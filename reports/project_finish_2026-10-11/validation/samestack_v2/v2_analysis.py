"""Usage: python3 -I v2_analysis.py <scripts_dir> <data_dir> <gold.json>"""
import glob, gzip, json, os, re, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from stats_lib import paired, holm, mde, n_needed
from scipy.stats import binomtest, wilcoxon

DATA, GOLD = sys.argv[2], sys.argv[3]
test = json.load(open(GOLD))["test"]
gold = [t["answer"].split("####")[-1].strip().replace(",", "") for t in test]


def last_boxed(t):
    """Balanced-brace scan for the last \\boxed{...} (independent of the run's regex)."""
    i = t.rfind("\\boxed")
    while i != -1:
        j = t.find("{", i)
        if j != -1:
            depth, k = 0, j
            while k < len(t):
                if t[k] == "{": depth += 1
                elif t[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return t[j + 1:k]
                k += 1
        i = t.rfind("\\boxed", 0, i)
    return None


def to_num(s):
    s = re.sub(r"\\text\{[^}]*\}|\\mathrm\{[^}]*\}|\\[,;:! ]|\\%|%|\\\$|\$|,|\s", "", s).rstrip(".")
    m = re.fullmatch(r"\\d?frac\{(-?[\d.]+)\}\{([\d.]+)\}", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def grade(t, g):
    b = last_boxed(t)
    if b is None:
        return 0, "nobox"
    v = to_num(b)
    if v is None:
        return 0, "unparsable"
    return int(abs(v - float(g)) < 1e-6), "ok"


runs = {}
for f in sorted(glob.glob(os.path.join(DATA, "*_s*.json"))):
    r = json.load(open(f))
    tx = json.load(gzip.open(f.replace(".json", "_texts.json.gz"), "rt"))
    mine = {k: np.array([grade(t, g)[0] for t, g in zip(tx[k], gold)]) for k in ("pre", "post")}
    nobox = {k: np.mean([last_boxed(t) is None for t in tx[k]]) for k in ("pre", "post")}
    runs.setdefault(r["arm"], {})[r["seed"]] = dict(
        stored_post=np.array(r["post_correct"]), stored_pre=np.array(r["pre_correct"]),
        mine_post=mine["post"], mine_pre=mine["pre"], nobox=nobox,
        post_trunc=np.mean(r["post_trunc"]), pre_trunc=np.mean(r["pre_trunc"]),
        train_trunc=np.mean([s["trunc_frac"] for s in r["step_log"]]),
        train_len=np.mean([s["mean_comp_len"] for s in r["step_log"]]),
        elapsed=r["elapsed_seconds"], eval_s=r["eval_seconds"],
        disagree_post=[i for i in np.where(mine["post"] != np.array(r["post_correct"]))[0].tolist()],
        disagree_pre=int((mine["pre"] != np.array(r["pre_correct"])).sum()))

R = {"arms": {}, "contrasts": {}, "item_level": {}, "power": {}}
for a, d in runs.items():
    v = list(d.values())
    R["arms"][a] = {
        "n_seeds": len(d), "seeds": sorted(d),
        "base_pre_acc_stored_mean": np.mean([x["stored_pre"].mean() for x in v]),
        "post_acc_stored_mean": np.mean([x["stored_post"].mean() for x in v]),
        "post_acc_mine_mean": np.mean([x["mine_post"].mean() for x in v]),
        "post_sd_across_seeds": np.std([x["stored_post"].mean() for x in v], ddof=1),
        "item_disagreements_post_total": sum(len(x["disagree_post"]) for x in v),
        "item_disagreements_pre_total": sum(x["disagree_pre"] for x in v),
        "items_graded": 1319 * len(v),
        "eval_trunc_pre": np.mean([x["pre_trunc"] for x in v]),
        "eval_trunc_post": np.mean([x["post_trunc"] for x in v]),
        "eval_nobox_post": np.mean([x["nobox"]["post"] for x in v]),
        "train_trunc_mean": np.mean([x["train_trunc"] for x in v]),
        "train_comp_len_mean": np.mean([x["train_len"] for x in v]),
        "elapsed_s_mean": np.mean([x["elapsed"] for x in v]), "elapsed_s_max": np.max([x["elapsed"] for x in v]),
    }
# sample disagreements for inspection
ex = []
for a, d in runs.items():
    for s, x in d.items():
        for i in x["disagree_post"][:2]:
            ex.append((a, s, i, int(x["stored_post"][i]), int(x["mine_post"][i]), gold[i]))
R["disagreement_examples"] = ex[:12]

for key in ("stored", "mine"):
    post = {a: {s: x[f"{key}_post"].mean() for s, x in d.items()} for a, d in runs.items()}
    C = {"g8-ppo": paired(post["grpo_g8"], post["ppo"]), "g8-g2": paired(post["grpo_g8"], post["grpo_g2"]),
         "g2-ppo": paired(post["grpo_g2"], post["ppo"])}
    ht, hs = holm({k: c["p_t"] for k, c in C.items()}), holm({k: c["p_signflip"] for k, c in C.items()})
    for k in C:
        C[k]["holm_p_t"], C[k]["holm_p_signflip"] = ht[k], hs[k]
    R["contrasts"][key] = C

# also delta-vs-base (post - pre, same run) contrasts as a covariate-adjusted check
post_d = {a: {s: x["stored_post"].mean() - x["stored_pre"].mean() for s, x in d.items()} for a, d in runs.items()}
R["contrasts"]["gain_over_own_pre"] = {"g8-ppo": paired(post_d["grpo_g8"], post_d["ppo"])}

# item-level secondary
rng = np.random.default_rng(1)
for k, (x, y) in {"g8-ppo": ("grpo_g8", "ppo"), "g8-g2": ("grpo_g8", "grpo_g2"), "g2-ppo": ("grpo_g2", "ppo")}.items():
    seeds = sorted(set(runs[x]) & set(runs[y]))
    X = np.stack([runs[x][s]["stored_post"] for s in seeds])  # seeds x items
    Y = np.stack([runs[y][s]["stored_post"] for s in seeds])
    D = X - Y
    b, c = int((D == 1).sum()), int((D == -1).sum())
    # per-seed McNemar
    per_seed = [binomtest(int((D[i] == 1).sum()), int((D[i] != 0).sum())).pvalue for i in range(len(seeds))]
    # two-way clustered bootstrap: resample seeds and items independently
    B, nS, nI = 4000, D.shape[0], D.shape[1]
    bs = np.empty(B)
    for j in range(B):
        si = rng.integers(0, nS, nS); ii = rng.integers(0, nI, nI)
        bs[j] = D[np.ix_(si, ii)].mean()
    # items-only bootstrap (seeds fixed) -> conditional on these seeds
    bi = np.array([D[:, rng.integers(0, nI, nI)].mean() for _ in range(4000)])
    item_mean = D.mean(0)
    nz = item_mean[item_mean != 0]
    R["item_level"][k] = {"n_seeds": nS, "mean_diff": D.mean(), "pooled_x_only": b, "pooled_y_only": c,
                          "pooled_mcnemar_naive_p": binomtest(b, b + c).pvalue,
                          "per_seed_mcnemar_p": per_seed,
                          "twoway_boot_ci": list(np.percentile(bs, [2.5, 97.5])),
                          "items_only_boot_ci": list(np.percentile(bi, [2.5, 97.5])),
                          "wilcoxon_items_seedavg_p": wilcoxon(nz).pvalue if len(nz) else None}

for k, c in R["contrasts"]["stored"].items():
    sd = c["sd"]
    R["power"][k] = {"sd_diff": sd, "mde_n7": mde(sd, 7), "mde_n10": mde(sd, 10),
                     "seeds_for_0.05": n_needed(sd, 0.05), "seeds_for_0.01": n_needed(sd, 0.01),
                     "seeds_for_0.005": n_needed(sd, 0.005)}
# TOST at the pre-specified +/-0.02 margin and a one-sided test of a v1-sized (0.05) difference
from scipy import stats as _st
for k, c in R["contrasts"]["stored"].items():
    n, m, sd = c["n"], c["mean"], c["sd"]
    se = sd / np.sqrt(n)
    lo = 1 - _st.t.cdf((m + 0.02) / se, n - 1)
    hi = _st.t.cdf((m - 0.02) / se, n - 1)
    R["power"][k]["tost_0.02_p"] = max(lo, hi)
    R["power"][k]["p_diff_ge_0.05"] = _st.t.cdf((m - 0.05) / se, n - 1)
print(json.dumps(R, indent=1, default=float))
