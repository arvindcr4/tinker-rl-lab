"""Final stats for same-stack GSM8K-CoT v2 (claim A), Modal A10G + Colab runs.

Adapted from the audit's claimA/scripts/v2_analysis.py (+ stats_lib.py, inlined) to include
ppo_8x8, the Colab completions, and the hardware-parity reruns.

  python3 -I v2_final_analysis.py MODAL_DIR COLAB_DIR GOLD_JSON OUT_JSON
MODAL_DIR: Modal volume copies (<arm>_s<seed>.json + _texts.json.gz)
COLAB_DIR: Colab runs, same format (parity reruns ppo_s42 / grpo_g8_s42 are kept separate)
"""
import glob, gzip, itertools, json, re, sys
import numpy as np
from scipy import stats
from scipy.stats import binomtest

MODAL, COLAB, GOLD, OUT = sys.argv[1:5]
MARGIN = 0.02
PARITY = {("ppo", 42), ("grpo_g8", 42)}


# ---- stats_lib (audit) ----
def signflip(d):
    d = np.asarray(d, float); obs = abs(d.sum())
    return sum(abs((np.array(s) * d).sum()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=len(d))) / 2 ** len(d)


def boot_ci(d, B=20000, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    return list(np.percentile(d[rng.integers(0, len(d), (B, len(d)))].mean(1), [2.5, 97.5]))


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds]); n = len(d)
    md, sd = d.mean(), d.std(ddof=1); se = sd / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    p_lo = 1 - stats.t.cdf((md + MARGIN) / se, n - 1); p_hi = stats.t.cdf((md - MARGIN) / se, n - 1)
    return {"n": n, "seeds": seeds, "diffs": [round(float(x), 4) for x in d], "mean": md, "sd": sd,
            "t_ci95": [md - tc * se, md + tc * se], "boot_ci95": boot_ci(d),
            "p_t": float(stats.ttest_1samp(d, 0).pvalue), "p_signflip": signflip(d),
            "tost_margin": MARGIN, "tost_p": float(max(p_lo, p_hi)), "equivalent_at_margin": bool(max(p_lo, p_hi) < 0.05)}


def holm(ps):
    order = sorted(ps, key=ps.get); out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k])); out[k] = run
    return out


def mde(sd, n, power=0.8, alpha=0.05):
    from scipy.optimize import brentq
    df = n - 1; tc = stats.t.ppf(1 - alpha / 2, df)
    return brentq(lambda dl: (1 - stats.nct.cdf(tc, df, dl / (sd / np.sqrt(n)))) - power, 1e-9, 10 * sd)


# ---- independent grader (audit) ----
def last_boxed(t):
    i = t.rfind("\\boxed")
    while i != -1:
        j = t.find("{", i)
        if j != -1:
            depth, k = 0, j
            while k < len(t):
                if t[k] == "{": depth += 1
                elif t[k] == "}":
                    depth -= 1
                    if depth == 0: return t[j + 1:k]
                k += 1
        i = t.rfind("\\boxed", 0, i)
    return None


def to_num(s):
    s = re.sub(r"\\text\{[^}]*\}|\\mathrm\{[^}]*\}|\\[,;:! ]|\\%|%|\\\$|\$|,|\s", "", s).rstrip(".")
    m = re.fullmatch(r"\\d?frac\{(-?[\d.]+)\}\{([\d.]+)\}", s)
    if m: return float(m.group(1)) / float(m.group(2))
    try: return float(s)
    except ValueError: return None


def grade(t, g):
    b = last_boxed(t)
    v = to_num(b) if b is not None else None
    return int(v is not None and abs(v - float(g)) < 1e-6)


gold = [t["answer"].split("####")[-1].strip().replace(",", "") for t in json.load(open(GOLD))["test"]]


def hardware_label(result):
    env = result.get("execution_environment") or result.get("env")
    if not isinstance(env, dict) or any(not isinstance(env.get(k), str) or not env[k].strip() for k in ("gpu", "platform")):
        raise ValueError("Per-run GPU and platform provenance is required; directory names are not evidence")
    return f"{env['gpu']} ({env['platform']})"


def load(path):
    r = json.load(open(path))
    hw = hardware_label(r)
    tx = json.load(gzip.open(path.replace(".json", "_texts.json.gz"), "rt"))
    mine = np.array([grade(t, g) for t, g in zip(tx["post"], gold)])
    return {"arm": r["arm"], "seed": r["seed"], "hw": hw, "post": np.array(r["post_correct"]), "pre": np.array(r["pre_correct"]),
            "post_acc": float(np.mean(r["post_correct"])), "pre_acc": float(np.mean(r["pre_correct"])),
            "regrade_disagree": int((mine != np.array(r["post_correct"])).sum()),
            "step_reward": [s["mean_reward"] for s in r["step_log"]], "last10": r["last10_avg"],
            "elapsed_s": r["elapsed_seconds"], "post_trunc": float(np.mean(r["post_trunc"])),
            "train_trunc": float(np.mean([s["trunc_frac"] for s in r["step_log"]])),
            "env": r.get("execution_environment") or r["env"],
            "critic_vl_pre_mean": (float(np.mean([s["value_loss_pre_fit"] for s in r["step_log"]])) if r["arm"].startswith("ppo") else None),
            "floor_half_var_mean": float(np.mean([s["reward_var_half"] for s in r["step_log"]]))}


modal = [load(f) for f in sorted(glob.glob(f"{MODAL}/*_s*.json"))]
colab = [load(f) for f in sorted(glob.glob(f"{COLAB}/*_s*.json"))]
parity_runs = [c for c in colab if (c["arm"], c["seed"]) in PARITY]
new_runs = [c for c in colab if (c["arm"], c["seed"]) not in PARITY]
runs = {}
for x in modal + new_runs:
    runs.setdefault(x["arm"], {})[x["seed"]] = x

R = {"experiment": "samestack_gsm8k_cot_v2 final", "n_runs": {a: len(d) for a, d in runs.items()},
     "hardware_by_run": {f"{a}_s{s}": x["hw"] for a, d in runs.items() for s, x in sorted(d.items())},
     "regrade_item_disagreements_total": int(sum(x["regrade_disagree"] for d in runs.values() for x in d.values())),
     "arms": {}, "contrasts": {}, "holm": {}, "item_level": {}, "power": {}, "hardware_parity": {}}
for a, d in sorted(runs.items()):
    v = list(d.values())
    R["arms"][a] = {"n_seeds": len(d), "seeds": sorted(d), "pre_acc_mean": np.mean([x["pre_acc"] for x in v]),
                    "post_acc_mean": np.mean([x["post_acc"] for x in v]), "post_sd": np.std([x["post_acc"] for x in v], ddof=1),
                    "gain_mean": np.mean([x["post_acc"] - x["pre_acc"] for x in v]),
                    "last10_train_reward": np.mean([x["last10"] for x in v]), "eval_trunc_post": np.mean([x["post_trunc"] for x in v]),
                    "train_trunc": np.mean([x["train_trunc"] for x in v]),
                    "critic_value_loss_pre_fit": (np.mean([x["critic_vl_pre_mean"] for x in v]) if a.startswith("ppo") else None),
                    "critic_floor_half_reward_var": (np.mean([x["floor_half_var_mean"] for x in v]) if a.startswith("ppo") else None),
                    "elapsed_s_mean": np.mean([x["elapsed_s"] for x in v])}

post = {a: {s: x["post_acc"] for s, x in d.items()} for a, d in runs.items()}
pairs = {"g8-ppo": ("grpo_g8", "ppo"), "g8-g2": ("grpo_g8", "grpo_g2"), "g2-ppo": ("grpo_g2", "ppo"),
         "g8-ppo8x8": ("grpo_g8", "ppo_8x8"), "ppo8x8-ppo": ("ppo_8x8", "ppo")}
for k, (x, y) in pairs.items():
    if x in post and y in post and len(set(post[x]) & set(post[y])) >= 2:
        R["contrasts"][k] = paired(post[x], post[y])
families = {"lead_specified (g8-ppo, g8-g2, g2-ppo)": ["g8-ppo", "g8-g2", "g2-ppo"],
            "script_preregistered (g8-ppo, g8-ppo8x8, g8-g2)": ["g8-ppo", "g8-ppo8x8", "g8-g2"]}
for fam, ks in families.items():
    ks = [k for k in ks if k in R["contrasts"]]
    R["holm"][fam] = {"p_t": holm({k: R["contrasts"][k]["p_t"] for k in ks}),
                      "p_signflip": holm({k: R["contrasts"][k]["p_signflip"] for k in ks})}
# Sensitivity: same-hardware seeds only (drops Colab-trained seeds from any contrast).
same_hw = {a: {s: v for s, v in d.items() if "A10G" in runs[a][s]["env"]["gpu"]} for a, d in post.items()}
R["contrasts_same_hardware_only"] = {k: paired(same_hw[x], same_hw[y]) for k, (x, y) in pairs.items()
                                     if x in same_hw and y in same_hw and len(set(same_hw[x]) & set(same_hw[y])) >= 2}

rng = np.random.default_rng(1)
for k, (x, y) in pairs.items():
    if k not in R["contrasts"]: continue
    seeds = sorted(set(runs[x]) & set(runs[y]))
    D = np.stack([runs[x][s]["post"] for s in seeds]) - np.stack([runs[y][s]["post"] for s in seeds])
    b, c = int((D == 1).sum()), int((D == -1).sum())
    bs = np.array([D[np.ix_(rng.integers(0, len(seeds), len(seeds)), rng.integers(0, D.shape[1], D.shape[1]))].mean() for _ in range(4000)])
    R["item_level"][k] = {"n_seeds": len(seeds), "mean_diff": D.mean(), "x_only": b, "y_only": c,
                          "pooled_mcnemar_naive_p": binomtest(b, b + c).pvalue, "twoway_boot_ci95": list(np.percentile(bs, [2.5, 97.5]))}
for k, c in R["contrasts"].items():
    R["power"][k] = {"sd_diff": c["sd"], "mde_80pct": mde(c["sd"], c["n"]) if c["sd"] > 0 else None, "n": c["n"]}

for cr in parity_runs:
    mr = runs[cr["arm"]][cr["seed"]]
    R["hardware_parity"][f"{cr['arm']}_s{cr['seed']}"] = {
        "modal_hw": mr["hw"], "colab_hw": cr["hw"], "colab_env": cr["env"],
        "pre_acc": {"modal": mr["pre_acc"], "colab": cr["pre_acc"], "delta": cr["pre_acc"] - mr["pre_acc"],
                    "item_agreement": float((mr["pre"] == cr["pre"]).mean())},
        "post_acc": {"modal": mr["post_acc"], "colab": cr["post_acc"], "delta": cr["post_acc"] - mr["post_acc"],
                     "item_agreement": float((mr["post"] == cr["post"]).mean())},
        "train_reward_step0": {"modal": mr["step_reward"][0], "colab": cr["step_reward"][0]},
        "last10_train_reward": {"modal": mr["last10"], "colab": cr["last10"], "delta": cr["last10"] - mr["last10"]},
        "train_reward_trace_max_abs_diff": float(np.max(np.abs(np.array(mr["step_reward"]) - np.array(cr["step_reward"])))),
        "elapsed_s": {"modal": mr["elapsed_s"], "colab": cr["elapsed_s"]}}
# Reference scale for parity deltas: across-seed SD of post accuracy within the arm.
for k, v in R["hardware_parity"].items():
    arm = k.rsplit("_s", 1)[0]
    v["post_delta_over_arm_seed_sd"] = v["post_acc"]["delta"] / R["arms"][arm]["post_sd"]
R["new_colab_runs"] = {f"{x['arm']}_s{x['seed']}": {"post_acc": x["post_acc"], "pre_acc": x["pre_acc"], "env": x["env"]} for x in new_runs}
json.dump(R, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"arms": {a: {k: R["arms"][a][k] for k in ("n_seeds", "pre_acc_mean", "post_acc_mean", "post_sd")} for a in R["arms"]},
                  "contrasts": {k: {kk: c[kk] for kk in ("n", "mean", "t_ci95", "p_t", "p_signflip", "tost_p")} for k, c in R["contrasts"].items()},
                  "holm": R["holm"], "parity": {k: {"pre": v["pre_acc"], "post": v["post_acc"]} for k, v in R["hardware_parity"].items()}},
                 indent=1, default=float))
