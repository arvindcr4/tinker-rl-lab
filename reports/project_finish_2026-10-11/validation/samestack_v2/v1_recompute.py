import json, sys
import numpy as np
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from stats_lib import paired, holm, mde, n_needed

R = sys.argv[1] if len(sys.argv) > 1 else "platform_hybrid/experiments/results/"
R = R.rstrip("/") + "/"
full = json.load(open(R + "samestack_gsm8k_cot_full.json"))
by = {}
for r in full["runs"]:
    pc, qc = np.array(r["pre_correct"]), np.array(r["post_correct"])
    assert len(qc) == r["n_eval"] == 200
    by.setdefault(r["arm"], {})[r["seed"]] = dict(pre=pc, post=qc, stored_post=r["heldout_post_acc"],
                                                  len=np.mean([s["mean_comp_len"] for s in r["step_log"]]),
                                                  elapsed=r["elapsed_seconds"])
out = {}
for a, d in by.items():
    mism = [s for s, x in d.items() if abs(x["post"].mean() - x["stored_post"]) > 1e-9]
    pre_same = all((x["pre"] == next(iter(d.values()))["pre"]).all() for x in d.values())
    out[a] = {"post_by_seed": {s: x["post"].mean() for s, x in sorted(d.items())},
              "pre_mean": np.mean([x["pre"].mean() for x in d.values()]), "pre_identical_across_seeds": pre_same,
              "post_mean": np.mean([x["post"].mean() for x in d.values()]), "stored_mismatch_seeds": mism,
              "train_comp_len_mean": np.mean([x["len"] for x in d.values()]),
              "elapsed_s_mean": np.mean([x["elapsed"] for x in d.values()])}
post = {a: {s: x["post"].mean() for s, x in d.items()} for a, d in by.items()}
c = {k: paired(post[x], post[y]) for k, (x, y) in
     {"g8-ppo": ("grpo_g8", "ppo"), "g8-g2": ("grpo_g8", "grpo_g2"), "g2-ppo": ("grpo_g2", "ppo")}.items()}
h = holm({k: v["p_t"] for k, v in c.items()})
hs = holm({k: v["p_signflip"] for k, v in c.items()})
for k in c:
    c[k]["holm_t"], c[k]["holm_sf"] = h[k], hs[k]
    c[k]["mde_n5"] = mde(c[k]["sd"], 5)
# item-level: pooled discordant counts g8 vs ppo
from scipy.stats import binomtest
for x, y in [("grpo_g8", "ppo"), ("grpo_g8", "grpo_g2")]:
    b = sum(int(((by[x][s]["post"] == 1) & (by[y][s]["post"] == 0)).sum()) for s in by[x])
    cc = sum(int(((by[x][s]["post"] == 0) & (by[y][s]["post"] == 1)).sum()) for s in by[x])
    c[f"{x}-{y}_pooled_discordant"] = {"x_only": b, "y_only": cc,
                                        "naive_mcnemar_p": binomtest(b, b + cc).pvalue}
print(json.dumps({"arms": out, "contrasts": c}, indent=1, default=float))
