"""Claim C exp 4 analysis: matched TRL vs Tinker (Qwen3-8B, 3 seeds x 30 steps).

  python3 -I matched_analyze.py RESULTS_DIR OUT_JSON
RESULTS_DIR holds trl_s*.json, tinker_s*.json, eval_heldout.json.
Training reward: per-seed last-10 mean; paired-by-seed difference (seed fixes the prompt stream),
t CI (n=3, df=2) and per-step paired bootstrap over (seed, step) cells.
Held-out: greedy acc on 200 test items per adapter; paired by item within seed, pooled across seeds.
"""
import glob, json, sys
import numpy as np
from scipy import stats

D, OUT = sys.argv[1], sys.argv[2]
runs = {}
for f in glob.glob(f"{D}/*_s*.json"):
    r = json.load(open(f)); runs[(r["side"], r["seed"])] = r
seeds = sorted({s for (_, s) in runs})
res = {"seeds": seeds, "per_seed": {}, "train": {}, "heldout": {}}
curves = {side: np.array([[x["mean_reward"] for x in runs[(side, s)]["step_log"]] for s in seeds]) for side in ("trl", "tinker")}
for side in curves:
    res["train"][side] = {"mean_curve": curves[side].mean(0).tolist(), "last10_by_seed": curves[side][:, -10:].mean(1).tolist(),
                          "first5_by_seed": curves[side][:, :5].mean(1).tolist(), "step0_by_seed": curves[side][:, 0].tolist(),
                          "trunc_by_seed": [float(np.mean([x.get("trunc_frac", np.nan) for x in runs[(side, s)]["step_log"]])) for s in seeds]}
d = curves["tinker"][:, -10:].mean(1) - curves["trl"][:, -10:].mean(1)
n = len(d)
tc = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
rng = np.random.default_rng(0)
diff_cells = curves["tinker"][:, -10:] - curves["trl"][:, -10:]
bs = [diff_cells[rng.integers(0, n, n)][:, rng.integers(0, 10, 10)].mean() for _ in range(10000)]
res["train"]["last10_gap_tinker_minus_trl"] = {"by_seed": d.tolist(), "mean": float(d.mean()), "t_ci95": [float(d.mean() - tc * se), float(d.mean() + tc * se)],
                                               "boot_seed_step_ci95": list(np.percentile(bs, [2.5, 97.5]))}
d0 = curves["tinker"][:, 0] - curves["trl"][:, 0]
res["train"]["step0_gap_tinker_minus_trl"] = {"by_seed": d0.tolist(), "note": "same prompts at step 0; 16 samples each"}
# stream check: TRL saw the dataset rows of the intended step
res["trl_stream_order_ok"] = {s: runs[("trl", s)].get("stream_order_ok") for s in seeds}
try:
    E = json.load(open(f"{D}/eval_heldout.json"))
    res["heldout"]["env"] = E["_env"]
    base = np.array(E["base"]["r_campaign"])
    res["heldout"]["base"] = {"acc": float(base.mean()), "acc_boxed_only": E["base"]["acc_boxed_only"], "trunc": E["base"]["trunc_frac"]}
    acc = {side: {s: np.array(E[f"{side}_s{s}"]["r_campaign"]) for s in seeds if f"{side}_s{s}" in E} for side in ("trl", "tinker")}
    for side in acc:
        res["heldout"][side] = {str(s): {"acc": float(v.mean()), "gain_vs_base": float(v.mean() - base.mean()),
                                         "acc_boxed_only": E[f"{side}_s{s}"]["acc_boxed_only"], "trunc": E[f"{side}_s{s}"]["trunc_frac"]}
                                for s, v in acc[side].items()}
    common = [s for s in seeds if s in acc["trl"] and s in acc["tinker"]]
    M = np.stack([acc["tinker"][s] - acc["trl"][s] for s in common])  # seeds x items
    bs = [M[rng.integers(0, len(common), len(common))][:, rng.integers(0, M.shape[1], M.shape[1])].mean() for _ in range(10000)]
    sd = M.mean(1)
    res["heldout"]["gap_tinker_minus_trl"] = {"by_seed": sd.tolist(), "mean": float(M.mean()),
                                              "twoway_boot_ci95": list(np.percentile(bs, [2.5, 97.5])),
                                              "t_ci95_seed": ([float(sd.mean() - stats.t.ppf(.975, len(sd) - 1) * sd.std(ddof=1) / np.sqrt(len(sd))),
                                                               float(sd.mean() + stats.t.ppf(.975, len(sd) - 1) * sd.std(ddof=1) / np.sqrt(len(sd)))] if len(sd) > 1 else None)}
except FileNotFoundError:
    res["heldout"] = "eval_heldout.json missing"
res["env"] = {f"{k[0]}_s{k[1]}": v.get("env") for k, v in runs.items()}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res["train"][k] for k in ("last10_gap_tinker_minus_trl", "step0_gap_tinker_minus_trl")}, indent=1))
print({s: (res["train"]["trl"]["last10_by_seed"][i], res["train"]["tinker"]["last10_by_seed"][i]) for i, s in enumerate(seeds)})
if isinstance(res["heldout"], dict): print(json.dumps({k: v for k, v in res["heldout"].items() if k != "env"}, indent=1))
