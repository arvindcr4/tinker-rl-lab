"""P7 review 2026-10-18: recompute N2 degenerate-group counts and binomial
all-correct-share predictions from the logged N2 reward tensors."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
methods = ["grpo", "aero", "areal", "gift"]
tot = deg = ones = 0
pred_step_num = pred_step_den = 0.0  # step-mean p: expected all-one / expected degenerate
pred_meth_num = pred_meth_den = 0.0
per_method = {}
for m in methods:
    rows = [json.loads(l) for l in open(D / f"{m}_s0_tensors.jsonl")]
    md = mo = mt = 0
    allr = []
    for r in rows:
        G = r["group_size"]
        ks = [int(sum(g)) for g in r["rewards"]]
        p = sum(ks) / (G * len(ks))
        allr += ks
        for k in ks:
            mt += 1
            if k in (0, G):
                md += 1
                mo += k == G
            pred_step_num += p ** G
            pred_step_den += p ** G + (1 - p) ** G
    pm = sum(allr) / (8 * len(allr))
    pred_meth_num += len(allr) * pm ** 8
    pred_meth_den += len(allr) * (pm ** 8 + (1 - pm) ** 8)
    per_method[m] = dict(steps=len(rows), groups=mt, degenerate=md,
                         all_correct=mo, mean_reward=round(pm, 4))
    tot += mt
    deg += md
    ones += mo
# Leave-one-method-out: the 16 prompts at each step are shared across methods,
# so estimate each (method, step, prompt) success rate from the other three
# methods' 24 rollouts on the same prompt (independent of the scored group).
rows_by_m = {m: [json.loads(l) for l in open(D / f"{m}_s0_tensors.jsonl")] for m in methods}
lomo_one = lomo_zero = 0.0
for s in range(len(rows_by_m["grpo"])):
    for j in range(16):
        ids = {rows_by_m[m][s]["prompt_indices"][j] for m in methods}
        assert len(ids) == 1, "prompt mismatch across methods"
        for m in methods:
            others = [o for o in methods if o != m]
            p = sum(sum(rows_by_m[o][s]["rewards"][j]) for o in others) / (8 * len(others))
            lomo_one += p ** 8
            lomo_zero += (1 - p) ** 8
lomo = dict(expected_all_correct=round(lomo_one, 1), expected_all_wrong=round(lomo_zero, 1),
            share=round(lomo_one / (lomo_one + lomo_zero), 4))
out = dict(per_method=per_method, groups=tot, degenerate=deg, all_correct=ones,
           all_wrong=deg - ones, binom_pred_leave_one_method_out=lomo,
           note_91_5pct="the earlier 91.5% (1,775/165) prediction could not be reproduced by any "
                        "definition tried (step-mean p: 99.99%; method-mean p: 100%; "
                        "4-method pooled p: 92.1%; LOMO p: 91.9%)",
           all_correct_share=round(ones / deg, 4),
           binom_pred_share_stepmean_p=round(pred_step_num / pred_step_den, 4),
           binom_pred_share_methodmean_p=round(pred_meth_num / pred_meth_den, 4))
print(json.dumps(out, indent=1))
o = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P7_zvf_controller"
o.mkdir(parents=True, exist_ok=True)
(o / "n2_degenerate_share.json").write_text(json.dumps(out, indent=1))
