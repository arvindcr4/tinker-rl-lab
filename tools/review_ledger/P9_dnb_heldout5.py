"""P9 Deep Review (2026-10-18): five-seed Qwen3-8B GSM8K held-out vs. base control.

Inputs: platform_tinker/reports/final/gsm8k_heldout_seed{042,137,256,512,999}.json,
        platform_tinker/reports/final/gsm8k_base_control_200.json (same 200 test items).
Output: platform_hybrid/experiments/results/review_2026-10-18/P9_dnb/heldout5.json
(1) one-sample t of seed accuracies vs the base point estimate (seeds as unit, df=4);
(2) item-paired t: per item, mean correctness over seeds minus base correctness (df=199).
"""

import json
import statistics as st
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "platform_tinker/reports/final"
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P9_dnb"


def item_correct(d):
    out = {}
    for i, ex in enumerate(d["examples"]):
        key = ex["idx"]
        out[str(key)] = float(ex["status"] == "correct")
    return out


seeds = ["042", "137", "256", "512", "999"]
runs = [json.loads((F / f"gsm8k_heldout_seed{s}.json").read_text()) for s in seeds]
base = json.loads((F / "gsm8k_base_control_200.json").read_text())
acc = [r["summary"]["accuracy"] for r in runs]
b = base["summary"]["accuracy"]
t1 = stats.ttest_1samp(acc, b)
items = [item_correct(r) for r in runs]
bi = item_correct(base)
common = sorted(set(bi).intersection(*[set(x) for x in items]))
diffs = [st.mean(x[k] for x in items) - bi[k] for k in common]
t2 = stats.ttest_1samp(diffs, 0.0)
res = {
    "per_seed_accuracy": dict(zip(seeds, acc)),
    "mean": round(st.mean(acc), 4),
    "sd": round(st.stdev(acc), 4),
    "base": b,
    "one_sample": {
        "t": round(float(t1.statistic), 3),
        "df": len(acc) - 1,
        "p": round(float(t1.pvalue), 3),
    },
    "item_paired": {
        "n_items": len(common),
        "mean_diff": round(st.mean(diffs), 4),
        "t": round(float(t2.statistic), 3),
        "df": len(common) - 1,
        "p": round(float(t2.pvalue), 3),
    },
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "heldout5.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
