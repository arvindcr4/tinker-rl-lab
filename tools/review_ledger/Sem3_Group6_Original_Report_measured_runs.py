"""Measured Qwen3-8B GSM8K GRPO runs (Tinker, W&B project tinker-rl-scaling) and the
matching held-out GSM8K evaluations, used to replace the unsourced Chapter 5 tables
of the Group 6 report (platform_tinker/reports/combined/base_final.tex).

Usage:
  python3 tools/review_ledger/Sem3_Group6_Original_Report_measured_runs.py <wandb_runs.json>

Inputs: the W&B export (read-only inventory pulled 2026-10-18) and
platform_tinker/reports/final/gsm8k_{base_control_200,heldout_seed*}.json.
Output: platform_hybrid/experiments/results/review_2026-10-18/Sem3_Group6_Original_Report/measured_runs.json
The seed is the unit of analysis; step-level rewards are never pooled.
"""

import json
import statistics
import sys
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT = (
    REPO
    / "platform_hybrid/experiments/results/review_2026-10-18/Sem3_Group6_Original_Report/measured_runs.json"
)

runs = json.load(open(sys.argv[1]))
rows = []
for r in runs:
    if (
        r["project"] != "tinker-rl-scaling"
        or r["state"] != "finished"
        or not r["name"].endswith("-tinker")
    ):
        continue
    s, c = r["summary"], r["config"]
    rows.append(
        dict(
            wandb_id=r["id"],
            name=r["name"],
            model=c.get("model"),
            seed=c.get("seed"),
            lora_rank=c.get("lora_rank"),
            learning_rate=c.get("learning_rate"),
            tinker_run_id=c.get("tinker_run_id"),
            total_steps=s.get("total_steps"),
            peak_train_reward=s.get("peak_accuracy", 0) / 100,
            last10_train_reward=s.get("mean_last10_reward"),
        )
    )

seed_rows = [r for r in rows if r["name"].startswith("gsm8k-qwen3-8b-seed")]
last10 = [r["last10_train_reward"] for r in seed_rows]

final = REPO / "platform_tinker/reports/final"
base = json.load(open(final / "gsm8k_base_control_200.json"))["summary"]
held = []
for r in sorted(seed_rows, key=lambda x: x["seed"]):
    f = final / f"gsm8k_heldout_seed{r['seed']:03d}.json"
    d = json.load(open(f))
    assert d["config"]["model"].split("://")[1] in r["tinker_run_id"], f
    held.append(
        dict(
            seed=r["seed"],
            correct=d["summary"]["correct"],
            attempted=d["summary"]["attempted"],
            accuracy=d["summary"]["accuracy"],
        )
    )
acc = [h["accuracy"] for h in held]
t = stats.ttest_1samp(acc, base["accuracy"])
out = dict(
    runs=rows,
    qwen3_8b_seed_level_last10_train_reward=dict(
        n_seeds=len(last10),
        values=last10,
        mean=statistics.mean(last10),
        sd=statistics.stdev(last10),
    ),
    heldout=dict(
        base=dict(correct=base["correct"], attempted=base["attempted"], accuracy=base["accuracy"]),
        per_seed=held,
        mean=statistics.mean(acc),
        sd=statistics.stdev(acc),
        one_sample_t=float(t.statistic),
        df=len(acc) - 1,
        p_two_sided=float(t.pvalue),
        note="single base evaluation supplies no baseline variance; not an equivalence test",
    ),
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
