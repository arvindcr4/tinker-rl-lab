"""Export the W&B runs behind P5 Exhibit 3 and Tables p5-ushape / p5-gsize.

Input: a read-only W&B dump (list of {project, id, name, state, created,
config, summary}) pulled 2026-10-18 (wandb_runs.json). Output: TSVs under
platform_hybrid/experiments/results/review_2026-10-18/P5_minreport/.

Usage: python tools/review_ledger/P5_minreport_export_wandb.py <wandb_runs.json>
"""
import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "platform_hybrid/experiments/results/review_2026-10-18/P5_minreport"


def main(dump):
    runs = json.load(open(dump))
    OUT.mkdir(parents=True, exist_ok=True)

    # Exhibit 3: zvf-training h2h2_* (Qwen3.5-4B, 15 steps, batch 2, G=8)
    cols = ["wandb_project", "wandb_id", "name", "state", "created", "arm", "seed", "model",
            "steps", "batch", "group_size", "lr", "rank", "temperature",
            "last10_avg", "first5_avg", "zero_loss_pct", "zero_reward_pct"]
    rows = []
    for r in runs:
        if r["project"] != "zvf-training" or not r["name"].startswith("h2h2_"):
            continue
        s = r["summary"]
        arm = r["name"].split("_")[1]
        rows.append([r["project"], r["id"], r["name"], r["state"], r["created"], arm, s.get("seed"),
                     "Qwen3.5-4B", s.get("steps"), s.get("batch"), s.get("group_size"), s.get("lr"),
                     s.get("rank"), s.get("temperature"), s.get("last10_avg"), s.get("first5_avg"),
                     s.get("zero_loss_pct"), s.get("zero_reward_pct")])
    rows.sort(key=lambda x: (x[5], x[6]))
    with open(OUT / "exhibit3_h2h2_wandb_export.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        w.writerows(rows)

    # 368-run zvf-audit project (Tables p5-ushape, p5-gsize)
    cols = ["wandb_id", "name", "state", "model", "group", "batch", "steps", "seed",
            "final_reward", "mean_zvf"]
    rows = []
    for r in runs:
        if r["project"] != "zvf-audit":
            continue
        c, s = r["config"], r["summary"]
        rows.append([r["id"], r["name"], r["state"], c.get("model"), c.get("group"), c.get("batch"),
                     c.get("steps"), c.get("seed"), s.get("final_reward"), s.get("mean_zvf")])
    with open(OUT / "zvf_audit_368_wandb_export.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
