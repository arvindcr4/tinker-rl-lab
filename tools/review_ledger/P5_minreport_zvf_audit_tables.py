"""Recompute P5 Tables p5-ushape and p5-gsize from the W&B zvf-audit export.

Input : platform_hybrid/experiments/results/review_2026-10-18/P5_minreport/zvf_audit_368_wandb_export.tsv
        (written by P5_minreport_export_wandb.py; 368 runs, W&B project zvf-audit)
Output: p5_ushape.tsv, p5_gsize.tsv in the same directory.
Runs without final_reward/mean_zvf in their W&B summary are excluded.
"""

import collections
import csv
import statistics as st
from pathlib import Path

D = (
    Path(__file__).resolve().parents[2]
    / "platform_hybrid/experiments/results/review_2026-10-18/P5_minreport"
)


def main():
    rows = list(csv.DictReader(open(D / "zvf_audit_368_wandb_export.tsv"), delimiter="\t"))
    ok = [
        r
        for r in rows
        if r["final_reward"] not in ("", "None") and r["mean_zvf"] not in ("", "None")
    ]
    print(f"{len(rows)} runs, {len(ok)} with final metrics")
    by = collections.defaultdict(list)
    byg = collections.defaultdict(list)
    for r in ok:
        m = r["model"].split("/")[-1]
        by[m].append((float(r["final_reward"]), float(r["mean_zvf"])))
        byg[(m, int(r["group"]))].append((float(r["mean_zvf"]), r["batch"]))
    with open(D / "p5_ushape.tsv", "w") as f:
        f.write("model\tn_runs\tmean_final_reward\tmean_zvf\n")
        for m, v in sorted(by.items(), key=lambda kv: st.mean(x[0] for x in kv[1])):
            line = f"{m}\t{len(v)}\t{st.mean(x[0] for x in v):.4f}\t{st.mean(x[1] for x in v):.4f}"
            f.write(line + "\n")
            print(line)
    with open(D / "p5_gsize.tsv", "w") as f:
        f.write("model\tG\tn_runs\tmean_zvf\tbatches\n")
        for (m, g), v in sorted(byg.items()):
            line = f"{m}\t{g}\t{len(v)}\t{st.mean(x[0] for x in v):.4f}\t{','.join(sorted(set(x[1] for x in v)))}"
            f.write(line + "\n")
            print(line)


if __name__ == "__main__":
    main()
