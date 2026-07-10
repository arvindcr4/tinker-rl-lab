#!/usr/bin/env python3
"""P5 — Item 48: MIN-REPORT field discriminative-entropy audit.

Motivation. Items 01, 14, 18, 27, 28, 37 measure *coverage* or
*truthfulness* (does the manifest declare a value, does it match the
measured telemetry?). Item 32 measures *predictive-sufficiency*. None
measure the information-theoretic question: **do the declared values
actually separate cells, or are they effectively constant?**

Falsifiable claim: at least one MIN-REPORT item that scores 100%
validated on Exhibit 11 has fewer than 4 unique values across the 98
mega cells (H < 2 bits) — it is *informatively vacuous*.

For every mega_20260704 cell we:

  1. parse the 7 manifest items (loss_form, ref_policy_kl,
     sampler_backend_precision, per_step_zvf_path, group_size_schedule,
     heldout_split, decontamination_notes);
  2. compute per-item: n_unique, top_value_freq, Shannon entropy H (bits),
     normalised entropy H / log2(n_unique);
  3. compute per-stratum entropy H(task_slice) and H(G-bucket);
     discriminative-entropy-ratio D = 1 - mean_stratum_H / overall_H
     (D≈1 ⇒ variance between strata; D≈0 ⇒ within strata);
  4. classify: VACUOUS (H<0.5 or k≤2), LOW (H<1.5),
     MEDIUM (H<2.5), HIGH (H≥2.5);
  5. write:
       experiments/results/p5p8/p5_field_discriminative_entropy.tsv
       experiments/results/p5p8/p5_field_discriminative_entropy_summary.json
       experiments/results/p5p8/figures/p5_field_discriminative_entropy.{png,pdf}

Headline: the gap between *validation rate* (Exhibit 11) and
*discriminative content* (this iter) identifies the work-list for the
manifest emitter: items where the standard is satisfied without
providing reviewer-actionable information.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAN_DIR = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
CELLS_TSV = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
FIG_DIR = OUT_DIR / "figures"

ITEMS = [
    ("item1", "loss_form",                   "loss_form"),
    ("item2", "ref_policy_kl",               "ref_policy_kl"),
    ("item3", "sampler_backend_precision",   "sampler_backend_precision"),
    ("item4", "per_step_zvf_path",           "per_step_zvf_path"),
    ("item5", "group_size_schedule",         "group_size_schedule"),
    ("item6", "heldout_split",               "heldout_split"),
    ("item7", "decontamination_notes",       "decontamination_notes"),
]

# Exhibit-11 validation rates from iter-20.
EX11 = {"item1": 0.244, "item2": 0.244, "item3": 0.640, "item4": 0.495,
        "item5": 0.743, "item6": 0.762, "item7": 0.618}


def H_bits(counter):
    total = sum(counter.values())
    if total <= 1:
        return 0.0
    return -sum((v/total) * math.log2(v/total) for v in counter.values() if v > 0)


def H_norm(counter):
    n = len(counter)
    if n <= 1:
        return 0.0
    return H_bits(counter) / math.log2(n)


def classify(H, k):
    if H < 0.5 or k <= 2:
        return "VACUOUS"
    if H < 1.5:
        return "LOW"
    if H < 2.5:
        return "MEDIUM"
    return "HIGH"


def load_strata():
    """cell_id -> {task_slice, G} from cells.tsv."""
    s = {}
    with CELLS_TSV.open() as f:
        header = f.readline().rstrip().split("\t")
        i_task = header.index("task_slice")
        i_G = header.index("G")
        i_cid = header.index("cell_id")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s[parts[i_cid]] = {"task_slice": parts[i_task], "G": int(parts[i_G])}
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cells = []
    for fn in sorted(os.listdir(MAN_DIR)):
        if fn.endswith(".json"):
            with (MAN_DIR / fn).open() as f:
                cells.append(json.load(f))
    n = len(cells)
    strata = load_strata()

    rows = []
    by_item = {}
    for key, label, jf in ITEMS:
        vals = [str(c.get(jf, "")) for c in cells]
        ctr = Counter(vals)
        k = len(ctr)
        H = H_bits(ctr)
        Hn = H_norm(ctr)
        top_v, top_n = ctr.most_common(1)[0]
        tf = top_n / n
        cls = classify(H, k)
        by_item[key] = (vals, H)
        rows.append({
            "item_key": key, "item_label": label, "json_field": jf,
            "n_unique": k, "top_value": top_v[:60], "top_freq": round(tf, 4),
            "shannon_H_bits": round(H, 4), "normalised_H": round(Hn, 4),
            "classification": cls, "exhibit11_validation": EX11.get(key, 0.0),
        })

    # per-stratum entropy & D-ratios
    for r in rows:
        key = r["item_key"]
        vals, H = by_item[key]
        # by task_slice
        bt = defaultdict(list)
        for c, v in zip(cells, vals):
            bt[strata.get(c["cell_id"], {}).get("task_slice", "?")].append(v)
        H_tasks = [H_bits(Counter(vs)) for vs in bt.values()]
        mean_H_t = sum(H_tasks) / len(H_tasks) if H_tasks else 0.0
        # by G-bucket
        bg = defaultdict(list)
        for c, v in zip(cells, vals):
            G = strata.get(c["cell_id"], {}).get("G", 0)
            bg["small" if G <= 8 else "large"].append(v)
        H_G = [H_bits(Counter(vs)) for vs in bg.values()]
        mean_H_g = sum(H_G) / len(H_G) if H_G else 0.0
        r["n_task_strata"] = len(bt)
        r["mean_H_per_task_stratum"] = round(mean_H_t, 4)
        r["n_G_strata"] = len(bg)
        r["mean_H_per_G_stratum"] = round(mean_H_g, 4)
        r["D_task"] = round(1 - mean_H_t / H, 4) if H > 1e-9 else 0.0
        r["D_G"] = round(1 - mean_H_g / H, 4) if H > 1e-9 else 0.0
        discr = min(1.0, H / 2.5)
        r["discriminative_score_2.5bit"] = round(discr, 4)
        r["gap_validation_minus_discriminative"] = round(
            r["exhibit11_validation"] - discr, 4)

    # write TSV
    cols = list(rows[0].keys())
    tsv = OUT_DIR / "p5_field_discriminative_entropy.tsv"
    with tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {tsv}")

    # write JSON
    counts = {k: sum(1 for r in rows if r["classification"] == k)
              for k in ("VACUOUS", "LOW", "MEDIUM", "HIGH")}
    vac_high = [r for r in rows
                if r["classification"] == "VACUOUS"
                and r["exhibit11_validation"] >= 0.5]
    summary = {
        "n_cells": n, "n_items": len(rows),
        "classification_counts": counts,
        "vacuous_items": [r["item_key"] for r in rows
                          if r["classification"] == "VACUOUS"],
        "high_items": [r["item_key"] for r in rows
                       if r["classification"] == "HIGH"],
        "vacuous_items_with_high_validation_rate": [
            {"item": r["item_key"], "label": r["item_label"],
             "H_bits": r["shannon_H_bits"], "n_unique": r["n_unique"],
             "top_freq": r["top_freq"],
             "exhibit11_validation": r["exhibit11_validation"],
             "gap": r["gap_validation_minus_discriminative"]}
            for r in vac_high],
        "headline_finding": (
            f"{counts['VACUOUS']}/{len(rows)} MIN-REPORT items are VACUOUS "
            f"(H<0.5 bits, <=2 unique values) across n={n} manifests; "
f"{len(vac_high)} of those also score >=50% validated by Exhibit 11. "
            f"The manifest emitter satisfies the standard without providing "
            f"reviewer-actionable information on these items."),
        "rows": rows,
    }
    js = OUT_DIR / "p5_field_discriminative_entropy_summary.json"
    with js.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {js}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        cc = {"VACUOUS": "#d62728", "LOW": "#ff7f0e",
              "MEDIUM": "#bcbd22", "HIGH": "#2ca02c"}
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [r["item_label"] for r in rows]
        H_vals = [r["shannon_H_bits"] for r in rows]
        bars = ax.bar(labels, H_vals,
                      color=[cc[r["classification"]] for r in rows],
                      edgecolor="black", linewidth=0.5)
        for h, ls, c in [(2.5, "--", "black"), (1.5, ":", "gray"),
                         (0.5, ":", "red")]:
            ax.axhline(h, color=c, linestyle=ls, linewidth=0.8)
        ax.set_ylabel("Shannon entropy H (bits)")
        ax.set_title(f"MIN-REPORT field discriminative entropy at n={n} cells\n"
                     f"{counts['VACUOUS']} VACUOUS, {counts['LOW']} LOW, "
                     f"{counts['MEDIUM']} MEDIUM, {counts['HIGH']} HIGH")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        ax.legend(handles=[Patch(facecolor=v, edgecolor="black", label=k)
                           for k, v in cc.items()] +
                          [Line2D([0], [0], color=c, linestyle=ls,
                                  label=f"{lbl} threshold")
                           for lbl, ls, c in [("HIGH", "--", "black"),
                                               ("MEDIUM", ":", "gray"),
                                               ("VACUOUS", ":", "red")]],
                  fontsize=7, loc="upper right")
        for bar, k in zip(bars, [r["n_unique"] for r in rows]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05, f"k={k}",
                    ha="center", fontsize=7)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            out = FIG_DIR / f"p5_field_discriminative_entropy.{ext}"
            plt.savefig(out, dpi=140)
            print(f"wrote {out}")
        plt.close(fig)
    except ImportError:
        print("matplotlib not available — skipping figure", file=sys.stderr)

    # console summary
    print()
    print(f"=== MIN-REPORT field discriminative-entropy audit (n={n}) ===")
    for r in rows:
        print(f"  {r['item_key']:6s} {r['item_label']:30s} k={r['n_unique']:3d} "
              f"H={r['shannon_H_bits']:.3f} bits  "
              f"top_freq={r['top_freq']:.3f}  class={r['classification']:8s}  "
              f"valid_rate={r['exhibit11_validation']:.3f}  "
              f"gap={r['gap_validation_minus_discriminative']:+.3f}")
    print()
    print("VACUOUS items with high validation: "
          f"{[r['item_key'] for r in vac_high]}")


if __name__ == "__main__":
    main()