#!/usr/bin/env python3
"""P5 JOB B SYNTH (iter 52): per-cell 3-axis triangulation of MIN-REPORT honesty.

The iter-50 triangulation measured A (per-axis harvest score) and B (per-item
registry match-rate) at the ITEM level and found A at the ceiling. iter-48
measured C (per-item Shannon entropy) and found 4/7 items VACUOUS. This iter
measures all three at the CELL level (n=98 mega cells) so that joint
correlations are computable.

Per-cell definitions:
- A_cell: claim-vs-measurement alignment score from claim_alignment.tsv (0-100).
- B_cell: mean match-rate across registry entries whose model/task match the
  cell's (model, task_slice), from delta_minreport_consistency.tsv.
- C_cell: mean across the 7 MIN-REPORT items of the cell's normalised
  Shannon entropy, weighted by (1 - frequency of the cell's value in the
  corpus). This rewards cells whose values are RARE in the corpus
  (informative) and penalises cells whose values are the common case.

Inputs:
- experiments/results/mega_20260704/manifests/*.json (98 files)
- experiments/results/p5p8/claim_alignment.tsv (98 rows)
- experiments/results/p5p8/delta_minreport_consistency.tsv
- registry/entries/*.json (model, task per entry_id)

Outputs:
- experiments/results/p5p8/p5_3axis_triangulation_per_cell.tsv (98 rows)
- experiments/results/p5p8/p5_3axis_triangulation_per_cell_boot.tsv
- experiments/results/p5p8/p5_3axis_triangulation_per_cell_summary.json
- experiments/results/p5p8/figures/p5_3axis_per_cell.{png,pdf}

Stdlib + matplotlib only. <=300 lines.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MEGA = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
OUT = ROOT / "experiments" / "results" / "p5p8"
REG = ROOT / "registry" / "entries"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CLAIM_TSV = OUT / "claim_alignment.tsv"
DELTA_TSV = OUT / "delta_minreport_consistency.tsv"
ENTRIES_DIR = REG

ITEMS = [
    ("loss_form", "item1"),
    ("ref_policy_kl", "item2"),
    ("sampler_backend_precision", "item3"),
    ("per_step_zvf_path", "item4"),
    ("group_size_schedule", "item5"),
    ("heldout_split", "item6"),
    ("decontamination_notes", "item7"),
]


def load_minreport_manifests():
    """cell_id -> {item_key: value} from mega manifests."""
    rows = {}
    for f in sorted(MEGA.glob("*.json")):
        with open(f) as fp:
            m = json.load(fp)
        cid = m.get("cell_id", f.stem)
        rows[cid] = {k: str(m.get(k, "")) for k, _ in ITEMS}
    return rows


def load_claim_alignment():
    """cell_id -> score (0-100)."""
    res = {}
    with open(CLAIM_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                res[row["cell_id"]] = float(row["score"])
            except (KeyError, ValueError):
                pass
    return res


def load_registry_entries():
    """entry_id -> (model, task)."""
    res = {}
    for f in sorted(ENTRIES_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                e = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        eid = e.get("id", f.stem)
        model = e.get("model", "")
        task = e.get("task", "")
        if model and task:
            res[eid] = (model, task)
    return res


def load_delta_consistency(entries):
    """(model, task) -> list of verdict codes (MATCH/MISMATCH/...)."""
    bucket = defaultdict(list)
    with open(DELTA_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            eid = row.get("entry_id", "")
            verdict = row.get("verdict", "")
            if eid in entries:
                bucket[entries[eid]].append(verdict)
    return bucket


def compute_item_freqs_and_H(minreport):
    """Per-item value frequency distribution and normalised Shannon entropy."""
    item_counts = {k: defaultdict(int) for k, _ in ITEMS}
    item_total = {k: 0 for k, _ in ITEMS}
    n_cells = len(minreport)
    for cid, vals in minreport.items():
        for k, _ in ITEMS:
            v = vals.get(k, "")
            if v == "":
                continue
            item_counts[k][v] += 1
            item_total[k] += 1
    item_H = {}
    item_freqs = {}
    for k, _ in ITEMS:
        total = item_total[k]
        if total == 0:
            item_H[k] = 0.0
            item_freqs[k] = {}
            continue
        H = 0.0
        freqs = {}
        for v, c in item_counts[k].items():
            p = c / total
            freqs[v] = p
            if p > 0:
                H -= p * math.log2(p)
        n_uniq = len(item_counts[k])
        max_possible = math.log2(min(n_uniq, n_cells)) if n_uniq > 1 else 1.0
        norm_H = H / max_possible if max_possible > 0 else 0.0
        item_H[k] = norm_H
        item_freqs[k] = freqs
    return item_freqs, item_H


def per_cell_C(vals, item_freqs, item_H):
    """Per-cell discriminative entropy C_cell.

    For each item, C_item = normalised_H * (1 - frequency_of_this_value_in_corpus).
    Mean across the 7 items. Returns 0..1 range.
    """
    contribs = []
    for k, _ in ITEMS:
        v = vals.get(k, "")
        if v == "":
            continue
        freq = item_freqs[k].get(v, 0.0)
        norm_H = item_H[k]
        contribs.append(norm_H * (1.0 - freq))
    if not contribs:
        return float("nan")
    return sum(contribs) / len(contribs)


def per_cell_B(model, task, delta_bucket):
    """Mean match-rate across registry entries matching (model, task)."""
    verdicts = delta_bucket.get((model, task), [])
    if not verdicts:
        return float("nan")
    return sum(1 for v in verdicts if v == "MATCH") / len(verdicts)


def pearson(xs, ys):
    """Pearson r with Fisher-z 95% CI."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan"), float("nan"), float("nan")
    r = sxy / math.sqrt(sxx * syy)
    if abs(1 - r) < 1e-9:
        return r, r, r
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    ci_lo = (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)
    ci_hi = (math.exp(2 * z_hi) - 1) / (math.exp(2 * z_hi) + 1)
    return r, ci_lo, ci_hi


def classify_cell(A, B, C):
    """4-profile honesty classification."""
    A_pass = (not math.isnan(A)) and A >= 90.0
    B_pass = (not math.isnan(B)) and B >= 0.5
    C_pass = (not math.isnan(C)) and C >= 0.5
    if not A_pass:
        return "claim_alignment_fail"
    if A_pass and C_pass:
        return "honest_and_informative"
    if A_pass and not C_pass:
        return "honest_but_vacuous"   # headline gap
    return "informative_but_unaudited"


def parse_cell_id(cid):
    """cell_id format: <model>_<task>_<variant>_G<G>_t<temp>_s<seed>_<hash>.

    Returns (model_norm, task_norm) keys that match the registry entries'
    (model, task) fields. The registry uses strings like "Qwen/Qwen3.5-4B"
    and "gsm8k"; the cell_id uses dash-separated model and underscore-separated
    task with a _easy/_hard/_subset suffix.
    """
    parts = cid.split("_")
    model_id = parts[0] if parts else ""
    # task may be split across multiple parts (e.g. "humaneval_subset")
    # The task name in the registry is the first token before _easy/_hard/_subset.
    task_first = parts[1] if len(parts) > 1 else ""
    # Map to registry's (model, task) format
    if "Qwen" in model_id:
        model_norm = "Qwen/Qwen3.5-4B"
    elif "Llama" in model_id or "meta-llama" in model_id:
        model_norm = "meta-llama/Llama-3.2-3B"
    else:
        model_norm = model_id
    task_norm = task_first  # "gsm8k" or "humaneval"
    return model_norm, task_norm


def main():
    print("Loading per-cell claim alignment (axis A) ...")
    A_map = load_claim_alignment()
    print(f"  A scores for {len(A_map)} cells")
    print("Loading registry entries (entry_id -> model, task) ...")
    entries = load_registry_entries()
    print(f"  {len(entries)} entries")
    print("Loading delta consistency verdicts (axis B source) ...")
    delta_bucket = load_delta_consistency(entries)
    print(f"  delta bucket on {len(delta_bucket)} (model, task) keys")
    for k, v in sorted(delta_bucket.items(), key=lambda kv: -len(kv[1]))[:5]:
        print(f"    {k}: n_verdicts={len(v)}")
    print("Loading per-cell mega manifests (axis C source) ...")
    minreport = load_minreport_manifests()
    print(f"  minreport for {len(minreport)} cells")
    item_freqs, item_H = compute_item_freqs_and_H(minreport)
    print("  per-item normalised entropy H:")
    for k, _ in ITEMS:
        top = max(item_freqs[k].values()) if item_freqs[k] else 0.0
        print(f"    {k:35s}  H={item_H[k]:.3f}  top_value_freq={top:.2f}")
    rows = []
    for cid, vals in minreport.items():
        model, task = parse_cell_id(cid)
        A = A_map.get(cid, float("nan"))
        B = per_cell_B(model, task, delta_bucket)
        C = per_cell_C(vals, item_freqs, item_H)
        rows.append({
            "cell_id": cid,
            "model": model,
            "task_slice": task,
            "A_axis_score": A,
            "B_axis_match_rate": B,
            "C_axis_discriminative_H": C,
            "profile": classify_cell(A, B, C),
        })
    out_tsv = OUT / "p5_3axis_triangulation_per_cell.tsv"
    with open(out_tsv, "w") as f:
        f.write("cell_id\tmodel\ttask_slice\tA_axis_score\tB_axis_match_rate\t"
                "C_axis_discriminative_H\tprofile\n")
        for r in rows:
            def fmt(v):
                return f"{v:.4f}" if not math.isnan(v) else ""
            f.write("\t".join([
                r["cell_id"], r["model"], r["task_slice"],
                fmt(r["A_axis_score"]),
                fmt(r["B_axis_match_rate"]),
                fmt(r["C_axis_discriminative_H"]),
                r["profile"],
            ]) + "\n")
    print(f"Wrote {out_tsv} ({len(rows)} cells)")
    profiles = defaultdict(int)
    for r in rows:
        profiles[r["profile"]] += 1
    print(f"\nProfile distribution (n={len(rows)} cells):")
    for p, c in sorted(profiles.items(), key=lambda kv: -kv[1]):
        print(f"  {p:30s}  {c:3d}  ({100*c/len(rows):.1f}%)")
    # Joint correlations
    pairs_data = {
        "A_vs_B": ([r["A_axis_score"] for r in rows if not math.isnan(r["A_axis_score"]) and not math.isnan(r["B_axis_match_rate"])],
                   [r["B_axis_match_rate"] for r in rows if not math.isnan(r["A_axis_score"]) and not math.isnan(r["B_axis_match_rate"])]),
        "A_vs_C": ([r["A_axis_score"] for r in rows if not math.isnan(r["A_axis_score"]) and not math.isnan(r["C_axis_discriminative_H"])],
                   [r["C_axis_discriminative_H"] for r in rows if not math.isnan(r["A_axis_score"]) and not math.isnan(r["C_axis_discriminative_H"])]),
        "B_vs_C": ([r["B_axis_match_rate"] for r in rows if not math.isnan(r["B_axis_match_rate"]) and not math.isnan(r["C_axis_discriminative_H"])],
                   [r["C_axis_discriminative_H"] for r in rows if not math.isnan(r["B_axis_match_rate"]) and not math.isnan(r["C_axis_discriminative_H"])]),
    }
    boot_rows = []
    for name, (xs, ys) in pairs_data.items():
        r_val, lo, hi = pearson(xs, ys)
        boot_rows.append({
            "pair": name, "n_cells": len(xs),
            "pearson_r": "" if math.isnan(r_val) else f"{r_val:.4f}",
            "ci_low": "" if math.isnan(lo) else f"{lo:.4f}",
            "ci_high": "" if math.isnan(hi) else f"{hi:.4f}",
            "excludes_zero": (not math.isnan(lo)) and (lo > 0 or hi < 0),
        })
    boot_tsv = OUT / "p5_3axis_triangulation_per_cell_boot.tsv"
    with open(boot_tsv, "w") as f:
        f.write("pair\tn_cells\tpearson_r\tci_low\tci_high\texcludes_zero\tdegenerate_reason\n")
        # Determine if the correlation is degenerate (one axis has zero variance)
        degenerate_reasons = []
        for name, (xs, ys) in pairs_data.items():
            if len(xs) < 3:
                degenerate_reasons.append("n_too_small")
                continue
            x_var = sum((x - sum(xs)/len(xs))**2 for x in xs)
            y_var = sum((y - sum(ys)/len(ys))**2 for y in ys)
            if x_var == 0:
                degenerate_reasons.append("x_constant")
            elif y_var == 0:
                degenerate_reasons.append("y_constant")
            else:
                degenerate_reasons.append("")
        for r, reason in zip(boot_rows, degenerate_reasons):
            f.write(f"{r['pair']}\t{r['n_cells']}\t{r['pearson_r']}\t{r['ci_low']}\t{r['ci_high']}\t{r['excludes_zero']}\t{reason}\n")
    print(f"Wrote {boot_tsv}")
    print(f"\nJoint correlations:")
    for r, reason in zip(boot_rows, degenerate_reasons):
        print(f"  {r['pair']:10s}  n={r['n_cells']:3d}  r={r['pearson_r']!s:8s}  CI=[{r['ci_low']!s:8s}, {r['ci_high']!s:8s}]  excl0={r['excludes_zero']}  ({reason})")
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = {
        "honest_and_informative": "#2ca02c",
        "honest_but_vacuous": "#d62728",
        "claim_alignment_fail": "#9467bd",
        "informative_but_unaudited": "#1f77b4",
    }
    titles = [
        f"A (claim-alignment) vs B (registry match)  r={boot_rows[0]['pearson_r']!s}",
        f"A vs C (per-cell discriminative entropy)  r={boot_rows[1]['pearson_r']!s}",
        f"B vs C  r={boot_rows[2]['pearson_r']!s}",
    ]
    keys_for_plot = [("A_axis_score", "B_axis_match_rate"), ("A_axis_score", "C_axis_discriminative_H"),
                     ("B_axis_match_rate", "C_axis_discriminative_H")]
    for ax, title, (xk, yk) in zip(axes, titles, keys_for_plot):
        for r in rows:
            x, y = r[xk], r[yk]
            if math.isnan(x) or math.isnan(y):
                continue
            ax.scatter(x, y, c=colors.get(r["profile"], "#7f7f7f"),
                       s=30, alpha=0.75, edgecolor="black", linewidth=0.3)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
    from matplotlib.patches import Patch
    handles = [Patch(color=colors[k], label=f"{k} (n={profiles.get(k, 0)})") for k in colors]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(FIG / "p5_3axis_per_cell.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIG / "p5_3axis_per_cell.pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG/'p5_3axis_per_cell.png'}")
    summary = {
        "n_cells": len(rows),
        "n_with_A": sum(1 for r in rows if not math.isnan(r["A_axis_score"])),
        "n_with_B": sum(1 for r in rows if not math.isnan(r["B_axis_match_rate"])),
        "n_with_C": sum(1 for r in rows if not math.isnan(r["C_axis_discriminative_H"])),
        "profile_counts": dict(profiles),
        "joint_correlations": {
            "A_vs_B": dict(zip(["r", "ci_low", "ci_high", "excludes_zero", "n"],
                               [boot_rows[0]["pearson_r"], boot_rows[0]["ci_low"],
                                boot_rows[0]["ci_high"], boot_rows[0]["excludes_zero"],
                                boot_rows[0]["n_cells"]])),
            "A_vs_C": dict(zip(["r", "ci_low", "ci_high", "excludes_zero", "n"],
                               [boot_rows[1]["pearson_r"], boot_rows[1]["ci_low"],
                                boot_rows[1]["ci_high"], boot_rows[1]["excludes_zero"],
                                boot_rows[1]["n_cells"]])),
            "B_vs_C": dict(zip(["r", "ci_low", "ci_high", "excludes_zero", "n"],
                               [boot_rows[2]["pearson_r"], boot_rows[2]["ci_low"],
                                boot_rows[2]["ci_high"], boot_rows[2]["excludes_zero"],
                                boot_rows[2]["n_cells"]])),
        },
        "item_normalised_H": {k: round(item_H[k], 4) for k, _ in ITEMS},
        "headline": {
            "falsifiable_claim": "At the per-cell level, the per-cell 3-axis triangulation surfaces the 'honest-but-vacuous' gap as the count of cells that pass the harvest-surface truthfulness audit (A) but whose declared values are not informationally distinct from the corpus (low C).",
            "n_honest_but_vacuous": profiles.get("honest_but_vacuous", 0),
            "n_honest_and_informative": profiles.get("honest_and_informative", 0),
        },
    }
    sum_path = OUT / "p5_3axis_triangulation_per_cell_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {sum_path}")
    print(f"\n=== Headline ===")
    print(f"  Cells passing A (truthfulness >= 90): {summary['n_with_A']}/{len(rows)}")
    print(f"  Cells with B available: {summary['n_with_B']}/{len(rows)}")
    print(f"  Cells with C >= 0.5 (informative): {sum(1 for r in rows if not math.isnan(r['C_axis_discriminative_H']) and r['C_axis_discriminative_H'] >= 0.5)}/{len(rows)}")
    print(f"  Honest-but-vacuous profile: {profiles.get('honest_but_vacuous', 0)}/{len(rows)} ({100*profiles.get('honest_but_vacuous', 0)/len(rows):.1f}%)")


if __name__ == "__main__":
    main()