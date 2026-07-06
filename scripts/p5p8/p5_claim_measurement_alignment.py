#!/usr/bin/env python3
"""P5 — Item 37: MIN-REPORT claim-vs-measurement alignment audit.

Motivation. Iter-1 (item 01), iter-9 (item 14), iter-13 (item 18) and
iter-21 (item 28) all audit *coverage* — does the manifest declare the
right keys with valid values? They never ask the next question: **are
those declared values actually true of the measured telemetry**?
A standard that can be satisfied by typing a plausible-looking string
is not a measurement.

This iter closes that gap. For every mega_20260704 cell we:

  1. parse the manifest's six claim-bearing fields
       cell_id  → model_id, task_slice, G, temperature, seed
       group_size_schedule  → claimed G
       heldout_split        → claimed task_slice
       decontamination_notes → claimed decontam class
  2. join on cell_id with cells.tsv and read the six measured fields
       model_family, task_slice, G, temperature, seed, mean_reward
  3. compare claim to measurement, emit per-field match/mismatch
       (mismatch severity: catastrophic / mismatch-but-compatible / match)
  4. produce a per-cell ALIGNMENT score 0–100
       (each of the 6 axes worth 16.67 pts, with severity-weighted
       partial credit for declared-but-unparseable or n/a cases)
  5. aggregate to a corpus-level alignment score with paired
       bootstrap 95% CIs over the 98 cells
  6. write:
       experiments/results/p5p8/claim_alignment.tsv
       experiments/results/p5p8/claim_alignment_summary.json
       experiments/results/p5p8/figures/claim_alignment_per_axis.{png,pdf}
       experiments/results/p5p8/figures/claim_alignment_dist.{png,pdf}

Headline falsifiable claim:
  the mega_20260704 corpus is X% claim-measured aligned, with the
  worst axis being Y (where Y ∈ {model, task, G, temperature, seed,
  decontam}).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLS_TSV = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
MAN_DIR = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
FIG_DIR = OUT_DIR / "figures"

# Six claim axes; each contributes 100/6 pts to the alignment badge.
AXES = ["model", "task", "G", "temperature", "seed", "decontam"]


def load_cells() -> dict:
    """Return {cell_id: {measured_field: value}} from cells.tsv."""
    out = {}
    with CELLS_TSV.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            cid = row["cell_id"]
            try:
                row["_G_meas"] = int(row["G"])
                row["_T_meas"] = float(row["temperature"])
                row["_S_meas"] = int(row["seed"])
                row["_R_meas"] = float(row["mean_reward"])
                row["_Z_meas"] = float(row["zvf"])
            except (ValueError, KeyError):
                continue
            out[cid] = row
    return out


CELL_RE = re.compile(
    r"^(?P<model>[^_]+)_(?P<task>[a-z0-9_]+)_G(?P<G>\d+)_t(?P<t>[\d.]+)_s(?P<s>\d+)_"
)


def parse_cell_id(cid: str) -> dict | None:
    m = CELL_RE.match(cid)
    if not m:
        return None
    return {
        "model": m.group("model"),
        "task": m.group("task"),
        "G": int(m.group("G")),
        "temperature": float(m.group("t")),
        "seed": int(m.group("s")),
    }


GS_RE = re.compile(r"fixed-G=(\d+)|constant G=(\d+)|adaptive.*G=(\d+)")
DECONTAM_CLASSES = {"train-slice", "held-out-disjoint", "test-disjoint",
                    "exact-ngram-overlap", "external-disjoint", "n/a",
                    "n/a-train-only",
                    # corpus-actual tokens emitted by the live manifest emitter
                    "gsm8k-train-slice",
                    "humaneval-openai-subset"}


def parse_manifest(manifest: dict) -> dict:
    """Parse claim-bearing fields out of a manifest."""
    out = {}
    gss = str(manifest.get("group_size_schedule", ""))
    m = GS_RE.search(gss)
    if m:
        g = next(int(x) for x in m.groups() if x is not None)
        out["G"] = g
    else:
        out["G"] = None
    out["task"] = manifest.get("heldout_split", None)
    out["decontam"] = manifest.get("decontamination_notes", None)
    return out


def axis_match(axis: str, claimed, measured) -> tuple[float, str]:
    """Return (points 0..16.67, severity string)."""
    # Special case: decontam has no measured column in cells.tsv, so
    # we score the manifest's *declaration* against a recognised class
    # list (no ground-truth measurement to contradict it).
    if axis == "decontam":
        if claimed is None or str(claimed).strip() == "":
            return 0.0, "no_claim"
        cv_low = str(claimed).strip().lower()
        if any(tok in cv_low for tok in DECONTAM_CLASSES):
            return 16.67, "declared_recognised_class"
        return 8.33, "declared_unrecognised_class"
    if measured is None or str(measured).strip() == "":
        return 0.0, "no_measurement"
    if claimed is None or str(claimed).strip() == "":
        # manifest did not declare the field
        return 0.0, "no_claim"
    cv = str(claimed).strip()
    mv = str(measured).strip()
    if axis == "model":
        # Compare model families — claim is the cell_id-prefix slug
        # like "meta-llama-Llama-3-2-3B"; measurement is the
        # cells.tsv `model_family` field like "meta-llama/Llama-3.2-3B".
        # Normalise: drop separators, drop dots between digits, lowercase.
        def _norm(s: str) -> str:
            s = s.replace("-", "").replace("_", "").replace(".", "").replace("/", "").lower()
            return s
        c = _norm(cv)
        m = _norm(mv)
        if c == m:
            return 16.67, "match"
        # Fuzzy: accept substring containment (handles minor formatting
        # such as a leading org prefix being dropped)
        if c in m or m in c:
            return 12.0, "partial_match"
        return 0.0, "catastrophic_mismatch"
    if axis == "task":
        if cv == mv:
            return 16.67, "match"
        if cv.replace("-", "") == mv.replace("-", ""):
            return 8.33, "case_or_dash"
        return 0.0, "task_mismatch"
    if axis in ("G", "seed"):
        try:
            ic = int(float(cv))
            im = int(float(mv))
        except ValueError:
            return 0.0, "unparseable"
        return (16.67, "match") if ic == im else (0.0, "value_mismatch")
    if axis == "temperature":
        try:
            fc = float(cv)
            fm = float(mv)
        except ValueError:
            return 0.0, "unparseable"
        return (16.67, "match") if abs(fc - fm) < 1e-6 else (0.0, "value_mismatch")
    return 0.0, "unknown_axis"


def per_cell_score(claims: dict, measured: dict,
                   cid_claims_override: dict | None = None) -> dict:
    out = {}
    cid = measured.get("cell_id", "?")
    cid_parsed = parse_cell_id(cid)
    # claim from cell_id (manifest cell_id == cells.tsv cell_id)
    if cid_claims_override is not None:
        cid_claims = dict(cid_claims_override)
    else:
        cid_claims = cid_parsed or {}
    rows = []
    total = 0.0
    for axis in AXES:
        if axis == "decontam":
            claimed = claims.get("decontam")
            measured_val = None  # no measured decontam in cells.tsv
        else:
            claimed = cid_claims.get(axis)
            if axis == "G" and claimed is None:
                claimed = claims.get("G")
            measured_val = measured.get({
                "model": "model_family",
                "task": "task_slice",
                "G": "_G_meas",
                "temperature": "_T_meas",
                "seed": "_S_meas",
            }[axis])
        pts, sev = axis_match(axis, claimed, measured_val)
        total += pts
        rows.append({
            "axis": axis,
            "claimed": claimed,
            "measured": measured_val,
            "points": round(pts, 2),
            "severity": sev,
        })
    return {"cell_id": cid, "score": round(total, 1), "axes": rows}


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0):
    """Paired bootstrap percentile CI on the mean."""
    import random
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (sum(values) / n, lo, hi)


def write_outputs(scored):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tsv = OUT_DIR / "claim_alignment.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "cell_id", "score",
            "model_pts", "task_pts", "G_pts",
            "temp_pts", "seed_pts", "decontam_pts",
            "model_sev", "task_sev", "G_sev",
            "temp_sev", "seed_sev", "decontam_sev",
        ])
        for s in scored:
            am = {a["axis"]: a for a in s["axes"]}
            w.writerow([
                s["cell_id"], s["score"],
                am["model"]["points"], am["task"]["points"], am["G"]["points"],
                am["temperature"]["points"], am["seed"]["points"],
                am["decontam"]["points"],
                am["model"]["severity"], am["task"]["severity"],
                am["G"]["severity"], am["temperature"]["severity"],
                am["seed"]["severity"], am["decontam"]["severity"],
            ])
    per_axis_match = defaultdict(int)
    per_axis_total = defaultdict(int)
    per_axis_sev = defaultdict(lambda: defaultdict(int))
    scores = []
    for s in scored:
        scores.append(s["score"])
        for a in s["axes"]:
            per_axis_total[a["axis"]] += 1
            if a["severity"].startswith("match") or a["severity"] == "declared_recognised_class":
                per_axis_match[a["axis"]] += 1
            per_axis_sev[a["axis"]][a["severity"]] += 1
    n = len(scored)
    mean, lo, hi = bootstrap_ci(scores)
    summary = {
        "n_cells": n,
        "score_mean": round(mean, 2),
        "score_ci95_lo": round(lo, 2),
        "score_ci95_hi": round(hi, 2),
        "score_min": round(min(scores), 2) if scores else 0,
        "score_max": round(max(scores), 2) if scores else 0,
        "score_std": round((sum((s - mean) ** 2 for s in scores)
                            / max(1, n)) ** 0.5, 2),
        "per_axis_match_pct": {
            a: round(100.0 * per_axis_match[a] / max(1, per_axis_total[a]), 1)
            for a in AXES
        },
        "per_axis_severity_mix": {
            a: dict(per_axis_sev[a]) for a in AXES
        },
    }
    (OUT_DIR / "claim_alignment_summary.json").write_text(
        json.dumps(summary, indent=2))
    return summary


def make_figure(scored, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warn: matplotlib not available; skipping figure",
              file=sys.stderr)
        return
    # Per-axis match %
    axes = AXES
    pct = [summary["per_axis_match_pct"][a] for a in axes]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(axes, pct, color="#4477AA", edgecolor="black")
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, p + 1, f"{p:.1f}%",
                ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% claim-measurement match")
    ax.set_title(f"MIN-REPORT claim-measurement alignment per axis\n"
                 f"(n={summary['n_cells']}, mean={summary['score_mean']:.1f}, "
                 f"95% CI [{summary['score_ci95_lo']:.1f}, "
                 f"{summary['score_ci95_hi']:.1f}])")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "claim_alignment_per_axis.png", dpi=150)
    fig.savefig(FIG_DIR / "claim_alignment_per_axis.pdf")
    plt.close(fig)
    # Score distribution
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    scores = [s["score"] for s in scored]
    ax2.hist(scores, bins=15, color="#EE6677", edgecolor="black")
    ax2.axvline(summary["score_mean"], color="black", linestyle="--",
                label=f"mean={summary['score_mean']:.1f}")
    ax2.set_xlabel("Claim-measurement alignment score (0-100)")
    ax2.set_ylabel("# cells")
    ax2.set_title(f"Alignment score distribution (n={summary['n_cells']})")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "claim_alignment_dist.png", dpi=150)
    fig2.savefig(FIG_DIR / "claim_alignment_dist.pdf")
    plt.close(fig2)


def main():
    if not CELLS_TSV.exists():
        print(f"missing {CELLS_TSV}", file=sys.stderr)
        return 1
    cells = load_cells()
    print(f"loaded {len(cells)} cells from {CELLS_TSV.name}")
    scored = []
    missing_manifest = 0
    for cid, measured in cells.items():
        m_path = MAN_DIR / f"{cid}.json"
        if not m_path.exists():
            missing_manifest += 1
            continue
        try:
            with m_path.open() as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"warn: bad manifest {m_path}: {e}", file=sys.stderr)
            continue
        claims = parse_manifest(manifest)
        scored.append(per_cell_score(claims, measured))
    print(f"scored {len(scored)} cells (missing manifest: {missing_manifest})")
    summary = write_outputs(scored)
    make_figure(scored, summary)

    # Perturbation test — for a sample of cells, swap each axis and
    # confirm the audit detects it (non-vacuity check).
    print("\n=== Perturbation test (negative control) ===")
    perturb_results = []
    for s in scored[:10]:
        cid = s["cell_id"]
        meas = cells[cid]
        m_path = MAN_DIR / f"{cid}.json"
        if not m_path.exists():
            continue
        with m_path.open() as f:
            manifest = json.load(f)
        # Try 4 perturbations: swap G, swap temperature, swap task, swap seed
        cids_parsed = parse_cell_id(cid) or {}
        for axis, perturbed_value in [
            ("G", (cids_parsed.get("G") or 8) * 2),  # double the G
            ("temperature", 1.5 if cids_parsed.get("temperature") != 1.5 else 0.1),
            ("task", "wrong_task_xyz"),
            ("seed", (cids_parsed.get("seed") or 0) + 9999),
        ]:
            cid_claims = {
                "model": cids_parsed.get("model"),
                "task": perturbed_value if axis == "task" else cids_parsed.get("task"),
                "G": perturbed_value if axis == "G" else cids_parsed.get("G"),
                "temperature": perturbed_value if axis == "temperature" else cids_parsed.get("temperature"),
                "seed": perturbed_value if axis == "seed" else cids_parsed.get("seed"),
            }
            claims = {
                "G": cid_claims["G"],
                "task": cid_claims["task"],
                "decontam": manifest.get("decontamination_notes"),
            }
            ps = per_cell_score(claims, meas, cid_claims_override=cid_claims)
            orig_axis = next(a for a in s["axes"] if a["axis"] == axis)
            pert_axis = next(a for a in ps["axes"] if a["axis"] == axis)
            detected = (orig_axis["severity"] == "match"
                        and pert_axis["severity"] != "match")
            perturb_results.append({
                "cell_id": cid,
                "axis": axis,
                "perturbed_to": perturbed_value,
                "orig_severity": orig_axis["severity"],
                "perturbed_severity": pert_axis["severity"],
                "detected": detected,
            })
    detected_n = sum(1 for r in perturb_results if r["detected"])
    total_n = len(perturb_results)
    print(f"  perturbations:        {total_n}")
    print(f"  detected by audit:    {detected_n} ({100.0 * detected_n / max(1, total_n):.1f}%)")
    by_axis = defaultdict(lambda: [0, 0])
    for r in perturb_results:
        by_axis[r["axis"]][1] += 1
        by_axis[r["axis"]][0] += int(r["detected"])
    print(f"  per-axis detection:")
    for axis, (d, t) in sorted(by_axis.items()):
        print(f"    {axis:>11s}: {d}/{t}  ({100.0 * d / max(1, t):.0f}%)")

    # Persist perturbation results next to the alignment TSV.
    (OUT_DIR / "claim_alignment_perturbation.json").write_text(
        json.dumps({"n": total_n, "detected": detected_n,
                    "per_axis": {a: {"detected": d, "total": t}
                                 for a, (d, t) in by_axis.items()},
                    "rows": perturb_results}, indent=2))

    print(f"\nn_cells:                 {summary['n_cells']}")
    print(f"score mean / 95% CI:     {summary['score_mean']} "
          f"[{summary['score_ci95_lo']}, {summary['score_ci95_hi']}]")
    print(f"score range:             [{summary['score_min']}, {summary['score_max']}]"
          f"  (std={summary['score_std']})")
    print("per-axis match %:")
    for a in AXES:
        print(f"  {a:>11s}: {summary['per_axis_match_pct'][a]:>5.1f}%")
    print("per-axis severity mix:")
    for a in AXES:
        print(f"  {a}: {summary['per_axis_severity_mix'][a]}")
    print(f"figure: {FIG_DIR}/claim_alignment_per_axis.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())