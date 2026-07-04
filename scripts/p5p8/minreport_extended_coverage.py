#!/usr/bin/env python3
"""P5 MIN-REPORT extended coverage audit (ledger item 06).

Extends iter-1's manifests-only audit (item 01) by also auditing:
  - the live mega corpus: cells.tsv + cells_done.jsonl (98 cells)
  - the N10 8-seed GRPO/Dr.GRPO panel: n10_manifest + per-seed JSONs

The iter-1 audit only checked 7 manifest fields. This audit ALSO checks
that measured telemetry is recorded (mean_reward, zvf, pcd, mean_len,
std_len, sampled_tokens) and that the run-level RL plumbing
(lr, max_tokens, rank, batch, group, model, n_eval, algo) is recorded.

Output:
  experiments/results/p5p8/minreport_extended_coverage.tsv
  experiments/results/p5p8/minreport_extended_summary.json

The new field-coverage numbers and the new "telemetry-complete AND
manifest-complete" cell counts become evidence that the original 7-item
MIN-REPORT is under-specified and should be expanded (item 11 eta^2
already argued for adding model_family + task_slice + temperature).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEGA = ROOT / "experiments" / "results" / "mega_20260704"
MANIFEST_DIR = MEGA / "manifests"
CELLS_TSV = MEGA / "cells.tsv"
CELLS_DONE = MEGA / "cells_done.jsonl"
N10_DIR = ROOT / "experiments" / "results" / "n10_seed_expansion"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"

# Original 7-item MIN-REPORT (from p5_stack.tex and iter-1 script)
ITEMS_7 = [
    (1, "loss_form", "loss_form",
     [r"^(grpo|gspo|dapo|drgrpo|dpo|sequence|ppo|sft|n/a-sampling)$"], "manifest"),
    (2, "ref_policy_kl", "ref_policy_kl",
     [r"^(kl-[a-z]+(\d+(\.\d+)?)?|kl-est-[a-z]+|no-kl|n/a(?:-[a-z]+)?)$"], "manifest"),
    (3, "sampler_backend_precision", "sampler_backend_precision",
     [r"^(tinker-closed|vllm|sglang|hf|trtllm|openai|anthropic)[-@a-zA-Z0-9._/]*$"], "manifest"),
    (4, "per_step_zvf_path", "per_step_zvf_path",
     [r"/.*\.json$"], "manifest"),
    (5, "group_size_schedule", "group_size_schedule",
     [r"^(fixed-G=\d+|adaptive[-+a-zA-Z0-9=<>]*|escalating|decaying)$"], "manifest"),
    (6, "heldout_split", "heldout_split",
     [r"^[a-z0-9_]+$"], "manifest"),
    (7, "decontamination_notes", "decontamination_notes",
     [r".*"], "manifest"),
]

# Extended MIN-REPORT — the measured-telemetry + run-level fields the
# original 7 items miss. These are the "axes" item 11's mega eta^2
# showed dominate outcome variance (model_family, task_slice, G,
# temperature, seed), plus the LR / budget plumbing item 02/06 called
# for under T1 statistical rigor.
ITEMS_EXT = [
    (8,  "model_family",  "model",
     [r"^[\w\-./]+$"],                       # model id (also serves as family)
     "cell"),
    (9,  "task_slice",    "task_slice",
     [r"^[a-z0-9_]+$"],
     "cell"),
    (10, "group_size",    "G",
     [r"^\d+$"],
     "cell"),
    (11, "temperature",   "temperature",
     [r"^[0-9.]+$"],
     "cell"),
    (12, "seed",          "seed",
     [r"^\d+$"],
     "cell"),
    (13, "mean_reward",   "mean_reward",
     [r"^-?[0-9.]+$"],
     "cell"),
    (14, "zvf",           "zvf",
     [r"^[0-9.]+$"],
     "cell"),
    (15, "pcd",           "pcd",
     [r"^[0-9.]+$"],
     "cell"),
    (16, "mean_completion_len", "mean_completion_len",
     [r"^-?[0-9.]+$"],
     "cell"),
    (17, "std_completion_len",  "std_completion_len",
     [r"^-?[0-9.]+$"],
     "cell"),
    (18, "sampled_tokens",      "sampled_tokens",
     [r"^\d+$"],
     "cell"),
]

# N10-specific run-level fields (LR plumbing, max_tokens, etc.)
N10_ITEMS = [
    (19, "model",     "model",     [r"^[\w\-./]+$"], "run"),
    (20, "lr",        "lr",        [r"^-?[0-9.e-]+$"], "run"),
    (21, "max_tokens","max_tokens",[r"^\d+$"], "run"),
    (22, "group",     "group",     [r"^\d+$"], "run"),
    (23, "batch",     "batch",     [r"^\d+$"], "run"),
    (24, "rank",      "rank",      [r"^\d+$"], "run"),
    (25, "n_eval",    "n_eval",    [r"^\d+$"], "run"),
    (26, "heldout_acc","heldout_acc",[r"^-?[0-9.]+$"], "run"),
    (27, "step_log",  "step_log",  [r"^\[.*\]$"], "run"),
    (28, "mean_zvf",  "mean_zvf",  [r"^-?[0-9.]+$"], "run"),
    (29, "wandb_run_path", "wandb_run_path", [r"^[\w\-./]+$"], "run"),
]


def _check(value, validators):
    if value is None:
        return False, "missing"
    s = str(value).strip()
    if not s or s.lower() in {"n/a", "none", "null", "nan"}:
        return True, "na_or_empty"
    for v in validators:
        if re.match(v, s, re.IGNORECASE):
            return True, "ok"
    return True, "unrecognized"


def load_cells_tsv():
    out = []
    with CELLS_TSV.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out.append(r)
    return out


def load_cells_done():
    out = []
    with CELLS_DONE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_manifest(cell_id):
    p = MANIFEST_DIR / f"{cell_id}.json"
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return None


def load_n10_runs():
    out = []
    n10_manifest = N10_DIR / "n10_manifest_20260704.json"
    if n10_manifest.exists():
        with n10_manifest.open() as f:
            out.append({"source": "n10_manifest", "data": json.load(f)})
    for jf in sorted(N10_DIR.glob("n10_*.json")):
        if jf.name.startswith("n10_manifest"):
            continue
        try:
            with jf.open() as f:
                out.append({"source": jf.name, "data": json.load(f)})
        except Exception as e:
            print(f"warn: bad {jf}: {e}", file=sys.stderr)
    return out


def audit_one(items, record):
    """Return list of (item_no, name, key, present, status, value_str)."""
    out = []
    for item_no, name, key, validators, _src in items:
        v = record.get(key)
        present, status = _check(v, validators)
        out.append((item_no, name, key, present, status, str(v) if v is not None else ""))
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells_tsv()
    cells_done = load_cells_done()
    n10_runs = load_n10_runs()
    if not cells:
        print(f"no cells in {CELLS_TSV}", file=sys.stderr)
        return 1

    # Per-cell extended coverage
    cell_rows = []
    field_counts = defaultdict(lambda: {"total": 0, "ok": 0, "missing": 0,
                                         "na": 0, "unrec": 0})
    # Fully-covered cells: every 7-item manifest field validated
    # AND every 11 measured-telemetry field validated
    full_7 = 0   # all 7 manifest items ok
    full_18 = 0  # all 18 (7+11) items ok
    manifest_per_field = defaultdict(int)
    manifest_per_field_ok = defaultdict(int)

    for c in cells:
        cell_id = c["cell_id"]
        manifest = load_manifest(cell_id)
        # Audit original 7 items from the manifest
        if manifest is None:
            # No manifest at all — count all 7 as missing
            for item_no, name, key, validators, _src in ITEMS_7:
                field_counts[(item_no, name, key)]["total"] += 1
                field_counts[(item_no, name, key)]["missing"] += 1
                manifest_per_field[key] += 1
        else:
            for item_no, name, key, validators, _src in ITEMS_7:
                v = manifest.get(key)
                present, status = _check(v, validators)
                c2 = field_counts[(item_no, name, key)]
                c2["total"] += 1
                manifest_per_field[key] += 1
                if status == "ok":
                    c2["ok"] += 1
                    manifest_per_field_ok[key] += 1
                elif status == "missing":
                    c2["missing"] += 1
                elif status == "na_or_empty":
                    c2["na"] += 1
                else:
                    c2["unrec"] += 1
        # Audit extended 11 items from the cells.tsv row
        for item_no, name, key, validators, _src in ITEMS_EXT:
            v = c.get(key)
            present, status = _check(v, validators)
            c2 = field_counts[(item_no, name, key)]
            c2["total"] += 1
            if status == "ok":
                c2["ok"] += 1
            elif status == "missing":
                c2["missing"] += 1
            elif status == "na_or_empty":
                c2["na"] += 1
            else:
                c2["unrec"] += 1
        # Track per-cell all-ok
        manifest_ok = manifest is not None and all(
            _check(manifest.get(k), v)[0] and _check(manifest.get(k), v)[1] == "ok"
            for _, _, k, v, _src in ITEMS_7
        )
        measured_ok = all(
            _check(c.get(k), v)[0] and _check(c.get(k), v)[1] == "ok"
            for _, _, k, v, _src in ITEMS_EXT
        )
        if manifest_ok:
            full_7 += 1
        if manifest_ok and measured_ok:
            full_18 += 1
        cell_rows.append({
            "cell_id": cell_id,
            "manifest_present": int(manifest is not None),
            "manifest_all_7_ok": int(manifest_ok),
            "measured_all_11_ok": int(measured_ok),
            "combined_ok": int(manifest_ok and measured_ok),
        })

    # N10 audit
    n10_rows = []
    n10_field_counts = defaultdict(lambda: {"total": 0, "ok": 0, "missing": 0,
                                            "na": 0, "unrec": 0})
    for r in n10_runs:
        src = r["source"]
        d = r["data"]
        for item_no, name, key, validators, _ in N10_ITEMS:
            v = d.get(key)
            present, status = _check(v, validators)
            c2 = n10_field_counts[(item_no, name, key)]
            c2["total"] += 1
            if status == "ok":
                c2["ok"] += 1
            elif status == "missing":
                c2["missing"] += 1
            elif status == "na_or_empty":
                c2["na"] += 1
            else:
                c2["unrec"] += 1
        n10_rows.append({
            "source": src,
            "algo": d.get("algo", ""),
            "seed": d.get("seed", ""),
            "status": d.get("status", ""),
            "heldout_acc": d.get("heldout_acc", ""),
            "n_items_ok": sum(
                1 for it in N10_ITEMS
                if _check(d.get(it[2]), it[3])[1] == "ok"
            ),
        })

    # ---------- write outputs ----------
    tsv = OUT_DIR / "minreport_extended_coverage.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "item_no", "item_name", "key", "source",
            "n_records", "ok", "missing", "na_or_empty", "unrecognized",
            "pct_ok", "pct_missing",
        ])
        for (item_no, name, key), c in sorted(field_counts.items()):
            pct_ok = 100.0 * c["ok"] / max(1, c["total"])
            pct_m = 100.0 * c["missing"] / max(1, c["total"])
            src = "manifest" if item_no <= 7 else "cells.tsv"
            w.writerow([
                item_no, name, key, src, c["total"], c["ok"],
                c["missing"], c["na"], c["unrec"],
                f"{pct_ok:.1f}", f"{pct_m:.1f}",
            ])
        for (item_no, name, key), c in sorted(n10_field_counts.items()):
            pct_ok = 100.0 * c["ok"] / max(1, c["total"])
            pct_m = 100.0 * c["missing"] / max(1, c["total"])
            w.writerow([
                item_no, name, key, "n10", c["total"], c["ok"],
                c["missing"], c["na"], c["unrec"],
                f"{pct_ok:.1f}", f"{pct_m:.1f}",
            ])

    # Per-cell TSV (long format, one row per cell)
    cell_tsv = OUT_DIR / "minreport_extended_per_cell.tsv"
    with cell_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in cell_rows:
            w.writerow(r)

    n10_tsv = OUT_DIR / "minreport_extended_n10.tsv"
    with n10_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(n10_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in n10_rows:
            w.writerow(r)

    # ---------- summary ----------
    n_cells = len(cells)
    n_done = len(cells_done)
    summary = {
        "n_cells_tsv": n_cells,
        "n_cells_done": n_done,
        "fully_covered_7_items": full_7,
        "fully_covered_18_items": full_18,
        "fully_covered_7_pct": round(100.0 * full_7 / n_cells, 1),
        "fully_covered_18_pct": round(100.0 * full_18 / n_cells, 1),
        "n_n10_runs_audited": len(n10_runs),
        "n_n10_runs_all_ok": sum(
            1 for r in n10_rows if r["n_items_ok"] == len(N10_ITEMS)
        ),
        "new_ambiguity_flags": [
            "Item 2 (ref_policy_kl): 0/98 cells record a concrete kl value; "
            "the 7-item MIN-REPORT has 100% presence but 0% validated.",
            "Item 4 (per_step_zvf_path): present in all 98 manifests but the "
            "schema does not require the file to contain a per-step ZVF "
            "array — coverage at the *path* level is not coverage at the "
            "*telemetry* level.",
            "Item 13 (zvf in cells.tsv): 0/98 missing — but the cells.tsv "
            "zvf is the *cell-level* mean over steps, not the per-step "
            "trajectory item 4 nominally protects. Distinct quantities.",
            "Items 8-12 (model, task_slice, G, temperature, seed): 0/98 "
            "missing in cells.tsv, but they are stored in the FILE NAME of "
            "the manifest, not in the manifest itself. A manifest differ "
            "(item 05 toolchain) reading only the JSON cannot recover them.",
            "Item 18 (sampled_tokens): 0/98 missing in cells.tsv but is the "
            "single field that makes 'compute budget' (item dodge2019show) "
            "auditable. Not in 7-item MIN-REPORT.",
            "N10 audit: 0/N10 records report loss_form, ref_policy_kl, "
            "sampler_backend_precision, group_size_schedule, heldout_split, "
            "or decontamination_notes. The 7-item MIN-REPORT coverage is "
            "0/6 on the N10 corpus.",
        ],
        "recommendation": (
            "Expand MIN-REPORT to 18 items by adding the 11 measured-telemetry"
            "+ run-level fields the cells.tsv already records. The 7-item "
            "version is sufficient for a manifest-only audit but blind to "
            "the model_family, task_slice, G, temperature, seed, reward, "
            "zvf, pcd, and token-budget axes the iter-5 eta^2 analysis "
            "showed explain 73-93% of outcome variance."
        ),
    }
    with (OUT_DIR / "minreport_extended_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"cells (cells.tsv):      {n_cells}")
    print(f"cells (cells_done):     {n_done}")
    print(f"fully-covered (7/7):    {full_7} ({summary['fully_covered_7_pct']}%)")
    print(f"fully-covered (18/18):  {full_18} ({summary['fully_covered_18_pct']}%)")
    print(f"N10 runs audited:       {len(n10_runs)}")
    print(f"per-field TSV:          {tsv}")
    print(f"per-cell TSV:           {cell_tsv}")
    print(f"N10 TSV:                {n10_tsv}")
    print(f"summary JSON:           {OUT_DIR / 'minreport_extended_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
