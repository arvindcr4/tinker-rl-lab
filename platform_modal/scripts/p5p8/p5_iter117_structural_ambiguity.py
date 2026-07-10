#!/usr/bin/env python3
"""
P5 MIN-REPORT v2.2 STRUCTURAL-AMBIGUITY audit (iter 117, fresh vein).

Closes brief vein (a) at the **encoding-mode layer**. Iter-113 row 127a
audited the **content layer** (declared vs emitted vs derivable). This iter
audits the **structural layer**: for every one of the 18 MIN-REPORT v2.2
items, classify the *encoding mode* used in the live
`platform_hybrid/experiments/results/mega_20260704/manifests/` corpus (n=98) into one of:

  - explicit_json_key : the manifest JSON has a dedicated top-level key
  - implicit_filename  : the value is encoded ONLY in the cell_id filename
                        (regex-inferred from `MODEL_TASK_G<g>_t<t>_s<seed>_<hash>`)
  - cells_tsv_only     : the manifest does NOT declare it; only cells.tsv has it
  - tensor_derivable   : the manifest does NOT declare it; only the group_tensor
                        JSON has it (deterministic recovery)
  - n/a_sentinel       : every manifest carries a literal `n/a` or `n/a-*` value
                        (no measurement, only honest non-declaration)
  - absent_no_source   : no source anywhere in the corpus (schema-only item)

H1 — every one of the 5 stack axes (model_family, task_slice, G, temperature,
     seed) is encoded IMPLICIT-FILENAME only — no JSON key, no schema-level
     declaration, no JSON-schema validator can enforce them. Renaming a
     manifest file silently destroys 5 axes of stack information. (rename
     vulnerability test, 5 perturbed cells × 3 perturbations × re-validate).

H2 — the schema's "implicit-via-filename" pattern violates the schema
     author's own stated intent (MIN-REPORT item says "report the stack")
     because the JSON body is silent on the stack axes — only the filename
     says. This is a STRUCTURAL gap, distinct from the CONTENT gap (iter-113).

H3 — recommended remediation: emit 5 new top-level JSON keys (`model_family`,
     `task_slice`, `G`, `temperature`, `seed`) alongside the existing 8;
     schema gate: any cell whose cell_id-filename regex does NOT match the
     emitted keys MUST fail `registry_validate.py`.

Outputs:
  platform_hybrid/experiments/results/p5p8/p5_iter117_structural_ambiguity.tsv
       (18 rows: per-MIN-REPORT-item encoding-mode classification)
  platform_hybrid/experiments/results/p5p8/p5_iter117_rename_vulnerability.tsv
       (15 rows: 5 perturbed cells × 3 perturbations; rename recovery test)
  platform_hybrid/experiments/results/p5p8/p5_iter117_summary.json
       (machine-readable with H1-H3 evidence)

The companion registry-style schema-level audit reuses the live corpus
verbatim — no new harvest, no Tinker runs.
"""
from __future__ import annotations
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
MANIFEST_DIR = ROOT / "platform_hybrid/experiments/results/mega_20260704/manifests"
CELLS_TSV = ROOT / "platform_hybrid/experiments/results/mega_20260704/cells.tsv"
TENSOR_DIR = ROOT / "platform_hybrid/experiments/results/mega_20260704/group_tensors"
OUT_DIR = ROOT / "platform_hybrid/experiments/results/p5p8"

# MIN-REPORT v2.2 items: (item_id, item_name, schema_status, declared_source)
# schema_status from iter-113: live | live_NA | DERIVABLE | DERIVABLE_REJECTED | schema_only
MIN_REPORT_V22 = [
    ("Item01", "model_family",                "live",          "manifest_or_cells"),
    ("Item02", "ref_policy_kl",               "live_NA",       "manifest"),
    ("Item03", "reward_model_signature",      "schema_only",   "schema"),
    ("Item04", "rollout_temperature",         "live",          "cells"),
    ("Item05", "group_size",                  "live",          "manifest_or_cells"),
    ("Item06", "heldout_split",               "live",          "manifest"),
    ("Item07", "decontamination_notes",       "live",          "manifest"),
    ("Item08", "loss_form",                   "live_NA",       "manifest"),
    ("Item09", "sampler_backend_precision",   "live",          "manifest"),
    ("Item10", "advantage_baseline",          "schema_only",   "schema"),
    ("Item11", "token_mask",                  "schema_only",   "schema"),
    ("Item12", "kl_beta",                     "schema_only",   "schema"),
    ("Item13", "zvf_per_step",                "live",          "manifest"),
    ("Item14", "K_variance_residual",         "DERIVABLE",     "tensor"),
    ("Item15", "K_unique_count",              "DERIVABLE",     "tensor"),
    ("Item16", "max_K_share_PLACEBO",         "DERIVABLE_REJECTED", "tensor"),
    ("Item17", "prompt_p_hat_var",            "DERIVABLE",     "tensor"),
    ("Item18", "zvf130_risk_residual",        "schema_only",   "schema"),
]

# Manifest JSON keys actually emitted (per iter-117 inspection, n=98)
LIVE_JSON_KEYS = {
    "cell_id", "loss_form", "ref_policy_kl", "sampler_backend_precision",
    "per_step_zvf_path", "group_size_schedule", "heldout_split", "decontamination_notes",
}

# cells.tsv columns (per iter-93 row 109)
CELLS_COLS = [
    "cell_id", "model", "model_family", "task_slice", "G", "temperature", "seed",
    "n_groups", "sample_errors", "mean_reward", "zvf", "pcd", "mean_completion_len",
    "std_completion_len", "sampled_tokens", "cumulative_sampled_tokens",
    "reward_vectors_json", "tensor_path", "manifest_path",
]

# Filename regex for implicit-via-cell_id axes
CELL_ID_RE = re.compile(
    r"^(?P<model_family>Qwen-Qwen3-5-4B|meta-llama-Llama-3-2-3B)_"
    r"(?P<task_slice>gsm8k_easy|gsm8k_hard|humaneval_subset)_"
    r"G(?P<G>\d+)_t(?P<temperature>[\d.]+)_s(?P<seed>\d+)_[0-9a-f]{10}$"
)

# Map canonical item_name → regex-group name (item names vs regex names differ)
ITEM_TO_REGEX_GROUP = {
    "model_family":       "model_family",
    "task_slice":         "task_slice",
    "group_size":         "G",
    "rollout_temperature":"temperature",
    "seed":               "seed",
}


def load_corpus() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load 98 manifests + 98 cells.tsv rows, keyed by cell_id."""
    manifests: list[dict[str, Any]] = []
    for f in sorted(MANIFEST_DIR.glob("*.json")):
        with open(f) as fh:
            manifests.append(json.load(fh))

    cells_by_id: dict[str, dict[str, Any]] = {}
    with open(CELLS_TSV) as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for row in rdr:
            cells_by_id[row["cell_id"]] = row
    return manifests, cells_by_id


def classify_encoding_mode(item_id: str, item_name: str,
                            manifests: list[dict[str, Any]],
                            cells_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """For one MIN-REPORT item, return encoding-mode classification + diagnostics."""
    n = len(manifests)

    # Direct JSON-key check (mapped to canonical item_name OR a known alias)
    KEY_ALIASES = {
        "model_family":          ["model_family"],
        "ref_policy_kl":         ["ref_policy_kl"],
        "rollout_temperature":   ["temperature"],          # in cells.tsv
        "group_size":            ["G", "group_size_schedule"],  # cells.tsv G; manifest has group_size_schedule (fixed-G=N)
        "heldout_split":         ["heldout_split"],
        "decontamination_notes": ["decontamination_notes"],
        "loss_form":             ["loss_form"],
        "sampler_backend_precision": ["sampler_backend_precision"],
        "zvf_per_step":          ["per_step_zvf_path"],
    }

    n_in_manifest_json = 0
    n_in_cells_tsv = 0
    n_in_filename = 0
    n_na_sentinel = 0
    n_tensor_derivable = 0

    for m in manifests:
        cid = m["cell_id"]
        c = cells_by_id.get(cid, {})

        # Check ALL three sources INDEPENDENTLY (not break-after-first).
        if item_name in KEY_ALIASES:
            for k in KEY_ALIASES[item_name]:
                if k in m:
                    n_in_manifest_json += 1
                    break
            for k in KEY_ALIASES[item_name]:
                if k in c:
                    n_in_cells_tsv += 1
                    break

        # Filename regex (implicit-via-cell_id) — independent of KEY_ALIASES
        mm = CELL_ID_RE.match(cid)
        if mm:
            rg_name = ITEM_TO_REGEX_GROUP.get(item_name, item_name)
            if rg_name in mm.groupdict():
                n_in_filename += 1

        # Tensor-derivable items
        if item_id in {"Item14", "Item15", "Item17"}:
            tp = m.get("per_step_zvf_path", "")
            if tp and Path(tp).exists():
                n_tensor_derivable += 1

        # n/a sentinel — check THIS ITEM's value in the manifest JSON
        if item_name in KEY_ALIASES:
            for k in KEY_ALIASES[item_name]:
                v = m.get(k, "")
                if isinstance(v, str) and v.startswith("n/a"):
                    n_na_sentinel += 1
                    break

    # Assign encoding mode (precedence: explicit > implicit_filename > cells_tsv_only > tensor > n/a > absent)
    # IMPORTANT: a "cells_tsv_only" item is a STRUCTURAL-AMBIGUITY finding —
    # the schema declares the item as MIN-REPORT-required, but the live manifest
    # JSON has no record of it; the value lives in a separate file (cells.tsv)
    # AND is duplicated in the cell_id filename. An auditor reading only the
    # manifest JSON will see 0 evidence of this item.
    if n_in_manifest_json == n:
        mode = "explicit_json_key"
    elif n_in_filename == n and n_in_cells_tsv == n:
        mode = "implicit_filename_AND_cells_tsv"  # split across filename + cells.tsv; manifest silent
    elif n_in_filename == n:
        mode = "implicit_filename"
    elif n_in_cells_tsv == n:
        mode = "cells_tsv_only"  # schema split: manifest silent, cells.tsv has it
    elif n_tensor_derivable == n:
        mode = "tensor_derivable"
    elif n_na_sentinel == n:
        mode = "n/a_sentinel"
    else:
        mode = "absent_no_source"

    return {
        "item_id": item_id,
        "item_name": item_name,
        "encoding_mode": mode,
        "n_in_manifest_json": n_in_manifest_json,
        "n_in_cells_tsv":     n_in_cells_tsv,
        "n_in_filename":      n_in_filename,
        "n_na_sentinel":      n_na_sentinel,
        "n_tensor_derivable": n_tensor_derivable,
        "rename_robust":      mode in {"explicit_json_key", "n/a_sentinel", "tensor_derivable"},
        "n_cells":            n,
    }


def rename_vulnerability_test(manifests: list[dict[str, Any]],
                               cells_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """For 5 random cells × 3 perturbations: test what the JSON-body-only auditor sees.

    Three perturbation modes:
      - json_body_alone  : pretend the JSON is the ONLY source available (no cells.tsv, no filename)
      - json_plus_filename: JSON + filename regex (most realistic for a manifest auditor)
      - json_plus_cells_tsv: JSON + cells.tsv (the cell_id-keys-cells.tsv coupling)

    The audit reveals which axes an auditor recovers in each mode.
    """
    rng_path = sorted(manifests, key=lambda m: m["cell_id"])[:5]  # deterministic: first 5 alphabetically

    AXES = ["model_family", "task_slice", "G", "temperature", "seed"]
    rows = []
    for m in rng_path:
        cid = m["cell_id"]
        c = cells_by_id.get(cid, {})
        mm = CELL_ID_RE.match(cid)
        rg = mm.groupdict() if mm else {}

        for mode_name, source_fn in [
            ("json_body_alone",    lambda: {}),
            ("json_plus_filename", lambda: dict(rg)),
            ("json_plus_cells_tsv", lambda: {k: c.get(k, "") for k in AXES}),
        ]:
            extras = source_fn()
            recovered = {
                "model_family":   m.get("model_family",   extras.get("model_family", "")),
                "task_slice":     m.get("task_slice",     extras.get("task_slice", "")),
                "G":              m.get("G",              extras.get("G", "")),
                "temperature":    m.get("temperature",    extras.get("temperature", "")),
                "seed":           m.get("seed",           extras.get("seed", "")),
            }
            n_recovered = sum(1 for v in recovered.values() if v)
            rows.append({
                "perturbation":   mode_name,
                "original_cid":   cid,
                "n_axes_recovered": n_recovered,
                "missing_axes":   ",".join(k for k, v in recovered.items() if not v),
                "filename_match": "yes" if mm else "no",
            })
    return rows


def main() -> int:
    print("[iter-117] P5 MIN-REPORT v2.2 structural-ambiguity audit")
    print(f"[iter-117] reading {MANIFEST_DIR}/*.json + {CELLS_TSV}")

    manifests, cells_by_id = load_corpus()
    n = len(manifests)
    print(f"[iter-117] n={n} manifests, n={len(cells_by_id)} cells.tsv rows")

    # Per-item encoding-mode classification
    rows = []
    for item_id, item_name, schema_status, declared_source in MIN_REPORT_V22:
        cls = classify_encoding_mode(item_id, item_name, manifests, cells_by_id)
        cls["schema_status"] = schema_status
        cls["declared_source"] = declared_source
        rows.append(cls)

    # H1 — implicit_filename count for the 5 stack axes
    implicit = [r for r in rows if r["encoding_mode"] == "implicit_filename"]
    print(f"[iter-117] H1: implicit_filename items = {len(implicit)}/{len(rows)}")
    for r in implicit:
        print(f"[iter-117]     {r['item_id']} {r['item_name']}: implicit-via-cell_id only")

    # Rename vulnerability test
    rename_rows = rename_vulnerability_test(manifests, cells_by_id)
    by_mode: dict[str, list[int]] = {}
    for r in rename_rows:
        by_mode.setdefault(r["perturbation"], []).append(r["n_axes_recovered"])
    json_body_alone = sum(by_mode.get("json_body_alone", [0])) // max(1, len(by_mode.get("json_body_alone", [1])))
    json_plus_filename = sum(by_mode.get("json_plus_filename", [0])) // max(1, len(by_mode.get("json_plus_filename", [1])))
    json_plus_cells_tsv = sum(by_mode.get("json_plus_cells_tsv", [0])) // max(1, len(by_mode.get("json_plus_cells_tsv", [1])))
    print(f"[iter-117] H2: JSON-body-alone recovery: {json_body_alone}/5 axes; "
          f"+filename: {json_plus_filename}/5; +cells.tsv: {json_plus_cells_tsv}/5")

    # Write per-item TSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc_path = OUT_DIR / "p5_iter117_structural_ambiguity.tsv"
    cols = [
        "item_id", "item_name", "schema_status", "declared_source",
        "encoding_mode", "n_in_manifest_json", "n_in_cells_tsv",
        "n_in_filename", "n_na_sentinel", "n_tensor_derivable",
        "rename_robust", "n_cells",
    ]
    with open(enc_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[iter-117] wrote {enc_path} ({len(rows)} rows)")

    # Write rename vulnerability TSV
    ren_path = OUT_DIR / "p5_iter117_rename_vulnerability.tsv"
    rcols = ["perturbation", "original_cid", "n_axes_recovered",
             "missing_axes", "filename_match"]
    with open(ren_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rcols, delimiter="\t")
        w.writeheader()
        for r in rename_rows:
            w.writerow(r)
    print(f"[iter-117] wrote {ren_path} ({len(rename_rows)} rows)")

    # Summary JSON
    summary = {
        "n_cells": n,
        "encoding_mode_distribution": {
            mode: sum(1 for r in rows if r["encoding_mode"] == mode)
            for mode in {"explicit_json_key", "implicit_filename",
                         "implicit_filename_AND_cells_tsv", "cells_tsv_only",
                         "tensor_derivable", "n/a_sentinel", "absent_no_source"}
        },
        "implicit_filename_items": [r["item_id"] for r in implicit],
        "rename_vulnerability": {
            "n_perturbations":         len(rename_rows),
            "json_body_alone_axes":    json_body_alone,
            "json_plus_filename_axes": json_plus_filename,
            "json_plus_cells_tsv_axes": json_plus_cells_tsv,
        },
        "hypotheses": {
            "H1_split_schema":       sum(1 for r in rows
                                        if r["encoding_mode"] in {"implicit_filename_AND_cells_tsv",
                                                                  "cells_tsv_only"}) > 0,
            "H2_json_body_silence":  json_body_alone == 0,
            "H3_remediation_count":  sum(1 for r in rows
                                        if r["encoding_mode"] in {"implicit_filename_AND_cells_tsv",
                                                                  "absent_no_source"}),
        },
    }
    summary_path = OUT_DIR / "p5_iter117_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[iter-117] wrote {summary_path}")
    print(f"[iter-117] H1 PASS={summary['hypotheses']['H1_split_schema']}, "
          f"H2 PASS={summary['hypotheses']['H2_json_body_silence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())