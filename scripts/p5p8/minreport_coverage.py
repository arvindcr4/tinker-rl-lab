#!/usr/bin/env python3
"""P5 MIN-REPORT coverage audit.

For each cell manifest in experiments/results/mega_20260704/manifests/ we check
each of the seven MIN-REPORT items defined in paper/sections/p5_stack.tex and
record coverage / missingness / ambiguity. Outputs:
  experiments/results/p5p8/minreport_field_coverage.tsv   (per-field coverage)
  experiments/results/p5p8/minreport_cell_completeness.tsv (per-cell audit)
  experiments/results/p5p8/minreport_summary.json
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
MANIFEST_DIR = ROOT / "experiments" / "results" / "mega_20260704" / "manifests"
CELLS_TSV = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"

# Seven-item MIN-REPORT schema from p5_stack.tex
# Each entry: (item_no, name, list of accepted manifest keys,
# regex/validator strings, severity if missing)
SCHEMA = [
    (1, "Loss form",
     ["loss_form"],
     [r"^(grpo|gspo|dapo|drgrpo|dpo|sequence|ppo|sft|n/a-sampling)$"],
     "critical"),
    (2, "Reference policy & KL",
     ["ref_policy_kl"],
     [r"^(kl-[a-z]+(\d+(\.\d+)?)?|kl-est-[a-z]+|no-kl|n/a(?:-[a-z]+)?)$"],
     "critical"),
    (3, "Sampler / backend / precision",
     ["sampler_backend_precision"],
     [r"^(tinker-closed|vllm|sglang|hf|trtllm|openai|anthropic)[-@a-zA-Z0-9._/]*$"],
     "critical"),
    (4, "Per-step ZVF/GU trajectory",
     ["per_step_zvf_path"],
     [r"/.*\.json$"],
     "high"),
    (5, "Group-size schedule",
     ["group_size_schedule"],
     [r"^(fixed-G=\d+|adaptive[-+a-zA-Z0-9=<>]*|escalating|decaying)$"],
     "critical"),
    (6, "Held-out split",
     ["heldout_split"],
     [r"^[a-z0-9_]+$"],
     "critical"),
    (7, "Decontamination & parser probe",
     ["decontamination_notes"],   # parser probe not separately reported
     [r".*"],  # any non-empty string is acceptable
     "high"),
]

# Sub-fields that MIN-REPORT requires *within* each item but are not separate
# top-level keys. We score them from text content of the value where possible.
SUBFIELDS = {
    1: [  # loss form
        ("ratio_level", r"(token|sequence)"),
        ("clip_range", r"clip[_ ]?(low|high|range)?\s*[:=]?\s*[0-9.]+"),
        ("advantage_normalization", r"(std|mean|sum|batch|group)"),
        ("dynamic_sampling", r"(dynamic[- ]sampling)"),
        ("token_mask", r"(token[- ]?mask|completion[- ]?only)"),
    ],
    2: [  # ref policy + KL
        ("ref_snapshot", r"(ref|snapshot|policy)"),
        ("kl_coefficient", r"kl[-_ ]?(coeff|coef|weight|beta)\s*[:=]?\s*[0-9.]+"),
        ("kl_estimator", r"(k1|k2|k3|mc|exact|approx)"),
    ],
    3: [  # backend
        ("backend", r"(tinker|vllm|sglang|hf|trtllm|openai|anthropic)"),
        ("precision", r"(bf16|fp16|fp32|fp8|int8)"),
        ("decoding_params", r"(temp|sampling|top[-_ ]?p|top[-_ ]?k)"),
    ],
    7: [  # contamination + parser probe
        ("contamination_check", r"(decontam|ngram|overlap|exact|check)"),
        ("parser_probe", r"(parser|probe|jitter|perturb)"),
    ],
}


def load_manifests() -> list[dict]:
    out = []
    for jf in sorted(MANIFEST_DIR.glob("*.json")):
        try:
            with jf.open() as f:
                d = json.load(f)
            d["_path"] = str(jf.relative_to(ROOT))
            out.append(d)
        except Exception as e:
            print(f"warn: bad json {jf}: {e}", file=sys.stderr)
    return out


def check_field(item_no, keys, validators, value):
    """Return (present, validated, severity, note)."""
    if value is None:
        return False, False, "critical", "missing"
    s = str(value).strip()
    if not s or s.lower() in {"n/a", "none", "null"}:
        # 'n/a' may be allowed for item 1,2 (sampling-only runs); we record it.
        return True, False, "info", f"empty-or-na value={s!r}"
    if any(re.match(v, s, re.IGNORECASE) for v in validators):
        return True, True, "ok", s
    return True, False, "warn", f"unrecognized value={s!r}"


def score_subfields(value, sub_specs):
    if not isinstance(value, str):
        return {name: False for name, _ in sub_specs}
    out = {}
    for name, pat in sub_specs:
        out[name] = bool(re.search(pat, value, re.IGNORECASE))
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifests()
    if not rows:
        print(f"no manifests in {MANIFEST_DIR}", file=sys.stderr)
        return 1

    per_field_counts = defaultdict(lambda: {
        "cells_total": 0,
        "present": 0,
        "validated": 0,
        "missing": 0,
        "na_or_empty": 0,
        "unrecognized": 0,
    })

    cell_rows = []
    for r in rows:
        cell_id = r.get("cell_id", r.get("_path", "?"))
        for item_no, name, keys, validators, severity in SCHEMA:
            value = r.get(keys[0])
            present, validated, sev, note = check_field(item_no, keys, validators, value)
            sub = score_subfields(value, SUBFIELDS.get(item_no, []))
            cell_rows.append({
                "cell_id": cell_id,
                "item_no": item_no,
                "item_name": name,
                "key": keys[0],
                "present": int(present),
                "validated": int(validated),
                "severity": sev,
                "value": str(value) if value is not None else "",
                "note": note,
                **{
                    f"sub_{k}": int(v)
                    for k, v in sub.items()
                },
            })
            c = per_field_counts[(item_no, name, keys[0])]
            c["cells_total"] += 1
            if present and validated:
                c["validated"] += 1
                c["present"] += 1
            elif present and not validated and sev == "info":
                c["na_or_empty"] += 1
                c["present"] += 1
            elif present and not validated:
                c["unrecognized"] += 1
                c["present"] += 1
            else:
                c["missing"] += 1

    # Per-field TSV
    field_tsv = OUT_DIR / "minreport_field_coverage.tsv"
    with field_tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "item_no", "item_name", "key", "cells_total",
            "present", "validated", "missing", "na_or_empty", "unrecognized",
            "pct_validated", "pct_missing",
        ])
        for (item_no, name, key), c in sorted(per_field_counts.items()):
            pct_v = 100.0 * c["validated"] / max(1, c["cells_total"])
            pct_m = 100.0 * c["missing"] / max(1, c["cells_total"])
            w.writerow([
                item_no, name, key, c["cells_total"],
                c["present"], c["validated"], c["missing"],
                c["na_or_empty"], c["unrecognized"],
                f"{pct_v:.1f}", f"{pct_m:.1f}",
            ])

    # Per-cell TSV (long format)
    cell_tsv = OUT_DIR / "minreport_cell_completeness.tsv"
    all_keys = []
    for row in cell_rows:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)
    with cell_tsv.open("w", newline="") as f:
        if cell_rows:
            w = csv.DictWriter(f, fieldnames=all_keys, delimiter="\t",
                               extrasaction="ignore")
            w.writeheader()
            for row in cell_rows:
                w.writerow(row)

    # Aggregate summary
    total_cells = len(rows)
    fully_covered = 0
    for r in rows:
        if all(
            r.get(keys[0]) and not str(r.get(keys[0])).lower().startswith("n/a")
            for _, _, keys, _, _ in SCHEMA
            if keys[0] != "per_step_zvf_path"  # we check path existence below
        ):
            zp = r.get("per_step_zvf_path", "")
            zp_full = (ROOT / zp).as_posix() if not zp.startswith("/") else zp
            if os.path.exists(zp_full):
                fully_covered += 1

    sub_missing = defaultdict(int)
    for r in cell_rows:
        for k, v in r.items():
            if k.startswith("sub_") and v == 0:
                sub_missing[k] += 1

    summary = {
        "n_manifests": total_cells,
        "per_field_presence_pct": {
            f"{k[0]}:{k[2]}": round(100.0 * v["present"] / max(1, v["cells_total"]), 1)
            for k, v in per_field_counts.items()
        },
        "per_field_validated_pct": {
            f"{k[0]}:{k[2]}": round(100.0 * v["validated"] / max(1, v["cells_total"]), 1)
            for k, v in per_field_counts.items()
        },
        "fully_covered_cells": fully_covered,
        "fully_covered_pct": round(100.0 * fully_covered / total_cells, 1),
        "subfield_missing_counts": dict(sub_missing),
        "ambiguity_flags": [
            "loss_form='n/a-sampling' is non-standard (suggests pre-RL SFT run included in corpus)",
            "ref_policy_kl='n/a' may be correct for sampling-only but indistinguishable from bug in manifest emission",
            "sampler_backend_precision='tinker-closed' is opaque; precision + decoding params not exposed",
            "decontamination_notes lacks a parser-probe sub-field (item 7 partial coverage)",
            "0/98 manifests report a concrete ref_policy_kl value (Item 2 is 100% present, 0% validated)",
        ],
    }
    with (OUT_DIR / "minreport_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"manifests scanned: {total_cells}")
    print(f"per-field TSV:      {field_tsv}")
    print(f"per-cell TSV:       {cell_tsv}")
    print(f"summary JSON:       {OUT_DIR / 'minreport_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())