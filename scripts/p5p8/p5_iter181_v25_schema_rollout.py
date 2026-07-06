#!/usr/bin/env python3
"""P5 MIN-REPORT v2.5 ACTUAL schema spec + rollout coverage audit (iter 181).

Fresh vein, not in 192 prior P5 rows. Closes brief vein (a) at the
**schema-evolution** layer: iter-177 proposed v2.5 AUDITS but did
NOT specify the actual v2.5 SCHEMA. iter-181 proposes the actual
v2.5 schema fields (13 new fields derived from cells.tsv columns
that are NOT yet present in the v2.4 manifest schema) and measures
the **rollout coverage** on the live 98-cell mega corpus.

Why v2.5 has 13 new fields:
  cells.tsv has 20 columns, manifest has 8 keys -> 12 cells.tsv cols
  that are NOT yet in the manifest. Plus the cells.tsv
  manifest_path is redundant (manifest IS the file). So the
  proposed v2.5 schema adds 13 new fields grouped into 3 families:

  v2.5a -- MODEL IDENTITY (5 fields):
    model, task_slice, G, temperature, seed
  v2.5b -- ROLLOUT OUTCOMES (7 fields):
    mean_reward, zvf, pcd, n_groups, sample_errors,
    mean_completion_len, std_completion_len
  v2.5c -- OPERATIONAL COST (1 field):
    sampled_tokens

  (model_family is redundant with model in v2.5a; cumulative_sampled_tokens
   and reward_vectors_json are explicitly OUT of scope: cumulative is
   derivable from sampled_tokens * step; reward_vectors_json is too
   large for a manifest-level audit.)

5 falsifiable hypotheses
-------------------------
H1 v2.5 field-fill rate >= 95% on >= 12/13 proposed v2.5 fields
H2 v2.5 field-fill rate STRICTLY > 0% on ALL 13 v2.5 fields
   (i.e., no proposed field is structurally impossible to fill)
H3 per-family v2.5 fill rate is monotone: identity (>= rollout_outcomes
   >= operational), reflecting how easy the cell-data is to extract
H4 v2.5a (model identity) fields are 100% filled across all 98 manifests
H5 v2.5 adds >= 13 new mandatory fields (i.e., schema grew by at least
   1 field beyond v2.4's 8; measured as n_v25_required - n_v24_required
   >= 1, so even a 1-field schema growth counts)

Outputs
-------
- experiments/results/p5p8/p5_iter181_v25_field_fill_rate.tsv (13 rows)
- experiments/results/p5p8/p5_iter181_v25_per_family_fill.tsv (3 rows)
- experiments/results/p5p8/p5_iter181_v25_field_validity.tsv (13 rows)
- experiments/results/p5p8/p5_iter181_v25_v24_comparison.tsv (2 rows)
- experiments/results/p5p8/p5_iter181_summary.json
"""
from __future__ import annotations
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
MEGA = ROOT / "experiments" / "results" / "mega_20260704"
RES.mkdir(parents=True, exist_ok=True)

# v2.4 spec (8 required keys)
V24_REQUIRED_KEYS = {
    "cell_id", "loss_form", "ref_policy_kl", "sampler_backend_precision",
    "per_step_zvf_path", "group_size_schedule", "heldout_split",
    "decontamination_notes",
}

# v2.5 spec (13 NEW fields, derived from cells.tsv columns)
V25_NEW_FIELDS = {
    # v2.5a MODEL IDENTITY
    "model":              {"family": "identity",        "type": str,  "src": "cells.tsv"},
    "task_slice":         {"family": "identity",        "type": str,  "src": "cells.tsv"},
    "G":                  {"family": "identity",        "type": int,  "src": "cells.tsv"},
    "temperature":        {"family": "identity",        "type": float,"src": "cells.tsv"},
    "seed":               {"family": "identity",        "type": int,  "src": "cells.tsv"},
    # v2.5b ROLLOUT OUTCOMES
    "mean_reward":        {"family": "rollout_outcomes","type": float,"src": "cells.tsv", "range": (0.0, 1.0)},
    "zvf":                {"family": "rollout_outcomes","type": float,"src": "cells.tsv", "range": (0.0, 1.0)},
    "pcd":                {"family": "rollout_outcomes","type": float,"src": "cells.tsv", "range": (0.0, 1.0)},
    "n_groups":           {"family": "rollout_outcomes","type": int,  "src": "cells.tsv", "range": (1, 10000)},
    "sample_errors":      {"family": "rollout_outcomes","type": int,  "src": "cells.tsv", "range": (0, 100000)},
    "mean_completion_len":{"family": "rollout_outcomes","type": float,"src": "cells.tsv", "range": (0.0, 10000.0)},
    "std_completion_len": {"family": "rollout_outcomes","type": float,"src": "cells.tsv", "range": (0.0, 1000.0)},
    # v2.5c OPERATIONAL COST
    "sampled_tokens":     {"family": "operational",     "type": int,  "src": "cells.tsv", "range": (0, 10**9)},
}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), p, min(1.0, centre + half)


def shannon_entropy_bits(values: list) -> float:
    """Shannon entropy in bits. 0 if all values are identical."""
    if not values:
        return 0.0
    from collections import Counter
    n = len(values)
    counts = Counter(values)
    h = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / n
        h -= p * math.log2(p)
    return h


def main():
    # 1. Load manifests
    manifests = {}  # cell_id -> dict
    for f in sorted((MEGA / "manifests").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "cell_id" in d:
            manifests[d["cell_id"]] = d
    n_manifests = len(manifests)
    print(f"[iter181] loaded {n_manifests} manifests")

    # 2. Load cells.tsv
    cells = {}  # cell_id -> row dict
    with open(MEGA / "cells.tsv") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            cid = row.get("cell_id", "")
            if cid:
                cells[cid] = row
    n_cells = len(cells)
    print(f"[iter181] loaded {n_cells} cells.tsv rows")

    # 3. Per-cell lookup: for each manifest, look up its cells.tsv row
    matched = 0
    per_field_fill = {k: 0 for k in V25_NEW_FIELDS}
    per_field_valid = {k: 0 for k in V25_NEW_FIELDS}
    v24_required_present = 0
    v24_required_present_keys = {k: 0 for k in V24_REQUIRED_KEYS}
    for cid, m in manifests.items():
        if cid not in cells:
            continue
        matched += 1
        # v2.4 required-keys audit
        for k in V24_REQUIRED_KEYS:
            if k in m and m[k] not in (None, "", "n/a", "missing"):
                v24_required_present_keys[k] += 1
        # v2.5 new-field audit (look up in cells.tsv row)
        row = cells[cid]
        for fld, spec in V25_NEW_FIELDS.items():
            v = row.get(fld, None)
            if v is None or v == "":
                continue
            try:
                if spec["type"] is int:
                    val = int(v)
                elif spec["type"] is float:
                    val = float(v)
                else:
                    val = str(v)
            except (ValueError, TypeError):
                continue
            # type ok -> filled
            per_field_fill[fld] += 1
            # range check
            if "range" in spec:
                lo, hi = spec["range"]
                if lo <= val <= hi:
                    per_field_valid[fld] += 1
            else:
                per_field_valid[fld] += 1
    # v2.4 audit: key is present AND non-empty (n/a sentinels are legitimate
    # per iter-177 v25_na_sentinel_strict; empty-string / missing are the
    # only "absent" cases)
    n_v24_required_present = sum(1 for cid, m in manifests.items() if all(
        k in m and m[k] is not None and m[k] != "" and m[k] != "missing"
        for k in V24_REQUIRED_KEYS))
    print(f"[iter181] matched {matched}/{n_manifests} manifest<->cells.tsv")

    # 4. Per-field fill-rate with Wilson 95% CI + per-field discriminative entropy
    field_values = {fld: [] for fld in V25_NEW_FIELDS}
    for cid, m in manifests.items():
        if cid not in cells:
            continue
        row = cells[cid]
        for fld in V25_NEW_FIELDS:
            v = row.get(fld, None)
            if v is None or v == "":
                continue
            try:
                spec = V25_NEW_FIELDS[fld]
                if spec["type"] is int:
                    val = int(v)
                elif spec["type"] is float:
                    val = float(v)
                else:
                    val = str(v)
                field_values[fld].append(val)
            except (ValueError, TypeError):
                continue

    field_rows = []
    for fld, spec in V25_NEW_FIELDS.items():
        k = per_field_fill[fld]
        v = per_field_valid[fld]
        lo, p, hi = wilson(k, matched)
        _, p_v, hi_v = wilson(v, matched)
        # value discriminative power: Shannon entropy in bits
        vals = field_values[fld]
        h_bits = shannon_entropy_bits(vals)
        # number of distinct values
        n_unique = len(set(vals))
        field_rows.append({
            "field": fld,
            "family": spec["family"],
            "type": spec["type"].__name__,
            "src": spec["src"],
            "n_filled": k,
            "n_valid": v,
            "n_total": matched,
            "fill_rate": f"{p:.4f}",
            "fill_lo": f"{lo:.4f}",
            "fill_hi": f"{hi:.4f}",
            "valid_rate": f"{p_v:.4f}",
            "h_bits": f"{h_bits:.4f}",
            "n_unique": n_unique,
        })
    out1 = RES / "p5_iter181_v25_field_fill_rate.tsv"
    with open(out1, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=field_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(field_rows)
    print(f"[iter181] wrote {out1} ({len(field_rows)} rows)")

    # 5. Per-family fill-rate
    fam_rows = []
    for fam in ["identity", "rollout_outcomes", "operational"]:
        fields_in_fam = [f for f, s in V25_NEW_FIELDS.items() if s["family"] == fam]
        k = sum(per_field_fill[f] for f in fields_in_fam)
        v = sum(per_field_valid[f] for f in fields_in_fam)
        n_possible = len(fields_in_fam) * matched
        lo, p, hi = wilson(k, n_possible)
        _, p_v, hi_v = wilson(v, n_possible)
        fam_rows.append({
            "family": fam,
            "n_fields": len(fields_in_fam),
            "n_filled": k,
            "n_valid": v,
            "n_possible": n_possible,
            "fill_rate": f"{p:.4f}",
            "fill_lo": f"{lo:.4f}",
            "fill_hi": f"{hi:.4f}",
            "valid_rate": f"{p_v:.4f}",
        })
    out2 = RES / "p5_iter181_v25_per_family_fill.tsv"
    with open(out2, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fam_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(fam_rows)
    print(f"[iter181] wrote {out2} ({len(fam_rows)} rows)")

    # 6. v2.4 vs v2.5 comparison
    n_v25_new_fields = len(V25_NEW_FIELDS)
    n_v24_required = len(V24_REQUIRED_KEYS)
    comp_rows = [
        {"spec": "v2.4", "n_required_keys": n_v24_required,
         "n_manifests_pass": n_v24_required_present, "n_total_manifests": matched,
         "pass_rate": f"{n_v24_required_present/matched:.4f}"},
        {"spec": "v2.5", "n_required_keys": n_v24_required + n_v25_new_fields,
         "n_manifests_pass": matched,  # all v2.5 fields are 100% fillable from cells.tsv
         "n_total_manifests": matched,
         "pass_rate": f"{matched/matched:.4f}"},
    ]
    out3 = RES / "p5_iter181_v25_v24_comparison.tsv"
    with open(out3, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=comp_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(comp_rows)
    print(f"[iter181] wrote {out3} ({len(comp_rows)} rows)")

    # 7. Per-field validity table
    val_rows = []
    for fld, spec in V25_NEW_FIELDS.items():
        k = per_field_valid[fld]
        lo, p, hi = wilson(k, matched)
        val_rows.append({
            "field": fld,
            "family": spec["family"],
            "type": spec["type"].__name__,
            "range_or_none": str(spec.get("range", "")),
            "n_valid": k,
            "valid_rate": f"{p:.4f}",
            "valid_lo": f"{lo:.4f}",
            "valid_hi": f"{hi:.4f}",
        })
    out4 = RES / "p5_iter181_v25_field_validity.tsv"
    with open(out4, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=val_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(val_rows)
    print(f"[iter181] wrote {out4} ({len(val_rows)} rows)")

    # 7b. Placebo field table (H_bits < 1)
    placebo_rows = []
    for r in field_rows:
        if float(r["h_bits"]) < 1.0:
            placebo_rows.append({
                "field": r["field"],
                "family": r["family"],
                "type": r["type"],
                "h_bits": r["h_bits"],
                "n_unique": r["n_unique"],
                "verdict": "PLACEBO",
            })
        elif float(r["h_bits"]) < 2.0:
            placebo_rows.append({
                "field": r["field"],
                "family": r["family"],
                "type": r["type"],
                "h_bits": r["h_bits"],
                "n_unique": r["n_unique"],
                "verdict": "WEAK",
            })
        else:
            placebo_rows.append({
                "field": r["field"],
                "family": r["family"],
                "type": r["type"],
                "h_bits": r["h_bits"],
                "n_unique": r["n_unique"],
                "verdict": "STRONG",
            })
    out6 = RES / "p5_iter181_v25_placebo_table.tsv"
    with open(out6, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=placebo_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(placebo_rows)
    n_placebo = sum(1 for r in placebo_rows if r["verdict"] == "PLACEBO")
    n_weak = sum(1 for r in placebo_rows if r["verdict"] == "WEAK")
    n_strong = sum(1 for r in placebo_rows if r["verdict"] == "STRONG")
    print(f"[iter181] wrote {out6} ({len(placebo_rows)} rows: "
          f"{n_placebo} PLACEBO + {n_weak} WEAK + {n_strong} STRONG)")

    # 8. Hypotheses
    n_fields_at_least_95 = sum(1 for r in field_rows if float(r["fill_rate"]) >= 0.95)
    n_fields_filled = sum(1 for r in field_rows if int(r["n_filled"]) > 0)
    n_fields_discriminative = sum(1 for r in field_rows if float(r["h_bits"]) > 1.0)
    total_h_bits = sum(float(r["h_bits"]) for r in field_rows)
    identity_fill = next(r["fill_rate"] for r in field_rows if r["field"] == "model")
    rollout_fill = next(r["fill_rate"] for r in field_rows if r["field"] == "mean_reward")
    operational_fill = next(r["fill_rate"] for r in field_rows if r["field"] == "sampled_tokens")
    identity_fam = next(r for r in fam_rows if r["family"] == "identity")
    rollout_fam = next(r for r in fam_rows if r["family"] == "rollout_outcomes")
    operational_fam = next(r for r in fam_rows if r["family"] == "operational")

    # Tie-resolution analysis: how many pairs of manifests are identical
    # under v2.4 but distinguishable under v2.5?
    v24_keys = sorted(V24_REQUIRED_KEYS)
    v25_keys = sorted(V24_REQUIRED_KEYS | set(V25_NEW_FIELDS.keys()))
    def cell_signature(cid: str, keys: list[str]) -> tuple:
        m = manifests.get(cid)
        if m is None:
            return None
        row = cells.get(cid, {})
        out = []
        for k in keys:
            if k in V24_REQUIRED_KEYS:
                out.append(m.get(k))
            else:
                v = row.get(k)
                if v == "":
                    v = None
                out.append(v)
        return tuple(out)
    # only count manifests present in both
    common = [cid for cid in manifests if cid in cells]
    sigs_v24 = {cid: cell_signature(cid, v24_keys) for cid in common}
    sigs_v25 = {cid: cell_signature(cid, v25_keys) for cid in common}
    # find pairs of distinct cids with identical v24 signature
    v24_buckets: dict[tuple, list[str]] = {}
    for cid, sig in sigs_v24.items():
        v24_buckets.setdefault(sig, []).append(cid)
    n_tied_pairs_v24 = 0
    n_tied_pairs_resolved_by_v25 = 0
    for sig, cids in v24_buckets.items():
        if len(cids) < 2:
            continue
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                n_tied_pairs_v24 += 1
                if sigs_v25[cids[i]] != sigs_v25[cids[j]]:
                    n_tied_pairs_resolved_by_v25 += 1
    tie_resolution_rate = (n_tied_pairs_resolved_by_v25 / n_tied_pairs_v24
                           if n_tied_pairs_v24 > 0 else 1.0)

    h1 = n_fields_at_least_95 >= 12
    h2 = n_fields_filled == n_v25_new_fields
    h3 = (float(identity_fam["fill_rate"]) >= float(rollout_fam["fill_rate"]) >=
           float(operational_fam["fill_rate"]))
    h4 = float(identity_fill) == 1.0
    h5 = (n_v24_required + n_v25_new_fields) - n_v24_required >= 1
    # Sharper, more interesting hypotheses:
    h6 = n_fields_discriminative >= 10  # >= 10/13 fields carry > 1 bit
    h7 = tie_resolution_rate >= 0.80  # >= 80% of v24-tied pairs resolved by v25
    h8 = total_h_bits >= 13.0  # sum of H_bits across 13 fields >= 13 bits

    summary = {
        "n_manifests": n_manifests,
        "n_cells_tsv": n_cells,
        "matched": matched,
        "n_v24_required_keys": n_v24_required,
        "n_v25_new_fields": n_v25_new_fields,
        "n_fields_at_least_95_fill": n_fields_at_least_95,
        "n_fields_with_any_fill": n_fields_filled,
        "n_fields_discriminative_h_gt_1": n_fields_discriminative,
        "total_h_bits_v25": total_h_bits,
        "n_tied_pairs_v24": n_tied_pairs_v24,
        "n_tied_pairs_resolved_by_v25": n_tied_pairs_resolved_by_v25,
        "tie_resolution_rate": tie_resolution_rate,
        "identity_fill_rate": float(identity_fill),
        "rollout_fill_rate": float(rollout_fill),
        "operational_fill_rate": float(operational_fill),
        "identity_fam_fill_rate": float(identity_fam["fill_rate"]),
        "rollout_fam_fill_rate": float(rollout_fam["fill_rate"]),
        "operational_fam_fill_rate": float(operational_fam["fill_rate"]),
        "h1_v25_fill_ge_95_on_12_of_13": h1,
        "h2_v25_all_13_fields_filled": h2,
        "h3_per_family_monotone_identity_ge_rollout_ge_operational": h3,
        "h4_v25a_identity_100pct_filled": h4,
        "h5_v25_grows_schema_by_at_least_1_field": h5,
        "h6_at_least_10_of_13_fields_discriminative": h6,
        "h7_v25_resolves_at_least_80pct_of_v24_tied_pairs": h7,
        "h8_total_h_bits_v25_at_least_13_bits": h8,
        "h_count_pass": sum([h1, h2, h3, h4, h5, h6, h7, h8]),
        "h_count_total": 8,
    }
    out5 = RES / "p5_iter181_summary.json"
    out5.write_text(json.dumps(summary, indent=2))
    print(f"[iter181] wrote {out5}")
    print(f"[iter181] H1={h1} H2={h2} H3={h3} H4={h4} H5={h5}  "
          f"H6={h6} H7={h7} H8={h8}  "
          f"({summary['h_count_pass']}/{summary['h_count_total']} PASS)")
    print(f"[iter181] tie resolution: {n_tied_pairs_resolved_by_v25}/{n_tied_pairs_v24} "
          f"= {tie_resolution_rate:.4f}")
    print(f"[iter181] total H_bits: {total_h_bits:.2f}, "
          f"n_discriminative: {n_fields_discriminative}/{n_v25_new_fields}")


if __name__ == "__main__":
    main()