#!/usr/bin/env python3
"""P5 MIN-REPORT v2.4 schema-vs-live-mega-manifest sub-field coverage audit (iter 205).

Fresh P5 vein, not in 215 prior P5 rows. Iter-121 (row 136) audited 8 top-level
manifest fields; iter-153 (row 170) audited the v2.4 emit-gap on
``bib/manifests/cells.tsv``; iter-114 (row 130) audited schema-mismatch on
12 sub-fields. None of the 215 prior P5 rows does a **per-sub-field**, **per-
manifest** coverage audit that classifies each MIN-REPORT-required sub-field
as ``EXTRACTABLE_LITERAL`` / ``EXTRACTABLE_PARSED`` / ``AMBIGUOUS`` /
``MISSING`` against the LIVE ``platform_hybrid/experiments/results/mega_20260704/manifests/``
corpus (98 cells, 4 model_family x 3 task_slice x 5 G x 2 temperature x 2 seed).

Iter-205 closes this gap by:
  (a) enumerating the 23 MIN-REPORT-required sub-fields grouped by the 7
      items (loss_form / reference_kl / sampler_backend / telemetry /
      group_size_schedule / heldout_split/ decontamination);
  (b) reading each of 98 manifests and classifying each sub-field into one
      of four recoverability states;
  (c) building a measured per-sub-field coverage table;
  (d) computing per-Item coverage (the 7-item min-report standard must hold
      atomically: 0/7 sub-fields = FAIL, 23/23 sub-fields = PASS);
  (e) computing per-(manifest) coverage distribution;
  (f) running a paired bootstrap on per-(model_family, task_slice)
      coverage rates to test whether coverage is uniform across the
      experimental design.

Outputs
-------
- platform_hybrid/experiments/results/p5p8/p5_iter205_subfield_class.tsv (23 rows: per
  sub-field, n_extract_literal, n_extract_parsed, n_ambiguous,
  n_missing, coverage_pct)
- platform_hybrid/experiments/results/p5p8/p5_iter205_item_coverage.tsv (7 rows: per
  Item, n_subfields_pass / total_subfields, manifest-level pass rate)
- platform_hybrid/experiments/results/p5p8/p5_iter205_per_manifest_coverage.tsv (98
  rows: per manifest, n_subfields_pass, n_subfields_partial,
  n_subfields_missing, derived badge level)
- platform_hybrid/experiments/results/p5p8/p5_iter205_stratified_coverage.tsv (4
  model_family x 3 task_slice cells: per-cell coverage rate with
  bootstrap CI)
- platform_hybrid/experiments/results/p5p8/p5_iter205_summary.json (H1-H6 verdicts +
  measured coverage rollups + missing-field list + ambiguous-field
  list)
"""
from __future__ import annotations
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(20260706)
N_BOOT = 2000
SEED = 20260706

WORKTREE = Path("/home/claude/tinker-rl-lab-minimax")
MANIFEST_DIR = WORKTREE / "platform_hybrid/experiments/results/mega_20260704/manifests"
CELLS_TSV = WORKTREE / "platform_hybrid/experiments/results/mega_20260704/cells.tsv"
OUT_DIR = WORKTREE / "platform_hybrid/experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Canonical MIN-REPORT v2.4 sub-field map (23 required sub-fields, 7 items)
# Each entry: (item_id, sub_field_id, parser_key_or_None)
# - parser_key=None means a literal sub-key in the manifest is required.
# - parser_key="group_size_schedule" means we must parse the string from
#   that top-level key (e.g. "fixed-G=2" -> {initial_g: 2, schedule:
#   "fixed"}).
MIN_REPORT_FIELDS: list[tuple[str, str, str | None]] = [
    # Item 1: loss_form (6 required sub-fields)
    ("loss_form", "importance_ratio_level", "loss_form"),
    ("loss_form", "clip_eps_low", "loss_form"),
    ("loss_form", "clip_eps_high", "loss_form"),
    ("loss_form", "length_normalization", "loss_form"),
    ("loss_form", "advantage_normalization", "loss_form"),
    ("loss_form", "token_mask", "loss_form"),
    # Item 2: reference_kl (3 required sub-fields)
    ("reference_kl", "reference_policy", "ref_policy_kl"),
    ("reference_kl", "kl_beta", "ref_policy_kl"),
    ("reference_kl", "kl_estimator", "ref_policy_kl"),
    # Item 3: sampler_backend (4 required sub-fields)
    ("sampler_backend", "backend", "sampler_backend_precision"),
    ("sampler_backend", "precision", "sampler_backend_precision"),
    ("sampler_backend", "temperature", "cells_tsv"),  # parsed from cells.tsv
    ("sampler_backend", "top_p", "sampler_backend_precision"),
    # Item 4: telemetry (3 required sub-fields)
    ("telemetry", "per_step_zvf", "per_step_zvf_path"),
    ("telemetry", "per_step_gu", "per_step_zvf_path"),
    ("telemetry", "source", "per_step_zvf_path"),
    # Item 5: group_size_schedule (3 required sub-fields)
    ("group_size_schedule", "initial_g", "group_size_schedule"),
    ("group_size_schedule", "schedule", "group_size_schedule"),
    ("group_size_schedule", "adaptation_rule", "group_size_schedule"),
    # Item 6: heldout_split (2 required sub-fields)
    ("heldout_split", "disjoint_from_reward_env", "heldout_split"),
    ("heldout_split", "description", "heldout_split"),
    # Item 7: decontamination (2 required sub-fields)
    ("decontamination", "performed", "decontamination_notes"),
    ("decontamination", "parser_robustness_probe", "decontamination_notes"),
]


def parse_group_size_schedule(s: str) -> dict:
    """Parse 'fixed-G=2' -> {schedule: 'fixed', initial_g: 2}.

    Returns {} if no match.
    """
    out: dict = {}
    if not s:
        return out
    if s.startswith("fixed-G="):
        try:
            out["initial_g"] = int(s.split("=", 1)[1])
            out["schedule"] = "fixed"
        except ValueError:
            pass
    elif s.startswith("adaptive"):
        out["schedule"] = "adaptive"
        out["adaptation_rule"] = None  # not encoded in shorthand
    return out


def parse_sampler_backend_precision(s: str) -> dict:
    """Parse 'tinker-closed' -> {backend: 'tinker', precision: 'closed'}.

    Returns {} if no match.
    """
    out: dict = {}
    if not s or "-" not in s:
        return out
    parts = s.split("-", 1)
    out["backend"] = parts[0]
    out["precision"] = parts[1] if len(parts) > 1 else None
    return out


def parse_heldout_split(s: str) -> dict:
    """Parse 'gsm8k_easy' -> {description: 'gsm8k_easy',
    disjoint_from_reward_env: True}.

    Heuristic: heldout split name is disjoint from reward env by definition
    (otherwise it wouldn't be a heldout).
    """
    return {"description": s, "disjoint_from_reward_env": True} if s else {}


def parse_decontamination_notes(s: str) -> dict:
    """Parse 'gsm8k-train-slice' -> {performed: True,
    parser_robustness_probe: None}.

    Heuristic: non-empty note indicates decontamination was performed.
    """
    return {"performed": bool(s), "parser_robustness_probe": None} if s else {}


def parse_loss_form(s: str) -> dict:
    """Parse loss_form. The mega manifests always emit 'n/a-sampling' which
    indicates the n/a-sampling shorthand (no reward shaping, no KL, sampled
    from base policy)."""
    out: dict = {}
    if not s:
        return out
    if s == "n/a-sampling":
        # n/a-sampling: no importance ratio (pure sampling / REINFORCE-like)
        out["importance_ratio_level"] = None
        out["clip_eps_low"] = None
        out["clip_eps_high"] = None
        out["length_normalization"] = None
        out["advantage_normalization"] = "none"
        out["token_mask"] = None
    return out


def parse_ref_policy_kl(s: str) -> dict:
    """Parse ref_policy_kl shorthand. Mega manifests emit 'n/a'."""
    out: dict = {}
    if not s:
        return out
    if s == "n/a":
        out["reference_policy"] = None
        out["kl_beta"] = None
        out["kl_estimator"] = None
    return out


def parse_per_step_zvf_path(s: str) -> dict:
    """Parse per_step_zvf_path. Non-empty path means per_step_zvf is True."""
    out: dict = {}
    if not s:
        return out
    out["per_step_zvf"] = True
    out["per_step_gu"] = None  # GU not separately logged in mega
    out["source"] = s
    return out


# Sub-field extractors by top-level parser_key
PARSERS = {
    "loss_form": parse_loss_form,
    "ref_policy_kl": parse_ref_policy_kl,
    "sampler_backend_precision": parse_sampler_backend_precision,
    "group_size_schedule": parse_group_size_schedule,
    "heldout_split": parse_heldout_split,
    "decontamination_notes": parse_decontamination_notes,
    "per_step_zvf_path": parse_per_step_zvf_path,
}


def load_manifests() -> list[dict]:
    """Load all 98 mega manifests."""
    rows: list[dict] = []
    for fp in sorted(MANIFEST_DIR.glob("*.json")):
        try:
            with fp.open() as fh:
                d = json.load(fh)
                d["_manifest_path"] = str(fp)
                rows.append(d)
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def load_cells_tsv() -> dict:
    """Load cells.tsv keyed by cell_id -> {temperature, G, ...}."""
    cells: dict = {}
    with CELLS_TSV.open() as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            try:
                cells[row["cell_id"]] = {
                    "temperature": float(row["temperature"]),
                    "G": int(row["G"]),
                    "model_family": row["model_family"],
                    "task_slice": row["task_slice"],
                    "seed": int(row["seed"]),
                }
            except (KeyError, ValueError):
                continue
    return cells


def classify_subfield(
    manifest: dict, item: str, sub_field: str, parser_key: str | None,
    cells_index: dict,
) -> str:
    """Classify a sub-field into one of:
      EXTRACTABLE_LITERAL  - sub-key present in manifest body with the
        right type.
      EXTRACTABLE_PARSED   - not a literal sub-key but unambiguously
        derivable from a top-level string via the canonical parser.
      AMBIGUOUS            - multiple plausible parses; requires domain
        choice; would need operator confirmation.
      MISSING              - cannot be recovered from the manifest body.

    Returns the classification string.
    """
    cell_id = manifest.get("cell_id", "")
    cell_row = cells_index.get(cell_id, {})

    # Special-case: temperature is in cells_tsv, not the manifest.
    if parser_key == "cells_tsv":
        if sub_field == "temperature":
            if "temperature" in cell_row:
                return "EXTRACTABLE_PARSED"
        return "MISSING"

    if parser_key is None:
        # No parser configured: assume MISSING (audit-mode strict).
        return "MISSING"

    raw = manifest.get(parser_key)
    if raw is None:
        return "MISSING"

    parser = PARSERS.get(parser_key)
    if parser is None:
        return "MISSING"
    parsed = parser(raw)

    if sub_field in parsed and parsed[sub_field] is not None:
        return "EXTRACTABLE_PARSED"
    # Heuristic: bool fields with True default from non-empty raw also count
    if sub_field == "performed" and raw:
        return "EXTRACTABLE_PARSED"
    if sub_field == "disjoint_from_reward_env" and raw:
        return "EXTRACTABLE_PARSED"
    if sub_field == "per_step_zvf" and raw:
        return "EXTRACTABLE_PARSED"
    return "AMBIGUOUS"


def main() -> None:
    manifests = load_manifests()
    cells_index = load_cells_tsv()
    n_manifests = len(manifests)
    print(f"[iter205] Loaded {n_manifests} manifests, {len(cells_index)} cells")

    # --- (a) per-sub-field classification ---
    subfield_class: dict = defaultdict(lambda: defaultdict(int))
    for manifest in manifests:
        for item, sub, parser_key in MIN_REPORT_FIELDS:
            cls = classify_subfield(manifest, item, sub, parser_key, cells_index)
            subfield_class[(item, sub)][cls] += 1

    subfield_rows: list[dict] = []
    for (item, sub), counts in subfield_class.items():
        lit = counts.get("EXTRACTABLE_LITERAL", 0)
        par = counts.get("EXTRACTABLE_PARSED", 0)
        amb = counts.get("AMBIGUOUS", 0)
        miss = counts.get("MISSING", 0)
        covered = lit + par  # recoverable sub-field
        cov_pct = 100 * covered / n_manifests
        subfield_rows.append({
            "item": item,
            "sub_field": sub,
            "n_extract_literal": lit,
            "n_extract_parsed": par,
            "n_ambiguous": amb,
            "n_missing": miss,
            "n_covered": covered,
            "coverage_pct": round(cov_pct, 2),
        })

    subfield_rows.sort(key=lambda r: (r["item"], r["sub_field"]))

    # --- (b) per-Item coverage ---
    by_item: dict = defaultdict(lambda: {"total": 0, "covered": 0})
    for r in subfield_rows:
        by_item[r["item"]]["total"] += 1
        if r["n_covered"] == n_manifests:
            by_item[r["item"]]["covered"] += 1

    # Per-manifest Item pass (all sub-fields of the Item covered)
    item_pass_by_manifest: dict = defaultdict(int)
    for manifest in manifests:
        for item, sub, parser_key in MIN_REPORT_FIELDS:
            cls = classify_subfield(manifest, item, sub, parser_key, cells_index)
            if cls in ("EXTRACTABLE_LITERAL", "EXTRACTABLE_PARSED"):
                item_pass_by_manifest[(manifest.get("cell_id", ""), item)] += 1

    item_rows: list[dict] = []
    for item in sorted(by_item.keys()):
        item_total_subfields = sum(
            1 for it, _, _ in MIN_REPORT_FIELDS if it == item
        )
        manifests_passing_item = sum(
            1 for m in manifests
            if item_pass_by_manifest[(m.get("cell_id", ""), item)] == item_total_subfields
        )
        item_rows.append({
            "item": item,
            "n_subfields": item_total_subfields,
            "n_subfields_full_coverage": by_item[item]["covered"],
            "manifests_passing_item": manifests_passing_item,
            "manifests_passing_pct": round(100 * manifests_passing_item / n_manifests, 2),
        })

    # --- (c) per-manifest coverage ---
    manifest_rows: list[dict] = []
    for manifest in manifests:
        n_lit = n_par = n_amb = n_mis = 0
        for item, sub, parser_key in MIN_REPORT_FIELDS:
            cls = classify_subfield(manifest, item, sub, parser_key, cells_index)
            if cls == "EXTRACTABLE_LITERAL":
                n_lit += 1
            elif cls == "EXTRACTABLE_PARSED":
                n_par += 1
            elif cls == "AMBIGUOUS":
                n_amb += 1
            else:
                n_mis += 1
        covered = n_lit + n_par
        missing = n_mis
        ambiguous = n_amb
        # Badge tier: 0/23 = bronze-fail, 1-7 = bronze, 8-15 = silver,
        # 16-22 = gold, 23 = platinum
        if covered == 0:
            tier = "FAIL"
        elif covered <= 7:
            tier = "bronze-partial"
        elif covered <= 15:
            tier = "silver-partial"
        elif covered <= 22:
            tier = "gold-partial"
        else:
            tier = "platinum"
        manifest_rows.append({
            "cell_id": manifest.get("cell_id", ""),
            "model_family": cells_index.get(manifest.get("cell_id", ""), {}).get("model_family", ""),
            "task_slice": cells_index.get(manifest.get("cell_id", ""), {}).get("task_slice", ""),
            "n_literal": n_lit,
            "n_parsed": n_par,
            "n_ambiguous": n_amb,
            "n_missing": n_mis,
            "n_covered": covered,
            "missing": missing,
            "ambiguous": ambiguous,
            "tier": tier,
        })

    # --- (d) per-(model_family, task_slice) coverage with bootstrap CI ---
    by_cell: dict = defaultdict(list)
    for r in manifest_rows:
        key = (r["model_family"], r["task_slice"])
        by_cell[key].append(r["n_covered"])

    rng = random.Random(SEED)
    stratified_rows: list[dict] = []
    for (mf, ts), coverages in sorted(by_cell.items()):
        n = len(coverages)
        point_mean = sum(coverages) / n
        boots = []
        for _ in range(N_BOOT):
            sample = [rng.choice(coverages) for _ in range(n)]
            boots.append(sum(sample) / n)
        boots.sort()
        ci_lo = boots[int(0.025 * N_BOOT)]
        ci_hi = boots[int(0.975 * N_BOOT)]
        stratified_rows.append({
            "model_family": mf,
            "task_slice": ts,
            "n_manifests": n,
            "mean_subfields_covered": round(point_mean, 2),
            "ci_lo": round(ci_lo, 2),
            "ci_hi": round(ci_hi, 2),
            "frac_above_15": round(sum(1 for c in coverages if c >= 15) / n, 4),
        })

    # --- Hypotheses ---
    # H1: per-Item pass rate is non-zero on EVERY of the 7 items
    h1_pass = all(r["manifests_passing_item"] > 0 for r in item_rows)
    # H2: at least 3 items have a manifest pass rate >= 50%
    h2_pass = sum(1 for r in item_rows if r["manifests_passing_pct"] >= 50) >= 3
    # H3: per-manifest coverage MEAN >= 7 (out of 23) - bronze threshold
    mean_cov = sum(r["n_covered"] for r in manifest_rows) / len(manifest_rows)
    h3_pass = mean_cov >= 7
    # H4: per-(mf, task_slice) coverage is uniform (max - min < 5)
    if stratified_rows:
        cell_means = [r["mean_subfields_covered"] for r in stratified_rows]
        spread = max(cell_means) - min(cell_means)
    else:
        spread = 0
    h4_pass = spread < 5
    # H5: ZERO manifests hit platinum (23/23)
    n_platinum = sum(1 for r in manifest_rows if r["tier"] == "platinum")
    h5_pass = n_platinum == 0  # STRICT (schema-impossibility finding)
    # H6: at least 90% of manifests are in a non-FAIL tier
    n_fail = sum(1 for r in manifest_rows if r["tier"] == "FAIL")
    h6_pass = n_fail <= 0.10 * n_manifests
    # H7: at least 5 sub-fields require AMBIGUOUS recovery (n_ambiguous = n_manifests)
    n_ambiguous_perm = sum(1 for r in subfield_rows if r["n_ambiguous"] == n_manifests)
    h7_pass = n_ambiguous_perm >= 5
    # H8: schema-impossibility — ZERO manifests atomically pass all 7 items
    n_atomically_pass = sum(
        1 for m in manifests
        if all(
            item_pass_by_manifest[(m.get("cell_id", ""), item)] == sum(
                1 for it, _, _ in MIN_REPORT_FIELDS if it == item
            )
            for item in {it for it, _, _ in MIN_REPORT_FIELDS}
        )
    )
    h8_pass = n_atomically_pass == 0  # STRICT impossibility finding

    # --- Missing / ambiguous fields list ---
    missing_fields: list[dict] = []
    for r in subfield_rows:
        if r["n_missing"] == n_manifests:
            missing_fields.append({
                "item": r["item"],
                "sub_field": r["sub_field"],
                "n_missing": r["n_missing"],
            })
    ambiguous_fields: list[dict] = []
    for r in subfield_rows:
        if r["n_ambiguous"] > 0:
            ambiguous_fields.append({
                "item": r["item"],
                "sub_field": r["sub_field"],
                "n_ambiguous": r["n_ambiguous"],
            })

    summary = {
        "n_manifests": n_manifests,
        "n_subfields": len(MIN_REPORT_FIELDS),
        "n_items": len({it for it, _, _ in MIN_REPORT_FIELDS}),
        "hypotheses": {
            "H1_per_item_non_zero": h1_pass,
            "H2_at_least_3_items_over_50pct": h2_pass,
            "H3_mean_coverage_at_least_bronze": h3_pass,
            "H4_uniform_across_mf_x_task": h4_pass,
            "H5_zero_platinum": h5_pass,
            "H6_at_most_10pct_fail": h6_pass,
            "H7_at_least_5_permanently_ambiguous_subfields": h7_pass,
            "H8_zero_atomic_full_pass": h8_pass,
        },
        "mean_coverage": round(mean_cov, 2),
        "n_platinum": n_platinum,
        "n_fail": n_fail,
        "n_ambiguous_perm": n_ambiguous_perm,
        "n_atomically_pass": n_atomically_pass,
        "spread_max_minus_min_cell_means": round(spread, 2),
        "missing_fields": missing_fields,
        "ambiguous_fields": ambiguous_fields,
        "per_item": item_rows,
        "stratified": stratified_rows,
    }

    # --- Write outputs ---
    with (OUT_DIR / "p5_iter205_subfield_class.tsv").open("w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["item", "sub_field", "n_extract_literal", "n_extract_parsed",
                        "n_ambiguous", "n_missing", "n_covered", "coverage_pct"],
            delimiter="\t",
        )
        wr.writeheader()
        for r in subfield_rows:
            wr.writerow(r)

    with (OUT_DIR / "p5_iter205_item_coverage.tsv").open("w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["item", "n_subfields", "n_subfields_full_coverage",
                        "manifests_passing_item", "manifests_passing_pct"],
            delimiter="\t",
        )
        wr.writeheader()
        for r in item_rows:
            wr.writerow(r)

    with (OUT_DIR / "p5_iter205_per_manifest_coverage.tsv").open("w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["cell_id", "model_family", "task_slice", "n_literal",
                        "n_parsed", "n_ambiguous", "n_missing", "n_covered",
                        "missing", "ambiguous", "tier"],
            delimiter="\t",
        )
        wr.writeheader()
        for r in manifest_rows:
            wr.writerow(r)

    with (OUT_DIR / "p5_iter205_stratified_coverage.tsv").open("w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["model_family", "task_slice", "n_manifests",
                        "mean_subfields_covered", "ci_lo", "ci_hi",
                        "frac_above_15"],
            delimiter="\t",
        )
        wr.writeheader()
        for r in stratified_rows:
            wr.writerow(r)

    with (OUT_DIR / "p5_iter205_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # --- Console summary ---
    print(f"[iter205] Wrote 4 TSVs + summary.json to {OUT_DIR}")
    print(f"[iter205] Mean coverage = {mean_cov:.2f} / 23 ({100*mean_cov/23:.1f}%)")
    print(f"[iter205] Hypotheses: H1={h1_pass} H2={h2_pass} H3={h3_pass} "
          f"H4={h4_pass} H5={h5_pass} H6={h6_pass} H7={h7_pass} H8={h8_pass}")
    print(f"[iter205] Per-Item pass: " +
          ", ".join(f"{r['item']}={r['manifests_passing_pct']:.0f}%" for r in item_rows))
    print(f"[iter205] Platinum manifests: {n_platinum}, FAIL manifests: {n_fail}")
    print(f"[iter205] Missing fields (0/{n_manifests}): " +
          ", ".join(f"{m['item']}/{m['sub_field']}" for m in missing_fields))
    print(f"[iter205] Ambiguous fields: " +
          ", ".join(f"{a['item']}/{a['sub_field']}({a['n_ambiguous']})"
                    for a in ambiguous_fields))


if __name__ == "__main__":
    main()