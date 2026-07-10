#!/usr/bin/env python3
"""P5 MIN-REPORT x external-reporting-standards gap audit (iter 17, JOB A).

Maps each of the seven MIN-REPORT items of platform_hybrid/paper/sections/p5_stack.tex
against the published axes of two complementary reporting standards:

  * Model Cards for Model Reporting (Mitchell et al., FAT* 2019) -
    9 recommended sections per paper Sec. 3: (1) intended use, (2) factors,
    (3) metrics, (4) evaluation data, (5) training data, (6) quantitative
    analyses, (7) ethical considerations, (8) caveats and recommendations,
    (9) explainability/interpretability. We audit 8 (the 9th is method-
    dependent and not a domain-agnostic MIN-REPORT axis).

  * Datasheets for Datasets (Gebru et al., CACM 2021) -
    7 sections per paper Sec. 3: (1) motivation, (2) composition,
    (3) collection process, (4) preprocessing/labeling/cleaning,
    (5) intended uses, (6) distribution/maintenance, (7) tasks.

For each external axis we probe every manifest in the worktree and record
whether the axis is covered (any plausible key+value pair is present),
not covered (no key found), or honest-n/a ("n/a-*" sentinel). We then
sum three quantitative gap scores per axis:

  coverage      = share of manifests with a non-empty value for the axis
  honest_na     = share of manifests with an honest-n/a declaration
  gap_score     = 1 - coverage - honest_na    # the operational gap

The script writes:
  platform_hybrid/experiments/results/p5p8/p5_minreport_external_alignment.tsv
  platform_hybrid/experiments/results/p5p8/p5_minreport_external_alignment.json

The seven MIN-REPORT items are then cross-tabulated against the 8 model-
card axes and the 7 datasheet axes, exposing which axes MIN-REPORT
fully covers, partially covers, and does not cover at all (the headline
gap).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIRS = [
    ROOT / "experiments" / "results" / "mega_20260704" / "manifests",
    ROOT / "experiments" / "results" / "quick_20260704",
]
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"

# ---------------------------------------------------------------------------
# External reporting-standards axes with their probe keys and value-regex
# ---------------------------------------------------------------------------
# (axis_id, axis_name, [alt-keys], value-regex (None = any non-empty),
#  source-paper)
# (axis_id, axis_name, [alt-keys], value-regex (None = any non-empty),
#  source-paper-key)
MC_AXES = [  # Model Cards for Model Reporting (Mitchell et al., FAT* 2019)
    ("mc_intended_use",
     "Intended use & out-of-scope uses",
     ["intended_use", "intended_uses", "primary_intended_use",
      "out_of_scope_uses", "use_cases", "primary_use"],
     r".*",
     "mitchell2019modelcards"),
    ("mc_factors",
     "Relevant factors & subpopulations",
     ["factors", "subpopulations", "subgroups", "demographics",
      "evaluation_factors"],
     r".*",
     "mitchell2019modelcards"),
    ("mc_metrics",
     "Metrics chosen (with motivations)",
     ["metrics", "decision_metrics", "performance_metrics",
      "evaluation_metrics", "metric_descriptions"],
     r".*",
     "mitchell2019modelcards"),
    ("mc_eval_data",
     "Evaluation data & details",
     ["evaluation_data", "eval_data", "test_data", "validation_data",
      "heldout_split", "test_set", "benchmark"],
     r".*",
     "mitchell2019modelcards"),
    ("mc_training_data",
     "Training data & details",
     ["training_data", "train_data", "training_corpus",
      "pretraining_data", "training_set", "finetuning_data"],
     r".*",
     "mitchell2019modelcards"),
    ("mc_quant_analyses",
     "Quantitative analyses (intersectional, breakdown)",
     ["quantitative_analyses", "intersectional_analyses",
      "subgroup_metrics", "breakdown_analyses", "per_slice_metrics"],
     r".*",
     "mitchell2019modelcards"),
    ("mc_ethical",
     "Ethical considerations, risks, harms",
     ["ethical_considerations", "ethical", "risks", "harms",
      "biases", "known_biases", "fairness_analyses"],
     r".*",
     "mitchell2019modelcards"),
    ("mc_caveats",
     "Caveats & recommendations",
     ["caveats", "caveats_and_recommendations", "limitations",
      "recommendations", "known_limitations"],
     r".*",
     "mitchell2019modelcards"),
]

DS_AXES = [  # Datasheets for Datasets (Gebru et al., CACM 2021)
    ("ds_motivation",
     "Motivation for dataset creation",
     ["motivation", "dataset_motivation", "purpose", "rationale",
      "creation_purpose"],
     r".*",
     "gebru2021datasheets"),
    ("ds_composition",
     "Dataset composition (instances, splits, missingness)",
     ["composition", "dataset_composition", "instances",
      "instance_count", "data_instances", "splits",
      "missing_values"],
     r".*",
     "gebru2021datasheets"),
    ("ds_collection",
     "Data collection process (sampling, time period, consent)",
     ["collection_process", "data_collection", "sampling_process",
      "collection_time", "consent", "data_source"],
     r".*",
     "gebru2021datasheets"),
    ("ds_preprocessing",
     "Preprocessing, cleaning, labeling",
     ["preprocessing", "cleaning", "labeling", "label_process",
      "data_cleaning", "data_preprocessing"],
     r".*",
     "gebru2021datasheets"),
    ("ds_intended_uses",
     "Intended & discouraged uses",
     ["intended_uses", "uses", "intended_use", "discouraged_uses",
      "use_restrictions"],
     r".*",
     "gebru2021datasheets"),
    ("ds_distribution",
     "Dataset distribution & maintenance",
     ["distribution", "maintenance", "updates", "version",
      "versioning", "distribution_maintenance"],
     r".*",
     "gebru2021datasheets"),
    ("ds_tasks",
     "Tasks for which the dataset is suitable",
     ["tasks", "supported_tasks", "benchmark_tasks", "task_suitability",
      "evaluation_tasks"],
     r".*",
     "gebru2021datasheets"),
]

# ---------------------------------------------------------------------------
# Cross-walk: which MIN-REPORT items (1..7) map to which external axes
# (hand-validated mapping against the two papers' Sec. 3 taxonomies)
# ---------------------------------------------------------------------------
MC_CROSSWALK = {
    "mc_intended_use":       [],
    "mc_factors":            [],
    "mc_metrics":            [],
    "mc_eval_data":          [6],   # Item 6 (held-out split) partially covers
    "mc_training_data":      [],
    "mc_quant_analyses":     [4],   # Item 4 (per-step ZVF trajectory) provides
                                    # quantitative breakdowns on the run
    "mc_ethical":            [],
    "mc_caveats":            [],
}

DS_CROSSWALK = {
    "ds_motivation":         [],
    "ds_composition":        [],
    "ds_collection":         [],
    "ds_preprocessing":      [7],   # Item 7 (decontamination) partially covers
    "ds_intended_uses":      [],
    "ds_distribution":       [],
    "ds_tasks":              [],
}

NA_PAT = re.compile(r"^\s*n/?a(\s|-|$)", re.IGNORECASE)


def probe_axis(manifest: dict, axis_keys: list[str], val_regex) -> tuple[str, str]:
    """Return ('covered', value) | ('na', 'n/a-...') | ('gap', '')."""
    found = None
    for k in axis_keys:
        v = manifest.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        found = (k, s)
        break
    if found is None:
        return "gap", ""
    k, s = found
    if NA_PAT.match(s):
        return "na", s
    if val_regex is None or re.search(val_regex, s, re.IGNORECASE):
        return "covered", s
    return "na", s  # unrecognised non-empty treated as honest-na


def load_manifests() -> list[dict]:
    out = []
    for mdir in MANIFEST_DIRS:
        if not mdir.is_dir():
            continue
        for jf in sorted(mdir.glob("*.json")):
            try:
                with jf.open() as f:
                    d = json.load(f)
            except Exception as e:
                print(f"warn: bad json {jf}: {e}", file=sys.stderr)
                continue
            if not any(any(k in d for k in keys)
                       for _, _, keys, _, _ in MC_AXES + DS_AXES):
                continue
            d["_path"] = jf.name
            d["_corpus"] = mdir.name
            out.append(d)
    return out


def audit_axes(manifests: list[dict]) -> dict:
    n = len(manifests)
    coverage = {}
    per_axis_status = defaultdict(list)  # per-axis per-manifest status
    for axes in (MC_AXES, DS_AXES):
        for aid, _name, keys, val_re, _src in axes:
            covered = na = gap = 0
            for m in manifests:
                status, _val = probe_axis(m, keys, val_re)
                per_axis_status[aid].append(status)
                if status == "covered":
                    covered += 1
                elif status == "na":
                    na += 1
                else:
                    gap += 1
            coverage[aid] = {
                "covered": covered,
                "na": na,
                "gap": gap,
                "coverage": covered / n if n else 0,
                "honest_na": na / n if n else 0,
                "gap_score": (n - covered - na) / n if n else 0,
            }
    return {"n": n, "axes": coverage, "per_axis_status": per_axis_status}


def crosswalk_summary(coverage: dict) -> list[dict]:
    """For each external axis, mark which MIN-REPORT item (if any) covers it."""
    out = []
    for axes, source, crosswalk in [
        (MC_AXES, "mitchell2019modelcards", MC_CROSSWALK),
        (DS_AXES, "gebru2021datasheets", DS_CROSSWALK),
    ]:
        for aid, name, _keys, _val, _src in axes:
            cov = coverage["axes"].get(aid, {})
            mr_items = crosswalk.get(aid, [])
            out.append({
                "axis_id": aid,
                "axis_name": name,
                "source": source,
                "minreport_items_partial": mr_items,
                "minreport_coverage": (
                    "partial" if mr_items else "none"
                ),
                "coverage": round(cov.get("coverage", 0), 3),
                "honest_na": round(cov.get("honest_na", 0), 3),
                "gap_score": round(cov.get("gap_score", 0), 3),
                "gap_status": (
                    "MIN-REPORT_NEEDED" if cov.get("gap_score", 0) >= 0.95 else
                    ("GAP_HIGH" if cov.get("gap_score", 0) >= 0.80 else
                     ("GAP_MID" if cov.get("gap_score", 0) >= 0.50 else
                      "GAP_LOW"))
                ),
            })
    return out


def write_outputs(manifests, coverage, cw_summary):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(manifests)
    tsv = OUT_DIR / "p5_minreport_external_alignment.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "axis_id", "axis_name", "source", "minreport_items_partial",
            "coverage", "honest_na", "gap_score", "gap_status",
        ])
        for r in cw_summary:
            w.writerow([
                r["axis_id"], r["axis_name"], r["source"],
                ";".join(str(x) for x in r["minreport_items_partial"]) or "-",
                f"{r['coverage']:.3f}", f"{r['honest_na']:.3f}",
                f"{r['gap_score']:.3f}", r["gap_status"],
            ])
    summary = {
        "n_manifests": n,
        "n_axes_total": len(cw_summary),
        "n_axes_full_coverage": sum(1 for r in cw_summary if r["coverage"] >= 0.95),
        "n_axes_partial": sum(1 for r in cw_summary
                              if 0.05 <= r["coverage"] < 0.95),
        "n_axes_zero_coverage": sum(1 for r in cw_summary if r["coverage"] < 0.05),
        "axes": cw_summary,
        "by_source": {
            "mitchell2019modelcards": [
                r for r in cw_summary if r["source"] == "mitchell2019modelcards"
            ],
            "gebru2021datasheets": [
                r for r in cw_summary if r["source"] == "gebru2021datasheets"
            ],
        },
        "headline_gap_axes": sorted(
            [{"axis": r["axis_id"], "gap": r["gap_score"]}
             for r in cw_summary if r["gap_score"] >= 0.95],
            key=lambda x: -x["gap"],
        ),
        "by_minreport_status": {
            "none": [r for r in cw_summary if not r["minreport_items_partial"]],
            "partial": [r for r in cw_summary if r["minreport_items_partial"]],
        },
    }
    with (OUT_DIR / "p5_minreport_external_alignment.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {tsv} ({len(cw_summary)} rows)")
    print(f"wrote {OUT_DIR / 'p5_minreport_external_alignment.json'}")
    print(f"manifests scanned: {n}")
    print(f"axes with full coverage (>=95%): {summary['n_axes_full_coverage']}/"
          f"{summary['n_axes_total']}")
    print(f"axes with zero coverage (<5%): {summary['n_axes_zero_coverage']}/"
          f"{summary['n_axes_total']}")
    print(f"axes with MIN-REPORT partial coverage: {len(summary['by_minreport_status']['partial'])}/{summary['n_axes_total']}")
    print(f"axes with NO MIN-REPORT coverage: {len(summary['by_minreport_status']['none'])}/{summary['n_axes_total']}")
    print("headline gap axes (gap>=95%):")
    for g in summary["headline_gap_axes"]:
        print(f"  {g['axis']}: gap={g['gap']:.3f}")


if __name__ == "__main__":
    manifests = load_manifests()
    if not manifests:
        print("ERROR: no manifests found", file=sys.stderr)
        sys.exit(1)
    coverage = audit_axes(manifests)
    cw_summary = crosswalk_summary(coverage)
    write_outputs(manifests, coverage, cw_summary)