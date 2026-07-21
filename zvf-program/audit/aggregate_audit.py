#!/usr/bin/env python3
"""Validate and aggregate the locked GRPO survival-audit result directory.

The program is deliberately stdlib-only. It refuses to assign scientific
verdicts when a seed is missing, a held-out set is undersized, a non-target
stack field differs, or a required metric is absent.

Each input file is one JSON record with the fields listed in
``preregistration.json``. Example:

    python3 aggregate_audit.py --input-dir results/full --output results/audit.json
    python3 aggregate_audit.py --input-dir results/full --allow-incomplete
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
LATEX_ARM_NAMES = {
    "dapo": "DAPO",
    "gspo": "GSPO",
    "drgrpo": "DrGRPO",
    "aero": "AERO",
}


class AuditError(ValueError):
    """A preregistration or result violates the locked audit contract."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{path}: top-level JSON value must be an object")
    return value


def percentile(values: list[float], probability: float) -> float:
    """Linear-interpolated quantile matching common scientific packages."""
    if not values:
        raise AuditError("cannot take a percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    differences: list[float], confidence: float, *, resamples: int = 10_000, seed: int = 20_260_714
) -> tuple[float, float]:
    if len(differences) < 2:
        raise AuditError("paired confidence interval requires at least two seeds")
    rng = random.Random(seed)
    n = len(differences)
    means = [
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    ]
    alpha = 1.0 - confidence
    return percentile(means, alpha / 2.0), percentile(means, 1.0 - alpha / 2.0)


def achieved_mde_80(differences: list[float]) -> float:
    """Normal-approximation two-sided alpha=.05, power=.80 MDE."""
    if len(differences) < 2:
        return math.inf
    return (1.959963984540054 + 0.8416212335729143) * statistics.stdev(differences) / math.sqrt(len(differences))


def benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Return BH rejections; included for the locked multiplicity contract."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    largest = -1
    m = len(ordered)
    for index, (_, p_value) in enumerate(ordered, start=1):
        if p_value <= alpha * index / m:
            largest = index
    return {name: index <= largest for index, (name, _) in enumerate(ordered, start=1)}


def validate_records(
    prereg: dict[str, Any], records: Iterable[tuple[Path, dict[str, Any]]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    core = prereg["core_stratum"]
    arms = core["arms"]
    seeds = set(core["seeds"])
    heldout_n = core["heldout"]["n"]
    required = prereg["required_result_fields"]
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    seen: set[tuple[str, int]] = set()
    errors: list[str] = []
    fingerprints: set[str] = set()

    for path, record in records:
        missing = [field for field in required if field not in record]
        if missing:
            errors.append(f"{path}: missing fields {missing}")
            continue
        arm, seed = record["arm"], record["seed"]
        if arm not in arms:
            errors.append(f"{path}: unregistered arm {arm!r}")
            continue
        if seed not in seeds:
            errors.append(f"{path}: seed {seed!r} is not preregistered")
            continue
        key = (arm, seed)
        if key in seen:
            errors.append(f"{path}: duplicate arm/seed record {key}")
            continue
        seen.add(key)
        error_count = len(errors)
        if record["heldout_n"] != heldout_n:
            errors.append(f"{path}: heldout_n={record['heldout_n']} (locked value is {heldout_n})")
        expected_changes = sorted(arms[arm]["allowed_changes"])
        observed_changes = sorted(record["treatment_changes"])
        if observed_changes != expected_changes:
            errors.append(
                f"{path}: treatment_changes={observed_changes}; expected {expected_changes}"
            )
        if not isinstance(record["stack_fingerprint"], str) or not record["stack_fingerprint"]:
            errors.append(f"{path}: stack_fingerprint must be a non-empty string")
        else:
            fingerprints.add(record["stack_fingerprint"])
        for field in ("heldout_score", "last10_reward", "mean_zvf", "mean_gu", "wall_clock_seconds"):
            if not isinstance(record[field], (int, float)) or not math.isfinite(record[field]):
                errors.append(f"{path}: {field} must be a finite number")
        if not isinstance(record["rollouts"], int) or record["rollouts"] < 0:
            errors.append(f"{path}: rollouts must be a non-negative integer")
        if not isinstance(record["collapse"], bool):
            errors.append(f"{path}: collapse must be boolean")
        manifest = Path(record["manifest_path"])
        if not manifest.is_absolute():
            manifest = path.parent / manifest
        if not manifest.is_file():
            errors.append(f"{path}: manifest_path does not exist: {manifest}")
        else:
            try:
                manifest_record = load_json(manifest)
            except AuditError as exc:
                errors.append(str(exc))
            else:
                trace = manifest_record.get("heldout_trace")
                if not isinstance(trace, list) or len(trace) != heldout_n:
                    errors.append(
                        f"{path}: manifest heldout_trace must contain {heldout_n} rows"
                    )
                else:
                    indices = [
                        row.get("index") if isinstance(row, dict) else None
                        for row in trace
                    ]
                    if indices != list(range(heldout_n)):
                        errors.append(f"{path}: manifest heldout indices are not contiguous")
                    hashes = [
                        row.get("completion_sha256") if isinstance(row, dict) else None
                        for row in trace
                    ]
                    valid_hashes = all(
                        isinstance(value, str)
                        and re.fullmatch(r"[0-9a-f]{64}", value)
                        for value in hashes
                    )
                    if not valid_hashes:
                        errors.append(f"{path}: manifest heldout rows lack valid completion hashes")
                    elif len(set(hashes)) != heldout_n:
                        errors.append(f"{path}: manifest completion hashes are not unique")
                    correct = sum(
                        row.get("correct") is True
                        for row in trace
                        if isinstance(row, dict)
                    )
                    if not math.isclose(
                        correct / heldout_n,
                        record["heldout_score"],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        errors.append(f"{path}: manifest trace disagrees with heldout_score")
        if len(errors) == error_count:
            indexed[key] = record

    if len(fingerprints) > 1:
        errors.append(
            "stack lock failed: more than one non-treatment stack_fingerprint was observed"
        )
    return indexed, errors


def missing_units(prereg: dict[str, Any], indexed: dict[tuple[str, int], dict[str, Any]]) -> list[str]:
    core = prereg["core_stratum"]
    return [
        f"{arm}/seed-{seed}"
        for arm in core["arms"]
        for seed in core["seeds"]
        if (arm, seed) not in indexed
    ]


def verdict(
    ci95: tuple[float, float],
    ci90: tuple[float, float],
    mde: float,
    equivalence_margin: float,
    published_delta: float | None,
) -> str:
    lower95, upper95 = ci95
    if upper95 < 0:
        return "REVERSES"
    if (
        ci90[0] >= -equivalence_margin
        and ci90[1] <= equivalence_margin
        and mde <= equivalence_margin
    ):
        return "DISAPPEARS"
    if lower95 > 0:
        if published_delta is not None and lower95 >= 0.5 * abs(published_delta):
            return "RETAINS"
        return "SURVIVES"
    return "INCONCLUSIVE"


def aggregate(prereg: dict[str, Any], indexed: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    core = prereg["core_stratum"]
    baseline = core["baseline_arm"]
    seeds = core["seeds"]
    margin = prereg["analysis"]["equivalence_margin"]
    results: dict[str, Any] = {}
    for arm, arm_spec in core["arms"].items():
        if arm == baseline:
            continue
        differences = [
            indexed[(arm, seed)]["heldout_score"]
            - indexed[(baseline, seed)]["heldout_score"]
            for seed in seeds
        ]
        ci95 = paired_bootstrap_ci(differences, 0.95)
        ci90 = paired_bootstrap_ci(differences, 0.90)
        mde = achieved_mde_80(differences)
        results[arm] = {
            "paired_differences": differences,
            "controlled_delta": statistics.fmean(differences),
            "ci95": list(ci95),
            "ci90": list(ci90),
            "achieved_mde_80": mde,
            "equivalence_margin": margin,
            "published_delta": arm_spec["published_delta"],
            "verdict": verdict(ci95, ci90, mde, margin, arm_spec["published_delta"]),
            "secondary_means": {
                field: statistics.fmean(indexed[(arm, seed)][field] for seed in seeds)
                for field in prereg["secondary_metrics"]
                if field != "collapse"
            },
            "collapse_rate": statistics.fmean(
                float(indexed[(arm, seed)]["collapse"]) for seed in seeds
            ),
        }
    return {
        "status": "COMPLETE",
        "schema_version": prereg["schema_version"],
        "n_seeds": len(seeds),
        "baseline_arm": baseline,
        "results": results,
    }


def render_latex_results(report: dict[str, Any]) -> str:
    """Render the complete aggregate as generated LaTeX macros for R08."""
    if report.get("status") != "COMPLETE":
        raise AuditError("LaTeX results require a COMPLETE aggregate")
    results = report.get("results")
    if not isinstance(results, dict):
        raise AuditError("complete aggregate has no results object")

    lines = [
        "% Generated by aggregate_audit.py; do not edit by hand.",
        f"\\newcommand{{\\AuditStatus}}{{{report['status']}}}",
        f"\\newcommand{{\\AuditNSeeds}}{{{int(report['n_seeds'])}}}",
    ]
    for arm, tex_name in LATEX_ARM_NAMES.items():
        result = results.get(arm)
        if not isinstance(result, dict):
            raise AuditError(f"complete aggregate is missing {arm!r}")
        ci95 = result.get("ci95")
        if not isinstance(ci95, list) or len(ci95) != 2:
            raise AuditError(f"complete aggregate has invalid {arm!r} ci95")
        lines.extend(
            [
                f"\\newcommand{{\\Audit{tex_name}Delta}}{{{float(result['controlled_delta']):+.5f}}}",
                f"\\newcommand{{\\Audit{tex_name}CILow}}{{{float(ci95[0]):+.5f}}}",
                f"\\newcommand{{\\Audit{tex_name}CIHigh}}{{{float(ci95[1]):+.5f}}}",
                f"\\newcommand{{\\Audit{tex_name}MDE}}{{{float(result['achieved_mde_80']):.5f}}}",
                f"\\newcommand{{\\Audit{tex_name}Verdict}}{{{result['verdict']}}}",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=HERE / "preregistration.json")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tex-output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    prereg = load_json(args.prereg)
    paths = sorted(args.input_dir.glob("*.json")) if args.input_dir.is_dir() else []
    records = [(path, load_json(path)) for path in paths]
    indexed, errors = validate_records(prereg, records)
    missing = missing_units(prereg, indexed)

    if errors or missing:
        report = {
            "status": "PREREGISTERED-NOT-RUN" if not indexed else "INCOMPLETE",
            "validated_units": len(indexed),
            "required_units": sum(
                1 for _ in prereg["core_stratum"]["arms"] for _ in prereg["core_stratum"]["seeds"]
            ),
            "errors": errors,
            "missing_units": missing,
            "verdicts_emitted": False,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        if args.tex_output:
            args.tex_output.unlink(missing_ok=True)
        print(rendered)
        return 0 if args.allow_incomplete else 2

    report = aggregate(prereg, indexed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.tex_output:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text(render_latex_results(report))
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
