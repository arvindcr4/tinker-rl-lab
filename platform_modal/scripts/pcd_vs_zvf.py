#!/usr/bin/env python3
"""Recompute PCD, exact-tie ZVF, and tolerance-based ZVF from stored rewards.

No training or external services are used. PCD remains the population variance
(ddof=0); canonical tolerance-based ZVF uses sample variance (ddof=1) <= 1e-6,
as specified in paper/sections/zvf_pipeline_spec.tex. The historical ``zvf_ind``
and output keys without a qualifier retain their exact-zero definitions.

The seeded micro-jitter probe tests exact-zero brittleness. It does NOT falsify
tolerance-based ZVF: both definitions are now reported, with an epsilon sweep.
The cross-run comparison still uses logged summaries; it is not a recomputed
PCD-vs-ZVF predictive comparison or evidence of learning benefit.

The historical pcd_vs_zvf_summary.tsv is read for provenance and never rewritten;
fresh summary values are written to pcd_vs_zvf_recomputed_summary.tsv instead.

Run from the repository root with Python 3; only the standard library is used.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path


RES = "platform_hybrid/experiments/results"
DEFAULT_EPSILON = 1e-6
DEFAULT_EPSILON_GRID = (0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2)


def pvar(xs):
    """Population variance, preserving the historical PCD computation."""
    if not xs:
        raise ValueError("variance needs at least one reward")
    n = len(xs)
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / n


def svar(xs):
    """Sample variance, matching rewards.var(axis=-1, ddof=1)."""
    if len(xs) < 2:
        raise ValueError("sample variance needs at least two rewards")
    return pvar(xs) * len(xs) / (len(xs) - 1)


def zvf_exact_ind(group):
    """Historical exact-zero population-variance indicator (epsilon=0)."""
    return float(pvar(group) == 0.0)


def zvf_ind(group):
    """Backward-compatible exact-zero indicator; NOT canonical tolerance ZVF."""
    return zvf_exact_ind(group)


def _validate_nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def zvf_tolerance_ind(group, epsilon=DEFAULT_EPSILON):
    """Canonical sample-variance threshold indicator (inclusive boundary)."""
    _validate_nonnegative(epsilon, "epsilon")
    return float(svar(group) <= epsilon)


def pcd(group):
    return pvar(group)


def phat(group):
    return sum(group) / len(group)


def rankdata(xs):
    """Average ranks, ties shared."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n < 2 or len(ys) != n:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / math.sqrt(sx * sy)


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


def load_groups(results_dir):
    """Load raw binary GSM8K groups in the historical lexicographic file order."""
    groups, sources = [], []
    for path in sorted(Path(results_dir).glob("tinker_gsm8k_zvf_s*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        if "per_problem" not in data:
            continue  # The glob also matches the per-seed summary file.
        start = len(groups)
        for pp in data["per_problem"]:
            group = [float(reward) for reward in pp["rewards"]]
            if len(group) < 2 or any(reward not in (0.0, 1.0) for reward in group):
                raise ValueError(f"{path}: expected binary groups with at least two rewards")
            groups.append(group)
        sources.append({
            "file": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "seed": data.get("seed"),
            "group_start": start,
            "n_groups": len(groups) - start,
        })
    if not groups:
        raise ValueError(f"no raw GSM8K reward groups found in {results_dir}")
    if len({len(group) for group in groups}) != 1:
        raise ValueError("GSM8K groups must all have the same group size")
    return groups, sources


def analyze_jitter(groups, seed=0, amplitude=1e-4, epsilon=DEFAULT_EPSILON,
                   epsilon_grid=DEFAULT_EPSILON_GRID):
    """Compute both definitions using one shared, deterministic jitter draw.

    Rewards are neither clipped nor renormalized, retaining the original probe.
    A local Random instance leaves the caller's global RNG state untouched.
    """
    _validate_nonnegative(amplitude, "amplitude")
    _validate_nonnegative(epsilon, "epsilon")
    if not groups or any(len(group) < 2 for group in groups):
        raise ValueError("need nonempty groups with at least two rewards each")
    if any(not math.isfinite(reward) for group in groups for reward in group):
        raise ValueError("rewards must be finite")
    thresholds = sorted(set(epsilon_grid) | {epsilon})
    for threshold in thresholds:
        _validate_nonnegative(threshold, "epsilon")
    rng = random.Random(seed)
    jittered = [[r + rng.uniform(0, amplitude) for r in group] for group in groups]
    n = len(groups)
    population_before = [pvar(group) for group in groups]
    population_after = [pvar(group) for group in jittered]
    sample_before = [svar(group) for group in groups]
    sample_after = [svar(group) for group in jittered]
    exact_before = [value == 0.0 for value in population_before]
    exact_after = [value == 0.0 for value in population_after]
    sensitivity = []
    for threshold in thresholds:
        before = [value <= threshold for value in sample_before]
        after = [value <= threshold for value in sample_after]
        sensitivity.append({
            "epsilon": threshold,
            "variance_ddof": 1,
            "n_groups": n,
            "before_count": sum(before),
            "after_count": sum(after),
            "before_fraction": sum(before) / n,
            "after_fraction": sum(after) / n,
            "changed_count": sum(a != b for a, b in zip(before, after)),
        })
    canonical = next(row for row in sensitivity if row["epsilon"] == epsilon)
    pcd_before, pcd_after = sum(population_before) / n, sum(population_after) / n
    summary = {
        "n_groups": n,
        "group_sizes": sorted({len(group) for group in groups}),
        "jitter_seed": seed,
        "jitter_amplitude": amplitude,
        "jitter_distribution": "independent random.Random(seed).uniform(0, amplitude)",
        "jitter_transform": "reward + jitter; no clipping or rescaling",
        "exact_zero_definition": "population variance (ddof=0) == 0; historical exact-tie proxy",
        "tolerance_definition": "sample variance (ddof=1) <= epsilon",
        "tolerance_epsilon": epsilon,
        "pcd_definition": "mean population variance (ddof=0)",
        "exact_zero_before_count": sum(exact_before),
        "exact_zero_after_count": sum(exact_after),
        "exact_zero_before_fraction": sum(exact_before) / n,
        "exact_zero_after_fraction": sum(exact_after) / n,
        "tolerance_before_count": canonical["before_count"],
        "tolerance_after_count": canonical["after_count"],
        "tolerance_before_fraction": canonical["before_fraction"],
        "tolerance_after_fraction": canonical["after_fraction"],
        "tolerance_changed_count": canonical["changed_count"],
        "pcd_before": pcd_before,
        "pcd_after": pcd_after,
        "pcd_delta": pcd_after - pcd_before,
        "epsilon_sensitivity": sensitivity,
        "claim_boundary": (
            "Exact-zero brittleness does not establish failure of tolerance-based ZVF. "
            "This stored-group perturbation is not a training or controller comparison; "
            "it does not establish that jitter supplies useful learning signal or that "
            "PCD improves learning. PCD changes slightly rather than being invariant."
        ),
        "legacy_output_aliases": {
            "zvf_batch_before_jitter": "exact_zero_before_fraction",
            "zvf_batch_after_jitter": "exact_zero_after_fraction",
            "mean_zvf_ind": "exact-zero population-variance indicator",
        },
    }
    details = [{
        "group_index": index,
        "population_variance_before": population_before[index],
        "population_variance_after": population_after[index],
        "sample_variance_before": sample_before[index],
        "sample_variance_after": sample_after[index],
        "exact_zero_before": int(exact_before[index]),
        "exact_zero_after": int(exact_after[index]),
        "tolerance_before": int(sample_before[index] <= epsilon),
        "tolerance_after": int(sample_after[index] <= epsilon),
    } for index in range(n)]
    return summary, details


def cross_run_metrics(path):
    """Preserve the historical logged-summary comparison and output names."""
    with Path(path).open() as handle:
        rows = list(csv.DictReader((line for line in handle if not line.startswith("#")),
                                   delimiter="\t"))
    use = [r for r in rows if r.get("last10_avg", "") not in ("", "nan")
           and r.get("mean_zvf", "") not in ("", "nan")]

    def col(row, key):
        try:
            return float(row[key])
        except (KeyError, ValueError, TypeError):
            return float("nan")

    zvf = [col(r, "mean_zvf") for r in use]
    reward = [col(r, "mean_reward") for r in use]
    outcome = [col(r, "last10_avg") for r in use]
    collapse = [float(r.get("failure_label", "") == "collapse") for r in use]
    return {
        "n_runs_crossrun": len(use),
        "spearman_zvf_outcome": spearman(zvf, outcome),
        "spearman_meanreward_outcome": spearman(reward, outcome),
        "spearman_zvf_collapse": spearman(zvf, collapse),
        "spearman_meanreward_collapse": spearman(reward, collapse),
    }


def write_tsv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(RES))
    parser.add_argument("--output-dir", type=Path,
                        help="defaults to results-dir; input reward tensors are never modified")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter-amplitude", type=float, default=1e-4)
    parser.add_argument("--variance-epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--epsilon-grid", type=float, nargs="+", default=DEFAULT_EPSILON_GRID)
    args = parser.parse_args(argv)
    groups, sources = load_groups(args.results_dir)
    analysis, details = analyze_jitter(groups, args.seed, args.jitter_amplitude,
                                      args.variance_epsilon, args.epsilon_grid)
    analysis["sources"] = sources
    analysis["source_order"] = "lexicographic filenames, then original per_problem order"
    analysis["schema_version"] = 1
    for source in sources:
        subset = details[source["group_start"]:source["group_start"] + source["n_groups"]]
        source["exact_zero_before_count"] = sum(row["exact_zero_before"] for row in subset)
        source["exact_zero_after_count"] = sum(row["exact_zero_after"] for row in subset)
        source["tolerance_before_count"] = sum(row["tolerance_before"] for row in subset)
        source["tolerance_after_count"] = sum(row["tolerance_after"] for row in subset)
        for index, row in enumerate(subset):
            row["source_file"] = source["file"]
            row["source_group_index"] = index

    mastered = sum(phat(group) == 1.0 for group in groups)
    incapable = sum(phat(group) == 0.0 for group in groups)
    analysis["mastered_count"] = mastered
    analysis["incapable_count"] = incapable
    analysis["mixed_count"] = len(groups) - mastered - incapable
    group_size = len(groups[0])
    shape = []
    for k in range(group_size + 1):
        bucket = [group for group in groups if abs(phat(group) - k / group_size) < 1e-9]
        if bucket:
            shape.append({
                "p_x": f"{k / group_size:.4f}", "n": len(bucket),
                "mean_zvf_ind": f"{sum(map(zvf_ind, bucket)) / len(bucket):.4f}",
                "mean_pcd": f"{sum(map(pcd, bucket)) / len(bucket):.4f}",
            })

    cross_run_path = args.results_dir / "zvf_summary.tsv"
    cross_run = cross_run_metrics(cross_run_path)
    analysis["cross_run_provenance"] = {
        "source_file": cross_run_path.name,
        "source_sha256": hashlib.sha256(cross_run_path.read_bytes()).hexdigest(),
        "current_input_recomputation": cross_run,
        "scope": "Provenance check only; historical cross-run artifact is retained unchanged.",
    }
    historical_path = args.results_dir / "pcd_vs_zvf_summary.tsv"
    analysis["historical_summary"] = {
        "file": historical_path.name,
        "sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
        "status": (
            "Preserved unchanged. Superseded for unqualified jitter/ZVF interpretation only: "
            "its before/after jitter ZVF fields denote exact-zero variance, not tolerance ZVF. "
            "Historical cross-run correlations are not revised by this tolerance audit."
        ),
    }
    metrics = {
        "n_groups": len(groups),
        # Retain historical names and precision in the NEW summary, with explicit aliases.
        "zvf_batch_before_jitter": f"{analysis['exact_zero_before_fraction']:.4f}",
        "zvf_batch_after_jitter": f"{analysis['exact_zero_after_fraction']:.4f}",
        "pcd_batch_before_jitter": f"{analysis['pcd_before']:.6f}",
        "pcd_batch_after_jitter": f"{analysis['pcd_after']:.6f}",
        **{key: value if isinstance(value, int) else f"{value:.4f}"
           for key, value in cross_run.items()},
        "zvf_exact_batch_before_jitter": analysis["exact_zero_before_fraction"],
        "zvf_exact_batch_after_jitter": analysis["exact_zero_after_fraction"],
        "zvf_tolerance_batch_before_jitter": analysis["tolerance_before_fraction"],
        "zvf_tolerance_batch_after_jitter": analysis["tolerance_after_fraction"],
        "zvf_tolerance_variance_ddof": 1,
        "zvf_tolerance_epsilon": args.variance_epsilon,
        "pcd_batch_delta": analysis["pcd_delta"],
        "jitter_seed": args.seed,
        "jitter_amplitude": args.jitter_amplitude,
    }
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "pcd_vs_zvf_shape.tsv", shape)
    write_tsv(output_dir / "pcd_vs_zvf_recomputed_summary.tsv",
              [{"metric": key, "value": value} for key, value in metrics.items()])
    write_tsv(output_dir / "pcd_vs_zvf_epsilon_sensitivity.tsv", analysis["epsilon_sensitivity"])
    write_tsv(output_dir / "pcd_vs_zvf_jitter_groups.tsv", details)
    (output_dir / "pcd_vs_zvf_tolerance_analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + "\n")

    print(f"Loaded {len(groups)} groups (G={group_size}); all-correct={mastered}, "
          f"all-wrong={incapable}, mixed={analysis['mixed_count']}")
    print("Exact-zero ZVF (historical alias): "
          f"{analysis['exact_zero_before_count']}/{len(groups)} -> "
          f"{analysis['exact_zero_after_count']}/{len(groups)}")
    print(f"Tolerance ZVF (sample variance <= {args.variance_epsilon:g}): "
          f"{analysis['tolerance_before_count']}/{len(groups)} -> "
          f"{analysis['tolerance_after_count']}/{len(groups)}")
    print(f"PCD: {analysis['pcd_before']:.12f} -> {analysis['pcd_after']:.12f}; "
          f"delta={analysis['pcd_delta']:+.6g}")
    print(analysis["claim_boundary"])
    print(f"Logged-summary cross-run comparison: {cross_run}")
    print(f"Wrote analysis artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
