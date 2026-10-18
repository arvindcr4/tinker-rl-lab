"""Synthetic anchor-heuristic search benchmark, protocol v2.

This is not MIPRO, a Gaussian process, or an empirical instruction-tuning result.
All optimizers share each seeded artificial landscape and the same budget.
Observed score variance is descriptive dispersion, not posterior uncertainty.
"""

from __future__ import annotations
import csv
import json
import math
import random
import statistics as st
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT, RESULTS_DIR
from _review01_contracts import (
    OUTPUT_ROOT,
    evidence_metadata,
    require_inputs,
    write_artifact_metadata,
)

WORKTREE = REPO_ROOT
RESULTS = RESULTS_DIR / "berkeley"
OUT = OUTPUT_ROOT / "synthetic_anchor_search"
INSTRUCTIONS = ["i0_baseline", "i1_step_by_step", "i2_short_answer", "i3_long_reasoning"]
N_SEEDS = 5
EVAL_BUDGET = 24
STRATEGIES = ("RANDOM", "GRIDSEARCH", "ANCHOR_UCB_HEURISTIC")


def load_anchor_features():
    path = RESULTS / "eureka_rqs_per_anchor.tsv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = {}
    for row in rows:
        name = row.get("model") or row.get("anchor")
        if not name or name in result:
            raise ValueError("Missing or duplicate anchor identity")
        result[name] = {
            key: float(row[key]) for key in ["r_mean", "r_var", "frac_above_0p5", "RQS"]
        }
    return result


def latent_score(anchor_feat, instruction, anchor_idx, inst_idx, rng):
    base = (
        0.55 * anchor_feat["r_mean"]
        + 0.2 * (1 - min(anchor_feat["r_var"], 1))
        + 0.15 * anchor_feat["frac_above_0p5"]
        + 0.1 * anchor_feat["RQS"]
    )
    bias = [
        [0, 0, 0, 0],
        [0.05, 0.04, -0.02, -0.03],
        [-0.04, -0.03, 0.03, 0.02],
        [0.02, 0.01, 0.04, 0.05],
    ][inst_idx][anchor_idx % 4]
    return min(1.0, max(0.0, base + bias + rng.uniform(-0.02, 0.02)))


def _validate_budget(anchors, instructions, budget, oracle):
    if not isinstance(budget, int) or budget < 0:
        raise ValueError("Evaluation budget must be a nonnegative integer")
    cells = [(a, i) for a in anchors for i in instructions]
    if len(set(cells)) != len(cells) or not cells or set(cells) != set(oracle):
        raise ValueError("Unique nonempty candidate grid must match oracle identities")
    if not all(math.isfinite(v) for v in oracle.values()):
        raise ValueError("Nonfinite synthetic score")
    return cells, min(budget, len(cells))


def _result(strategy, observations, oracle):
    values = [row["score"] for row in observations]
    return {
        "strategy": strategy,
        "observations": observations,
        "history": values,
        "best_so_far": max(values) if values else None,
        "cum_regret": sum(max(oracle.values()) - v for v in values),
        "observed_score_variance": st.pvariance(values) if len(values) > 1 else None,
        "evaluations": len(values),
        "evidence_type": "synthetic_benchmark",
    }


def random_strategy(anchors, instructions, budget, oracle, rng):
    cells, budget = _validate_budget(anchors, instructions, budget, oracle)
    rng.shuffle(cells)
    return _result(
        "RANDOM",
        [{"anchor": a, "instruction": i, "score": oracle[a, i]} for a, i in cells[:budget]],
        oracle,
    )


def gridsearch_strategy(anchors, instructions, budget, oracle, rng):
    cells, budget = _validate_budget(anchors, instructions, budget, oracle)
    return _result(
        "GRIDSEARCH",
        [{"anchor": a, "instruction": i, "score": oracle[a, i]} for a, i in cells[:budget]],
        oracle,
    )


def mipro_bo_strategy(anchors, instructions, budget, oracle, rng):
    """Legacy name for a bounded anchor-UCB heuristic; not a GP or MIPRO implementation."""
    cells, budget = _validate_budget(anchors, instructions, budget, oracle)
    observed = []
    remaining = set(cells)
    order = list(anchors)
    rng.shuffle(order)
    for anchor in order[:budget]:
        instruction = rng.choice(instructions)
        observed.append(
            {"anchor": anchor, "instruction": instruction, "score": oracle[anchor, instruction]}
        )
        remaining.remove((anchor, instruction))
    while len(observed) < budget:

        def acquisition(cell):
            values = [row["score"] for row in observed if row["anchor"] == cell[0]]
            return (
                (
                    st.mean(values)
                    + 1.5 * math.sqrt((st.pvariance(values) if len(values) > 1 else 0.1) + 0.001)
                )
                if values
                else float("inf")
            )

        cell = max(sorted(remaining), key=acquisition)
        remaining.remove(cell)
        observed.append({"anchor": cell[0], "instruction": cell[1], "score": oracle[cell]})
    return _result("ANCHOR_UCB_HEURISTIC", observed, oracle)


def early_discovery(oracle, observations, cutoff=8):
    """Exact per-anchor top-quartile discovery within a GLOBAL evaluation budget.

    For 12 anchors the legacy >=9 within 8 criterion is impossible; report that
    design error instead of substituting exhaustive-search final maxima.
    """
    anchors = sorted({a for a, _ in oracle})
    thresholds = {
        a: float(np.quantile([v for (anchor, _), v in oracle.items() if anchor == a], 0.75))
        for a in anchors
    }
    found = {
        row["anchor"] for row in observations[:cutoff] if row["score"] >= thresholds[row["anchor"]]
    }
    required = math.ceil(0.75 * len(anchors))
    return {
        "n_hits": len(found),
        "n_anchors": len(anchors),
        "cutoff": cutoff,
        "thresholds": thresholds,
        "required_hits": required,
        "status": "infeasible_legacy_global_budget" if required > cutoff else "descriptive",
        "threshold_met": len(found) >= required,
    }


def cohen_d(a, b):
    """Sample-pooled standardized effect; unavailable for insufficient/constant data."""
    if len(a) < 2 or len(b) < 2:
        return None
    variance = ((len(a) - 1) * st.variance(a) + (len(b) - 1) * st.variance(b)) / (
        len(a) + len(b) - 2
    )
    return (st.mean(a) - st.mean(b)) / math.sqrt(variance) if variance > 0 else None


def summarize_comparisons(runs):
    paired = []
    for run in runs:
        heuristic = run["strategies"]["ANCHOR_UCB_HEURISTIC"]
        random_result = run["strategies"]["RANDOM"]
        if heuristic["evaluations"] != random_result["evaluations"]:
            raise ValueError("Optimizer comparisons require matched evaluation budgets")
        paired.append(
            {
                "landscape_seed": run["landscape_seed"],
                "quality_difference": heuristic["best_so_far"] - random_result["best_so_far"],
                "regret_difference": heuristic["cum_regret"] - random_result["cum_regret"],
            }
        )
    differences = [row["quality_difference"] for row in paired]
    sd = st.stdev(differences) if len(differences) > 1 else 0
    return {
        "paired_landscape_differences": paired,
        "H1_mean_regret_difference": st.mean(row["regret_difference"] for row in paired),
        "H2_early_discovery": [run["early_discovery"] for run in runs],
        "H3_mean_selection_quality_difference": st.mean(differences),
        "H3_paired_d_z": st.mean(differences) / sd if sd else None,
        "H4_fraction_of_known_synthetic_optimum": [
            run["strategies"]["ANCHOR_UCB_HEURISTIC"]["best_so_far"] / run["oracle_max"]
            if run["oracle_max"] > 0
            else None
            for run in runs
        ],
        "H5_observed_score_variances": [
            {
                strategy: value["observed_score_variance"]
                for strategy, value in run["strategies"].items()
            }
            for run in runs
        ],
        "inference_status": "synthetic_descriptive_only",
    }


def main():
    path = RESULTS / "eureka_rqs_per_anchor.tsv"
    require_inputs(path)
    feats = load_anchor_features()
    anchors = sorted(feats)
    if not anchors:
        raise ValueError("No anchor features")
    budget = min(EVAL_BUDGET, len(anchors) * len(INSTRUCTIONS) - 1)
    runs = []
    for seed in range(N_SEEDS):
        landscape_seed = 20260704 + seed * 113
        rng = random.Random(landscape_seed)
        oracle = {
            (a, inst): latent_score(feats[a], inst, ai, ii, rng)
            for ai, a in enumerate(anchors)
            for ii, inst in enumerate(INSTRUCTIONS)
        }
        searches = {}
        for strategy, function in zip(
            STRATEGIES, (random_strategy, gridsearch_strategy, mipro_bo_strategy), strict=False
        ):
            searches[strategy] = function(
                anchors, INSTRUCTIONS, budget, oracle, random.Random(1000 + seed)
            )
        runs.append(
            {
                "landscape_seed": landscape_seed,
                "search_seed": 1000 + seed,
                "strategies": searches,
                "early_discovery": early_discovery(
                    oracle, searches["ANCHOR_UCB_HEURISTIC"]["observations"]
                ),
                "oracle_max": max(oracle.values()),
            }
        )
    metadata = evidence_metadata(
        "seeded_artificial_landscapes_equal_budget_anchor_UCB_grid_random",
        [path],
        kind="synthetic_benchmark",
    )
    summary = {
        **metadata,
        "evaluation_budget_per_strategy": budget,
        "runs": runs,
        "comparisons": summarize_comparisons(runs),
        "empirical_optimizer_claim": False,
        "posterior_uncertainty": None,
        "verdict": "SYNTHETIC_DEMONSTRATION_ONLY",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic_anchor_search.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (OUT / "mipro_pareto_regret_curve.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "landscape_seed",
                "search_seed",
                "strategy",
                "evaluation",
                "anchor",
                "instruction",
                "score",
                "running_best",
                "cumulative_regret",
                "observed_score_variance",
            ]
        )
        for run in runs:
            for strategy, result in run["strategies"].items():
                scores = []
                for step, observation in enumerate(result["observations"], 1):
                    scores.append(observation["score"])
                    writer.writerow(
                        [
                            run["landscape_seed"],
                            run["search_seed"],
                            strategy,
                            step,
                            observation["anchor"],
                            observation["instruction"],
                            observation["score"],
                            max(scores),
                            sum(run["oracle_max"] - score for score in scores),
                            st.pvariance(scores) if len(scores) > 1 else None,
                        ]
                    )
    with (OUT / "mipro_pareto_paired.tsv").open("w") as stream:
        rows = summary["comparisons"]["paired_landscape_differences"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write_artifact_metadata(OUT, metadata)
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
