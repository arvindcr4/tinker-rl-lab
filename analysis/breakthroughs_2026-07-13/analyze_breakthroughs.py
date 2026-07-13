#!/usr/bin/env python3
"""Cross-dataset breakthrough audit for Tinker RL experiment data.

This script is deterministic and uses only checked-in experiment artifacts. It
tests the exact pass@k/pass^k/ZVF identity, a conditional group-size optimum,
the fixed-mean polarization paradox, and saturation asymmetry in the empirical
G' controller.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "platform_hybrid" / "experiments" / "results"
OUT = Path(__file__).resolve().parent
SEED = 20_260_713
BOOTSTRAPS = 5_000


def load_prompt_probabilities() -> tuple[list[float], dict[str, object]]:
    source = RESULTS / "zvf_iter46_per_prompt_isog.tsv"
    seen: dict[tuple[str, str], float] = {}
    source_rows = 0
    conflicts = 0
    with source.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            row = line.rstrip("\n").split("\t")
            if not row or row[0] == "source":
                continue
            source_rows += 1
            key = (row[1], row[2])
            value = float(row[3])
            if key in seen and seen[key] != value:
                conflicts += 1
            seen[key] = value
    return list(seen.values()), {
        "source_rows": source_rows,
        "unique_prompt_seed_keys": len(seen),
        "duplicate_rows_from_repeated_group_settings": source_rows - len(seen),
        "conflicting_p_x_within_key": conflicts,
    }


def pass_power(probabilities: list[float], k: int) -> float:
    return sum(p**k for p in probabilities) / len(probabilities)


def fail_power(probabilities: list[float], k: int) -> float:
    return sum((1.0 - p) ** k for p in probabilities) / len(probabilities)


def contrastive_yield(probabilities: list[float], k: int) -> float:
    return 1.0 - pass_power(probabilities, k) - fail_power(probabilities, k)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def duality_curve(probabilities: list[float]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_error = 0.0
    for k in range(2, 17):
        pass_k = pass_power(probabilities, k)
        fail_k = fail_power(probabilities, k)
        pass_at_k = 1.0 - fail_k
        zvf = pass_k + fail_k
        scissor_gap = pass_at_k - pass_k
        usable = 1.0 - zvf
        error = abs(scissor_gap - usable)
        max_error = max(max_error, error)
        rows.append(
            {
                "k": k,
                "pass_power_k": round(pass_k, 9),
                "fail_power_k": round(fail_k, 9),
                "pass_at_k": round(pass_at_k, 9),
                "zvf": round(zvf, 9),
                "contrastive_yield": round(usable, 9),
                "scissor_gap": round(scissor_gap, 9),
                "identity_abs_error": error,
                "yield_per_rollout": round(usable / k, 9),
                "yield_over_sqrt_k": round(usable / math.sqrt(k), 9),
            }
        )

    rng = random.Random(SEED)
    best_counts: Counter[int] = Counter()
    g4_minus_g5: list[float] = []
    for _ in range(BOOTSTRAPS):
        sample = [probabilities[rng.randrange(len(probabilities))] for _ in probabilities]
        utilities = {
            k: contrastive_yield(sample, k) / math.sqrt(k) for k in range(2, 17)
        }
        best_counts[max(utilities, key=utilities.get)] += 1
        g4_minus_g5.append(utilities[4] - utilities[5])
    g4_minus_g5.sort()

    summary = {
        "n_prompt_seed_tasks": len(probabilities),
        "mean_p": sum(probabilities) / len(probabilities),
        "identity_max_abs_error": max_error,
        "best_k_yield_per_rollout": max(rows, key=lambda row: row["yield_per_rollout"])["k"],
        "best_k_yield_over_sqrt_k": max(rows, key=lambda row: row["yield_over_sqrt_k"])["k"],
        "bootstrap_best_k_counts": dict(sorted(best_counts.items())),
        "g4_minus_g5_point": next(row["yield_over_sqrt_k"] for row in rows if row["k"] == 4)
        - next(row["yield_over_sqrt_k"] for row in rows if row["k"] == 5),
        "g4_minus_g5_ci95": [g4_minus_g5[124], g4_minus_g5[4_874]],
    }
    return rows, summary


def polarization_audit(probabilities: list[float], k: int = 5) -> list[dict[str, object]]:
    mu = sum(probabilities) / len(probabilities)
    low, high = max(0.0, mu - 0.35), min(1.0, mu + 0.35)
    high_weight = (mu - low) / (high - low)
    polarized = [high] * round(1_000 * high_weight) + [low] * (1_000 - round(1_000 * high_weight))

    rows = []
    for label, sample in (("observed_middle", probabilities), ("equal_mean_polarized", polarized)):
        sample_mu = sum(sample) / len(sample)
        variance = sum((p - sample_mu) ** 2 for p in sample) / len(sample)
        pass_k = pass_power(sample, k)
        fail_k = fail_power(sample, k)
        rows.append(
            {
                "cohort": label,
                "n": len(sample),
                "mean_p": round(sample_mu, 9),
                "variance_p": round(variance, 9),
                "k": k,
                "pass_power_k": round(pass_k, 9),
                "fail_power_k": round(fail_k, 9),
                "zvf": round(pass_k + fail_k, 9),
                "contrastive_yield": round(1.0 - pass_k - fail_k, 9),
            }
        )
    return rows


def controller_asymmetry() -> tuple[list[dict[str, object]], dict[str, object]]:
    source = RESULTS / "p5p8" / "p7_iter203_emp_per_obs.tsv"
    grouped: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    source_rows = 0
    with source.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source_rows += 1
            if row["fires"] != "1":
                continue
            correct = int(row["k"])
            boundary = "all_correct" if correct == 8 else "all_wrong" if correct == 0 else "other"
            values = grouped[(row["method"], boundary)]
            values[0] += 1
            values[1] += float(row["g_for_prompt"])
            values[2] += float(row["gu_for_prompt"])
            values[3] += float(row["cost_ratio"])

    rows: list[dict[str, object]] = []
    for (method, boundary), values in sorted(grouped.items()):
        n, total_g, total_gu, total_cost_ratio = values
        rows.append(
            {
                "method": method,
                "boundary": boundary,
                "n_fires": int(n),
                "mean_chosen_g": round(total_g / n, 6),
                "mean_recovered_contrast": round(total_gu / n, 9),
                "mean_cost_ratio": round(total_cost_ratio / n, 9),
            }
        )

    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "contrast": 0.0})
    for row in rows:
        boundary = str(row["boundary"])
        totals[boundary]["n"] += int(row["n_fires"])
        totals[boundary]["contrast"] += int(row["n_fires"]) * float(row["mean_recovered_contrast"])
    total_fires = sum(v["n"] for v in totals.values())
    summary = {
        boundary: {
            "n_fires": int(values["n"]),
            "share_of_fires": values["n"] / total_fires,
            "mean_recovered_contrast": values["contrast"] / values["n"],
        }
        for boundary, values in sorted(totals.items())
    }
    summary["counterfactual_all_wrong_only"] = {
        "escalation_overhead_removed_fraction": summary["all_correct"]["share_of_fires"],
        "status": "experiment proposal; performance preservation is not established by this audit",
    }
    summary["source_rows"] = source_rows
    summary["fired_rows"] = int(total_fires)
    return rows, summary


def controller_summary_rows(
    detailed_rows: list[dict[str, object]], summary: dict[str, object]
) -> list[dict[str, object]]:
    rows = []
    for boundary in ("all_correct", "all_wrong"):
        selected = [row for row in detailed_rows if row["boundary"] == boundary]
        n = sum(int(row["n_fires"]) for row in selected)
        rows.append(
            {
                "boundary": boundary,
                "n_fires": n,
                "share_of_fires": summary[boundary]["share_of_fires"],
                "mean_recovered_contrast": summary[boundary]["mean_recovered_contrast"],
                "mean_cost_ratio": sum(
                    int(row["n_fires"]) * float(row["mean_cost_ratio"])
                    for row in selected
                )
                / n,
            }
        )
    return rows


def crosscheck_existing_curve(duality_rows: list[dict[str, object]]) -> dict[str, object]:
    source = RESULTS / "berkeley" / "passk_reliability_curve.tsv"
    expected = {int(row["k"]): row for row in duality_rows}
    max_pass_diff = 0.0
    max_pass_at_diff = 0.0
    compared = 0
    with source.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            k = int(row["k"])
            if k not in expected:
                continue
            compared += 1
            max_pass_diff = max(
                max_pass_diff,
                abs(float(row["pass_pow_k"]) - float(expected[k]["pass_power_k"])),
            )
            max_pass_at_diff = max(
                max_pass_at_diff,
                abs(float(row["pass_at_k_bestof"]) - float(expected[k]["pass_at_k"])),
            )
    return {
        "rows_compared": compared,
        "max_abs_pass_power_diff": max_pass_diff,
        "max_abs_pass_at_diff": max_pass_at_diff,
        "tolerance_note": "existing curve is rounded to 5 decimal places",
    }


def main() -> None:
    probabilities, prompt_quality = load_prompt_probabilities()
    duality_rows, duality_summary = duality_curve(probabilities)
    polarization_rows = polarization_audit(probabilities)
    controller_rows, controller_summary = controller_asymmetry()

    write_tsv(OUT / "duality_curve.tsv", duality_rows)
    write_tsv(OUT / "polarization_paradox.tsv", polarization_rows)
    polarization_chart_rows = [
        {
            "cohort": row["cohort"],
            "metric": metric,
            "value": row[field],
            "mean_p": row["mean_p"],
            "variance_p": row["variance_p"],
            "zvf": row["zvf"],
            "contrastive_yield": row["contrastive_yield"],
        }
        for row in polarization_rows
        for metric, field in (
            ("pass^5", "pass_power_k"),
            ("contrastive_yield", "contrastive_yield"),
        )
    ]
    write_tsv(OUT / "polarization_chart.tsv", polarization_chart_rows)
    write_tsv(OUT / "controller_boundary_asymmetry.tsv", controller_rows)
    write_tsv(
        OUT / "controller_boundary_summary.tsv",
        controller_summary_rows(controller_rows, controller_summary),
    )

    summary = {
        "generated_at": "2026-07-13",
        "sources": [
            "platform_hybrid/experiments/results/zvf_iter46_per_prompt_isog.tsv",
            "platform_hybrid/experiments/results/p5p8/p7_iter203_emp_per_obs.tsv",
            "platform_hybrid/experiments/results/berkeley/passk_reliability_curve.tsv",
        ],
        "duality": duality_summary,
        "polarization": polarization_rows,
        "controller": controller_summary,
        "data_quality": {
            "prompt_source": prompt_quality,
            "existing_curve_crosscheck": crosscheck_existing_curve(duality_rows),
            "controller_expected_shape": {
                "expected_rows": 4 * 40 * 16,
                "observed_rows": controller_summary["source_rows"],
                "complete": controller_summary["source_rows"] == 4 * 40 * 16,
            },
        },
        "claim_boundaries": {
            "descriptive": [
                "pass@k - pass^k equals 1 - ZVF algebraically and numerically",
                "G=4 maximizes contrastive_yield/sqrt(G) in the 505-task uncertain-middle cohort",
                "92.3% of empirical controller fires are all-correct observations",
            ],
            "not_established": [
                "causal improvement in held-out accuracy",
                "global G=4 optimality outside the uncertain-middle cohort",
                "performance preservation under an all-wrong-only controller",
            ],
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
