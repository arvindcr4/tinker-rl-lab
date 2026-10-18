"""Pure helpers for candidate publication analysis v2 (no artifact I/O)."""

import collections
import itertools
import json
import math
import random
import statistics
from collections.abc import Sequence


def h(probability, group_size=8):
    return probability**group_size + (1 - probability) ** group_size


def mean_preserving_points(mean, maximum_radius=0.3):
    if not 0 <= mean <= 1 or not 0 <= maximum_radius <= 1:
        raise ValueError("probability/radius outside range")
    radius = min(maximum_radius, mean, 1 - mean)
    return mean - radius, mean + radius


def empirical_delta_expectation(probabilities, group_size):
    """Exact finite-sample diagnostic, including the observed-mean plug-in bias.

    Independent Bernoulli draws within and across prompts are an assumption.
    Enumeration is limited to one million count configurations.
    """
    from scipy.stats import binom

    if (
        type(group_size) is not int
        or group_size < 2
        or not probabilities
        or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
    ):
        raise ValueError("valid probabilities and integer group size >= 2 required")
    if (group_size + 1) ** len(probabilities) > 1_000_000:
        raise ValueError("exact enumeration limit exceeded")
    m = len(probabilities)
    expectation = 0.0
    for counts in itertools.product(range(group_size + 1), repeat=m):
        mass = math.prod(
            binom.pmf(k, group_size, p) for k, p in zip(counts, probabilities, strict=False)
        )
        residual = (
            h(sum(counts) / (m * group_size), group_size)
            - sum(k in (0, group_size) for k in counts) / m
        )
        expectation += mass * residual
    latent = h(statistics.mean(probabilities), group_size) - statistics.mean(
        h(p, group_size) for p in probabilities
    )
    return {
        "empirical_plugin_expectation": float(expectation),
        "latent_probability_delta": latent,
        "sampling_plugin_bias": float(expectation - latent),
        "sign_guarantee": False,
    }


def trajectory_summary(rows, samples=5000, seed=0):
    """Cluster bootstrap by seed, retaining every step and G within that seed.

    Reports a row-weighted observed mixture and separate group-size estimates.
    No population mechanism or platform transport is identified by this CI.
    """
    if not rows:
        raise ValueError("empty arithmetic population")
    clusters = collections.defaultdict(list)
    identities = set()
    for row in rows:
        key = (str(row["seed"]), int(row["G"]), str(row["id"]))
        if key in identities:
            raise ValueError("duplicate seed/group/step identity")
        identities.add(key)
        value = float(row["delta_div"])
        if not math.isfinite(value):
            raise ValueError("nonfinite residual")
        clusters[key[0]].append(value)
    if type(samples) is not int or samples < 1:
        raise ValueError("positive bootstrap sample count required")
    values = [value for trajectory in clusters.values() for value in trajectory]
    keys = sorted(clusters)
    result = {
        "n_rows": len(values),
        "n_independent_seed_clusters": len(keys),
        "seed_ids": keys,
        "mean": statistics.mean(values),
        "ci": None,
        "estimand": "row-weighted observed trajectory/group-size mixture",
        "uncertainty_unit": "whole seed trajectories; repeated steps never independent",
        "status": "insufficient_independent_clusters"
        if len(keys) < 2
        else "descriptive_cluster_bootstrap",
    }
    if len(keys) >= 2:
        import numpy as np

        rng = random.Random(seed)
        draws = [
            statistics.mean(v for key in rng.choices(keys, k=len(keys)) for v in clusters[key])
            for _ in range(samples)
        ]
        result["ci"] = np.percentile(draws, [2.5, 97.5]).tolist()
    return result


def robust_incremental(data, reduced, full, restriction="r = 0, r2 = 0"):
    """HC3 joint Wald F on the identical analyzed row set in both models."""
    import numpy as np

    # Optional analysis dependency has no distributed stubs/py.typed marker.
    import statsmodels.formula.api as smf

    full_model = smf.ols(full, data=data, missing="drop")
    cohort = data.loc[full_model.data.row_labels].copy()
    f4 = smf.ols(full, data=cohort, missing="raise").fit(cov_type="HC3")
    f3 = smf.ols(reduced, data=cohort, missing="raise").fit(cov_type="HC3")
    test = f4.wald_test(restriction, use_f=True, scalar=True)
    result = {
        "F": float(test.statistic),
        "p": float(test.pvalue),
        "delta_r2": float(f4.rsquared - f3.rsquared),
        "covariance_type": f4.cov_type,
        "restriction": restriction,
        "n": int(f4.nobs),
        "df_num": float(test.df_num),
        "df_denom": float(test.df_denom),
        "cohort_row_ids": [str(i) for i in cohort.index],
        "run_ids": cohort["run"].tolist() if "run" in cohort else None,
        "design_rank": int(np.linalg.matrix_rank(f4.model.exog)),
        "design_columns": len(f4.params),
    }
    if not all(math.isfinite(result[k]) for k in ("F", "p", "df_num", "df_denom")):
        raise ValueError("undefined robust incremental test")
    return result


def equivalence_classes(records, query) -> list[list[str]]:
    """Identical normalized signatures INCLUDING known/unknown states.

    Pairwise R0 compatibility is not transitive and is never unioned here.
    Delta declarations are included conservatively in the immutable signature.
    """
    classes: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
    for identity, record in sorted(records.items()):
        fields = query.field_view(record)
        signature: dict[str, object] = {
            field: {
                "known": query.is_known(cell),
                "value": query._key(field, cell["value"]) if query.is_known(cell) else None,
            }
            for field, cell in sorted(fields.items())
        }
        signature["variant_deltas_applied"] = sorted(
            record.get("variant_deltas_applied", []),
            key=lambda row: json.dumps(row, sort_keys=True),
        )
        classes[
            json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ].append(identity)
    return sorted(classes.values(), key=lambda members: members[0])


def coverage_median(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("nonempty finite coverage population required")
    return statistics.median(values)
