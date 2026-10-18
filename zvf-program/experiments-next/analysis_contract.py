"""Prospective pool-analysis-v2 contracts (stdlib only, offline-testable)."""

import hashlib
import json
import math
import statistics

VERSION = "pool-analysis-v2"
IDENTITY = (
    "model_revision",
    "checkpoint_identity",
    "dataset_revision",
    "prompt_manifest_sha256",
    "reward_parser_sha256",
    "tokenizer_revision",
    "sampler_backend",
    "sampling_config_sha256",
    "prompt_selection_policy",
)
ZVF_DEFINITION = {
    "version": "binary-exact-v1",
    "reward_domain": [0, 1],
    "variance": "population",
    "epsilon": 0,
    "singleton": "rejected",
}
T1_DECISION = {
    "primary_interval": "wilson",
    "primary_batching_design": "shuffled",
    "minimum_batch_size": 64,
    "curriculum_global_inference": "blocked_pending_validation",
    "evidence_tier": "exploratory_empirical",
    "controller_readiness_established": False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_rewards(rewards, *, binary=True):
    if (
        not isinstance(rewards, list)
        or len(rewards) < 2
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or (binary and x not in (0, 1))
            for x in rewards
        )
    ):
        raise ValueError(
            "rewards require at least two finite numbers in the registered binary domain"
        )
    return rewards


def zero_variance(rewards):
    validate_rewards(rewards)
    return min(rewards) == max(rewards)


def validate_design(*, groups, counts=(), available=None, n_prompts=None):
    if not groups or any(isinstance(g, bool) or not isinstance(g, int) or g < 2 for g in groups):
        raise ValueError("group sizes must be integers >= 2")
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in counts):
        raise ValueError("batch/trial/draw/stratum counts must be positive integers")
    if available is not None and not any(g <= available for g in groups):
        raise ValueError("no feasible group size in requested grid")
    if n_prompts is not None and n_prompts < 1:
        raise ValueError("nonempty prompt population required")


def completeness(pool):
    actual = len(pool.get("prompts", []))
    planned = pool.get("planned_prompts", pool.get("n_prompts"))
    state = pool.get("sampling_completeness", "legacy_unverified")
    if pool.get("truncated"):
        state = "truncated"
    if pool.get("status") != "complete" or actual != planned:
        state = "truncated" if pool.get("truncated") else "shortfall"
    return {
        "sampling_completeness": state,
        "planned_prompts": planned,
        "actual_prompts": actual,
        "exploratory_only": state != "full" or bool(pool.get("exploratory_only")),
    }


def validate_full_pool(pool):
    info = completeness(pool)
    if info["exploratory_only"]:
        raise ValueError(f"full sampling contract required; got {info}")
    if not pool.get("prompts"):
        raise ValueError("full pool requires nonempty prompts")
    r = pool.get("rollouts_per_prompt")
    if not isinstance(r, int) or isinstance(r, bool) or r < 2:
        raise ValueError("rollouts_per_prompt must be an integer >= 2")
    ids = [row.get("idx") for row in pool.get("prompts", [])]
    if ids != list(range(len(ids))):
        raise ValueError("prompt IDs must be a contiguous unique prefix")
    for row in pool.get("prompts", []):
        validate_rewards(row.get("rewards"))
        ph = row.get("p_hat")
        if (
            isinstance(ph, bool)
            or not isinstance(ph, (int, float))
            or not math.isfinite(ph)
            or not math.isclose(ph, statistics.mean(row["rewards"]), abs_tol=1e-12)
        ):
            raise ValueError("p_hat must equal the observed binary reward mean")
        if len(row["rewards"]) != r:
            raise ValueError("rollout count differs from plan")
        tokens = row.get("token_counts")
        if (
            not isinstance(tokens, list)
            or len(tokens) != r
            or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens)
        ):
            raise ValueError("token counts must match rewards and be nonnegative integers")
    return info


def validate_identity(contract):
    if not isinstance(contract, dict) or contract.get("version") != VERSION:
        raise ValueError("missing versioned analysis contract")
    for key in IDENTITY:
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ValueError(f"missing identity field {key}")
    return contract


def validate_resume(previous, expected):
    for key, value in expected.items():
        if key not in previous or previous[key] != value:
            raise ValueError(f"resume refused: {key} differs or is missing")
    validate_identity(previous.get("analysis_contract"))
    if not previous.get("sampler_path"):
        raise ValueError("resume refused: resolved frozen sampler identity missing")


def strata_all(items, count):
    if count < 1 or count > len(items):
        raise ValueError("strata count must be between 1 and population size")
    return [items[i * len(items) // count : (i + 1) * len(items) // count] for i in range(count)]


def stratified_interval(values, weights):
    """Conservative weighted Hoeffding CI; valid with one independent draw/stratum.

    Avoids pretending a single observation estimates within-stratum variance.
    """
    if len(values) != len(weights) or not values or not math.isclose(sum(weights), 1):
        raise ValueError("normalized matching stratum weights required")
    estimate = sum(x * w for x, w in zip(values, weights, strict=False))
    half = math.sqrt(math.log(40) * sum(w * w for w in weights) / 2)
    return max(0, estimate - half), min(1, estimate + half)
