"""Unexecuted candidate preflight for a future authorized length-regime amendment.

Consumes small measured generation pools, never generates data or allocates compute.
The caller must bind actual model revision/template/decoding/cap and prompt/seed IDs.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence
from .replay import length_cv, ReplayContractError
from .protocol import canonical_fingerprint


def length_feasibility_preflight(
    pools: Sequence[Mapping[str, Any]],
    *,
    generation_identity: Mapping[str, Any],
    minimum_cv: float = 0.35,
    safety_margin: float = 0.05,
) -> dict[str, Any]:
    required = {"model_revision", "prompt_template_sha256", "decoding", "max_completion_length"}
    if not required <= generation_identity.keys() or not pools:
        raise ReplayContractError(
            "complete generation identity and representative measured pools required"
        )
    if not all(math.isfinite(v) and v >= 0 for v in (minimum_cv, safety_margin)):
        raise ReplayContractError("finite nonnegative thresholds required")
    identity = canonical_fingerprint(generation_identity)
    seen = set()
    scores = []
    for pool in pools:
        key = (pool["prompt_id"], pool["seed"])
        if key in seen or pool.get("generation_fingerprint") != identity:
            raise ReplayContractError("duplicate pool or changed generation identity")
        seen.add(key)
        lengths = pool["completion_lengths"]
        if len(lengths) != 16 or any(
            type(x) is not int or not 0 < x <= generation_identity["max_completion_length"]
            for x in lengths
        ):
            raise ReplayContractError("expected sixteen valid measured completion lengths")
        scores.append(max(length_cv(subset) for subset in itertools.combinations(lengths, 6)))
    passed = all(score >= minimum_cv + safety_margin for score in scores)
    return {
        "schema_version": "flagship-length-preflight-candidate-v1",
        "generation_fingerprint": identity,
        "measured_pool_fingerprint": canonical_fingerprint(pools),
        "selected_cv": scores,
        "minimum_cv": minimum_cv,
        "safety_margin": safety_margin,
        "status": "FEASIBLE_CANDIDATE" if passed else "STOP_INFEASIBLE_NO_FULL_ALLOCATION",
        "allocation_authorized": False,
        "limitation": "Representative measured preflight only; an authorized amended protocol must require this before full allocation.",
    }
