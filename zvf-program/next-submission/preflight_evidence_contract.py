"""Pure v2 request/telemetry contracts shared across provider and matrix gates."""

import hashlib
import json
import math
import re

TELEMETRY_VERSION = "generated-and-optimizer-observed-v2"
# Explicit historical semantics: A004 only proves infrastructure; its eligibility
# counter never establishes that an optimizer changed parameters.
HISTORICAL_TELEMETRY = {None: "A004-eligibility-only"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def seal_request(request, *, decoder, tracking, source_files):
    request = dict(request)
    request.update(decoder=decoder, tracking=tracking, contract_version="preflight-evidence-v2")
    scientific = {
        "runtime_packages": request["runtime_packages"],
        "source_files": source_files,
        "decoder": decoder,
        "trainer": "trl-custom-rollout-g8-v2",
    }
    request["scientific_manifest"] = scientific
    request["execution_manifest"] = {
        k: request.get(k) for k in ("provider", "hardware_flavor", "gpu")
    }
    request["stack_fingerprint"] = digest(scientific)
    request.pop("fingerprint", None)
    request["identity_manifest"] = json.loads(json.dumps(request))
    request["fingerprint"] = digest(request["identity_manifest"])
    return request


def validate_identity(request):
    for key, pattern in (("source_commit", r"[0-9a-f]{40}"), ("protocol_sha256", r"[0-9a-f]{64}")):
        if not isinstance(request.get(key), str) or re.fullmatch(pattern, request[key]) is None:
            raise ValueError(f"missing or malformed {key}")
    identity = request.get("identity_manifest")
    scientific = request.get("scientific_manifest")
    if not isinstance(identity, dict) or digest(identity) != request.get("fingerprint"):
        raise ValueError("unit fingerprint cannot be recomputed from retained identity_manifest")
    if any(request.get(k) != v for k, v in identity.items()):
        raise ValueError("receipt differs from retained identity manifest")
    if not isinstance(scientific, dict) or digest(scientific) != request.get("stack_fingerprint"):
        raise ValueError("stack fingerprint cannot be recomputed from scientific_manifest")


def validate_counts(audit, arm, run_config):
    groups, generated = audit.get("rollout_groups"), audit.get("generated_rollouts")
    if type(groups) is not int or groups <= 0 or type(generated) is not int or generated <= 0:
        raise ValueError("positive generated rollouts and groups required")
    fractions = [audit.get(f"{k}_fraction") for k in ("all_wrong", "all_correct", "mixed")]
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in fractions
    ):
        raise ValueError("invalid group fractions")
    if not math.isclose(sum(fractions), 1, abs_tol=1e-9):
        raise ValueError("group fractions do not partition groups")
    counts = [round(v * groups) for v in fractions]
    if any(
        not math.isclose(v * groups, c, abs_tol=1e-9)
        for v, c in zip(fractions, counts, strict=False)
    ):
        raise ValueError("group fractions imply fractional groups")
    expected = 8 * groups if arm == "grpo_g8" else 2 * groups + 6 * counts[2]
    if generated != expected:
        raise ValueError("generated-rollout G2-to-G8/G8 physical accounting mismatch")
    steps = run_config.get("optimizer_steps")
    if type(steps) is not int or steps < 1 or steps > run_config.get("max_steps", 0):
        raise ValueError("observed optimizer steps outside requested horizon")
    if audit.get("telemetry_version") != TELEMETRY_VERSION:
        return False  # explicit historical A004 mapping, never applied-update evidence
    if audit.get("update_eligible_groups") != counts[2]:
        raise ValueError("eligible groups disagree with reward contrast")
    truncated = audit.get("truncated_generated_rollouts")
    if (
        type(truncated) is not int
        or not 0 <= truncated <= generated
        or audit.get("unknown_finish_rollouts") != 0
    ):
        raise ValueError("genuine completion finish accounting missing")
    if not math.isclose(
        audit.get("completion_clipped_fraction", -1), truncated / generated, abs_tol=1e-9
    ):
        raise ValueError("clipping must count genuine generated completions")
    observations = audit.get("update_observations")
    if not isinstance(observations, list):
        raise ValueError("optimizer observations missing")
    seen = set()
    applied = 0
    for o in observations:
        step = o.get("optimizer_step")
        if type(step) is not int or not 1 <= step <= steps or step in seen:
            raise ValueError("invalid or duplicate optimizer observation")
        seen.add(step)
        if o.get("update_applied"):
            norm = o.get("finite_gradient_norm")
            if (
                type(o.get("active_loss_tokens")) is not int
                or o["active_loss_tokens"] <= 0
                or type(norm) not in (int, float)
                or not math.isfinite(norm)
                or norm <= 0
                or o.get("finite_parameters") is not True
                or o.get("parameter_changed") is not True
                or o.get("optimizer_step_observed") is not True
            ):
                raise ValueError(
                    "applied update lacks measured tokens, finite parameters, gradient or parameter change"
                )
            count = o.get("update_eligible_groups")
            if type(count) is not int or count <= 0:
                raise ValueError("applied update has no eligible groups")
            applied += count
    if audit.get("updated_groups") != applied or not 0 <= applied <= counts[2]:
        raise ValueError("applied groups disagree with optimizer observations")
    return applied > 0


def completed_receipt(base, *, result, manifest, verification, cleanup, return_code):
    """Provider adapters must supply observed lifecycle facts, never inferred absence."""
    return {
        **base,
        "status": "completed",
        "payload": result,
        "manifest": manifest,
        "remote_verification": verification,
        "hf_repo": result["remote"]["hf_repo"],
        "return_code": return_code,
        "failed_step": None,
        "allocation_started": True,
        "cleanup": cleanup,
    }


def validate_diagnostic_seed(seed, protocol):
    plan = protocol["paired_seed_plan"]
    reserved = set(plan["seeds"]) | {plan["seed_24_reserve"]}
    if type(seed) is not int or seed <= 0 or seed in reserved:
        raise ValueError(
            "diagnostic seeds must be positive and disjoint from confirmatory/reserve seeds"
        )
