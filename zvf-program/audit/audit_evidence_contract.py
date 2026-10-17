"""Pure shared admission checks. Version 2 rejects invalid scientific evidence.

Legacy completion_sha256 values are retained as opaque row attestations; repeated
completion text is legal. Example identity is the locked split index, not text.
"""

import math
import re

METRICS = ("heldout_score", "last10_reward", "mean_zvf", "mean_gu", "wall_clock_seconds")


def record_errors(record, manifest, *, heldout_n):
    errors = []
    if type(heldout_n) is not int or heldout_n <= 0:
        return ["positive heldout_n required"]
    for field in ("fingerprint", "stack_fingerprint"):
        if re.fullmatch(r"[0-9a-f]{64}", str(record.get(field, ""))) is None:
            errors.append(f"{field} must be a SHA-256 value")
    verification = record.get("remote_verification") or {}
    if (verification.get("wandb") or {}).get("state") != "finished":
        errors.append("remote W&B completion not verified")
    if not isinstance(verification.get("hf_repo"), str) or not verification["hf_repo"]:
        errors.append("remote HF repository missing")
    if re.fullmatch(r"[0-9a-f]{40}", str(verification.get("hf_commit", ""))) is None:
        errors.append("remote exact HF revision missing")
    for field in METRICS:
        value = record.get(field)
        if type(value) not in (int, float) or not math.isfinite(value):
            errors.append(f"{field} must be finite")
        elif field != "wall_clock_seconds" and not 0 <= value <= 1:
            errors.append(f"{field} must be in [0, 1]")
        elif field == "wall_clock_seconds" and value <= 0:
            errors.append("wall_clock_seconds must be positive")
    if not any(e.startswith(("mean_zvf", "mean_gu")) for e in errors):
        if not math.isclose(record["mean_gu"], 1 - record["mean_zvf"], abs_tol=1e-9):
            errors.append("GU must equal 1 - ZVF")
    if type(record.get("rollouts")) is not int or record["rollouts"] <= 0:
        errors.append("rollouts must be positive integer")
    if type(record.get("collapse")) is not bool:
        errors.append("collapse must be boolean")
    if record.get("evidence_class") != "confirmatory":
        errors.append("outer evidence_class must be confirmatory")
    if not isinstance(manifest, dict):
        return errors + ["manifest missing"]
    if manifest.get("evidence_class") != record.get("evidence_class"):
        errors.append("manifest evidence_class mismatch")
    inner = manifest.get("audit_record", {})
    config = manifest.get("run_config", {})
    for field in (
        "arm",
        "seed",
        "heldout_n",
        *METRICS,
        "rollouts",
        "collapse",
        "stack_fingerprint",
    ):
        if field not in record or inner.get(field) != record[field]:
            errors.append(f"manifest audit_record {field} mismatch")
    for outer, inside in (
        ("fingerprint", "unit_fingerprint"),
        ("stack_fingerprint", "stack_fingerprint"),
    ):
        if not record.get(outer) or config.get(inside) != record.get(outer):
            errors.append(f"manifest {inside} mismatch")
    trace = manifest.get("heldout_trace")
    if not isinstance(trace, list) or len(trace) != heldout_n:
        return errors + ["heldout_trace cardinality mismatch"]
    if [r.get("index") if isinstance(r, dict) else None for r in trace] != list(range(heldout_n)):
        errors.append("heldout example identities must be the locked contiguous split indices")
    if any(not isinstance(r, dict) or type(r.get("correct")) is not bool for r in trace):
        errors.append("heldout outcomes must be booleans")
    if any(
        not isinstance(r, dict)
        or not re.fullmatch(r"[0-9a-f]{64}", str(r.get("completion_sha256", "")))
        for r in trace
    ):
        errors.append("heldout_hashes invalid")
    score = record.get("heldout_score")
    if type(score) in (int, float) and math.isfinite(score):
        if not math.isclose(
            sum(r.get("correct") is True for r in trace if isinstance(r, dict)) / heldout_n,
            score,
            abs_tol=1e-12,
        ):
            errors.append("heldout score mismatch")
    return errors
