"""Shared conservative contracts for the review-01 candidate analyses (no I/O at import)."""
from pathlib import Path
import hashlib
import json
import math

from _paths import REPO_ROOT, RESULTS_DIR

PROTOCOL = "pro-review-01-v2"
OUTPUT_ROOT = RESULTS_DIR / "protocols" / PROTOCOL


def require_inputs(*paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required analysis inputs: " + ", ".join(missing))


def evidence_metadata(method, paths=(), kind="retrospective_diagnostic"):
    require_inputs(*paths)
    return {"protocol": PROTOCOL, "evidence_type": kind, "analysis_method": method,
            "confirmatory": False, "selection_adjustment": "none; exploratory only",
            "inputs": [{"path": str(Path(p).relative_to(REPO_ROOT)) if Path(p).is_relative_to(REPO_ROOT) else str(p),
                        "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]}


def spearman_or_none(xs, ys):
    from scipy.stats import spearmanr
    if len(xs) != len(ys):
        raise ValueError("Mismatched correlation inputs")
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    if not all(math.isfinite(float(x)) for x in [*xs, *ys]):
        return None
    return float(spearmanr(xs, ys).statistic)


def write_artifact_metadata(directory, metadata):
    """Manifest explicitly applies the evidence type/method to every generated file."""
    directory = Path(directory)
    artifacts = sorted(p.name for p in directory.iterdir() if p.is_file() and p.name != "artifact_manifest.json")
    (directory / "artifact_manifest.json").write_text(json.dumps(
        {"protocol": PROTOCOL, "artifacts": {name: metadata for name in artifacts}}, indent=2) + "\n")


def reject_synthetic_or_failed(record):
    """Legacy artifacts without provenance remain descriptive, never relabeled measured."""
    if not isinstance(record, dict):
        raise ValueError("Trajectory provenance must be a mapping")
    kinds = [str(record.get(key, "")).lower().replace("-", "_") for key in ("data_kind", "evidence_type")]
    mode = str(record.get("mode", "")).lower().replace("-", "_")
    if any(kind == "synthetic" or kind.startswith("synthetic_") for kind in kinds) or mode in {"dryrun", "dry_run"}:
        raise ValueError("Synthetic fixture is not an empirical trajectory")
    if record.get("state") in {"failed", "incomplete", "running", "interrupted", "cancelled", "canceled"}:
        raise ValueError("Incomplete attempt is not an empirical trajectory")


def json_ready(value):
    """Unavailable scalars are JSON null, never nonstandard NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
