"""Tie-aware finite rank statistics for exploratory observed cohorts."""

import math

import numpy as np
from scipy.stats import kendalltau, rankdata


def _pairs(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("rank statistics require aligned vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("rank statistics require finite pairs")
    return x, y


def rank_correlation(x, y):
    x, y = _pairs(x, y)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def spearman(x, y, b=2000, seed=20260702):
    x, y = _pairs(x, y)
    if isinstance(b, bool) or not isinstance(b, int) or b < 1:
        raise ValueError("positive integer permutation count required")
    obs = rank_correlation(x, y)
    if not math.isfinite(obs):
        return math.nan, math.nan
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    rng = np.random.default_rng(seed)
    count = sum(
        abs(float(rx @ rng.permutation(ry) / denominator)) >= abs(obs) - 1e-12 for _ in range(b)
    )
    return obs, (count + 1) / (b + 1)


def kendall(x, y):
    x, y = _pairs(x, y)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(kendalltau(x, y, variant="b").statistic)


def observed_trace(row, source_root):
    """Load explicitly identified immutable run bytes; never infer from summaries."""
    import hashlib
    import json
    from pathlib import Path

    if not all(row.get(key) for key in ("trace_file", "trace_sha256", "run_id")):
        return None, {
            "status": "unavailable",
            "reason": "missing_explicit_trace_identity",
            "trace_file": "",
            "trace_sha256": "",
            "run_id": "",
        }
    root = Path(source_root).resolve()
    path = (root / row["trace_file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("trace path escapes source root")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row["trace_sha256"]:
        raise ValueError("trace source hash mismatch")
    record = json.loads(raw)
    for metadata in (row, record):
        kind = str(metadata.get("evidence_kind", "")).lower()
        if (
            "synthetic" in kind
            or "reconstruct" in kind
            or any(
                metadata.get(key) in (True, 1, "true", "True", "1")
                for key in ("synthetic", "is_synthetic", "reconstructed")
            )
        ):
            raise ValueError("synthetic or reconstructed traces are not measured evidence")
    if record.get("run_id") != row["run_id"]:
        raise ValueError("trace run identity mismatch")
    values = np.asarray(record.get("reward_trace"), dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("ordered finite reward trace required")
    if (
        isinstance(row["n_steps"], bool)
        or not math.isfinite(float(row["n_steps"]))
        or float(row["n_steps"]) != len(values)
    ):
        raise ValueError("trace length disagrees with cohort row")
    return values, {
        "status": "observed",
        "reason": "",
        "trace_file": str(path.relative_to(root)),
        "trace_sha256": digest,
        "run_id": row["run_id"],
    }


def load_reward_trace(path):
    """Read ordered rewards without inventing zeros or compressing missing steps."""
    import json
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return []
    record = json.loads(path.read_text())
    values = record.get("reward_trace", record.get("rewards"))
    if values is None and "step_log" in record:
        values = [step.get("reward") for step in record["step_log"]]
    if values is None:
        return []
    if not isinstance(values, list) or any(v is None or isinstance(v, bool) for v in values):
        raise ValueError("ordered reward trace cannot contain missing or boolean steps")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ordered reward trace must be finite")
    return values.tolist()


def write_manifest(script_file, output_dir, input_paths, *, seed=20260702):
    """Persist the exact bytes and implementation that produced candidate artifacts."""
    import hashlib
    import importlib.metadata
    import json
    from pathlib import Path
    import subprocess

    script = Path(script_file).resolve()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=script.parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    sources = []
    for path in sorted({Path(p).resolve() for p in input_paths}):
        sources.append(
            dict(
                path=str(path),
                status="observed" if path.is_file() else "unavailable",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            )
        )
    payload = dict(
        analysis_version="review27_v3",
        script=script.name,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        code_revision=revision,
        seed=seed,
        inputs=sources,
        dependencies={
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")
        },
        interpretation="Exploratory candidate analysis; historical artifacts are not replaced",
    )
    target = Path(output_dir) / (script.stem + "_manifest.json")
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return target
