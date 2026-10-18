"""Full-history, unit-checked plot inputs with reproducible local sidecars."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from _review02_contracts import result_provenance

METRICS = {
    "train/erf": ("Effective rollout fraction", "fraction"),
    "eval/erf": ("Effective rollout fraction", "fraction"),
    "train/accuracy": ("Training accuracy", "fraction"),
    "eval/accuracy": ("Evaluation accuracy", "fraction"),
    "zvf": ("Zero-variance fraction", "fraction"),
    "zero_variance_frac": ("Zero-variance fraction", "fraction"),
    "train/zvf": ("Zero-variance fraction", "fraction"),
}


def metric_spec(metric, label=None, unit=None):
    if metric in METRICS:
        default_label, default_unit = METRICS[metric]
        return dict(metric=metric, label=label or default_label, unit=unit or default_unit)
    if not label or unit not in ("fraction", "percent", "scalar"):
        raise ValueError(
            "unknown metric requires explicit label and unit (fraction, percent, scalar)"
        )
    return dict(metric=metric, label=label, unit=unit)


def archive_history(run, spec, output, smoothing=1):
    if smoothing < 1:
        raise ValueError("smoothing window must be positive")
    # scan_history is complete retrieval; history(samples=...) is sampled.
    raw = list(run.scan_history())
    rows, omitted = [], 0
    for row in raw:
        if row.get("_step") is None or row.get(spec["metric"]) is None:
            omitted += 1
            continue
        step, value = row["_step"], row[spec["metric"]]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (step, value)
        ):
            raise ValueError("plot rows require finite numeric steps and values")
        if spec["unit"] == "fraction" and not 0 <= value <= 1:
            raise ValueError("fraction outside [0,1]; refusing clipped plot")
        if spec["unit"] == "percent" and not 0 <= value <= 100:
            raise ValueError("percent outside [0,100]")
        rows.append({"step": step, "value": value})
    rows.sort(key=lambda r: r["step"])
    if not rows or len({r["step"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique numeric steps required")
    for i, row in enumerate(rows):
        row["plotted_value"] = sum(
            r["value"] for r in rows[max(0, i - smoothing + 1) : i + 1]
        ) / min(i + 1, smoothing)
    path = Path(str(output) + "." + str(run.id) + ".rows.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    metadata = result_provenance(
        [path],
        run_id=str(run.id),
        run_path=str(getattr(run, "path", "")),
        metric_spec=spec,
        retrieval_time=datetime.now(timezone.utc).isoformat(),
        retrieval_method="scan_history_full",
        n_retrieved=len(raw),
        n_plotted=len(rows),
        omitted_missing_metric_or_step=omitted,
        x_axis="W&B _step, logging step (not necessarily training step)",
        smoothing={"method": "trailing_mean", "window": smoothing},
        estimand="recorded per-log-step metric",
        uncertainty_method="none_descriptive_trajectory",
    )
    Path(str(path) + ".manifest.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return rows, metadata
