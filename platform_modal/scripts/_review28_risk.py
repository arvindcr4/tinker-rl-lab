"""Provenance-gated, versioned retrospective risk analysis (no synthetic rows).

Input TSV requires explicit run identities, measured-channel flags, and observed
outcomes. Historical hand-imputed tables are deliberately not accepted as input.
"""

import argparse
import csv
import json
import math
import hashlib
from pathlib import Path
from _review28_statistics import auc

PROTOCOL = "review28_v2"
CHANNELS = ("mean_zvf", "rolling_lag1", "drift_slope")


def measured(row, name):
    return str(row.get(name + "_measured", "")).lower() in ("true", "1")


def score(row):
    """Frozen max of three channels; same function for development and holdout."""
    if not all(measured(row, name) for name in CHANNELS):
        return None
    try:
        z, lag, drift = (float(row[name]) for name in CHANNELS)
    except (KeyError, ValueError, TypeError):
        return None
    if not all(map(math.isfinite, (z, lag, drift))) or not 0 <= z <= 1 or not -1 <= lag <= 1:
        return None
    logistic = lambda x: 1 / (1 + math.exp(-max(-700, min(700, x))))
    return max(
        logistic(6 * (0.30 - z)), logistic(10 * (lag - 0.45)), logistic(200 * (drift - 0.004))
    )


def evaluate(rows, threshold=None, *, calibrate=True):
    eligible = []
    excluded = []
    seen = set()
    for row in rows:
        key = tuple(row.get(k) for k in ("experiment", "method", "seed"))
        if not all(v is not None and str(v) != "" for v in key) or key in seen:
            raise ValueError("Missing or duplicate complete run identity")
        seen.add(key)
        value = score(row)
        observed = str(row.get("outcome_observed", "")).lower() in ("true", "1")
        label = str(row.get("failure", ""))
        if value is None or not observed or label not in ("0", "1"):
            excluded.append(dict(row, exclusion_reason="unmeasured_channel_or_unobserved_outcome"))
            continue
        eligible.append(dict(row, risk=value, failure=int(label), protocol=PROTOCOL))
    if calibrate and threshold is None and {r["failure"] for r in eligible} == {0, 1}:
        candidates = sorted({0.0, 1.0, *(r["risk"] for r in eligible)})

        def balanced(t):
            return (
                sum(
                    sum((r["risk"] >= t) == bool(label) for r in eligible if r["failure"] == label)
                    / sum(r["failure"] == label for r in eligible)
                    for label in (0, 1)
                )
                / 2
            )

        threshold = max(candidates, key=lambda t: (balanced(t), t))
    for row in eligible:
        row["predicted_failure"] = None if threshold is None else int(row["risk"] >= threshold)
    return eligible, excluded, threshold


def main(iteration):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-input", type=Path, required=True)
    parser.add_argument("--heldout-input", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "results" / PROTOCOL,
    )
    args = parser.parse_args()
    read = lambda path: list(csv.DictReader(path.open(), delimiter="\t"))
    development, excluded, threshold = evaluate(read(args.measured_input))
    heldout, heldout_excluded, _ = (
        evaluate(read(args.heldout_input), threshold, calibrate=False)
        if args.heldout_input
        else ([], [], threshold)
    )
    dev_keys = {tuple(r[k] for k in ("experiment", "method", "seed")) for r in development}
    if any(tuple(r[k] for k in ("experiment", "method", "seed")) in dev_keys for r in heldout):
        raise ValueError("Development and holdout run identities overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(
        protocol=PROTOCOL,
        iteration=iteration,
        status="retrospective_development_only",
        input_sha256=hashlib.sha256(args.measured_input.read_bytes()).hexdigest(),
        heldout_sha256=hashlib.sha256(args.heldout_input.read_bytes()).hexdigest()
        if args.heldout_input
        else None,
        input_path=str(args.measured_input.resolve()),
        score_version="measured_three_channel_max_v2",
        grouping_unit="experiment_method_seed",
        observation_cutoff="completed_trajectory_retrospective",
        historical_outputs="superseded: synthetic/mismatched-score inference",
        score_channels=CHANNELS,
        threshold=threshold,
        development=development,
        excluded=excluded,
        heldout=heldout,
        heldout_excluded=heldout_excluded,
        development_auc=auc([r["risk"] for r in development], [r["failure"] for r in development]),
        heldout_auc=auc([r["risk"] for r in heldout], [r["failure"] for r in heldout]),
    )
    (args.output_dir / f"zvf_iter{iteration}_meta.json").write_text(json.dumps(report, indent=2))
