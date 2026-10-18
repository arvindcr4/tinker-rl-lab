#!/usr/bin/env python3
"""NIAH v2: censored retrospective recovery diagnostics and disjoint-window association.

The trajectory analogy is exploratory, not an independent forecasting validation.
H2 uses only observations before the final ten-step target, without overlap.
"""

import json
import csv
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT, RESULTS_DIR
from _review01_contracts import (
    OUTPUT_ROOT,
    evidence_metadata,
    require_inputs,
    spearman_or_none,
    write_artifact_metadata,
    reject_synthetic_or_failed,
)

ROOT = REPO_ROOT
SRC = RESULTS_DIR / "groupsize_zvf_sweep.json"
OUT = OUTPUT_ROOT / "niah"
TOL = 0.05
W_GRID = [1, 2, 3, 5, 8, 13, 21]
W_FIX = 5


def window_mean(seq, p, w):
    if p < 0 or w <= 0 or p + w > len(seq):
        raise ValueError("Window must be completely observed")
    return st.mean(seq[p : p + w])


def earliest_recovery(rewards, target, width, tolerance=TOL):
    return next(
        (
            p
            for p in range(len(rewards) - width + 1)
            if abs(window_mean(rewards, p, width) - target) <= tolerance
        ),
        None,
    )


def recovery_order(pstar):
    values = list(pstar.values())
    complete = bool(values) and all(v is not None for v in values)
    return {
        "recovery_coverage": sum(v is not None for v in values) / len(values) if values else 0,
        "status": "observed" if complete else "INCOMPLETE",
        "monotone_nonincreasing": complete
        and all(b <= a for a, b in zip(values, values[1:], strict=False)),
    }


def disjoint_window(rewards, width, terminal_width=10):
    boundary = len(rewards) - terminal_width
    return window_mean(rewards, boundary - width, width), st.mean(rewards[boundary:])


def analyze(runs):
    if not runs:
        raise ValueError("No trajectories")
    records = []
    for run in runs:
        reject_synthetic_or_failed(run)
        steps = sorted(run["step_log"], key=lambda row: row["step"])
        if len({s["step"] for s in steps}) != len(steps):
            raise ValueError("Duplicate trajectory step")
        rewards = [float(row["mean_reward"]) for row in steps]
        if len(rewards) < max(W_GRID) + 10 or not all(math.isfinite(x) for x in rewards):
            raise ValueError("Finite complete trajectory required for disjoint windows")
        terminal = st.mean(rewards[-10:])
        pstar = {width: earliest_recovery(rewards, terminal, width) for width in W_GRID}
        records.append(
            {
                "G": run["group_size"],
                "seed": run["seed"],
                "rewards": rewards,
                "terminal": terminal,
                "pstar_by_W": pstar,
                "recovery_order": recovery_order(pstar),
            }
        )
    h2 = []
    for width in W_GRID:
        pairs = [disjoint_window(r["rewards"], width) for r in records]
        h2.append(
            {
                "width": width,
                "spearman": spearman_or_none([a for a, _ in pairs], [b for _, b in pairs]),
                "predictor_target_overlap": False,
            }
        )
    recovered = [r for r in records if r["pstar_by_W"][W_FIX] is not None]
    h1, h5 = [], []
    for record in records:
        positions = list(range(len(record["rewards"]) - W_FIX + 1))
        recovery = [
            int(abs(window_mean(record["rewards"], p, W_FIX) - record["terminal"]) <= TOL)
            for p in positions
        ]
        third = max(1, len(positions) // 3)
        downward = sum(b < a for a, b in zip(recovery, recovery[1:], strict=False))
        h1.append(
            {
                "G": record["G"],
                "seed": record["seed"],
                "early_recovery": st.mean(recovery[:third]),
                "late_recovery": st.mean(recovery[-third:]),
                "downward_transitions": downward,
                "monotone_nondecreasing": downward == 0,
            }
        )
        middle = (len(positions) - 1) / 2
        ramp = [p / (len(positions) - 1) for p in positions]
        ushape = [abs(p - middle) / middle for p in positions]

        def residual_sse(predictor, recovery=recovery):
            xmean, ymean = st.mean(predictor), st.mean(recovery)
            denom = sum((x - xmean) ** 2 for x in predictor)
            slope = (
                sum((x - xmean) * (y - ymean) for x, y in zip(predictor, recovery, strict=False))
                / denom
                if denom
                else 0
            )
            intercept = ymean - slope * xmean
            return sum(
                (y - intercept - slope * x) ** 2 for x, y in zip(predictor, recovery, strict=False)
            )

        ramp_sse, u_sse = residual_sse(ramp), residual_sse(ushape)
        h5.append(
            {
                "G": record["G"],
                "seed": record["seed"],
                "sse_position_ramp": ramp_sse,
                "sse_ushape": u_sse,
                "ramp_has_lower_residual": ramp_sse < u_sse,
            }
        )
    grid, pooled_pstar = [], {}
    for width in W_GRID:
        first = None
        for position in range(min(len(r["rewards"]) for r in records) - width + 1):
            rate = st.mean(
                abs(window_mean(r["rewards"], position, width) - r["terminal"]) <= TOL
                for r in records
            )
            grid.append({"width": width, "position": position, "recovery_rate": rate})
            if first is None and rate >= 0.5:
                first = position
        pooled_pstar[width] = first
    return {
        "runs": [{k: v for k, v in r.items() if k != "rewards"} for r in records],
        "H1_position_cliff": h1,
        "H2_disjoint_window_association": h2,
        "H3_pooled_recovery_grid": {
            "grid": grid,
            "pstar_by_W": pooled_pstar,
            **recovery_order(pooled_pstar),
        },
        "H4_recovery_modulation": {
            "spearman_complete_cases": spearman_or_none(
                [math.log2(r["G"]) for r in recovered], [r["pstar_by_W"][W_FIX] for r in recovered]
            ),
            "n_recovered": len(recovered),
            "n_censored": len(records) - len(recovered),
            "status": "INCOMPLETE" if len(recovered) != len(records) else "descriptive",
            "equivalence_established": False,
        },
        "H5_position_vs_ushape_fit": h5,
        "verdict": "EXPLORATORY_ONLY",
    }


def main():
    require_inputs(SRC)
    data = json.loads(SRC.read_text())
    reject_synthetic_or_failed(data)
    result = analyze(data["runs"])
    metadata = evidence_metadata("censored_recovery_and_disjoint_window_rank_association", [SRC])
    result.update(metadata)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "niah_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    outputs = {
        "niah_h1_position_cliff.tsv": result["H1_position_cliff"],
        "niah_h2_effective_window.tsv": result["H2_disjoint_window_association"],
        "niah_h3_grid.tsv": result["H3_pooled_recovery_grid"]["grid"],
        "niah_h3_pstar.tsv": [
            {"width": width, "earliest_recovery": start}
            for width, start in result["H3_pooled_recovery_grid"]["pstar_by_W"].items()
        ],
        "niah_h4_group_modulation.tsv": [
            {"G": row["G"], "seed": row["seed"], "earliest_recovery": row["pstar_by_W"][W_FIX]}
            for row in result["runs"]
        ],
        "niah_h5_middle_falsify.tsv": result["H5_position_vs_ushape_fit"],
    }
    for name, rows in outputs.items():
        with (OUT / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    write_artifact_metadata(OUT, metadata)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
