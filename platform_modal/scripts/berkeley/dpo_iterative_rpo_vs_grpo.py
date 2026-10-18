"""DPO/GRPO v2: local gradient relationship and explicitly unmeasured IRPO hypotheses.

A shared winner/loser construction does not imply objective, trajectory or
optimal-group-size equivalence. These are retrospective/theoretical diagnostics.
"""

import json
import csv
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _analysis_common import read_tsv_lines as _read_tsv

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT, RESULTS_DIR
from _review01_contracts import (
    OUTPUT_ROOT,
    evidence_metadata,
    require_inputs,
    write_artifact_metadata,
    json_ready,
)

ROOT = WORKTREE = REPO_ROOT
RES = RESULTS = RESULTS_DIR
OUT = OUTPUT_ROOT / "dpo_iterative_rpo_vs_grpo"


# Input TSVs
ZVFLINK = RESULTS / "group_size_iter115_zvf_linkage.tsv"
JOINTFIT = RESULTS / "group_size_iter127_joint_fit.tsv"
OPTG = RESULTS / "group_size_iter127_optimal_g.tsv"
ISORWD = RESULTS / "group_size_iter123_iso_reward.tsv"
NOISE = RESULTS / "group_size_iter123_noise_mech.tsv"
EFFECT = RESULTS / "group_size_iter123_effect_size.tsv"
BOUNDED = RESULTS / "group_size_iter127_bounded_cone.tsv"
COMP = RESULTS / "group_size_iter127_complementarity.tsv"


def _parse_float(s):
    """Tolerant numeric parse: strip quotes, commas, handle +/-."""
    s = s.strip().strip('"').replace(",", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_zvflink():
    rows = _read_tsv(ZVFLINK)
    out = []
    for r in rows[1:]:
        if len(r) < 7:
            continue
        try:
            out.append(
                {
                    "T": int(r[0]),
                    "acc_G4": _parse_float(r[1]),
                    "acc_G32": _parse_float(r[2]),
                    "retention": _parse_float(r[3]),
                    "GU_G4": _parse_float(r[4]),
                    "GU_G32": _parse_float(r[5]),
                    "GU_ratio": _parse_float(r[6]),
                }
            )
        except (ValueError, IndexError):
            continue
    return out


def load_iso_reward():
    """Parse iter123 T=64M acc-by-G dict."""
    rows = _read_tsv(ISORWD)
    out = {}
    for r in rows[1:]:
        if len(r) < 3 or r[0] != "iso_reward_setup":
            continue
        # headline looks like: {"G=4": 0.64, "G=8": 0.8, ...}
        h = r[2].strip().strip('"').strip("{").strip("}").replace('"', "")
        for kv in h.split(","):
            k, v = kv.split(":")
            out[int(k.strip().split("=")[1])] = float(v)
    return out


def load_snr_scaling():
    rows = _read_tsv(NOISE)
    out = {
        "n_G": None,
        "slope": None,
        "ci_lo": None,
        "ci_hi": None,
        "R2": None,
        "p": None,
        "theoretical": None,
        "at_G": {},
    }
    for r in rows[1:]:
        if len(r) < 3:
            continue
        h = r[2]
        if r[1] == "n_G":
            # headline is like 'SNR-at-G pool: n=4 G-values: [2, 4, 8, 16]'
            import re

            m = re.search(r"n=(\d+)", h)
            out["n_G"] = int(m.group(1)) if m else None
        elif r[1] == "ols_log10_snr_vs_log10_G":
            # headline: 'OLS log10(SNR) ~ log10(G): slope=+0.366/decade  95%CI [+0.148,+0.583]  R^2=0.844  p=8.11e-02'
            import re

            m = re.search(r"slope=([+-]?\d+\.\d+)", h)
            out["slope"] = float(m.group(1)) if m else None
            m = re.search(r"95%CI\s*\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]", h)
            if m:
                out["ci_lo"] = float(m.group(1))
                out["ci_hi"] = float(m.group(2))
            m = re.search(r"R\^2=([\d.]+)", h)
            out["R2"] = float(m.group(1)) if m else None
            m = re.search(r"p=([\d.eE+-]+)", h)
            out["p"] = float(m.group(1)) if m else None
        elif r[1] == "theoretical_pred_+0.5":
            # headline like 'Theoretical SNR~G^0.5 predicts slope=+0.500/decade; ...'
            import re

            m = re.search(r"slope=([+-]?\d+\.\d+)", h)
            out["theoretical"] = float(m.group(1)) if m else None
        elif r[1] == "pred_at_G":
            # G=2=0.043/0.051, G=4=0.060/0.072, G=8=0.094/0.102, G=16=0.087/0.144
            for kv in h.split(","):
                kv = kv.strip()
                if not kv.startswith("G="):
                    continue
                g_str, rest = kv.split("=", 1)[1].split("=", 1)
                pred, emp = rest.split("/")
                out["at_G"][int(g_str)] = (float(pred), float(emp))
    return out


def load_joint_fit():
    rows = _read_tsv(JOINTFIT)
    out = {"params": {}, "cells": []}
    for r in rows[1:]:
        section, key, headline = r[0], r[1], r[2]
        if not key.startswith("row_"):
            out["params"][key] = headline
    return out


def load_optimal_g():
    rows = _read_tsv(OPTG)
    out = {}
    for r in rows[1:]:
        if len(r) < 3 or not r[1].startswith("T="):
            continue
        try:
            T = int(r[1].split("=")[1])
            # headline like 'T=1e+06: G*(T)=8, acc(G*)=0.480, G*(T)_pred=8.0'
            import re

            m = re.search(r"G\*\(T\)=(\d+)", r[2])
            if m:
                out[T] = int(m.group(1))
        except (ValueError, IndexError):
            continue
    return out


def load_bounded():
    rows = _read_tsv(BOUNDED)
    out = {"supported": None, "cells": []}
    for r in rows[1:]:
        if len(r) < 3:
            continue
        if r[1] == "supported":
            out["supported"] = r[2].strip()
        elif r[1].startswith("T="):
            parts = r[2].split(",")
            T = int(r[1].split("=")[1])
            try:
                acc32 = float(parts[0].split("=")[1])
                acc64 = float(parts[1].split("=")[1])
                delta = float(parts[2].split("=")[1])
                out["cells"].append({"T": T, "acc_G32": acc32, "acc_G64": acc64, "delta": delta})
            except (ValueError, IndexError):
                continue
    return out


def dpo_implicit_reward_equivalence():
    """Report observed contrast yield; ratios cannot establish usable pair counts."""
    return [
        {
            **row,
            "iterative_rpo_feasible_at_G4": None,
            "iterative_rpo_feasible_at_G32": None,
            "status": "unmeasured_hypothesis",
            "note": "Actual usable-group or winner-loser-pair counts are required; yield ratio is not feasibility",
        }
        for row in load_zvflink()
    ]


def snr_scaling_validation():
    """Reported GRPO slope diagnostics, with no inference about unmeasured IRPO."""
    source = load_snr_scaling()
    available = all(source[key] is not None for key in ["ci_lo", "ci_hi", "theoretical"])
    return {
        **source,
        "slope_in_ci_of_theory": source["ci_lo"] <= source["theoretical"] <= source["ci_hi"]
        if available
        else None,
        "interpretation": "Descriptive interval containment is not objective or algorithm equivalence",
    }


def write_tsv(path, rows, header):
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def irpo_optimal_g_vs_iter127():
    return [
        {
            "T": budget,
            "GRPO_optimal_G": size,
            "Iterative_RPO_optimal_G": None,
            "status": "unmeasured_hypothesis",
            "rationale": "No IRPO training comparison was performed",
        }
        for budget, size in load_optimal_g().items()
    ]


def dpo_pair_gradient_scale(beta, log_ratio_difference):
    """Scale multiplying gradient of the unaveraged pair loss -D, frozen ref."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = beta * log_ratio_difference
    return beta * (math.exp(-z) / (1 + math.exp(-z)) if z >= 0 else 1 / (1 + math.exp(z)))


def dpo_loss_vs_grpo_loss():
    return [
        {
            "setting": "one winner-loser pair; frozen reference snapshot equals current policy",
            "GRPO_loss_form": "-D (unaveraged pair, population-SD advantages +1,-1)",
            "DPO_loss_form": "-log sigmoid(beta * (D-D_ref))",
            "gradient_scale_at_snapshot": "beta/2 relative to -D",
            "general_gradient_scale": "beta * sigmoid(-beta*(D-D_ref))",
            "match": "Aligned local update directions only, after scale and loss normalization",
            "interpretation": "No trajectory or optimal group-size equivalence. Differentiably tied reference gives zero gradient. Unscaled gradient tends to zero as beta tends to zero.",
        }
    ]


def main():
    inputs = [ZVFLINK, OPTG, NOISE]
    require_inputs(*inputs)
    summary = {
        **evidence_metadata(
            "frozen_reference_pair_gradient_identity_and_unmeasured_IRPO_hypotheses", inputs
        ),
        "contrast_yield": dpo_implicit_reward_equivalence(),
        "optimal_group_sizes": irpo_optimal_g_vs_iter127(),
        "loss_relationship": dpo_loss_vs_grpo_loss(),
        "grpo_snr_source": snr_scaling_validation(),
        "headline_claim": "A frozen equal reference aligns one-pair local directions up to beta/2 scaling; objective and trajectory equivalence are not established",
        "IRPO_comparison_status": "unmeasured_hypothesis",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dpo_iterative_rpo_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, allow_nan=False) + "\n"
    )
    for name, rows in [
        ("dpo_iterative_rpo_grpo_equivalence.tsv", summary["contrast_yield"]),
        ("dpo_iterative_rpo_snr_scaling.tsv", [summary["grpo_snr_source"]]),
        ("dpo_iterative_rpo_optimal_g.tsv", summary["optimal_group_sizes"]),
        ("dpo_iterative_rpo_loss_equivalence.tsv", summary["loss_relationship"]),
    ]:
        write_tsv(OUT / name, rows, list(rows[0]) if rows else ["status"])
    write_artifact_metadata(
        OUT,
        {
            key: summary[key]
            for key in ["protocol", "evidence_type", "analysis_method", "inputs", "confirmatory"]
        },
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
