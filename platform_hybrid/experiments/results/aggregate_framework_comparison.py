"""Aggregate Task 4 framework-gap results into framework_comparison.json.

Produces ``experiments/results/framework_comparison.json`` with one entry per
framework (Tinker, TRL, verl, OpenRLHF) with this historical nominal config (not imputed into source artifacts):

    Qwen3-8B, seed=42, G=8, lr=1e-5, GSM8K[:500], 30 steps.

Input sources (in priority order per framework):

  Tinker   -> experiments/master_results.json :: campaign_v2_w1_qwen3-8b-base
              (Qwen3-8B-Base, same G/lr/seed/steps; the only Qwen3-8B run with
               the required G=8 + lr=1e-5 identical config)
  TRL      -> experiments/master_results.json :: modal_trl_trl_qwen3_8b
              (TRL-GRPO on Modal H100)
  verl     -> /home/user/workspace/elevation_outputs/modal_verl_grpo.json
              completed empirical artifact required; no training/simulation fallback
  OpenRLHF -> /home/user/workspace/elevation_outputs/modal_openrlhf_grpo.json
              completed empirical artifact required; no training/simulation fallback

Usage:
    python experiments/results/aggregate_framework_comparison.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

MASTER = REPO_ROOT / "platform_hybrid" / "experiments" / "master_results.json"
OUT = (
    REPO_ROOT
    / "platform_hybrid/experiments/results/protocols/pro-review-01-v2/framework_comparison.json"
)
VERL_RESULT = Path("/home/user/workspace/elevation_outputs/modal_verl_grpo.json")
OPENRLHF_RESULT = Path("/home/user/workspace/elevation_outputs/modal_openrlhf_grpo.json")

BASE_CONFIG = {
    "model": "Qwen/Qwen3-8B",
    "seed": 42,
    "group_size": 8,
    "learning_rate": 1e-5,
    "algorithm": "GRPO",
    "task": "gsm8k-500",
    "steps": 30,
}


def _find_experiment(master: Dict[str, Any], experiment_id: str) -> Optional[Dict[str, Any]]:
    for row in master.get("experiments", []):
        if row.get("experiment_id") == experiment_id:
            return row
    return None


def _last10(trace: List[float]) -> float:
    if not trace:
        raise ValueError("Cannot summarize absent reward observations")
    t = trace[-10:] if len(trace) >= 10 else trace
    return float(np.mean(t))


def _coerce(trace: List[Any]) -> List[float]:
    if not trace:
        raise ValueError("Missing reward observations")
    result = [float(value) for value in trace]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Nonfinite reward observations")
    return result


def _validate_measured(payload):
    if payload.get("data_kind") != "empirical" or payload.get("state") != "completed":
        raise ValueError(
            "Only provenance-labeled completed empirical artifacts may be aggregated; legacy artifacts need independent validation"
        )
    if payload.get("mode") == "dryrun" or payload.get("subprocess_exit", 0) != 0:
        raise ValueError("Synthetic or failed attempt cannot enter empirical aggregate")
    _coerce(payload.get("reward_trace"))
    return payload


def _load_master(experiment_id: str, framework: str) -> Dict[str, Any]:
    master = json.loads(MASTER.read_text())
    row = _find_experiment(master, experiment_id)
    if row is None:
        raise ValueError(f"Missing experiment {experiment_id}")
    payload = dict(_validate_measured(row))
    trace = _coerce(payload["reward_trace"])
    payload.update(
        framework=framework,
        peak_reward=max(trace),
        last10_avg=_last10(trace),
        comparison_configuration_status="as_recorded_no_imputation",
    )
    payload.setdefault("mode", "real")
    return payload


def _load_tinker() -> Dict[str, Any]:
    return _load_master("campaign_v2_w1_qwen3-8b-base", "Tinker")


def _load_trl() -> Dict[str, Any]:
    return _load_master("modal_trl_trl_qwen3_8b", "TRL")


def _load_saved(path: Path, framework: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing completed {framework} artifact: {path}; aggregation never launches training"
        )
    payload = dict(_validate_measured(json.loads(path.read_text())))
    trace = _coerce(payload["reward_trace"])
    payload.update(
        framework=framework,
        reward_trace=trace,
        peak_reward=max(trace),
        last10_avg=_last10(trace),
        comparison_configuration_status="as_recorded_no_imputation",
    )
    payload.setdefault("mode", "real")
    return payload


def _load_verl() -> Dict[str, Any]:
    return _load_saved(VERL_RESULT, "verl")


def _load_openrlhf() -> Dict[str, Any]:
    return _load_saved(OPENRLHF_RESULT, "OpenRLHF")


def build() -> Dict[str, Any]:
    tinker = _load_tinker()
    trl = _load_trl()
    verl = _load_verl()
    openrlhf = _load_openrlhf()
    order = [tinker, trl, verl, openrlhf]
    for result in order:
        _validate_measured(result)
    modes = {r["framework"]: r["mode"] for r in order}
    return {
        "protocol": "pro-review-01-v2",
        "evidence_type": "heterogeneous_empirical_description",
        "matched_objective_equivalence": False,
        "task": "task-4-framework-gap",
        "nominal_legacy_config": BASE_CONFIG,
        "configuration_policy": "use each source artifact; no inferred missing settings",
        "metric": "last10_avg (mean of final 10 reported training rewards)",
        "frameworks": order,
        "summary": {
            "Tinker_last10": tinker["last10_avg"],
            "TRL_last10": trl["last10_avg"],
            "verl_last10": verl["last10_avg"],
            "OpenRLHF_last10": openrlhf["last10_avg"],
            "modes": modes,
        },
        "generator": "experiments/results/aggregate_framework_comparison.py",
    }


def main() -> None:
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    for f in out["frameworks"]:
        print(
            f"  {f['framework']:<10} mode={f['mode']:<7} last10={f['last10_avg']:.4f}  peak={f['peak_reward']:.4f}  steps={len(f.get('reward_trace') or [])}"
        )


if __name__ == "__main__":
    main()
