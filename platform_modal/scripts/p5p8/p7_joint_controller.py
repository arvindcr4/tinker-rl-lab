#!/usr/bin/env python3
"""Retrospective P7 joint-policy accounting, with separate cost/outcome axes.

The versioned rule preserves the original step-first policy: a fired step
escalates every prompt to G=16; otherwise contrast prompts use G=2 and
boundary prompts retain G=8. Decisions use the recorded G=8 sample, so the
budget is an oracle/retrospective allocation, not an executable pilot-inclusive
saving. Boundary restoration is a same-sample plug-in prediction for appending
G-8 draws. It is zero at empirical p=0/1, and is never an observed outcome.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import random
from collections import defaultdict
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT

WORK = REPO_ROOT
N2_DIR = WORK / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
OUT_DIR = WORK / "platform_hybrid/experiments/results/p5p8/corrected-review20-v2"
G_BASE, G_ESC, G_DUALFORMER = 8, 16, 2
METHODS = ("grpo", "aero", "areal", "gift")
DDIV_TAU_GRID = (0.03, 0.04, 0.05, 0.06, 0.07)
N_BOOT, SEED = 2000, 20260705
JOINT_RULE_ID = "joint_step_priority_g8_g16_g2_v2"
METRICS = (
    "rollout_saves",
    "extra_rollouts",
    "net_rollouts_saved",
    "predicted_restored_prompts",
    "cost_ratio",
)


def iid_zvf(p_hat: float, G: int) -> float:
    if not 0 <= p_hat <= 1 or G < 1:
        raise ValueError("invalid probability or group size")
    return p_hat**G + (1.0 - p_hat) ** G


def stable_seed(*parts):
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()
    return SEED + int.from_bytes(digest[:4], "big")


def load_n2_with_step_ddiv() -> dict[str, list[dict]]:
    out = {}
    for method in METHODS:
        rows, seen = [], set()
        with (N2_DIR / f"{method}_s0_tensors.jsonl").open() as fh:
            for line in fh:
                rec = json.loads(line)
                step = rec["step"]
                if step in seen:
                    raise ValueError("duplicate method/step")
                seen.add(step)
                groups = rec["rewards"]
                if not groups or any(
                    len(g) != G_BASE or any(v not in (0, 1) for v in g) for g in groups
                ):
                    raise ValueError("expected nonempty binary G=8 prompt groups")
                ks = [sum(g) for g in groups]
                observed = sum(k in (0, G_BASE) for k in ks) / len(ks)
                predicted = sum(iid_zvf(k / G_BASE, G_BASE) for k in ks) / len(ks)
                for prompt_idx, k in enumerate(ks):
                    rows.append(
                        {
                            "method": method,
                            "step": step,
                            "prompt_idx": prompt_idx,
                            "K": k,
                            "p_hat": k / G_BASE,
                            "zvf_actual": float(k in (0, G_BASE)),
                            "contrast_prompt": 0 < k < G_BASE,
                            "boundary_prompt": k in (0, G_BASE),
                            "zvf_iid_g8": iid_zvf(k / G_BASE, G_BASE),
                            "zvf_iid_g16": iid_zvf(k / G_BASE, G_ESC),
                            "step_zvf_iid": predicted,
                            "step_zvf_obs": observed,
                            "step_delta_div": predicted - observed,
                        }
                    )
        out[method] = rows
    return out


def controller_prompt_action(record, tau, policy="joint"):
    """One shared step-first action rule; no action implies restoration."""
    if policy not in ("joint", "dualformer", "ddiv"):
        raise ValueError("unknown controller")
    boundary = bool(record["zvf_actual"])
    fired = policy != "dualformer" and record["step_delta_div"] >= tau
    group_size = G_ESC if fired else (G_DUALFORMER if not boundary and policy != "ddiv" else G_BASE)
    restoration_probability = 0.0
    if boundary and group_size > G_BASE:
        # Append experiment: an all-failure group needs a success and vice versa.
        p = record["p_hat"]
        additional = group_size - G_BASE
        restoration_probability = (
            1 - (1 - p) ** additional if record["K"] == 0 else 1 - p**additional
        )
    return {
        "applied_G": group_size,
        "fired": fired,
        "predicted_restoration_probability": restoration_probability,
    }


def _account(records, tau, policy):
    total, saved, extra, boundary_fires, predicted, deescalated = 0, 0, 0, 0, 0.0, 0
    fired_steps, fired_prompts = set(), 0
    for record in records:
        action = controller_prompt_action(record, tau, policy)
        group_size = action["applied_G"]
        total += group_size
        saved += max(0, G_BASE - group_size)
        extra += max(0, group_size - G_BASE)
        deescalated += group_size < G_BASE
        predicted += action["predicted_restoration_probability"]
        if action["fired"]:
            fired_steps.add((record.get("method"), record["step"]))
            fired_prompts += 1
            boundary_fires += bool(record["zvf_actual"])
    baseline = len(records) * G_BASE
    return {
        "rule_id": JOINT_RULE_ID if policy == "joint" else f"{policy}_g8_v2",
        "tau": tau,
        "n_contrast": deescalated,
        "n_fired_steps": len(fired_steps),
        "n_fired_prompts": fired_prompts,
        "n_boundary_escalated": boundary_fires,
        "predicted_restored_prompts": predicted,
        "observed_restored_prompts": None,
        "restoration_status": "same_sample_plugin_prediction_only",
        "budget_status": "retrospective_allocation_excludes_pilot_cost",
        "rollout_saves": saved,
        "extra_rollouts": extra,
        "net_rollouts_saved": saved - extra,
        "g_total": total,
        "baseline_rollouts": baseline,
        "cost_ratio": total / baseline if baseline else None,
    }


def joint_controller(records: list[dict], tau: float) -> dict:
    return _account(records, tau, "joint")


def dualformer_only(records: list[dict]) -> dict:
    return _account(records, None, "dualformer")


def ddiv_only(records: list[dict], tau: float) -> dict:
    return _account(records, tau, "ddiv")


def bootstrap_controller_metrics(records, tau, B=N_BOOT, seed=SEED):
    """Paired step-cluster bootstrap, conditional on recorded decision signals.

    Prompts in a step share the trigger and stay together in each draw. All
    cost and modeled-outcome metrics use exactly the same sampled clusters.
    """
    if B < 1:
        raise ValueError("B must be positive")
    by_step = defaultdict(list)
    for record in records:
        by_step[(record.get("method"), record["step"])].append(record)
    clusters = list(by_step.values())
    if not clusters:
        return {metric: {"pt": None, "lo": None, "hi": None} for metric in METRICS}
    point = joint_controller(records, tau)
    # Aggregate each whole cluster once; selected clusters contribute repeatedly.
    contributions = [joint_controller(cluster, tau) for cluster in clusters]
    rng = random.Random(seed)
    values = {metric: [] for metric in METRICS}
    for _ in range(B):
        sample = [contributions[rng.randrange(len(clusters))] for _ in clusters]
        for metric in METRICS:
            value = (
                sum(row["g_total"] for row in sample)
                / sum(row["baseline_rollouts"] for row in sample)
                if metric == "cost_ratio"
                else sum(row[metric] for row in sample)
            )
            values[metric].append(value)
    return {
        metric: {
            "pt": point[metric],
            "lo": sorted(values[metric])[int(B * 0.025)],
            "hi": sorted(values[metric])[min(B - 1, int(B * 0.975))],
        }
        for metric in METRICS
    }


def bootstrap_ci_per_method_metric(records, tau, B, seed):
    """Legacy entry point, now explicitly measured in net rollouts only."""
    return {
        **bootstrap_controller_metrics(records, tau, B, seed)["net_rollouts_saved"],
        "metric": "net_rollouts_saved",
        "unit": "rollouts",
    }


def main():
    n2 = load_n2_with_step_ddiv()
    joint_rows, comparison_rows = [], []
    for method in METHODS:
        for tau in DDIV_TAU_GRID:
            row = {"method": method, **joint_controller(n2[method], tau)}
            ci = bootstrap_controller_metrics(n2[method], tau, N_BOOT, stable_seed(method, tau))
            for metric in METRICS:
                row[f"{metric}_lo"], row[f"{metric}_hi"] = ci[metric]["lo"], ci[metric]["hi"]
            joint_rows.append(row)
        for name, result in (
            ("joint", joint_controller(n2[method], 0.05)),
            ("dualformer", dualformer_only(n2[method])),
            ("ddiv", ddiv_only(n2[method], 0.05)),
        ):
            comparison_rows.append({"method": method, "controller": name, **result})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("p7_joint_controller.tsv", joint_rows),
        ("p7_joint_controller_boot.tsv", comparison_rows),
    ):
        with (OUT_DIR / filename).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "rule_id": JOINT_RULE_ID,
        "bootstrap_unit": "step_cluster",
        "n_boot": N_BOOT,
        "metric_units": {
            "net_rollouts_saved": "rollouts",
            "predicted_restored_prompts": "expected initially-boundary prompts",
        },
        "observed_restoration_status": "unavailable_no_intervention_outcomes",
        "comparisons": comparison_rows,
        "tau_sweep": joint_rows,
    }
    with (OUT_DIR / "p7_joint_controller_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
    print(f"Joint policy {JOINT_RULE_ID}: wrote separate rollout/outcome metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()
