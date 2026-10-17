#!/usr/bin/env python3
"""StateFlow v2: retrospective trajectory descriptions, not stopped-policy evaluation.

Terminal-reward-normalized states use future information. Suffix rewards cannot
measure a checkpoint frozen at an earlier step. Retention remains unavailable
until common held-out checkpoint evaluations and an online rule are supplied.
"""
import json
import math
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT, RESULTS_DIR
from _review01_contracts import OUTPUT_ROOT, evidence_metadata, require_inputs, write_artifact_metadata, reject_synthetic_or_failed, spearman_or_none

ROOT = REPO_ROOT
SRC = RESULTS_DIR / "groupsize_zvf_sweep.json"
OUT = OUTPUT_ROOT / "stateflow"
THETA_LO, THETA_HI = 0.5, 0.9


def dp_3seg(signal):
    """Least-squares three-segment retrospective fit; unavailable if fewer than 3 points."""
    if len(signal) < 3:
        return None, None
    prefix, prefix2 = [0.], [0.]
    for value in signal:
        prefix.append(prefix[-1]+value)
        prefix2.append(prefix2[-1]+value*value)
    def cost(a, b):
        total = prefix[b]-prefix[a]
        return prefix2[b]-prefix2[a]-total*total/(b-a)
    best = min((cost(0, a)+cost(a, b)+cost(b, len(signal)), a, b)
               for a in range(1, len(signal)-1) for b in range(a+1, len(signal)))
    return best[1:]


def _standardize(values):
    mean = st.mean(values)
    sd = st.pstdev(values) or 1.
    return [(value-mean)/sd for value in values]


def fsm_states(rew, R_T):
    labels = []
    current = 0
    for reward in rew:
        fraction = reward / R_T if R_T > 0 else 0
        if current < 1 and fraction >= THETA_LO:
            current = 1
        if current < 2 and fraction >= THETA_HI:
            current = 2
        labels.append(current)
    return labels, next((i for i, x in enumerate(labels) if x >= 1), None), next((i for i, x in enumerate(labels) if x >= 2), None)


def state_ordering(labels, rewards, entropy):
    if not (len(labels) == len(rewards) == len(entropy)):
        raise ValueError("State and observation lengths differ")
    if set(labels) != {0, 1, 2} or any(b < a or b > a + 1 for a, b in zip(labels, labels[1:])):
        return {"status": "INCOMPLETE", "reward_increases": False, "entropy_decreases": False}
    rew = [st.mean(r for r, label in zip(rewards, labels) if label == state) for state in range(3)]
    ent = [st.mean(e for e, label in zip(entropy, labels) if label == state) for state in range(3)]
    return {"status": "observed", "reward_increases": all(a < b for a, b in zip(rew, rew[1:])),
            "entropy_decreases": all(a > b for a, b in zip(ent, ent[1:]))}


def checkpoint_retention(scores, stop_step, final_step):
    """Only independently evaluated held-out checkpoints can define retention."""
    if scores is None or stop_step not in scores or final_step not in scores:
        return None
    if scores[final_step] <= 0:
        return None
    return scores[stop_step] / scores[final_step]


def analyze_run(run):
    reject_synthetic_or_failed(run)
    steps = sorted(run["step_log"], key=lambda s: s["step"])
    if not steps or len({s["step"] for s in steps}) != len(steps):
        raise ValueError("Empty or duplicate step trajectory")
    rewards = [s["mean_reward"] for s in steps]
    entropy = [s["entropy"] for s in steps]
    if not all(math.isfinite(float(value)) for value in rewards+entropy+[s["advantage_variance"] for s in steps]):
        raise ValueError("Nonfinite trajectory observation")
    terminal = st.mean(rewards[-10:])
    labels, b1, b2 = fsm_states(rewards, terminal)
    c1, c2 = dp_3seg([a-b for a, b in zip(_standardize(rewards), _standardize(entropy))])
    fitted_labels = [0 if i < c1 else 1 if i < c2 else 2 for i in range(len(steps))] if c1 is not None else None
    match = sum(a == b for a, b in zip(labels, fitted_labels))/len(labels) if fitted_labels is not None else None
    low_variance = next((i for i, s in enumerate(steps) if s["advantage_variance"] < .5), None)
    return {"group_size": run["group_size"], "seed": run["seed"], "state_labels": labels,
            "consolidation_entry": b1, "convergence_entry": b2,
            "terminal_reward": terminal,
            "segmentation": {"boundaries": [c1, c2], "step_match": match,
                             "mean_boundary_offset": (abs(b1-c1)+abs(b2-c2))/2 if None not in (b1, b2, c1, c2) else None},
            "state_means": [{"state": state,
                             "reward": st.mean([r for r, label in zip(rewards, labels) if label == state]) if state in labels else None,
                             "entropy": st.mean([e for e, label in zip(entropy, labels) if label == state]) if state in labels else None}
                            for state in range(3)],
            "ordering": state_ordering(labels, rewards, entropy),
            "low_advantage_variance_entry": low_variance, "advantage_variance_threshold": .5,
            "low_variance_minus_convergence": low_variance-b2 if low_variance is not None and b2 is not None else None,
            "heldout_checkpoint_retention": None, "stopping_policy_status": "unmeasured",
            "online_usable": False, "gradient_inactivity_established": False}


def main():
    require_inputs(SRC)
    data = json.loads(SRC.read_text())
    reject_synthetic_or_failed(data)
    runs = [analyze_run(run) for run in data["runs"]]
    if not runs:
        raise ValueError("No trajectories")
    metadata = evidence_metadata("terminal_normalized_retrospective_state_labels", [SRC])
    matched = [r["segmentation"]["step_match"] for r in runs if r["segmentation"]["step_match"] is not None]
    converged = [r for r in runs if r["convergence_entry"] is not None]
    lags = [r["low_variance_minus_convergence"] for r in runs if r["low_variance_minus_convergence"] is not None]
    summary = {**metadata, "runs": runs,
               "H1_state_validity": {"median_step_match": st.median(matched) if matched else None, "n_observed": len(matched)},
               "H2_transition_order": {"complete_transitions": sum(r["ordering"]["status"] == "observed" for r in runs),
                                       "both_monotone": sum(r["ordering"]["reward_increases"] and r["ordering"]["entropy_decreases"] for r in runs)},
               "H3_state_aware_stop": {"status": "insufficient_data", "heldout_retention": None, "realized_compute_savings": None},
               "H4_group_size_schedule": {"spearman": spearman_or_none([math.log2(r["group_size"]) for r in converged], [r["convergence_entry"] for r in converged]),
                                          "n_censored": len(runs)-len(converged)},
               "H5_convergence_low_variance_lag": {"median_lag": st.median(lags) if lags else None,
                                                    "n_observed": len(lags), "n_censored": len(runs)-len(lags)},
               "stopping_policy_effectiveness": None,
               "verdict": "RETROSPECTIVE_ONLY", "reason": "No stopped-checkpoint held-out evaluations"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stateflow_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for number, name, fields, rows in [
        (1, "state_validity", ["group_size", "seed", "step_match", "mean_boundary_offset"],
         [[r["group_size"], r["seed"], r["segmentation"]["step_match"], r["segmentation"]["mean_boundary_offset"]] for r in runs]),
        (2, "transition_order", ["group_size", "seed", "status", "reward_increases", "entropy_decreases"],
         [[r["group_size"], r["seed"], *r["ordering"].values()] for r in runs]),
        (3, "state_aware_stop", ["group_size", "seed", "status", "heldout_retention"],
         [[r["group_size"], r["seed"], "insufficient_data", None] for r in runs]),
        (4, "group_size_schedule", ["group_size", "seed", "convergence_entry"],
         [[r["group_size"], r["seed"], r["convergence_entry"]] for r in runs]),
        (5, "conv_low_variance_lag", ["group_size", "seed", "lag"],
         [[r["group_size"], r["seed"], r["low_variance_minus_convergence"]] for r in runs]),
    ]:
        with (OUT / f"stateflow_h{number}_{name}.tsv").open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(fields)
            writer.writerows(rows)
    write_artifact_metadata(OUT, metadata)
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
