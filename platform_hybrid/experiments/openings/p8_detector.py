#!/usr/bin/env python
"""
P8 method-trace classifier -- independent-run exploratory diagnostic.

Goal: replace the prose-only claim ("full-telemetry AUROC 0.63 vs reward-only 0.35")
with a runnable script that reads the real per-step reward tensors and reports the
AUROC numbers actually obtained when independent-run coverage permits it.
Otherwise writes status=not_reproducible with null AUROCs, never step-leaking folds.

Data: experiments/results/n2_reward_tensor_resume/*_tensors.jsonl
      methods = grpo (genuine) / aero, areal, gift (surrogate)
      40 steps each, 16 prompts x 8 samples per step.

Labeling: positive = surrogate (aero/areal/gift), negative = genuine (grpo).
Each STEP is one sample -> 4 methods x 40 steps = 160 rows.

Model: LogisticRegression with independent-run stratified group CV.
       Pipeline = SimpleImputer(median) + StandardScaler + LogisticRegression
       (imputer guards against any NaN feature so the fit never errors).

Two feature sets are compared:
  * full   -- all honestly-computed telemetry features
  * reward -- reward-level information only (reward_mean, frac_all_zero, frac_all_one)

Every feature below is RECOMPUTED from the raw `rewards` / `lengths` / `advantages`
arrays inside each tensor line -- we do NOT trust the pre-baked scalar fields, so the
computation is auditable end to end.

NOTE on features the P8 prose mentions but which are NOT honestly derivable from a
single step's reward tensor:
  * `loss`          -- a training-side optimizer scalar. It happens to be present as a
                       pre-baked field in the jsonl, but it is NOT reconstructable from
                       the reward/length arrays, so we OMIT it to keep every feature
                       self-derived and reproducible.
  * `lag1_autocorr` -- defined over the temporal reward sequence, not computable from a
                       single step in isolation. We DO reconstruct a reproducible
                       per-method version (lag-1 autocorr of the step reward_mean series)
                       rather than reading the pre-baked field. See build_features().
"""

import json
import glob
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments/results/n2_reward_tensor_resume"
OUT_PATH = ROOT / "experiments/results/p8_openings/detector_reproduced.json"

SURROGATE = {"aero", "areal", "gift"}
GENUINE = {"grpo"}

# Feature sets ---------------------------------------------------------------
# reward-only baseline: information you get from just looking at reward values.
REWARD_ONLY_FEATURES = ["reward_mean", "frac_all_zero", "frac_all_one"]
# full telemetry: reward-only PLUS group-variance / length / advantage structure.
FULL_FEATURES = REWARD_ONLY_FEATURES + [
    "zvf",              # fraction of prompt-groups with zero reward variance
    "reward_std",       # std of all rewards in the step
    "mean_len",         # mean completion length
    "cv_len",           # coeff. of variation of lengths (std/mean)
    "len_reward_corr",  # corr(length, reward) across all samples
    "adv_abs_mean",     # mean |advantage|
    "adv_std",          # std of advantages
    "lag1_autocorr",    # lag-1 autocorr of the reward_mean step-series (per method)
]


def load_steps():
    """Return list of per-step dicts with recomputed features + label."""
    rows = []
    for path in sorted(glob.glob(str(DATA_DIR / "*_tensors.jsonl"))):
        method_steps = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                method_steps.append(d)
        # sort by step to make the autocorr series deterministic
        method_steps.sort(key=lambda x: x.get("step", 0))
        reward_mean_series = []
        feats_seq = []
        for d in method_steps:
            f = build_features(d)
            feats_seq.append((d, f))
            reward_mean_series.append(f["reward_mean"])
        # lag-1 autocorr of the reward_mean series is a per-method scalar; broadcast it
        for prefix_index, (d, f) in enumerate(feats_seq):
            lag1 = lag1_autocorr(reward_mean_series[:prefix_index + 1])
            method = d["method"]
            f["lag1_autocorr"] = lag1
            label = 1 if method in SURROGATE else 0
            rows.append({"run_id": d.get("run_id") or str(path), "method": method, "step": d.get("step"),
                         "label": label, "features": f})
    return rows


def build_features(d):
    """Recompute all step-level features from the raw arrays. No pre-baked fields."""
    rewards = np.asarray(d["rewards"], dtype=float)        # (n_prompts, group_size)
    lengths = np.asarray(d["lengths"], dtype=float)
    advantages = np.asarray(d["advantages"], dtype=float)

    n_groups = rewards.shape[0]

    # zero-variance groups: every sample in the group has identical reward
    group_var = rewards.var(axis=1)
    zvf = float(np.mean(group_var == 0.0))

    all_zero = np.all(rewards == 0.0, axis=1)
    all_one = np.all(rewards == 1.0, axis=1)
    frac_all_zero = float(np.mean(all_zero))
    frac_all_one = float(np.mean(all_one))

    reward_mean = float(rewards.mean())
    reward_std = float(rewards.std())

    mean_len = float(lengths.mean())
    len_std = float(lengths.std())
    cv_len = float(len_std / mean_len) if mean_len > 0 else 0.0

    # corr(length, reward) across every sample; NaN if either is constant.
    rflat = rewards.reshape(-1)
    lflat = lengths.reshape(-1)
    if rflat.std() > 0 and lflat.std() > 0:
        len_reward_corr = float(np.corrcoef(lflat, rflat)[0, 1])
    else:
        len_reward_corr = float("nan")  # imputer handles this

    adv_abs_mean = float(np.abs(advantages).mean())
    adv_std = float(advantages.std())

    return {
        "reward_mean": reward_mean,
        "frac_all_zero": frac_all_zero,
        "frac_all_one": frac_all_one,
        "zvf": zvf,
        "reward_std": reward_std,
        "mean_len": mean_len,
        "cv_len": cv_len,
        "len_reward_corr": len_reward_corr,
        "adv_abs_mean": adv_abs_mean,
        "adv_std": adv_std,
        "n_groups": n_groups,
    }


def lag1_autocorr(series):
    x = np.asarray(series, dtype=float)
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    a, b = x[:-1], x[1:]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def make_matrix(rows, feature_names):
    X = np.array([[r["features"].get(f, np.nan) for f in feature_names] for r in rows],
                 dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    return X, y


def cv_auroc(X, y, seed=0, groups=None):
    """Out-of-run CV; require enough independent runs of each label."""
    if groups is None:
        raise ValueError("Independent run IDs required; random step folds leak run identity")
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError("Missing run identity")
    for group in set(groups):
        if len(set(y[groups == group])) != 1:
            raise ValueError("A run has inconsistent labels")
    counts = [len(set(groups[y == label])) for label in set(y)]
    if len(counts) != 2 or min(counts) < 2:
        raise ValueError("At least two independent runs per class required")
    folds = min(5, min(counts))
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y, groups))
    if any(len(set(y[train])) < 2 or len(set(y[test])) < 2 for train, test in splits):
        raise ValueError("Insufficient class coverage in independent-run folds")
    proba = cross_val_predict(pipe, X, y, cv=splits, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba)), proba


def main():
    rows = load_steps()
    n_pos = sum(r["label"] for r in rows)
    n_neg = len(rows) - n_pos
    methods = sorted({r["method"] for r in rows})

    groups = [r["run_id"] for r in rows]
    X_full, y = make_matrix(rows, FULL_FEATURES)
    X_rew, _ = make_matrix(rows, REWARD_ONLY_FEATURES)

    try:
        auroc_full, _ = cv_auroc(X_full, y, seed=0, groups=groups)
    except ValueError as error:
        out = {"status": "not_reproducible", "claim_eligible": False,
               "reason": str(error), "cv": "StratifiedGroupKFold(independent_runs)",
               "n_rows": len(rows), "methods": methods,
               "auroc_full_telemetry": None, "auroc_reward_only": None}
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out))
        return out
    auroc_rew, _ = cv_auroc(X_rew, y, seed=0, groups=groups)

    # stability across a few CV seeds
    seeds = [0, 1, 2, 3, 4]
    full_seeds = [cv_auroc(X_full, y, s, groups=groups)[0] for s in seeds]
    rew_seeds = [cv_auroc(X_rew, y, s, groups=groups)[0] for s in seeds]

    out = {
        "description": "Exploratory P8 method-trace classifier. Positive=surrogate "
                       "(aero/areal/gift), negative=genuine (grpo). LogisticRegression, "
                       "independent-run stratified group CV, out-of-fold AUROC.",
        "data_dir": str(DATA_DIR.relative_to(ROOT)),
        "methods": methods,
        "n_rows": len(rows),
        "n_positive_surrogate": n_pos,
        "n_negative_genuine": n_neg,
        "full_features": FULL_FEATURES,
        "reward_only_features": REWARD_ONLY_FEATURES,
        "omitted_prose_features": {
            "loss": "training-side optimizer scalar; not derivable from reward/length "
                    "tensor, so omitted for reproducibility",
        },
        "auroc_full_telemetry": round(auroc_full, 4),
        "auroc_reward_only": round(auroc_rew, 4),
        "auroc_full_telemetry_seed_mean": round(float(np.mean(full_seeds)), 4),
        "auroc_full_telemetry_seed_std": round(float(np.std(full_seeds)), 4),
        "auroc_reward_only_seed_mean": round(float(np.mean(rew_seeds)), 4),
        "auroc_reward_only_seed_std": round(float(np.std(rew_seeds)), 4),
        "auroc_full_by_seed": {str(s): round(v, 4) for s, v in zip(seeds, full_seeds)},
        "auroc_reward_by_seed": {str(s): round(v, 4) for s, v in zip(seeds, rew_seeds)},
        "cv": "StratifiedGroupKFold(independent_runs)",
        "pipeline": "SimpleImputer(median) -> StandardScaler -> LogisticRegression",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fh:
        json.dump(out, fh, indent=2)

    print(f"rows={len(rows)}  pos(surrogate)={n_pos}  neg(genuine)={n_neg}  methods={methods}")
    print(f"AUROC full-telemetry (seed0) = {auroc_full:.4f}")
    print(f"AUROC reward-only   (seed0) = {auroc_rew:.4f}")
    print(f"AUROC full  over seeds {seeds}: mean={np.mean(full_seeds):.4f} std={np.std(full_seeds):.4f}")
    print(f"AUROC rew   over seeds {seeds}: mean={np.mean(rew_seeds):.4f} std={np.std(rew_seeds):.4f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
