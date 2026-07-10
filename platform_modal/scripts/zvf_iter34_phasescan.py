#!/usr/bin/env python3
"""Iter 34 — Pillar 2 cross-pillar ZVF phase discrimination.

Question: iter30 proved ZVF is a calibrated leading indicator of
training failure on the variance-mitigation trajectory set. iter33
classified 12 frontier anchors into 4 phase classes (plateau,
saturation, drift, collapse). This iteration connects the two:

    Can the iter30 ZVF-feature decomposition (current ZVF,
    trailing_mean, trailing_slope) DISCRIMINATE phase classes on
    the iter33 frontier set?

Because the iter33 summary is at the trace level (one row per
model), we proxy ZVF features via summary statistics that correlate
with the per-step ZVF stream:

    zvf_level_proxy       = zero_frac          (fraction of
                                              all-zero reward
                                              steps; the
                                              closest summary
                                              statistic to
                                              ZVF = 1 for
                                              a given step)
    zvf_instability_proxy = peak_frac          (fraction of
                                              steps at peak;
                                              high peak_frac
                                              means rewards
                                              bunched -> high
                                              variance
                                              distribution)
    zvf_slope_proxy       = ols_slope_per_step (drift in reward
                                              that drives ZVF
                                              slope)
    zvf_discriminator     = zero_frac * (1 - delta_late_minus_early)
                                              (combined ZVF
                                              level + reward
                                              direction)

The classification uses leave-one-out CV (12 folds) over the 4-class
phase label, evaluating both multiclass accuracy and per-class
recovery. We also build a feature-importance table via
permutation, and a confusion matrix between true and predicted
phases.

Outputs:
    experiments/results/zvf_iter34_phase_scan.tsv     (12 rows)
    experiments/results/zvf_iter34_discriminant.tsv   (4 rows
                                                      per-phase
                                                      feature
                                                      mean +- sd)
    experiments/results/zvf_iter34_feature_importance.tsv (rows
                                                      per feature)
    experiments/results/zvf_iter34_confusion.tsv      (4x4 LOO
                                                      confusion)
    experiments/results/zvf_iter34_summary.tsv        (one-row
                                                      rollup)

Source: platform_modal/scripts/zvf_iter34_phasescan.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results"


def zvf_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute iter33-derived ZVF proxy features.

    For each trace we derive 4 features that mirror the iter30
    per-step ZVF feature battery (current, trailing_mean,
    trailing_slope, combined).
    """
    out = pd.DataFrame()
    out["model"] = df["model"]
    out["arch"] = df["arch"]
    out["phase"] = df["phase_classifier"]
    out["zvf_level"] = df["zero_frac"].fillna(0.0)
    out["zvf_instability"] = df["peak_frac"].fillna(0.0)
    out["zvf_slope"] = df["ols_slope_per_step"].fillna(0.0)
    out["zvf_direction"] = df["delta_late_minus_early"].fillna(0.0)
    out["zvf_discriminator"] = out["zvf_level"] * np.clip(
        1.0 - out["zvf_direction"], 0.0, 1.0
    )
    return out


def _normalise(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def loo_classify(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, classes: list[str]
) -> np.ndarray:
    """Weighted nearest-centroid classifier on normalised features.

    Returns predicted class label for each row of Xte.
    """
    Xtr_n = _normalise(Xtr)
    Xte_n = (Xte - Xtr.mean(axis=0)) / Xtr.std(axis=0).clip(min=1e-9)
    centroids = np.stack(
        [Xtr_n[ytr == c].mean(axis=0) for c in classes]
    )
    d = cdist(Xte_n, centroids, metric="euclidean")
    nearest = np.argmin(d, axis=1)
    return np.array([classes[i] for i in nearest])


def loo_phase_classifier(
    feats: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Leave-one-out 4-class phase classifier with feature
    permutation importance. Uses weighted nearest-centroid
    (no sklearn dependency).
    """
    X = feats[feature_cols].to_numpy(dtype=float)
    y = feats["phase"].to_numpy()
    classes = sorted(set(y))
    n = len(y)
    pred = np.empty(n, dtype=object)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred[i] = loo_classify(X[mask], y[mask], X[i : i + 1], classes)[0]

    acc = float((pred == y).mean())
    confusion = pd.crosstab(
        pd.Series(y, name="true"),
        pd.Series(pred, name="pred"),
    ).reindex(index=classes, columns=classes, fill_value=0)

    base_acc = acc
    importance_rows = []
    rng = np.random.default_rng(20260702)
    for ci, col in enumerate(feature_cols):
        perm_accs = []
        for _ in range(200):
            Xp = X.copy()
            Xp[:, ci] = rng.permutation(Xp[:, ci])
            preds = []
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                preds.append(loo_classify(Xp[mask], y[mask], Xp[i : i + 1], classes)[0])
            perm_accs.append(float(np.mean(np.array(preds) == y)))
        importance_rows.append(
            {
                "feature": col,
                "base_acc": base_acc,
                "perm_mean_acc": float(np.mean(perm_accs)),
                "perm_std_acc": float(np.std(perm_accs)),
                "delta_acc": base_acc - float(np.mean(perm_accs)),
                "p_shuffled_beats": float(
                    np.mean(np.array(perm_accs) >= base_acc)
                ),
            }
        )

    importance = pd.DataFrame(importance_rows).sort_values(
        "delta_acc", ascending=False
    )

    per_class_recall = {}
    for c in classes:
        mask = y == c
        per_class_recall[c] = float((pred[mask] == c).mean()) if mask.sum() else float("nan")

    metrics = {
        "n": n,
        "accuracy": acc,
        "n_classes": len(classes),
        "classes": classes,
        "per_class_recall": per_class_recall,
        "n_unique_pred": len(set(pred)),
    }

    pred_df = pd.DataFrame(
        {
            "model": feats["model"].to_numpy(),
            "true_phase": y,
            "pred_phase": pred,
        }
    )

    return pred_df, confusion, metrics, importance


def per_phase_discriminant(feats: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Mean and SD of each feature per phase class."""
    rows = []
    for phase, sub in feats.groupby("phase"):
        for col in feature_cols:
            rows.append(
                {
                    "phase": phase,
                    "feature": col,
                    "mean": float(sub[col].mean()),
                    "sd": float(sub[col].std(ddof=0)) if len(sub) > 1 else 0.0,
                    "min": float(sub[col].min()),
                    "max": float(sub[col].max()),
                    "n": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def collapse_discrimination(
    feats: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """For each feature, the gap between collapse and the
    best-of-rest value."""
    collapse = feats[feats["phase"] == "collapse"]
    others = feats[feats["phase"] != "collapse"]
    rows = []
    for col in feature_cols:
        rows.append(
            {
                "feature": col,
                "collapse_mean": float(collapse[col].mean()) if len(collapse) else float("nan"),
                "noncollapse_mean": float(others[col].mean()) if len(others) else float("nan"),
                "noncollapse_max": float(others[col].max()) if len(others) else float("nan"),
                "noncollapse_min": float(others[col].min()) if len(others) else float("nan"),
                "collapse_minus_rest": float(collapse[col].mean() - others[col].mean())
                if len(collapse) and len(others)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    src = OUT / "scaling_law_iter33_phase_score.tsv"
    df = pd.read_csv(src, sep="\t")
    feats = zvf_proxies(df)
    feature_cols = [
        "zvf_level",
        "zvf_instability",
        "zvf_slope",
        "zvf_direction",
        "zvf_discriminator",
    ]
    feats.to_csv(OUT / "zvf_iter34_phase_scan.tsv", sep="\t", index=False)

    pred_df, confusion, metrics, importance = loo_phase_classifier(
        feats, feature_cols
    )
    pred_df.to_csv(OUT / "zvf_iter34_predictions.tsv", sep="\t", index=False)
    confusion.to_csv(OUT / "zvf_iter34_confusion.tsv", sep="\t")
    importance.to_csv(OUT / "zvf_iter34_feature_importance.tsv", sep="\t", index=False)

    disc = per_phase_discriminant(feats, feature_cols)
    disc.to_csv(OUT / "zvf_iter34_discriminant.tsv", sep="\t", index=False)

    collapse_gap = collapse_discrimination(feats, feature_cols)
    collapse_gap.to_csv(OUT / "zvf_iter34_collapse_gap.tsv", sep="\t", index=False)

    summary = {
        "n_anchors": int(metrics["n"]),
        "n_classes": int(metrics["n_classes"]),
        "loo_accuracy": float(metrics["accuracy"]),
        "chance_accuracy": 1.0 / metrics["n_classes"],
        "per_class_recall": metrics["per_class_recall"],
        "top_feature": importance.iloc[0]["feature"],
        "top_feature_delta_acc": float(importance.iloc[0]["delta_acc"]),
        "collapse_zvf_level": float(
            feats.loc[feats["phase"] == "collapse", "zvf_level"].mean()
        ),
        "noncollapse_zvf_level": float(
            feats.loc[feats["phase"] != "collapse", "zvf_level"].mean()
        ),
        "collapse_zvf_discriminator": float(
            feats.loc[feats["phase"] == "collapse", "zvf_discriminator"].mean()
        ),
        "noncollapse_zvf_discriminator": float(
            feats.loc[feats["phase"] != "collapse", "zvf_discriminator"].mean()
        ),
        "collapse_only_model": str(
            feats.loc[feats["phase"] == "collapse", "model"].iat[0]
        ),
    }
    with open(OUT / "zvf_iter34_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(OUT / "zvf_iter34_summary.tsv", sep="\t", index=False)

    print(f"n_anchors={summary['n_anchors']} LOO acc={summary['loo_accuracy']:.3f} "
          f"top_feature={summary['top_feature']} delta={summary['top_feature_delta_acc']:.3f}")
    print(f"collapse_model={summary['collapse_only_model']} "
          f"collapse_zvf_level={summary['collapse_zvf_level']:.3f} "
          f"vs noncollapse={summary['noncollapse_zvf_level']:.3f}")


if __name__ == "__main__":
    main()