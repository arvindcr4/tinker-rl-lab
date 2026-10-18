"""Deep Review 6c172429 audit of P2 (ZVF) numbers that do not depend on the
simulated variance-mitigation corpus.

  * 12-anchor phase classifier (Section 10.2-10.3): leakage-free LOO
    nearest-centroid on the stored features of zvf_iter34_phase_scan.tsv, a
    label-permutation null, and the majority-class baseline;
  * Section 7 Spearman on the five-anchor corpus.

Writes experiments/results/zvf_review_6c17_*.tsv. No new training.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RES = Path(__file__).resolve().parents[2] / "platform_hybrid/experiments/results"
FEATS = ["zvf_level", "zvf_instability", "zvf_slope", "zvf_direction", "zvf_discriminator"]


def write(name, rows):
    path = RES / f"zvf_review_6c17_{name}.tsv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.name}")


def loo_nearest_centroid(X, y):
    """Strict LOO: normalisation and centroids from the training fold only.
    A held-out point can only receive a class present in its training fold."""
    pred = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        mu, sd = X[m].mean(0), X[m].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr, xte = (X[m] - mu) / sd, (X[i] - mu) / sd
        classes = sorted(set(y[m]))
        cent = np.stack([Xtr[y[m] == c].mean(0) for c in classes])
        pred.append(classes[int(np.argmin(((cent - xte) ** 2).sum(1)))])
    return np.array(pred)


def main():
    rows = list(csv.DictReader(open(RES / "zvf_iter34_phase_scan.tsv"), delimiter="\t"))
    X = np.array([[float(r[f]) for f in FEATS] for r in rows])
    y = np.array([r["phase"] for r in rows])
    pred = loo_nearest_centroid(X, y)
    acc = float((pred == y).mean())
    rng = np.random.default_rng(20261018)
    null = np.array(
        [(loo_nearest_centroid(X, rng.permutation(y)) == y).mean() for _ in range(2000)]
    )
    labels, counts = np.unique(y, return_counts=True)
    out = [
        {"model": r["model"], "true_phase": t, "pred_phase_strict_loo": p}
        for r, t, p in zip(rows, y, pred, strict=False)
    ]
    write("iter34_strict_loo_predictions", out)
    summary = [
        {
            "n": len(y),
            "strict_loo_accuracy": round(acc, 4),
            "majority_baseline": round(counts.max() / len(y), 4),
            "label_permutation_p": round(float((null >= acc).mean()), 4),
            **{f"recall_{c}": round(float((pred[y == c] == c).mean()), 4) for c in labels},
        }
    ]
    write("iter34_strict_loo_summary", summary)
    print(summary[0])

    frac = [0.0, 0.1333, 0.0, 0.0, 0.55]
    zero = [0.0, 0.0667, 0.0, 0.0, 0.55]
    coll = [0, 0, 0, 0, 1]
    write(
        "scaling_spearman",
        [
            {"proxy": "frac_below_0p1", "rho": round(spearmanr(frac, coll)[0], 4)},
            {"proxy": "zero_fraction", "rho": round(spearmanr(zero, coll)[0], 4)},
        ],
    )


if __name__ == "__main__":
    main()
