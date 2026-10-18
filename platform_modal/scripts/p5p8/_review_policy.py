"""Versioned, CPU-only policy primitives for corrected review20/21 analyses.

Threshold search on supplied labels is a retrospective oracle diagnostic, not
held-out deployment performance. No model fitting or artifact I/O here.
"""

from __future__ import annotations
import math

PROTOCOL_VERSION = "review-policy-v2"


def alert_threshold(investigation_cost, miss_loss):
    if investigation_cost < 0 or miss_loss < 0:
        raise ValueError("costs must be nonnegative")
    return investigation_cost / miss_loss if miss_loss > 0 else math.inf


def optimal_threshold(
    scores, labels, investigation_cost, miss_loss, sensing_cost=0, max_alerts=None
):
    """Exact >= threshold policy including no-alert/all-alert, preserving ties."""
    if len(scores) != len(labels) or not len(labels):
        raise ValueError("nonempty aligned scores and labels required")
    if any(not math.isfinite(float(s)) for s in scores) or any(y not in (0, 1) for y in labels):
        raise ValueError("finite scores and binary labels required")
    if min(investigation_cost, miss_loss, sensing_cost) < 0:
        raise ValueError("costs must be nonnegative")
    pairs = sorted(zip(scores, labels, strict=False), reverse=True)
    n, positives = len(labels), int(sum(labels))
    best = dict(
        tau=math.inf,
        tp=0,
        fp=0,
        fn=positives,
        kstar=0,
        total_cost=miss_loss * positives + sensing_cost * n,
    )
    tp = 0
    for i, (score, label) in enumerate(pairs):
        tp += int(label)
        if i + 1 < n and pairs[i + 1][0] == score:
            continue
        k = i + 1
        if max_alerts is not None and k > max_alerts:
            continue
        total = investigation_cost * k + miss_loss * (positives - tp) + sensing_cost * n
        if total < best["total_cost"]:
            best = dict(
                tau=float(score), tp=tp, fp=k - tp, fn=positives - tp, kstar=k, total_cost=total
            )
    best.update(
        n_rows=n,
        alert_rate=best["kstar"] / n,
        recall=best["tp"] / positives if positives else None,
        cost_per_dec=best["total_cost"] / n,
        cost_per_caught=best["total_cost"] / best["tp"] if best["tp"] else None,
        selection_status="test_optimized_oracle",
    )
    return best
