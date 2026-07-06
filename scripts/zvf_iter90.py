#!/usr/bin/env python3
"""Pillar 2 Iter 90 — ZVF Recovery Asymmetry.

Distinct from prior iters:
  iter74 Markov, iter78 EWS lead-time, iter82 hazard+survival,
  iter86 decision-theoretic (tau,K) cost-savings.

Iter 90 asks: WHEN the ZVF alarm fires, is the alarm TRANSIENT or
SUSTAINED? And what is the per-step heldout-acc trajectory in each
case? This is the recovery framing complement to iter86 — if many
alarms are transient, iter86's "stop" rule is over-conservative for
those methods, and the proper decision-theoretic fix is to wait K
post-alarm steps before stopping.

Input: experiments/results/variance_mitigation.tsv (45 traces:
       9 methods x 5 seeds x ~123 steps).
Output:
    experiments/results/zvf_iter90_episodes.tsv     (one row per alarm episode)
    experiments/results/zvf_iter90_post_episode.tsv (one row per (method, post-cat))
    experiments/results/zvf_iter90_recovery.tsv     (per-method recovery rate)
    experiments/results/zvf_iter90_summary.tsv      (one-line headline)
    figures/zvf_iter90_recovery.{pdf,png}
"""

from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "experiments" / "results" / "variance_mitigation.tsv"
OUT  = ROOT / "experiments" / "results"
FIG  = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TAU       = 0.5   # alarm threshold (locked to iter78/82/86)
K_RECOVER = 5     # alarm-to-recovery window (matches iter86 k_persist)
ROLL      = 10    # rolling window for heldout-Acc smoothing


def load_traces():
    """Return traces[(method, seed)] = list of {step, zvf, ha, collapse}."""
    out = defaultdict(list)
    with open(SRC) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            out[(row["method"], int(row["seed"]))].append({
                "step"    : int(row["step"]),
                "zvf"     : float(row["zvf"]),
                "ha"      : float(row["heldout_acc"]),
                "collapse": int(row["collapse"]),
            })
    for k in out:
        out[k].sort(key=lambda d: d["step"])
    return out


def rolling(seq, k):
    n = len(seq)
    out = [None] * n
    s = 0.0
    for i, v in enumerate(seq):
        s += v
        if i >= k:
            s -= seq[i - k]
        if i >= k - 1:
            out[i] = s / k
    return out


def detect_episodes(trace):
    """Return list of dicts: {start, end, len_alarm,
                                recovered: bool, recovery_step or None}."""
    episodes, cur = [], None
    n = len(trace)
    ha_roll = rolling([d["ha"] for d in trace], ROLL)
    for i, d in enumerate(trace):
        alarmed = d["zvf"] > TAU
        if alarmed and cur is None:
            cur = {"start": i, "end": i}
        elif alarmed and cur is not None:
            cur["end"] = i
        elif not alarmed and cur is not None:
            j = i
            recovered_within = False
            recovery_step = None
            while j < min(n, cur["end"] + 1 + K_RECOVER):
                if j >= n:
                    break
                if trace[j]["zvf"] <= TAU:
                    recovered_within = True
                    recovery_step = j
                    break
                j += 1
            cur["len_alarm"]      = cur["end"] - cur["start"] + 1
            cur["recovered"]      = recovered_within
            cur["recovery_step"]  = recovery_step
            cur["ha_at_alarm"]    = trace[cur["start"]]["ha"]
            episodes.append(cur)
            cur = None
    if cur is not None:
        cur["len_alarm"]      = cur["end"] - cur["start"] + 1
        cur["recovered"]      = False
        cur["recovery_step"]  = None
        cur["ha_at_alarm"]    = trace[cur["start"]]["ha"]
        episodes.append(cur)
    return episodes


def post_window(trace, start, L=15):
    end = min(len(trace), start + L + 1)
    return [trace[i]["ha"] for i in range(start, end)]


def main():
    traces = load_traces()
    print(f"[iter90] loaded {len(traces)} traces")
    # Per-episode dump
    with open(OUT / "zvf_iter90_episodes.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method", "seed", "start", "end", "len_alarm",
                    "recovered", "recovery_step", "ha_at_alarm"])
        for (m, s), tr in traces.items():
            for ep in detect_episodes(tr):
                w.writerow([m, s, ep["start"], ep["end"], ep["len_alarm"],
                            int(ep["recovered"]),
                            -1 if ep["recovery_step"] is None else ep["recovery_step"],
                            f"{ep['ha_at_alarm']:.6f}"])
    # Per-method recovery rate + per-cat post-ha delta
    by_method   = defaultdict(lambda: {"n_ep": 0, "n_rec": 0, "ha_post_rec": [],
                                        "ha_post_sus": [], "ha_never": []})
    for (m, s), tr in traces.items():
        ep = detect_episodes(tr)
        seen_alarm = bool(ep)
        b = by_method[m]
        # Never-alarmed trajectories: their rolling heldout-acc avg.
        if not seen_alarm:
            b["ha_never"].extend([d["ha"] for d in tr if d["step"] >= 5])
        for e in ep:
            b["n_ep"] += 1
            if e["recovered"]:
                b["n_rec"] += 1
                rs = e["recovery_step"]
                if rs is not None:
                    b["ha_post_rec"].extend(post_window(tr, rs))
            else:
                es_ = e["end"]
                if es_ < len(tr):
                    b["ha_post_sus"].extend(post_window(tr, es_))
    # Per-(method, post-cat) rows
    with open(OUT / "zvf_iter90_post_episode.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method", "cat", "n_obs", "mean_ha", "median_ha"])
        for m, b in by_method.items():
            for cat, vals in (("recovered", b["ha_post_rec"]),
                              ("sustained", b["ha_post_sus"]),
                              ("never_alarmed", b["ha_never"])):
                if not vals:
                    w.writerow([m, cat, 0, "NA", "NA"])
                    continue
                sv = sorted(vals)
                med = sv[len(sv) // 2]
                w.writerow([m, cat, len(vals),
                            f"{sum(vals)/len(vals):.6f}", f"{med:.6f}"])
    # Per-method recovery table
    rows = []
    with open(OUT / "zvf_iter90_recovery.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method", "n_episodes", "n_recovered", "n_sustained",
                    "recovery_rate", "n_never_alarmed_traces", "trajectory_count"])
        for m, b in by_method.items():
            ne, nr = b["n_ep"], b["n_rec"]
            ns     = ne - nr
            rec    = nr / ne if ne else 0.0
            nna    = len([1 for (mm, _), tr in traces.items()
                          if mm == m and not detect_episodes(tr)])
            tc     = sum(1 for (mm, _) in traces if mm == m)
            w.writerow([m, ne, nr, ns, f"{rec:.4f}", nna, tc])
            rows.append((m, ne, nr, ns, rec, nna, tc))
    # Headline summary
    total_ep  = sum(r[1] for r in rows)
    total_rec = sum(r[2] for r in rows)
    total_sus = sum(r[3] for r in rows)
    pooled_rec = (total_rec / total_ep) if total_ep else 0.0
    never_tr   = sum(r[5] for r in rows)
    summary ={
        "n_traces":               len(traces),
        "n_pooled_episodes":      total_ep,
        "n_pooled_recovered":     total_rec,
        "n_pooled_sustained":     total_sus,
        "pooled_recovery_rate":   round(pooled_rec, 4),
        "tau":                    TAU,
        "k_recover":              K_RECOVER,
        "best_method_recovery":   max(rows, key=lambda r: r[4])[0] if rows else None,
        "best_recovery_rate":     round(max((r[4] for r in rows), default=0.0), 4),
        "worst_method_recovery":  min(rows, key=lambda r: r[4])[0] if rows else None,
        "worst_recovery_rate":    round(min((r[4] for r in rows), default=0.0), 4),
        "by_method_recovery":     {r[0]: round(r[4], 4) for r in rows},
    }
    with open(OUT / "zvf_iter90_summary.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["key", "value"])
        for k, v in summary.items():
            if isinstance(v, dict):
                w.writerow([f"per_method.{k}", json.dumps(v)])
            else:
                w.writerow([k, v])
    with open(OUT / "zvf_iter90_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # Plot
    methods = [r[0] for r in rows]
    rates   = [r[4] for r in rows]
    plt.figure(figsize=(8, 4))
    order = sorted(range(len(methods)), key=lambda i: rates[i])
    plt.barh([methods[i] for i in order],
             [rates[i] for i in order],
             color=["#3b8ec2" if rates[i] > pooled_rec else "#d6604d"
                    for i in order])
    plt.axvline(pooled_rec, color="black", linestyle="--", linewidth=1)
    plt.xlabel(f"Recovery rate within K={K_RECOVER} steps after alarm fires")
    plt.title(f"ZVF alarm recovery rate by method "
              f"(pooled={pooled_rec:.2%}, tau={TAU})")
    plt.tight_layout()
    plt.savefig(FIG / "zvf_iter90_recovery.pdf")
    plt.savefig(FIG / "zvf_iter90_recovery.png", dpi=140)
    plt.close()
    print(f"[iter90] DONE  ep={total_ep} rec={total_rec} "
          f"sus={total_sus} pooled_rate={pooled_rec:.4f}")
    print(f"[iter90] summary.json saved")
    return summary


if __name__ == "__main__":
    main()
