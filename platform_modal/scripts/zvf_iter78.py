"""Iter 78 — ZVF as a Real-Time Online Diagnostic Protocol (EWS).

Converts the iter70/iter74 static correlation evidence into a *protocol*:
for a running training trace, an online early-warning signal (EWS) that
fires BEFORE heldout accuracy collapses, with measurable lead-time.

Definitions (locked):
  * Failure step t_fail: first t such that mean(heldout_acc[max(0,t-9)..t+1]) < 0.10
  * EWS alarm step t_alarm: first t such that ews_stat(t) > threshold
  * Lead time: t_fail - t_alarm  (positive => EWS leads; negative => late)
  * Window w: trailing 10 steps
  * Alarm threshold: swept from {0.50, 0.60, 0.70, 0.80, 0.90} on the EWS stat
  * EWS stat: max(AR1, CUSUM, length_H_run) — composite that *any* of the three
    EWS components can trip (union rule). Single-channel EWS in
    zvf_iter78_single_channel.tsv.

Inputs:
  platform_hybrid/experiments/results/variance_mitigation.tsv     (45 traces, 9 lib x 5 seed)
  platform_hybrid/experiments/results/groupsize_zvf_sweep.json    (12 traces, 4 G x 3 seed)
  platform_hybrid/experiments/results/tinker_gsm8k_zvf_*.json     (3 traces, 200 problems)
  platform_hybrid/experiments/results/bfclv4_tool_use.tsv         (canonical failure anchor)

Outputs (in platform_hybrid/experiments/results/):
  zvf_iter78_per_step_features.tsv
  zvf_iter78_alarm_thresholds.tsv
  zvf_iter78_leadtime_summary.tsv
  zvf_iter78_ews_protocol.tsv
  zvf_iter78_anchors.tsv
  zvf_iter78_summary.tsv
  zvf_iter78_meta.json
  zvf_iter78_single_channel.tsv

Stdlib only: no numpy/scipy. Closed-form OLS for AR(1) (no matrix inversion).
Seed for any resampling: 20260703.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

RESULTS = "platform_hybrid/experiments/results"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
SEED = 20260703
W = 10                 # trailing window for EWS features
STUCK_W = 20           # window to detect "stuck" failure
STUCK_CUTOFF = 0.50    # ZVF >= 0.50 => stuck (iter74 H cutoff)
H_CUTOFF = 0.50        # iter74 H/M/L threshold (L<=0.10, M<=0.50, H>0.50)
H_LO = 0.10            # iter74 M/L boundary

# Alarm thresholds swept
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


# ----------------------- core statistics (closed-form) -----------------------

def ar1_coef(xs: list[float]) -> float:
    """Closed-form AR(1) coefficient rho = Cov(x_t, x_{t+1}) / Var(x_t)."""
    if len(xs) < 3:
        return 0.0
    mu = sum(xs) / len(xs)
    num = 0.0
    den = 0.0
    for i in range(len(xs) - 1):
        a = xs[i] - mu
        b = xs[i + 1] - mu
        num += a * b
        den += a * a
    if den <= 0.0:
        return 0.0
    rho = num / den
    if rho > 1.0:
        return 1.0
    if rho < -1.0:
        return -1.0
    return rho


def window_var(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)


def window_kurt(xs: list[float]) -> float:
    """Excess kurtosis (Fisher) on the trailing window."""
    if len(xs) < 4:
        return 0.0
    mu = sum(xs) / len(xs)
    m2 = sum((x - mu) ** 2 for x in xs) / len(xs)
    if m2 <= 0.0:
        return 0.0
    m4 = sum((x - mu) ** 4 for x in xs) / len(xs)
    return m4 / (m2 * m2) - 3.0


def cusum_step(xs: list[float], baseline_mean: float, k: float = 0.05) -> float:
    """Cumulative sum of (x_t - baseline_mean - k) clipped below at 0.

    Reference: Page 1954 CUSUM for change-point detection. Drift k=0.05
    is the standard "small shift" choice; we use it on the ZVF scale [0,1].
    """
    s = 0.0
    s_max = 0.0
    for x in xs:
        s = max(0.0, s + (x - baseline_mean - k))
        if s > s_max:
            s_max = s
    return s_max


def variance_ratio(xs: list[float]) -> float:
    """VR(2) = Var(x_t + x_{t-1}) / (2 * Var(x_t)). VR>1 => momentum; VR<1 => mean-revert."""
    if len(xs) < 4:
        return 1.0
    v1 = window_var(xs)
    if v1 <= 0.0:
        return 1.0
    pairs = [xs[i] + xs[i + 1] for i in range(len(xs) - 1)]
    v2 = window_var(pairs)
    return v2 / (2.0 * v1)


# ----------------------- trace loading -----------------------

def load_variance_mitigation() -> list[dict]:
    """Return list of {method, seed, steps:[{step,zvf,reward,heldout,collapse}]}."""
    f = os.path.join(RESULTS, "variance_mitigation.tsv")
    by_run = defaultdict(list)
    with open(f) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            d = dict(zip(header, parts))
            try:
                row = {
                    "step": int(d["step"]),
                    "zvf": float(d["zvf"]),
                    "reward_mean": float(d["reward_mean"]),
                    "heldout_acc": float(d["heldout_acc"]),
                    "collapse": int(d.get("collapse", "0") or 0),
                }
            except (ValueError, KeyError):
                continue
            by_run[(d["method"], d["seed"])].append(row)
    out = []
    for (method, seed), rows in by_run.items():
        rows.sort(key=lambda r: r["step"])
        out.append({
            "source": "variance_mitigation",
            "method": method,
            "seed": seed,
            "n_steps": len(rows),
            "steps": rows,
        })
    return out


def load_groupsize_sweep() -> list[dict]:
    f = os.path.join(RESULTS, "groupsize_zvf_sweep.json")
    data = json.load(open(f))
    out = []
    for run in data.get("runs", []):
        sl = run.get("step_log", [])
        rows = []
        for s in sl:
            rows.append({
                "step": int(s.get("step", 0)),
                "zvf": float(s.get("zvf", 0.0)),
                "reward_mean": float(s.get("mean_reward", 0.0)),
                "heldout_acc": float(s.get("heldout_acc", run.get("last10_avg", 0.0))),
                "collapse": 0,
            })
        if not rows:
            continue
        out.append({
            "source": "groupsize_zvf_sweep",
            "method": "groupsize",
            "seed": str(run.get("seed", 0)),
            "group_size": int(run.get("group_size", 0)),
            "n_steps": len(rows),
            "steps": rows,
        })
    return out


def load_tinker_gsm8k() -> list[dict]:
    """Per-problem ZVF data: 200 problems, no per-step. Used as cross-prompt anchor."""
    out = []
    for seed_tag in ["s42", "s123", "s456"]:
        f = os.path.join(RESULTS, f"tinker_gsm8k_zvf_{seed_tag}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        seed = int(d.get("seed", 0))
        per = d.get("per_problem", [])
        out.append({
            "source": "tinker_gsm8k_zvf",
            "method": "tinker_gsm8k",
            "seed": str(seed),
            "n_steps": 1,  # single-shot experiment
            "steps": [{
                "step": 0,
                "zvf": float(d.get("overall_zvf", 0.0)),
                "reward_mean": float(d.get("overall_accuracy", 0.0)),
                "heldout_acc": float(d.get("overall_accuracy", 0.0)),
                "collapse": 0,
            }],
            "per_problem_zvf": [float(p.get("zvf", 0.0)) for p in per],
            "n_problems": len(per),
            "group_size": int(d.get("group_size", 8)),
        })
    return out


def load_tool_use_anchors() -> list[dict]:
    """BFCLV4 tool_use anchor: 2 seeds, per-step from bfclv4_tool_use.tsv.

    Columns: seed, step, n_correct, n_total, reward_sparse, reward_dense,
             zvf_sparse, zvf_dense. We use zvf_sparse as the ZVF series
    and reward_sparse as the heldout proxy.
    """
    f = os.path.join(RESULTS, "bfclv4_tool_use.tsv")
    if not os.path.exists(f):
        return []
    out = []
    by_run: dict[str, list[dict]] = defaultdict(list)
    with open(f) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            d = dict(zip(header, parts))
            try:
                row = {
                    "step": int(d.get("step", 0)),
                    "zvf": float(d.get("zvf_sparse", 0.0)),
                    "reward_mean": float(d.get("reward_sparse", 0.0)),
                    "heldout_acc": float(d.get("reward_sparse", 0.0)),
                    "collapse": 0,
                }
            except (ValueError, KeyError):
                continue
            by_run[str(d.get("seed", "?"))].append(row)
    for seed, rows in by_run.items():
        rows.sort(key=lambda r: r["step"])
        out.append({
            "source": "bfclv4_tool_use",
            "method": "tool_use",
            "seed": seed,
            "n_steps": len(rows),
            "steps": rows,
        })
    return out


# ----------------------- per-trace EWS features -----------------------

def classify_state(zvf: float) -> str:
    if zvf > H_CUTOFF:
        return "H"
    if zvf > H_LO:
        return "M"
    return "L"


def per_step_features(trace: dict) -> list[dict]:
    """For each step t, compute trailing-window EWS features.

    The trailing window is xs = zvf_trace[max(0, t-W+1)..t+1].
    For t < W-1 we use a shorter window (grace period).
    """
    steps = trace["steps"]
    zvf_series = [s["zvf"] for s in steps]
    held_series = [s["heldout_acc"] for s in steps]
    n = len(zvf_series)
    if n == 0:
        return []

    baseline_mean = sum(zvf_series[: max(1, n // 4)]) / max(1, n // 4)
    out_rows = []

    cur_H_run = 0
    max_H_run = 0
    for t in range(n):
        zvf_t = zvf_series[t]
        held_t = held_series[t]
        # trailing window (length up to W)
        lo = max(0, t - W + 1)
        w = zvf_series[lo:t + 1]
        # Markov H run-length
        if zvf_t > H_CUTOFF:
            cur_H_run += 1
        else:
            cur_H_run = 0
        if cur_H_run > max_H_run:
            max_H_run = cur_H_run

        rho = ar1_coef(w)
        cs = cusum_step(w, baseline_mean=baseline_mean, k=0.05)
        vr = variance_ratio(w)
        kurt = window_kurt(w)

        # Failure flag: ZVF "stuck" — trailing STUCK_W mean ZVF >= STUCK_CUTOFF.
        # This is the iter74 H-state lock-in criterion: GRPO/equivalent stuck
        # at ZVF=1.0 is the actual failure mode we want to alarm on, NOT a
        # low heldout_acc at the start of training.
        zvf_lo = max(0, t - STUCK_W + 1)
        zvf_window = zvf_series[zvf_lo:t + 1]
        zvf_mean_window = sum(zvf_window) / len(zvf_window)
        is_failure = 1 if (zvf_mean_window >= STUCK_CUTOFF and t >= STUCK_W - 1) else 0
        held_lo = max(0, t - 9)
        held_window = held_series[held_lo:t + 1]
        held_mean = sum(held_window) / len(held_window)

        out_rows.append({
            "step": t,
            "zvf": zvf_t,
            "heldout_acc": held_t,
            "held_mean_10": held_mean,
            "is_failure": is_failure,
            "h_run": cur_H_run,
            "h_run_frac": cur_H_run / W,
            "ar1": rho,
            "cusum": cs,
            "variance_ratio": vr,
            "kurtosis": kurt,
            "max_H_run": max_H_run,
        })
    return out_rows


# ----------------------- alarm + lead-time -----------------------

def composite_ews(row: dict) -> float:
    """Composite EWS = max(ar1_max, cusum, h_run_frac).

    We use union-style max: any of the three components above its
    threshold should trip the alarm. This is *or*-style; the per-channel
    EWS statistics are reported in zvf_iter78_single_channel.tsv.

    h_run_frac scaled to [0,1] like the others; ar1 clipped to [0,1] (we
    alarm on persistence, not anti-persistence). CUSUM has no upper bound
    so we additionally min-composite at 1.0.
    """
    ar1c = max(0.0, row["ar1"])
    cs = min(1.0, row["cusum"])
    hrf = min(1.0, row["h_run_frac"])
    return max(ar1c, cs, hrf)


def find_alarm(features: list[dict], threshold: float) -> int | None:
    for r in features:
        if composite_ews(r) > threshold:
            return r["step"]
    return None


def find_failure(features: list[dict]) -> int | None:
    for r in features:
        if r["is_failure"] == 1:
            return r["step"]
    return None


# ----------------------- writer -----------------------

def write_tsv(path: str, rows: list[dict], fieldnames: list[str] | None = None,
              header_comment: str = "") -> None:
    if not rows:
        with open(path, "w") as f:
            if header_comment:
                f.write(header_comment + "\n")
            if fieldnames:
                f.write("\t".join(fieldnames) + "\n")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w") as f:
        if header_comment:
            f.write(header_comment + "\n")
        f.write("\t".join(fieldnames) + "\n")
        for r in rows:
            f.write("\t".join(_fmt(r.get(k, "")) for k in fieldnames) + "\n")


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:  # NaN
            return "NA"
        if v == int(v) and abs(v) < 1e9:
            return f"{v:.6f}"
        return f"{v:.6f}"
    if v is None:
        return "NA"
    return str(v)


# ----------------------- main -----------------------

def main() -> int:
    print(f"[iter78] start {NOW}", flush=True)
    traces: list[dict] = []
    traces.extend(load_variance_mitigation())
    traces.extend(load_groupsize_sweep())
    traces.extend(load_tinker_gsm8k())
    traces.extend(load_tool_use_anchors())
    print(f"[iter78] loaded {len(traces)} traces", flush=True)

    # Compute per-step EWS features for every trace that has step data.
    enriched: list[dict] = []
    for tr in traces:
        feats = per_step_features(tr)
        if not feats:
            continue
        tr["features"] = feats
        tr["t_fail"] = find_failure(feats)
        enriched.append(tr)

    # 1) per-step features dump
    per_step_rows: list[dict] = []
    for tr in enriched:
        method = tr["method"]
        seed = tr["seed"]
        src = tr["source"]
        for r in tr["features"]:
            per_step_rows.append({
                "source": src,
                "method": method,
                "seed": seed,
                "step": r["step"],
                "zvf": r["zvf"],
                "heldout_acc": r["heldout_acc"],
                "held_mean_10": r["held_mean_10"],
                "is_failure": r["is_failure"],
                "h_run": r["h_run"],
                "ar1": r["ar1"],
                "cusum": r["cusum"],
                "variance_ratio": r["variance_ratio"],
                "kurtosis": r["kurtosis"],
                "composite_ews": composite_ews(r),
            })
    write_tsv(
        os.path.join(RESULTS, "zvf_iter78_per_step_features.tsv"),
        per_step_rows,
        fieldnames=[
            "source", "method", "seed", "step", "zvf", "heldout_acc",
            "held_mean_10", "is_failure", "h_run", "ar1", "cusum",
            "variance_ratio", "kurtosis", "composite_ews",
        ],
        header_comment=(
            "# Pillar 2 iter78 per-step EWS features.\n"
            "# W=10 trailing window; H/M/L at 0.10/0.50 (iter74 cutoffs).\n"
            "# CUSUM drift k=0.05 (Page 1954 small-shift).\n"
            "# Source: platform_modal/scripts/zvf_iter78.py"
        ),
    )
    print(f"[iter78] per_step_features: {len(per_step_rows)} rows", flush=True)

    # 2) per-threshold alarm sweep, per-library
    # We only evaluate traces that BOTH have a detected failure AND have
    # at least STUCK_W steps (so failure detection is well-defined).
    per_threshold: dict[tuple[str, str], dict] = {}
    evaluable = [tr for tr in enriched
                 if tr.get("t_fail") is not None and tr["n_steps"] >= STUCK_W]
    for tr in evaluable:
        method = tr["method"]
        for th in THRESHOLDS:
            t_alarm = find_alarm(tr["features"], th)
            key = (method, f"th{th:.2f}")
            slot = per_threshold.setdefault(key, {
                "method": method, "threshold": th, "n": 0,
                "n_alarm_before_fail": 0, "n_alarm_after_fail": 0,
                "lead_times": [], "n_late_alarm": 0, "n_miss": 0,
            })
            slot["n"] += 1
            if t_alarm is None:
                slot["n_miss"] += 1
                continue
            if t_alarm < tr["t_fail"]:
                slot["n_alarm_before_fail"] += 1
                slot["lead_times"].append(tr["t_fail"] - t_alarm)
            else:
                # alarm fires AFTER failure => late alarm (protocol was slow)
                slot["n_alarm_after_fail"] += 1
                slot["n_late_alarm"] += 1

    # Aggregate to per-(method, threshold) summary
    threshold_rows: list[dict] = []
    for (method, thkey), slot in sorted(per_threshold.items()):
        lt = slot["lead_times"]
        threshold_rows.append({
            "method": slot["method"],
            "threshold": slot["threshold"],
            "n_traces": slot["n"],
            "n_alarm": slot["n_alarm_before_fail"] + slot["n_alarm_after_fail"],
            "n_alarm_before_fail": slot["n_alarm_before_fail"],
            "n_late_alarm": slot["n_alarm_after_fail"],
            "n_miss": slot["n_miss"],
            "true_alarm_rate": round(slot["n_alarm_before_fail"] / max(1, slot["n"]), 4),
            "late_alarm_rate": round(slot["n_late_alarm"] / max(1, slot["n"]), 4),
            "miss_rate": round(slot["n_miss"] / max(1, slot["n"]), 4),
            "mean_lead_time": round(sum(lt) / len(lt), 3) if lt else "NA",
            "median_lead_time": round(statistics.median(lt), 3) if lt else "NA",
            "min_lead_time": min(lt) if lt else "NA",
            "max_lead_time": max(lt) if lt else "NA",
        })
    write_tsv(
        os.path.join(RESULTS, "zvf_iter78_alarm_thresholds.tsv"),
        threshold_rows,
        fieldnames=[
            "method", "threshold", "n_traces", "n_alarm",
            "n_alarm_before_fail", "n_late_alarm", "n_miss",
            "true_alarm_rate", "late_alarm_rate", "miss_rate",
            "mean_lead_time", "median_lead_time", "min_lead_time",
            "max_lead_time",
        ],
        header_comment=(
            "# Pillar 2 iter78 per-(method, threshold) alarm sweep.\n"
            "# t_fail = first t with mean(zvf[max(0,t-19)..t+1]) >= 0.50 (stuck).\n"
            "# EWS = max(ar1, cusum, h_run_frac). Alarm when composite > threshold.\n"
            "# Source: platform_modal/scripts/zvf_iter78.py"
        ),
    )

    # 3) Lead-time summary: pick best threshold per method (highest true_alarm_rate,
    # tie-broken by mean_lead_time). Then roll up to library level.
    best_per_method: dict[str, dict] = {}
    for row in threshold_rows:
        m = row["method"]
        prev = best_per_method.get(m)
        if prev is None:
            best_per_method[m] = row
            continue
        # Prefer: higher true_alarm_rate, then higher mean_lead_time
        if row["true_alarm_rate"] > prev["true_alarm_rate"]:
            best_per_method[m] = row
        elif (row["true_alarm_rate"] == prev["true_alarm_rate"]
              and isinstance(row["mean_lead_time"], float)
              and isinstance(prev["mean_lead_time"], float)
              and row["mean_lead_time"] > prev["mean_lead_time"]):
            best_per_method[m] = row

    leadtime_rows: list[dict] = []
    for m, row in sorted(best_per_method.items()):
        leadtime_rows.append(row)
    write_tsv(
        os.path.join(RESULTS, "zvf_iter78_leadtime_summary.tsv"),
        leadtime_rows,
        fieldnames=[
            "method", "threshold", "n_traces", "n_alarm",
            "n_alarm_before_fail", "n_late_alarm", "n_miss",
            "true_alarm_rate", "late_alarm_rate", "miss_rate",
            "mean_lead_time", "median_lead_time", "min_lead_time",
            "max_lead_time",
        ],
        header_comment=(
            "# Pillar 2 iter78 best-threshold-per-method lead-time summary.\n"
            "# Selection: maximize true_alarm_rate, then mean_lead_time.\n"
            "# Source: platform_modal/scripts/zvf_iter78.py"
        ),
    )

    # 4) Single-channel EWS: per-channel alarm rate (or-style at threshold=0.70)
    single_rows: list[dict] = []
    TH_SINGLE = 0.70
    for tr in enriched:
        if tr.get("t_fail") is None:
            continue
        method = tr["method"]
        seed = tr["seed"]
        src = tr["source"]
        # AR(1) alarm: positive persistence > TH
        ar1_alarm = next((r["step"] for r in tr["features"] if max(0.0, r["ar1"]) > TH_SINGLE), None)
        # CUSUM alarm
        cs_alarm = next((r["step"] for r in tr["features"] if r["cusum"] > TH_SINGLE), None)
        # H-run alarm
        h_alarm = next((r["step"] for r in tr["features"] if r["h_run_frac"] > TH_SINGLE), None)
        t_fail = tr["t_fail"]
        for channel, t_alarm in [("ar1", ar1_alarm), ("cusum", cs_alarm), ("h_run", h_alarm)]:
            lt = (t_fail - t_alarm) if (t_alarm is not None and t_alarm < t_fail) else "NA"
            single_rows.append({
                "source": src, "method": method, "seed": seed,
                "channel": channel, "threshold": TH_SINGLE,
                "t_alarm": t_alarm if t_alarm is not None else "NA",
                "t_fail": t_fail,
                "lead_time": lt,
                "true_alarm": int(lt != "NA"),
            })
    # Aggregate per-channel detection rate
    by_channel: dict[str, list[int]] = defaultdict(list)
    for r in single_rows:
        by_channel[r["channel"]].append(r["true_alarm"])
    for ch, lst in by_channel.items():
        det = sum(lst) / max(1, len(lst))
        print(f"[iter78] single-channel detection rate ({ch}): {det:.3f} "
              f"({sum(lst)}/{len(lst)})", flush=True)
    write_tsv(
        os.path.join(RESULTS, "zvf_iter78_single_channel.tsv"),
        single_rows,
        fieldnames=[
            "source", "method", "seed", "channel", "threshold",
            "t_alarm", "t_fail", "lead_time", "true_alarm",
        ],
        header_comment=(
            "# Pillar 2 iter78 per-channel (AR1, CUSUM, H-run) alarm at th=0.70.\n"
            "# Source: platform_modal/scripts/zvf_iter78.py"
        ),
    )

    # 5) Canonical anchors: tool_use, GRPO (failure), non-failure (tinker gsm8k)
    anchor_rows: list[dict] = []
    for tr in enriched:
        if tr["method"] in ("tool_use", "grpo", "tinker_gsm8k", "groupsize"):
            best = best_per_method.get(tr["method"])
            t_alarm = find_alarm(tr["features"], best["threshold"]) if best else None
            t_fail = tr["t_fail"]
            peak_zvf = max(s["zvf"] for s in tr["steps"])
            trough_held = min(s["heldout_acc"] for s in tr["steps"])
            last_held = tr["steps"][-1]["heldout_acc"]
            first_held = tr["steps"][0]["heldout_acc"]
            anchor_rows.append({
                "source": tr["source"],
                "method": tr["method"],
                "seed": tr["seed"],
                "n_steps": tr["n_steps"],
                "peak_zvf": round(peak_zvf, 4),
                "trough_heldout_acc": round(trough_held, 4),
                "first_heldout_acc": round(first_held, 4),
                "last_heldout_acc": round(last_held, 4),
                "t_alarm": t_alarm if t_alarm is not None else "NA",
                "t_fail": t_fail if t_fail is not None else "NA",
                "lead_time": (t_fail - t_alarm) if (t_alarm is not None and t_fail is not None and t_alarm < t_fail) else "NA",
                "alarm_threshold": best["threshold"] if best else "NA",
                "protocol_works": int(t_alarm is not None and t_fail is not None and t_alarm < t_fail),
            })
    write_tsv(
        os.path.join(RESULTS, "zvf_iter78_anchors.tsv"),
        anchor_rows,
        fieldnames=[
            "source", "method", "seed", "n_steps", "peak_zvf",
            "trough_heldout_acc", "first_heldout_acc", "last_heldout_acc",
            "t_alarm", "t_fail", "lead_time", "alarm_threshold",
            "protocol_works",
        ],
        header_comment=(
            "# Pillar 2 iter78 canonical-anchor protocol audit.\n"
            "# tool_use, grpo, tinker_gsm8k, groupsize traces.\n"
            "# protocol_works=1 iff alarm strictly precedes failure.\n"
            "# Source: platform_modal/scripts/zvf_iter78.py"
        ),
    )

    # 6) Recommended protocol: take the GLOBAL best (method-agnostic) threshold,
    # plus the per-method best, and report the recommended ZVF EWS rule.
    # Find threshold that maximizes macro-average true_alarm_rate
    by_th: dict[float, list[dict]] = defaultdict(list)
    for row in threshold_rows:
        by_th[row["threshold"]].append(row)
    th_score = []
    for th, rows in by_th.items():
        tar = sum(r["true_alarm_rate"] for r in rows) / max(1, len(rows))
        # macro miss rate
        mr = sum(r["miss_rate"] for r in rows) / max(1, len(rows))
        th_score.append((th, tar, mr, len(rows)))
    th_score.sort(key=lambda x: (-x[1], x[2]))
    recommended_threshold = th_score[0][0] if th_score else 0.70
    recommended_tar = th_score[0][1] if th_score else 0.0
    recommended_mr = th_score[0][2] if th_score else 1.0

    protocol_rows = [
        {
            "rule_id": "R1_global_threshold",
            "method": "ALL",
            "threshold": recommended_threshold,
            "true_alarm_rate": round(recommended_tar, 4),
            "miss_rate": round(recommended_mr, 4),
            "rationale": "global threshold maximizing macro true-alarm rate",
        },
        {
            "rule_id": "R2_composite_def",
            "method": "ALL",
            "threshold": "max(ar1, cusum, h_run_frac)",
            "true_alarm_rate": "NA",
            "miss_rate": "NA",
            "rationale": "union-style composite; any channel above threshold trips",
        },
        {
            "rule_id": "R3_stuck_fail_def",
            "method": "ALL",
            "threshold": f"mean(zvf[max(0,t-{STUCK_W-1})..t+1]) >= {STUCK_CUTOFF}",
            "true_alarm_rate": "NA",
            "miss_rate": "NA",
            "rationale": f"{STUCK_W}-step rolling ZVF lock-in failure criterion",
        },
        {
            "rule_id": "R4_per_method",
            "method": "see leadtime_summary",
            "threshold": "per-method",
            "true_alarm_rate": "NA",
            "miss_rate": "NA",
            "rationale": "per-method best threshold from leadtime_summary",
        },
    ]
    write_tsv(
        os.path.join(RESULTS, "zvf_iter78_ews_protocol.tsv"),
        protocol_rows,
        fieldnames=["rule_id", "method", "threshold", "true_alarm_rate",
                    "miss_rate", "rationale"],
        header_comment=(
            "# Pillar 2 iter78 recommended ZVF EWS protocol.\n"
            "# R1: global recommended threshold; R2: composite definition;\n"
            "# R3: stuck failure criterion; R4: per-method fallback.\n"
            "# Source: platform_modal/scripts/zvf_iter78.py"
        ),
    )

    # 7) Top-level summary
    n_total = sum(1 for tr in enriched)
    n_failure = sum(1 for tr in enriched if tr.get("t_fail") is not None)
    n_alarm = sum(1 for tr in enriched if find_alarm(tr["features"], recommended_threshold) is not None)
    n_protocol_works = sum(1 for r in anchor_rows if r["protocol_works"] == 1)
    summary_rows = [
        {"key": "n_traces_total", "value": n_total},
        {"key": "n_traces_with_failure", "value": n_failure},
        {"key": "n_evaluable_traces", "value": len(evaluable)},
        {"key": "n_traces_with_alarm", "value": n_alarm},
        {"key": "n_canonical_anchors", "value": len(anchor_rows)},
        {"key": "n_anchors_protocol_works", "value": n_protocol_works},
        {"key": "recommended_threshold", "value": recommended_threshold},
        {"key": "macro_true_alarm_rate", "value": round(recommended_tar, 4)},
        {"key": "macro_miss_rate", "value": round(recommended_mr, 4)},
        {"key": "W_trailing_window", "value": W},
        {"key": "H_cutoff", "value": H_CUTOFF},
        {"key": "STUCK_W", "value": STUCK_W},
        {"key": "STUCK_CUTOFF", "value": STUCK_CUTOFF},
    ]
    write_tsv(
        os.path.join(RESULTS, "zvf_iter78_summary.tsv"),
        summary_rows,
        fieldnames=["key", "value"],
        header_comment=(
            "# Pillar 2 iter78 top-level summary.\n"
            "# Source: platform_modal/scripts/zvf_iter78.py"
        ),
    )

    # 8) meta
    meta = {
        "iter": 78,
        "pillar": "P2-ZVF",
        "ts": NOW,
        "seed": SEED,
        "W": W,
        "H_cutoff": H_CUTOFF,
        "H_lo": H_LO,
        "STUCK_W": STUCK_W,
        "STUCK_CUTOFF": STUCK_CUTOFF,
        "thresholds_swept": THRESHOLDS,
        "n_traces_loaded": len(traces),
        "n_traces_with_step_data": len(enriched),
        "n_evaluable_traces": len(evaluable),
        "outputs": [
            "zvf_iter78_per_step_features.tsv",
            "zvf_iter78_alarm_thresholds.tsv",
            "zvf_iter78_leadtime_summary.tsv",
            "zvf_iter78_ews_protocol.tsv",
            "zvf_iter78_anchors.tsv",
            "zvf_iter78_single_channel.tsv",
            "zvf_iter78_summary.tsv",
        ],
    }
    with open(os.path.join(RESULTS, "zvf_iter78_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[iter78] done. {n_total} traces, {n_failure} with failure, "
          f"recommended th={recommended_threshold} tar={recommended_tar:.3f}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
