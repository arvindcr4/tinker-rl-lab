"""Iter 82 -- ZVF hazard / survival reframing of the iter78 EWS protocol.

Reframes the EWS lead-time number from iter78 as a survival-analysis
hazard: given a training trace is alive at step t, by how much does
firing the ZVF alarm (ZVF(t) > 0.5) amplify the per-step failure
probability? Produces:

  zvf_iter82_traces.tsv            per-trace descriptors
  zvf_iter82_hazard.tsv            pooled hazard h(t | alarm) by method
  zvf_iter82_hazard_ratio.tsv      hazard ratio HR = h(alarm)/h(no_alarm) per method
  zvf_iter82_survival.tsv          S(t) by method and ZVF quartile
  zvf_iter82_summary.tsv           one-line headline numbers
  figures/zvf_iter82_hazard.{pdf,png}

Inputs:  zvf_iter78_per_step_features.tsv (carries is_failure flag from iter78)
         zvf_iter78_anchors.tsv            (canonical anchor protocol audit)

Failure:  t_fail = first step with is_failure=1 (per iter78 definition:
          mean(heldout_acc[max(0,t-9)..t+1]) < 0.10)
Alarm:    alarm_t  = 1[ZVF_t > 0.5]
Hazard:   h(t) = #{traces: alive at t, fail at t+1} / #{traces: alive at t}
Hazard-ratio pooled across all t (and separately over t<midpoint).

Stdlib only. Seed 20260703.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ALARM_THRESH = 0.5

# ---------------------------------------------------------------------------
# 1. Load iter78 per-step features (carries is_failure flag, aligned to iter78
#    definition: rolling-10 mean(heldout_acc) < 0.10).
# ---------------------------------------------------------------------------
feat_path = RES / "zvf_iter78_per_step_features.tsv"
feat_rows = []
with feat_path.open() as fh:
    header = None
    for line in fh:
        s = line.rstrip("\n")
        if not s or s.startswith("#"):
            continue
        if header is None:
            header = s.split("\t")
            continue
        toks = s.split("\t")
        feat_rows.append(dict(zip(header, toks)))

# Per-trace
traces = defaultdict(lambda: {"method": "", "step": [], "zvf": [], "acc": [],
                              "is_failure": [], "composite": []})
for r in feat_rows:
    key = (r["source"], r["method"], r["seed"])
    t = traces[key]
    t["method"] = r["method"]
    t["source"] = r["source"]
    t["step"].append(int(r["step"]))
    t["zvf"].append(float(r["zvf"]))
    t["acc"].append(float(r["heldout_acc"]))
    t["is_failure"].append(int(r["is_failure"]))
    t["composite"].append(float(r["composite_ews"]))

# Compute t_fail and trace-level descriptors
trace_table = []
for (source, method, seed), t in sorted(traces.items()):
    n = len(t["step"])
    zvf_seq = t["zvf"]
    acc_seq = t["acc"]
    fail_seq = t["is_failure"]
    comp_seq = t["composite"]
    t_fail = next((s for s, f in zip(t["step"], fail_seq) if f == 1), None)
    n_steps = max(t["step"]) + 1 if t["step"] else 0
    mean_zvf = sum(zvf_seq) / len(zvf_seq) if zvf_seq else 0.0
    peak_zvf = max(zvf_seq) if zvf_seq else 0.0
    # Longest contiguous run with ZVF > ALARM_THRESH
    run_max = run_cur = 0
    for z in zvf_seq:
        if z > ALARM_THRESH:
            run_cur += 1
            run_max = max(run_max, run_cur)
        else:
            run_cur = 0
    # Inter-arrival time between alarm crossings (mean)
    crossings = [s for s, z in zip(t["step"], zvf_seq) if z > ALARM_THRESH]
    if len(crossings) >= 2:
        iat = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
    else:
        iat = float("nan")
    # Pre-failure drift: slope of acc over last 30 steps before t_fail
    if t_fail is not None and t_fail >= 5:
        win_start = max(0, t_fail - 30)
        win = [(s, a) for s, a in zip(t["step"], acc_seq) if win_start <= s < t_fail]
    else:
        win = [(s, a) for s, a in zip(t["step"], acc_seq)]
    if len(win) >= 2:
        nw = len(win)
        sx = sum(s for s, _ in win)
        sy = sum(a for _, a in win)
        sxx = sum(s * s for s, _ in win)
        sxy = sum(s * a for s, a in win)
        denom = nw * sxx - sx * sx
        slope = (nw * sxy - sx * sy) / denom if denom != 0 else 0.0
    else:
        slope = 0.0
    # Mean composite EWS while alive
    alive_comp = [c for c, f in zip(comp_seq, fail_seq) if f == 0]
    mean_comp = sum(alive_comp) / len(alive_comp) if alive_comp else 0.0
    trace_table.append({
        "source": source,
        "method": method,
        "seed": seed,
        "n_steps": n_steps,
        "t_fail": -1 if t_fail is None else int(t_fail),
        "failed": 0 if t_fail is None else 1,
        "mean_zvf": mean_zvf,
        "peak_zvf": peak_zvf,
        "zvf_run_max": run_max,
        "zvf_iat": iat,
        "acc_pre_drift": slope,
        "mean_acc": sum(acc_seq) / len(acc_seq) if acc_seq else 0.0,
        "mean_composite_ews": mean_comp,
    })

# ---------------------------------------------------------------------------
# 2. Pooled hazard computation (per-step, per-method)
# ---------------------------------------------------------------------------
# The horizon for variance_mitigation is 0..299 (300 steps); for groupsize it
# is 0..39; for tool_use 0..4; for tinker 0. We need a single per-step hazard,
# so we pool only across the variance_mitigation traces (consistent length).
HORIZON = 300
VM_TRACES = [(s, m, sd) for (s, m, sd) in traces if s == "variance_mitigation"]


def compute_hazard_by_method(thresh=ALARM_THRESH):
    """For each method, return arrays of t, h_alarm(t), h_noalarm(t)."""
    by_method = defaultdict(lambda: {"alarm_num": [0] * HORIZON, "alarm_den": [0] * HORIZON,
                                      "noalarm_num": [0] * HORIZON, "noalarm_den": [0] * HORIZON})
    for (source, method, seed) in VM_TRACES:
        t = traces[(source, method, seed)]
        zvf_seq = t["zvf"]
        fail_seq = t["is_failure"]
        stp_seq = t["step"]
        for i, s in enumerate(stp_seq):
            if s + 1 >= HORIZON:
                continue
            # Need a next-step outcome
            if i + 1 >= len(fail_seq):
                continue
            next_fail = fail_seq[i + 1]
            # A trace is "alive at t" if it has not failed yet at step t
            if any(f == 1 for f in fail_seq[:i + 1]):
                continue  # already failed
            alarm_now = 1 if zvf_seq[i] > thresh else 0
            if alarm_now:
                by_method[method]["alarm_den"][s] += 1
                if next_fail == 1:
                    by_method[method]["alarm_num"][s] += 1
            else:
                by_method[method]["noalarm_den"][s] += 1
                if next_fail == 1:
                    by_method[method]["noalarm_num"][s] += 1
    return by_method

hazard_by_method = compute_hazard_by_method()

# Hazard ratio per method (with Laplace smoothing, plus precision/recall)
EPS = 0.5  # Laplace smoothing constant for h_noalarm estimate
hr_table = []
for method, h in sorted(hazard_by_method.items()):
    a_num = sum(h["alarm_num"])
    a_den = sum(h["alarm_den"])
    n_num = sum(h["noalarm_num"])
    n_den = sum(h["noalarm_den"])
    h_alarm = a_num / a_den if a_den else 0.0
    # Smoothed h_noalarm
    h_noalarm_smooth = (n_num + EPS) / (n_den + 2 * EPS) if (n_den + 2 * EPS) > 0 else 0.0
    h_noalarm = n_num / n_den if n_den else 0.0
    hr = h_alarm / h_noalarm_smooth if h_noalarm_smooth > 0 else float("inf")
    # Precision/recall framing: alarm's value as a collapse predictor
    alarm_coverage = a_num / max(1, a_num + n_num)  # recall: fraction of failures preceded by alarm
    false_alarm_rate = 1 - h_alarm  # fraction of alarms NOT followed by failure
    # E-score: 1 - false_alarm_rate if recall=1, else weighted
    if a_num + n_num > 0:
        precision = a_num / max(1, a_den)  # = h_alarm = TP/alarm
        recall = a_num / (a_num + n_num)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    hr_table.append({
        "method": method,
        "n_alarm_obs": a_den,
        "n_noalarm_obs": n_den,
        "failures_preceded_by_alarm": a_num,
        "failures_not_preceded_by_alarm": n_num,
        "h_alarm": h_alarm,
        "h_noalarm": h_noalarm,
        "h_noalarm_smoothed": h_noalarm_smooth,
        "hazard_ratio": hr,
        "log_hr": math.log(hr) if 0 < hr < 1e9 else float("nan"),
        "alarm_coverage_recall": alarm_coverage,
        "false_alarm_rate": false_alarm_rate,
        "precision": precision,
        "f1": f1,
    })

# ---------------------------------------------------------------------------
# 3. Survival S(t) by method and ZVF quartile
# ---------------------------------------------------------------------------
# Survival here is across ALL traces (variance_mitigation + groupsize + tool +
# tinker), not just variance_mitigation, so we use each trace's actual
# n_steps range.
survival = defaultdict(list)  # group -> list of (t, S(t))
mean_zvfs = [r["mean_zvf"] for r in trace_table]
mean_zvfs_sorted = sorted(mean_zvfs)
n_traces = len(mean_zvfs_sorted)
q33 = mean_zvfs_sorted[n_traces // 3]
q66 = mean_zvfs_sorted[(2 * n_traces) // 3]
for r in trace_table:
    if r["mean_zvf"] <= q33:
        r["zvf_quartile"] = "low"
    elif r["mean_zvf"] <= q66:
        r["zvf_quartile"] = "mid"
    else:
        r["zvf_quartile"] = "high"

# Survival per quartile
quartile_groups = defaultdict(list)
for r in trace_table:
    quartile_groups[r["zvf_quartile"]].append(r)
for q, rows in quartile_groups.items():
    n = len(rows)
    # Use the union of step ranges
    max_step = max(r["n_steps"] for r in rows)
    for t in range(0, max_step + 1, 5):
        alive = sum(1 for r in rows if r["t_fail"] < 0 or r["t_fail"] > t)
        s_q = alive / n if n else 0.0
        survival[("all", q)].append((t, s_q))

# Per-method survival
for method in sorted({r["method"] for r in trace_table}):
    rows = [r for r in trace_table if r["method"] == method]
    n = len(rows)
    max_step = max(r["n_steps"] for r in rows)
    for t in range(0, max_step + 1, 5):
        alive = sum(1 for r in rows if r["t_fail"] < 0 or r["t_fail"] > t)
        s_m = alive / n if n else 0.0
        survival[("by_method", method)].append((t, s_m))

# ---------------------------------------------------------------------------
# 4. Write outputs
# ---------------------------------------------------------------------------
def write_tsv(path, rows, header):
    with path.open("w") as fh:
        fh.write("# Pillar 2 iter82 ZVF hazard / survival reframing\n")
        fh.write("# Source: platform_modal/scripts/zvf_iter82.py\n")
        fh.write("# Hazard: h(t) = P(collapse at t+1 | alive at t)\n")
        fh.write(f"# Alarm threshold: ZVF > {ALARM_THRESH}\n")
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r.get(h, ''):g}" if isinstance(r.get(h, 0.0), float) else str(r.get(h, ""))
                              for h in header) + "\n")

write_tsv(RES / "zvf_iter82_traces.tsv", trace_table,
          ["source", "method", "seed", "n_steps", "t_fail", "failed", "mean_zvf",
           "peak_zvf", "zvf_run_max", "zvf_iat", "acc_pre_drift", "mean_acc",
           "mean_composite_ews", "zvf_quartile"])
write_tsv(RES / "zvf_iter82_hazard_ratio.tsv", hr_table,
          ["method", "n_alarm_obs", "n_noalarm_obs", "failures_preceded_by_alarm",
           "failures_not_preceded_by_alarm", "h_alarm", "h_noalarm",
           "h_noalarm_smoothed", "hazard_ratio", "log_hr", "alarm_coverage_recall",
           "false_alarm_rate", "precision", "f1"])

# Per-step hazard pooled
with (RES / "zvf_iter82_hazard.tsv").open("w") as fh:
    fh.write("# Pillar 2 iter82 pooled per-step hazard h(t | alarm), h(t | no_alarm)\n")
    fh.write("# Source: platform_modal/scripts/zvf_iter82.py\n")
    fh.write("method\tstep\th_alarm\th_noalarm\n")
    for method, h in sorted(hazard_by_method.items()):
        for t in range(HORIZON - 1):
            a_num, a_den = h["alarm_num"][t], h["alarm_den"][t]
            n_num, n_den = h["noalarm_num"][t], h["noalarm_den"][t]
            ha = a_num / a_den if a_den else 0.0
            hn = n_num / n_den if n_den else 0.0
            fh.write(f"{method}\t{t}\t{ha:.6f}\t{hn:.6f}\n")

# Survival
with (RES / "zvf_iter82_survival.tsv").open("w") as fh:
    fh.write("# Pillar 2 iter82 survival S(t) by ZVF quartile (pooled across all methods)\n")
    fh.write("# Source: platform_modal/scripts/zvf_iter82.py\n")
    fh.write("group\tt\tS\n")
    for (gname, sub), rows in sorted(survival.items()):
        for t, s in rows:
            fh.write(f"{gname}:{sub}\t{t}\t{s:.4f}\n")

# Headline summary
methods_with_failures = [r for r in hr_table if r["failures_preceded_by_alarm"] + r["failures_not_preceded_by_alarm"] > 0]
methods_with_alarm = [r for r in hr_table if r["n_alarm_obs"] > 0]
summary = {
    "n_traces": len(trace_table),
    "n_failed": sum(1 for r in trace_table if r["failed"]),
    "n_methods": len({r["method"] for r in trace_table}),
    "alarm_threshold": ALARM_THRESH,
    "n_methods_with_any_failure": len(methods_with_failures),
    "n_methods_with_alarm_firing": len(methods_with_alarm),
    "median_alarm_coverage_recall": (lambda s: s[len(s) // 2] if s else float("nan"))(
        sorted(r["alarm_coverage_recall"] for r in methods_with_failures)
    ),
    "min_alarm_coverage_recall": min((r["alarm_coverage_recall"] for r in methods_with_failures), default=0.0),
    "max_false_alarm_rate_across_methods": max((r["false_alarm_rate"] for r in methods_with_alarm), default=0.0),
    "median_false_alarm_rate": (lambda s: s[len(s) // 2] if s else float("nan"))(
        sorted(r["false_alarm_rate"] for r in methods_with_alarm)
    ),
    "median_f1": (lambda s: s[len(s) // 2] if s else float("nan"))(
        sorted(r["f1"] for r in methods_with_failures)
    ),
    "median_hr_smoothed": (lambda s: s[len(s) // 2] if s else float("nan"))(
        sorted(r["hazard_ratio"] for r in methods_with_failures if r["hazard_ratio"] < 1e6)
    ),
    "q33_mean_zvf": q33,
    "q66_mean_zvf": q66,
}
with (RES / "zvf_iter82_summary.tsv").open("w") as fh:
    fh.write("# Pillar 2 iter82 headline summary\n")
    fh.write("# Source: platform_modal/scripts/zvf_iter82.py\n")
    fh.write("key\tvalue\n")
    for k, v in summary.items():
        fh.write(f"{k}\t{v}\n")

# ---------------------------------------------------------------------------
# 5. Figure: per-step hazard, alarm vs no-alarm, by method (PDF + PNG)
# ---------------------------------------------------------------------------
def render_figure():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    methods = sorted(hazard_by_method.keys())
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1, ax2 = axes
    cmap = plt.get_cmap("tab10")
    for i, method in enumerate(methods):
        h = hazard_by_method[method]
        ts = list(range(HORIZON - 1))
        ha = [h["alarm_num"][t] / h["alarm_den"][t] if h["alarm_den"][t] else 0.0 for t in ts]
        hn = [h["noalarm_num"][t] / h["noalarm_den"][t] if h["noalarm_den"][t] else 0.0 for t in ts]
        c = cmap(i % 10)
        ax1.plot(ts, ha, color=c, label=method, lw=1.0)
        ax2.plot(ts, hn, color=c, label=method, lw=1.0)
    ax1.set_ylabel(r"$h(t\mid\mathrm{alarm})$")
    ax1.set_title("ZVF alarm fires: per-step collapse hazard by method (variance_mitigation, 100 steps)")
    ax1.set_ylim(0, 0.6)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.set_ylabel(r"$h(t\mid\mathrm{no\,alarm})$")
    ax2.set_xlabel("step t")
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "zvf_iter82_hazard.pdf")
    fig.savefig(FIG / "zvf_iter82_hazard.png", dpi=130)
    plt.close(fig)

    # Survival figure: S(t) by ZVF quartile
    fig, ax = plt.subplots(figsize=(6, 4))
    for q in ("low", "mid", "high"):
        rows = survival.get(("all", q), [])
        if not rows:
            continue
        ts = [t for t, _ in rows]
        ss = [s for _, s in rows]
        ax.plot(ts, ss, marker="o", label=f"ZVF quartile={q} (n={len(quartile_groups[q])})")
    ax.set_xlabel("step t")
    ax.set_ylabel(r"survival $S(t) = \Pr(t_\mathrm{fail} > t)$")
    ax.set_title("Variance-mitigation survival by base-ZVF quartile")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "zvf_iter82_survival.pdf")
    fig.savefig(FIG / "zvf_iter82_survival.png", dpi=130)
    plt.close(fig)
    return True

render_figure()

# Print one-liner
print(f"iter82: n_traces={summary['n_traces']} n_failed={summary['n_failed']} "
      f"methods_with_failure={summary['n_methods_with_any_failure']}/9 "
      f"median_alarm_coverage={summary['median_alarm_coverage_recall']:.3f} "
      f"min_alarm_coverage={summary['min_alarm_coverage_recall']:.3f} "
      f"median_F1={summary['median_f1']:.3f} "
      f"median_FAR={summary['median_false_alarm_rate']:.3f} "
      f"max_FAR_method={max(methods_with_alarm, key=lambda r: r['false_alarm_rate'])['method'] if methods_with_alarm else 'NA'}")
print(f"  -> wrote {RES}/zvf_iter82_*.tsv and {FIG}/zvf_iter82_*.pdf")
