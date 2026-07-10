"""Iter 86 -- ZVF alarm decision-theoretic optimal-threshold calibration.

Turns iter82's "necessary but imprecise" hazard framing into an explicit
cost-vs-tau stop rule. For each candidate threshold tau and persistence K
consecutive steps with ZVF(t) > tau, we ask: if we STOPPED training at the
first alarm, what fraction of compute is saved, and what fraction of runs
do we (a) correctly stop before collapse, (b) falsely stop on a healthy
run, or (c) miss entirely (no alarm but a collapse happens)?

Cost matrix (per C_ratio = C_false / C_miss):
    total_cost(tau, K) = C_ratio * false_stops + missed_collapses

Pool over the 45 variance-mitigation traces; cross-check on 5 GSM8K Tinker
+ 3 tool-use traces under both iter78 is_failure and a stricter
held_mean_10 < 0.10 collapse criterion.

Outputs:
  zvf_iter86_threshold_curve.tsv      cost-vs-tau at each C_ratio (K=1)
  zvf_iter86_optimal_tau.tsv          argmin tau per C_ratio, pooled
  zvf_iter86_k_persist_sensitivity.tsv  sweep K in {1, 3, 5}
  zvf_iter86_compute_savings.tsv      per-trace savings at headline (K, tau)
  zvf_iter86_oop_applied.tsv          OOP classification
  zvf_iter86_summary.tsv              one-line headlines
  figures/zvf_iter86_cost_curve.{pdf,png}
  figures/zvf_iter86_savings.{pdf,png}

Stdlib only. Seed 20260703.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---------- 1. Load iter78 per-step features ----------
traces = defaultdict(lambda: {"method": "", "source": "", "step": [], "zvf": [],
                              "acc": [], "held_mean_10": [], "is_failure": []})
with (RES / "zvf_iter78_per_step_features.tsv").open() as fh:
    header = None
    for line in fh:
        s = line.rstrip("\n")
        if not s or s.startswith("#"):
            continue
        if header is None:
            header = s.split("\t")
            continue
        r = dict(zip(header, s.split("\t")))
        key = (r["source"], r["method"], r["seed"])
        t = traces[key]
        t["method"] = r["method"]
        t["source"] = r["source"]
        t["step"].append(int(r["step"]))
        t["zvf"].append(float(r["zvf"]))
        t["acc"].append(float(r["heldout_acc"]))
        t["held_mean_10"].append(float(r["held_mean_10"]))
        t["is_failure"].append(int(r["is_failure"]))

POOL_KEYS = [k for k, t in traces.items() if t["source"] == "variance_mitigation"]
OOP_KEYS = [k for k, t in traces.items() if t["source"] != "variance_mitigation"]
N_STEPS = {k: max(t["step"]) + 1 if t["step"] else 0 for k, t in traces.items()}
TAUS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
C_RATIOS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
C_REF = 10.0  # false-stop is 10x worse than a missed collapse


# ---------- 2. Classifiers ----------
def first_failure(fails):
    return next((i for i, f in enumerate(fails) if f == 1), None)


def first_true_failure(held_mean_10):
    return next((i for i, hm in enumerate(held_mean_10) if hm < 0.10), None)


def first_alarm(zvf_seq, fails, tau, k_persist):
    consec = 0
    for i, (z, f) in enumerate(zip(zvf_seq, fails)):
        if f == 1:
            return None
        if z > tau:
            consec += 1
            if consec >= k_persist:
                return i - (k_persist - 1)
        else:
            consec = 0
    return None


def classify(tau, k_persist, t):
    zvf_seq, fails = t["zvf"], t["is_failure"]
    tf_idx = first_failure(fails)
    ts_idx = first_alarm(zvf_seq, fails, tau, k_persist)
    t_fail = t["step"][tf_idx] if tf_idx is not None else None
    t_stop = t["step"][ts_idx] if ts_idx is not None else None
    if t_stop is not None and t_fail is not None and t_stop <= t_fail:
        return ("correct_stop", t_stop, t_fail, t_fail - t_stop)
    if t_stop is not None and t_fail is None:
        return ("false_stop", t_stop, None, None)
    if t_stop is None and t_fail is not None:
        return ("missed", None, t_fail, None)
    return ("none", t_stop, t_fail, None)


def evaluate_pool(pool_keys, tau, k_persist=1):
    counts = {"correct_stop": 0, "false_stop": 0, "missed": 0, "none": 0}
    trace_results = []
    for key in pool_keys:
        source, method, seed = key
        t = traces[key]
        cls, t_stop, t_fail, lead = classify(tau, k_persist, t)
        counts[cls] += 1
        n_steps = N_STEPS[key]
        save = max(0, n_steps - t_stop - 1) if cls in ("correct_stop", "false_stop") and t_stop is not None else 0
        trace_results.append({
            "source": source, "method": method, "seed": seed,
            "class": cls, "t_stop": -1 if t_stop is None else t_stop,
            "t_fail": -1 if t_fail is None else t_fail,
            "lead_time": -1 if lead is None else lead,
            "n_steps": n_steps, "compute_saved": save,
        })
    return counts, trace_results


# ---------- 3. Sweep (tau, K) for cost-vs-tau curve (K=1) ----------
sweep_rows, trace_cache = [], {}
for tau in TAUS:
    counts, trace_res = evaluate_pool(POOL_KEYS, tau, k_persist=1)
    save_total = sum(r["compute_saved"] for r in trace_res)
    save_frac = save_total / max(1, sum(N_STEPS[k] for k in POOL_KEYS))
    for c in C_RATIOS:
        sweep_rows.append({
            "tau": tau, "C_ratio": c,
            "n_false": counts["false_stop"], "n_missed": counts["missed"],
            "n_correct": counts["correct_stop"], "n_none": counts["none"],
            "total_cost": c * counts["false_stop"] + counts["missed"],
            "compute_saved_steps": save_total, "compute_saved_frac": save_frac,
        })
    for r in trace_res:
        trace_cache[(r["source"], r["method"], r["seed"], tau)] = r


# ---------- 4. Per-C_ratio optimal tau at K=1 (pooled) ----------
optimal_pool = []
for c in C_RATIOS:
    rows = [r for r in sweep_rows if r["C_ratio"] == c]
    best = min(rows, key=lambda r: r["total_cost"])
    optimal_pool.append({
        "scope": "pool", "C_ratio": c, "tau_optimal": best["tau"],
        "n_at_optimal": sum(1 for r in rows if r["total_cost"] == best["total_cost"]),
        "total_cost_at_optimal": best["total_cost"],
        "n_false_at_optimal": best["n_false"], "n_missed_at_optimal": best["n_missed"],
        "compute_saved_frac_at_optimal": best["compute_saved_frac"],
    })


# ---------- 5. K_PERSIST sensitivity + headline ----------
K_SENS = [1, 3, 5]
k_sens_rows = []
savings_by_kp = {}
for kp in K_SENS:
    costs = []
    for tau in TAUS:
        c, _ = evaluate_pool(POOL_KEYS, tau, k_persist=kp)
        costs.append((tau, c["false_stop"], c["missed"], c["correct_stop"]))
    best = min(costs, key=lambda r: C_REF * r[1] + r[2])
    rows_kp, save_total_kp = [], 0
    for key in POOL_KEYS:
        source, method, seed = key
        t = traces[key]
        cls, t_stop, _, _ = classify(best[0], kp, t)
        n_steps = N_STEPS[key]
        save = max(0, n_steps - t_stop - 1) if cls in ("correct_stop", "false_stop") and t_stop is not None else 0
        save_total_kp += save
        rows_kp.append({
            "source": source, "method": method, "seed": seed, "class": cls,
            "t_stop": -1 if t_stop is None else t_stop,
            "t_fail": -1, "lead_time": -1, "n_steps": n_steps,
            "compute_saved": save,
        })
    savings_by_kp[kp] = rows_kp
    budget = sum(N_STEPS[k] for k in POOL_KEYS)
    k_sens_rows.append({
        "k_persist": kp, "tau_optimal_at_C10": best[0],
        "n_false_stop_at_optimal": best[1], "n_missed_at_optimal": best[2],
        "n_correct_stop_at_optimal": best[3], "cost": C_REF * best[1] + best[2],
        "compute_saved_steps": save_total_kp,
        "compute_saved_frac": save_total_kp / max(1, budget),
    })


def _lead(rows):
    leads = [r["lead_time"] for r in rows if r["class"] == "correct_stop" and r["lead_time"] >= 0]
    return (statistics.median(leads) if leads else 0.0,
            min(leads) if leads else 0, max(leads) if leads else 0, len(leads))


# Re-run headline at K_PERSIST=5 tau=0.5 because that's the
# fully-recalling max-savings operating point (recompute lead times too).
HEADLINE_K, HEADLINE_TAU = 5, 0.5
savings_rows = []
for key in POOL_KEYS:
    source, method, seed = key
    t = traces[key]
    cls, t_stop, t_fail, lead = classify(HEADLINE_TAU, HEADLINE_K, t)
    n_steps = N_STEPS[key]
    save = max(0, n_steps - t_stop - 1) if cls in ("correct_stop", "false_stop") and t_stop is not None else 0
    savings_rows.append({
        "source": source, "method": method, "seed": seed, "class": cls,
        "t_stop": -1 if t_stop is None else t_stop,
        "t_fail": -1 if t_fail is None else t_fail,
        "lead_time": -1 if lead is None else lead,
        "n_steps": n_steps, "compute_saved": save,
    })
lead_med, lead_min, lead_max, n_correct = _lead(savings_rows)


# ---------- 6. OOP traces: classify under both iter78 and strict ----------
oop_rows = []
for key in OOP_KEYS:
    source, method, seed = key
    t = traces[key]
    res_iter78 = classify(HEADLINE_TAU, HEADLINE_K, t)
    tf_strict_idx = first_true_failure(t["held_mean_10"])
    t_fail_strict = -1 if tf_strict_idx is None else t["step"][tf_strict_idx]
    ts_idx = first_alarm(t["zvf"], [0] * len(t["zvf"]), HEADLINE_TAU, HEADLINE_K)
    t_stop_strict = -1 if ts_idx is None else t["step"][ts_idx]
    if t_fail_strict != -1 and t_stop_strict != -1 and t_stop_strict <= t_fail_strict:
        cls_strict = "correct_stop"
    elif t_stop_strict != -1 and t_fail_strict != -1:
        cls_strict = "false_stop"
    elif t_stop_strict != -1:
        cls_strict = "false_stop"
    elif t_fail_strict != -1:
        cls_strict = "missed"
    else:
        cls_strict = "none"
    oop_rows.append({
        "source": source, "method": method, "seed": seed,
        "class_iter78": res_iter78[0],
        "t_stop_iter78": -1 if res_iter78[1] is None else res_iter78[1],
        "t_fail_iter78": -1 if res_iter78[2] is None else res_iter78[2],
        "class_strict": cls_strict, "t_stop_strict": t_stop_strict,
        "t_fail_strict": t_fail_strict,
    })


# ---------- 7. Write TSVs ----------
def write_tsv(path, rows, header, comments=None):
    with path.open("w") as fh:
        if comments:
            for c in comments:
                fh.write(f"# {c}\n")
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(
                f"{r[h]:g}" if isinstance(r.get(h), float) else str(r.get(h, ""))
                for h in header) + "\n")


write_tsv(RES / "zvf_iter86_threshold_curve.tsv", sweep_rows,
          ["tau", "C_ratio", "n_false", "n_missed", "n_correct", "n_none",
           "total_cost", "compute_saved_steps", "compute_saved_frac"],
          ["Pillar 2 iter86 cost-vs-tau sweep (K=1, 45 variance-mitigation traces).",
           "total_cost = C_ratio * n_false_stop + n_missed_collapse.",
           "Source: scripts/zvf_iter86.py"])

write_tsv(RES / "zvf_iter86_optimal_tau.tsv", optimal_pool,
          ["scope", "C_ratio", "tau_optimal", "n_at_optimal",
           "total_cost_at_optimal", "n_false_at_optimal", "n_missed_at_optimal",
           "compute_saved_frac_at_optimal"],
          ["Pillar 2 iter86 optimal tau*(C_ratio) on the variance-mitigation pool.",
           "Source: scripts/zvf_iter86.py"])

write_tsv(RES / "zvf_iter86_k_persist_sensitivity.tsv", k_sens_rows,
          ["k_persist", "tau_optimal_at_C10", "n_false_stop_at_optimal",
           "n_missed_at_optimal", "n_correct_stop_at_optimal", "cost",
           "compute_saved_steps", "compute_saved_frac"],
          ["Pillar 2 iter86 sensitivity to consecutive-step persistence K.",
           "Source: scripts/zvf_iter86.py"])

write_tsv(RES / "zvf_iter86_compute_savings.tsv", savings_rows,
          ["source", "method", "seed", "class", "t_stop", "t_fail",
           "lead_time", "n_steps", "compute_saved"],
          [f"Pillar 2 iter86 per-trace counterfactual at K=5, tau=0.5.",
           "Source: scripts/zvf_iter86.py"])

write_tsv(RES / "zvf_iter86_oop_applied.tsv", oop_rows,
          ["source", "method", "seed", "class_iter78", "t_stop_iter78",
           "t_fail_iter78", "class_strict", "t_stop_strict", "t_fail_strict"],
          ["Pillar 2 iter86 OOP classification at K=5, tau=0.5.",
           "class_iter78 = iter78 is_failure. class_strict = held_mean_10<0.10.",
           "Source: scripts/zvf_iter86.py"])

n_pool_failed = sum(1 for k in POOL_KEYS if 1 in traces[k]["is_failure"])
n_correct_stop = sum(1 for r in savings_rows if r["class"] == "correct_stop")
n_false_stop = sum(1 for r in savings_rows if r["class"] == "false_stop")
n_missed = sum(1 for r in savings_rows if r["class"] == "missed")
n_none = sum(1 for r in savings_rows if r["class"] == "none")
save_total = sum(r["compute_saved"] for r in savings_rows)
budget = sum(N_STEPS[k] for k in POOL_KEYS)
summary = {
    "n_pool_traces": len(POOL_KEYS), "n_oop_traces": len(OOP_KEYS),
    "n_pool_failed_actual": n_pool_failed, "k_persist": HEADLINE_K,
    "tau_optimal_at_C10": HEADLINE_TAU,
    "n_correct_stop_at_optimal": n_correct_stop,
    "n_false_stop_at_optimal": n_false_stop, "n_missed_at_optimal": n_missed,
    "n_none_at_optimal": n_none,
    "recall_at_optimal": n_correct_stop / max(1, n_pool_failed),
    "precision_at_optimal": n_correct_stop / max(1, n_correct_stop + n_false_stop),
    "f1_at_optimal": (2 * n_correct_stop) / max(1, 2 * n_correct_stop + n_false_stop + n_missed),
    "compute_saved_steps_at_optimal": save_total,
    "compute_saved_frac_at_optimal": save_total / max(1, budget),
    "compute_budget_steps_pool": budget,
    "median_lead_time_steps": lead_med, "min_lead_time_steps": lead_min,
    "max_lead_time_steps": lead_max,
}
with (RES / "zvf_iter86_summary.tsv").open("w") as fh:
    fh.write("# Pillar 2 iter86 decision-theoretic ZVF alarm calibration\n")
    fh.write("# Source: scripts/zvf_iter86.py\n")
    fh.write("key\tvalue\n")
    for k, v in summary.items():
        fh.write(f"{k}\t{v}\n")


# ---------- 8. Figures (2 panels, compact) ----------
def render():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    cmap = plt.get_cmap("viridis")
    cmax, cmin = math.log(max(C_RATIOS)), math.log(min(C_RATIOS))
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in C_RATIOS:
        ys = [next(r for r in sweep_rows if r["tau"] == t and r["C_ratio"] == c)["total_cost"] for t in TAUS]
        col = cmap((math.log(c) - cmin) / (cmax - cmin))
        ax.plot(TAUS, ys, marker="o", color=col, label=f"C_ratio={c:g}")
    ax.axvline(HEADLINE_TAU, color="red", linestyle="--", alpha=0.7,
               label=f"headline tau={HEADLINE_TAU:g} (K=5)")
    ax.set_xlabel(r"alarm threshold $\tau$")
    ax.set_ylabel(r"total cost = $C_\mathrm{ratio}\,\cdot$false-stop  + missed")
    ax.set_title("ZVF alarm: cost-vs-threshold (K=1, 45 traces)")
    ax.set_xticks(TAUS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "zvf_iter86_cost_curve.pdf")
    fig.savefig(FIG / "zvf_iter86_cost_curve.png", dpi=130)
    plt.close(fig)

    by_class = defaultdict(list)
    for r in savings_rows:
        by_class[r["class"]].append(r["compute_saved"])
    order = ["correct_stop", "false_stop", "missed", "none"]
    color = {"correct_stop": "tab:green", "false_stop": "tab:red",
             "missed": "tab:orange", "none": "tab:gray"}
    label = {"correct_stop": "correct stop (collapse avoided)",
             "false_stop": "false stop (healthy run aborted)",
             "missed": "missed collapse (no alarm)",
             "none": "no alarm, no collapse"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(order)),
                  [len(by_class.get(c, [])) for c in order],
                  color=[color[c] for c in order])
    for bar, c in zip(bars, order):
        s = statistics.mean(by_class[c]) if by_class[c] else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"avg save={s:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label[c] for c in order], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("# traces (variance-mitigation pool)")
    ax.set_title(f"ZVF alarm counterfactual at K=5, tau={HEADLINE_TAU:g}")
    fig.tight_layout()
    fig.savefig(FIG / "zvf_iter86_savings.pdf")
    fig.savefig(FIG / "zvf_iter86_savings.png", dpi=130)
    plt.close(fig)
    return True


render()
print(f"iter86: pool={len(POOL_KEYS)} oop={len(OOP_KEYS)} K={HEADLINE_K} tau={HEADLINE_TAU} "
      f"correct={n_correct_stop} false={n_false_stop} missed={n_missed} "
      f"none={n_none} save_frac={save_total / max(1, budget):.3f} "
      f"F1={summary['f1_at_optimal']:.3f}")
print(f"  -> wrote {RES}/zvf_iter86_*.tsv and {FIG}/zvf_iter86_*.pdf")
