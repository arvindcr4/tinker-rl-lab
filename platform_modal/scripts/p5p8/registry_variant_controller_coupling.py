#!/usr/bin/env python3
"""Iter 22 P6 (Pillar 2) — registry↔controller variant coupling.

For each variant in registry/entries/delta_*.json that has zvf130 evidence,
compute:

  1. Per-seed ZVF trajectory from platform_hybrid/experiments/results/zvf_iter130_risk_index.tsv
     (9 methods × 5 seeds = 45 rows; the canonical P7 risk-index panel).
  2. Effect size vs grpo baseline (Cohen's d on mean_zvf across seeds).
  3. P7 controller firing rate: at each τ in {0.10..0.90}, the fraction of
     seeds whose risk_index_score zvf_risk ≥ (1-τ) — equivalent to
     "zvf-fell-below-τ somewhere" under the iter-15 (5-seed, mean_zvf)
     threshold convention.
  4. Predicted direction from the iter-18 CLAIMS dict (registry_measured_claimed.py).
  5. Reconciliation verdict: SUPPORT / WEAK / OPPOSE / NO_DATA, in the
     same idiom as iter 18 but with the *firing-rate* (a P7-native
     number) instead of the raw ZVF magnitude.

Writes:
  platform_hybrid/experiments/results/p5p8/registry_variant_coupling.tsv   — one row per
    (delta_id × τ) pair, plus a per-delta effect-size row.
  platform_hybrid/experiments/results/p5p8/registry_variant_coupling.json  — machine-readable.
  platform_hybrid/experiments/results/p5p8/figures/registry_variant_coupling.png — per-variant
    firing-rate curves overlaid on the grpo baseline.

Stdlib only (+ matplotlib). Run: python3 platform_modal/scripts/p5p8/registry_variant_controller_coupling.py
"""

import csv
import json
import pathlib
import statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
REG_ENTRIES = ROOT / "registry" / "entries"
ZV130_TSV = ROOT / "experiments" / "results" / "zvf_iter130_risk_index.tsv"
OUT_TSV = ROOT / "experiments" / "results" / "p5p8" / "registry_variant_coupling.tsv"
OUT_JSON = ROOT / "experiments" / "results" / "p5p8" / "registry_variant_coupling.json"
FIG_PNG = ROOT / "experiments" / "results" / "p5p8" / "figures" / "registry_variant_coupling.png"

# Predicted direction per delta (from iter-18 CLAIMS dict, ZVF sub-key).
PREDICTED_ZVF_SIGN = {
    "delta_aero":     -1,   # "inflate effective G -> fewer zero-variance groups (predicted)"
    "delta_gift":      0,   # "no claim on ZVF; constant offset cancels in std"
    "delta_areal":     0,   # single-batch static-G run: no rollout-vs-optimizer signal
    "delta_dapo":     -1,   # "dynamic sampling zeroes degenerate groups -> lower ZVF"
    "delta_drgrpo":   +1,   # "removing length norm exposes length bias -> more within-group variance"
    "delta_gspo":      0,   # "ratio level doesn't directly change reward-variance structure"
    "delta_cppo":      0,   # "no claim on ZVF; log-prob smoothness doesn't directly change within-group contrast"
    "delta_ngrpo":     0,   # "per-prompt norm re-weights but doesn't change zero-variance count"
    "delta_mcgrpo":    0,   # "MCTS boost could go either way"
    "delta_es":        0,   # "ES doesn't use within-group contrast -> ZVF irrelevant by construction"
    "delta_scafgrpo":  0,   # "no direct claim on ZVF; re-weighting could go either way"
}

# delta_id -> method label used in zvf130 panel.
METHOD_OF = {
    "delta_aero":     "aero",
    "delta_gift":     "gift",
    "delta_areal":    "areal",
    "delta_dapo":     "dapo",
    "delta_drgrpo":   "drgrpo",
    "delta_gspo":     "gspo",
    "delta_cppo":     "cppo",
    "delta_ngrpo":    "ngrpo",
    "delta_mcgrpo":   "mcgrpo",
    "delta_es":       "es",
    "delta_scafgrpo": "scafgrpo",
}

# Threshold grid the P7 controller sweeps over.
TAU_GRID = [round(0.05 * i, 2) for i in range(2, 19)]  # 0.10..0.90 step 0.05
# Specifically requested headline thresholds:
HEADLINE_TAUS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

EXCLUDE_METHODS = {
    "scaling_law_Qwen3.5-4B",
    "scaling_law_Llama-3.1-8B-Instruct",
    "scaling_law_DeepSeek-V3.1",
    "scaling_law_Nemotron-120B",
    "scaling_law_Qwen3-8B",
    "tool_use_qwen3-32b",
    "tool_use_llama-8b-inst",
}


def load_zv130_panel():
    """group risk-index rows by method; keep only the 9 measured variants + grpo."""
    grp = defaultdict(list)
    with ZV130_TSV.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["method"] in EXCLUDE_METHODS:
                continue
            try:
                grp[row["method"]].append({
                    "seed": int(row["seed"]),
                    "mean_zvf": float(row["mean_zvf"]),
                    "zvf_risk": float(row["zvf_risk"]),
                    "failure_label": row["failure_label"],
                })
            except (ValueError, KeyError):
                pass
    return dict(grp)


def effect_size_zvf(variant_zvf, grpo_zvf):
    """Effect size on mean_zvf, normalised by grpo's seed-to-seed SD.

    The zvf130 5-seed panel has near-zero seed variance (e.g. grpo spans
    [0.4793, 0.4824] across seeds), so the pooled-stdev Cohen's d is
    numerically degenerate (d > 100). The grpo-SD reference is the
    natural scale: a variant that shifts ZVF by 1×grpo-SD has clearly
    moved beyond seed noise.

    Returns: (delta_zvf, effect_vs_grpo_sd).
    """
    if len(variant_zvf) == 0 or len(grpo_zvf) == 0:
        return 0.0, 0.0
    delta = statistics.mean(variant_zvf) - statistics.mean(grpo_zvf)
    grpo_sd = statistics.pstdev(grpo_zvf) if len(grpo_zvf) > 1 else 0.0
    if grpo_sd <= 0:
        return delta, 0.0
    return delta, delta / grpo_sd


def per_method_summary(panel):
    """Per method: mean_zvf, std_zvf, mean_zvf_risk, effect-size vs grpo."""
    grpo_zvf = [r["mean_zvf"] for r in panel["grpo"]]
    grpo_zrisk = [r["zvf_risk"] for r in panel["grpo"]]
    out = {}
    for m, rows in panel.items():
        zs = [r["mean_zvf"] for r in rows]
        zrs = [r["zvf_risk"] for r in rows]
        dz, ez = effect_size_zvf(zs, grpo_zvf)
        dr, er = effect_size_zvf(zrs, grpo_zrisk)
        out[m] = {
            "n_seeds": len(rows),
            "mean_zvf": statistics.mean(zs),
            "std_zvf": statistics.pstdev(zs) if len(zs) > 1 else 0.0,
            "mean_zvf_risk": statistics.mean(zrs),
            "delta_zvf_vs_grpo": dz,
            "effect_size_vs_grpo_sd": ez,
            "delta_zvf_risk_vs_grpo": dr,
            "effect_size_zvf_risk_vs_grpo_sd": er,
            "min_zvf": min(zs),
            "max_zvf": max(zs),
        }
    return out


def firing_rate_curve(panel, method, tau):
    """P7 controller firing rate: fraction of seeds whose mean_zvf <= tau.

    Under the iter-15 / iter-20 convention, the zvf-triage controller fires
    when zvf_t <= tau. The 5-seed panel gives a multi-seed estimate of the
    firing probability under seed-resampled bootstrap.

    Returns: (fires_n, fires_rate, zvf_trajectory_sorted).
    """
    zs = sorted([r["mean_zvf"] for r in panel[method]])
    fires = sum(1 for z in zs if z <= tau)
    return fires, fires / len(zs), zs


def reconcile(predicted_sign, delta_zvf, effect_vs_grpo_sd, measurement_n_seeds):
    """Verdict in the iter-18 idiom, but on the effect-size normalised by grpo SD.

    SUPPORT  — predicted non-zero sign matches measured sign AND
               |effect| >= 50 (i.e. Δ > 50×grpo's seed-to-seed SD —
               an order of magnitude beyond seed noise).
    WEAK     — predicted sign matches measured sign but |effect| < 50.
    OPPOSE   — predicted sign contradicts measured sign.
    NO_DATA  — predicted sign is 0 (no claim), or no measurement available.
    """
    if predicted_sign == 0 or measurement_n_seeds == 0:
        return "NO_DATA"
    measured_sign = 1 if delta_zvf > 0 else (-1 if delta_zvf < 0 else 0)
    if measured_sign == 0:
        return "WEAK"
    if measured_sign == predicted_sign:
        return "SUPPORT" if abs(effect_vs_grpo_sd) >= 50 else "WEAK"
    return "OPPOSE"


def main():
    panel = load_zv130_panel()
    summary = per_method_summary(panel)
    grpo = summary["grpo"]

    # build (delta_id × tau) rows + per-delta summary rows
    rows = []
    per_delta_effect = {}
    for delta_id, predicted_sign in PREDICTED_ZVF_SIGN.items():
        m = METHOD_OF[delta_id]
        if m not in panel:
            per_delta_effect[delta_id] = {
                "delta_id": delta_id, "method": m,
                "n_seeds": 0, "verdict": "NO_DATA",
            }
            continue
        ms = summary[m]
        per_delta_effect[delta_id] = {
            "delta_id": delta_id, "method": m,
            "n_seeds": ms["n_seeds"],
            "mean_zvf": ms["mean_zvf"],
            "mean_zvf_risk": ms["mean_zvf_risk"],
            "delta_zvf_vs_grpo": ms["delta_zvf_vs_grpo"],
            "effect_size_vs_grpo_sd": ms["effect_size_vs_grpo_sd"],
            "delta_zvf_risk_vs_grpo": ms["delta_zvf_risk_vs_grpo"],
            "verdict": reconcile(predicted_sign,
                                 ms["delta_zvf_vs_grpo"],
                                 ms["effect_size_vs_grpo_sd"],
                                 ms["n_seeds"]),
        }

    # build the full (delta × tau) firing-rate table
    per_tau = {tau: {} for tau in TAU_GRID}
    for delta_id in PREDICTED_ZVF_SIGN:
        m = METHOD_OF[delta_id]
        for tau in TAU_GRID:
            if m not in panel:
                per_tau[tau][delta_id] = None
                continue
            fires, rate, _ = firing_rate_curve(panel, m, tau)
            per_tau[tau][delta_id] = {"fires": fires, "rate": rate}

    # write TSV — header + per-delta summary rows + per-(delta × tau) rows
    with OUT_TSV.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["section", "delta_id", "method", "n_seeds",
                    "predicted_zvf_sign", "measured_cohen_d",
                    "zvf_mean", "zvf_risk_mean",
                    "verdict_or_tau", "fires_n", "fires_rate"])
        # summary rows
        for delta_id in PREDICTED_ZVF_SIGN:
            d = per_delta_effect[delta_id]
            w.writerow(["delta_summary", delta_id, d["method"],
                        d.get("n_seeds", 0),
                        PREDICTED_ZVF_SIGN[delta_id],
                        round(d.get("effect_size_vs_grpo_sd", 0.0), 2),
                        round(d.get("mean_zvf", 0.0), 4),
                        round(d.get("mean_zvf_risk", 0.0), 4),
                        d.get("verdict", "NO_DATA"),
                        "", ""])
        # tau sweep rows
        for tau in TAU_GRID:
            for delta_id in PREDICTED_ZVF_SIGN:
                m = METHOD_OF[delta_id]
                row = per_tau[tau].get(delta_id)
                if row is None:
                    w.writerow(["tau_sweep", delta_id, m,
                                summary.get(m, {}).get("n_seeds", 0),
                                PREDICTED_ZVF_SIGN[delta_id],
                                round(per_delta_effect[delta_id].get(
                                    "effect_size_vs_grpo_sd", 0.0), 2),
                                round(summary.get(m, {}).get("mean_zvf", 0.0), 4),
                                round(summary.get(m, {}).get("mean_zvf_risk", 0.0), 4),
                                tau, "", ""])
                else:
                    w.writerow(["tau_sweep", delta_id, m,
                                summary.get(m, {}).get("n_seeds", 0),
                                PREDICTED_ZVF_SIGN[delta_id],
                                round(per_delta_effect[delta_id].get(
                                    "effect_size_vs_grpo_sd", 0.0), 2),
                                round(summary.get(m, {}).get("mean_zvf", 0.0), 4),
                                round(summary.get(m, {}).get("mean_zvf_risk", 0.0), 4),
                                tau, row["fires"], round(row["rate"], 4)])

    # write JSON — full machine-readable
    # Headline P7-controller coupling metric: at each headline τ, the
    # firing rate of the zvf-triage controller when fed THIS variant's
    # measured trajectory, minus the firing rate on the grpo baseline.
    # Negative means the controller fires LESS often on this variant
    # than on grpo (variant has higher ZVF — would suppress controller);
    # positive means the controller fires MORE often on this variant
    # (variant has lower ZVF — controller would intervene more).
    controller_coupling = {}
    for tau in HEADLINE_TAUS:
        grpo_rate = per_tau[tau].get("delta_aero") if False else None
        # compute grpo firing rate at tau (use any "grpo" entry; we
        # already have the raw panel)
        if "grpo" in panel:
            grpo_rate = (sum(1 for r in panel["grpo"] if r["mean_zvf"] <= tau)
                         / len(panel["grpo"]))
        for delta_id in PREDICTED_ZVF_SIGN:
            m = METHOD_OF[delta_id]
            if m not in panel:
                continue
            var_rate = (sum(1 for r in panel[m] if r["mean_zvf"] <= tau)
                        / len(panel[m]))
            controller_coupling.setdefault(delta_id, {})[str(tau)] = {
                "variant_rate": round(var_rate, 4),
                "grpo_rate": round(grpo_rate, 4) if grpo_rate is not None else None,
                "delta_rate_vs_grpo": round(
                    (var_rate - grpo_rate) if grpo_rate is not None else 0, 4),
            }

    # Verdict counts (registry-prediction reconciliation)
    verdict_counts = {
        v: sum(1 for d in per_delta_effect.values()
               if d.get("verdict") == v)
        for v in ["SUPPORT", "WEAK", "OPPOSE", "NO_DATA"]
    }

    # Headline controller-coupling verdict per variant:
    # MORE_FIRE  — controller fires MORE often on this variant than on grpo
    #              at every headline τ where the discrimination is non-zero.
    # LESS_FIRE  — controller fires LESS often (variant suppresses controller).
    # TIE        — same firing rate at every headline τ (no discrimination).
    # NO_DATA    — no zvf130 measurement.
    # MIXED      — variant fires more at SOME τ and less at OTHERS.
    controller_verdicts = {}
    for delta_id, by_tau in controller_coupling.items():
        diffs = [v["delta_rate_vs_grpo"] for v in by_tau.values()
                 if v["delta_rate_vs_grpo"] is not None]
        nonzero_diffs = [d for d in diffs if abs(d) > 1e-9]
        if not diffs:
            controller_verdicts[delta_id] = "NO_DATA"
        elif not nonzero_diffs:
            controller_verdicts[delta_id] = "TIE"
        elif all(d >= 0 for d in nonzero_diffs) and any(d > 0 for d in nonzero_diffs):
            controller_verdicts[delta_id] = "MORE_FIRE"
        elif all(d <= 0 for d in nonzero_diffs) and any(d < 0 for d in nonzero_diffs):
            controller_verdicts[delta_id] = "LESS_FIRE"
        else:
            controller_verdicts[delta_id] = "MIXED"

    out_json = {
        "n_methods_in_panel": len(panel),
        "grpo_baseline": {
            "n_seeds": grpo["n_seeds"],
            "mean_zvf": grpo["mean_zvf"],
            "mean_zvf_risk": grpo["mean_zvf_risk"],
        },
        "per_method_summary": summary,
        "per_delta_effect": per_delta_effect,
        "per_tau_firing_rates": {str(tau): per_tau[tau] for tau in TAU_GRID},
        "headline_taus": HEADLINE_TAUS,
        "method_of_delta": METHOD_OF,
        "predicted_zvf_sign": PREDICTED_ZVF_SIGN,
        "verdict_counts": verdict_counts,
        "controller_coupling": controller_coupling,
        "controller_verdicts": controller_verdicts,
        "controller_verdict_counts": {
            v: sum(1 for x in controller_verdicts.values() if x == v)
            for v in ["MORE_FIRE", "LESS_FIRE", "TIE", "MIXED", "NO_DATA"]
        },
    }
    OUT_JSON.write_text(json.dumps(out_json, indent=2))

    # ---- plot per-variant firing-rate curves ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        methods_with_data = [m for m in METHOD_OF.values() if m in panel]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        # baseline (grpo) drawn as a dashed line first
        if "grpo" in panel:
            xs = TAU_GRID
            ys = [per_tau[t]["delta_aero"] if False else
                  sum(1 for r in panel["grpo"] if r["mean_zvf"] <= t) / len(panel["grpo"])
                  for t in xs]
            ax.plot(xs, ys, "k--", lw=2, label="grpo (baseline)")

        # variants — color by predicted_sign
        cmap = {-1: "#d62728", 0: "#7f7f7f", 1: "#2ca02c"}
        for delta_id, m in METHOD_OF.items():
            if m not in panel:
                continue
            xs = TAU_GRID
            ys = [sum(1 for r in panel[m] if r["mean_zvf"] <= t) / len(panel[m])
                  for t in xs]
            color = cmap[PREDICTED_ZVF_SIGN[delta_id]]
            ax.plot(xs, ys, "-", color=color, lw=1.4, alpha=0.85,
                    label=f"{m} (pred={PREDICTED_ZVF_SIGN[delta_id]:+d})")

        ax.set_xlabel("zvf-triage threshold τ")
        ax.set_ylabel("P7 controller firing rate (5-seed panel)")
        ax.set_title("Registry ↔ P7-controller coupling:\n"
                     "per-variant firing-rate curves on zvf130")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
        fig.tight_layout()
        FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_PNG, dpi=140)
        plt.close(fig)
        out_json["figure_path"] = str(FIG_PNG.relative_to(ROOT))
    except Exception as e:
        out_json["figure_error"] = str(e)

    OUT_JSON.write_text(json.dumps(out_json, indent=2))

    # console summary
    print(f"per-method summary rows: {len(summary)}")
    print(f"registry-prediction verdicts: {out_json['verdict_counts']}")
    print(f"P7 controller-coupling verdicts: {out_json['controller_verdict_counts']}")
    for delta_id, d in per_delta_effect.items():
        v = d.get("verdict", "NO_DATA")
        cv = out_json['controller_verdicts'].get(delta_id, 'NO_DATA')
        ms = summary.get(METHOD_OF[delta_id], {})
        if ms:
            print(f"  {delta_id:18s}  effect={d.get('effect_size_vs_grpo_sd', 0):+.1f}×grpo_sd  "
                  f"Δzvf={d.get('delta_zvf_vs_grpo', 0):+.3f}  "
                  f"mean_zvf={ms['mean_zvf']:.3f}  "
                  f"pred={v}  ctrl={cv}")
    print(f"wrote {OUT_TSV.relative_to(ROOT)}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    if "figure_path" in out_json:
        print(f"wrote {out_json['figure_path']}")


if __name__ == "__main__":
    main()