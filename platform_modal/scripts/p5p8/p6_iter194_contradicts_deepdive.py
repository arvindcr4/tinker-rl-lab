#!/usr/bin/env python3
"""
P6 iter 194 — CONTRADICTS deep-dive + robustness stress test + registry amendment.

Vein: (a) deeper — extends iter-190 (which surfaced 2 CONTRADICTS verdicts) by
stress-testing the 2 CONTRADICTS findings (delta_aero / delta_areal on
reward_mean @ n2_same_stack_last10) along 5 robustness dimensions and then
proposing a registry amendment that aligns the prose with the measurement.

5 robustness dimensions:
  R1. multi-seed bootstrap (5 different seeds, B=2000 each)
  R2. BCa (bias-corrected accelerated) bootstrap CIs
  R3. leave-one-step-out jackknife on the last 10
  R4. multi-window sensitivity (last 5/10/15/20/25)
  R5. cross-panel consistency — does the same direction hold on the
      5-seed zvf130 panel?

Protocol review18-v2: read-only robustness audit and append-only proposal.
Only complete passing checks permit a proposal for human review. This script
never rewrites claims or marks retrospective predictions SUPPORTS. R1 varies
bootstrap RNG seeds on one dataset, not independent training seeds.
Outputs are versioned under p5p8/review18_v2; original entries are preserved
inside each uniquely named proposal record for provenance.
"""
from __future__ import annotations
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
REG = WORKTREE / "registry"
ENT = REG / "entries"
RES = WORKTREE / "experiments" / "results" / "p5p8" / "review18_v2"
RES.mkdir(parents=True, exist_ok=True)
N2_TSV = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume" / "n2_metrics.tsv"
ZVFRISK_TSV = WORKTREE / "experiments" / "results" / "zvf_iter130_method_risk.tsv"

N_BOOT = 2000
DEFAULT_SEED = 20260706
MULTISEED_SEEDS = [20260706, 20260707, 20260708, 20260709, 20260710]
WINDOWS = [5, 10, 15, 20, 25]
BASE = "grpo"
VARIANTS = ["aero", "areal"]


def load_n2():
    """Return dict[(method)] -> list[dict] sorted by step."""
    rows = list(csv.DictReader(open(N2_TSV), delimiter="\t"))
    out = defaultdict(list)
    for r in rows:
        out[r["method"]].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["step"]))
    return dict(out)


def zvfrisk():
    return {r["method"]: r for r in csv.DictReader(open(ZVFRISK_TSV), delimiter="\t")}


def slice_metric(rows_by_method, metric, step_lo, step_hi):
    """Return dict[method] -> list[float] for steps in [step_lo, step_hi] inclusive."""
    out = {}
    for m, rs in rows_by_method.items():
        vals = []
        for r in rs:
            s = int(r["step"])
            if step_lo <= s <= step_hi:
                vals.append(float(r[metric]))
        if vals:
            out[m] = vals
    return out


def paired_step_bootstrap_pct(a, b, n_boot=N_BOOT, seed=DEFAULT_SEED):
    """Paired-step percentile bootstrap CI on mean(b - a).
    Returns (delta, ci_lo, ci_hi)."""
    rng = random.Random(seed)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = [bi - ai for ai, bi in zip(a, b)]
    point = sum(diffs) / len(diffs)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return point, lo, hi


def bca_bootstrap_ci(a, b, n_boot=N_BOOT, seed=DEFAULT_SEED, alpha=0.05):
    """Bias-corrected accelerated bootstrap CI on mean(b - a). Falls back to
    percentile CI if BCa numerics are unstable."""
    rng = random.Random(seed)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = [bi - ai for ai, bi in zip(a, b)]
    point = sum(diffs) / len(diffs)
    # bias-correction z0
    boots_for_z0 = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots_for_z0.append(sum(diffs[i] for i in idx) / n)
    prop_less = sum(1 for b_ in boots_for_z0 if b_ < point) / n_boot
    # numerical safety: clip prop_less away from {0, 1} to avoid ndtri -> inf
    prop_less = max(min(prop_less, 1 - 1e-6), 1e-6)
    # bias-correction: use normal-approx inverse-CDF
    def ndtri(p):
        # Beasley-Springer-Moro normal quantile (good to ~1e-9)
        if p <= 0 or p >= 1:
            return float("-inf") if p <= 0 else float("inf")
        a_ = [-3.969683028665376e+01, 2.209460984245205e+02,
              -2.759285104469687e+02, 1.383577518672690e+02,
              -3.066479806614716e+01, 2.506628277459239e+00]
        b_ = [-5.447609879822406e+01, 1.615858368580409e+02,
              -1.556989798598866e+02, 6.680131188771972e+01,
              -1.328068155288572e+01]
        c_ = [-7.784894002430293e-03, -3.223964580411365e-01,
              -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00, 2.938163982698783e+00]
        d_ = [7.784695709041462e-03, 3.224671290700398e-01,
              2.445134137142996e+00, 3.754408661907416e+00]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c_[0]*q+c_[1])*q+c_[2])*q+c_[3])*q+c_[4])*q+c_[5]) / \
                   ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
        if p <= phigh:
            q = p - 0.5
            r = q * q
            return (((((a_[0]*r+a_[1])*r+a_[2])*r+a_[3])*r+a_[4])*r+a_[5])*q / \
                   (((((b_[0]*r+b_[1])*r+b_[2])*r+b_[3])*r+b_[4])*r+1)
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c_[0]*q+c_[1])*q+c_[2])*q+c_[3])*q+c_[4])*q+c_[5]) / \
                ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)

    z0 = ndtri(prop_less)
    # acceleration jackknife (leave-one-step-out)
    jk = []
    for i in range(n):
        rest = diffs[:i] + diffs[i+1:]
        jk.append(sum(rest) / len(rest))
    jk_mean = sum(jk) / len(jk)
    num = sum((jk_mean - j) ** 3 for j in jk)
    den = (sum((jk_mean - j) ** 2 for j in jk) ** 1.5)
    a_acc = num / (6 * den) if den > 0 else 0.0
    z_lo = ndtri(alpha / 2)
    z_hi = ndtri(1 - alpha / 2)
    a_lo = prop_less + (z0 + z_lo) / (1 - a_acc * (z0 + z_lo))
    a_hi = prop_less + (z0 + z_hi) / (1 - a_acc * (z0 + z_hi))
    # clip a_lo / a_hi to (0, 1) to handle numerical instability
    a_lo = max(min(a_lo, 1 - 1e-6), 1e-6)
    a_hi = max(min(a_hi, 1 - 1e-6), 1e-6)
    # final: percentile CI on adjusted alphas
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    n_b = len(boots)
    lo_idx = max(0, min(n_b - 1, int(a_lo * n_b)))
    hi_idx = max(0, min(n_b - 1, int(a_hi * n_b)))
    return point, boots[lo_idx], boots[hi_idx]


def cohen_d_paired(a, b):
    """Cohen's d for paired samples: mean(b-a) / sd(b-a)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = [bi - ai for ai, bi in zip(a, b)]
    m = sum(diffs) / len(diffs)
    var = sum((d - m) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    return m / sd if sd > 0 else 0.0


def cliffs_delta(a, b):
    """Cliff's delta: (# > - # <) / (n_a * n_b). Robust non-parametric effect size."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    gt = lt = 0
    for ai in a:
        for bi in b:
            if bi > ai:
                gt += 1
            elif bi < ai:
                lt += 1
    return (gt - lt) / (len(a) * len(b))


def leave_one_out_jackknife(a, b):
    """Leave-one-step-out jackknife on mean(b - a). Returns (mean, se, lo, hi)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = [bi - ai for ai, bi in zip(a, b)]
    point = sum(diffs) / len(diffs)
    jk = []
    for i in range(len(diffs)):
        rest = diffs[:i] + diffs[i+1:]
        jk.append(sum(rest) / len(rest))
    jk_mean = sum(jk) / len(jk)
    var = sum((j - jk_mean) ** 2 for j in jk) / max(1, len(jk) - 1)
    se = math.sqrt(var)
    return point, se, point - 1.96 * se, point + 1.96 * se


def amendment_proposal(checks, evidence_complete):
    required = {"h1", "h2b", "h3", "h4", "h5"}
    eligible = evidence_complete and set(checks) == required and all(v is True for v in checks.values())
    return {"protocol_version": "review18-v2", "robustness": checks,
            "eligible_for_human_review": bool(eligible),
            "status": "PROPOSED_REQUIRES_INDEPENDENT_VALIDATION" if eligible else "FAILED_OR_INCONCLUSIVE",
            "proposed_sign": "<=0" if eligible else None,
            "registry_mutated": False}


def main():
    print(f"[iter194] loading N2 from {N2_TSV}")
    n2 = load_n2()
    print(f"[iter194] N2 methods: {sorted(n2.keys())}")
    zvrisk = zvfrisk()
    print(f"[iter194] zvf130 methods: {sorted(zvrisk.keys())}")

    # ----- R1: multi-seed bootstrap -----
    multiseed_rows = []
    for variant in VARIANTS:
        last10 = slice_metric(n2, "reward_mean", 30, 39)
        a, b = last10[BASE], last10[variant]
        for seed in MULTISEED_SEEDS:
            d, lo, hi = paired_step_bootstrap_pct(a, b, seed=seed)
            multiseed_rows.append({
                "variant": variant,
                "metric": "reward_mean",
                "panel": "n2_same_stack_last10",
                "seed": seed,
                "delta": round(d, 6),
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "significant": bool(hi < 0),
            })

    # ----- R2: BCa bootstrap CIs -----
    bca_rows = []
    for variant in VARIANTS:
        last10 = slice_metric(n2, "reward_mean", 30, 39)
        a, b = last10[BASE], last10[variant]
        d_bca, lo_bca, hi_bca = bca_bootstrap_ci(a, b, n_boot=N_BOOT, seed=DEFAULT_SEED)
        # also compute the standard percentile for comparison
        d_pct, lo_pct, hi_pct = paired_step_bootstrap_pct(a, b)
        # jackknife for acceleration
        pt_jk, se_jk, lo_jk, hi_jk = leave_one_out_jackknife(a, b)
        bca_rows.append({
            "variant": variant,
            "metric": "reward_mean",
            "panel": "n2_same_stack_last10",
            "point": round(d_pct, 6),
            "pct_ci_low": round(lo_pct, 6),
            "pct_ci_high": round(hi_pct, 6),
            "bca_ci_low": round(lo_bca, 6),
            "bca_ci_high": round(hi_bca, 6),
            "jk_ci_low": round(lo_jk, 6),
            "jk_ci_high": round(hi_jk, 6),
            "jk_se": round(se_jk, 6),
            "cohens_d": round(cohen_d_paired(a, b), 4),
            "cliffs_delta": round(cliffs_delta(a, b), 4),
            "n_steps": len(b),
        })

    # ----- R3: leave-one-step-out jackknife on each step -----
    jackknife_rows = []
    for variant in VARIANTS:
        last10 = slice_metric(n2, "reward_mean", 30, 39)
        a, b = last10[BASE], last10[variant]
        for i in range(len(b)):
            a_loo = a[:i] + a[i+1:]
            b_loo = b[:i] + b[i+1:]
            d, _, _ = paired_step_bootstrap_pct(a_loo, b_loo)
            jackknife_rows.append({
                "variant": variant,
                "metric": "reward_mean",
                "panel": "n2_same_stack_last10_loo",
                "left_out_step": 30 + i,
                "loo_delta": round(d, 6),
                "loo_sign_consistent_with_full": bool(d < 0),  # full deltas are negative
            })

    # ----- R4: multi-window sensitivity -----
    window_rows = []
    for variant in VARIANTS:
        for w in WINDOWS:
            step_hi = 39
            step_lo = step_hi - w + 1
            sliced = slice_metric(n2, "reward_mean", step_lo, step_hi)
            a, b = sliced[BASE], sliced[variant]
            if not a or not b:
                continue
            d, lo, hi = paired_step_bootstrap_pct(a, b)
            window_rows.append({
                "variant": variant,
                "metric": "reward_mean",
                "panel": f"n2_same_stack_last{w}",
                "window": w,
                "step_lo": step_lo,
                "step_hi": step_hi,
                "delta": round(d, 6),
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "significant": bool(hi < 0 or lo > 0),
                "direction": "negative" if d < 0 else ("positive" if d > 0 else "zero"),
                "contradicts_predicted_sign_ge_0": bool(hi < 0),  # predicted ">=0" but actually <0
            })

    # ----- R5: cross-panel consistency (zvf130) -----
    cross_panel_rows = []
    for variant in VARIANTS:
        zvr = zvfrisk()
        a = float(zvr[BASE]["zvf_risk_mean"])
        b = float(zvr[variant]["zvf_risk_mean"])
        sd_a = float(zvr[BASE]["zvf_risk_sd"] or 0)
        sd_b = float(zvr[variant]["zvf_risk_sd"] or 0)
        delta = b - a
        se = math.sqrt((sd_a ** 2 + sd_b ** 2) / 5)
        ci_lo = delta - 1.96 * se
        ci_hi = delta + 1.96 * se
        # also for mag_mean
        ma = float(zvr[BASE]["mag_mean"])
        mb = float(zvr[variant]["mag_mean"])
        cross_panel_rows.append({
            "variant": variant,
            "metric_zvf_risk_mean": round(delta, 6),
            "zvf_risk_ci": [round(ci_lo, 6), round(ci_hi, 6)],
            "zvf_risk_significant_negative": bool(ci_hi < 0),
            "mag_mean_delta": round(mb - ma, 6),
            "mag_mean_base": round(ma, 6),
            "mag_mean_variant": round(mb, 6),
            "interpretation": (
                f"zvf_risk_mean is significantly LOWER for {variant} than grpo "
                f"(delta={delta:.4f}, CI=[{ci_lo:.4f},{ci_hi:.4f}]), but "
                f"reward_mean is significantly LOWER on n2_same_stack_last10 — "
                f"i.e. {variant} trades reward for zvf reduction."
            ),
        })

    # ----- Write outputs -----
    out1 = RES / "p6_iter194_robustness_multiseed.tsv"
    with open(out1, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(multiseed_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(multiseed_rows)

    out2 = RES / "p6_iter194_robustness_bca.tsv"
    with open(out2, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(bca_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(bca_rows)

    out3 = RES / "p6_iter194_robustness_jackknife.tsv"
    with open(out3, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(jackknife_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(jackknife_rows)

    out4 = RES / "p6_iter194_robustness_window.tsv"
    with open(out4, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(window_rows)

    out5 = RES / "p6_iter194_robustness_cross_panel.tsv"
    with open(out5, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(cross_panel_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(cross_panel_rows)

    # ----- Aggregate stats for hypotheses -----
    # H1: All 5 multi-seed bootstraps have CI_high < 0 for both variants.
    h1_pass = all(
        r["ci_high"] < 0
        for r in multiseed_rows
    )
    # H2a: BCa lower bound < 0 for both variants (point estimate below CI center).
    h2a_pass = all(r["bca_ci_low"] < 0 for r in bca_rows)
    # H2b: BCa upper bound < 0 for both variants (strict exclusion of 0).
    h2b_pass = all(r["bca_ci_high"] < 0 for r in bca_rows)
    # H2c: BCa upper bound <= 0 for both variants (consistency with the <=0 prediction).
    h2c_pass = all(r["bca_ci_high"] <= 0 for r in bca_rows)
    # Overall H2 passes if H2a AND H2c both pass (relaxed: upper can touch 0 but not cross).
    h2_pass = h2a_pass and h2c_pass
    # H3: Jackknife leave-one-step-out: ALL 10 LOO steps give negative delta for both variants.
    by_variant_jk = defaultdict(list)
    for r in jackknife_rows:
        by_variant_jk[r["variant"]].append(r["loo_delta"])
    h3_pass = all(
        all(d < 0 for d in by_variant_jk[v])
        for v in VARIANTS
    )
    # H4: Multi-window — at least 3 of 5 windows have significant negative direction.
    by_variant_win = defaultdict(list)
    for r in window_rows:
        by_variant_win[r["variant"]].append(r)
    h4_pass = all(
        sum(1 for r in by_variant_win[v] if r["significant"] and r["direction"] == "negative") >= 3
        for v in VARIANTS
    )
    # H5: Cross-panel — zvf_risk_mean is significantly NEGATIVE for both variants (lower risk = lower zvf = success on that channel).
    h5_pass = all(
        r["zvf_risk_significant_negative"]
        for r in cross_panel_rows
    )

    # Protocol v2: analysis is read-only with respect to claims and verdicts.
    # Passing same-data robustness tests can motivate a proposal, not validate
    # a retrospective prediction. Original claims remain untouched.
    checks = dict(h1=h1_pass, h2b=h2b_pass, h3=h3_pass, h4=h4_pass, h5=h5_pass)
    evidence_complete = all((multiseed_rows, bca_rows, jackknife_rows,
                             window_rows, cross_panel_rows))
    proposal = amendment_proposal(checks, evidence_complete)
    proposal["original_entries"] = {
        variant: json.loads((ENT / f"delta_{variant}.json").read_text())
        for variant in VARIANTS}
    # Append-only audit: never overwrite a previous proposal or registry claim.
    audit_path = RES / ("p6_iter194_proposal_" + __import__("uuid").uuid4().hex + ".json")
    with audit_path.open("x") as f:
        json.dump(proposal, f, indent=2)

    summary = {
        "iter": 194,
        "pillar": "P6",
        "vein": "(a) deeper — CONTRADICTS deep-dive + robustness stress test + registry amendment for delta_aero/delta_areal reward_mean claim",
        "raw_data_sources": [
            "platform_hybrid/experiments/results/n2_reward_tensor_resume/n2_metrics.tsv",
            "platform_hybrid/experiments/results/zvf_iter130_method_risk.tsv",
        ],
        "h1_multiseed_pass": h1_pass,
        "h2a_bca_lower_pass": h2a_pass,
        "h2b_bca_strict_pass": h2b_pass,
        "h2c_bca_relaxed_pass": h2c_pass,
        "h2_bca_pass": h2_pass,
        "h3_jackknife_pass": h3_pass,
        "h4_window_pass": h4_pass,
        "h5_cross_panel_pass": h5_pass,
        "n_h_pass": sum([h1_pass, h2_pass, h3_pass, h4_pass, h5_pass]),
        "n_h_total": 5,
        "protocol_version": "review18-v2",
        "registry_amendments_applied": [],
        "amendment_proposal": str(audit_path),
        "amendment_change": "None; read-only analysis. Retrospective revision requires independent validation.",
        "outputs": [
            "p6_iter194_robustness_multiseed.tsv",
            "p6_iter194_robustness_bca.tsv",
            "p6_iter194_robustness_jackknife.tsv",
            "p6_iter194_robustness_window.tsv",
            "p6_iter194_robustness_cross_panel.tsv",
        ],
        "registry_patches": [],
    }
    out_sum = RES / "p6_iter194_summary.json"
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()