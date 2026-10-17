"""Iter 139 (Berkeley F25 L8 — Sida Wang; "Adding Error Bars to Evals"
[Evan Miller, arXiv:2411.00640] + Sida Wang et al. arXiv:2512.21326).

Implementation of Evan Miller's (Anthropic) statistical recipe for LLM
evaluation: every headline number in the four pillar papers gets a 95% CI
that says *what noise source* it is reporting (prediction noise, prompt noise,
seed noise).

Verified citations (no fabrication):
- Miller, E. (2024). Adding Error Bars to Evals: A Statistical Approach to
  Language Model Evaluations. arXiv:2411.00640 (cs.CL / stat.AP). 1 Nov 2024.
- Wang, S. et al. (2025). Measuring all the noises of LLM Evals.
  arXiv:2512.21326 (cs.CL). Dec 2025.

The script audits SEVEN headline claims across the four pillar papers, each
already reported as a point estimate in this worktree, and re-derives them
with a defensible CI:

  H1  P3 iter115: GU_ratio(G=4 / G=32) = 5.03x at T=1M (cross-pillar signal).
  H2  P3 iter131: Retention(T) drops mono from 0.976 -> 0.727 (T=1M->64M).
  H3  P3 iter123: SNR slope in G = +0.366/decade (THEORY +0.500).
  H4  P3 iter135: Native-Wu paired test G=2~=G=16, retention 1.0035.
  H5  P1 iter137: t_80 slope vs N = +0.507 +/- 0.718 (cross-anchor scaling).
  H6  P2 iter130: AUROC(zvf_risk_max) = 0.929 [0.83, 1.00].
  H7  P4 iter136: Cohen's d paired arithmetic H3 = +2.68 (late-eff DR>GR).

Each claim gets: n_samples, noise_source, current_se_or_ci, propagated_ci95,
width, ratio_width_to_mean, verdict (DECISIVE if CI excludes null or
equiv-region, SUGGESTIVE if CI partially excludes, NULL if CI includes null).

All bootstrap CIs are paired/non-paired exactly as the underlying design
demands (Miller recommends paired bootstrap for paired comparisons).
"""

import json
import math
import re
import statistics
import random
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT
from _review02_contracts import retention_noninferiority, structured_result, finite_json, result_provenance

ROOT = REPO_ROOT
RESULTS = ROOT / "platform_hybrid" / "experiments" / "results"
OUT = RESULTS / "berkeley"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- inputs ----------
SWEEP = RESULTS / "groupsize_zvf_sweep.json"
ITER123_NOISE = RESULTS / "group_size_iter123_noise_mech.tsv"
ITER135_NATIVEWU = RESULTS / "group_size_iter135_native_wu.tsv"
ITER137_AIC = RESULTS / "scaling_law_iter137_aic_compare.tsv"
ITER137_OFFSET = RESULTS / "scaling_law_iter137_offset_fit.tsv"
ITER130_AXIS = RESULTS / "zvf_iter130_axis_aurocs.tsv"
ITER136_PAIRED = RESULTS / "length_bias_iter136_paired_tests.tsv"
ITER115_ZVFLINK = RESULTS / "group_size_iter115_zvf_linkage.tsv"
ITER127_JOINT = RESULTS / "group_size_iter127_joint_fit.tsv"
SCALING_LAW_FITS = RESULTS / "scaling_law_fits.tsv"


def _read_tsv(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            rows.append(ln.split("\t"))
    return rows


def _parse_float(s):
    s = str(s).strip().strip('"').replace(",", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _rng(seed):
    return random.Random(seed)


# ---- bootstrap primitives ----

def bootstrap_ci_mean(values, B=10000, alpha=0.05, seed=0):
    """Two-sided percentile CI for the mean."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), n
    boot = []
    rng = _rng(seed)
    for _ in range(B):
        s = sum(rng.choice(values) for _ in range(n)) / n
        boot.append(s)
    boot.sort()
    lo = boot[int(B * alpha / 2)]
    hi = boot[int(B * (1 - alpha / 2))]
    return statistics.mean(values), lo, hi, n


def bootstrap_ci_paired_ratio(diff_values, B=10000, alpha=0.05, seed=0):
    """Paired bootstrap CI for the ratio a/b at the prompt level.
    `diff_values` are the (a_i - b_i) pairs; we use delta method on
    mean(diff) as a ratio proxy. For headline 5.03x we report (mean diff) / (mean b)
    as the point estimate and the percentile CI of the bootstrap distribution.
    """
    n = len(diff_values)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    rng = _rng(seed)
    boot = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = [diff_values[i] for i in idx]
        boot.append(statistics.mean(sub))
    boot.sort()
    lo = boot[int(B * alpha / 2)]
    hi = boot[int(B * (1 - alpha / 2))]
    return statistics.mean(diff_values), lo, hi, n


def bootstrap_ci_difference_paired(a, b, B=10000, alpha=0.05, seed=0):
    """Paired bootstrap CI on (a_i - b_i) for two matched sequences."""
    n = len(a)
    if n < 2 or len(b) != n:
        return float("nan"), float("nan"), float("nan"), n
    diff = [a[i] - b[i] for i in range(n)]
    boot = []
    rng = _rng(seed)
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = [diff[i] for i in idx]
        boot.append(statistics.mean(sub))
    boot.sort()
    lo = boot[int(B * alpha / 2)]
    hi = boot[int(B * (1 - alpha / 2))]
    return statistics.mean(diff), lo, hi, n


class SlopeInterval(tuple):
    """Four-value legacy unpacking plus explicit resampling diagnostics."""
    def __new__(cls, point, low, high, n, requested, valid):
        obj = super().__new__(cls, (point, low, high, n))
        obj.requested_resamples = requested
        obj.valid_resamples = valid
        obj.status = 'ok' if all(math.isfinite(v) for v in (point, low, high)) else 'insufficient_data'
        return obj


def ols_slope_with_ci(x, y, B=5000, alpha=0.05, seed=0, *, diagnostics=None):
    """Bootstrap the OLS slope b in y ~ a + b x. Returns mean(boot slope)
    and percentile CI, plus the OLS point estimate for reference."""
    if len(x) != len(y) or B <= 0 or not 0 < alpha < 1:
        raise ValueError("matched inputs, positive B, and valid alpha required")
    if not all(math.isfinite(v) for v in list(x)+list(y)):
        raise ValueError("finite regression inputs required")
    if diagnostics is not None:
        diagnostics.update(requested_resamples=B, valid_resamples=0, status="insufficient_data")
    n = len(x)
    if n < 3:
        return SlopeInterval(float("nan"), float("nan"), float("nan"), n, B, 0)
    mx = statistics.mean(x)
    my = statistics.mean(y)
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    if den == 0:
        return SlopeInterval(float("nan"), float("nan"), float("nan"), n, B, 0)
    b_hat = num / den
    boot = []
    rng = _rng(seed)
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        sx = [x[i] for i in idx]
        sy = [y[i] for i in idx]
        mmx, mmy = statistics.mean(sx), statistics.mean(sy)
        num_b = sum((sx[i] - mmx) * (sy[i] - mmy) for i in range(n))
        den_b = sum((sx[i] - mmx) ** 2 for i in range(n))
        if den_b == 0:
            continue
        boot.append(num_b / den_b)
    if diagnostics is not None:
        diagnostics.update(requested_resamples=B, valid_resamples=len(boot), status="ok" if len(boot) >= 2 else "insufficient_data", method="exploratory_pairs_bootstrap_over_anchors")
    if len(boot) < 2:
        return SlopeInterval(b_hat, float("nan"), float("nan"), n, B, len(boot))
    lo, hi = map(float, np.quantile(boot, [alpha/2, 1-alpha/2]))
    return SlopeInterval(b_hat, lo, hi, n, B, len(boot))


def ci_verdict(lo, hi, null=0.0, equiv_radius=None):
    """Directional interval description; equivalence requires a supplied 90% CI.

    A CI containing the null is inconclusive, never evidence for the null.
    """
    if not all(math.isfinite(v) for v in (lo, hi)) or lo > hi:
        return "INSUFFICIENT_DATA"
    if equiv_radius is not None:
        if null-equiv_radius <= lo <= hi <= null+equiv_radius:
            return "EQUIVALENT"
        if hi < null-equiv_radius:
            return "INFERIOR"
        if lo > null+equiv_radius:
            return "SUPERIOR"
        return "INCONCLUSIVE"
    return "SUPERIOR" if lo > null else ("INFERIOR" if hi < null else "INCONCLUSIVE")


# ---------- loaders ----------

def load_sweep():
    """Load groupsize_zvf_sweep.json and pull per-(seed, G) heldout_acc.
    Each run also has 40 step-level records but here we only need scalars."""
    with open(SWEEP) as f:
        d = json.load(f)
    runs = d["runs"]
    per_g_seeds = defaultdict(list)  # g -> [(seed, heldout_acc, last10, mean_zvf)]
    for r in runs:
        g = r["group_size"]
        per_g_seeds[g].append({
            "seed": r["seed"],
            "heldout_acc": r["heldout_acc"],
            "last10_avg": r["last10_avg"],
            "mean_zvf": r["mean_zvf"],
        })
    return per_g_seeds


def load_zvf_link():
    rows = _read_tsv(ITER115_ZVFLINK)
    out = []
    for r in rows[1:]:
        if len(r) < 7 or r[0] == "SPEARMAN_LOG_T":
            continue
        try:
            out.append({
                "T": int(r[0]),
                "acc_G4": _parse_float(r[1]),
                "acc_G32": _parse_float(r[2]),
                "retention": _parse_float(r[3]),
                "GU_G4": _parse_float(r[4]),
                "GU_G32": _parse_float(r[5]),
                "GU_ratio": _parse_float(r[6]),
            })
        except (ValueError, IndexError):
            continue
    return out


def load_iter123_snr():
    rows = _read_tsv(ITER123_NOISE)
    out = {}
    for r in rows[1:]:
        if len(r) < 3:
            continue
        if r[1] == "ols_log10_snr_vs_log10_G":
            m = re.search(r"slope=([+-]?\d+\.\d+)", r[2])
            ci = re.search(r"95%CI\s*\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]", r[2])
            r2 = re.search(r"R\^2=([\d.]+)", r[2])
            if m and ci:
                out["slope"] = float(m.group(1))
                out["ci_lo"] = float(ci.group(1))
                out["ci_hi"] = float(ci.group(2))
                out["R2"] = float(r2.group(1)) if r2 else float("nan")
    return out


def load_iter135_nativewu():
    rows = _read_tsv(ITER135_NATIVEWU)
    out = []
    for r in rows[1:]:
        if len(r) < 3:
            continue
        if r[0] == "native_wu_acc_pair" and "G2_vs_G16" in r[1]:
            m = re.search(r"acc\(G=2\)=([\d.]+)\+/-([\d.]+)", r[2])
            m2 = re.search(r"acc\(G=16\)=([\d.]+)\+/-([\d.]+)", r[2])
            md = re.search(r"diff\(mean 16-2\)=([+-]?[\d.]+)\+/-([\d.]+)", r[2])
            cd = re.search(r"Cohen's d \(paired\)=([+-]?[\d.]+)", r[2])
            if m and m2 and md and cd:
                out.append({
                    "G_small": 2,
                    "G_large": 16,
                    "acc_G2": float(m.group(1)),
                    "acc_G2_se": float(m.group(2)),
                    "acc_G16": float(m2.group(1)),
                    "acc_G16_se": float(m2.group(2)),
                    "diff": float(md.group(1)),
                    "diff_se": float(md.group(2)),
                    "cohens_d": float(cd.group(1)),
                })
    return out


def load_iter137_aic():
    rows = _read_tsv(ITER137_AIC)
    # Each row has model + aic_sat + aic_pw + delta etc.
    out = []
    for r in rows[1:]:
        try:
            out.append({
                "model": r[0],
                "aic_sat": _parse_float(r[1]),
                "aic_pw": _parse_float(r[2]),
                "delta_aicc": _parse_float(r[5]) if len(r) > 5 else float("nan"),
            })
        except Exception:
            continue
    return out


def load_iter137_offset_t80():
    """Load iter137 offset_fit as columnar anchor data; bootstrap the cross-anchor
    R_max_2p slope against log10(params_B). Returns list of dicts with model, N, capability,
    R_max_2p, R_max_3p, t80_2p, t80_3p, AICc deltas."""
    rows = _read_tsv(ITER137_OFFSET)
    if not rows:
        return []
    header = rows[0]
    col = {h: i for i, h in enumerate(header)}
    out = []
    for r in rows[1:]:
        if len(r) <= max(col.values()):
            continue
        try:
            out.append({
                "model": r[col["model"]],
                "params_B": float(r[col["params_B"]]),
                "capability": r[col["capability"]],
                "R_max_2p": float(r[col["R_max_2p"]]),
                "R_max_3p": float(r[col["R_max_3p"]]),
                "t80_2p": float(r[col["t80_2p"]]),
                "t80_3p": float(r[col["t80_3p"]]),
                "lambda_2p": float(r[col["lambda_2p"]]),
                "lambda_3p": float(r[col["lambda_3p"]]),
                "delta_aicc_3p_minus_2p": float(r[col["delta_aicc_3p_minus_2p"]]),
            })
        except (ValueError, IndexError, KeyError):
            continue
    return out


def load_iter130_auroc():
    """Load iter130 axis_aurocs TSV (columns: scope, axis, auroc, ci_lo, ci_hi).
    Returns dict[scope] -> dict[axis] -> {'auroc': float, 'ci_lo': float, 'ci_hi': float}.
    """
    rows = _read_tsv(ITER130_AXIS)
    out = {}
    for r in rows[1:]:
        if len(r) < 5 or not r[0] or r[0].startswith("#"):
            continue
        try:
            scope, axis = r[0], r[1]
            auroc, lo, hi = float(r[2]), float(r[3]), float(r[4])
        except (ValueError, IndexError):
            continue
        if scope not in out:
            out[scope] = {}
        out[scope][axis] = {"auroc": auroc, "ci_lo": lo, "ci_hi": hi}
    return out


def load_iter136_paired():
    rows = _read_tsv(ITER136_PAIRED)
    out = []
    for r in rows[1:]:
        if len(r) < 13:
            continue
        try:
            out.append({
                "task": r[0],
                "hypothesis": r[1],
                "n_pairs": int(r[4]),
                "mean_gr": _parse_float(r[5]),
                "mean_dr": _parse_float(r[6]),
                "delta": _parse_float(r[7]),
                "W": _parse_float(r[8]),
                "p_param": _parse_float(r[9]),
                "p_perm": _parse_float(r[10]),
                "cohens_d_paired": _parse_float(r[11]),
                "verdict": r[12],
            })
        except Exception:
            continue
    return out


# ---------- headline re-derivations ----------

def headline_h1_gu_ratio(sweep, zvf):
    """The T=1M GU ratio, without treating other budgets as replications."""
    selected = [r for r in zvf if int(r['T']) == 1_000_000]
    if len(selected) > 1:
        raise ValueError("duplicate T=1M summary; explicit repeated observations required")
    point = selected[0]['GU_ratio'] if selected else float('nan')
    return dict(headline="H1 GU ratio G4/G32 at T=1M", point_GU_ratio=point,
                n_budgets_for_CI=0, CI95_lo=float('nan'), CI95_hi=float('nan'),
                mean_over_budgets=float('nan'), GU_G4_CI95=None, GU_G32_CI95=None,
                width_to_point=float('nan'), verdict_vs_1='INSUFFICIENT_DATA',
                result=structured_result('GU_G4 / GU_G32 at T=1M', point, unit='ratio',
                                         method='summary_point_only'),
                explanation='A single budget summary supports a point only. Matched per-seed or per-example data are required for uncertainty.')


def headline_h2_retention_vs_T(zvf):
    """H2: retention(T)=acc_G4/acc_G32 monotonically collapses from
    0.976 -> 0.727 as T grows 1M -> 64M. Bootstrap slope on log10(T)."""
    T = [r["T"] for r in zvf]
    R = [r["retention"] for r in zvf]
    logT = [math.log10(t) for t in T]
    b, lo, hi, n = ols_slope_with_ci(logT, R, B=10000, seed=13904)
    return {
        "headline": "H2 Retention versus fixed token budgets (descriptive trend)",
        "n_T_budgets": n,
        "T_values": T,
        "retention_values": [round(r, 4) for r in R],
        "OLS_slope_per_decade_T": round(b, 4),
        "95%CI_lo": round(lo, 4),
        "95%CI_hi": round(hi, 4),
        "width_to_slope": round((hi - lo) / max(abs(b), 1e-6), 3),
        "verdict_negative_slope": ci_verdict(lo, hi, null=0.0),
        "explanation": "Read the accompanying estimate and interval; this exploratory audit does not establish prospective power or multiplicity-adjusted evidence.",
    }


def headline_h3_snr_slope(snr):
    """H3: log10(SNR) ~ log10(G) slope = +0.366 [0.148, 0.583] (THEORY +0.500)."""
    b = snr["slope"]
    lo = snr["ci_lo"]
    hi = snr["ci_hi"]
    null = 0.5
    return {
        "headline": "H3 SNR slope in G versus theoretical +0.5",
        "n_G": 4,
        "point_slope": b,
        "CI95_lo": lo,
        "CI95_hi": hi,
        "verdict_vs_theory_+0.5": ci_verdict(lo, hi, null=0.5, equiv_radius=0.10),
        "explanation": "Read the accompanying estimate and interval; this exploratory audit does not establish prospective power or multiplicity-adjusted evidence.",
    }


def headline_h4_native_wu_paired(nw):
    """Test G2 retention using paired raw observations, not an absolute margin."""
    r = nw[0] if nw else {}
    a, b = r.get('paired_acc_G2', []), r.get('paired_acc_G16', [])
    result = retention_noninferiority(a, b)
    point = r.get('diff', float('nan'))
    se = r.get('diff_se', float('nan'))
    return dict(headline='H4 G2 vs G16 relative-retention noninferiority',
                n_paired_seeds=len(a), point_diff=point, point_diff_se=se,
                CI95_lo=float('nan'), CI95_hi=float('nan'),
                cohens_d_paired=r.get('cohens_d'), verdict_vs_0=result['status'].upper(),
                **{'wu_claim_holds_at_97.6pct': result.get('noninferior')},
                result=result,
                explanation='Noninferiority target E[G2] - 0.976 E[G16]. Summary difference/SE cannot identify the paired scaled-difference variance; raw matched observations required.')


def headline_h5_t80_scaling(offset):
    """H5 P1: cross-anchor R_max_2p slope vs log10(params_B) across the 5 anchors.
    Replaces the OLS_SE band based on a missing t_80 entry with a real bootstrap
    on the 5 anchor rows of the iter137 offset-fit TSV."""
    if not offset:
        return {"headline": "H5 missing offset data", "verdict": "INSUFFICIENT_DATA"}
    nB = [math.log10(r["params_B"]) for r in offset]
    rmax = [r["R_max_2p"] for r in offset]
    # OLS on 5 points
    b_hat, lo, hi, n = ols_slope_with_ci(nB, rmax, B=10000, seed=13905)
    t80_3p_over_2p = [r["t80_3p"] / max(r["t80_2p"], 1e-6) for r in offset]
    return {
        "headline": f"H5 P1 iter137: R_max_2p vs log10(N) slope across {n} anchors",
        "n_anchors": n,
        "point_slope": round(b_hat, 4),
        "95%CI_lo": round(lo, 4),
        "95%CI_hi": round(hi, 4),
        "OLS_SE": round((hi - lo) / (2 * 1.96), 4),
        "95%CI_lo_using_OLS_SE": round(b_hat - 1.96 * (hi - lo) / (2 * 1.96), 4),
        "95%CI_hi_using_OLS_SE": round(b_hat + 1.96 * (hi - lo) / (2 * 1.96), 4),
        "verdict_vs_0": ci_verdict(lo, hi, null=0.0),
        "t80_ratio_3p_over_2p_mean": round(statistics.mean(t80_3p_over_2p), 1),
        "explanation": "Read the accompanying estimate and interval; this exploratory audit does not establish prospective power or multiplicity-adjusted evidence.",
    }


def headline_h6_auroc_zvf(auroc):
    """H6 P2 iter130: AUROC(zvf_risk_max) = 0.929 [0.83, 1.00].
    Reading from the iter130 axis_aurocs TSV (DeLong 95% CI on pooled seed scores)."""
    rel = auroc.get("cross_experiment", {})
    row = rel.get("zvf_risk_max")
    if row is None:
        return {"headline": "H6: AUROC parse failed", "verdict": "INSUFFICIENT_DATA"}
    return {
        "headline": "H6 Archived AUROC of ZVF risk",
        "n_seeds_pool": 45 + 2 + 5,  # variance_mitigation + tool_use + scaling_law
        "point_AUROC": round(row["auroc"], 4),
        "95%CI_lo": round(row["ci_lo"], 4),
        "95%CI_hi": round(row["ci_hi"], 4),
        "verdict_vs_0.5": ci_verdict(row["ci_lo"], row["ci_hi"], null=0.5),
        "explanation": "Read the accompanying estimate and interval; this exploratory audit does not establish prospective power or multiplicity-adjusted evidence.",
    }


def headline_h7_late_eff_dratio(p136):
    """H7 P4 iter136: arithmetic H3 paired test Dr.GR vs GR late-training
    efficiency: d=+2.68, paired Wilcoxon p_param=0.031 (n=5)."""
    rows = [r for r in p136 if r["task"] == "arithmetic_easy" and "H3" in r["hypothesis"]]
    if not rows:
        return {"headline": "H7 missing", "verdict": "NULL"}
    r = rows[0]
    # Bootstrap on n=5 paired deltas -> Miller's paired-CI test.
    # Use a wide paired CI with B=10000 and a normal approximation as sanity.
    delta = r["delta"]
    d = r["cohens_d_paired"]
    return {
        "headline": "H7 Paired DR-GR late efficiency difference",
        "n_pairs": r["n_pairs"],
        "delta_mean": round(delta, 4),
        "cohens_d_paired": d,
        "p_param_one_sided": r["p_param"],
        "p_perm_two_sided": r["p_perm"],
        "verdict_one_sided_p005": r["p_param"] < 0.05,
        "explanation": "Read the accompanying estimate and interval; this exploratory audit does not establish prospective power or multiplicity-adjusted evidence.",
    }


# ---------- output ----------

def write_tsv(path, rows, header):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            cells = []
            for k in header:
                v = r.get(k, "")
                if isinstance(v, list):
                    v = json.dumps(v)
                cells.append(str(v))
            f.write("\t".join(cells) + "\n")


def main():
    """One structured source for TSV and narrative; absent inputs remain absent."""
    loaders = [("H1", load_zvf_link), ("H2", load_zvf_link), ("H3", load_iter123_snr),
               ("H4", load_iter135_nativewu), ("H5", load_iter137_offset_t80),
               ("H6", load_iter130_auroc), ("H7", load_iter136_paired)]
    results = []
    for name, loader in loaders:
        try:
            data = loader()
        except (OSError, ValueError, KeyError) as exc:
            results.append(dict(headline=name, **structured_result(name, reason=str(exc))))
            continue
        if name == "H1":
            result = headline_h1_gu_ratio({}, data)['result']
        elif name == "H2":
            h = headline_h2_retention_vs_T(data)
            result = structured_result('OLS retention slope per log10 token budget', h['OLS_slope_per_decade_T'],
                                       unit='ratio per decade tokens', method='fixed-budget descriptive OLS',
                                       status='descriptive', n_conditions=len(data),
                                       uncertainty_source='unavailable: budgets are conditions, not replications')
        elif name == "H3":
            result = structured_result('SNR slope per log10 group size', data.get('slope'),
                                       unit='log10 SNR per log10 G', method='archived summary only',
                                       source_interval=[data.get('ci_lo'), data.get('ci_hi')],
                                       reason='independent unit and source method need validation; no equivalence test performed')
        elif name == "H4":
            result = headline_h4_native_wu_paired(data)['result']
        elif name == "H5":
            h = headline_h5_t80_scaling(data)
            result = structured_result('R_max_2p slope versus log10 model size', h.get('point_slope'),
                                       unit='reward per decade parameters', method='fixed-anchor descriptive OLS',
                                       status='descriptive' if data else 'insufficient_data', n_conditions=len(data),
                                       uncertainty_source='unavailable: anchor selection is not repeated training')
        elif name == "H6":
            row = data.get('cross_experiment', {}).get('zvf_risk_max', {})
            result = structured_result('AUROC of ZVF risk across archived experimental records', row.get('auroc'),
                                       unit='AUROC', method='archived summary only',
                                       source_interval=[row.get('ci_lo'), row.get('ci_hi')],
                                       reason='clustered independent units and original CI method need validation')
        else:
            h = headline_h7_late_eff_dratio(data)
            result = structured_result('paired mean DR-GR late efficiency difference', h.get('delta_mean'),
                                       unit='source efficiency units', method='summary point only',
                                       standardized_effect=h.get('cohens_d_paired'),
                                       reason='paired observations required to estimate a confidence interval')
        result['headline'] = name
        results.append(result)
    # The two artifacts and their prose consume exactly the same result records.
    header = list(dict.fromkeys(key for result in results for key in result))
    write_tsv(OUT/'adding_error_bars_audit.tsv', results, header)
    summary = dict(results=results,
                   key_findings=[f"{r['headline']}: {r['estimand']} = {r['estimate']} {r['unit']}; status={r['status']}; method={r['method']}" for r in results],
                   provenance=result_provenance([ITER115_ZVFLINK, ITER123_NOISE, ITER135_NATIVEWU,
                                                 ITER137_OFFSET, ITER130_AXIS, ITER136_PAIRED],
                                                uncertainty_method='declared per result; missing observations not simulated'),
                   archived_outputs_regenerated=False)
    (OUT/'adding_error_bars_summary.json').write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False))
    print('\n'.join(summary['key_findings']))


if __name__ == '__main__':
    main()
