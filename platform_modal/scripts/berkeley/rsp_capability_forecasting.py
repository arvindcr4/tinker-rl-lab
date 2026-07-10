#!/usr/bin/env python3
"""
Row 22 (B-F24): F24 L11 Ben Mann (Anthropic) -- RSP + Measuring Agent Capabilities.

Ports Anthropic's Responsible Scaling Policy (RSP, 2023) discipline and the
conservative capability-elicitation / forecasting protocol of Phuong et al.,
"Evaluating Frontier Models for Dangerous Capabilities" (arXiv:2403.13793,
DeepMind 2024) onto TinkerRL-Bench's Pillar-1 scaling-law bootstrap fits.

Mapping RSP -> our benchmark:
  * capability threshold R*        <- a reward asymptote the model can "cross"
  * red line   (actual capability) <- point-estimate ceiling r_max_mean
  * yellow line (trigger eval)     <- upper-CI ceiling r_max_hi  (conservative)
  * safety buffer                  <- distance yellow trips before red
  * elicitation gap (assume more)  <- R_max_policy / RQS-adjusted ceiling
  * forecast reliability gate      <- bootstrap CI width / lam-at-bound rate
  * forecasting horizon            <- t_80 (steps to reach 80% of asymptote)

Five falsifiable hypotheses (see docstring per block). Real data only:
  platform_hybrid/experiments/results/scaling_law_bootstrap_ci.tsv       (5 models, 1000-boot CIs)
  platform_hybrid/experiments/results/berkeley/eureka_rqs_per_anchor.tsv (RQS elicitation quality)
  platform_hybrid/experiments/results/berkeley/sweagent_aci_decomp.tsv   (R_max_policy elicitation-adj)
"""
import csv, json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "berkeley")


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def spearman(xs, ys):
    """Spearman rho with average-rank ties; returns (rho, n)."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if not (math.isnan(x) or math.isnan(y))]
    n = len(pairs)
    if n < 3:
        return float("nan"), n

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx = rank([p[0] for p in pairs])
    ry = rank([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan"), n
    return num / (dx * dy), n


def write_tsv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


# ---------------------------------------------------------------- load & merge
boot = read_tsv(os.path.join(RES, "scaling_law_bootstrap_ci.tsv"))
rqs = {r["model"]: fnum(r["RQS"]) for r in read_tsv(os.path.join(OUT, "eureka_rqs_per_anchor.tsv"))}
aci = {r["model"]: r for r in read_tsv(os.path.join(OUT, "sweagent_aci_decomp.tsv"))}

M = []
for r in boot:
    m = r["model"]
    M.append(dict(
        model=m, params_B=fnum(r["params_B"]), n_steps=fnum(r["n_steps"]),
        r_max=fnum(r["r_max_mean"]), r_lo=fnum(r["r_max_lo"]), r_hi=fnum(r["r_max_hi"]),
        t80=fnum(r["t_80_mean"]), t80_lo=fnum(r["t_80_lo"]), t80_hi=fnum(r["t_80_hi"]),
        lam_bound=fnum(r["lam_at_bound_rate"]), rss=fnum(r["rss"]),
        rqs=rqs.get(m, float("nan")),
        r_max_policy=fnum(aci.get(m, {}).get("R_max_policy_decomp"), float("nan")),
    ))

summary = {"pillar": "B-F24", "row": 22,
           "lecture": "F24 L11 Ben Mann (Anthropic) -- RSP + measuring agent capabilities",
           "citations": {
               "RSP": "Anthropic's Responsible Scaling Policy, 2023-09-19 (anthropic.com)",
               "Phuong2024": "Phuong et al., Evaluating Frontier Models for Dangerous "
                             "Capabilities, arXiv:2403.13793, DeepMind 2024-03-20"},
           "n_models": len(M), "hypotheses": {}}

# ================================================================ H1
# RSP yellow/red safety buffer in CEILING space.
# red line: point ceiling r_max crosses R*.  yellow line: upper-CI r_hi crosses R*.
# The conservative (yellow) rule fires earlier -> "protected zone".
# DECISIVE if all models have r_hi - r_max > 0 AND >=1 threshold where the
# conservative rule catches a model the point rule misses.
h1_rows = []
buffers = []
for m in M:
    buf = m["r_hi"] - m["r_max"]
    buffers.append(buf)
    h1_rows.append([m["model"], f"{m['r_max']:.4f}", f"{m['r_hi']:.4f}",
                    f"{buf:.4f}", f"{(m['r_hi']-m['r_lo']):.4f}"])
protected_counts = []
for Rstar in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    red = sum(1 for m in M if m["r_max"] >= Rstar)
    yellow = sum(1 for m in M if m["r_hi"] >= Rstar)
    protected = yellow - red
    protected_counts.append(protected)
    h1_rows.append([f"THRESHOLD_R*={Rstar:.2f}", f"red_cross={red}",
                    f"yellow_cross={yellow}", f"protected={protected}", ""])
p1 = write_tsv("rsp_h1_yellow_red_buffer.tsv",
               ["model_or_threshold", "r_max_or_red", "r_hi_or_yellow",
                "buffer_or_protected", "ci_width"], h1_rows)
h1_all_pos = all(b > 0 for b in buffers)
h1_catches = max(protected_counts) >= 1
h1_dec = h1_all_pos and h1_catches
summary["hypotheses"]["H1_yellow_red_buffer"] = {
    "all_buffers_positive": h1_all_pos, "min_buffer": min(buffers),
    "max_protected_at_threshold": max(protected_counts),
    "verdict": "DECISIVE" if h1_dec else "NULL", "evidence": p1}

# ================================================================ H2
# Elicitation gap (Phuong: measured capability is a LOWER bound; assume more capable).
# gap = R_max_policy (elicitation-adjusted) - R_max_observed >= 0, and the gap
# should be LARGER for poorly-elicited (low-RQS) models.
# DECISIVE if gap>=0 for all AND rho(gap, 1-RQS) > 0.5.
h2_rows = []
gaps, inv_rqs = [], []
for m in M:
    obs = m["r_max"]
    pol = m["r_max_policy"]
    if math.isnan(pol):
        continue
    gap = pol - obs
    gaps.append(gap)
    inv_rqs.append(1.0 - m["rqs"] if not math.isnan(m["rqs"]) else float("nan"))
    h2_rows.append([m["model"], f"{obs:.4f}", f"{pol:.4f}", f"{gap:.4f}",
                    f"{m['rqs']:.4f}", f"{(1.0-m['rqs']):.4f}"])
rho_gap, n_gap = spearman(gaps, inv_rqs)
h2_rows.append(["SPEARMAN_gap_vs_(1-RQS)", f"rho={rho_gap:.4f}", f"n={n_gap}",
                "", "", ""])
p2 = write_tsv("rsp_h2_elicitation_gap.tsv",
               ["model", "R_max_observed", "R_max_policy", "elicitation_gap",
                "RQS", "one_minus_RQS"], h2_rows)
h2_all_nonneg = all(g >= -1e-9 for g in gaps)
h2_dec = h2_all_nonneg and (not math.isnan(rho_gap)) and rho_gap > 0.5
summary["hypotheses"]["H2_elicitation_gap"] = {
    "all_gaps_nonneg": h2_all_nonneg, "rho_gap_vs_inv_rqs": rho_gap, "n": n_gap,
    "verdict": "DECISIVE" if h2_dec else "NULL", "evidence": p2}

# ================================================================ H3
# Forecast reliability gate. RSP: only rely on a forecast if it is reliable.
# rel_ci = (t80_hi - t80_lo)/t80_mean.  A model whose crossing-time CI spans
# >5x its own point estimate is "unforecastable" -> RSP demands max-conservative
# (assume immediate crossing). Quantify the unforecastable set.
# DECISIVE if we can partition models into forecastable vs unforecastable with a
# clean gap (>=1 model each side, and the split is not marginal).
h3_rows = []
rel_cis = []
for m in M:
    rel = (m["t80_hi"] - m["t80_lo"]) / m["t80"] if m["t80"] > 0 else float("nan")
    rel_cis.append((m["model"], rel, m["lam_bound"], m["rss"]))
    flag = "UNFORECASTABLE" if (not math.isnan(rel) and rel > 5.0) else "forecastable"
    h3_rows.append([m["model"], f"{m['t80']:.4f}", f"{m['t80_lo']:.4f}",
                    f"{m['t80_hi']:.4f}", f"{rel:.3f}", f"{m['lam_bound']:.3f}", flag])
unforecast = [x for x in rel_cis if not math.isnan(x[1]) and x[1] > 5.0]
forecast = [x for x in rel_cis if not math.isnan(x[1]) and x[1] <= 5.0]
# clean gap: min(unforecastable rel_ci) >> max(forecastable rel_ci)
gap_clean = (len(unforecast) >= 1 and len(forecast) >= 1 and
             min(x[1] for x in unforecast) > 2.0 * max(x[1] for x in forecast))
p3 = write_tsv("rsp_h3_forecast_reliability.tsv",
               ["model", "t80_mean", "t80_lo", "t80_hi", "rel_ci_width",
                "lam_at_bound_rate", "gate"], h3_rows)
summary["hypotheses"]["H3_forecast_reliability"] = {
    "n_unforecastable": len(unforecast), "n_forecastable": len(forecast),
    "unforecastable_models": [x[0] for x in unforecast],
    "clean_partition_gap": gap_clean,
    "verdict": "DECISIVE" if gap_clean else "NULL", "evidence": p3}

# ================================================================ H4  (the punchline)
# Scale is NOT a reliable RSP forecasting variable. RSP/frontier-safety implicitly
# assumes capability is forecastable from model scale. On the SAME verifiable-reward
# stack, test rho(params_B, r_max) and rho(params_B, t80); compare to
# rho(RQS, r_max) (elicitation quality). Expect scale NULL/negative, RQS strong.
# DECISIVE if |rho(params, r_max)| < 0.5 (scale fails) AND rho(RQS, r_max) > 0.5.
params = [m["params_B"] for m in M]
r_maxs = [m["r_max"] for m in M]
t80s = [m["t80"] for m in M]
rqss = [m["rqs"] for m in M]
rho_pr, n1 = spearman(params, r_maxs)
rho_pt, n2 = spearman(params, t80s)
rho_qr, n3 = spearman(rqss, r_maxs)
h4_rows = [
    ["rho(params_B, r_max)", f"{rho_pr:.4f}", f"n={n1}",
     "scale->ceiling: RSP forecasting variable"],
    ["rho(params_B, t80)", f"{rho_pt:.4f}", f"n={n2}",
     "scale->horizon: RSP forecasting variable"],
    ["rho(RQS, r_max)", f"{rho_qr:.4f}", f"n={n3}",
     "elicitation->ceiling: stack-relative variable"],
]
p4 = write_tsv("rsp_h4_scale_vs_stack.tsv",
               ["correlation", "spearman_rho", "n", "interpretation"], h4_rows)
h4_scale_fails = (not math.isnan(rho_pr)) and abs(rho_pr) < 0.5
h4_stack_wins = (not math.isnan(rho_qr)) and rho_qr > 0.5
h4_dec = h4_scale_fails and h4_stack_wins
summary["hypotheses"]["H4_scale_not_forecasting_variable"] = {
    "rho_params_rmax": rho_pr, "rho_params_t80": rho_pt, "rho_rqs_rmax": rho_qr,
    "scale_fails": h4_scale_fails, "stack_wins": h4_stack_wins,
    "verdict": "DECISIVE" if h4_dec else "NULL", "evidence": p4}

# ================================================================ H5
# Temporal forecast buffer & the RSP decision rule. Conservative planning fires
# at the EARLIEST plausible crossing t80_lo; lead time (buffer) = t80_mean - t80_lo.
# DECISIVE if lead>0 for all forecastable models AND the mean reserved fraction
# (lead / t80_mean) is material (>= 0.25 -- a real safety margin, not noise).
h5_rows = []
fracs = []
for m in M:
    lead = m["t80"] - m["t80_lo"]
    frac = lead / m["t80"] if m["t80"] > 0 else float("nan")
    fracs.append(frac)
    h5_rows.append([m["model"], f"{m['t80']:.4f}", f"{m['t80_lo']:.4f}",
                    f"{lead:.4f}", f"{frac:.3f}"])
valid_fracs = [f for f in fracs if not math.isnan(f)]
mean_frac = sum(valid_fracs) / len(valid_fracs) if valid_fracs else float("nan")
all_lead_pos = all((m["t80"] - m["t80_lo"]) > 0 for m in M if m["t80"] > 0)
h5_rows.append(["MEAN_reserved_fraction", "", "", "", f"{mean_frac:.3f}"])
p5 = write_tsv("rsp_h5_temporal_buffer.tsv",
               ["model", "t80_mean", "t80_lo_conservative", "lead_time_buffer",
                "reserved_fraction"], h5_rows)
h5_dec = all_lead_pos and (not math.isnan(mean_frac)) and mean_frac >= 0.25
summary["hypotheses"]["H5_temporal_buffer"] = {
    "all_leads_positive": all_lead_pos, "mean_reserved_fraction": mean_frac,
    "verdict": "DECISIVE" if h5_dec else "NULL", "evidence": p5}

# ---------------------------------------------------------------- tally & write
decisive = [k for k, v in summary["hypotheses"].items() if v["verdict"] == "DECISIVE"]
summary["n_decisive"] = len(decisive)
summary["decisive"] = decisive
summary["overall"] = ("DECISIVE" if len(decisive) >= 3
                      else "SUGGESTIVE" if len(decisive) == 2 else "NULL")
with open(os.path.join(OUT, "rsp_capability_forecasting_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"n_models={len(M)}  DECISIVE {len(decisive)}/5 -> {summary['overall']}")
for k, v in summary["hypotheses"].items():
    print(f"  {k}: {v['verdict']}")
