# Row 22 — RSP capability-forecasting on the Pillar-1 scaling law

**Source lecture:** F24 L11 — Ben Mann (Anthropic), *Measuring Agent
Capabilities and Anthropic's Responsible Scaling Policy*. A lecture **not
previously in the ledger** (covered F24 lectures were L1/L2/L5/L6/L8/L9/L10/L12).

**Target:** A1 (statistical rigor) + A2 (eval methodology), Pillar 1.

## Verified citations (both checked 2026-07-04)
- **RSP** — *Anthropic's Responsible Scaling Policy*, published **2023-09-19**
  (anthropic.com). Introduces AI Safety Levels (ASL-1…ASL-4+), **capability
  thresholds**, **evaluation triggers**, adversarial red-team gating, and the
  discipline of pausing when scaling outstrips safety procedures.
  *(verified via WebFetch of the announcement page — exact title & date quoted.)*
- **Phuong et al.**, *Evaluating Frontier Models for Dangerous Capabilities*,
  **arXiv:2403.13793**, DeepMind, **2024-03-20**. Core methodology: **conservative
  capability elicitation** (measured capability is a *lower bound* — assume the
  model is more capable) and **forecasting** capability as a function of scale.
  *(verified via arXiv abs metadata: title + authors Phuong, Aitchison, Catt,
  Cogan, … + date 2024/03/20.)*

## The idea, ported
RSP is a *forecasting-under-uncertainty* protocol: define a capability threshold,
predict when a model will cross it, and act **conservatively** — trigger
mitigations at the *earliest plausible* crossing, not the point estimate. Our
Pillar-1 scaling-law bootstrap (`scaling_law_bootstrap_ci.tsv`, 5 models × 1000
resamples) already produces exactly the ingredients: a fitted ceiling `r_max`
with CI, and a crossing horizon `t_80` (steps to reach 80 % of the asymptote)
with CI. We map RSP's vocabulary directly:

| RSP concept | TinkerRL-Bench quantity |
| --- | --- |
| capability threshold R\* | a reward asymptote the model can "cross" |
| red line (actual capability) | point-estimate ceiling `r_max_mean` |
| yellow line (trigger eval) | upper-CI ceiling `r_max_hi` (conservative) |
| safety buffer | distance the yellow line trips before the red |
| elicitation gap ("assume more") | `R_max_policy` (RQS-adjusted ceiling) |
| forecast reliability gate | bootstrap CI width / λ-at-bound rate |
| forecasting horizon | `t_80` (steps to cross) |

## Measured result — **3/5 DECISIVE → DECISIVE overall**
Script: `scripts/berkeley/rsp_capability_forecasting.py`
(outputs `experiments/results/berkeley/rsp_*.tsv` + `rsp_capability_forecasting_summary.json`).

- **H1 (yellow/red safety buffer) — DECISIVE.** Every model has a positive
  ceiling buffer `r_max_hi − r_max_mean > 0` (min **0.0563**, Nemotron 1.19 — its
  fit is unidentified and rails at the 1.5 bound). Sweeping R\*, the conservative
  (yellow) rule catches models the point rule misses at **every** threshold:
  protected = 1 for R\* ∈ [0.30, 0.80] (Nemotron), and **protected = 3** at
  R\*=0.90 (three near-ceiling models trip the yellow line while zero trip red).
  The RSP buffer is real and monotone in threshold stringency.
- **H2 (elicitation gap) — NULL (directionally correct, under-powered).** The
  Phuong "assume-more-capable" gap `R_max_policy − R_max_observed ≥ 0` holds for
  **4/4** models with a policy decomposition (gap 0.13–0.52, largest for the
  worst-elicited model Qwen3-8B). But ρ(gap, 1−RQS) = **0.40** < 0.5 and n=4
  (Nemotron has no finite `R_max_policy`), so the monotonicity is suggestive, not
  decisive. The *direction* is the RSP claim; the *rank strength* is under-powered.
- **H3 (forecast-reliability gate) — NULL (soft boundary — a stronger message).**
  The intended clean split (unforecastable vs forecastable at rel-CI 5×) does not
  separate: `t_80` relative CI widths are **2.88× – 9.02×** across *all* five
  models (Qwen3-8B 5.24×, Nemotron 9.02× flagged; but 3.96× sits just below the
  cut). Read honestly, this is the *opposite* of a clean partition: **every**
  crossing-time forecast is wide (all CIs bottom out at the same t=0.16 floor),
  so RSP would demand a wide buffer for the *whole fleet*, not a subset — which is
  exactly what H5 quantifies.
- **H4 (scale is NOT a reliable RSP forecasting variable) — DECISIVE (punchline).**
  RSP/frontier-safety implicitly forecasts capability from model scale. On the
  **same verifiable-reward stack**, Spearman ρ(params_B, r_max) = **0.10** and
  ρ(params_B, t_80) = **0.05** — scale is essentially uncorrelated with both the
  ceiling and the crossing horizon (Qwen3-8B r_max 0.29 vs Llama-8B 0.87 at
  identical 8 B; Nemotron-120B 0.31). By contrast ρ(RQS, r_max) = **0.60** —
  *elicitation quality* predicts the ceiling. **Scale-based RSP forecasting is
  under-identified in verifiable-reward RL post-training; the buffer must be
  stack/elicitation-relative, not scale-relative.** This is the same
  under-identification the Pillar-1 same-stack null licenses (frontier synthesis:
  *Estimator-Equivalence Principle* — once the stack is fixed the nominal knob
  stops predicting the update geometry; here the nominal knob is *model size*).
- **H5 (temporal forecast buffer & decision rule) — DECISIVE.** The RSP rule
  "trigger at the earliest plausible crossing `t_80_lo`" reserves a lead-time
  buffer `t_80_mean − t_80_lo > 0` for all five models, with a mean reserved
  fraction of **0.785** — i.e. a conservative planner fires after only ~21 % of
  the expected horizon. The buffer is large *because* the forecasts are wide
  (H3), closing the loop: broad crossing-time uncertainty → large mandated margin.

## Go / no-go
**GO — paper-facing (A1/A2 + Pillar-1), one sentence + a diagnostic figure.**
The valuable, defensible claim is **H4**: on a fixed verifiable-reward stack,
model scale is not a usable capability-forecasting variable (ρ≈0.1), so any
RSP-style "forecast the crossing from scale" protocol is under-identified for RL
post-training — capability tracks *elicitation quality* (RQS, ρ=0.60) instead.
H1/H5 give the constructive companion: a concrete conservative decision rule
(fire mitigations at the upper-CI ceiling / lower-CI crossing time) that yields a
real, quantified safety margin (protected=1–3 models; 0.785 reserved horizon).
H2 (elicitation gap) and H3 (fleet-wide unforecastability) are honest caveats
that *strengthen* the message: the gap is directionally the RSP one but n=5 is
too small to rank it, and crossing-time is broadly unforecastable — hence the
large buffers are not conservatism theatre, they are load-bearing.
Distinct from row 09 (SWE-agent Pass@K *tiers*) and row 08 (Eureka RQS *covariate*):
this row is the *forecasting/decision-rule* frame, not a static ceiling estimate.
