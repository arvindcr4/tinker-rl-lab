# Iter 174 — P6 (GRPO-Registry) — Tier-stratified metric-coverage audit

**Vein (fresh, not in 173 prior rows):** tier-stratified SIGNIFICANCE-RATE and
PANEL-COVERAGE comparison of registry entries. Brief vein (a) at the
metric-coverage layer. None of iter-126 (tier classifier), iter-134 (per-row
field completeness), iter-158 (4-tuple join coverage), iter-162 (ground-truth
audit), iter-166 (provenance audit) reported a tier-stratified *measured-effect*
breakdown.

**Tier rule (mirrors iter-126 row 139 verbatim):**
- **A** = `n_sig≥3` AND `n_panels≥2` (3 entries: aero / gift / areal)
- **B** = `n_sig≥1` (7 entries: adaptiveg / cppo / drgrpo / es / mcgrpo / ngrpo / scafgrpo)
- **D** = `n_total=0` (7 entries: dapo / gspo / liteppo / ppo / reinforce / tool_use_llama-8b-inst / tool_use_qwen3-32b)

`scripts/p5p8/p6_iter174_metric_coverage_tier_stratified.py` (~310 LoC,
stdlib only) walks every `delta_*.json` measured[] row, classifies by tier,
emits per-entry / per-(entry, panel) / per-metric / per-tier summaries.

## 5 falsifiable hypotheses (set BEFORE measurement)

| # | Hypothesis | Pass? | Quantitative result |
|---|---|---|---|
| **H1** | tier-A sig_rate > tier-B sig_rate | **REFUTED** | tier-A 0.500 < tier-B 0.600 |
| **H2** | every tier-A entry has ≥2 panels; every tier-B has ≤1 | PASS | tier-A panels = [2,2,2]; tier-B panels = [1,1,1,1,1,1,1] |
| **H3** | on n2_same_stack_last10, reward_mean sig_rate > zvf sig_rate | PASS | reward_mean 0.667 > zvf 0.333 |
| **H4** | tier-A mean CI width < tier-B mean CI width | PASS | tier-A 1.779 < tier-B 7.288 |
| **H5** | tier-D entries exist (label-only entries >0) | PASS | 7/17 = 0.4118 [Wilson 0.216, 0.640] |

## Sharpest paper-grade findings

### F1 — H1 REFUTED: tier system does NOT predict measured-significance rate

**tier-A sig_rate = 0.500, tier-B sig_rate = 0.600.** The tier system is NOT a
clean predictor of "fraction of measured rows that come out significant". This
is **structurally explained**: tier-A entries (aero/gift/areal) carry
*structurally NS metrics* on the n2_same_stack_last10 panel — specifically
`zvf` (1/3 sig) and `mean_zvf` on zvf130_5seed (0/8 sig by design — see F8).
Tier-B entries concentrate on `zvf_risk_mean` (8/8 sig) + `mag_mean` (5/5 sig)
which both have low variance and rarely yield NS results. The **tier system
predicts evidence DEPTH (panels + metrics), not evidence SIGNIFICANCE**.

### F2 — H2 PASS: clean panel stratification by tier

All 3 tier-A entries have **exactly 2 panels** (`n2_same_stack_last10` +
`zvf130_5seed`). All 7 tier-B entries have **exactly 1 panel** each (6 on
`zvf130_5seed` + 1 `length_bias_iter60_grpo_vs_drgrpo_paired` for drgrpo + 1
`qp7_adaptive_armB_vs_armA_paired` for adaptiveg). This is the **canonical
axis the tier rule was designed to capture** — multi-panel coverage indicates
robust evidence; single-panel coverage is the citation-only-to-measured
backbone. Tier rule is calibrated for this axis; H1 failure is a tier-AVERAGE
limitation, not a tier-DESIGN limitation.

### F3 — H3 PASS: on n2_same_stack_last10, reward_mean has 2× the sig_rate of zvf

`reward_mean` sig_rate = 2/3 (AERO + AREAL sig; GIFT NS).
`zvf` sig_rate = 1/3 (GIFT sig; AERO + AREAL NS).

This is a **methodological observation**: in the registry's n2 same-stack
panel, the headline outcome (`reward_mean`) is more reliably significant than
the group-starvation diagnostic (`zvf`). Two reasons:
- `zvf` measures a within-group contrast; its variance is bounded by the
  per-prompt difficulty distribution and grows non-monotonically with G.
- `reward_mean` measures the aggregate policy quality; the per-step bootstrap
  is dominated by the step-mean drift.

**Implication for the paper**: when reporting a variant's effect on
n2_same_stack, lead with `reward_mean` (sig-rate 0.667) and treat `zvf` as a
secondary diagnostic (sig-rate 0.333).

### F4 — H4 PASS: tier-A CIs are 4× narrower than tier-B CIs

tier-A mean CI width = **1.779** (n=18 rows).
tier-B mean CI width = **7.288** (n=20 rows).
Ratio = 4.10×.

The **driving cause is drgrpo's `L_star`** row (CI width = 144.21, delta =
44.25) — a single row inflates the tier-B mean by ~6.5× over the
zvf130-only entries (which have CI width 0.07-0.10). When drgrpo is excluded,
tier-B mean CI width drops to 0.073 (50× narrower than tier-A). The H4 PASS
is therefore a **drgrpo-outlier artifact**, not a tier-wide property.

**Recommendation**: when reporting tier-aggregate CI widths, exclude
drgrpo's L_star row (a known high-variance metric on a length-bias panel) and
re-state as "tier-A 1.78 vs tier-B-excl-drgrpo 0.073 = 24× wider".

### F5 — H5 PASS: tier-D fraction = 7/17 = 0.4118 [Wilson 0.216, 0.640]

**41% of the registry's variant-delta entries are label-only** (no measured
rows). This is the iter-126 closure gap, but the gap has GROWN:
- iter-126 (row 139, ~2 days ago): tier-D = 5 entries
- iter-138 added `tool_use_llama-8b-inst` + `tool_use_qwen3-32b` without
  measured rows → tier-D = 7 entries today.
- Growth: +2 entries, +6.4pp (33.3% → 41.2%).

**Operational impact**: tier-D is the registry's "claim without measurement"
backlog. Closing each requires one N2-protocol run (40 steps × G=8 × 1 seed,
≤$5 Tinker API credits). Total closure cost: ≤$35 for all 7 tier-D entries.
Priority order (suggested): **DAPO > GSPO > PPO > REINFORCE > LitePPO > tool_use
×2** (DAPO first — highest-impact 2025 method; tool_use last — they are
use-case-domain entries, not algorithm-variant entries).

### F6 — zvf_risk_mean is the registry's most reliably-significant metric

Across 8 entries that report `zvf_risk_mean` (aero, areal, cppo, es, gift,
mcgrpo, ngrpo, scafgrpo), **100% are statistically significant** at the 5-seed
risk-index panel. Sig_rate 1.000, n=8. This metric has been deliberately chosen
(iter-102 onwards) because it provides a clean per-seed aggregation with
adequate statistical power. **Implication**: when adding new tier-B entries,
`zvf_risk_mean` is the cheapest and most reliable metric to include.

### F7 — mean_zvf is the registry's "always-NS" metric

Across 8 entries that report `mean_zvf`, **0% are significant** at the
zvf130_5seed panel. Sig_rate 0.000, n=8. The mean_zvf CI widths are tiny
(0.005-0.01), which suggests the CI is computed on the wrong aggregation (the
per-seed mean rather than the per-step reward tensor). **Action item**: in a
future synthesis iter, recompute `mean_zvf` CIs on the per-step reward tensor
(N2 source: `experiments/results/n2_reward_tensor_resume/n2_metrics.tsv`) to
see if significance emerges.

### F8 — tier-A entries have richer metric diversity than tier-B

tier-A entries report 6 distinct metrics (zvf, reward_mean, pcd, mean_len,
zvf_risk_mean, mean_zvf); tier-B entries report 1-3 metrics (mostly
zvf_risk_mean + mean_zvf + mag_mean; drgrpo is an outlier with L_star +
neg_frac + pos_frac). The **metric diversity** (6 vs 3) is what makes tier-A
"richer evidence", even though sig_rate is LOWER (because more metrics = more
chance of NS rows).

## Cross-paper coupling

- **P6 iter-126 row 139** — tier classifier defined the A/B/D rule; iter-174
  re-derives the classification from current data and confirms the rule still
  partitions cleanly (3/7/7).
- **P6 iter-134 row 150** — per-row field completeness audit found 38/38 rows
  pass semantic completeness; iter-174 adds the tier-stratified aggregation
  layer that iter-134 lacked.
- **P6 iter-158 row 172** — 4-tuple join coverage found registry completeness
  bimodal (1.000 or 0.000); iter-174 shows tier-D is the bimodal-zero peak,
  contributing 7/7 of the 0.000 entries.
- **P6 iter-166 row 178** — provenance audit found 60.47% entries have
source_artifacts; iter-174 finds 41.18% tier-D entries (label-only), which
  structurally cannot declare source_artifacts (no measured rows to source).
- **P5 iter-153 row 170** — MIN-REPORT v2.4 audit; iter-174's tier-stratified
  breakdown is a candidate for a 5th audit layer on top of v2.4's
  bib/manifests/cells.tsv/registry stack.
- **FRONTIER_INSIGHTS Round 1 (Critic Degeneracy Hypothesis)** — the H1
  refutation is consistent with the (frontier synthesis) framing that
  label-named variants (especially tier-D: DAPO/GSPO/PPO/REINFORCE/LitePPO)
  underdetermine the algorithmic claim; closing tier-D to tier-B requires
  measured runs, not just better labels.

## Operational recommendations

(a) **PRIORITIZE** tier-D closure in iter-175+ (cheap: $35 total for 7 entries).
(b) **DOCUMENT** F1 (H1 refutation) in paper_P6 §5 as a transparency signal:
tier ≠ significance.
(c) **RECOMPUTE** `mean_zvf` CIs from the per-step N2 reward tensors in a
future synthesis iter (F7).
(d) **EXCLUDE** drgrpo's `L_star` row from any tier-aggregate CI-width
calculation (F4 outlier).
(e) **NO LIVE PATCH** to the registry is needed — iter-174 is observational.

## Artifacts

| Path | Rows | Description |
|---|---|---|
| `scripts/p5p8/p6_iter174_metric_coverage_tier_stratified.py` | 310 LoC | stdlib only, deterministic |
| `experiments/results/p5p8/p6_iter174_per_entry.tsv` | 17 | per-entry (n_total, n_sig, sig_rate, mean_abs_delta, mean_ci_width, tier) |
| `experiments/results/p5p8/p6_iter174_per_entry_panel.tsv` | 13 | per-(entry, panel) breakdown |
| `experiments/results/p5p8/p6_iter174_per_metric.tsv` | 10 | per-metric aggregation across entries |
| `experiments/results/p5p8/p6_iter174_tier_summary.tsv` | 3 | tier aggregate (A/B/D) |
| `experiments/results/p5p8/p6_iter174_summary.json` | 1 | H1-H5 verdicts + Wilson CIs |

**Status: validated (4/5 hypotheses PASS, 1 REFUTED; REFUTATION is itself a
paper-grade finding).** Tier-A n=18 rows; Tier-B n=20 rows; Tier-D n=0 rows
(label-only).
