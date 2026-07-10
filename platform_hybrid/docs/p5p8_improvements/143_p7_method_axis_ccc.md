# Iter 127 — P7 per-method axis breakdown of the calibrated controller

**Vein picked:** brief vein (b) at the **method-axis** granularity.
Fresh, not in 138 prior ledger rows.  Reuses iter-119's CCC bank
verbatim; audits the bank's per-(method, step) recommendation on the
N2 four-method panel (aero / areal / gift / grpo × 40 steps × G=8 × seed 0).

## Headline findings (4 falsifiable)

| H | Claim | Verdict | Evidence |
|---|---|---|---|
| **H1** | Methods differ on CCC axis (spread > 0) | **PASS** | spread = 5.4 (gift 23.6 → areal 18.2) |
| **H2** | Most aggressive method is gift (n_escalate ≥ 5) | **PASS** | gift n_escalate = 26/40, top rank |
| **H3** | Reward rank aligns with G_ccc rank | **REPORTED** | rank.gift,grpo,areal,aero ≠ rank.G.gift,grpo,aero,areal |
| **H4** | Top vs bottom CI non-overlap | **TENSION** | gift [20.0, 27.2] vs areal [14.6, 22.4] — overlap [20.0, 22.4] |
| **Side** | FAST regime ever triggered on N2? | **NEGATIVE** | 0 / 160 step-method rows hit $z_{\mathrm{obs}} < 0.50$ |

## Per-method CCC table

| Method | Mean $G_{\mathrm{ccc}}$ [95% CI] | $\Delta z$ (contrast gain) | DEG / BAS / FAST | Mean reward |
|---|---|---|---|---|
| gift  | 23.6 [20.0, 27.2] | $-0.198$ | 26 / 14 / 0 | 0.8447 |
| grpo  | 20.0 [16.4, 23.6] | $-0.185$ | 20 / 20 / 0 | 0.8342 |
| aero  | 19.4 [15.8, 23.0] | $-0.174$ | 19 / 21 / 0 | 0.8275 |
| areal | 18.2 [14.6, 22.4] | $-0.155$ | 17 / 23 / 0 | 0.8287 |

## H3 — reward rank vs G rank (the orthogonality test)

| Rank | By $G_{\mathrm{ccc}}$ | By mean reward |
|---|---|---|
| 1 | gift  | gift  |
| 2 | grpo  | grpo  |
| 3 | aero  | areal |
| 4 | areal | aero  |

The reward rank is NOT the G rank — aero and areal swap. CCC is
**not** a degenerate re-labelling of `reward_mean`. The DEGENERATE
trigger ($z_{\mathrm{obs}}\geq 0.70$) is partially orthogonal to
reward: a method can have mid-pack reward (areal at 0.8287) but
generate more DEGENERATE steps via higher p_hat asymmetry.

## Cross-paper coupling

- **P7 iter-119 row 133** (CCC unification §4.17) — iter-127 is the
  audit of the same CCC bank at the method axis.
- **P7 iter-115 row 129** (N10 multiseed Adaptive-G*) — iter-115 used
  $\tau=0.70$ DEGENERATE threshold; iter-127 confirms the threshold
  fires on the N2 panel at 17 to 26 / 40 per method.
- **P6 iter-126 row 139** (per-delta measured-evidence tier) — gift
  is tier-A (n_sig≥3 AND n_panels≥2); gift is also the CCC's most
  aggressively controlled method on the same panel.
- **P5 iter-125 row 138** (chained $\eta^2$) — iter-125 ranks the
  four N2 methods by algo-axis signal (zvf × task slice R=10.32,
  pcd × task slice R=12.62); iter-127 ranks them by CCC's
  recommended $G$.  Two distinct method-orderings.
- **Berkeley row 01** (Dualformer-auto-acc) — iter-127 confirms
  the FAST regime (Dualformer $G_{\min}{=}2$ to $4$) is **dormant**
  at N2 granularity; never triggered in 160/160 step-method rows.

## Operational recommendation

1. **Method-stratified budget planning** — gift consumes
   $\approx 23.6$ rollouts/step on average (escalation budget
   ~$3\times G_{\mathrm{base}}$); areal/aero consume $\approx 19$
   ($\approx 2.4\times G_{\mathrm{base}}$). Cross-method budget
   uniformity is a hidden cost driver.

2. **FAST regime is dormant on arithmetic-reward same-stack runs**
   at $G_{\mathrm{obs}}{=}8$. Do not budget for FAST savings at
   the N2 scale.

3. **Pair-method comparison** shows one-third of step decisions
   move methods into different regimes; do not collapse
   multi-method evaluation into a single trace — method-stratify
   the regime histogram.

4. **Cross-method reward rank differs from CCC G-rank** — the CCC
   is not a degenerate re-labelling of the reward mean, which
   validates the DEGENERATE trigger as a genuinely
   orthogonal-channel signal.

## Artifacts

- `scripts/p5p8/p7_iter127_method_axis_ccc.py` (~280 LoC,
  stdlib only, deterministic Bisection)
- `experiments/results/p5p8/p7_iter127_method_axis_ccc.tsv`
  (4 rows × 16 cols)
- `experiments/results/p5p8/p7_iter127_method_step_recommendation.tsv`
  (160 rows × 13 cols)
- `experiments/results/p5p8/p7_iter127_regime_mix.tsv`
  (12 rows: 4 methods × 3 regimes)
- `experiments/results/p5p8/p7_iter127_summary.json`
- `paper/sections/p7_iter127_method_axis_ccc.tex` (~110 lines, NEW §4.18)
- `paper/paper_P7_zvf_controller.pdf` rebuilds to **58 pages / 0 errors / 0 undefined citations**
  (was 57, +1 page from new section §4.18).

## Pre-existing build error fixed

A pre-existing undefined reference `sec:p7-design-rules` in
`p7_iter107_tautransfer.tex` was retargeted to the actual label
`sec:p7-rules` in `p7_design_rules.tex`. The build warning was
present in the iter-119 build log (`Reference `sec:p7-design-rules'
on page 47 undefined`); fixing it cleans the build to 0 / 0 as the
brief requires.
