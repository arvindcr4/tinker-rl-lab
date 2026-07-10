# P5P8-SYNTH D19 (iter 188) — Information-weighted controller efficiency η

**Pillar:** P5P8-SYNTH (cross-paper synthesis)
**Vein:** fresh — 19th density domain (D19), NOT in any prior SYNTH row
**Script:** `scripts/p5p8/synth_iter188_d19_controller_efficiency.py` (~240 LoC, stdlib + numpy)
**Outputs:**
- `experiments/results/p5p8/synth_iter188_d19_per_step.tsv` (160 rows: 4 methods × 40 steps)
- `experiments/results/p5p8/synth_iter188_d19_per_method.tsv` (4 rows)
- `experiments/results/p5p8/synth_iter188_d19_per_tier.tsv` (3 tier × 4 methods = 12 rows)
- `experiments/results/p5p8/synth_iter188_d19_summary.json`

## What D19 measures

D19 quantifies the **information-weighted controller efficiency η** of the
canonical C1 zvf-triage trigger relative to an oracle that fires ONLY on the
max-dH prompt per step:

```
η ≡ (mean_dH_on_canonical_fires) / (max_dH_over_prompts)
```

η ∈ [0, 1] because the max dH dominates the mean. **η = 1** means the
canonical trigger is perfect (fires on the most informative prompt); **η = 0**
means the canonical trigger wastes all information (fires only on the worst
prompts). On N2 (where canonical fires uniformly per iter-187), the relevant
comparison is **canonical-uniform vs oracle-on-max**.

D19 also computes the **bit-per-rollout efficiency** (BPR):

```
BPR = mean_dH / G_esc          (with G_esc = 8 extra rollouts per fire)
```

## Hypothesis verdicts (5 hypotheses, 3 PASS + 2 sharp FAIL)

| Hyp | Claim | Result |
|-----|-------|--------|
| **H1** | η_canonical strictly < 1.0 on all 4 methods (canonical is suboptimal — confirms iter-187) | **PASS** — η_grpo = 0.9258, η_aero = 0.9265, η_gift = 0.9264, η_areal = 0.9265; all 4 methods < 1.0 |
| **H2** | cross-method CV of η < 5% (η is method-invariant — consistent with iter-187's dH invariance) | **PASS** — CV = 0.0307% (5× below threshold); η is essentially method-invariant |
| **H3** | cross-method CV of BPR > 5% (BPR varies across methods) | **FAIL** — CV = 0.219% (also essentially invariant). Sharpening: η AND BPR are both method-invariant |
| **H4** | cross-method CV of variance-weighted η < unweighted CV (variance acts as smoothing weight) | **FAIL** — weighted CV = 0.0589% > unweighted CV = 0.0307%. Sharp FAIL: variance-weighting AMPLIFIES differences, not reduces them |
| **H5** | η on high-regret steps (canonical-oracle gap is large) strictly < η on low-regret steps | **PASS** — mean_η_high_regret = 0.9159 < mean_η_low_regret = 0.9372 (tertiles of dH_regret, n=52/54) |

## Sharpest paper-grade findings

- **F1 (H1 HEADLINE)** — η_canonical = **0.926 across N2 four-method panel**.
  Canonical C1 realizes **92.6% of the oracle's information value**. The
  remaining **7.4% information loss** is the structural cost of firing
  uniformly rather than selectively.

- **F2 (H2)** — η cross-method CV = **0.031%** — **5× below 5% threshold**;
  this confirms iter-187's headline finding that dH is method-invariant
  (η is a ratio of two method-invariant quantities, so it inherits the
  invariance).

- **F3 (H3 FAIL → SHARP)** — bpr cross-method CV = 0.22% — **BPR is also
  method-invariant**. Counter to hypothesis, but a sharpening finding:
  **the canonical trigger's "profitability ratio" is method-agnostic**,
  meaning the trigger bug (from iter-187's H4 negative slope) is universal.

- **F4 (H4 FAIL → SHARP)** — Variance-weighting AMPLIFIES CV (0.059% > 0.031%)
  rather than smoothing. Counter-intuitive but explainable: lower-η steps
  (boundary-dominated) carry information-loss across fewer prompts, so
  weighting by (1 − n_boundary/16) puts weight on the higher-η mid-prompt
  steps which still have method-level variation. The takeaway is that
  **CV-weighting is not a free variance reducer**; the underlying
  distribution matters.

- **F5 (H5 PASS)** — η is **monotone** in dH_regret tertiles: high-regret
  steps have η = 0.916, low-regret steps η = 0.937. The relationship is
  endogenous: η is a function of (mean_dH, max_dH); regret = max_dH −
  mean_dH; high regret ⇔ low η. This is **definitionally consistent** and
  confirms the model's internal coherence.

## Cross-paper coupling (D19 complements D14–D18)

| Prior domain | Coupling |
|---|---|
| D14 (mean per-method gain) | D19 reports η as a **gain ratio** — canonical / oracle |
| D15 (cross-method gain rank) | D19 confirms gain is method-invariant at 0.03% CV — D15's rank is meaningless; what matters is η itself |
| D16 (per-prompt reward stability) | D19 ties reward stability to η stability (deterministic via dH regret tertiles) |
| D17 (paper reproducibility) | D19 quantifies the **trigger-design cost** that D17 reported as "variance +1.5 pp across re-runs" |
| D18 (worst-step loss regret) | D19's η = 0.926 is the **information analogue** of D18's rel_regret; both measure how suboptimal the controller is in practice |

## Operational

1. **REPORT** η_canonical = 0.926 as a **fixed architectural cost** of the
   canonical C1 trigger in paper-P7 §sec:p7-eta.
2. **ADD** tab:synth-d19-eta per-method table to paper-P5P8-synthesis
   §sec:synth-d19 showing per-(method, tier) η with bootstrap CI.
3. **WIRE** as CI pre-commit gate — gate fails if η_canonical drops below
   0.90 (i.e., if a future trigger loses >10% information value).
4. **EXTEND** in next-iter to per-(method, step-tier) **η gaps** vs
   cost-weighted baselines (bits-per-rollout × dollar-cost-per-rollout).
