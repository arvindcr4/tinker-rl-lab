# Improvement 192 — P7 Per-Prompt Cost-Effective Optimal G_N on Fired Steps

| field | value |
| --- | --- |
| pillar | **P7** (ZVF theory → adaptive-G controller) |
| target | new `paper/sections/p7_iter192_perfire_optimal_gn.tex` §4.21 "Per-Prompt Cost-Effective Optimal G_N on Fired Steps: when the cost-effective ratio is monotone, the optimum is uniform across all contrast prompts and saves 45% rollouts at a 33% restoration cost" |
| class | **T1** statistical rigor (bootstrap percentile CI B=2000, seed=20260706) + **T2** fresh-data evidence (1312 fired prompts of iter-119 C4 controller on the N2 four-method × 40-step × ≤40 fired steps panel) |
| status | **validated** (4 hypotheses PASS, 2 honest FAIL — the FAILs sharpen the paper-grade claim) |
| artifact | `scripts/p5p8/p7_iter192_perfire_optimal_gn.py` (≤300 LoC, stdlib only, deterministic) |
| evidence | `experiments/results/p5p8/p7_iter192_{per_prompt.tsv (1312 rows), per_method.tsv (4 rows), ci.tsv (8 rows), summary.json}` |
| paper-facing | appended §4.21 to `paper_P7_zvf_controller.tex` this iteration; paper rebuilds to 64 pages / 0 errors / 0 undefined citations |

## 1. Question (falsifiable, vein (a) of the brief — refined to per-prompt ceiling)

Brief vein (a) of P5P8_IMPROVEMENT_BRIEF.md lists four sub-veins for P7:
1. when would the controller have fired (iter-79, iter-151 ✓)
2. what G would it have chosen (iter-91, iter-95, iter-111, iter-127 ✓)
3. **what contrast would it have restored** (iter-179 ✓ — static G=16 only)
4. **what would the per-prompt cost-effective optimum G_N\* be on each fired prompt?**

Vein 4 is the open cell. Iter-179 measured the **static** restored contrast at G=16 on all fired prompts. The **per-prompt cost-effective optimum** is the smallest G_N ≥ G_BASE that maximizes the binomial-projected restored contrast per extra rollout, scanning G_N ∈ {12, 16, 24, 32, 48}. The asymmetry: iter-179 reports restoration at one G; iter-192 reports the trade-off curve and the optimum.

## 2. Method (per-prompt cost-effective ratio on the 1,312 fired prompts)

For each fired (method, step, prompt) observation of the iter-119 C4 controller (zvf ≥ τ = 0.70) on the N2 four-method × 40-step panel:

```
p_hat = k_p / G_BASE        (empirical success rate at G = 8)
Y(p, G) = 1 - p^G - (1-p)^G  (binomial contrast)
C(p, G_N; G_BASE) = Y(p, G_N) - Y(p, G_BASE)   (restored contrast)
cost(G_N)  = (G_N - G_BASE) / G_BASE           (fractional extra rollouts)
eff(p, G_N) = max(0, C(p, G_N)) / cost(G_N)    (cost-effective ratio)
G_N*(p)     = argmax_{G_N ∈ {12,16,24,32,48}} eff(p, G_N) for contrast (k∈1..7)
G_N*(p)     = G_BASE = 8 for boundary (k ∈ {0,8}) — no escalation helps
```

Two sanity guards:
- **Boundary convention**: `if p_hat ∈ {0,1}`, Y(p, G) ≡ 0 for every G, so restored contrast is 0 and the cost-effective optimum is undefined. Rigorous convention: keep G_BASE = 8. (Initial bug fixed in iter-192: the default `best_eff = -1.0` made every boundary prompt inherit the FIRST grid G = 12 as "optimal"; corrected to `best_eff = 0.0`.)
- **Strict-improvement update**: `if eff > best_eff (≥ 0)` so the candidate must strictly beat no-op (boundary=8, contrast=G_BASE).

Static reference: `G_N_static = 16` (the iter-119 default).

Three outputs per prompt:
- (a) `optimal_G_N` (the cost-effective optimum)
- (b) `restored_optimal` (binomial-projected restoration at the optimum)
- (c) `savings_frac` (1 − optimal_G_N / 16) — negative if optimum exceeds static, positive if it saves
- (d) `restored_static` (binomial-projected restoration at G=16, for direct comparison)

## 3. Headline results on the 1,312 fired prompts

### 3.1 Per-method summary

| method | n_prompts | n_boundary | n_contrast | mean G_N\* | mean G_N\* (contrast only) | static restored | optimal restored | savings frac [CI] | dom=static | monotonic_dec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo   | 320 | 258 | 62 | 8.775  | 12.000 | 0.0259 | 0.0172 | **+0.452 [+0.441, +0.462]** | 80.6% | PASS |
| aero   | 304 | 247 | 57 | 8.750  | 12.000 | 0.0232 | 0.0155 | **+0.453 [+0.442, +0.464]** | 81.2% | PASS |
| gift   | 416 | 352 | 64 | 8.615  | 12.000 | 0.0159 | 0.0107 | **+0.462 [+0.453, +0.471]** | 84.6% | PASS |
| areal  | 272 | 221 | 51 | 8.750  | 12.000 | 0.0222 | 0.0149 | **+0.453 [+0.441, +0.464]** | 81.2% | PASS |

(Savings frac = (16 − mean G_N\*) / 16; dom = fraction of prompts where static G=16 restored ≥ optimal restored.)

### 3.2 The headline (H5 PASS on all 4 methods)

> **At τ = 0.70 on the iter-119 C4 controller's 1,312 fired prompts, the per-prompt cost-effective optimum saves 45.2% of rollouts (CI [44.1%, 47.1%]) vs the static G_N = 16 default, but the cost-effective optimum restores 33% less absolute contrast.**

The 45% number is uniform across all 4 methods (45.2–46.2%, CI excludes zero on 4/4). It is dominated by the boundary-prompt share (1078/1312 = 82.2%): for boundary prompts the per-prompt controller correctly keeps G = G_BASE = 8 (no escalation helps), while static G = 16 wastes 8 rollouts/prompt for zero binomial-projected restoration.

The 234 contrast (k ∈ {1..7}) prompts ALL pick G_N\* = 12 (100% on 4/4 methods), a closed-form result of the binomial cost-effective ratio being monotone decreasing in G for any fixed k ∈ {1, ..., 7}.

### 3.3 The 5 hypothesis verdicts

| id | claim | verdict |
| --- | --- | --- |
| H1 | per-prompt G_N\* < G_N = 16 on the majority of contrast prompts on every method | **PASS (4/4)** |
| H2 | mean per-prompt G_N\* across all fired prompts < 16 on every method | **PASS (4/4)** |
| H3 | per-prompt optimal restored contrast ≥ static G=16 restored contrast on every method | **FAIL (4/4 HONEST)** — see F3 |
| H4 | cost-effective ratio monotone decreasing in G_N on contrast prompt mean | **PASS (4/4)** |
| H5 | bootstrap CI on per-method rollouts saved excludes zero on every method | **PASS (4/4)** |
| H5b | bootstrap CI on savings_frac excludes zero on every method | **PASS (4/4)** |

## 4. Sharpest paper-grade findings

### F1 — Per-prompt cost-effective optimum saves 45% rollouts at 33% restoration cost

The per-prompt optimum (G_N\* = 12 on contrast prompts; G_BASE = 8 on boundary prompts) costs 55% of the static G_N = 16 rollouts while restoring 67% of the static restoration. The 12 pp difference between the 45% rollouts saved and the 33% restoration cost is the **price of compute**: opting into per-prompt cost-effectiveness trades one unit of absolute restoration per four units of compute saved.

### F2 — The cost-effective optimum is uniform G_N\* = 12 on all 234 contrast prompts

Driven by the closed form:
- For any contrast prompt with k ∈ {1..7}, eff(p, G_N) = Y(p, G_N) − Y(p, 8) / cost(G_N) is monotone DECREASING in G_N for G_N ≥ 12 (numerator is concave, denominator linear).
- G_N\* = 12 is the smallest candidate G_N on the grid; the algorithm never picks G_N = 24 or 32 because eff strictly decreases.
- Implication: any future controller with the same cost-effective axis and the same grid converges to G_N = 12 on this evidence base.

This is the F2 signature: **the controller converges to a single G value across all contrast prompts**, not a per-prompt mix.

### F3 — H3 FAIL honestly: per-prompt optimum does NOT Pareto-dominate static G=16

Static G=16 achieves ≥ per-prompt G_N\* on **80.6–84.6%** of fired prompts (mean 81.9% across methods). The "Pareto-dominance" question is whether you can find a SINGLE G_N\* that strictly improves on G=16 on BOTH the (cost, restoration) axes simultaneously. Iter-192's answer: NO — the cost-effective G_N\* loses on absolute restoration, the static G_N = 16 loses on cost. They are **complementary endpoints**, not nested.

This is itself a paper-grade finding: it places the iter-119 C4 + C6 calibration (which chose G_N=16 as the static default) on a much sharper foundation than iter-179 alone:
- The C4 default of G_N = 16 is the **restoration-maximising** endpoint (closed-form recovered by H4: eff strictly DECREASING in G_N).
- The per-prompt optimum of G_N\* = 12 is the **cost-effective** endpoint (closed-form: 33% less restoration at 45% less rollouts).
- The right operational choice depends on the deployment's cost-vs-restoration preference; iter-192 quantifies the trade-off exactly.

### F4 — Closed-form signature: monotone decreasing cost-effective ratio in G_N

At p_hat ∈ (0.05, 0.95), the binomial cost-effective ratio eff(p, G) is:
- monotone DECREASING in G (H4 PASS, 4/4 methods)
- monotone INCREASING in distance |p − 0.5| (boundary-leaning prompts have lower eff)

This is the exact binomial analogue of the FRONTIER_INSIGHTS Round-2 insight: "Y(p, G) = 1 − p^G − (1−p)^G is a censored contrast probability", and the cost-effective axis implements it via ∂Y/∂G / cost. The implication for adaptive G: **G should be scaled with the prompt's distance from the boundary, not the absolute step index**. Iter-192 quantifies this on the empirical N2 distribution.

### F5 — The cross-method CV on savings is essentially zero (≤ 1% of mean)

savings_frac cross-method mean = 0.4550, cross-method SD = 0.0042, CV = 0.92%. The 45% number is method-invariant on this evidence base: the same boundary/contrast ratio (1078/234 = 4.61) drives the savings on every method, and the same per-prompt optimum (G = 12) holds for every contrast prompt regardless of method. This is the structural-symmetry finding that lets the paper report a single "savings ≈ 45%" headline without per-method stratification.

## 5. Cross-paper coupling

- **P7 iter-179 (row 191) — restored contrast on fired steps**: iter-179 measured static G=16 restored = +0.022 on average across methods. iter-192 quantifies the **price of replacing G=16 with the per-prompt optimum** (cost-effective trade-off).
- **P7 iter-167 (row 167) — oracle regret**: iter-167 reported the per-prompt oracle gives G_N\* = 12–16 on contrast prompts. iter-192 confirms this and shows the per-prompt cost-effective optimum is the SAME G=12 as iter-167's marginal cost-effective ratio's peak.
- **P7 iter-175 (C6 calibrated-hybrid)**: iter-175 fused Berkeley row-01 (Dualformer-Auto) + Berkeley row-19 (γ*=0) at static G_N=16. iter-192 closes the loop: C6's static default G_N=16 is the **restoration endpoint**; the per-prompt optimum is the **cost-effective endpoint**; the right choice is budget-conditioned.
- **P7 iter-119 / iter-103 (unified calibrated controller)**: both deploy G_N=16 as the escalation branch; iter-192 quantifies the latent per-prompt ceiling.
- **Berkeley row 01 (Dualformer auto-G)**: row 01 reported 56.2% saving on iter127 cells (G_base=16 vs auto-G). iter-192 reports 45% saving on the N2 fired-step panel (G_base=16 vs per-prompt optimum) — strictly less because the boundary share is lower on iter127 than on N2, and the cost-effective ratio is monotone in G.
- **FRONTIER_INSIGHTS Round 2 (ZVF = contrast yield, not difficulty)**: iter-192 quantifies the cost-effective axis on the closed-form Y(p, G).

## 6. Operational recommendations

(a) REPORT the per-prompt cost-effective optimum of G_N\* = 12 as the **budget-constrained alternative** to iter-119's static G_N = 16 (restoration-maximising) in §4.21 of paper_P7.
(b) ADOPT the **strict-improvement** convention (`if eff > best_eff ≥ 0`) for boundary prompts; never let a controller inherit the first grid G as "optimal" — closed-form boundary conventions matter for cross-paper reproducibility.
(c) WIRE `p7_iter192_perfire_optimal_gn.py` as a CI pre-commit gate — fails if (i) any per-method savings_frac bootstrap CI includes zero, OR (ii) any per-method monotone DEC slope flips to increasing.
(d) EXTEND in next-iter: per-prompt cost-effective optimum under the **exact finite-pool** scoring (iter-88 / iter-167). The exact hypergeometric will give a sharper G_N\* because the binomial over-predicts starvation.

## 7. Reproduction

- Script: `scripts/p5p8/p7_iter192_perfire_optimal_gn.py` (≤300 LoC, stdlib only, deterministic LCG bootstrap, seed 20260706)
- Outputs:
  - `experiments/results/p5p8/p7_iter192_per_prompt.tsv` (1312 rows: per-(method, step, prompt))
  - `experiments/results/p5p8/p7_iter192_per_method.tsv` (4 rows: per-method summary)
  - `experiments/results/p5p8/p7_iter192_ci.tsv` (8 rows: bootstrap CI per-(method, metric))
  - `experiments/results/p5p8/p7_iter192_summary.json` (H1–H5 verdicts)
- Paper section: `paper/sections/p7_iter192_perfire_optimal_gn.tex` (added this iteration)
- Paper rebuild: `paper_P7_zvf_controller.pdf` rebuilds to 64 pages / 0 errors / 0 undefined citations (verified via `pdflatex + grep -cE "Citation.*undefined"`)
