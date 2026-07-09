# P3 Hypothesis Stress Test — Group Size Non-Monotonicity and G=4 ~ G=32 Equivalence

Executed per `research_prompts/design/hypothesis-stress-test.md` (Ready-to-Copy Prompt contract).
Role: skeptical reviewer testing causal logic. Date: 2026-07-04.
Paper under test: `paper/paper_P3_group_size.tex` and its `sections/` inputs; data checked against `experiments/results/`.

---

## Input

### Hypothesis
Group size affects trainability non-monotonically; G=4 approximately equals G=32 token-normalized on capable models, extending the GRPO-is-secretly-DPO equivalence (arXiv 2510.00977, G=2 ~ G=16) to broader scale.

### Proposed mechanism (derived from the paper's own results/discussion sections, stated explicitly)
Reconstructed from `sections/group_size.tex`, `sections/group_size_reconcile.tex`, `sections/frontier_synthesis_group_size.tex`, `sections/p3_conclusion.tex`:

1. **All-pairs contrast identity.** The group-centered GRPO update is algebraically an all-pairs preference (DPO) update: `sum_i (r_i - r̄) s_i = (1/G) sum_{i,j} (r_i - r_j) s_i` (Eq. `fs-contrast`, frontier synthesis). For binary rewards with K successes the gradient scales with the pair count K(G−K); zero-variance groups are projected out (ZVF(p,G) = p^G + (1−p)^G).
2. **Capable-model saturation of contrast.** On capable models (high per-prompt success rate p), almost every non-degenerate group already yields a usable win–loss pair even at G=4; extra rollouts buy only redundant pairs (frontier synthesis paragraph explaining the measured 100.3% retention of G=2 vs G=16 on the near-ceiling task; `group_size_effect_dpo_check.tsv`).
3. **Sub-√G variance reduction.** The measured SNR grows at only ~52% of the √G ideal (SNR(G=16)=2.16 vs predicted 4.13; `group_size_iter15_snr.tsv`), so the marginal variance benefit of large G is heavily discounted; reward is statistically flat in G (0.971–0.984 across G∈{2,4,8,16}) and TOST passes equivalence for 5/6 pairs at ε=0.02 (`group_size_iter15_equivalence.tsv`).
4. **Per-token efficiency compensates.** Token-normalized, G=4 runs ~8× more optimizer updates than G=32 at fixed T, and the per-token gradient-efficiency estimator GE_tok is monotone decreasing in G at every budget (max always at G=4; Table `tab:groupsize-tokennorm` in `group_size_reconcile.tex`).

Chain: capable model → high contrast-yield per group even at G=4 → preference-density (not variance) is the binding quantity → sub-√G SNR gain of G=32 is offset by G=4's 8× step count at fixed tokens → token-normalized G=4 ≈ G=32.

### Known counterexamples (as supplied, mapped to repo artifacts)
- **It-Takes-Two SNR condition.** arXiv 2510.00977 predicts equivalence only where Monte-Carlo variance is not the binding constraint; outside that SNR regime G re-enters through the σ_R²/G baseline-noise term.
- **Contrast-yield analysis (task cites "iter138"; in this worktree the artifact is iter115).** `group_size_iter115_zvf_linkage.tsv` / `sections/group_size_iter115.tex` Finding 2: G=4 carries **4.15–5.03× more contrast yield (GU) per group** than G=32 at every budget, *yet retention collapses monotonically* 0.976 → 0.833 → 0.750 → 0.727 as T goes 1M → 4M → 16M → 64M (Spearman ρ = −1.000). No iter138 files exist under `experiments/results/` (max is iter135); the 4–5× number is iter115's.
- **Token-normalization confound.** Fixed-step comparison inverts the result (G=4 best, 52.1% last-10 reward, Table 6) vs fixed-token (G=32 best at T≥16M); the choice of normalization axis is itself a free parameter (`group_size_reconcile.tex`).

---

## Output

### 1) Weakest link

**The equivalence half of the hypothesis ("G=4 ≈ G=32 token-normalized on capable models") has never been tested on measured data: no measured G=32 cell exists anywhere in the benchmark, and every regime in which equivalence *is* observed is a regime where the outcome metric has almost no dynamic range.**

Concretely, the causal chain breaks at the joint "high contrast-yield at G=4 → token-normalized equivalence at G=32":

- The only measured equivalence evidence is G∈{2,4,8,16} on Qwen2.5-0.5B / arithmetic at p≈0.98 — a near-ceiling task whose entire held-out range across G is ~1.3 pp (`groupsize_zvf_sweep.json`; `group_size_effect_dpo_check.tsv` explicitly marks G=32 as "not measured on arithmetic sweep").
- The only G=32 data at all is the Qwen3-8B/GSM8K token-budget table, which the paper's own appendix flags as an *illustrative reanalysis*: "reconstructed point estimates from existing GRPO ablation logs (the FALLBACK_ROWS table in the script), **not** freshly measured per-seed runs; per-cell seed counts are not retained and no measured G×T sweep over {4,8,16,32,64} was executed" (`group_size_reconcile.tex`, caption of `tab:groupsize-tokennorm`).
- And within that reconstruction, the equivalence claim survives **only** at T=1M — the under-trained regime where both arms score ≤0.42 and the Δ CI spans zero (`group_size_g4_vs_g32_broader_scale.tsv`: retention 97.6% at T=1M, then 83.3%/75.0%/72.7%; `group_size_iter115_tost.tsv`: TOST at Wu's 2.4 pp bound FAILS with p=1.0 at every T≥4M; `group_size_iter135_threshold_tstar.tsv`: extrapolated T* for retention <95% is ~0.97M tokens).

So the hypothesis rests on an extrapolation across three axes simultaneously — group size (16→32), model scale (0.5B→"capable"), and task difficulty (ceiling→frontier) — with the single bridging dataset being synthetic, and that synthetic dataset *itself* contradicting the claim at every budget above 1M tokens.

### 2) Why this link is fragile

Three independent failure surfaces, all documented inside the paper:

**(a) Equivalence-by-compression.** Both regimes where "G=4 ≈ G=32-family" holds are regimes where *any* two G values would look equal because the metric cannot move: the measured arithmetic sweep is at ceiling (reward range 0.971–0.984, smaller than the 1.96·SE band at every G), and the T=1M reconstruction row is at the floor (accs 0.41 vs 0.42, both under-trained, CI [−0.07, +0.05] spanning zero). An equivalence claim validated only where the measurement has no headroom is unfalsifiable in exactly the regimes it was validated in — it is metric compression, not mechanism. The paper concedes this: "the budget at which it does generalize is the short-budget regime where … the problem is under-trained for all G" (`group_size.tex`).

**(b) The mechanism's own premise is contradicted by the paper's linkage analysis.** If preference density were the binding quantity (mechanism steps 2–3), G=4's 4.15–5.03× per-group GU advantage should protect it as budgets grow. It does not: retention collapses monotonically to 72.7% while the GU ratio stays >4× (`group_size_iter115_zvf_linkage.tsv`). The paper's own conclusion: "contrast yield is NOT the binding constraint … the mechanism that drives the G=4 penalty must therefore be gradient noise (σ_R²/G), not contrast starvation" (`group_size_iter115.tex`, Finding 2). This is precisely the It-Takes-Two SNR condition: equivalence holds only while baseline-estimator noise is not binding, and the paper's data says it becomes binding by T≈4M. The hypothesis extends the equivalence exactly into the regime where its enabling condition fails.

**(c) The normalization axis is a free parameter doing hidden work.** "Token-normalized" is one of at least three defensible budget axes (optimizer steps, optimizer-visible tokens, wall-clock/rollout cost), and the paper shows the verdict flips with the axis: fixed-step favors G=4 (52.1% best last-10 reward, Table 6); fixed-token favors G=32 at T≥16M; the per-token efficiency estimator GE_tok favors G=4 at *every* budget. A hypothesis whose truth value inverts under a defensible re-choice of normalization, with no independent argument for why optimizer-visible tokens is the causally correct axis, is not yet a causal claim. (Iter115 additionally converts the iso-accuracy penalty to 7.9× compute at acc=0.70 — under *that* normalization G=4 is catastrophically non-equivalent.)

### 3) Disconfirming check

**Lowest-cost first (Tier 0, zero new compute, ~hours of analysis):** Re-slice the *existing measured logs* (`experiments/results/groupsize_zvf_sweep.json`, per-step `step_log`, G∈{2,4,8,16}, 3 seeds, 40 steps) into token-matched pairs: compare G=4 at step 4s against G=16 at step s (equal optimizer-visible tokens, since tokens/step ∝ G), restricted to the mid-training window where per-step p∈[0.3, 0.8] — before ceiling compression engages. The hypothesis (token-normalized equivalence via preference density) predicts token-matched mean reward and held-out accuracy equal within ε=0.02 at every matched checkpoint. This tests the token-normalized equivalence claim on the only *real* runs in hand, outside the compressed regime, at zero cost. Its limits (G≤16, 0.5B model) are exactly why Tier 1 exists.

**Decisive check (Tier 1, the minimal run that can separate the readings — this is the handoff to the Minimal Decisive Experiment prompt):** Run ONE measured, token-matched pair — G=4 vs G=32, 3 seeds each, on a capable model at mid-difficulty (Qwen3-4B-class or Qwen3-8B on GSM8K, initial pass rate p₀ ∈ [0.3, 0.6], i.e., on the learning frontier, not at ceiling), at T = 4M optimizer-visible tokens per arm — the smallest budget at which the paper's reconstruction predicts non-equivalence (predicted Δ = +0.11). Evaluate held-out accuracy with paired bootstrap and TOST at ε ∈ {0.024, 0.05}; log per-step advantage variance, ZVF, and GU per group so the same run identifies the mechanism (contrast starvation vs gradient noise). Cost: 6 runs × 4M tokens = 24M sampled tokens total — roughly one row of the existing reconstruction, and the cheapest single cell that can either falsify the hypothesis or force retraction of the paper's illustrative table.

### 4) Result pattern that would force revision

Numeric triggers, pre-registered:

- **Primary (Tier 1):** If measured Δ = acc(G=32) − acc(G=4) at token-matched T=4M has a 95% CI lower bound > +0.024 (Wu's bound; equivalently retention < 90% with CI upper bound < 0.976), the hypothesis must be revised from "G=4 ≈ G=32 token-normalized on capable models" to "equivalence is budget-conditional and holds only in the under-trained regime, T ≤ T_equiv ≈ 1M tokens on this benchmark."
- **Mechanism kill-shot (same run):** If simultaneously GU(G=4)/GU(G=32) ≥ 2 while retention < 90%, the preference-density mechanism itself is falsified for this regime (gradient noise σ_R²/G is binding, confirming iter115 Finding 2 on measured data), and the paper's "preference-density dial" framing must be demoted to the contrast-saturated (near-ceiling) regime only.
- **Symmetric trigger (protects against motivated reasoning):** If instead TOST passes at ε=0.05 (p < 0.05) at T=4M — i.e., measured equivalence where the reconstruction predicted Δ=+0.11 — then the FALLBACK_ROWS illustrative table and every downstream artifact built on it (iter107/111/115/135 retention curve, TOST table, compute-cost projection, T* extrapolations) must be withdrawn or re-derived from measured runs.
- **Tier 0 trigger:** If token-matched G=4@step-4s vs G=16@step-s on the existing measured logs differ by > 0.02 mean reward (paired across the 3 seeds, sign-consistent in ≥ 2 mid-training windows with p ∈ [0.3, 0.8]), token-normalized equivalence fails even within the measured G range, and the hypothesis is revised *before* spending any new compute.

---

## Notes on failure-mode compliance (per the prompt's iteration guide)

- *No superficial critique:* the weakest link is located at a specific joint of the mechanism chain, with the paper's own file paths and numbers as evidence.
- *Lowest-cost check first:* Tier 0 costs zero compute (re-slicing existing logs); the decisive Tier 1 run is sized to the single cheapest informative cell (24M tokens).
- *Numeric revision trigger:* CI bound > +0.024 / retention < 90% / GU ratio ≥ 2 / TOST ε=0.05 / paired Δ > 0.02, all pre-stated.
- *Discrepancy flag:* the task's "iter138 contrast-yield analysis" does not exist in this worktree (sections and results stop at iter135); the 4–5× contrast-yield-per-prompt figure is iter115's (`group_size_iter115_zvf_linkage.tsv`, GU ratio 4.15–5.03×). Treated as the intended counterexample.

## Handoff (for Minimal Decisive Experiment)

- `{{research_question}}` ← Section 3, Tier 1 paragraph (verbatim).
- `{{decision_needed}}` ← Section 4 (verbatim).
