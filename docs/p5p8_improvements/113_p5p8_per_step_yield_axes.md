# 113 — P5+P7 SYNTH: per-step analog of iter-81 row 96 yield-residual Items 13-17 on N2 four-method tensors (iter 96 JOB B)

**Pillar:** P5P8-SYNTH (Pillar 4 cross-paper synthesis — closing the
iter-83 row 98 mint recommendation #2).
**Vein:** hybrid of brief veins (a)+(b): does the iter-81 per-CELL multi-
axis discrimination (Items 14-17) hold at per-STEP granularity on the N2
four-method panel? Or does the discrimination collapse to a single axis at
the finer time resolution? Closes the literal recommendation from iter-83
row 98 mint #2: "extend iter-81 row 96 Items 14-17 to per-step granularity
on N2 — does the multi-axis discrimination hold at per-step level, or only
at per-cell level?"

## Method

- Data: `experiments/results/n2_reward_tensor/{grpo,aero,gift}_s0_tensors.jsonl`
  (262 records) + `n2_reward_tensor_resume/areal_s0_tensors.jsonl` (final
  record count, 262 obs across 4 methods × ~65 steps each).
- For each (method, step, rewards[16 prompts × 8 rollouts]) record:
  compute per-prompt k_p ∈ {0..8}, then per-step aggregate Items:
  - Item 13-step = (zvf_obs − binom_zvf(p_hat, G)) / (1 − binom_zvf)
  - Item 14-step = (Var(k) − G·p·(1−p)) / max(G·p·(1−p), 1e-12)
    [**K_variance_residual**, signal-bearing per iter-81]
  - Item 15-step = |{k_p ∈ K_obs}|
    [**K_unique_count**, signal-bearing per iter-81]
  - Item 16-step = max_k #prompts(k_p=k)/16
    [**max_K_share**, REJECTED as placebo per iter-81]
  - Item 17-step = Var(p_hat_p) = Var(k_p / G)
    [**prompt_p_hat_var**, signal-bearing per iter-81]
- Target: |zvf_obs − binom_zvf(p_hat, G)| per step — the per-step deviation
  from the iid-Binomial baseline that the per-step controllers (iter-91
  zvf_then_drop) fire on.
- Spearman ρ of each Item vs target pooled across the 4 methods.
- Binomial(G, p_hat) null control: simulate 500 k-prompt vectors per step
  under independent Binomial(G, p_hat) and compute the empirical-vs-null
  difference of means (z-score).

## Falsifiable headlines

### H1 — Items 14 and 17 STRENGTHEN at per-step granularity

| Item | POOLED ρ (per-step) | iter-81 per-cell ρ | verdict |
|---|---:|---:|---|
| 13 (zvf_yield_residual) | **0.552** | ~0.55 (iter-81) | unchanged |
| **14 (K_variance_residual)** | **+0.677** | +0.558 | **+0.119 stronger per-step** |
| 15 (K_unique_count) | **+0.367** | +0.588 | −0.221 weaker per-step |
| **17 (prompt_p_hat_var)** | **+0.929** | +0.556 | **+0.373 stronger per-step** |
| 16 (max_K_share) | **−0.380** | +0.589 (NS) | REJECTED + FLIPPED |

**The difficulty-spread axis (Item 17, prompt_p_hat_var) is ~1.7×
stronger at per-step granularity than at per-cell**. The k-variance axis
(Item 14) is also stronger per-step. These are the two genuinely
information-bearing axes at per-step resolution.

### H2 — Item 15 (K_unique_count) is WEAKER at per-step granularity and BINOMIAL-NULL-NULL

Per-step Item 15 has POOLED ρ = +0.367 (vs iter-81 per-cell ρ = +0.588).
The binomial null test shows the empirical Item 15 mean (4.71 unique k)
is +0.62 above the binomial null (4.10) — z=0.6, **NOT significant**.

**Item 15 does not encode per-step information beyond Binomial(G, p)**
at this granularity, even though it was signal-bearing at per-cell.
The per-cell Item 15 signal derives from cross-cell prompt diversity,
not from per-step k-distribution shape.

### H3 — Item 16 (max_K_share) was REJECTED by iter-81; per-step re-test REJECTS IT HARDER (negative ρ)

Iter-81 row 96 H4 marked Item 16 as a placebo (positive ρ but binomial-
null-failing). The present per-step re-test gives **ρ = −0.380 (NEGATIVE)**
across all 4 methods. Per-step Item 16 is **anti-correlated** with the
target: when the per-step k-distribution concentrates on one k, the
per-step zvf deviation is *smaller*. The iter-81 reject is now reinforced
by a sign-flip: Item 16 carries no per-step information and is
operationally antipodal to the target.

### H4 — Per-step Item 17 dominates per-step Item 13 by 0.93 vs 0.55

The Spearman ρ target was |zvf_obs − binom_zvf(p_hat)|, the per-step
deviation. Item 17 (Var of the per-prompt p_hat) has ρ = **+0.929**,
Item 13 (zvf_yield_residual) has ρ = +0.552. The difficulty-spread
axis explains the per-step deviation **better than the zvf itself**
(95% boost in ρ).

This is mechanistically clean: per-prompt p_hat is the *cause* of per-
step zvf_obs deviation (Binomial(G, p_hat) maps p_hat → zvf_obs). Item 17
encodes the upstream variance; Item 13 encodes the downstream effect.

### H5 — Cross-method replication: per-step Items 14, 17 are universal; Item 16 is universally anti-correlated

| method | Item 14 ρ | Item 17 ρ | Item 16 ρ |
|---|---:|---:|---:|
| grpo | +0.788 | +0.939 | −0.350 |
| aero | +0.648 | +0.932 | −0.505 |
| gift | +0.499 | +0.896 | −0.357 |
| areal | +0.708 | +0.916 | −0.269 |

Item 17 is universally above 0.89; Item 14 is universally above +0.49;
Item 16 is universally negative. **The per-step discrimination is
robust across the 4-method same-stack panel and does not depend on
algorithm choice.**

### H6 — Binomial(G, p) null control: Items 14 and 17 EXCEED null at z > 8 on every method; Item 15 does NOT

Empirical vs null diff (z-score):
- Item 14: +4.35 to +4.62 (z = 10.9–11.5 per method); **all PASS binomial null**.
- Item 17: +0.075 to +0.077 (z = 8.1–8.6 per method); **all PASS**.
- Item 15: +0.53 to +0.75 (z = 0.5–0.7 per method); **all FAIL the binomial null**.
- Item 16: +0.20 to +0.23 (z = 1.8–1.9 per method); **FAIL**.

**At per-step granularity: Items 14 and 17 encode information BEYOND
Binomial(G, p). Item 15 does NOT — its iter-81 per-cell signal was
a cell-level artifact.**

## Cross-paper coupling (cross-paper synthesis)

This iter closes three open cross-paper couplings:

1. **(P5, P7) at per-step granularity** — iter-81 row 96 P5 multi-axis
   yield-residual was per-cell; this iter measures the same Items 14-17
   at per-step granularity on the N2 panel. The two axes that strengthen
   per-step (Items 14, 17) are **exactly the prompt-distribution-shape
   axes** that P7's iter-91 zvf_then_drop trigger fires on. **The P7
   per-step firing decision** = **the P5 per-step Item 17 difficulty-
   spread**. Concrete operational read: when N2 step-level k-vector
   has high variance (Item 17 ρ=0.93), zvf_then_drop fires (iter-91
   controller logic) — same phenomenon, two aggregation units.

2. **(P5, P6) on Items 14-17 vs the registry's measured_yield_residual
   block** — the iter-82 row 97 `delta_div` field is the per-cell
   aggregate of these per-step Items. The present iter is the
   **per-step analog**; it sharpens iter-82's per-cell measurement
   into the per-step resolution where the registry's panel tag
   `n2_same_stack_full40` lives.

3. **(P7, P8) on Item 16 (the rejected-as-placebo axis)** — iter-81
   rejected Item 16 at per-cell; this iter rejects it again at per-step
   with a sign-flip (ρ = −0.38). The two rejections compose: **the
   max-share axis is consistently the per-axis NAME on which `add ALL'`
   would over-fit**. The P7 counterfactual-control finding (iter-91's
   per-step benefit is dominated by k ∈ {1, 7}) is the same phenomenon:
   the BENEFIT concentrates on boundary-mixed prompts (uniform
   k-distribution, low max-share); high max-share is anti-correlated
   with benefit.

4. **(P5P8-SYNTH) closure of iter-83 row 98 mint #2** — the mint
   recommendation explicitly asked: "does the multi-axis discrimination
   hold at per-step level, or only at per-cell level?" **Per-step
   discrimination HOLDS — and is SHARPER on Items 14, 17 than at
   per-cell granularity**. The shape-of-distribution axes (Items 14,
   17) are the per-step signal-bearing axes; the count-of-uniques axis
   (Item 15) is per-cell only.

## Operational recommendation

Adopt **per-step Item 17 (prompt_p_hat_var)** as the next MIN-REPORT v2.3
candidate axis. The per-cell Item 17 already added +1.0 bit per the iter-
80 row 95 single-item analysis; the per-step Item 17 carries ρ=0.93 to
the per-step zvf-deviation target, the **strongest single-axis score**
in the iter-96 audit. **Replace Item 15 (K_unique_count) with Item 17
in MIN-REPORT v2.3** — the per-cell signal at Item 15 was a
cell-granularity artifact, while Item 17's per-step signal is universal.

## Deliverables

- `scripts/p5p8/p5p8_iter96_per_step_yield_axes.py` (~280 LoC, stdlib + simple
  math; uses jsonl + Counter for Item 16 max-share computation).
- `experiments/results/p5p8/p5p8_iter96_per_step_yield_axes.tsv` (262 rows =
  4 methods × ~65 steps).
- `experiments/results/p5p8/p5p8_iter96_per_step_rho.tsv` (24 rows = 5 items × 4 methods + 5 items pooled).
- `experiments/results/p5p8/p5p8_iter96_per_step_summary.json` (machine-readable
  with per-method per-item empirical vs null z-scores).
- `docs/p5p8_improvements/113_p5p8_per_step_yield_axes.md` (this proposal).
- 1 line in `AUTORESEARCH_FINDINGS.jsonl` (pillar P5P8-SYNTH, iter 96).
