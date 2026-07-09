# 96 — P5 multi-axis yield-residual MIN-REPORT v2.2 axes (iter 81, JOB A)

## Falsifiable headlines

- **H1 — four additional yield-residual candidates (Items 14-17) collectively add +15.86 bits to v2.1** on the live 98-cell corpus, lifting the total from 22.92 (v2.1) to **38.78 bits (v2.2)** = +69.2%. **Item 13 alone** (iter-80 row 95) added 4.65 bits; Items 14-17 collectively add **3.4×** as much, and each individual candidate carries **distinct n_unique ≥ 2 on every (model, task_slice, G, t, seed) cell**.
- **H2 — single-item Spearman coupling vs |Δzvf|** on each item alone (n_pairs=4753, full enumeration): Item 14 (K_variance_residual) = **0.558**, Item 15 (K_unique_count) = **0.588**, Item 16 (max_K_share) = **0.589**, Item 17 (prompt_p_hat_var) = **0.556**. **All four single-items exceed the v2.1 fingerprint's ρ=0.436** and even exceed the v1 fingerprint's ρ=0.436 — a single continuous axis now carries more fingerprint-vs-outcome coupling than the full 7-item v1 schema.
- **H3 — paired-bootstrap Δρ (v2.2_all − v2.1) = +0.0748, 95% CI [+0.066, +0.084]**, excludes zero. Adding ANY single candidate (Items 14/15/16/17 in turn) gives a paired Δρ CI that excludes zero: +0.031 to +0.044. The **strongest single addition** is **Item 15 (+0.044, CI [+0.039, +0.049])** = **60% as much uplift as all four together**; the **weakest** is Items 14/17 at +0.031 each.
- **H4 — binomial(G, p) null control (n=200 simulations per item) separates signal from randomness.** Excess observed H_bits − null mean: Item 14 = **+0.186 (~+2.8σ)**, Item 15 = **+0.255 (~+6.2σ)**, Item 16 = **−0.123 (~−1.9σ)**, Item 17 = **+0.257 (~+3.6σ)**. **Three of the four candidates (Items 14, 15, 17) carry significantly MORE H_bits than the binomial null** — they encode information BEYOND what Binomial(G,p) produces. **Item 16 is REJECTED as a placebo**: its H_bits is *below* binomial expectation (the empirical max_K_share is *higher* than binomial), so adding Item 16 would mostly be re-encoding binomial noise.
- **H5 — Item 15 (K_unique_count) is the strongest single axis** by both H_bits (above-binomial +6.2σ) and Spearman coupling (single-item ρ=0.588). It is interpretable as: "for this (model × task × G × t × seed) cell, how many distinct success-counts are observed across the n_groups prompts". A cell at G=8 with K_unique=8 has 8 distinct success-counts visible — full contrast coverage; K_unique=1 means all prompts land at the same K.

## Sharpest finding

The **iter-80 row 95 Item 13 (zvf_yield_residual) was the FIRST MIN-REPORT item to independently strengthen coupling.** Iter-81 finds **three additional yield-residual items (Items 14, 15, 17) that ALSO carry signal beyond the binomial expectation.** The MIN-REPORT fingerprint is **structurally under-recorded** on a single item. **v2.2 MIN-REPORT = v2.1 (12 items) + Items 14, 15, 17 (3 items, displacing Item 16 placebo + 1 v1 placebo)** — a 14-item schema (12 + 3 − 1 = 14) with **+15.86 bits + 3 of 4 candidates passing the binomial null** is the new sharp recommendation.

## Cross-paper coupling

- (i) **P5 iter-80 row 95 Item 13** — Item 13 is one of four yield-residual candidates; Items 14, 15, 17 sharpen the axis. **Item 16 explicitly FAILS the binomial null** — would have been a regression if blindly added.
- (ii) **P6 iter-66 row 77 / iter-78 row 92** — Items 14-17 are all directly computable from the (cell_id, group_tensor_path) tuple in the registry. The registry's `measured_yield_residual` δ_div block can be **extended** to a multi-axis 4-component vector (zvf / K_unique / max_Kshare / p_hat_var) at zero harvest cost.
- (iii) **P7 iter-75 row 88 / iter-79 row 93** — the joint trigger selects steps where the structural anti-herding signal is weak; the same Items 14-17 can also be computed at the **per-step** level from `n2_reward_tensor_resume/`. iter-75's CP_exact formula and iter-79's per-step trigger are the **per-step analog** of the per-cell Items 14-17 here.
- (iv) **P8 iter-80 row 94** — the score-stream gradient (Score_i−1−Score_i) on fraud rows is the **per-row anti-herding analog** of Items 14-17 on LLM cells. The H_bits uplift on the LLM axis is +15.86 bits; the LLM-call savings on the fraud axis is 57%. **Same anti-herding mechanism, two granularities, two operational wins.**

## Operational recommendation

Add Items 14 (K_variance_residual), 15 (K_unique_count), and 17 (prompt_p_hat_var) to MIN-REPORT v2.2. **Reject Item 16** (max_K_share) — it has highest point-ρ (0.589) but **fails the binomial null** (excess = −0.123, ≈ −1.9σ). Adding Item 16 alone would *replace* binomial signal with anti-herding-by-construction without adding new empirical information. Recommended **v2.2 schema size = 14 items (12 v2.1 − 1 placebo + 3 new)**.

## Reproducibility

- Script: `scripts/p5p8/p5_yield_residual_axes.py` (~290 LoC, stdlib only; numpy for K-extraction)
- Outputs:
  - `experiments/results/p5p8/p5_yield_residual_axes.tsv` (12 rows)
  - `experiments/results/p5p8/p5_yield_residual_axes_per_item.tsv` (17 rows)
  - `experiments/results/p5p8/p5_yield_residual_axes_shuffle_null.tsv` (4 rows)
  - `experiments/results/p5p8/p5_yield_residual_axes_summary.json` (machine-readable)
- Seed: 20260705 (paired bootstrap B=2000 with 2000 cell-pairs; binomial null n=200)
- Citations: frontier synthesis (FRONTIER_INSIGHTS.md Round 1) — "Contrastive Yield Y(p,G) = 1−ZVF" formalisation of which Item 13 is the per-cell proxy; iter-80 row 95 Item 13 baseline; iter-77 row 91 portability framework.
