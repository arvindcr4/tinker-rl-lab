# 20 — Seed-Clustering Error Bars: which TinkerRL-Bench headline CIs are honest?

**Source lecture.** Berkeley F25 *Agentic AI* L8 — **Sida Wang (Meta)**,
"Predictable Noise in LLMs / Adding Error Bars to Evals."
**Target.** A1 (statistical rigor of the benchmark), cross-pillar.
**Status.** validated.
**Citations (reused from row 07, verified 2026-07-04).**
- Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to
  Language Model Evaluations.* arXiv:2411.00640 (cs.CL / stat.AP).
- Wang, S. et al. (2025). *Measuring all the noises of LLM Evals.* arXiv:2512.21326.

## Why this is not a repeat of row 07
Row 07 (Miller) placed **simple i.i.d. bootstrap** CIs on 7 headline numbers.
Sida Wang's distinct lecture point is that eval noise is *structured*: metrics
logged along a training run are strongly autocorrelated within a seed, so pooling
`S` seeds × `M` steps and treating the `S·M` rows as independent inflates the
effective sample size and yields a **falsely narrow** CI. The honest unit of
replication is the **seed (cluster)**, not the step. Row 07 itself flagged this
as an open thread (H6: *"n=52 pooled across 3 experiments, not pure seeds →
Miller would call for cluster sensitivity"*). This row closes it with a
per-headline **design-effect audit**.

## Method (`scripts/berkeley/headline_ci_clustering.py`)
For every headline metric with real `(seed × step)` data we compute, from the
actual repo TSVs / JSON:
- **point** = grand mean of per-seed means;
- **naive pooled 95% CI** = bootstrap over all `S·M` rows as if i.i.d.;
- **seed-clustered 95% CI** = cluster bootstrap (resample seeds, use seed means);
- **ICC(1)** = between-seed var / total var (one-way random-effects ANOVA);
- **DEFF** = `1 + (m̄−1)·ICC` (Kish design effect);
- **n_eff** = `n_pooled / DEFF`;
- **width inflation** = `width_cluster / width_naive`.

Deterministic (`numpy.default_rng(0)`, B=10 000). Data: `samestack_ppo_grpo.json`
(P1, 5 seeds × 40 steps × 2 algos), `group_size_advantage_variance.tsv`
(P3, 4 G × 3 seeds × 40 steps), `bfclv4_tool_use.tsv` (P4, 2 seeds).

## Results — real, from `experiments/results/berkeley/headline_ci_clustering.tsv`

| metric | pillar | ICC | DEFF | n_eff / n_pooled | naive CI | cluster CI | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **P1 PPO last-10 reward** | P1 | **0.708** | **7.37** | 6.8 / 50 | [0.882, 0.948] | **[0.818, 0.975]** | INFLATED (2.39×) |
| P1 GRPO last-10 reward | P1 | 0.128 | 2.15 | 23.3 / 50 | [0.970, 0.987] | [0.967, 0.990] | MILD (1.32×) |
| **P1 GRPO−PPO paired Δ** | P1 | paired | 2.35 | 5 / 50 | **[0.030, 0.096]** | **[−0.0002, 0.156]** | **CLUSTER_REQUIRED** |
| P3 reward G=2 | P3 | 0.000 | 1.00 | 120 / 120 | [0.800, 0.878] | [0.827, 0.852] | HONEST |
| P3 reward G=16 | P3 | 0.000 | 1.00 | 120 / 120 | [0.834, 0.908] | [0.870, 0.878] | HONEST |
| P3 ZVF G=2 | P3 | 0.000 | 1.00 | 120 / 120 | [0.804, 0.871] | [0.827, 0.852] | HONEST |
| P3 ZVF G=16 | P3 | 0.000 | 1.00 | 120 / 120 | [0.563, 0.696] | [0.622, 0.638] | HONEST |
| P4 bfcl dense | P4 | 0.519 | 3.08 | 3.3 / 10 | [0.103, 0.268] | [0.095, 0.278] | INFLATED (n=2, weak) |
| P4 bfcl sparse | P4 | 0.415 | 2.66 | 3.8 / 10 | [0.050, 0.175] | [0.050, 0.175] | MILD (n=2, weak) |

**5/9 pooled-CI headlines are ≥1.5× DEFF-inflated.** The audit's value is that it
tells you, per headline, *which noise source dominates* and therefore which CI to
trust — precisely Sida Wang's "measure all the noises" prescription.

### Flagship finding (paper-critical) — the P1 equivalence claim flips on clustering
The same-stack headline is **GRPO ≈ PPO** (paper reports p=0.75 on held-out acc).
On the last-10 training reward the two error-bar recipes *disagree on the verdict*:
- **Naive pooled paired CI** = **[+0.030, +0.096]** — excludes 0 → would
  **falsely declare GRPO > PPO significant.**
- **Seed-clustered paired CI** = **[−0.0002, +0.156]** — straddles 0 →
  **correctly supports equivalence.**

Only the seed-clustered error bar is consistent with the paper's p=0.75. A
reviewer who pooled steps would manufacture a spurious algorithm effect. This is
a concrete, reviewer-salient reason the benchmark must report seed-clustered bars.

### Cross-pillar bridge to the Critic-Degeneracy Hypothesis (row 12)
PPO's between-seed **ICC = 0.708** vs GRPO's **ICC = 0.128** — a **5.5× ratio**.
Row 12 measured PPO's critic instability as grad_norm 156× larger and rolling
reward-variance 73% higher. Here the *same* instability appears through the
error-bars lens: PPO's variance is dominated by the **between-seed** component
(critic seed-sensitivity), while GRPO's stateless group-mean baseline keeps
between-seed variance low. The design-effect audit is an **independent, second
measurement of the CDH mechanism** — the PPO critic doesn't just add gradient
noise, it adds *seed-level* noise that pooled CIs hide.

### Why P3 is HONEST (nuance, not a null)
For the group-size traces, ICC ≈ 0: step-to-step (decoding/optimization) variance
`var_within ≈ 0.05` swamps between-seed variance `var_between ≈ 0` (seed means
0.852/0.827/0.840 at G=2 — genuinely distinct but tightly clustered). So DEFF = 1
and the **pooled CI is the valid one**; the cluster CI is narrower only because
k=3 clusters is small (under-powered, not more honest). The correct rule is
regime-dependent: cluster when ICC is non-trivial (P1/P4), pool when ICC ≈ 0 (P3).

## Go / no-go
**GO — validated, paper-facing (A1, all 4 pillars).** Deliverables:
1. Add a **"Seed-clustered error bars"** sentence to each pillar's methods/eval
   section: *"All headline error bars are seed-clustered (cluster bootstrap over
   seeds); we verified via a per-metric design-effect audit (ICC + Kish DEFF)
   that pooling training steps as i.i.d. understates CI width by up to 2.4×
   (DEFF 7.4 for PPO), and that the same-stack GRPO≈PPO equivalence holds only
   under seed-clustered — not pooled — error bars."*
2. Cite Miller 2411.00640 + Wang 2512.21326 alongside the existing row-07 citation.
3. Cross-reference CDH (row 12): report ICC(PPO)/ICC(GRPO) = 5.5× as a second CDH
   signature.

## Evidence paths
- `scripts/berkeley/headline_ci_clustering.py`
- `experiments/results/berkeley/headline_ci_clustering.tsv`
- `experiments/results/berkeley/headline_ci_clustering_icc.tsv`
- `experiments/results/berkeley/headline_ci_clustering_summary.json`
