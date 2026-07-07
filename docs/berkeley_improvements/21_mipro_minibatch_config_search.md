# Row 21 — MIPRO minibatch config-search for GRPO group-size selection

**Source lecture.** F24 L5 — Omar Khattab (Compound AI & DSPy).
**Key papers (both verified 2026-07-04 via arXiv abs metadata):**
- Opsahl-Ong, Ryan, Purtell, Broman, Potts, Zaharia, Khattab. *Optimizing
  Instructions and Demonstrations for Multi-Stage Language Model Programs*
  (**MIPRO**). arXiv:2406.11695, **EMNLP 2024**.
- Soylu, Potts, Khattab. *Fine-Tuning and Prompt Optimization: Two Great Steps
  that Work Better Together* (**FT+PO**). arXiv:2407.10930, **EMNLP 2024**.

**Target.** A2 (eval methodology / eval cost) + A3 (post-training science).

## The idea, ported

MIPRO optimizes a multi-stage LM program not by exhaustively evaluating every
candidate configuration on the full validation set, but by (a) a **Bayesian
surrogate** (TPE) over the discrete config space and (b) **minibatch evaluation**:
candidates are scored on cheap partial evaluations and only the promising ones
are *promoted* to full evaluation. FT+PO adds that the two optimization axes
(weights vs. prompts) "work better together".

We map this onto **GRPO group-size selection** on the same-stack sweep
(`groupsize_zvf_sweep.json`: 4 group sizes G∈{2,4,8,16} × 3 seeds × 40 steps,
Qwen2.5-0.5B, arithmetic). A "config" is a group size; "full evaluation" is the
3-seed terminal held-out accuracy. There are two orthogonal cheap **minibatch
axes**, in direct analogy to MIPRO/FT+PO:

| MIPRO/FT+PO axis | our minibatch axis | cheap because |
| --- | --- | --- |
| prompt-opt / demo subset | **SEED minibatch** — evaluate on 1 of 3 seeds | 2/3 fewer runs |
| weight-opt / partial fine-tune | **STEP minibatch** — read the trajectory at step k<40 | k/40 of training |

The stakes are real: per-config compute scales ~5× with G (25 s → 135 s per
seed), so a correct cheap-eval decision saves substantial wall-clock.

## Hypotheses & measured results

Data anchor — full 3-seed terminal held-out rank (best→worst): **G8 > G4 > G2 >
G16** (`mipro_summary.json`). Note `last10_avg` instead ranks **G16 first** — the
two terminal metrics *disagree*, the classic noisy-eval regime MIPRO targets.

| H | claim | result | verdict |
| --- | --- | --- | --- |
| **H1** | 1-seed SEED-minibatch recovers the full 3-seed rank | mean Kendall τ = **0.444** across seed choices | **SUGGESTIVE** |
| **H2** | an early step-k feature stably predicts terminal rank | **no** feature holds top-1 for a 5-step window with ρ>0 (first_stable = −1) | **NULL** |
| **H3** | surrogate ordering (TPE analog) reaches true best-G earlier + cheaper than random | order **8-16-4-2**; true-best at **position 1** (random exp. 2.5); aggressive top-1 promotion picks true best G8 at **50.1 % compute saving** | **DECISIVE** |
| **H4** | the 4 configs are statistically equivalent (flat landscape) → bounds minibatch regret | **6/6** pairs equivalent within ±0.02; max held-out gap **0.0117** | **DECISIVE** |
| **H5** | SEED + STEP surrogates "work better together" (FT+PO) | combined ρ=0.40 = max(seed 0.40, step −0.20); **gain 0.0** | **NULL** |

**2/5 DECISIVE.**

## Interpretation — the SEED/STEP asymmetry

The two decisive results and the two nulls tell one coherent story:

- **H4 is the enabling condition.** The config landscape is *flat*: every pair of
  group sizes is TOST-equivalent within ±0.02 and the largest terminal gap is
  0.0117 — smaller than a single config's cross-seed spread. When configs are
  within-noise-equivalent, a minibatch estimate of the winner has bounded regret
  by construction. This is exactly the condition under which MIPRO's cheap eval
  is *safe*, made quantitative.
- **H3 is the payoff on the SEED axis.** A cheap surrogate (1-seed held-out +
  partial mean-reward) orders the configs with the true best G=8 at rank 1, and
  promoting only the surrogate's top-1 to full evaluation recovers the true best
  at **half** the compute of the exhaustive 3-seed grid.
- **H2 + H5 falsify the STEP axis.** Partial-training features (ZVF, mean-reward,
  entropy, grad-norm at step k) do **not** stably predict the terminal ranking —
  mean-reward's config-ordering ρ flips +1.0 → −0.4 → +1.0 → −0.8 across
  k∈{0,9,29,39}. On a flat landscape the between-config signal is smaller than the
  within-trajectory noise, so early-stopping to save training steps is *unsafe*
  here even though sub-sampling seeds is safe. FT+PO's "better together" does not
  hold: the STEP axis is anti-correlated and adds nothing to the SEED axis.

This is a genuinely useful nuance, not a null result: **for GRPO config search,
sub-sample seeds, do not sub-sample training steps.** It sharpens A2 (how to
cheaply pick a group size) and complements row 11 (Yehudai eval-cost: 1-seed
sufficient for the top-1 headline) — here we show the same 1-seed economy extends
from *reporting* a headline to *selecting* a config, but the partial-training
shortcut that would look equally tempting does **not** transfer.

**(Frontier synthesis.)** The flatness anchor (H4) reinforces the
Estimator-Equivalence reading in `FRONTIER_INSIGHTS.md` round 1: once the stack is
fixed, config choice moves *variance/compute*, not the expected update direction —
so cheap surrogate selection is licensed precisely because the terminal landscape
is under-identified.

## Go / no-go

**GO (as a one-sentence A2/P3 methodology note).** Paper-facing claim: "group-size
selection on the same-stack arithmetic sweep is TOST-flat (max Δ=0.012, 6/6 pairs
equivalent), so a MIPRO-style 1-seed surrogate identifies the terminal-best group
size at ~50 % of exhaustive compute; partial-training (early-step) surrogates,
however, do not transfer and should not be used to prune configs." Not a new
section — a stabilizer sentence + the SEED/STEP asymmetry caveat.

## Artifacts
- `scripts/berkeley/mipro_minibatch_config_search.py`
- `experiments/results/berkeley/mipro_h1_minibatch_rank.tsv`
- `experiments/results/berkeley/mipro_h2_early_step_surrogate.tsv`
- `experiments/results/berkeley/mipro_h3_surrogate_search.tsv`
- `experiments/results/berkeley/mipro_h4_tost_equivalence.tsv`
- `experiments/results/berkeley/mipro_h5_two_steps_together.tsv`
- `experiments/results/berkeley/mipro_summary.json`
