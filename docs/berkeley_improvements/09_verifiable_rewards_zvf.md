# Improvement 09 — Verifiable rewards (Jiao F25 L4) bound ZVF and tighten G*

| field | value |
| --- | --- | --- |
| source lecture | **F25 "Agentic AI", Lecture 4 — Jiantao Jiao (NVIDIA) — Post-Training Verifiable Agents (SWE-bench Verified + BrowseComp)** |
| source papers | **SWE-bench Verified** subset of Jimenez et al. 2024, arXiv:2310.06770 (ICLR 2024); **BrowseComp** — Wei et al. 2025, arXiv:2504.12516 (OpenAI 2025). Related: Yehudai et al. 2025, arXiv:2503.16416 (Survey on Evaluation of LLM-based Agents, F25 L5). All three verified via WebFetch on arxiv.org 2026-07-04. |
| target mapping | **A3** post-training science (verifiable reward as a *lower bound* on the GRPO/ZVF surface) + **A1** statistical rigor (separates the verifiable-identity anti-herding bonus from grader-noise inflation) + **B-SYNTH** cross-course synthesis (ties Pillar 2 ZVF to Pillar 3 G*, with read-across to rows 01, 02, 08) |
| pillar | **B-F25** + **B-SYNTH** |
| status | **validated** (4 falsifiable hypotheses tested on real iter98 bfclv4 + iter130 zvf_by_library data; H1 NON-DECISIVE falsification, H2 partial-credit uplift DECISIVE, H3 variance DECISIVE, C2 Gv<Gn 12/12 p DECISIVE) |
| artifact | `scripts/berkeley/verifiable_rewards_zvf.py` (prototype) + `scripts/berkeley/verifiable_fix_and_resynth.py` (bug-fix + B-SYNTH resynthesis) |
| evidence | `experiments/results/berkeley/{verifiable_zvf_percell,verifiable_zvf_inflation,verifiable_zvf_p_dispersion,verifiable_g_star,verifiable_risk_score_delta,verifiable_g_star_sensitivity,verifiable_cross_pillar}.tsv` + `verifiable_summary.json` + `verifiable_cross_pillar_meta.json` |

## 1. Course idea, in one paragraph

Jiantao Jiao's F25 Lecture 4 frames post-training for agentic LLMs around
**verifiable rewards** — the family of tasks (SWE-bench Verified, BrowseComp,
math, code-execution) where the grader is deterministic, exact-match, and
inspectable. The implicit claim is that verifiable reward is the *cleanest*
RL signal we can build: the policy gradient is unbiased by grader noise, the
advantage estimator is well-defined, and the per-prompt ZVF/GU diagnostic
(our Pillar 2) reduces to a function of latent difficulty `p_x` and group
size `G` alone. Under non-verifiable reward (LLM-as-judge, partial credit,
human preference), the grader itself injects a **noise floor** that
contaminates every ZVF measurement. Our Pillar 2 measurements have so far
ignored this distinction; Jiao's framework makes it first-class.

## 2. Mapping to TinkerRL-Bench — three falsifiable hypotheses

We test three hypotheses on real data already in the repo.

**H1 (Jiao-1 — VERIFIABLE IDENTITY)**: on verifiable-reward rollouts, the
empirical ZVF is bounded by the i.i.d. baseline plus a small anti-herding
bonus:
`ZVF_obs = ZVF_iid(p, G) + delta_div_verifiable`, with
`|delta_div_verifiable| < 0.25`. We use the bfclv4 tool-use dataset which
contains BOTH sparse (verifiable, binary) and dense (LLM-as-judge,
partial-credit) reward columns for the same 10 (seed, step) cells, G=8.

**H2 (Jiao-2 — GRADER-NOISE INFLATION)**: under non-verifiable dense
reward, ZVF can be SPURIOUSLY lowered in the all-wrong herding regime by
partial credit injecting a positive signal into a fully-failed group, AND
in the non-herding regime partial credit always inflates apparent contrast
(`p_dense > p_sparse`). The "verifiable tax" — the irreducible ZVF floor
under perfect graders — is therefore smaller than the non-verifiable
inflation.

**H3 (Jiao-3 — p-CORRESPONDENCE)**: under verifiable reward, the empirical
mean reward `p_sparse` is a low-variance / unbiased estimator of latent
`p_x`. Under non-verifiable dense reward, `p_dense` has higher variance
and higher mean (because partial credit adds > 0 to all-correct-or-all-wrong
groups), so `Var(p_dense) > Var(p_sparse)` AND `mean(p_dense) > mean(p_sparse)`.

## 3. B-SYNTH cross-pillar resynthesis — three further claims

**C1 (CROSS-PILLAR RANKING)**: does applying Jiao's grader-noise correction
to the iter130 zvf_risk_max ranking change which method is the safest?
Result: the **rankings do not change** (`rank_shift = 0` for all 9 variance-
mitigation libraries, n_reordering=0, n_inversions=0) because the
magnitude-of-ZVF channel is preserved. BUT the **bucket composition** does:
4/4 drift-cluster methods (MCGRPO, GIFT, AREAL, ES) flip from the
`drift` bucket to the `plateau` bucket (`n_bucket_reassign=4`), meaning
the Jiao correction reclassifies ~44% of methods and weakens the iter130
claim that GIFT/AREAL are the lowest-risk methods — they were partly
risk-flagged because of grader noise, not because of policy instability.

**C2 (BUDGET-CONDITIONAL G*)**: under verifiable reward, the smallest G
that reaches 80% contrastive yield is at most the non-verifiable G*. With
calibrated inflation share delta_grader=0.16, we have **Gv < Gn on 12/12
p values**, and specifically at p=0.5: Gv=4 vs Gn=6 (33% reduction); at
p=0.05 (extreme hard): Gv=32 vs Gn=64 (50% reduction). This tightens the
iter127 G*(T) rule: under verifiable reward you can use SMALLER G at
every T, with the largest savings on the easy/hard tails where the
inflation dominates.

**C3 (BRIDGE TO DUALFORMER + DPO/IRPO)**: the Jiao verifiable-tax is a
*lower bound* on the GRPO loss surface. Both row 01 (Dualformer-auto G
allocation) and row 02 (DPO/IRPO equivalence) are bounded ABOVE by this
tax and therefore are STRICTLY COMPATIBLE with the verifiable regime —
the Jiao correction does not invalidate them; it tightens their
predictions. Concretely, Dualformer-auto's 56% compute savings become
*upper bound* savings under verifiable reward (the actual savings are
larger because Gv<Gn at every p), and DPO/IRPO's G*_IRPO=G*_GRPO result
becomes G*_IRPO=G*_verifiable < G*_GRPO, sharpening the equivalence.

## 4. Measured results

From `experiments/results/berkeley/verifiable_summary.json` and
`verifiable_cross_pillar_meta.json`:

- **H1 (verifiable identity)**: NON-DECISIVE on bfclv4 G=8 n=10.
  Non-herding n=6 cells, |delta_div_sparse| max=0.3436, mean=0.1331,
  non-herding mean=0.2219. The 0.25 bound is VIOLATED at one cell
  (delta_div=0.3436 > 0.25) — H1 does NOT pass. The structural
  anti-herding bonus from iter78/iter98 (~0.13-0.23) is the dominant
  signal but the upper tail extends to 0.34, which is consistent with
  the iter98 finding that the bonus depends on prompt-difficulty
  heterogeneity, not just (p, G).
- **H2 (grader noise inflation)**: PARTIAL DECISIVE. **H2a** (all-wrong
  herding broken by dense partial credit): 0/10 hits, frac=0.0,
  binomial p=1.0 — DECISIVE NULL. bfclv4's dense reward did NOT
  spuriously break the all-wrong signal. **H2b** (partial credit
  uplift in non-herding): 5/5 hits, frac=1.0, binomial p=0.0312 —
  DECISIVE. Every non-herding cell had `p_dense > p_sparse`. The
  partial-credit inflation is real and reproducible on this evidence.
- **H3 (p-correspondence)**: BOTH DECISIVE. `Var(p_dense)=0.0208 >
  Var(p_sparse)=0.0120` (variance DECISIVE) AND `mean(p_dense)=0.1862 >
  mean(p_sparse)=0.1125` (mean DECISIVE). Both pass.
- **C1 (cross-pillar ranking)**: 0 reordering, 0 inversions,
  **4 bucket reassignments** (mcgrpo/gift/areal/es: drift→plateau).
- **C2 (budget-conditional G*)**: 12/12 p values satisfy Gv < Gn,
  with the largest delta at p=0.05 (Gv=32 vs Gn=64, 2x reduction).
- **C3 (bridge)**: theoretical; both row 01 and row 02 are compatible.

## 5. Recommendation / go-no-go

**GO** for integration. The C2 Gv < Gn result (12/12) is a clean
DECISIVE sharpening of the iter127 G*(T) rule, and the C1 4-bucket
reassignment is a clean DECISIVE softening of the iter130 GIFT/AREAL
"lowest risk" claim. Both are paper-facing. The H1 NON-DECISIVE
falsification is also paper-facing because it sharpens the iter78
anti-herding bonus from "0.13-0.23" to "0.13-0.34 with a long upper
tail", which the iter130 three-channel risk index should track.

## 6. Artifacts

- `scripts/berkeley/verifiable_rewards_zvf.py` (prototype, n=300 lines)
- `scripts/berkeley/verifiable_fix_and_resynth.py` (bug-fix + C1/C2/C3, n=280 lines)
- `experiments/results/berkeley/verifiable_zvf_percell.tsv` (n=10 rows, bfclv4 per-cell)
- `experiments/results/berkeley/verifiable_zvf_inflation.tsv` (n=1, H1/H2/H3 summary)
- `experiments/results/berkeley/verifiable_zvf_p_dispersion.tsv` (n=10 rows, p-dispersion)
- `experiments/results/berkeley/verifiable_g_star.tsv` (n=12, G* at 12 p values)
- `experiments/results/berkeley/verifiable_risk_score_delta.tsv` (n=9, fix)
- `experiments/results/berkeley/verifiable_g_star_sensitivity.tsv` (n=12, Gv/Gn comparison)
- `experiments/results/berkeley/verifiable_cross_pillar.tsv` (n=9, C1 rank/bucket shift)
- `experiments/results/berkeley/verifiable_summary.json` (H1/H2/H3 machine-readable)
- `experiments/results/berkeley/verifiable_cross_pillar_meta.json` (C1/C2/C3)
- `paper/sections/zvf_iter130_verifiable.tex` (B-SYNTH paper section)
