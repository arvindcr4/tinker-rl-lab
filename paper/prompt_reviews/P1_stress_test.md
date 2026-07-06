# P1 Hypothesis Stress Test (hypothesis-stress-test.md contract)

Role executed: skeptical reviewer testing causal logic.
Target: Pillar-1 scaling paper (`paper/paper_P1_scaling.tex` + `paper/sections/scaling_laws.tex`,
`p1_results_intro.tex`, `p1_conclusion.tex`, `frontier_synthesis_scaling.tex`).
All numbers below were re-checked against `experiments/results/` (files cited inline).

---

## Input

**Hypothesis.** GRPO reward gain does NOT scale with parameter count across
0.6B–671B; the dominant axis is capability class (instruct-vs-base, collapse
risk), and the saturation model R(t) = R_max(1 − e^{−λt}) is structurally
misspecified (monotonicity rejected 5/5 anchors).

**Proposed mechanism (derived from the paper's own results/discussion, stated explicitly).**
The paper's Discussion (`p1_conclusion.tex`, Sec. "Discussion and Limitations")
supplies the causal chain:

1. GRPO's learning signal is within-group reward variance — "the effective
   learning signal in GRPO depends on within-group reward variance, which is
   itself coupled to model accuracy and group size."
2. Instruct anchors start at or above their eventual ceiling on GSM8K
   (R(1) = 1.0 for Qwen3.5-4B and Llama-3.1-8B-Instruct, 0.875 for
   DeepSeek-V3.1; `scaling_laws.tex` par. "Where the model fits and where it
   does not"), so groups are reward-homogeneous, advantages vanish, and the
   trace is flat from step 1.
3. Base-class anchors either produce near-zero usable signal (Nemotron-120B:
   zero-reward fraction 0.55, collapse) or sit low and flat (Qwen3-8B,
   R̄ = 0.285).
4. Therefore the measured reward level is set by pre-RL capability class, not
   by N: the cross-scale slope on log10 N is null (Table `tab:scaling-cross`;
   extended 12-anchor null, `tab:scaling-powerlaw`), R_max is bimodal by class
   (iter125: capable {0.817, 0.869, 0.844} vs incapable {0.285, 0.182}, gap
   0.531), and the strict-R(0)=0 saturation model is degenerate (λ pinned at
   the bound on 4/5 traces; constant model wins AIC 5/5; holdout improvement
   ≤ 0.0016 RMSE) and violated (monotonicity rejected 5/5, iter125).

**Known counterexamples (declared).**
- arXiv 2507.18014 (Nimmaturi et al., `nimmaturi2025predictive`) claims
  predictive GRPO scaling laws with a three-phase template.
- The paper's own iter117/121/125 fits: iter117 finds the t80-vs-N regression
  degenerate (4/5 anchors at the λ bound; the λ-free regression is a single
  point — `scaling_law_iter117_t80_scaling.tsv`); iter121 finds n = 5 has
  detection power 0.22–0.40 even for true slopes of 0.01–0.2 per decade
  (`scaling_law_iter121_power_curve.tsv`); iter125's bimodality dip test is
  p = 0.056 (`scaling_law_iter125_bimodality.tsv`).
- Only 5 frontier anchors, single-seed on most (Discussion: "frontier-scale
  anchors are single-seed API runs and should be read as case studies").
- Scope note: the hypothesis says 0.6B–671B, but the P1 anchor set actually
  spans 4B–685B; the 0.6B end of the benchmark never enters the scaling
  regression.

---

## 1) Weakest link

**The inference from "R_max is bimodal and the split aligns with
instruct-vs-base labels" to "capability class is the DOMINANT axis of GRPO
outcome."** This is the only affirmative causal claim in the hypothesis (the
other two clauses are nulls), it is the load-bearing clause — and it rests on a
2-vs-3 split of five single-seed anchors in which "capability class" is
perfectly confounded with harness compatibility (chat template + reward
parser), model family, and one collapse run. The paper's own paired
base-vs-instruct data contradicts the direction of the class effect.

## 2) Why this link is fragile

**(a) The statistics cannot support "dominant."** The Hartigan dip test on the
5 R_max values gives p = 0.056 — not significant at 0.05
(`scaling_law_iter125_bimodality.tsv`, confirmed in `scaling_law_iter125_meta.json`).
A *perfect* 2-vs-3 label alignment on n = 5 has permutation probability
1/C(5,2) = 0.10 under the null, so the class alignment can never reach p < 0.05
at this n. Worse, the paper's own iter129 likelihood analysis directly rejects
the clause as stated: the Bayes factor for (params + capability) vs
(params alone) is log BF = −9.53 — "very strong" *against* capability being an
identifiable predictor of R_max beyond parameter count
(`scaling_laws.tex` par. `par:scaling-iter129`, Table `tab:scaling-iter129-fit`).
The hypothesis asserts as dominant an axis the paper's own model comparison
says adds nothing at n = 5.

**(b) The paper's own paired data flips the sign of the class effect.**
`experiments/results/base_instruct_paired.tsv`:
- Qwen3-8B **Base**: train reward 0.8250 → 0.8562 — squarely in the "capable"
  band.
- Qwen3-8B **Instruct**: train reward 0.2925 → 0.3105 — in the "incapable"
  band — while its *held-out accuracy* is 0.820 → 0.833.
- Llama-3.1-8B: the reverse (Base 0.125, Instruct 0.950).

So base-vs-instruct does not have a stable sign even inside the paper's own
data, and a 0.53-point train-reward vs held-out-accuracy gap on the *same*
Qwen3-8B-Instruct checkpoint is the signature of a chat-template/answer-parser
mismatch, not of capability. Note also that the P1 anchor labeled
"Qwen3-8B / dense-base" has R̄ = 0.285 — numerically matching the paired
table's **Instruct** run (0.2925), not its Base run (0.825). Either the anchor
label or the class attribution is suspect; either way the "incapable" cluster
may be a harness artifact.

**(c) The "collapse risk" half of the clause rides on one single-seed API
run.** Nemotron-120B is the only collapse in the set; the paper itself
concedes the MoE-vs-dense gap loses significance once Nemotron is removed
(`tab:scaling-moe-vs-dense` caption: gap shrinks to +0.20, p > 0.05,
"descriptive rather than causal"). One unreplicated trace cannot anchor a
class-level "collapse risk" attribute.

**(d) If (a)–(c) fall, the two null clauses degrade with them.** The
no-scaling clause becomes *untested* rather than true: the three instruct
anchors start at ceiling (R(1) = 1.0, 1.0, 0.875), so measured *gain* is
pinned at ≈ 0 by construction, and iter121's synthetic-recovery analysis shows
the n = 5 design recovers a true slope of +0.1/decade only 26% of the time and
+0.2/decade only 40% (`scaling_law_iter121_power_curve.tsv`) — over the 2.2
decades from 4B to 685B, an undetectable +0.1/decade is +0.22 reward, larger
than most gaps in the table. "Does NOT scale" is an affirmative claim built on
a test with ≤ 26% power against realistic alternatives. And the
misspecification clause's headline evidence — "monotonicity rejected 5/5" —
tests a strawman: R(t) = R_max(1 − e^{−λt}) is a *mean* function, but the
iter125 test counts pairwise violations in raw per-step rewards quantized at
1/8 (≈ 8 rollouts/step, per-step σ ≈ 0.19 by iter121) against an arbitrary
"5% iid noise floor" (`par:scaling-iter125`). A *correctly specified*
saturating mean plus binomial sampling noise at σ ≈ 0.19 produces ~40–50%
pairwise violations on a near-flat trace — almost exactly the observed
0.29–0.46. The sound misspecification evidence is the λ-bound degeneracy,
AIC, and holdout nulls ("wrong model for ceiling-start traces"), which is a
much weaker claim than "structurally misspecified for GRPO"; it also defuses
the apparent conflict with arXiv 2507.18014, whose three phases are observed
on runs that *start below ceiling* — a regime these anchors never sample.

## 3) Disconfirming check

Lowest-cost first — zero new training compute, existing artifacts only
(~half a day of scripting):

**Harness-vs-capability audit of the "incapable" cluster.**
1. **Label audit:** map each of the 5 anchor traces
   (`scaling_law_iter117_meta.json` trace_file entries, e.g.
   `scale_gsm8k_qwen3-8b.json`) back to the run configs and exact model IDs;
   confirm whether the "Qwen3-8B / dense-base" anchor (R̄ = 0.285) is actually
   the Base checkpoint or the Instruct checkpoint that scores 0.2925 in
   `base_instruct_paired.tsv`.
2. **Transcript re-score:** sample 50 zero-reward rollouts each from the
   Qwen3-8B and Nemotron-120B anchor logs; classify each as
   (i) correct-but-unparsed (correct final answer present, reward parser or
   chat template missed it), (ii) genuinely wrong, (iii) degenerate output.
3. **Lenient re-score:** recompute R̄ and R_max for all 5 anchors under a
   lenient answer extractor (last number in the completion), then recompute
   the bimodality gap and dip statistic and the cross-scale slope.

## 4) Result pattern that would force revision

- **Revision trigger (numeric):** if ≥ 30% of sampled zero-reward rollouts
  from either "incapable" anchor are correct-but-unparsed, OR lenient
  re-scoring raises either incapable anchor's R̄ by ≥ 0.15 (collapsing the
  0.531 bimodality gap below 0.25 or moving any anchor across the cluster
  boundary), then the dominant axis must be re-attributed from "capability
  class (instruct-vs-base)" to "harness compatibility (template/parser)," and
  the bimodality clause of the hypothesis is withdrawn.
- **Hard-kill trigger:** if the label audit shows the R̄ = 0.285 anchor is the
  Instruct checkpoint (matching `base_instruct_paired.tsv` row 2), the
  instruct-vs-base clustering is void as stated and must be re-derived from
  verified labels before any class claim is made.
- **Survival pattern:** < 10% correct-but-unparsed, all lenient re-scored
  means within ±0.05 of the originals, and all 5 anchor labels verified — the
  capability-class reading survives this check; the next-cheapest decisive
  test is then a matched pair (same family, same N, base vs instruct, ≥ 3
  seeds, prompts stratified so step-1 solve rate ∈ [0.1, 0.9]) to unpin the
  ceiling-locked gain measurement, with pre-registered trigger: a
  gain-on-log10N slope ≥ +0.05/decade with bootstrap 95% CI excluding 0
  forces revision of the "does not scale" clause.

---

## Secondary fragilities (noted, not the weakest link)

- Scope mismatch: hypothesis says "0.6B–671B" but the scaling regression only
  ever sees 4B–685B; the 0.6B–1.7B paired cells are `source-missing` in
  `base_instruct_paired.tsv`.
- All misspecification evidence is conditional on T ≤ 30 steps and GSM8K; the
  Discussion itself bars extrapolation ("extrapolation to longer-horizon or
  non-math tasks is unwarranted").
- The "monotonicity rejected 5/5" framing should be restated as "the
  noiseless saturation mean is degenerate on ceiling-start traces (λ at bound
  4/5, AIC constant-wins 5/5, holdout Δ ≤ 0.0016)" — same conclusion, but no
  longer vulnerable to the noise-floor objection in 2(d).

## Handoff (→ Minimal Decisive Experiment prompt)

- `{{research_question}}` = the Disconfirming check (Section 3, verbatim).
- `{{decision_needed}}` = the Result pattern (Section 4, verbatim).
