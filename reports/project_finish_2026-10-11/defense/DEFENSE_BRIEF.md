# Phase 2 defense brief

> **Numbers marked v2 are provisional (16 October 2026).** Three PPO seeds and the exposure-matched `ppo_8x8` arm are still running. Single source: thesis Table 6.2 and `validation/samestack_v2/v2_analysis.json`. Update the v2 line below, slide 5, `DEFENSE_PREP.md` and `REVIEWER_GUIDE.md` together.
>
> **v2:** GRPO G8 − PPO +0.8 pp [−0.3, +1.9], seven paired seeds, equivalent within ±2 pp (TOST p = 0.016); G8 − G2 +0.8 pp [+0.02, +1.6], ten seeds, Holm p = 0.14; G2 − PPO +0.04 pp; base 0.654.

Twelve minutes of talk, ninety seconds of demo, then questions. Use the included corrected review deck. It now shows the implemented registry comparison and ownership boundary, with the prompt budget and inferential qualifications aligned to the thesis. Its measured scores are unchanged.

Do not present `platform_hybrid/sem 4 work/submissions/mtech-final-review/Arvind_MTech_Thesis_Review.pptx`.

## The talk

**0:00 Title.** "This is a measurement thesis. I recorded the controlled factors, disclosed the remaining confounds, and reported what the evidence can carry."

**0:40 The rule.** Two instruments. P1–P7 study RL diagnostics, comparisons and reporting. P8 is a supervised synthetic-fraud side study. E1–E14 asks what one 40-step Tinker checkpoint scores. Only Tinker and TRL finished runs, on different base checkpoints. This is not a completed seven-library benchmark.

**2:00 The phenomenon.** Eight correct samples: mean 1, standard deviation 0, advantages 0. All-wrong groups look the same, so ZVF cannot tell mastery from failure. Phase-1 main configuration sits at 0.72–0.77. Selected measured cohort means range from 0.16 to 0.84. The executable Colab audit finds that the old correlation used 17 finite cells, nine synthetic, although the report printed 23. Its outcome definitions mix training reward and other metrics, so it cannot validate held-out prediction. Three stored GSM8K tensors (Qwen3-8B, sampling only) reproduce 0.130, 0.190 and 0.155, pooled 0.1583. The reward-adjusted residual also uses this mixed cohort and mixed source outcomes including late training reward. It adds no independent held-out validation.

**3:30 Four answers.**

- ZVF is recomputable; predictive validity was not established.
- Algorithm labels: no detectable GRPO advantage over PPO once the token cap and critic are fixed.
- `stackdiff` flags both stack exhibits from records alone: R5 on checkpoint, reward and template. It is a reporting check; 342 of 378 pairs are underspecified.
- Governance produced receipts and a terminal state for every lane, not a training delta.

**5:00 The result that changed.** Unsaturated Qwen2.5-1.5B, GSM8K-CoT. The first rerun (v1) showed GRPO at group size 8 ahead of PPO by +5.0 percentage points. It had two flaws: a 200-token cap that bound (the base model scored 0.200) and a cold, untuned critic, so the PPO arm ended below its start (0.195). Holm-adjusted sign-flip p was 0.19. The corrected v2 rerun uses a 512-token cap, a warm-started critic, the full 1,319 test items and an independent re-grade that agrees on every item. Result: GRPO G8 minus PPO is +0.8 points, interval [−0.3, +1.9], equivalent within ±2 points. All arms gain about 4 points from 0.654. Say it plainly: once the cap and critic were fixed there was no detectable GRPO advantage. Three PPO seeds and the exposure-matched PPO arm are still running. On saturated 0.5B addition both methods are at about 0.99, paired difference −0.002, p = 0.374.

**7:00 Implementation.** Show the actual manifest comparison. Both records say DAPO, but `stackdiff` reports R5 on checkpoint, reward, template, dataset and evaluation metric, at coverage 0.87. On the 17× Tinker-versus-TRL pair it reports R5 on checkpoint, reward and template, at coverage 0.58, which is exactly the confound that pair hid. Levels come only from known differing fields; missing fields show up as coverage. Across 378 pairs: R0 39, R2 1, R3 10, R4 26, R5 302, with 342 underspecified because 16 entries are hand-entered. This is a reporting check, not a predicted outcome difference. The command and expected output are in `REGISTRY_EXAMPLE.md`.

**8:15 Campaign.** Read four rows and then the prohibition. VerilogEval 129/312. 150 of 183 failures are extraction failures, plausibly related to the 1,024-token cap with thinking on; no per-sample finish reason was recorded, so the cause is unproven. Omni-MATH 2,271/4,428 = 51.29% with two unjudged rows counted as failures; 51.31% is the judged-row figure. Capped answers are right 34.5% of the time, uncapped 85.9%. SWE-bench Pro 2/731, and 593 of 713 patches were corrupt before any test ran. These are harness scores, not capability scores. WebArena 90/812 is a lower bound because 108 judge tasks were ungraded. AgentDojo 88/97 = 0.907 is benign utility, Wilson interval [0.833, 0.950]. Do not average the rows. The small paired base-versus-trained table is inconclusive in every lane.

**9:30 C1.** After eight parser-zero answers, at least one of 24 fresh responses matches the reference in 11/64 questions. The cohort is the deterministic first 64 eligible questions; Wilson 9.88–28.21% is descriptive. Three numerical witnesses sit beside that score and are not folded into it. One case matches an incorrect reference after eight correct opening answers. This is not a training improvement.

**10:15 Contribution.** The diagnostic, the eight-item minimum report, and the ledger with named gates. Novelty, if asked: a search through July 2026 found no equivalent GRPO run datasheet with rollout provenance. That is bounded, not a proof of absence.

**11:00 Limitations and next study.** Small seed counts and missing execution records limit inference. Some client custom-loss loops are inspectable, while historical managed internals remain incomplete. The v2 rerun already fixed the critic, cap and evaluation items; the remaining step is the exposure-matched PPO arm. For the registry, the next study is a matched Base/Instruct × harness experiment that tests whether `stackdiff` predicts which gaps come from the harness.

**11:40 Demo.** Then stop.

## Demo

From the repository root, before entering:

```bash
./submission/demo/demo.sh --self-test
./submission/demo/demo.sh
```

Open `submission/demo/output/demo_report.html` directly. If they ask for a fresh process, `./submission/demo/demo.sh --serve`. Narration is in `submission/demo/DEFENSE_RUNBOOK.md`. The 68.75% on that page is the mean of one recorded artifact of 80 binary rewards. If the SHA check fails, show the failure.

Offline fallback: open the HTML generated before the talk. The default demo needs no network; if its integrity check fails, show that failure instead of bypassing it.

Rechecked on 11 October 2026: the demo self-test passed (7 tests). In the full repository, `tools/check_thesis_evidence.py` passed all 21 selected arithmetic checks. That checker and all its dependencies are not bundled here; it does not rerun graders or authenticate provenance.

## Answers to have cold

**Did GRPO beat PPO?** No. Ceiling null on 0.5B addition. On 1.5B GSM8K-CoT the v1 gap of +5.0 points came from a binding 200-token cap and a cold critic; the corrected v2 rerun gives +0.8 points [−0.3, +1.9], equivalent within ±2 points (TOST p = 0.016), provisional at seven PPO seeds. v2 could detect a 1.5-point difference at n = 7 and rejects a 5-point one (p ≈ 4 × 10⁻⁵). Holm family p = 0.25 is audit H7, the iter-136 efficiency contrast, not this comparison.

**Is the U-shape in ZVF a finding?** Mostly not. ZVF is close to the average of p^G + (1 − p)^G over prompts of varying difficulty. On the stored Qwen3-8B tensors a beta-binomial with intraclass correlation 0.16 predicts 0.170 against 0.158 observed. The audit has nine model labels, not six. The 92.3% all-correct escalation share is what the reward level predicts (91.5%), and no controller ran in that trace. Advantage Collapse Rate (arXiv 2605.21125) and Gradient Starvation (arXiv 2605.07689) study the same quantity.

**Is the 17× Tinker-versus-TRL swing a backend effect?** No. On the same Tinker harness, switching Base to Instruct alone accounts for 64% of the gap (0.856 to 0.338). Reward function, template, decoding, batch, data, loss and optimiser also differ, and TRL logged reward 0.0 on 25 of 30 steps.

**Do the four labels tie within 0.034?** The band is not equivalence. Three seeds per arm give a seed SD of 0.105; TOST fails at every margin up to ±0.15. Paired by seed, the seed explains 96% of the variance.

**What did GRPO do to Qwen3-8B on held-out GSM8K?** The Instruct checkpoint scores 82.0% before RL and 83.3% after. The +1.3 points are not significant, p = 0.26. That 82.0% is the same Instruct checkpoint, not a separate base model.

**Are the methods equivalent?** No. A null at accuracy 0.99 cannot separate no effect from no headroom. The group-size TOST at ±0.02 does not pass for G = 16 versus G = 2 (p = 0.056).

**Does ZVF predict accuracy?** Predictive validity was not established. On the eight non-simulated cells the old correlation is ρ = −0.37 (p = 0.36), the opposite sign. The historical analysis mixes synthetic cells and heterogeneous targets, including late training reward, so it cannot validate held-out prediction. ZVF describes unavailable group-relative signal, including all-correct and all-wrong groups; it does not establish a universal absence of association with outcome.

**Why is standard deviation zero for a perfect group?** Relative advantages need differences inside the group. Equal rewards supply none.

**What did the fourteen suites show about training?** Nothing causal. No per-suite base model at the campaign's own scope. The later small paired rerun is inconclusive in every lane.

**Why is WebArena not 11.1% exactly?** 90 successes out of 812 attempts. 108 tasks were never graded. The rate if those had passed would be 198/812 = 0.244. The reported figure is the lower bound.

**Which Omni-MATH percentage?** 51.29% over all 4,428 dispositions. 51.31% over 4,426 judged rows. The recovered terminal note attaches the second decimal to the first denominator. The thesis keeps the arithmetic and leaves the note unchanged.

**Is 41% on VerilogEval a capability claim?** No. It is one actor under one generation budget, full coverage, pass@1: 129/312 = 41.35%, design-cluster bootstrap interval [34.3%, 48.7%] over 156 paired designs. 150 of 183 failures are extraction failures, plausibly related to the 1,024-token cap with thinking on, without establishing causality. Mean response was 1,002 of 1,024 tokens; per-sample finish reasons were not recorded, so that is an inference. On 50 shared items the same adapter scores 33/50 with thinking off at 4,096 tokens, against 23/50 in this run. Engine and temperature also changed, so that is direction, not a corrected score. Some extracted code may come from unfinished reasoning drafts. The retained full run used the original managed sampler, which this package cannot replay.

**Is 2/731 on SWE-bench Pro a problem-solving score?** Mostly not. 593 of 713 patches fail `git apply` as corrupt: the hunk-header line counts do not match the bodies. The official script then ran the tests on the unpatched repository. With `--recount`, 711 parse. Both passes are among the 120 that parsed. 2/731 is what was measured; it mostly measures diff formatting. The runner now repairs headers. The stored score predates that.

**Does the algorithm explain at most 6% of ZVF variance?** No longer claimed. The old interval resampled pooled steps, a null band. Resampled within method: η² = 0.045, interval [0.009, 0.148]. Graded suggestive, not decisive. Still too narrow, because steps are autocorrelated.

**Did anything in the length-bias family survive correction?** No. The old p < 0.001 was a t-test on three identical values. The exact sign-flip over three seeds floors at 0.25, and that is what it gets. After Benjamini–Hochberg over the fourteen tests, the smallest q is 0.116.

**Was the 0.5B PPO-versus-GRPO comparison item-paired?** No. Paired by seed only. The two arms of a seed share 3 to 7 of 200 eval items. About 40% of PPO's eval items were seen in training, against about 7% for GRPO. Both at 0.99, so the null stands; it is not equivalence.

**What is 11/64?** After eight parser-zero answers, at least one reference match in 24 fresh draws, on a deterministic first-64 eligible subset. The cohort is not a random sample. Assistant review is not independent human validation. Parser v3 was checked and not deployed.

**Why so few seeds?** The completed studies were limited by their budget. Open-source comparisons use three to eight seeds; Tinker single-seed figures are descriptive. No universal seed count guarantees power. A follow-up should prespecify a meaningful effect size, variability assumption and power rationale before collecting results.

**Is the code a reproduction of every run?** No. The submission ZIP is a review package. Weights, hidden keys, and provider logs are not in it. Six original contracts need access this repository does not have.

## Still not something a rebuild can finish

- Read the PDF and approve the claims, the authorship line, and the expanded AI-tool declaration drafted on 16 October 2026 (it now names the agentic tools; see the README note).
- Guide, Head of Department, and examiner signatures. The prepared certificate is not an approval.
- Institutional similarity certificate, portal format, and the upload itself.
- WebArena regrade: the native judge model is retired, so a regrade would be a different judge.
- SWE-bench Multilingual is 1/190 and 4/110 under two serving contracts. Do not add them.
- Original contracts still closed on external access: SDAB, BinaryAudit payload, LifeSciBench package, AgentHarm, AppBench, FrontierMath hosted evaluation.
- Four suite-receipt pins stay unfrozen on purpose: BinaryAudit has no LICENSE file. Frontier-SWE is Proximal-Labs/frontier-swe at `422b9bb95deb8efe436becb0ed3c44be23611e10` (2026-08-07), and GitHub reports no license there. LifeSciBench has no public artifact. SDAB is provider-only.

## Earlier implementation additions (3 October)

`make_gspo_loss_fn` is in the trainer, off by default, clip 3e-4 / 4e-4 as in Zheng et al. 2025. A prompt-conditioned value baseline is also in the trainer, off by default: advantages are reward minus V(prompt), then normalized across the batch. Neither has a measured run. Do not put either on a slide. The VerilogEval v1.0.0 pin is the full commit `4b9b16e92f1d9cc520afbfa3ecd5a2f20a350fd5`. v2.0.0 peels to `c498220d0a52248f8e3fdffe279075215bde2da6`.
