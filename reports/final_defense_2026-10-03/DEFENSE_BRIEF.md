# Phase 2 defense brief

Twelve minutes of talk, ninety seconds of demo, then questions. The deck is `TinkerRL_Phase2_Defense_2026-10-03.pptx`. Rebuild from the repository root with `NODE_PATH=/tmp/pptxgenjs-run/node_modules node reports/final_defense_2026-10-03/build_defense_deck.js` after `npm install --prefix /tmp/pptxgenjs-run pptxgenjs@3.12.0`. Every number below is the 3 October 2026 public thesis, not the September review deck.

Do not present `platform_hybrid/sem 4 work/submissions/mtech-final-review/Arvind_MTech_Thesis_Review.pptx`.

## The talk

**0:00 Title.** "This is a measurement thesis. I held the stack fixed, changed one factor, and reported what the evidence can carry."

**0:40 The rule.** Two instruments. P1–P8 asks what GRPO does. E1–E14 asks what one 40-step Tinker checkpoint scores. Only Tinker and TRL finished runs, on different base checkpoints. This is not a completed seven-library benchmark.

**2:00 The phenomenon.** Eight correct samples: mean 1, standard deviation 0, advantages 0. All-wrong groups look the same, so ZVF cannot tell mastery from failure. Phase-1 main configuration sits at 0.72–0.77. Across runs the range is 0.16–0.84. Association with held-out outcome is ρ = 0.27, n = 23, interval [−0.37, 0.88]. Three stored GSM8K tensors (Qwen3-8B, sampling only) reproduce 0.130, 0.190 and 0.155, pooled 0.1583. After mean reward is regressed out, r = +0.21, 95% CI [−0.31, +0.62], n = 17.

**3:30 Four answers.**

- ZVF is recomputable and is not a predictor.
- Algorithm labels are unanswered at this budget.
- One manifest pair is flagged by `stackdiff` with no training rerun. Whether that flag matters for outcomes was not tested beyond that pair.
- Governance produced receipts and a terminal state for every lane, not a training delta.

**5:00 The one number they will remember.** Unsaturated Qwen2.5-1.5B, GSM8K-CoT, five paired seeds: GRPO at group size 8 minus an untuned-critic PPO is +5.0 percentage points, interval [+1.5, +8.5]. Paired t p = 0.016. Exact sign-flip p = 0.0625. The critic was not tuned, prompt exposure differed, and the token cap can bind. On saturated 0.5B addition both methods are at about 0.99, paired difference −0.002, p = 0.374. Group size 8 minus 2 on the unsaturated rerun is −0.5 percentage points and resolves neither a difference nor equivalence at ±0.02.

**7:00 What I am not claiming.** No scaling law. No length-bias result. No group-size optimum. None of the seven audited headlines is decisive. The suggestive one is H7, the iter-136 efficiency contrast, and Holm across that family of eight moves its p to 0.25. That correction is not the +5.0 pp gap. Both can show a sign-flip p of 0.0625 because five pairs agreed. The 99.9% versus 73.4% comparison is training reward on different model sizes. Do not defend it.

**8:15 Campaign.** Read four rows and then the prohibition. VerilogEval 129/312. 150 of 183 failures are extraction failures, most likely driven by the 1,024-token cap with thinking on rather than by format; no per-sample finish reason was recorded. Omni-MATH 2,271/4,428 = 51.29% with two unjudged rows counted as failures; 51.31% is the judged-row figure. Capped answers are right 34.5% of the time, uncapped 85.9%. SWE-bench Pro 2/731, and 593 of 713 patches were corrupt before any test ran. These are harness scores, not capability scores. WebArena 90/812 is a lower bound because 108 judge tasks were ungraded. AgentDojo 88/97 = 0.907 is benign utility, Wilson interval [0.833, 0.950]. Do not average the rows. The small paired base-versus-trained table is inconclusive in every lane.

**9:30 C1.** Frozen score 11/64, Wilson 9.88–28.21%. Three numerical witnesses sit beside that score and are not folded into it. One case matches an incorrect reference after eight correct opening answers. This is not a training improvement.

**10:15 Contribution.** The diagnostic, the eight-item minimum report, and the ledger with named gates. Novelty, if asked: a search through July 2026 found no equivalent GRPO run datasheet with rollout provenance. That is bounded, not a proof of absence.

**11:00 Expected pushes.** Seed count below ten. Tinker internals are closed. Some execution sources are lost. "Inconclusive" stays inconclusive.

**11:40 Demo.** Then stop.

## Demo

From the repository root, before entering:

```bash
./submission/demo/demo.sh --self-test
./submission/demo/demo.sh
```

Open `submission/demo/output/demo_report.html` directly. If they ask for a fresh process, `./submission/demo/demo.sh --serve`. Narration is in `submission/demo/DEFENSE_RUNBOOK.md`. The 68.75% on that page is the mean of one recorded artifact of 80 binary rewards. If the SHA check fails, show the failure.

Offline fallback, no network: `./submission/demo/defense_fallback/run.sh`.

Checked on 3 October 2026: demo self-test passed (7 tests). `tools/check_thesis_evidence.py` passed on the selected arithmetic. That checker does not rerun graders or authenticate provenance.

## Answers to have cold

**Did GRPO beat PPO?** Not as a general result. Ceiling null on 0.5B addition. One suggestive gap against an untuned critic on 1.5B GSM8K-CoT, paired t p = 0.016, exact sign-flip p = 0.0625. Holm family p = 0.25 is audit H7, the iter-136 efficiency contrast, not this gap.

**What did GRPO do to Qwen3-8B on held-out GSM8K?** The Instruct checkpoint scores 82.0% before RL and 83.3% after. The +1.3 points are not significant, p = 0.26. That 82.0% is the same Instruct checkpoint, not a separate base model.

**Are the methods equivalent?** No. A null at accuracy 0.99 cannot separate no effect from no headroom. The group-size TOST at ±0.02 does not pass for G = 16 versus G = 2 (p = 0.056).

**Does ZVF predict accuracy?** No. The interval on ρ contains zero. It is stronger as a description of catastrophic all-equal groups than as a predictor of the graded outcome.

**Why is standard deviation zero for a perfect group?** Relative advantages need differences inside the group. Equal rewards supply none.

**What did the fourteen suites show about training?** Nothing causal. No per-suite base model at the campaign's own scope. The later small paired rerun is inconclusive in every lane.

**Why is WebArena not 11.1% exactly?** 90 successes out of 812 attempts. 108 tasks were never graded. The rate if those had passed would be 198/812 = 0.244. The reported figure is the lower bound.

**Which Omni-MATH percentage?** 51.29% over all 4,428 dispositions. 51.31% over 4,426 judged rows. The recovered terminal note attaches the second decimal to the first denominator. The thesis keeps the arithmetic and leaves the note unchanged.

**Is 41% on VerilogEval a capability claim?** No. It is one actor under one generation budget, full coverage, pass@1: 129/312 = 41.35%, design-cluster bootstrap interval [34.3%, 48.7%] over 156 paired designs. 150 of 183 failures are extraction failures, most likely driven by the 1,024-token cap with thinking on. Mean response was 1,002 of 1,024 tokens; per-sample finish reasons were not recorded, so that is an inference. On 50 shared items the same adapter scores 33/50 with thinking off at 4,096 tokens, against 23/50 in this run. Engine and temperature also changed, so that is direction, not a corrected score. Some extracted code may come from unfinished reasoning drafts. The retained full run used the original managed sampler, which this package cannot replay.

**Is 2/731 on SWE-bench Pro a problem-solving score?** Mostly not. 593 of 713 patches fail `git apply` as corrupt: the hunk-header line counts do not match the bodies. The official script then ran the tests on the unpatched repository. With `--recount`, 711 parse. Both passes are among the 120 that parsed. 2/731 is what was measured; it mostly measures diff formatting. The runner now repairs headers. The stored score predates that.

**Does the algorithm explain at most 6% of ZVF variance?** No longer claimed. The old interval resampled pooled steps, a null band. Resampled within method: η² = 0.045, interval [0.009, 0.148]. Graded suggestive, not decisive. Still too narrow, because steps are autocorrelated.

**Did anything in the length-bias family survive correction?** No. The old p < 0.001 was a t-test on three identical values. The exact sign-flip over three seeds floors at 0.25, and that is what it gets. After Benjamini–Hochberg over the fourteen tests, the smallest q is 0.116.

**Was the 0.5B PPO-versus-GRPO comparison item-paired?** No. Paired by seed only. The two arms of a seed share 3 to 7 of 200 eval items. About 40% of PPO's eval items were seen in training, against about 7% for GRPO. Both at 0.99, so the null stands; it is not equivalence.

**What is 11/64?** The frozen C1 endpoint on a deterministic first-64 subset. The cohort is not a random sample. Assistant review is not independent human validation. Parser v3 was checked and not deployed.

**Why not ten seeds?** Cost. Henderson et al. recommend ten. No family here reaches that. Quantitative claims rest on the multi-seed open-source runs. Tinker single-seed figures are marked descriptive.

**Is the code a reproduction of every run?** No. The submission ZIP is a review package. Weights, hidden keys, and provider logs are not in it. Six original contracts need access this repository does not have.

## Still not something a rebuild can finish

- Read the PDF and approve the claims, the authorship line, and the AI-tool declaration.
- Guide, Head of Department, and examiner signatures. The prepared certificate is not an approval.
- Institutional similarity certificate, portal format, and the upload itself.
- WebArena regrade: the native judge model is retired, so a regrade would be a different judge.
- SWE-bench Multilingual is 1/190 and 4/110 under two serving contracts. Do not add them.
- Original contracts still closed on external access: SDAB, BinaryAudit payload, LifeSciBench package, AgentHarm, AppBench, FrontierMath hosted evaluation.
- Four suite-receipt pins stay unfrozen on purpose: BinaryAudit has no LICENSE file. Frontier-SWE is Proximal-Labs/frontier-swe at `422b9bb95deb8efe436becb0ed3c44be23611e10` (2026-08-07), and GitHub reports no license there. LifeSciBench has no public artifact. SDAB is provider-only.

## Code that landed with this pack

`make_gspo_loss_fn` is in the trainer, off by default, clip 3e-4 / 4e-4 as in Zheng et al. 2025. A prompt-conditioned value baseline is also in the trainer, off by default: advantages are reward minus V(prompt), then normalized across the batch. Neither has a measured run. Do not put either on a slide. The VerilogEval v1.0.0 pin is the full commit `4b9b16e92f1d9cc520afbfa3ecd5a2f20a350fd5`. v2.0.0 peels to `c498220d0a52248f8e3fdffe279075215bde2da6`.
