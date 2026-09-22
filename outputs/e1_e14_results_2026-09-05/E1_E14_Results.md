# E1–E14 experiment results — 5 September 2026

**The campaign is incomplete: 2 full-suite results, 5 partial lanes and 7 lanes blocked on external inputs.**
The results below come from saved native-evaluation records. They were reconciled today; no new experiment or paid compute allocation was launched.

| Experiment | Benchmark | Verified result | Suite status |
|---|---|---|---|
| E1 | SWE-bench Pro | 2/731 = 0.274% pass@1 | Full-suite result |
| E2 | FrontierSWE | 1/17 tasks; replay normalized score 0.8628 | Partial |
| E3 | SDAB | No exact-suite result | Blocked |
| E4 | BankerToolBench | 1/100 tasks; recovery metric 0.3115 | Partial |
| E5 | APEX-Agents | 11/480 attempted; 7 native-scored; prefix mean 0.050505 | Partial |
| E6 | WebBench | No exact-suite result | Blocked |
| E7 | BinaryAudit | 1/46 tasks; verifier reward 0.0 after agent error | Partial |
| E8 | LifeSciBench | 0/750 tasks | Blocked |
| E9 | MLE-bench | 40/75 competitions with native grades; suite score unavailable | Partial |
| E10 | AgentHarm | No private-suite result | Blocked |
| E11 | VerilogEval | 129/312 = 41.35% pass@1 | Full-suite result |
| E12 | AppBench | No exact-suite result | Blocked |
| E13 | OpenReward Games | No exact-suite result | Blocked |
| E14 | FrontierMath | No private-suite result | Blocked |

## What the results mean

E1 resolved 2 of 731 cases. It has 713 native evaluations, 14 generation failures and four lost generation artifacts; all 731 remain in the denominator. Its one-checkpoint evaluation used 476 Tinker generations and 255 Modal vLLM generations. This backend mix is recorded in the original receipt.

E11 passed 129 of 312 evaluated instances: 67/156 code-completion and 62/156 specification-to-RTL instances. The canonical score is 41.35%; 129/311 is only a secondary sensitivity calculation.

E5's 0.050505 prefix mean treats unscored attempts as zero for an attempt summary; it is not a native 480-task score. The seven scored tasks average 0.079365. E4's 0.3115 is a verifier recovery metric, not the simple fraction 37/128 and not a suite score.

E9 has 97 legacy receipts across 50 competition IDs, with 41 graded receipts covering 40 unique competitions. The separate merged-vLLM arm has seven receipts, one valid grade (H&M: 0.02132) and six invalid submissions. These arms are not combined. One old Spooky Author submission is missing locally; the later run's complete submission is retained.

## Colab and Tinker

The live Tinker query returned checkpoint not found, and the original training run returned an empty checkpoint list. The pinned Hugging Face adapter revision remains accessible, including its 2,245,975,768-byte weights file. This audit verified Hub metadata, not a fresh download of every weight byte.

Colab reported no active session. Previous A100 recovery canaries failed with out-of-memory and CUDA illegal-memory-access errors. The existing 14-source public-portfolio Colab preflight only verified source pins; it produced no E1–E14 benchmark result. All checked E1–E14 Modal deployments were idle.

The companion notebook is self-contained and can be opened in Colab to review and reproduce the result calculations. It is an evidence-review notebook, not an experiment runner or a claim that Colab executed these benchmarks.

## Requirements to finish

- **E2 — FrontierSWE:** Revision-bound benchmark authorization and 16 remaining tasks; original Tinker checkpoint unavailable.
- **E3 — SDAB:** Private 80-task bundle, runtime/reset contract and native grader.
- **E4 — BankerToolBench:** Working model/verifier endpoint and budget: the prior native-grader-only estimate exceeds the recorded remaining cap.
- **E5 — APEX-Agents:** Working inference runtime, full auxiliary assets, sufficient budget and a declared execution protocol.
- **E6 — WebBench:** Official live environment, reset procedure, ground truth and native scorer.
- **E7 — BinaryAudit:** Revision-bound benchmark authorization, working model runtime and the remaining 45 tasks.
- **E8 — LifeSciBench:** Official task package and native grader.
- **E9 — MLE-bench:** 35 legacy competitions lack native grades; original sampler unavailable; separate merged arm is frozen.
- **E10 — AgentHarm:** Private held-out task files and authorized native grading route.
- **E12 — AppBench:** Official deployment, task artifacts and native grading route.
- **E13 — OpenReward Games:** Provider-defined immutable held-out games suite and grading contract.
- **E14 — FrontierMath:** Authorized hosted/private Epoch evaluation.

The last recorded additional-spend allowance is $50.00, with $5.6301 counted and $44.3699 remaining. This is the existing ledger, not a current provider invoice. The earlier E4 grader-only estimate (~$59.45) and E5 compute scenarios (~$143–$289 before environments and judging) exceed it; these dated estimates are not fresh quotes.

The checked provider channels still have no new grant: [WebBench request](https://github.com/Halluminate/WebBench/issues/2), [AppBench request](https://github.com/AfterQuery/appbench.ai-docs/issues/1), and [OpenReward wrapper request](https://github.com/EnvCommons/wordle/issues/3). A targeted incoming-mail search found no provider access reply. [Epoch's evaluation information](https://epoch.ai/frontiermath/tiers-1-4/about) remains the reference for the hosted FrontierMath route.

## Validation and provenance

394 local checks passed: denominator arithmetic, E1 image-manifest hash, E5 task receipt hashes, E9 receipt integrity and available graded-artifact hashes. The existing completion-gate tests passed 15/15. Four missing files were restored only after their bytes matched the original recorded hashes. No native grades, model outputs or historical receipts were changed.

The source manifest identifies every audited file and SHA-256 digest. Live provider observations, all individual E9 receipt rows, restoration provenance and validation checks are included beside this report.

These outcomes do not establish multi-seed performance or complete the broader research claims. The unfinished experiment work remains open.
