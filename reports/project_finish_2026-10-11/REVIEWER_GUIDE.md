# Tinker RL Lab: reviewer guide

Prepared 11 October 2026 for the PES project evaluation. This guide explains the research contribution and routes a reviewer to existing evidence. It introduces no new experiment, benchmark score, or institutional approval. It is not an official marking rubric.

## The problem and the contribution

**A change in an RL result can come from the algorithm, the model, the serving stack, the reward parser, or the evaluation scope. This project builds the measurements needed to tell those explanations apart.** Its central question is when a comparison is attributable and when the available evidence cannot support it.

The deliverable is a measurement system with four connected parts: a recomputable diagnostic of missing group-relative learning signal; a minimum report of the experimental stack; a machine-readable registry for comparing those records; and a benchmark ledger that retains scope, coverage, failures, and external blockers. Controlled experiments test the interpretation of the diagnostics. The campaign then exercises the reporting discipline on one trained checkpoint.

This makes the work assessable through working implementation, explicit experimental design, traceable evidence, and qualified conclusions. A high benchmark number is not required to demonstrate those contributions, and no state-of-the-art performance is claimed.

## Read the right version

| Material | Status and intended use |
|---|---|
| [11 October thesis review copy](thesis/Thesis_Report_ArvindCR_2026-10-11_REVIEW.pdf) | Current review copy prepared from the corrected draft, with an explicit audit scope. Start here for this evaluation package. It is not a certified institutional submission. |
| [3 October public thesis and C1 addendum](../public_revision_2026-10-03/README.md) | Historical frozen public release and the source of the existing defense. Retained for comparison. |
| [4 October thesis draft](../public_revision_2026-10-04/thesis/README.md) | Unreviewed successor with Appendix J and five documented corrections. Consult it for those corrections; its existence does not promote it to a reviewed release. |
| [11 October defense](defense/TinkerRL_PES_Defense_2026-10-11.pptx) | Twelve-slide review deck with implementation, contribution, study scope and causal qualifications corrected. Use this deck for the current review. |
| [Historical identified submission](../../SUBMISSION.md) | Separate university thesis and package, with build instructions and remaining author/institutional actions. A public derivative is not a newly certified submission. |
| [Defense preparation](DEFENSE_PREP.md) | A concise speaking route and questions to rehearse alongside the existing deck. |

The 4 October draft corrects a Mantel correlation to +0.32; the TRL wall-time range to 136.4–183.7 seconds; unsupported PPO metadata in Table A.3; the sources for reconstructed token-budget rows; and E13's recorded length-finish count to 73/262. These are reporting corrections, not new model results. The draft README records the exact changes and its publication limitations.

## What belongs to the candidate

The shared repository originated with Semester 3 Group 6: six students produced the initial environment, integrations, experiments, literature foundation, group report, and main-track submission. Its boundary is `capstone-final-2026-04-25` at `21a99ef7`.

Semester 4 is Arvind C R's individual continuation under Ramesh Prakash Guledgudd. The documented additions include the P1–P8 research direction and analyses, diagnostic and reporting work, registry/controller studies, expanded experiments, the E1–E14 campaign, and the thesis. P1–P8 are internal research reports; their existence is not a claim of eight accepted publications. The controller's design and simulation are distinct from an experimentally demonstrated training benefit.

Start with [PROJECT_HISTORY.md](../../PROJECT_HISTORY.md), then [the Semester 4 provenance record](../../platform_hybrid/sem%204%20work/PROVENANCE.md) and thesis §1.6. Shared runtime code remains shared; commit/file counts are not used as a measure of academic contribution. The author must approve the thesis's authorship and AI-tool declarations.

## Evidence worth examining

Chapter and table numbers below refer to the thesis review copy. The linked source files expose the text and its underlying artifact references.

| Question | Defensible finding | Where to inspect it |
|---|---|---|
| Can the diagnostic be independently calculated? | Three stored GSM8K reward tensors give ZVF 0.130, 0.190, and 0.155; pooled 0.1583. Both uniformly correct and uniformly wrong groups have zero relative signal. | [Chapter 6 §6.3](thesis/ch06_results_core.md); [Table 1.1](thesis/ch01_introduction.md); [offline demonstration](../../submission/demo/README.md). |
| Does ZVF predict task performance? | The Colab lineage audit finds 17 finite cells rather than the reported 23; nine are synthetic and outcome definitions are mixed. The historical correlation cannot validate held-out prediction. ZVF remains a recomputable signal diagnostic. | Chapter 6 §6.3.4 and Table 1.1. |
| What survives a same-stack comparison? | On Qwen2.5-1.5B GSM8K-CoT, GRPO G8 minus minimal untuned-critic PPO is +5.0 percentage points over five paired seeds; paired-t CI [+1.5, +8.5], p = 0.016, exact sign-flip p = 0.0625. The baseline and test choice limit the result. | Chapter 6 §6.5.5 / Table 6.1; [summary](../../platform_hybrid/experiments/results/samestack_gsm8k_cot.json) and [raw study record](../../platform_hybrid/experiments/results/samestack_gsm8k_cot_full.json). |
| What did the reporting system demonstrate? | The eight-item minimum report and registry make stack differences explicit. `stackdiff` returns R5 for one recorded manifest pair (see [the runnable example](defense/REGISTRY_EXAMPLE.md)); broad predictive validity of its flag was not established. | [Chapter 7 §§7.1–7.2](thesis/ch07_results_infra.md). |
| How should benchmark outcomes be interpreted? | VerilogEval is 129/312; Omni-MATH is 2,271/4,426 judged rows or 2,271/4,428 dispositions; WebArena is a 90/812 lower bound with 108 ungraded tasks. Each describes its own actor/stack/scope. | [Chapter 8](thesis/ch09_results_campaign.md), [Table C.3](thesis/ch_back_notation.md), and [Appendix D](thesis/ch_back_campaign_detail.md). |
| What did the C1 extension add? | Frozen endpoint 11/64; three separate numerical witnesses after assistant-assisted review. Deterministic cohort selection and reference ambiguities prevent treating this as a population recovery rate or training improvement. | Chapter 7 §7.5, [Appendix G](thesis/ch_back_c1_evidence.md), [public case ledgers](../../research/public_results/README.md). |

Two comparisons must remain separate: the Chapter 6 controlled experiments and the E1–E14 actor evaluation. The latter's small paired base-versus-trained results are inconclusive in every lane (Table 8.B). No cross-suite average is meaningful. Original contracts, replacement benchmarks, and incomplete coverage retain their own identities.

## A ten-minute examination route

1. **Minutes 0–2: question and ownership.** Read this guide, thesis §1.6, and Table 1.1. Identify one concrete contribution and the study that tests it.
2. **Minutes 2–5: strongest controlled evidence.** Inspect Table 6.1 and its five-seed source. Ask why the paired t-test and exact sign-flip test lead to different strength of evidence.
3. **Minutes 5–7: implementation.** Run the demonstration below. It exposes the group-normalization mechanism and checks a recorded artifact against a fixed hash.
4. **Minutes 7–10: auditability.** Inspect Table C.3 and a lane receipt, then compare C1's frozen score with its separate semantic classifications. This demonstrates how the system records a result without overstating it.

From the repository root or the root of the extracted review ZIP, using Python 3:

```sh
python3 reports/project_finish_2026-10-11/verify_review_pack.py
python3 submission/demo/run_demo.py --self-test
python3 submission/demo/run_demo.py
```

Open `submission/demo/output/demo_report.html`. These selected checks and the deterministic demo require no model, credentials, GPU, or network. The four synthetic demo groups illustrate the mechanism; the separate recorded artifact's mean reward of 68.75% is not a new benchmark result.

In the full repository checkout, also run `python3 tools/check_thesis_evidence.py`. Its historical evidence dependencies are not all included in the review ZIP. That checker verifies selected stored arithmetic; it does not rerun native graders, authenticate upstream execution, validate every thesis claim, or reproduce training.

## Completion boundary

The project can be evaluated on its implemented measurement system and the evidence it actually produced. Completion of every original benchmark contract is not a prerequisite for explaining these contributions: unavailable native scopes remain explicitly blocked, and no replacement is passed off as the original. This guide does not predict a grade or substitute for the evaluator's criteria.

Institutional completion still requires the author's approval of the PDF, claims and AI-tool declaration; required guide/department/examiner signatures; any institutional similarity certificate; confirmed portal requirements; and an upload receipt. See the [submission checklist](../../SUBMISSION.md#before-the-actual-submission--author-checklist). External benchmark access and any separately authorized reruns are distinct from preparing a finished review package.
