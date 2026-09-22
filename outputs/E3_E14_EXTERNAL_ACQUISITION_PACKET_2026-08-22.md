# E3/E6/E8/E9/E10/E12/E13/E14 external acquisition packet

Prepared: 2026-08-22 (Asia/Kolkata)

This packet records the shortest legitimate path to every remaining external input. Public assets already present locally are not re-requested. No provider-controlled task, hidden answer, licence, terms acceptance, or hosted score is treated as obtained until its provider issues a receipt.

| Lane | What is already local | External item still required | Official route | Prepared request |
|---|---|---|---|---|
| E3 SDAB | Contract adapter, 50 focused tests, 4/5 preflight gates | Immutable 80-task identity, live deterministic runtime, traffic generator, native grader | founders@emulated.so | `outputs/e3_sdab/ACCESS_REQUEST_SDAB_2026-08-09.md` |
| E6 WebBench | Pinned public CSV, licence, 2,647 task hashes, disjointness proof, 7/8 gates | Official browser environment, reset/credentials, ground truth, native verifier, allowed write-task scope | jerry@halluminate.ai | `outputs/e6_webbench/ACCESS_REQUEST_HALLUMINATE_2026-08-09.md` |
| E8 LifeSciBench | Protocol adapter and public announcement metadata | Official 750-task package, licence/grant, artifacts, interface, rubric grader, manifest and contamination statement | OpenAI contact-sales form; benchmark materials have no dedicated public inbox | `outputs/e8_lifescibench/ACCESS_REQUEST_lifescibench_2026-08-09.md` |
| E9 MLE-bench | Pinned official repo, full 75-ID split, all 75 rule sets accepted, six native model grades, native verifier image | Remaining 69 competition runs; full production agent image; training/eval disjointness manifest | `outputs/e9_mle_bench/kaggle_rule_acceptance_receipt_2026-08-22.json` | `outputs/e9_mle_bench/lane_status_2026-08-09.md` |
| E10 AgentHarm | Public split and official Inspect evaluator pinned | Three `test_private` files or AISI-hosted evaluation; approved policy/semantic judge | HF dataset discussion, then inspect_evals issue/AISI contact | `outputs/e10_agentharm/AISI_ACCESS_REQUEST.md` |
| E12 AppBench | Six public task rows pinned | Written licence, official deployment/reset artifacts, verifier, two qualified graders and re-adjudication | research@afterquery.com, support@afterquery.com | `outputs/e12_appbench/ACCESS_REQUEST_2026-08-22.md` |
| E13 OpenReward Games | Wordle source/runtime, public train/test seeds, deterministic native reward path | Explicit wrapper licence, provider-defined versioned held-out suite, deployed revision attestation | hello@openreward.ai | `outputs/e13_openreward_games/ACCESS_REQUEST_2026-08-09.md` |
| E14 FrontierMath | Public protocol and sample characterization only | Epoch-run hosted evaluation; private questions and grader are not distributed | math_evals@epoch.ai | `outputs/e14_frontiermath/ACCESS_REQUEST_2026-08-09.md` |

## Actions that can be requested immediately

1. **Sent 2026-08-22:** E3, E6, E12, E13, and E14 provider emails. Message IDs are sealed in `outputs/E3_E14_EXTERNAL_ACQUISITION_RECEIPT_2026-08-22.json`.
2. **Submitted 2026-08-24:** E8 request through OpenAI's contact-sales form using the prepared text; the page returned its sales-team confirmation.
3. **Opened 2026-08-24:** E10 request as [AgentHarm discussion #9](https://huggingface.co/datasets/ai-safety-institute/AgentHarm/discussions/9). The request prefers an AISI-hosted run over delivery of private harmful-behaviour files.
4. **Completed 2026-08-22:** all 75 Kaggle competition rule sets were accepted and verified through the authenticated gated download endpoint. This clears the user-terms gate only; it is not a suite score.

## Inputs providers will ask the requester to decide

- Affiliation and whether the result is internal or intended for publication.
- Whether provider pre-publication review and embargo terms are acceptable.
- E6 side-effect authorization for write-class browser tasks.
- E10 non-logging deployment or AISI-hosted execution.
- E12 human-grader quote and budget.
- E14 hosted-evaluation access mode and potentially substantial commissioned-evaluation budget.

## Secret handling

The `sk-...` value pasted in chat is not stored in this repository or included in any request. Rotate it before use. Its provider must be identified before it can be configured, and it cannot substitute for licences, terms acceptance, private task access, or human graders.
