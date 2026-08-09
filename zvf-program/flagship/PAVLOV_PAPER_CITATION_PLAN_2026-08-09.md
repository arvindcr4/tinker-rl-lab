# Pavlov paper-citation plan — 2026-08-09

Status: **insertion plan only**. This file is the sole deliverable of the
readiness-audit lane. It does not edit any manuscript, add bibliography keys,
or promote a protocol, preflight, receipt, or rejected run into scientific
evidence.

The canonical roster is the twelve active roots in
[`platform_hybrid/paper/PAPERS_README.md`](../../platform_hybrid/paper/PAPERS_README.md).
R01, R02, R06, R07, U01, and P08_fraud remain absorbed historical roots and
must not receive independent citations or claims.

## Evidence vocabulary and hard boundary

Use these labels in every insertion and in any future review:

- **P — protocol**: [`PAVLOVS_LIST_TASK_CONTRACT.md`](PAVLOVS_LIST_TASK_CONTRACT.md),
  [`pavlovs_domain_contract.json`](pavlovs_domain_contract.json), and the
  deterministic manifest preview. This is prospective design evidence only.
- **X — observed xLAM slice**:
  [`autoresearch/orchestrator-260809-0922/base_eval_100.json`](../../autoresearch/orchestrator-260809-0922/base_eval_100.json).
  It is a single-turn strict function-calling base-model measurement: 100
  held-out xLAM examples, 7/100 perfect calls, mean strict reward 0.070. xLAM
  is not one of the Pavlov T1–T12 or E1–E14 suites.
- **R — rejected smoke**:
  [`autoresearch/orchestrator-260809-0922/rejected_untracked_smoke.json`](../../autoresearch/orchestrator-260809-0922/rejected_untracked_smoke.json).
  Four completed steps, no trained checkpoint, and no held-out result. It is
  provenance for a failed tracking gate, never an improvement result.
- **M — portfolio evidence**: immutable, seed-level results across the exact
  training/evaluation suites required by the row below. M does not exist yet.

Safe wording may describe P and, where explicitly permitted below, X. It must
not say that P, X, or R demonstrates model improvement, company usefulness,
production readiness, or a winning algorithm/controller.

## Complete suite key (all 26 IDs are mandatory accounting items)

### Training suites

| ID | Contract suite | Domain tags |
|---|---|---|
| T1 | `openreward_train` | multi-domain, tool-use, browser, science, ML, games, long-horizon |
| T2 | `swe_gym_train` | code, long-horizon |
| T3 | `browsergym_train` | browser, computer-use, enterprise, tool-use, long-horizon |
| T4 | `bfcl_train` | tool-use, code |
| T5 | `scienceworld_train` | science, long-horizon, tool-use |
| T6 | `unix_ctf_train` | security, ML, code, tool-use |
| T7 | `agentdojo_train` | alignment, security, tool-use, enterprise |
| T8 | `rtlcoder_train` | chip-design, code |
| T9 | `crafter_train` | games, long-horizon |
| T10 | `visual_app_train` | design, computer-use, code |
| T11 | `api_bank_rlvr_train` | finance, enterprise, tool-use, long-horizon |
| T12 | `openr1_math_train` | math |

### Primary held-out suites

| ID | Contract suite | Domain tags |
|---|---|---|
| E1 | `swe_bench_pro_eval` | code, long-horizon |
| E2 | `frontier_swe_eval` | code, ML, long-horizon |
| E3 | `sdab_eval` | code, ML, long-horizon, enterprise |
| E4 | `banker_toolbench_eval` | finance, enterprise, tool-use, long-horizon |
| E5 | `apex_agents_eval` | multi-domain, finance, enterprise, long-horizon, tool-use |
| E6 | `webbench_eval` | browser, computer-use, enterprise |
| E7 | `binaryaudit_eval` | security, code, long-horizon |
| E8 | `lifescibench_eval` | science, long-horizon, tool-use |
| E9 | `mle_bench_eval` | ML, code, long-horizon |
| E10 | `agentharm_eval` | alignment, security, tool-use |
| E11 | `verilog_eval` | chip-design, code |
| E12 | `appbench_eval` | design, computer-use, code |
| E13 | `openreward_games_eval` | games, long-horizon, tool-use |
| E14 | `frontiermath_eval` | math |

The T/E key covers every contract domain tag: `alignment`, `browser`,
`chip_design`, `code`, `computer_use`, `design`, `enterprise`, `finance`,
`games`, `long_horizon`, `math`, `ml`, `multi_domain`, `science`, `security`,
and `tool_use` (rendered above with readable hyphens where appropriate).

`math500_eval` is secondary only. `gsm8k_calibration` is calibration-only and
is never a T-suite or an E-suite.

## Receipt and result gates

For every M claim, attach all contract-required receipts:

1. immutable dataset/benchmark revision and license;
2. disjoint train/evaluation split manifest and task-ID hashes;
3. container/environment digest;
4. model revision and, for training, adapter revision;
5. seed, stack, run ID, budget, and verifier identity;
6. per-step loss/reward/telemetry and final metrics;
7. held-out result receipt with domain, horizon, reward/verifier, artifact vs.
   stateful, and seen vs. unseen-family slices; and
8. W&B/Hugging Face links or equivalent durable artifact locations.

The current X receipt contains row-level prompt/target/response hashes and
cost, but not the full contract receipt tuple. The rejected R record explicitly
cannot satisfy a result gate. The campaign manifest is therefore still
`BLOCKED` until model revisions, licenses, dataset revisions, and disjoint
task-ID hashes are recorded.

Call the complete eight-item receipt bundle **C0** below. C0 is necessary for
an M result claim but is not itself a result.

## Active-paper insertion plan

Each insertion point below is an anchor in the current source. The wording is
safe to use today, but should be accompanied by a citation to P only after the
manuscript's bibliography/venue format is updated by the primary owner.

### P1 — scaling limits / identifiability

- **Canonical active root:** `platform_hybrid/paper/paper_P1_scaling.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/sections/p1_conclusion.tex`, in `Discussion and
  Limitations`, immediately after the paragraph beginning
  `Submission scope.` and before `\section{Conclusion}`.
- **Suite map:** closest contract analogues are T4 (tool calls), T8 (code),
  and T12 (math), with E11, E14, and the tool-use E4 control. Any
  program-level usefulness or external-validity claim requires the full
  T1–T12 and E1–E14 portfolio.
- **Safe wording:** “The Pavlov’s List contract is a prospective gate for
  future multi-domain fine-tuning and held-out evaluation; it supplies no new
  evidence for this limits/identifiability audit. GSM8K remains a calibration
  control, not a usefulness result.”
- **Prohibited wording:** “Pavlov scaling shows a positive law,” “the xLAM
  baseline establishes scale-dependent usefulness,” or “the 12/14 suite
  portfolio has been run.”
- **Gate:** M requires paired model-scale cells, pinned revisions, multiple
  seeds, and C0 receipts for every cited suite; X and R cannot satisfy it.

### P2 — descriptive ZVF diagnostic

- **Canonical active root:** `platform_hybrid/paper/paper_P2_zvf.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/sections/p2_conclusion.tex`, after the final
  `Discussion and Limitations` paragraph ending `...cross-examination sets.`
  and before `\section{Conclusion}`.
- **Suite map:** diagnostic transfer across Pavlov domains requires all T1–T12
  and E1–E14. The existing GSM8K reward tensors and historical tool-use rows
  are not those suites.
- **Safe wording:** “The Pavlov portfolio is a prospective cross-domain test
  surface for whether diagnostics transfer beyond the named cells; this paper
  makes no such transfer claim.”
- **Prohibited wording:** “Low ZVF predicts Pavlov usefulness,” “xLAM validates
  ZVF,” or “ZVF is a causal controller signal.”
- **Gate:** per-suite reward tensors, ZVF/GU traces, and linked held-out
  outcomes are required in addition to C0.

### P3 — group size / preference density

- **Canonical active root:** `platform_hybrid/paper/paper_P3_group_size.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/sections/p3_conclusion.tex`, after the discussion
  paragraph ending `...group baseline may carry.` and before
  `\section{Conclusion}`.
- **Suite map:** a group-size external-validity claim requires all T1–T12 and
  E1–E14. T4/T8/T12 are the nearest tool/code/math analogues to the present
  benchmark, not replacements for the portfolio.
- **Safe wording:** “The contract specifies where a future token-matched,
  seed-paired group-size study would be evaluated; no Pavlov group-size
  optimum is claimed here.”
- **Prohibited wording:** “xLAM proves the best G,” “G=4 is universally
  optimal,” or “portfolio held-out retention has been established.”
- **Gate:** fixed-token, seed-paired arms and held-out metrics across the
  selected suites, with C0 and complete group-size manifests.

### P4 — bounded length-bias null

- **Canonical active root:** `platform_hybrid/paper/paper_P4_length_bias.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/sections/p4_conclusion.tex`, after the paragraph
  ending `...rather than on this paper's claims.` and before
  `\section{Conclusion}`.
- **Suite map:** long-horizon/stateful length tests should cover T1, T2, T3,
  T5, T6, T7, T9, T10, and T11, with T4/T8/T12 as controls; the primary
  held-out gate remains E1–E14.
- **Safe wording:** “The Pavlov contract identifies longer-horizon and
  stateful environments for a future length-confounding test; the present
  200-token GSM8K null does not address them.”
- **Prohibited wording:** “No length bias exists in agentic workloads,”
  “Dr. GRPO improves Pavlov usefulness,” or “the xLAM slice is a length-bias
  evaluation.”
- **Gate:** uncapped/length-confounded runs, causal mediation and truncation
  receipts, plus C0 and held-out artifact/state checks.

### P5 — MIN-REPORT-RL reporting standard

- **Canonical active root:** `platform_hybrid/paper/paper_P5_minreport.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/sections/p5_conclusion.tex`, immediately after the
  final sentence `Report the stack, not the label.` (the source wraps after
  `Report`).
- **Suite map:** T1–T12 and E1–E14 may be named as schema/manifest entry types;
  no suite execution is needed for the reporting-standard claim. Full 26-suite
  M evidence is required only if P5 adds a model efficacy or usefulness claim.
- **Safe wording:** “For future Pavlov runs, the eight-item MIN-REPORT-RL
  record should carry the applicable T/E suite IDs and their immutable
  receipts. This standard is a provenance requirement, not a result from those
  suites.”
- **Prohibited wording:** “The schema proves the model works,” “the xLAM
  receipt is a portfolio result,” or “suite IDs imply coverage.”
- **Gate:** schema applicability is immediate; any performance sentence needs
  C0 plus durable per-suite result receipts. X may be a worked receipt example;
  R must be marked rejected provenance.

### P6 — GRPO-Registry resource

- **Canonical active root:** `platform_hybrid/paper/paper_P6_registry.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/sections/p6_conclusion.tex`, after the final
  conclusion paragraph ending `...we did not catalog.`
- **Suite map:** all T1–T12 and E1–E14 can be represented as registry entry
  types; no execution is needed for the schema/resource claim. A portfolio
  usefulness claim requires all 26.
- **Safe wording:** “The registry can record Pavlov T/E identifiers, split
  hashes, stack fingerprints, and result status; an entry records provenance
  and does not certify performance.”
- **Prohibited wording:** “Registry coverage means benchmark coverage,”
  “xLAM makes the registry’s model entry successful,” or “R is a training
  result.”
- **Gate:** every populated entry needs C0; X can be a provenance entry only,
  and R must retain `inadmissible_provenance_only`.

### P7 — ZVF controller audit / test plan

- **Canonical active root:** `platform_hybrid/paper/paper_P7_zvf_controller.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/sections/p7_limitations.tex`, after the paragraph
  ending `...dry-run placeholders` and before
  `\subsection{Proposed decisive experiments}`.
- **Suite map:** controller external validity requires all T1–T12 and E1–E14;
  the current single-task evidence is not a portfolio subset receipt.
- **Safe wording:** “The Pavlov contract is a prospective external-validity
  gate for any future controller study; adaptive G remains a proposal until a
  seed-paired, fixed-token bakeoff and held-out portfolio evaluation succeed.”
- **Prohibited wording:** “The controller wins on Pavlov,” “xLAM validates
  adaptive G,” or “the rejected smoke demonstrates a training signal.”
- **Gate:** static G=16 and naive-boundary comparators, fixed tokens, multiple
  seeds, per-domain held-outs, and C0.

### P8 — exploratory workshop artifact

- **Canonical active root:**
  `platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex`, after
  the `Reproducibility and Limitations` paragraph ending
  `...we do not claim they do.` and before
  `\input{sections/conclusion_workshop}`.
- **Suite map:** current-topic analogues are T4/T8/T12 and E4/E11/E14; the
  full portfolio T1–T12/E1–E14 is required for a Pavlov/main-track claim.
- **Safe wording:** “The observed xLAM slice is a separate single-turn strict
  tool-call base baseline (7/100 perfect calls; mean strict reward 0.070). It
  is not a stateful, artifact-producing, multi-domain Pavlov result; those
  evaluations remain prospective.”
- **Prohibited wording:** “xLAM proves agentic usefulness,” “tool-use 0% rows
  and xLAM are comparable improvement arms,” or “the workshop artifact covers
  all 53 companies.”
- **Gate:** same-stack xLAM training and frozen held-out evaluation for any
  xLAM delta, then C0 and state/artifact receipts for the full portfolio.

### P9 — tiered benchmark / artifact paper

- **Canonical active root:**
  `platform_hybrid/paper/neurips_2026_variants/paper_P9_dnb.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/neurips_2026_variants/sections/reproducibility_card_dnb.tex`,
  after the `Hash manifest` paragraph ending `...agreement.` and before the
  `Submission-time status` paragraph.
- **Suite map:** current-topic analogues are T4/T8/T12 and E4/E11/E14; a
  program-level artifact/usefulness claim requires all T1–T12/E1–E14.
- **Safe wording:** “The xLAM receipt may be listed as one strict tool-use base
  baseline. It does not replace stateful tool, browser, code, artifact, or
  domain-family receipts required by the Pavlov contract.”
- **Prohibited wording:** “The tiered artifact has completed the Pavlov
  portfolio,” “xLAM is a post-trained improvement,” or “partial tool-use JSONL
  is held-out portfolio evidence.”
- **Gate:** raw tool-use JSONL, verifier/state artifacts, pinned splits, and
  complete C0 receipts; preserve the current A/B/C evidence grades.

### P10 — ZVF calibration theory

- **Canonical active root:** `zvf-program/theory/paper_P10_zvf_theory.tex`.
- **Source and insertion point:**
  `zvf-program/theory/paper_P10_zvf_theory.tex`, immediately after the
  `August 2026 evidence boundary` paragraph and before `\section{Introduction}`.
- **Suite map:** current theory needs only T12/E14 as a math analogue; GSM8K
  remains calibration-only. Any expansion to usefulness requires all 26.
- **Safe wording:** “The Pavlov contract is an application boundary for future
  multi-domain validation; the results here remain conditional theory and
  math/calibration evidence.”
- **Prohibited wording:** “The theorem proves agentic usefulness,” “FrontierMath
  or xLAM validates the controller,” or “calibration is a primary portfolio
  evaluation.”
- **Gate:** no new gate for the current theorem claim; C0 and all relevant M
  receipts are required before extending the claim beyond math/calibration.

### P11 — single-stack reproducibility audit

- **Canonical active root:**
  `zvf-program/audit/paper_P11_reproducibility_audit.tex`.
- **Source and insertion point:**
  `zvf-program/audit/paper_P11_reproducibility_audit.tex`, immediately after
  the `August 2026 evidence boundary` paragraph and before the first protocol
  figure.
- **Suite map:** the audit protocol is suite-agnostic; a Pavlov application
  requires all T1–T12/E1–E14. The current 40 units are a bounded GSM8K
  single-stack audit, not portfolio evidence.
- **Safe wording:** “The fail-closed audit gate is compatible with the Pavlov
  receipt contract; this paper’s 40-unit GSM8K audit does not establish
  cross-suite or company-family usefulness.”
- **Prohibited wording:** “The 40 units validate Pavlov readiness,” “INCONCLUSIVE
  algorithm deltas are improvements,” or “xLAM is an audited GRPO arm here.”
- **Gate:** per-suite stack/treatment fingerprints, held-out rows, seed-level
  verdicts, and C0; preserve the existing `INCONCLUSIVE` outcomes.

### P12 — GRPO/PPO/SAO diagnostic and contract

- **Canonical active root:**
  `platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex`.
- **Source and insertion point:**
  `platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex`,
  immediately after the `August 2026 evidence boundary` paragraph and before
  the first figure.
- **Suite map:** PPO/SAO external validity requires all T1–T12/E1–E14. The
  current checked-in GRPO artifacts are companion evidence, not a Pavlov
  subset.
- **Safe wording:** “The Pavlov contract supplies the future matched-budget,
  stateful evaluation surface for PPO/SAO; this paper reports no PPO/SAO
  training outcome or portfolio usefulness result.”
- **Prohibited wording:** “xLAM validates EGM/ZUF,” “TriageRL improves held-out
  Pavlov success,” or “the GRPO companion artifacts are an independent
  replication.”
- **Gate:** matched seed-paired PPO/SAO training, fixed held-outs, causal
  controller comparisons, and C0 across the full portfolio.

## Meeting-ready priority order

1. **P5, then P6 — provenance first.** These are the only immediate
   schema/resource insertions: T/E IDs and receipt fields can be specified
   without executing a suite. Keep any performance sentence behind C0 and M.
2. **P8, then P9 — bounded observed baseline.** Discuss X as one strict,
   single-turn xLAM base-model slice (7/100; 0.070) only. It is a baseline
   receipt, not post-training, stateful, artifact, or portfolio evidence; R is
   rejected provenance only.
3. **P1–P4 and P7 — historical claims with prospective gates.** Insert only
   the protocol boundary and the paper-specific subset/full-portfolio gate.
   No limits, transfer, group-size, length, or controller efficacy sentence
   can use X, R, or P as M.
4. **P10–P12 — calibration/audit/algorithm boundary.** Retain GSM8K and the
   40-unit audit as their bounded roles; any math-only extension uses T12/E14,
   while usefulness, PPO/SAO, or cross-domain claims require all 26 suites.
5. **Absorbed roots — audit-only.** Do not add independent insertion points
   for R01, R02, R06, R07, U01, or P08_fraud; cite their live parent only for
   the historical material listed below.

## Absorbed-root handling

Do not insert new citations into these historical roots. If their evidence is
needed, cite the live parent and state the absorption explicitly:

| Absorbed root | Live parent | Historical path | Permitted use |
|---|---|---|---|
| R02 | P2 | `platform_hybrid/paper/archive/absorbed/R02_main_zvf/main_zvf.tex` | historical ZVF/tool-use collapse only |
| R06 | P5 | `platform_hybrid/paper/archive/absorbed/R06_min_report/min_report_rl.tex` | historical reporting-standard material |
| R07 | P6 | `platform_hybrid/paper/archive/absorbed/R07_grpo_registry/grpo_registry.tex` | historical registry material |
| R01 | P9 | `platform_hybrid/paper/archive/absorbed/R01_acm/acm_main.tex` | historical benchmark packaging |
| U01 | thesis + P9 | `platform_hybrid/paper/archive/absorbed/U01_main_compendium/main.tex` | historical compendium only |
| P08_fraud | thesis | `platform_hybrid/paper/archive/absorbed/P08_fraud/paper_P8_fraud.tex` | measurement discipline; no RL evidence |

None is an independent Pavlov/xLAM result.

## Verification checklist for the primary owner

- [ ] All twelve live source paths P1–P12 appear once in the insertion plan.
- [ ] T1–T12 each appear in the suite key and in at least one paper gate.
- [ ] E1–E14 each appear in the suite key and in at least one paper gate.
- [ ] X, R, and M are kept separate in prose, tables, and receipts.
- [ ] P5/P6 are treated as schema/resource papers: suite IDs are representable
      without implying execution.
- [ ] No manuscript is edited until the primary owner authorizes the insertion.
- [ ] Before any main-track efficacy wording, run the validator and manifest
      preview, then attach all M receipts; do not launch paid work from this
      plan.
