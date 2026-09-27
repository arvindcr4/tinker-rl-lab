# 3. System Requirements Specification

This chapter states what the Tinker RL Lab system must do, what qualities it must hold while doing it, and how each requirement is verified. It inherits the specification of the Phase-1 report and extends it across the Semester-4 additions: the multi-framework benchmark rosters, the Zero-Variance Fraction (ZVF) diagnostic and the adaptive group-size controller derived from it, the minimum reporting standard and the machine-readable GRPO stack registry, and the E1–E14 evaluation campaign with its fail-closed completion gate. It is written as a specification rather than as a retrospective description, but every requirement below corresponds to a component that exists in the repository, and every functional requirement is paired with the concrete test that checks it. Where a requirement is not satisfied — the cloud capacity for two evaluation lanes, the white-box gradient path at the largest model scale — the shortfall is named in §3.6 and §3.8 rather than written around.

## 3.1 Overview and Actors

The central object of the Phase-1 specification was the **run**: one configuration, one seed, one framework, one training budget, producing per-step telemetry and a persisted set of raw group reward tensors (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Semester 4 introduces two further first-class objects. The first is the **lane**: a single benchmark suite evaluated under a named scope — an original contract with the suite's owner, or a declared replacement scope — carrying its own provider, payload, licence, quota and terminal state. The second is the **receipt**: an immutable, hash-bound record that a declared step actually happened, against declared inputs, at a declared revision. Requirements that concern runs, lanes and receipts are distinguished below where the distinction matters.

Four actors interact with the system, and the fourth is new in Semester 4.

**The experimenter** declares work. Under the original design this meant declaring a run through a single configuration object and nothing else: the model, the dataset, the reward function, the decoding parameters (temperature, top-$p$, maximum tokens), the group size $G$, the learning rate, the KL coefficient $\beta$ and the seed all live in one file, and the framework is a switch rather than a rewrite (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Under the campaign design the same actor declares a lane through a sealed resource request: the lane specification fixes an immutable task bundle, a pinned native runtime, a native grader, a budget and an applicable licence in advance, and those declared inputs are then checked for internal consistency before any launch is permitted (source: zvf-program/flagship/e1_e14_completion_gate.py).

**The back-end** executes the training loop and, under the managed path, is the Tinker service; under the white-box path it is a local or rented GPU. For evaluation, the corresponding actor is the **provider grader**: the benchmark owner's own native evaluator at a pinned revision. The system is permitted to record a suite score only when that evaluator produced it, or when a declared replacement harness reproduces it under a scope that is labelled as a replacement (source: zvf-program/flagship/e1_e14_completion_gate.py).

**The auditor** is a downstream script or an independent reader who recomputes published diagnostics from stored raw artefacts. In Semester 4 this role acquires force: the audit suite of §3.4 is executed as a gate with a non-zero exit status rather than as advisory tooling, and the completion gate for the campaign lanes is fail-closed by construction, in that a missing or malformed declaration stops the procedure instead of degrading it.

**The campaign operator** holds launch authority for paid work. Spend on any lane requires a bound authorization receipt issued by the operator; the system will not mint one for itself, and lanes are recorded as blocked when the receipt or the technical chain behind it is absent (source: outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md).

A fifth party, the **external access holder**, is an actor only in the negative sense: several lanes depend on a private payload, a hosted evaluation service or a quota grant that only the provider can issue. The specification requires that such a lane be recorded with an explicit terminal state and a named reopen condition rather than silently dropped, which is why the campaign's terminal-state vocabulary exists as a first-class output (source: outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md).

## 3.2 Functional Requirements

Table 3.1 lists the functional requirements. Each row states the requirement and names the source that defines it; §3.5 maps each to the artefact that implements it and the test that verifies it.

| ID | Requirement | Source |
|---|---|---|
| FR-1 | One declarative file fully determines a run (model, dataset, reward, decoding, G, learning rate, KL coefficient, seed, output); no framework-specific parameters are needed on any supported back-end | Phase-1 report |
| FR-2 | A configuration is executable on multiple frameworks through thin adapters that share the reward grader and decoding configuration. Two rosters are in scope and are kept apart: the cross-RL-library roster (TRL, Stable-Baselines3, CleanRL, Tianshou, PufferLib, rl_games, d3rlpy) and the cross-launcher LLM-RL roster (Tinker, TRL, SkyRL, veRL, OpenRLHF, Atropos, HF reference launchers). Completed-run parity is **not** met: only Tinker and TRL produced completed runs; the veRL and OpenRLHF entries are dry-run placeholders | `_shared_methods.tex`; `framework_comparison.json` |
| FR-3 | Every step computes ZVF and its all-correct / all-wrong decomposition from the reward tensors | Phase-1 report |
| FR-4 | Every step emits a structured telemetry record (reward, ZVF, gradient utilisation, collapse counts, entropy, completion length, KL) to a log and to the tracker; the controller work adds a per-step counterfactual against a matched fixed-G control | Phase-1 report; `controller_cf_per_step.tsv` |
| FR-5 | Raw per-group reward tensors are persisted for every run, so diagnostics can be recomputed | Phase-1 report |
| FR-6 | Sweeps over G, seed and baseline-versus-intervention arms run under a fixed wall-clock or token budget | Phase-1 report |
| FR-7 | For single-GPU models, per-layer LoRA gradient norms are recorded; not required at managed scale, where they are not exposed | Phase-1 report |
| FR-8 | Every run emits a provenance record binding configuration, grader version and rollout hashes; a campaign lane additionally declares its provider package and grant documents against published schema versions, each with a 64-character hash | Phase-1 report; `e1_e14_completion_gate.py` |
| FR-9 | A stack-conditioned result carries the seven manifest fields of the eight-item standard (loss form, reference KL, sampler/backend with base-checkpoint revision and hash, telemetry, group-size schedule, held-out split, decontamination) plus its eighth, evaluation item (held-out pass@k); a 0–100 badge is emitted, and an unreported field is distinguished from one reported as absent | `paper_P6_registry.tex`; `minreport_audit_summary.json` |
| FR-10 | The stack registry is queryable by stack, field or status | `registry/schema.json`; `registry/query.py` |
| FR-11 | Each named variant stores an explicit delta against the GRPO reference, and `stackdiff` returns an R0–R5 flip-risk verdict for same-label entries | `paper_P6_registry.tex`; `registry/provenance/` |
| FR-12 | An adaptive group-size controller with escalation asymmetry and hysteresis records its counterfactual against the best static recipe at matched and unequal rollout counts; it measures the trade and does not claim a win | `paper_P7_zvf_controller.tex`; `controller_cf_summary.json` |
| FR-13 | Each lane records results against a declared evidence class (exact complete, partial exact, partial recovery, externally blocked, local-setup-ready-provider-input-required) tied to the source status | `e1_e14_completion_gate.py` |
| FR-14 | The completion gate validates, and never executes, a lane package; a valid package proves only that its inputs are consistent and available; signatures are checked against provider trust roots, and the gate refuses rather than warns | `e1_e14_completion_gate.py` |
| FR-15 | Original-contract and replacement-scope results are never pooled, no cross-suite aggregate is computed, and every accuracy carries its denominator | `E1_E14_FINAL_RESULTS_2026-09-19.md` |
| FR-16 | The export path builds an anonymised review package whose build runs the audit suite, and refuses a package built with audits skipped | `run_all_audits.py`; `export_guard_audit.py` |

: Functional requirements. Sources: the Phase-1 report is `platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex`; other files are under `platform_hybrid/`, `platform_local/` or `zvf-program/flagship/`.

## 3.3 Non-Functional Requirements

| ID | Requirement | Source |
|---|---|---|
| NFR-1 Attribution | Comparative claims hold the stack fixed except the factor under test; confounded factors are reported as confounded | `_shared_methods.tex` |
| NFR-2 Reproducibility | Multi-arm claims rest on at least three seeds; single-seed results are descriptive only | Phase-1 report |
| NFR-3 Recomputability | Every derived diagnostic is reproducible from stored raw artefacts, independent of logged summaries | Phase-1 report |
| NFR-4 Denominator discipline | Held-out figures carry their n; negative, null and underpowered results are reported as such | `main_eai_body.tex` |
| NFR-5 Isolation | Concurrent experiments run in separate processes, not threads (after the tracker thread-safety defect) | Phase-1 report |
| NFR-6 Budget | Spend is recorded per run against a declared envelope; per-run bounds and lead-issued authorisation receipts apply even under an unlimited total | `Pending_Experiments.md` |
| NFR-7 Fail-closed | Validation defaults to refusal; an unverifiable package is not scored | `e1_e14_completion_gate.py` |
| NFR-8 Least privilege | Evidence records hold no key material; credential checks record only liveness and scope | `Pending_Experiments.md` |
| NFR-9 Lost-source auditability | Execution sources for sealed artefacts live in tracked paths; sealed requests and receipts suffice to pin a faithful re-implementation | `E1_E14_FINAL_RESULTS_2026-09-19.md` |
| NFR-10 External traceability | External payloads are bound by hash and validated against trust roots and schema versions; an unreported field is null, not zero or false | `e1_e14_completion_gate.py`; `paper_P6_registry.tex` |

: Non-functional requirements.

## 3.4 Test-Case Specification

The test cases below are the concrete checks attached to the requirements. TC-1 through TC-5 are inherited from the Phase-1 specification and are restated here because later chapters depend on them; TC-6 onward are the Semester-4 additions, and TC-6 is the suite that carries the largest share of the functional requirements.

**TC-1 — Recompute ZVF and the collapse decomposition from stored tensors.** Recompute the published ZVF and collapse figures from the persisted per-group reward tensors of the main configuration, using the documented analysis script. The published values are a ZVF of approximately 0.72–0.77 and an all-correct fraction of 0.65–0.71; recomputation must match (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Verifies FR-3 and FR-5.

**TC-2 — Recompute the detector metric from raw tensors.** Recompute the Phase-3 detector's area under the ROC curve from raw tensors under five-fold stratified cross-validation. The published figure is an AUROC of $0.84 \pm 0.01$ against $0.43$ for the reward-only comparator (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Verifies FR-5 and NFR-3.

**TC-3 — Re-run a matched-budget arm comparison.** Re-run the baseline and the curriculum arm at a matched token budget over three seeds; the Phase-1 result is that both arms average $+0.028$ on the held-out set with no curriculum advantage (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Verifies FR-1, FR-2 and FR-6.

**TC-4 — Recompute headline diagnostics.** Recompute each headline diagnostic quoted in a results chapter directly from stored artefacts, with the recomputation recorded next to the published value (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Verifies NFR-3.

**TC-5 — Provenance verification.** Verify a run's provenance record end to end, from configuration to rollout hashes to the grader version (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Verifies FR-8.

**TC-6 — Execute the submission audit suite as a gate.** Run `python platform_local/run_all_audits.py`. The suite executes nine audits — `claim_issues`, `sync_issues`, `anon_issues`, `strength_issues`, `package_issues`, `workflow_issues`, `export_guard_issues`, `caveat_issues` and `scientific_issues` — emits one `METRIC <name>=<count>` line per audit, and returns exit status 0 only when every audit reports zero issues (source: platform_local/run_all_audits.py; source: utils/audit_utils.py). Individual checks that this test enforces include the presence of scope and caveat language in the review documents (`readme_missing_key_results_scope_header`, `readme_missing_heldout_language`, `checklist_missing_preliminary_key_results_label`), the absence of overstated checkpoint claims (`misleading_checkpoint_availability_claim`, `submission_overstates_checkpoint_release`), the presence of every reviewer caveat (`heldout_scope`, `tool_eval_protocol`, `codegen_subset`, `capacity_confound`, `moe_diagnostics`, `budget_and_splits`, `replication_release`), the absence of banned marketing formulations (source: platform_local/claim_strength_audit.py), the survival of the technical caveats into the anonymous build (`anonymous_missing:<label>` for the seven needles), and the presence of the audit call and its skip flag in the export script (source: platform_local/paper_sync_audit.py; source: platform_local/export_guard_audit.py). Verifies FR-16, NFR-4 and NFR-7.

**TC-7 — Structural checks on the evaluation script and its result files.** For each held-out evaluation script, assert by parsing the source — not by reading its documentation — that it exposes a seed argument, locks the dataset split to the test split, samples with a non-zero temperature, consumes a checkpoint path, and writes the declared metadata keys including the split, the sampling flag, the seed and the model source. For each result file, assert that the schema version is 2, that the evaluation status is one of completed or failed, that the required configuration and summary fields are present, that the recorded split is the test split, that the attempted count equals the sum of correct and incorrect, and that a failed evaluation carries a failure reason and a completed evaluation carries non-zero attempts (source: platform_local/scientific_audit.py). Verifies FR-8 and NFR-4.

**TC-8 — Build the report from source.** Compile the main report and its supplementary through the full LaTeX chain and flag an empty journal field in the bibliography (source: platform_local/scientific_audit.py). Verifies FR-16.

**TC-9 — Dry-run the completion gate against a declared lane.** Present a provider package to the gate and require it to validate the declaration without executing anything: the expected outcomes are acceptance with the declared evidence class, or a specific refusal naming the missing element. A package must never be reported as a benchmark result by this procedure (source: zvf-program/flagship/e1_e14_completion_gate.py). Verifies FR-13, FR-14 and NFR-7.

**TC-10 — Registry query and null discipline.** Query the registry for a labelled entry and its delta record, and confirm that a field absent from the source is returned as unreported rather than as a zero value, and that a same-label pair with divergent behaviour triggers the flip-risk verdict (source: platform_hybrid/paper/paper_P6_registry.tex; source: platform_hybrid/registry/query.py). Verifies FR-9, FR-10, FR-11 and NFR-10.

## 3.5 Requirements Traceability Matrix

| Requirement | Implemented in | Verified by |
|---|---|---|
| FR-1 Configuration object | `platform_local/unified/launcher.py`; run configurations under `platform_hybrid/experiments/` | TC-3 |
| FR-2 Back-end-agnostic execution | `platform_local/trl_integrations/`; `platform_hybrid/experiments/results/framework_comparison.json` | TC-3; parity shortfall recorded in §3.2 |
| FR-3 Per-step ZVF | Training-loop telemetry (Phase-1 design) | TC-1 |
| FR-4 Per-step telemetry | Telemetry writer; `platform_hybrid/experiments/results/p5p8/controller_cf_per_step.tsv` | TC-1, TC-4 |
| FR-5 Raw tensor persistence | Persisted per-group reward tensors | TC-1, TC-2 |
| FR-6 Sweeps | Sweep runner | TC-3 |
| FR-7 White-box gradients | Single-GPU training path | Not automatable in the audit suite; checks in Chapter 5 §5.5 (per-layer gradient profile) |
| FR-8 Provenance record | `zvf-program/flagship/e1_e14_completion_gate.py`; run provenance records | TC-5, TC-7, TC-9 |
| FR-9 Minimum-report manifest | `platform_hybrid/experiments/results/p5p8/minreport_audit_summary.json` | TC-10 |
| FR-10 Registry query | `platform_hybrid/registry/schema.json`, `query.py`, `entries/` | TC-10 |
| FR-11 Variant deltas, stack-diff | `platform_hybrid/registry/provenance/` | TC-10 |
| FR-12 Adaptive controller | `platform_hybrid/experiments/results/p5p8/controller_cf_summary.json` | TC-4 |
| FR-13 Lane harness | `zvf-program/e1_wave10/`, `zvf-program/e5_successor27/`, `zvf-program/e2_core/`, `zvf-program/e13_balrog/`, `zvf-program/e6e9/check_quota_status.py` | TC-9 |
| FR-14 Completion gate | `zvf-program/flagship/e1_e14_completion_gate.py` | TC-9 |
| FR-15 Scope separation | `outputs/E1_E14_FINAL_RESULTS_2026-09-19.md` | TC-6, TC-9 |
| FR-16 Blind-review packaging | `platform_local/run_all_audits.py`, `export_guard_audit.py` | TC-6, TC-8 |
| NFR-1 Attribution | Method sections and comparison artefacts | TC-3, TC-10 |
| NFR-2 Reproducibility | Seed management in the runner | TC-3 |
| NFR-3 Recomputability | Documented analysis procedures | TC-1, TC-2, TC-4 |
| NFR-4 Denominator discipline | Caveat and claim-strength checks | TC-6, TC-7 |
| NFR-5 Isolation | Process-level execution | Operationally checked; not in the audit suite |
| NFR-6 Budget | Spend receipts | TC-9 |
| NFR-7 Fail-closed behaviour | Gate constants and suite exit status | TC-6, TC-9 |
| NFR-8 Least privilege | Credential-gate receipts | TC-9 |
| NFR-9 Auditability | Tracked execution paths | TC-9 |
| NFR-10 External traceability | Hash equality and trust-root validation | TC-9, TC-10 |

: Requirements traceability matrix: each functional requirement mapped to the artefacts that implement it and the audit check that verifies it.

Two rows deserve comment. FR-7 has no automated check because the white-box gradient path is a measurement procedure rather than an invariant; its evidence is the result data itself. NFR-5 is likewise checked operationally rather than by the suite, because process isolation is a property of how runs are launched and is not visible in any post-hoc artefact.

## 3.6 Hardware Requirements

| Compute surface | Requirement | Status |
|---|---|---|
| Managed training back-end (Tinker) | Sampling, training and checkpointing for the ~35B-total, ~3B-active MoE actor in bfloat16 with a pinned LoRA; per-layer gradients are not exposed, which is why FR-7 is scoped to small models | Met |
| White-box GPU | One CUDA device with at least 24 GB (an L4-class cloud notebook), enough for the 1.5B configuration and a reduced-length 3B re-test | Met |
| Serverless GPU (Modal) | Large scoring sweeps and the hosted sampler bridge; the bridge is a precondition for the lanes that use it | Met |
| Evaluation-lane cloud capacity | E9 needs a documented minimum of 4 vCPUs (8 requested) and E6 needs 16, against a live quota of 1 in each region, with increase requests open; a local E9 container build needs at least 30 GiB of free disk against 31 GiB available; E2 needs a GCP binding that only the project owner can create | Not met at 2026-09-21 |
| Orchestration workstation | Orchestration, analysis, figures, the audit suite and the report toolchain | Met |

: Hardware requirements and their status.

(sources: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md; platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex; outputs/UNBLOCK_CARRYOUT_2026-09-21.md; outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md; outputs/PES_Phase2_Review_2026-09-12/finish/e9_completion/status_2026-09-19.json)

## 3.7 Software Requirements

The runtime is Python 3.11 with PyTorch and the Hugging Face stack (Transformers, PEFT, Datasets) for model handling, tokenisation and adapters; the RL back-ends are TRL, veRL through its HybridFlow path, and OpenRLHF, driven through the shared launcher, with the Tinker client SDK for the managed path (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Experiment tracking is a weights-and-biases project; telemetry is stored as JSONL and as persisted tensors; analysis is carried out by the documented scripts named in the Phase-1 specification (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

Semester 4 adds five software components to the requirement set. First, the audit suite: `platform_local/run_all_audits.py` together with the nine audit modules it imports and the shared data model in `utils/audit_utils.py`, which defines the issue, result and suite types and the `METRIC` line format; the suite requires a Python interpreter with no third-party dependencies beyond the standard library for its structural checks (source: platform_local/run_all_audits.py; source: utils/audit_utils.py). Second, the completion gate and the per-lane harnesses under `zvf-program/`, which require Ed25519 signature verification and JSON Schema validation against the published provider-package, provider-grant and trust-root schema versions (source: zvf-program/flagship/e1_e14_completion_gate.py). Third, the registry: a schema, an entry store and a query interface, together with the provenance records under `platform_hybrid/registry/` (source: platform_hybrid/registry/query.py). Fourth, the report toolchain: a LaTeX distribution capable of the multi-pass build with BibTeX, plus a markdown assembler that concatenates the chapter files and, when available, renders the document through pandoc (source: platform_local/scientific_audit.py; source: outputs/PES_Phase2_Third_Review_2026-09-24/thesis/assemble_thesis.py). Fifth, the judgement-layer harness under `zvf-program/jev_lab/`, which records its own receipts and is used only for classification, faithfulness checking and value ordering (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md).

## 3.8 Constraints and Assumptions

**Rewards are binary and verifiable.** The specification assumes a programmatic checker — exact match for the mathematics tasks and unit tests for the code subset. Nothing in the requirement set addresses learned reward models, and none of the diagnostics should be read as applying to them (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**Held-out sets are small.** Several Phase-1 evaluation sets have $n$ between 8 and 20, and the controlled Semester-4 comparisons use 200 to 500 items (Table 1.1 gives the n behind each headline), with the consequence that learning-gain magnitudes are noise-limited by construction. This is a constraint on what the results can mean, not a defect of the analysis: the small-$n$ figures are reported with their denominators and are not used to support comparative claims (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**The managed back-end is a black box at the layer level.** Per-layer gradients are not exposed on the managed path, so claims about the relationship between reward spread and gradient magnitude are confined to the small-model white-box configuration and are directional only (source: platform_hybrid/paper/paper_P7_zvf_controller.tex).

**Group-size differences must not be attributed to sampling noise.** The comparison protocols share a deterministic prompt schedule, so that method-to-method differences in ZVF reflect training dynamics rather than independent draws of prompts (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**Framework parity is incomplete.** The requirement in FR-2 covers two seven-library rosters, but only two of the seven cross-launcher entries produced completed runs in this release, and the remaining comparison entries are dry-run placeholders. Comparative framework claims are therefore constrained to the rosters and the completed runs actually recorded, and the repository states the dry-run status explicitly (source: platform_hybrid/paper/sections/_shared_methods.tex; source: platform_hybrid/experiments/results/framework_comparison.json).

**The scale study reports descriptive limits rather than a scaling law.** The cross-scale analysis spans more than seventy runs across seven libraries and five model families, and its central result is a negative one: no reliable positive cross-scale slope is recovered and no saturation exponent is identifiable. The specification inherits that boundary — the requirement is to report identifiability limits, not to assert a law (source: platform_hybrid/paper/paper_P1_scaling.tex).

**External dependencies are assumed unavailable until proven otherwise.** Payload grants, hosted evaluations and quota increases are treated as absent until a response arrives; a lane with no provider response is closed as externally blocked with a reopen condition rather than left pending indefinitely (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md).

**Arithmetic and string lookups stay in code.** Whatever judgement layer is used for triage, classification or value ordering, verification-type checks are performed by deterministic code. This rule was banked after the judgement layer returned an incorrect answer to a parity probe in a healthy window, and it constrains the architecture: no requirement in this chapter may be verified by a language model where a deterministic check is possible (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md).

**Named variants are labels, not specifications.** The assumption that a method name identifies a method is false on the available evidence: entries sharing a label diverge substantially in measured behaviour, which is why the registry's delta records and flip-risk verdicts are specified as requirements rather than as documentation niceties (source: platform_hybrid/paper/paper_P6_registry.tex).
