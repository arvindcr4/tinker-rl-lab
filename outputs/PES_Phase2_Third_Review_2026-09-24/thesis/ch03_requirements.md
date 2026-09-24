# 3. System Requirements Specification

This chapter states what the Tinker RL Lab system must do, what qualities it must hold while doing it, and how each requirement is verified. It inherits the specification of the Phase-1 report and extends it across the Semester-4 additions: the multi-framework benchmark rosters, the Zero-Variance Fraction (ZVF) diagnostic and the adaptive group-size controller derived from it, the minimum reporting standard and the machine-readable GRPO stack registry, and the E1–E14 held-out evaluation campaign with its fail-closed completion gate. It is written as a specification rather than as a retrospective description, but every requirement below corresponds to a component that exists in the repository, and every functional requirement is paired with the concrete test that checks it. Where a requirement is not satisfied — the cloud capacity for two evaluation lanes, the white-box gradient path at the largest model scale — the shortfall is named in §3.6 and §3.8 rather than written around.

## 3.1 Overview and Actors

The central object of the Phase-1 specification was the **run**: one configuration, one seed, one framework, one training budget, producing per-step telemetry and a persisted set of raw group reward tensors (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Semester 4 introduces two further first-class objects. The first is the **lane**: a single benchmark suite evaluated under a named scope — an original contract with the suite's owner, or a declared replacement scope — carrying its own provider, payload, licence, quota and terminal state. The second is the **receipt**: an immutable, hash-bound record that a declared step actually happened, against declared inputs, at a declared revision. Requirements that concern runs, lanes and receipts are distinguished below where the distinction matters.

Four actors interact with the system, and the fourth is new in Semester 4.

**The experimenter** declares work. Under the original design this meant declaring a run through a single configuration object and nothing else: the model, the dataset, the reward function, the decoding parameters (temperature, top-$p$, maximum tokens), the group size $G$, the learning rate, the KL coefficient $\beta$ and the seed all live in one file, and the framework is a switch rather than a rewrite (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Under the campaign design the same actor declares a lane through a sealed resource request: the lane specification fixes an immutable task bundle, a pinned native runtime, a native grader, a budget and an applicable licence in advance, and those declared inputs are then checked for internal consistency before any launch is permitted (source: zvf-program/flagship/e1_e14_completion_gate.py).

**The back-end** executes the training loop and, under the managed path, is the Tinker service; under the white-box path it is a local or rented GPU. For evaluation, the corresponding actor is the **provider grader**: the benchmark owner's own native evaluator at a pinned revision. The system is permitted to record a suite score only when that evaluator produced it, or when a declared replacement harness reproduces it under a scope that is labelled as a replacement (source: zvf-program/flagship/e1_e14_completion_gate.py).

**The auditor** is a downstream script or an independent reader who recomputes published diagnostics from stored raw artefacts. In Semester 4 this role acquires force: the audit suite of §3.4 is executed as a gate with a non-zero exit status rather than as advisory tooling, and the completion gate for the campaign lanes is fail-closed by construction, in that a missing or malformed declaration stops the procedure instead of degrading it.

**The campaign operator** holds launch authority for paid work. Spend on any lane requires a bound authorization receipt issued by the operator; the system will not mint one for itself, and lanes are recorded as blocked when the receipt or the technical chain behind it is absent (source: outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md).

A fifth party, the **external access holder**, is an actor only in the negative sense: several lanes depend on a private payload, a hosted evaluation service or a quota grant that only the provider can issue. The specification requires that such a lane be recorded with an explicit terminal state and a named reopen condition rather than silently dropped, which is why the campaign's terminal-state vocabulary exists as a first-class output (source: outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md).

## 3.2 Functional Requirements

**FR-1 — Single configuration object.** One declarative file shall fully determine a run: model, dataset, reward function, decoding parameters, group size, learning rate, KL coefficient, seed and output location. No framework-specific parameters may be required from the experimenter to obtain a valid run on any supported back-end (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**FR-2 — Back-end-agnostic execution through thin adapters.** A configuration shall be executable on multiple frameworks through adapters that map the shared configuration onto each framework's own interface, while sharing the reward grader and the decoding configuration rather than re-implementing them. Two framework rosters are in scope and must not be conflated. The cross-RL-library roster contains TRL, Stable-Baselines3, CleanRL, Tianshou, PufferLib, rl_games and d3rlpy; the cross-launcher LLM-RL roster contains Tinker, TRL, SkyRL, veRL, OpenRLHF, Atropos and the Hugging Face reference launchers (source: platform_hybrid/paper/sections/_shared_methods.tex). Completed-run parity across the second roster is a requirement the system does **not** yet meet: of its seven entries, only Tinker and TRL produced completed runs in this release, and the veRL and OpenRLHF entries in the comparison artefact are dry-run placeholders (source: platform_hybrid/paper/sections/_shared_methods.tex; source: platform_hybrid/experiments/results/framework_comparison.json).

**FR-3 — Per-step signal-starvation measurement.** The training loop shall compute, at every step, the fraction of prompt-groups whose within-group reward variance is zero, together with the decomposition of that fraction into all-correct and all-wrong groups. These are the ZVF and the collapse decomposition, and they are derived quantities of the reward tensors rather than proxies (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**FR-4 — Per-step telemetry.** Each step shall emit a structured record carrying reward, ZVF, gradient utilisation, the all-correct and all-wrong collapse counts, policy entropy, completion length and KL divergence, written both to a machine-readable log and to the experiment tracker (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Semester-4 controller work extends this record with the per-step counterfactual comparison between the adaptive schedule and its matched fixed-group-size control (source: platform_hybrid/experiments/results/p5p8/controller_cf_per_step.tsv).

**FR-5 — Raw tensor persistence.** The system shall persist the raw per-group reward tensors for every run, so that any derived diagnostic can be recomputed deterministically after the fact rather than trusted from a logged value (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**FR-6 — Sweeps under a fixed budget.** The runner shall support sweeps over group size, seed and baseline-versus-intervention arms, executed under a fixed wall-clock or token budget so that arms are comparable in compute rather than in step count (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**FR-7 — White-box gradient path.** For models small enough to train on a single GPU, the system shall record per-layer LoRA gradient norms alongside the usual telemetry, so that the relationship between reward spread and gradient magnitude can be examined directly rather than inferred. This path is scoped to the small-model configurations and is not a requirement at the large managed scale, where per-layer gradients are not exposed (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**FR-8 — Per-run provenance record.** Every run shall emit a machine-readable provenance record binding the configuration, the grader or verifier version, and the rollout hashes, so that a published number can be tied to the exact inputs that produced it (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). In the campaign, the equivalent obligation is stronger: a lane declares its provider package and grant documents against published schema versions, each carrying a 64-character hash, and the declared inputs must be internally consistent and available locally before the lane is admissible (source: zvf-program/flagship/e1_e14_completion_gate.py).

**FR-9 — Minimum-reporting manifest and audit.** A run submitted as a stack-conditioned result shall carry the seven-field minimum-report manifest — loss form, reference KL, sampler backend including base-checkpoint revision and hash, telemetry, group-size schedule, held-out split, and decontamination — plus the eighth evaluation item, a held-out pass@$k$ report; the resource shall emit a completeness badge on a 0–100 scale and shall distinguish an unreported field from a field reported as absent (source: platform_hybrid/paper/paper_P6_registry.tex; source: platform_hybrid/experiments/results/p5p8/minreport_audit_summary.json).

**FR-10 — Registry query surface.** The stack registry shall be queryable: a schema plus a set of entry documents and a query interface that returns entries by stack, field or status, so that a reader can ask what a given labelled method actually did rather than what its name implies (source: platform_hybrid/registry/schema.json; source: platform_hybrid/registry/query.py).

**FR-11 — Variant-delta records and stack-diff.** For each named method variant the registry shall store an explicit delta against the GRPO reference — for example DAPO's asymmetric clipping, dynamic sampling, token-level loss, overlong-reward shaping and removed KL term; Dr. GRPO's removal of the length and standard-deviation normalisations; GSPO's sequence-level ratio — and the stack-diff procedure shall return a flip-risk verdict on the 0–5 scale when two entries differ in label but not in substance (source: platform_hybrid/paper/paper_P6_registry.tex; source: platform_hybrid/registry/provenance/).

**FR-12 — Adaptive group-size controller.** The system shall implement a controller that adjusts group size in response to the measured signal, with an escalation asymmetry, hysteresis to prevent oscillation, and a callback that diverts degenerate groups away from the fixed schedule, and it shall record the counterfactual comparison against the best static recipe at matched and at unequal rollout counts (source: platform_hybrid/paper/paper_P7_zvf_controller.tex; source: platform_hybrid/experiments/results/p5p8/controller_cf_summary.json). The controller is specified as a training-time policy, not as a promotion mechanism: the requirement is to measure the trade, not to claim a win.

**FR-13 — Lane harness with evidence classes.** Each evaluation lane shall be driven by a harness that records per-lane results against a declared evidence class — exact complete, partial exact, partial recovery, externally blocked, or local-setup-ready-provider-input-required — and shall keep the class assignment tied to the source status rather than to the operator's optimism (source: zvf-program/flagship/e1_e14_completion_gate.py).

**FR-14 — Fail-closed completion gate.** A declared lane package shall be validated, not executed, by the completion gate. A valid package proves only that the declared immutable inputs are internally consistent and available locally; accepting a package as a benchmark result is explicitly out of scope for the gate. Signature verification against provider trust roots shall be performed, and the gate shall refuse rather than warn (source: zvf-program/flagship/e1_e14_completion_gate.py).

**FR-15 — Scope-separation contract in reporting.** Original-contract and replacement-scope results shall never be pooled, no cross-suite aggregate shall be computed, and every reported accuracy shall carry its coverage denominator. This is a requirement on the outputs, not only on the analysis code (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md).

**FR-16 — Blind-review packaging with an unskippable audit guard.** The export path shall produce an anonymised review package whose build runs the audit suite, and the guard shall detect and refuse a package built with the audits skipped (source: platform_local/run_all_audits.py; source: platform_local/export_guard_audit.py).

## 3.3 Non-Functional Requirements

**NFR-1 — Attribution.** Every comparative claim shall be made with the whole stack held fixed except the factor under test, and framework, algorithm, model family and scale shall be reported as confounded where they are confounded (source: platform_hybrid/paper/sections/_shared_methods.tex).

**NFR-2 — Reproducibility.** Multi-arm claims shall rest on at least three seeds; single-seed results shall be labelled descriptive or exploratory and shall not be used to support a directional claim (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**NFR-3 — Recomputability.** Every published derived diagnostic shall be reproducible from stored raw artefacts by a documented procedure, with no dependence on a logged summary value (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**NFR-4 — Honesty and denominator discipline.** Held-out figures shall be reported with their denominator $n$; negative and null results shall be reported as such; where an analysis was underpowered, that shall be stated rather than left implicit (source: platform_hybrid/paper/main_eai_body.tex).

**NFR-5 — Isolation.** Concurrent experiments shall run in separate processes rather than threads, following a tracker thread-safety defect observed in Phase 1 (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**NFR-6 — Budget.** Managed-compute spend shall be recorded per run and counted against a declared envelope; even where the standing authorization removes the cumulative cap and sets the total to unlimited, per-run resource and timeout bounds remain in force and paid launches remain bound to a lead-issued authorization receipt (source: outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md).

**NFR-7 — Fail-closed behaviour.** A validation boundary shall default to refusal. The gate neither invokes an adapter nor accepts a package as a result, and a package that cannot be verified is not scored (source: zvf-program/flagship/e1_e14_completion_gate.py).

**NFR-8 — Least privilege and no credential retention.** The system shall hold no key material in its evidence records: credential checks record only liveness and scope, and the resulting receipts state explicitly that no key material is stored (source: outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md).

**NFR-9 — Auditability under lost-source conditions.** Execution sources for anything declared sealed shall live in tracked paths, so that a lost working directory cannot orphan a sealed artefact; where loss nonetheless occurs, the sealed requests and receipts shall remain sufficient to pin a faithful re-implementation (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md).

**NFR-10 — Traceability of external artefacts.** Externally supplied payloads and hosted results shall be bound by hash equality and validated against published trust roots and schema versions, and an unreported field shall be represented as a null rather than as a zero or a false value (source: zvf-program/flagship/e1_e14_completion_gate.py; source: platform_hybrid/paper/paper_P6_registry.tex).

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
| FR-7 White-box gradients | Single-GPU training path | Not automatable in the audit suite; checks in the P1 chapter |
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
| NFR-4 Honesty | Caveat and claim-strength checks | TC-6, TC-7 |
| NFR-5 Isolation | Process-level execution | Operationally checked; not in the audit suite |
| NFR-6 Budget | Spend receipts | TC-9 |
| NFR-7 Fail-closed behaviour | Gate constants and suite exit status | TC-6, TC-9 |
| NFR-8 Least privilege | Credential-gate receipts | TC-9 |
| NFR-9 Auditability | Tracked execution paths | TC-9 |
| NFR-10 External traceability | Hash equality and trust-root validation | TC-9, TC-10 |

: Requirements traceability matrix: each functional requirement mapped to the artefacts that implement it and the audit check that verifies it.

Two rows deserve comment. FR-7 has no automated check because the white-box gradient path is a measurement procedure rather than an invariant; its evidence is the result data itself. NFR-5 is likewise checked operationally rather than by the suite, because process isolation is a property of how runs are launched and is not visible in any post-hoc artefact.

## 3.6 Hardware Requirements

The system requires four compute surfaces. The **managed training back-end** executes GRPO at the scale of the campaign actor, a mixture-of-experts model of roughly 35 billion total parameters with about 3 billion active per token, evaluated in bfloat16 with a LoRA adapter at a pinned revision (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md). The requirement here is not a particular accelerator but the service contract: the back-end exposes sampling, training and checkpointing, and does **not** expose per-layer gradients, which is why FR-7 is scoped to the small-model path (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

The **white-box GPU** requirement is a single CUDA device with at least 24 GB of memory, met in Phase 1 by a cloud notebook instance with an L4-class GPU, sufficient for the 1.5-billion-parameter configuration and for a scaled re-test at 3 billion parameters on reduced sequence lengths (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). The **batch scoring and held-out evaluation** surface is a serverless GPU path used for large scoring sweeps and for the hosted sampler bridge that the tool-use lane depends on; that bridge is a required component, and its availability is a precondition for that lane rather than an assumption (source: outputs/UNBLOCK_CARRYOUT_2026-09-21.md).

The **evaluation-lane compute** is the Semester-4 addition with the sharpest unmet requirement. Two lanes are gated on cloud capacity that is currently below the required level: one needs at least eight standard on-demand vCPUs in its region against a current grant of one of eight, and the other needs at least sixteen against a current grant of one of sixteen, with quota increase requests filed and still open (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md; source: outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md). A third lane requires a local container build for which the recipe reconstruction needs at least 30 GiB of free disk against 31 GiB available, which is marginal rather than comfortable (source: outputs/UNBLOCK_CARRYOUT_2026-09-21.md). A fourth lane requires Cloud project resources whose binding must be minted by the project owner; that binding is a hard prerequisite the system cannot satisfy on its own (source: outputs/UNBLOCK_CARRYOUT_2026-09-21.md). The **orchestration workspace** is an ordinary developer workstation used for orchestration, analysis and figure generation, and carries the audit suite and the report toolchain.

## 3.7 Software Requirements

The runtime is Python 3.11 with PyTorch and the Hugging Face stack (Transformers, PEFT, Datasets) for model handling, tokenisation and adapters; the RL back-ends are TRL, veRL through its HybridFlow path, and OpenRLHF, driven through the shared launcher, with the Tinker client SDK for the managed path (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex). Experiment tracking is a weights-and-biases project; telemetry is stored as JSONL and as persisted tensors; analysis is carried out by the documented scripts named in the Phase-1 specification (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

Semester 4 adds five software components to the requirement set. First, the audit suite: `platform_local/run_all_audits.py` together with the nine audit modules it imports and the shared data model in `utils/audit_utils.py`, which defines the issue, result and suite types and the `METRIC` line format; the suite requires a Python interpreter with no third-party dependencies beyond the standard library for its structural checks (source: platform_local/run_all_audits.py; source: utils/audit_utils.py). Second, the completion gate and the per-lane harnesses under `zvf-program/`, which require Ed25519 signature verification and JSON Schema validation against the published provider-package, provider-grant and trust-root schema versions (source: zvf-program/flagship/e1_e14_completion_gate.py). Third, the registry: a schema, an entry store and a query interface, together with the provenance records under `platform_hybrid/registry/` (source: platform_hybrid/registry/query.py). Fourth, the report toolchain: a LaTeX distribution capable of the multi-pass build with BibTeX, plus a markdown assembler that concatenates the chapter files and, when available, renders the document through pandoc (source: platform_local/scientific_audit.py; source: outputs/PES_Phase2_Third_Review_2026-09-24/thesis/assemble_thesis.py). Fifth, the judgement-layer harness under `zvf-program/jev_lab/`, which records its own receipts and is used only for classification, faithfulness checking and value ordering (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md).

## 3.8 Constraints and Assumptions

**Rewards are binary and verifiable.** The specification assumes a programmatic checker — exact match for the mathematics tasks and unit tests for the code subset. Nothing in the requirement set addresses learned reward models, and none of the diagnostics should be read as applying to them (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**Held-out sets are small.** Several evaluation sets have $n$ between 8 and 20, with the consequence that learning-gain magnitudes are noise-limited by construction. This is a constraint on what the results can mean, not a defect of the analysis: the small-$n$ figures are reported with their denominators and are not used to support comparative claims (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**The managed back-end is a black box at the layer level.** Per-layer gradients are not exposed on the managed path, so claims about the relationship between reward spread and gradient magnitude are confined to the small-model white-box configuration and are directional only (source: platform_hybrid/paper/paper_P7_zvf_controller.tex).

**Group-size differences must not be attributed to sampling noise.** The comparison protocols share a deterministic prompt schedule, so that method-to-method differences in ZVF reflect training dynamics rather than independent draws of prompts (source: platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex).

**Framework parity is incomplete.** The requirement in FR-2 covers two seven-library rosters, but only two of the seven cross-launcher entries produced completed runs in this release, and the remaining comparison entries are dry-run placeholders. Comparative framework claims are therefore constrained to the rosters and the completed runs actually recorded, and the repository states the dry-run status explicitly (source: platform_hybrid/paper/sections/_shared_methods.tex; source: platform_hybrid/experiments/results/framework_comparison.json).

**The scale study reports descriptive limits rather than a scaling law.** The cross-scale analysis spans more than seventy runs across seven libraries and five model families, and its central result is a negative one: no reliable positive cross-scale slope is recovered and no saturation exponent is identifiable. The specification inherits that boundary — the requirement is to report identifiability limits, not to assert a law (source: platform_hybrid/paper/paper_P1_scaling.tex).

**External dependencies are assumed unavailable until proven otherwise.** Payload grants, hosted evaluations and quota increases are treated as absent until a response arrives; a lane with no provider response is closed as externally blocked with a reopen condition rather than left pending indefinitely (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md).

**Arithmetic and string lookups stay in code.** Whatever judgement layer is used for triage, classification or value ordering, verification-type checks are performed by deterministic code. This rule was banked after the judgement layer returned an incorrect answer to a parity probe in a healthy window, and it constrains the architecture: no requirement in this chapter may be verified by a language model where a deterministic check is possible (source: outputs/E1_E14_FINAL_RESULTS_2026-09-19.md).

**Named variants are labels, not specifications.** The assumption that a method name identifies a method is false on the available evidence: entries sharing a label diverge substantially in measured behaviour, which is why the registry's delta records and flip-risk verdicts are specified as requirements rather than as documentation niceties (source: platform_hybrid/paper/paper_P6_registry.tex).
