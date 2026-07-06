None of the four clears a SOUND bar. P7 is the only one with a real interventional scientific idea, but its controller evidence is underpowered and internally contradictory. P6 is the closest to a concrete artifact, but mostly proves a registry can pass its own schema. P5 is a reporting-position paper with some worked artifacts and internal audits, but not an externally validated standard. P8, as the claimed RL-run fraud/anomaly pillar, is simply not present in the attached text.

Line refs below are to the flattened LaTeX.

P5 — MIN-REPORT-RL

Verdict: THIN — there is a real reporting schema/worked-manifest idea, but the paper’s own evidence says the toolchain is mostly specification, the core audits are internal/closed, and the coverage validation is shallow.

Most damaging objections

Artifact exists as schema/prose/examples, not as an enforceable released standard. The paper states that the five toolchain components are “specifications,” with some behaviors already exercised and others “design targets” (P5 §“The Toolchain That Makes It Enforceable,” lines 6058–6067). It then admits the toolchain is “not a released product suite” and framework adapters remain to be built (P5 §Limitations, lines 6247–6250). The two DAPO manifests and stackdiff example demonstrate expressivity on one hand-picked label-collision case, not deployable enforcement across the field (P5 Appendix, lines 6300–6415).

Validation is dominated by internal records and closed-stack claims. The evidence base explicitly separates released benchmark artifacts from “internal program records” including the 368-run W&B audit, 12-cell Tinker head-to-head, backend-swap audit, and open-trainer reimplementation (P5 §Reading Guide, lines 623–639). The headline 17× backend flip and DAPO label flip are compelling as anecdotes, but the limitations concede these headline exhibits have not passed the full artifact-release pipeline and the closed stack cannot be independently re-executed (P5 §Limitations, lines 6229–6235).

The coverage audit undercuts the schema-validation claim. The paper reports 100% key presence in 98 manifests, but then says all twelve subfields enumerated in the standard score 0/98, making the standard “satisfiable as a shallow key set” but not yet enforceable as a structured stack record (P5 §Exhibit 6, lines 789–826). That is not evidence of a validated reporting standard; it is evidence that the current standard can be gamed by key stubs.

Novelty pressure: standardized ML documentation is not new. Model Cards already proposed structured reporting for trained models, including intended use and evaluation procedures; Datasheets did the same for datasets; the NeurIPS reproducibility program already combined code policy, reproducibility challenge, and checklist-style reporting. W&B and MLflow already track run metrics, hyperparameters/parameters, artifacts, output files, and run metadata. P5’s real novelty must therefore be the GRPO-specific stack levers, ZVF/GU telemetry, and comparability semantics—not “a checklist” or “run metadata.” 
MLflow AI Platform
+4
arXiv
+4
arXiv
+4

Highest-value fix: ship a working trl-min-report-rl emitter, grpo-stackdiff, and auditor; regenerate all 98/103 manifests with typed validated subfields, public artifact hashes, raw telemetry paths, and CI failures on missing fields; then require at least two independent external groups to reproduce the backend/label-collision audits using the standard.

P6 — GRPO-Registry

Verdict: THIN — there is a plausible JSON/CLI registry artifact, but the paper’s own counts, coverage, and drift checks show a fragile bookkeeping artifact rather than a validated scientific contribution.

Most damaging objections

The registry population is internally inconsistent. The abstract claims three variant-delta records and twelve seed stack entries (P6 Abstract, lines 6577–6601). The resource overview repeats “12 seed stack records and 3 variant-delta records” (P6 §Overview, lines 7060–7074). The population section switches to twenty stack records plus eleven variant-delta records passing validation, i.e. 31/31 (P6 §Populating the Registry, lines 7218–7231). Later, the validator section says the registry holds 35 entries: 20 stack records and 15 variant-delta records (P6 §Schema validator, lines 10006–10008). The limitations still say “all twelve stack entries” (P6 §Limitations, lines 11849–11851). For a registry paper, inconsistent cardinality is not cosmetic; it is a direct artifact-integrity failure.

The badge validates reportability, not correctness. The schema convention says null means unreported and the badge scores “reporting coverage, not configuration virtue” (P6 §Registry Schema, lines 7098–7106). The validator finds nine red-flag leaves with >50% null rate and mean MIN-REPORT coverage of only 0.3576 across the 35-entry corpus (P6 §Schema validator, lines 10071–10096). It also catches stale audit drift: delta_drgrpo had cached measured count 0 while the live entry had measured_n=3 (P6 §Schema validator, lines 10055–10069). This is useful linting, not validation that the registry correctly represents executed RL stacks.

Closed-stack self-reporting makes the registry honest-but-thin. The limitations concede that for closed stacks, the registry records what the operator exposes; null semantics and R5 make unauditability visible but “cannot manufacture the missing facts,” and the badge is not a correctness proof or quality score (P6 §Limitations, lines 11859–11864). That means the central artifact can faithfully encode ignorance without resolving it.

Novelty pressure: compared with W&B/MLflow, P6 is not novel as experiment tracking; those systems already log/search parameters, metrics, artifacts, output files, and run metadata. Its possible novelty is a GRPO-specific ontology of variant deltas and stack comparability. But the paper never proves that ontology is correct across independent implementations, only that entries pass its own JSON schema. 
Weights & Biases Documentation
+1

Highest-value fix: freeze a single versioned registry snapshot; reconcile 12/15/20/31/35-entry contradictions; attach each entry to immutable raw artifacts, commit hashes, config dumps, telemetry files, and source-paper evidence; add independently submitted external entries; and run a blind curator audit where stackdiff predictions are compared against known implementation differences.

P7 — ZVF-controller

Verdict: THIN — there is a real theory/controller attempt, but the claimed controller is not decisively validated, and the counterfactual sections contradict each other on what the controller saves or should do.

Most damaging objections

The live controller A/B is not adequate validation. The paper claims a zvf-triage callback and adaptive-G controller, and the E3 four-arm audit reports GRPO, Dr.GRPO, DAPO dynamic sampling, and adaptive-G in one open trainer (P7 Abstract, lines 12135–12162; P7 §E3, lines 12970–12994). But adaptive-G only ties Dr.GRPO on held-out gain: both are +0.575, while adaptive-G uses 186 rollouts versus Dr.GRPO’s 120 (P7 Table E3, lines 12989–12994). The authors then admit the adaptive-G tie is the weakest reading and claim only “competitive with, not superior to” the best fixed recipe (P7 §E3, lines 12999–13014). The limitations concede single task, small n, and deltas within the noise band needed for a Tier-A claim (P7 §Limitations, lines 19051–19062).

The control signal is not validated as an outcome predictor. PCD is proposed as a control-loop input after the micro-jitter falsification collapses ZVF from 0.158 to 0 while PCD remains unchanged (P7 §PCD/Jitter, lines 12844–12884). But the paper itself says PCD is not an outcome predictor and that the decisive predictive horse race still needs per-group tensors on all anchor runs (P7 §PCD, lines 12886–12894). Later it says the replacement diagnostic did not clear the stated early-window predictor bar of ρ ≥ 0.45 (P7 synthesis, lines 19018–19027). That leaves the controller with a plausible diagnostic, not a validated decision rule.

The counterfactual controller story is internally inconsistent. One section says the N2 four-method run has zero saved prompts across all methods and thresholds, because high ZVF comes from easy saturated prompts and the controller’s honest move is not to escalate (P7 §Counterfactual evaluation, lines 13047–13059). A later Bayesian section says the Bayesian controller saves 466.75 prompts and is Pareto-dominant for contrast restoration (P7 §Bayesian refinement, lines 13243–13266). Then the calibrated Pareto table says all eight controllers save zero prompts because the regime is fully saturated (P7 §Calibrated Pareto, lines 13332–13336). Much later, an empirical GU construction says bumping from G=8 to G=16 is prescribed on 159/160 cells (P7 empirical counterfactual, lines 18848–18867). These are not small presentation inconsistencies; they are incompatible interpretations of the same controller objective.

Novelty pressure: P7 cannot claim the broad idea of fixing homogeneous/zero-gradient groups. AERO already targets zero-advantage dead zones using adaptive rollout, selective rejection, and a Bayesian posterior, reporting large compute and wall-clock reductions while matching or improving GRPO metrics; NGRPO targets homogeneous incorrect groups by turning them into learning signals via advantage calibration and asymmetric clipping. P7’s defensible novelty is narrower: ZVF/PCD telemetry plus a particular adaptive group-size controller. 
arXiv
+1

Highest-value fix: run a preregistered controller trial: ≥3 task families including hard/drifting prompt sets, ≥5 seeds per arm, fixed rollout budget, released per-group tensors, and baselines GRPO/Dr.GRPO/DAPO/AERO/adaptive-G. Primary endpoint: held-out performance at fixed rollout budget. Secondary endpoints: ZVF/PCD calibration, trigger precision/recall against labeled starvation events, and cost. Delete or reconcile the contradictory N2 counterfactual sections before defense.

P8 — fraud/anomaly detection for RL runs

Verdict: VAPORWARE — as the claimed RL-run fraud/anomaly pillar, it is absent. The attached P8 is a synthetic credit-card-fraud XGBoost-vs-LLM side-probe, not an RL-run anomaly benchmark.

Most damaging objections

The claimed RL-run anomaly artifact is not in the paper. The title is “LLM vs. XGBoost in Credit-Card Fraud: Sensor and Scribe, Not Scorer” (P8 title, lines 19382–19383). The abstract is about a custom synthetic credit-card fraud dataset, XGBoost AUC, Qwen SFT AUC, and a hybrid fraud architecture (P8 Abstract, lines 19431–19457). The program context explicitly says the main program is RL, but “here” the label under test is “AI” in fraud detection (P8 Introduction, lines 19514–19518). There is no labeled RL-run anomaly dataset, no raw RL telemetry anomaly corpus, no anomaly-family labels, no train/test split over RL runs, and no AUROC/AUPRC/calibration for RL-run anomaly detection.

Even as a credit-card fraud paper, the validation is synthetic and unstable. The dataset is generated with sklearn.make_classification: 50,000 rows, 20 anonymized numeric features, 1% label noise, and 1% fraud rate; the paper admits it lacks temporal drift, verification latency, and covariate shift (P8 §Setup, lines 19547–19561). XGBoost is evaluated on a 10,000-row natural held-out split, while the LLM is evaluated on a 500-row positive-enriched held-out subset because the natural-rate set has too few positives for stable LLM AUC (P8 §Setup, lines 19565–19603). That is not a clean model comparison, much less an RL anomaly benchmark.

The paper contradicts itself on its own headline AUC. The abstract says the current reproducible quick artifacts give XGBoost AUC 0.7955 and LLM AUC 0.48268 (P8 Abstract, lines 19431–19439). Later, the calibration table says XGB-20raw reaches 0.9988 and XGB-24full reaches 0.9991 on the released 10,000-row split, then says 0.7955 is retained only for “archaeological honesty” while the iter-4 reproducible number is 0.9988 (P8 §Calibration, lines 19717–19750). The limitations then again call 0.7955 and 0.48268 the “current artifact-backed AUCs” (P8 §Limitations, lines 27218–27225). This is fatal for any empirical paper.

Most of the “LLM role” claims are not measured. The limitations state that only the scorer head-to-head carries a number of their own; sensor, scribe, and cold-start roles are grounded in external literature and regulatory text, and integrated performance is future work (P8 §Limitations, lines 27261–27266). That is an architecture memo, not a validated fraud/anomaly detection contribution.

Highest-value fix: replace P8 with an actual RL-run anomaly benchmark: raw RL logs/manifests/reward tensors from many stacks; expert-labeled anomaly families such as backend swap, surrogate algorithm, contamination/parser failure, metric spoofing, reward hacking, seed failure, and telemetry manipulation; public train/validation/test splits; rule, Isolation Forest, XGBoost, and sequence-model baselines; AUROC, AUPRC, calibration, family-wise recall, and blind inter-paper audit results.

Overall ranking by scientific merit

P7 — ZVF-controller. Most scientific content: model, diagnostic, intervention. Still THIN, not SOUND, because the controller evidence is underpowered and internally inconsistent.

P6 — GRPO-Registry. Most concrete artifact surface: schema, entries, CLI. Still THIN, because it mostly validates its own bookkeeping and has unresolved entry-count/version contradictions.

P5 — MIN-REPORT-RL. Coherent reporting thesis, but mostly a position/specification backed by internal records and shallow manifest coverage. THIN.

P8 — fraud/anomaly detection. VAPORWARE for the stated RL-run anomaly pillar; the attached paper is about synthetic credit-card fraud and does not contain the claimed RL-run benchmark.

Defense-sinking question for the weakest pillar:
“Please point to the section/table containing the labeled RL-run anomaly benchmark: raw RL telemetry, anomaly labels, train/test split, baselines, AUROC/AUPRC, calibration, and family-wise error analysis. If it is not in P8, why was a synthetic credit-card XGBoost/LLM side-probe submitted as the RL-run fraud/anomaly pillar?”