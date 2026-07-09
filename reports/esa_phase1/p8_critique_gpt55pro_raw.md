Bottom line upfront

The idea is salvageable, but the current framing is too fragile for NeurIPS/ICLR. Against EvilGenie/TRACE/RHB/METR, your distinction is mostly defensible. Against RFT-FaultBench / RFT-FM, it is in serious trouble. The paper cannot safely claim “first labeled benchmark for RL post-training run failures from telemetry” unless you narrow the claim to provenance/integrity failures: backend identity, algorithm identity, checkpoint identity, seed independence, data/eval channel integrity, and telemetry authenticity.

Right now, a hostile reviewer’s one-line rejection would be:

“This is a synthetic telemetry anomaly dataset with mostly standard detectors, and the closest prior already benchmarks RFT training failures from telemetry.”

That is fixable, but only by changing the center of gravity from anomaly detection to auditable RL-run integrity.

1. Novelty: distinct idea, unsafe claim
Is infrastructure-level RL-run integrity genuinely distinct from reward-hacking benchmarks?

Yes, conceptually. EvilGenie, TRACE, RHB, and METR-style work mostly ask whether an agent/model exploits a task/evaluator to appear successful. EvilGenie is about reward hacking in programming settings, including hardcoding tests or editing test files; RHB studies tool-using agents exploiting shortcut opportunities such as skipping verification or tampering with evaluation-relevant functions; METR discusses models modifying tests/scoring code or exploiting task environments. 
arXiv
+2
arXiv
+2

Your proposed benchmark can instead ask:

“Was the reported RLVR training run actually the run it claims to be?”

That is a different audit target. You are not detecting whether the model cheated a unit test. You are detecting whether the experiment execution is corrupted: wrong backend, wrong algorithm, wrong checkpoint, collapsed seeds, contaminated evaluator, spoofed reward logs, or broken parser.

That distinction is real.

But reviewers may still call it a relabel

They will call it a relabel if you continue to anchor the paper in “reward hacking.” TRACE is already framed as reward-hack detection via trajectories and explicitly includes categories such as tool abuse, context exploitation, and execution-environment hacks. It is synthetic but human-verified, with 517 trajectories and 54 subcategories. 
arXiv
+1

So if your pitch is “prior work detects reward hacking; we detect reward hacking in telemetry,” you lose.

The safer pitch is:

“Prior reward-hacking benchmarks audit model behavior. RLVR-Integrity audits experiment provenance and training-run integrity. It asks whether the run’s claimed algorithm, backend, checkpoint, seeds, reward channel, evaluator channel, and telemetry channel are trustworthy.”

That is much sharper.

Strongest prior-art threat

The strongest threat is not EvilGenie, TRACE, RHB, or METR. It is RFT-FaultBench / RFT-FM from Towards Robust LLM Post-Training: Automatic Failure Management for Reinforcement Fine-Tuning. That work claims a benchmark for fine-grained RFT failures with 5 fault families, 16 fault types, 779 training runs, 22,549 train-step records, and 1,457,288 trajectory-level records, plus an automatic failure-management framework. It also uses reward, KL, entropy, response length, and related training-dynamics signals. 
arXiv
+1

Even worse for your claim, RFT-FaultBench includes reward faults, policy-generation faults, optimization-dynamics faults, credit-assignment faults, and tool/environment faults; it also injects anomalies into the training process and reports that sequence baselines already do well on visible failures but struggle on harder settings. 
arXiv
+1

So your “first labeled benchmark” claim is currently dead unless rewritten as something like:

“First benchmark focused specifically on provenance and integrity violations in RLVR/RFT runs, complementing prior work on reward hacking, training-dynamics failures, and infrastructure fault tolerance.”

Secondary threats: L4 diagnoses large-scale LLM training failures using cross-job, spatial, and temporal log patterns, and RobustRL handles GPU-machine errors in RL post-training via role-aware detection/restart/reconnect. These do not kill your idea, but they make “infrastructure-level monitoring” sound less novel unless you emphasize scientific-integrity/provenance rather than generic failure diagnosis. 
arXiv
+1

2. Benchmark validity: synthetic telemetry injection is the biggest scientific weakness
Will reviewers dismiss injected anomalies as unrealistic?

Yes, unless you change the construction. “We take clean runs and programmatically inject anomalies into telemetry” sounds like you are benchmarking detectors on artifacts of your injection function, not on RL training failures.

A hostile reviewer will say:

“This benchmark does not test whether a detector can detect backend swaps, seed collapse, or contamination. It tests whether it can detect your scripted perturbations of time series.”

This is especially damaging for anomalies like backend swap, algorithm substitution, eval contamination, and telemetry spoofing. If the backend did not actually change, then “backend swap” is not a backend swap. It is a stylized distribution shift.

Are backend swaps and seed collapse trivially easy?

They may be trivially easy in the current design.

Backend swap becomes a giveaway if the injection creates a sharp change point in KL, entropy, reward, length, or gradient utilization. Seed collapse becomes trivial if multiple reported seeds produce identical or near-identical trajectories. Reward-scale drift becomes trivial if it is a smooth monotone rescaling of reward with no corresponding shift in realized task success.

You need to prove that the benchmark is not solvable by dumb tests:

“Did the mean jump?”

“Did variance suddenly shrink?”

“Are seed trajectories duplicate-correlated?”

“Does reward scale change but everything else stay the same?”

“Does the injected onset create a visible discontinuity?”

If a rule baseline solves 90% of the benchmark, the benchmark is not worthless, but its claim must become “integrity can be caught by cheap invariants,” not “we need a new telemetry auditor.”

The telemetry-spoofing case is conceptually dangerous

If telemetry itself is manipulated and the detector only sees telemetry, then perfect spoofing is information-theoretically undetectable. You cannot audit a corrupted channel using only that same corrupted channel.

For telemetry manipulation, you need an independent channel:

recompute rewards from saved samples;

hash-chain telemetry records;

log signed config/checkpoint/container digests;

record raw rollouts separately from aggregate metrics;

compare trainer-side, evaluator-side, and auditor-side metrics;

sample random replay windows.

Otherwise a reviewer will correctly say: “Your detector cannot detect telemetry spoofing; it detects inconsistent spoofing.”

How to make labels credible

You need three label tiers, not one:

Tier A: process-induced labels. The anomaly is injected into the actual training process, not post-hoc telemetry. Example: actually load the wrong checkpoint, actually run PPO while the config claims GRPO, actually collapse seeds, actually contaminate rollout prompts with eval items.

Tier B: post-hoc telemetry counterfactuals. These are useful, but they must be explicitly labeled as synthetic counterfactual corruptions, not real failures.

Tier C: natural incidents. These need incident evidence: config diffs, artifact hashes, commit/container digests, replay logs, evaluator logs, and at least two human annotators or maintainers agreeing on root cause.

The benchmark should publish label provenance: for every positive label, say whether it is process-induced, telemetry-mutated, or naturally observed; what evidence proves it; what severity was intended; what severity was observed; and whether a post-hoc verifier confirmed it.

Minimum benchmark design fix

Do not inject all anomalies only into saved telemetry. For the core benchmark, inject at least the following into the live harness:

wrong checkpoint loaded but correct label reported;

PPO/variant run but GRPO claimed;

seed collapse through actual seed reuse or broken seed propagation;

evaluator/parser failure through actual verifier bug;

reward normalization drift inside reward computation;

contamination through actual rollout/eval overlap.

Then keep post-hoc telemetry mutations as a separate “stress test” split.

3. Method: currently looks like off-the-shelf anomaly detection
Is “in-loop telemetry-stream detection” a real contribution?

Not yet. “Sub-second online detector over reward/KL/entropy/length/GU/ZVF” sounds like standard multivariate time-series anomaly detection with a domain-specific feature schema.

That is not a NeurIPS/ICLR method contribution by itself.

The method becomes publishable only if it exploits RLVR-specific integrity invariants that generic anomaly detectors do not know.

What would make the method genuinely novel?

You need to move from “anomaly detector” to integrity auditor.

Strong method directions:

Claim-vs-observation consistency checks.
The detector should test whether observed telemetry is consistent with the claimed algorithm, backend, checkpoint, seed plan, reward normalization, and evaluator channel. For example: do GRPO-style grouped reward/advantage statistics match the claimed group structure? Does reward normalization imply the observed reward/KL/gradient-utilization relationship? Do reported independent seeds show statistically plausible cross-run independence?

Dual-channel reward auditing.
Randomly recompute rewards from stored rollout samples and compare logged reward against realized reward. This directly attacks metric spoofing and verifier/parser failures.

Provenance-aware telemetry.
Include artifact hashes, container digests, git commits, checkpoint fingerprints, dataset/eval hashes, RNG lineage, backend/library versions, and evaluator signatures. A pure time-series detector is weak; an integrity system should use provenance evidence.

Online change-point detection with calibrated alert budgets.
The detector should guarantee something operationally meaningful: for example, “less than one false abort per 100 clean runs while detecting 80% of severe corruption within 200 steps.”

Cross-framework invariance.
Train on TRL/veRL, test on OpenRLHF/Tinker; train on one model/task, test on another. If it generalizes across frameworks, the method is not just learning logging fingerprints.

Actionable interventions.
The in-loop contribution becomes real if the system can pause, quarantine checkpoint, trigger reward replay, compare artifact hashes, or restart evaluator. Detection alone is weaker than detection plus low-cost verification.

The “latent-space hidden-state trajectory auditing” stretch goal is probably a distraction. It is expensive, not universally available, and not clearly tied to infrastructure integrity. I would cut it unless you have a killer result showing it catches integrity violations invisible to telemetry and provenance.

4. Metrics and baselines: decent start, but missing operational and anti-leakage tests

Your listed metrics are necessary but not sufficient. AUROC and AUPRC are fine for a leaderboard, but integrity monitoring is an operations problem. A detector with good AUROC but frequent false stops is unusable.

Missing metrics

Add:

False alarms per clean run and false aborts per 1,000 training steps.

Recall at operational FPR, e.g. 1%, 0.1%, or one false alarm per N clean runs.

Detection delay after actual onset, with censored cases handled explicitly.

Onset localization error, not just run-level detection.

Severity-stratified recall, because catching cartoon failures is not impressive.

Leave-framework-out performance.

Leave-injection-mechanism-out performance.

Natural-failure performance, reported separately from synthetic.

Overhead, including wall-clock slowdown, CPU/GPU usage, storage, and telemetry bandwidth.

Correct triage/action rate, if you claim in-loop intervention.

Evasion robustness, especially for gradual drift and telemetry spoofing.

Missing baselines

You need stronger and more embarrassing baselines:

RFT-FM or at least RFT-FaultBench-style feature baselines, because that is the closest prior.

Change-point detectors: CUSUM, Page-Hinkley, BOCPD/online Bayesian change point, ADWIN, EWMA.

Multivariate time-series baselines: TranAD, Anomaly Transformer, TCN/GRU autoencoder, maybe USAD/OmniAnomaly.

Hand-written invariants: seed-correlation test, reward-recompute mismatch, config/checkpoint hash mismatch, backend/library version mismatch, group-size/advantage-statistics consistency.

L4-like cross-run/spatial/temporal log comparison if logs are included.

“Metadata-only” and “telemetry-only” baselines. This is crucial. If metadata-only wins, your learned detector is unnecessary. If telemetry-only fails on spoofing, that proves you need independent provenance channels.

Human/SRE baseline on a small subset, especially for natural failures.

Is “LLM-as-scribe-not-scorer” meaningful?

It is meaningful only if you stop presenting it as a serious detector baseline. “Our XGBoost beats an LLM reading a summary” is a strawman headline.

A better role:

“The detector produces calibrated alerts; the LLM converts evidence into an incident report for humans.”

Evaluate the LLM on triage quality, hallucination rate, evidence faithfulness, and time saved for a human reviewer. Do not make “LLM-alone loses” the paper’s main story. Reviewers will not care.

5. Killer risk and the one minimal experiment
Single most likely rejection reason

The likely rejection reason is:

“The benchmark is synthetic telemetry corruption and overlaps heavily with existing RFT failure-management work; the method is mostly standard anomaly detection.”

That combines novelty and validity into one fatal objection.

The one minimal experiment that most de-risks it

Replace the proposed “inject 3 families into ~20 existing runs” with this:

Process-level transfer experiment: create actual, live training-process corruptions for three integrity families—backend/algorithm substitution, seed collapse, and reward-normalization drift—across at least two frameworks. Train detectors on synthetic/post-hoc or one-framework data, then test on held-out process-induced corruptions in another framework, with matched benign controls and no explicit metadata leakage.

The experiment should include:

actual corrupted runs, not only modified telemetry;

clean controls with benign distribution shifts, such as LR schedule change, checkpoint resume, batch-size change, reward-model update, and hardware jitter;

detectors evaluated without giveaway fields;

rules, change-point detectors, sequence models, and provenance/invariant baselines;

recall at fixed false-alarm budget;

detection latency;

a small natural-failure set if available;

a direct comparison to RFT-FaultBench/RFT-FM framing, even if only conceptual plus baseline reimplementation.

The decisive result would be:

“A provenance-aware RLVR integrity auditor trained without target-framework examples detects real process-induced integrity failures in a held-out framework at low false-alarm rate, while generic anomaly detectors and simple rules either miss subtle cases or false-alarm on benign training changes.”

That would de-risk both “synthetic artifact” and “off-the-shelf anomaly detection.”

Verdict

Needs-major-pivot, not abandon.

The core insight is valuable: RLVR runs can be scientifically invalid even when model outputs look plausible, and telemetry/provenance auditing is underdeveloped. But the current plan is too close to “time-series anomaly detection on injected telemetry,” and the “first benchmark” claim is unsafe because RFT-FaultBench is a direct prior-art threat.

Top 3 concrete changes

Reframe from “RL training anomaly benchmark” to “RLVR run-integrity/provenance benchmark.”
Explicitly separate your work from reward-hacking benchmarks and RFT-FaultBench. The audited object is not the model trajectory; it is the trustworthiness of the reported training run.

Move core labels from post-hoc telemetry injection to live process-induced corruption.
Keep synthetic telemetry injection only as an auxiliary stress split. The main benchmark must include real wrong-checkpoint loads, real seed collapse, real reward drift, real evaluator/parser failure, real algorithm/backend substitution, and natural incidents with evidence.

Make the method provenance-aware, not just anomaly-aware.
Add artifact hashes, reward replay, evaluator-side logs, RNG lineage, config/container digests, cross-seed independence tests, and algorithm-specific consistency checks. Then show cross-framework generalization and low false-alarm online detection.