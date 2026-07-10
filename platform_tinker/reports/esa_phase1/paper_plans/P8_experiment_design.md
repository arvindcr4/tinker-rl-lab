# P8 Experiment Design (post-critique) — PoT-Audit: a Proof-of-Training integrity auditor for RLVR runs

Incorporates Gemini Deep Think + GPT-5.5 Pro critiques of the P8 plan (both: **NEEDS MAJOR PIVOT, not abandon** — "scorching hot problem, wrong execution"). Raw: `p8_critique_deepthink_raw.md`, `p8_critique_gpt55pro_raw.md`.

## What both critics agreed on (the pivot)
1. **Reframe:** "RL training anomaly benchmark" → **"RLVR run-integrity / proof-of-training auditor."** Audited object = *trustworthiness of the reported training run* (matters for decentralized/outsourced RLHF & open-weights recipe claims), NOT the model trajectory. Separate explicitly from reward-hacking (EvilGenie/TRACE/METR) and from **RFT-FaultBench** (GPT's named prior-art threat — verify + differentiate + reimplement as baseline).
2. **Live code-level fault injection, NOT post-hoc telemetry splicing.** (Deep Think: "synthetic injection death warrant" — ML learns the injection artifact, not RL physics.) Actually mutate code/config and *run the training*. Post-hoc injection kept only as an auxiliary stress split.
3. **Method must be provenance-aware + RL-invariant, not off-the-shelf anomaly detection.**
   - Provenance layer (GPT): artifact/config/container hashes, **reward-replay** (recompute reward from logged completions, compare to logged), RNG lineage / cross-seed independence, algorithm-consistency checks.
   - RL-dynamics layer (Deep Think): a **physics-informed auditor** on the coupling KL ↔ ΔEntropy ↔ Advantage-variance ↔ Gradient-Utilization that should hold under the *claimed* algorithm; flag when the optimization manifold breaks.
4. **Drop trivial anomalies from the headline** (seed-collapse `assert var>0`, 0%-verifier `assert mean>0` — giveaways). Focus **stealthy**: eval contamination, reward-scale drift, surrogate-algorithm substitution.
5. **Metric = Recall @ ≤0.1% FPR** (primary) + **detection latency** (steps earlier than the reward curve reveals it). Must NOT false-alarm on **benign** shifts: LR-schedule change, checkpoint resume, batch-size change, reward-model update, hardware jitter, natural RL "aha" KL/length spikes. AUROC hides FPR — secondary only.
6. **Baselines:** Z-score/threshold rules · CUSUM/changepoint (Ruptures) · deep TS (Chronos/TimesFM zero-shot) · **VLM (GPT-4o/Gemini) reading W&B plot PNGs** (fair human-audit proxy, not LLM-on-raw-floats) · generic Isolation-Forest/Transformer MTSAD · RFT-FaultBench reimpl.

## Anomaly families (stealthy, code-level induced)
- **A. Surrogate-algorithm substitution** — claims GRPO, actually runs Dr.GRPO/PPO/AERO. *(We already have REAL per-step tensors for grpo/aero/gift in `experiments/results/n2_reward_tensor_resume/` — zero-compute substrate.)*
- **B. Reward-scale / normalization drift** — code hook rescales/renorms reward mid-run.
- **C. Eval/rollout contamination** — inject a fraction of held-out eval items into training rollouts.
- (seed-collapse & 0%-verifier retained ONLY as trivial-rule sanity checks, not headline.)

## Data we already have (real, per-step)
- `n2_reward_tensor_resume/{grpo,aero,gift}_s*_tensors.jsonl`: per step/method/seed — `rewards, lengths, advantages, zvf, frac_all_zero/one, pcd, reward_mean, cv_len, lag1_autocorr, loss`.
- `arithmetic_metrics.jsonl` (100 steps): reward/total, kl_v1/v2, entropy, frac_all_good/bad, format, correct.
- 111 `group_tensors/*.json`: per-group reward_vectors + completion_lengths + config.

## The method (2 layers → "proof-of-training")
1. **Deterministic provenance checks** (high-precision, zero false alarm): reward-replay hash mismatch → reward drift/spoofing; RNG/seed lineage → seed collapse; config/reward-fn/container digest → backend/algorithm swap declared vs actual; cross-seed independence test.
2. **Learned RL-invariant residual** (the novel core): model the algorithm-conditional invariant f(KL, ΔH, Var[A], GU) ≈ 0 that holds for a *correct* run of the claimed algorithm; the residual spikes under stealthy corruption *before* the reward curve moves. Train per-algorithm; test cross-framework.

## The ONE de-risking experiment to run FIRST (this week, cheap)
Per both critics — a **process-level transfer** test, not telemetry editing:
1. Run **3 clean GRPO** + **3 corrupted** short runs (surrogate-substitution, reward-drift, contamination) on the arithmetic/gsm8k harness; add **benign controls** (LR-change, resume, batch-size change).
2. Show **CUSUM/Z-score on the reward curve MISS or lag** the stealthy corruption (headline reward looks fine), while the **reward-replay + RL-invariant auditor catches the KL/ZVF/GU decoupling N steps earlier** and does **not** fire on benign controls.
3. **Kill criterion:** if simple rules match the auditor on *real executed* runs → pivot or abandon.

**Zero-compute head-start (today):** for family A, use the existing real `grpo` vs `aero`/`gift` tensors — test whether the algorithm-consistency invariant separates them while CUSUM-on-reward does not. If yes, family A is de-risked with no new training.

## Prior-art threat — CONFIRMED REAL
**RFT-FaultBench** = arXiv **2605.04431** (May 2026), "Towards Robust LLM Post-Training: Automatic Failure Diagnosis" — "the first benchmark for fine-grained failures in reinforcement fine-tuning: **5 fault families, 16 fault types, 779 runs**." So the "first benchmark" claim is DEAD. Read it before writing. Differentiation must be: theirs = fault **diagnosis/management** (what broke); ours = **run-integrity / provenance auditing** (is the run trustworthy / is it the algorithm it claims?), **adversarial-stealth** corruptions (surrogate substitution, contamination) they likely don't cover, and **RL-invariant early detection at ≤0.1% FPR** with cross-framework transfer. If RFT-FaultBench already covers provenance/stealth → narrow to the RL-invariant early-detection method as the contribution, or pivot P8 to a different open pillar.
