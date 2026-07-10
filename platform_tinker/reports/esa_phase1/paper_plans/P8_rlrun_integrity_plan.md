# P8 — Paper Plan: RLVR-Integrity, a benchmark for detecting infrastructure-level integrity failures in RL post-training runs

Status: DRAFT plan (2026-07-06). Converts P8 from the reviewers' VAPORWARE verdict (synthetic credit-card fraud) into an original, buildable contribution grounded in this project's own telemetry.

## 1. The reframe (bad → good)
- **From:** "LLM vs XGBoost on synthetic credit-card fraud" — reviewers (both models): *wrong domain*, self-contradictory AUC (0.7955 vs 0.9988), not RL-run anomaly detection. VAPORWARE.
- **To:** **RLVR-Integrity** — the *first labeled benchmark* for detecting **infrastructure-level integrity failures of the RL post-training run itself**, from its telemetry, plus a lightweight **in-loop** detector.

## 2. Novelty / differentiation (verified open white space)
Existing reward-hacking benchmarks operate at the **model→task output** level:
- **EvilGenie** (arXiv 2511.21654), **TRACE** (arXiv 2601.20103), **RHB** — the *model* games *task* rewards in code/agent settings.
- METR reward-hacking studies — model behaviour, not run integrity.

**Gap we own:** no labeled dataset/method detects **the training run being corrupt at the infrastructure level** — backend swaps, surrogate-algorithm substitution, checkpoint mislabels, telemetry manipulation. This is *provenance/integrity of the experiment*, orthogonal to output reward hacking. The paper must state this contrast explicitly in the intro + related work.

## 3. Anomaly taxonomy — grounded in REAL failures this project hit
(Every family below is a real integrity failure the adversarial reviews flagged in our own runs — that's the paper's credibility hook.)
1. **Backend swap** — managed-default/engine change mid-run (we observed the "17× framework gap").
2. **Surrogate-algorithm substitution** — labeled GRPO, actually PPO/variant.
3. **Base↔Instruct checkpoint mislabel** — (our own retracted base-vs-instruct bug).
4. **Eval contamination during rollout** — model self-generates/encounters held-out items.
5. **Reward-scale / normalization drift** — silent change in reward magnitude.
6. **Seed collapse** — single-seed run reported as multi-seed (our single-seed frontier issue).
7. **Telemetry manipulation / metric spoofing** — logged reward ≠ realized reward.
8. **Verifier/parser failure** — silent 0% (our tool-use 0% case).

## 4. Dataset construction
- **Source:** our multi-framework harness (TRL / veRL / OpenRLHF / Tinker) emits real per-step telemetry: reward, ZVF, GU, entropy, completion length, KL. We already have 70+ runs of it.
- **Method:** take clean runs, **programmatically inject** each anomaly family with controlled ground truth → labels + severity + injection step.
- **Splits:** by run (no leakage), stratified by family; public train/val/test.
- **Honesty:** injected anomalies ≠ fully natural; label a held-out subset of *naturally observed* failures too, and report the synthetic-injection caveat prominently.

## 5. Task + metrics (fixes every review demand)
- **Run-level** and **step-level** detection.
- **AUROC, AUPRC** (PR because anomalies are rare), **per-family recall @ fixed FPR**, **detection latency** (steps-to-detect — the in-loop value prop), **calibration (ECE)**, family-wise error analysis.

## 6. Baselines
- Rule thresholds · Isolation Forest · XGBoost on run features · sequence model (Transformer/LSTM over telemetry stream) · LLM-judge on run summary.
- **Honest headline:** a simple, well-calibrated telemetry detector beats the LLM-alone (the review noted LLM-alone was "catastrophic") — LLM is the *scribe/triage*, not the *scorer*. (This salvages the one real finding from the old P8.)

## 7. The novel method (the in-loop angle)
- **Telemetry-stream auditor** that flags integrity anomalies **during** training (sub-second), enabling *interception/kill* — vs all prior post-hoc detection. This is the core contribution.
- **Stretch (Gemini's angle):** latent-space trajectory auditing (hidden-state drift) for anomalies invisible in scalar telemetry — clearly labeled as exploratory.

## 8. Experiments
- Detection performance per family + overall; ablation (which telemetry channels matter); latency vs accuracy; cross-framework transfer (train on TRL, test on veRL); robustness to adaptive/stealthy injection.

## 9. What makes it defensible (maps to reviewer objections)
- Real RL telemetry (not credit-card) · labeled families · AUROC/AUPRC/calibration/family recall · multiple baselines · one consistent artifact (no 0.7955-vs-0.9988 contradiction) · explicit differentiation from EvilGenie/TRACE.

## 10. Minimal first experiment (this week, on the harness)
Pick 3 families (backend-swap, seed-collapse, reward-drift), inject into ~20 existing runs, train Isolation Forest + a small telemetry Transformer, report AUROC/AUPRC + latency vs a rule baseline. If the signal is clean, expand to all 8 families.
