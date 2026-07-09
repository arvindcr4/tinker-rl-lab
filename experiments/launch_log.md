# Experiment Launch Log

## 2026-07-04

### N12 — 4th length-coupling risk channel (rho(dZ,dL)) in max-fusion index
- **Script:** `/home/claude/tinker-rl-lab/experiments/results/N12_20260704/run_n12.py`
- **Status:** COMPLETED SUCCESSFULLY (zero-cost re-analysis, no training).
- **Inputs:** iter130 n=52 panel (`zvf_iter130_risk_index.tsv`), archived Dr.GRPO length-trace files (`drgrpo_vs_grpo.json`, `drgrpo_gsm8k_cot_full.json`), and iter136 per-run `rho(dZ,dL)` summary (`length_bias_iter136_step_coupling.tsv`).
- **Outputs:**
  - `experiments/results/N12_20260704/n12_risk_index.tsv`
  - `experiments/results/N12_20260704/n12_axis_aurocs.tsv`
  - `experiments/results/N12_20260704/n12_method_risk.tsv`
  - `experiments/results/N12_20260704/n12_real_length_traces_rho.tsv`
  - `experiments/results/N12_20260704/n12_meta.json`
  - `experiments/results/N12_20260704/figures/n12_auroc_comparison.{png,pdf}`
  - `experiments/results/N12_20260704/figures/n12_risk_scatter.{png,pdf}`
  - `experiments/results/N12_20260704/figures/n12_roc_curves.{png,pdf}`
- **Key result:** 4-channel max-fusion AUROC on the cross-experiment panel is **0.929 [0.824, 0.994]**, unchanged from the iter130 3-channel max-fusion (0.929 [0.823, 0.994]). New 4-channel weights: magnitude=0.25, CSD=0.45, drift=0.15, length-coupling=0.15. The length-coupling prior is method-level: `grpo` mean |rho(dZ,dL)|=0.138, variance-mitigation proxy (Dr.GRPO)=0.186.
- **Note:** Per-step length traces for the full n=52 panel are not archived as a single file; method-level priors were derived from the existing real Dr.GRPO length-trace panel.

### N2 — Reward-tensor instrumentation (grpo/aero/gift/areal, GSM8K, Qwen3.5-4B LoRA r32)
- **Script:** `/home/claude/tinker-rl-lab/experiments/tinker-runs/scripts/n2_reward_tensor_20260704.py`
- **PID:** 1015998 (launched via nohup ~11:30 local)
- **Log:** `/home/claude/tinker-rl-lab/experiments/tinker-runs/logs/n2_reward_tensor_20260704.out`
- **W&B:** `arvindcr4-pes-university/tinker-new-research/09zdaztp`
- **Verified status (11:33 local):** RUNNING — verified.
  - PID alive (etime ~2.5 min at check).
  - Log shows real progress: grpo arm step 1/40 completed (`reward 0.820 | zvf 0.688 | pcd 0.060`); no auth errors, no traceback.
  - Output files being written: `experiments/results/n2_reward_tensor/grpo_s0_tensors.jsonl` and `n2_metrics.tsv` (first data row present: grpo step 0, zvf 0.6875, real loss value).
  - W&B API: state=running, heartbeat 2026-07-04T09:32:44Z (UTC; fresh at time of check), created 09:30:13Z.
- **Expected duration:** ~2–4 h (4 arms x 40 steps, sequential in one process).
- **Next check:** in ~1 h, then every 1–2 h until done:
  1. `ps -p 1015998` (alive?).
  2. `tail -20 experiments/tinker-runs/logs/n2_reward_tensor_20260704.out` — step counter should advance (~40 steps/arm, arms in order grpo → aero → gift → areal) and no traceback.
  3. `wc -l experiments/results/n2_reward_tensor/n2_metrics.tsv` — should grow by ~1 row/step.
  4. W&B run 09zdaztp should remain state=running with recent heartbeat; at completion expect state=finished plus artifact `zvf-tensor-instrumented`.
  - If PID dead with state!=finished: inspect log tail for traceback; JSONL/TSV are flushed per step so partial data is usable.

### N10 — gsm8k_cot seed expansion GRPO vs Dr.GRPO (Qwen3.5-4B, 8 seeds × 15 steps pilot)
- **Script:** `/home/claude/tinker-rl-lab/experiments/tinker-runs/scripts/n10_gsm8k_cot_seed_expansion_20260704.py`
- **PID:** 1059111 (launched via nohup ~12:30 local 2026-07-04)
- **Log:** `/home/claude/tinker-rl-lab/experiments/tinker-runs/logs/n10_gsm8k_cot_seed_expansion_20260704.out`
- **W&B project:** `arvindcr4-pes-university/tinker-new-research` — one run per (algo, seed) cell, e.g. `n10_grpo_s42_20260704`.
- **Model:** `Qwen/Qwen3.5-4B` (≤8B; LoRA r16), GSM8K train, greedy held-out eval on 128 GSM8K test prompts.
- **Protocol:** sequential (1 cell at a time), 8 seeds × 2 algos (`grpo`, `dr_grpo`) × 15 steps = 240 training steps; group=8, batch=8, lr=1e-5, max_tokens=256, k_epochs=2. Constant-vs-per-length normalization distinguishes Dr.GRPO from GRPO.
- **Why N10:** it is the highest-priority feasible training experiment in the current wave — a direct seed-panel expansion of the P4 Dr.GRPO/GRPO gsm8k_cot evidence base (previously n=3 seeds). It is prerequisite to N11, which is intentionally deferred.
- **Expected duration:** ~3–6 h (polite sequential scheduling behind the already-running N2 reward-tensor run and mega cell campaign).
- **Outputs:** per-cell JSON in `experiments/results/n10_seed_expansion/n10_{algo}_s{seed}.json` plus combined `n10_manifest_20260704.json`.
- **Caveats / pilot nature:** 15 steps is a reduced pilot (original protocol used 30 steps on Modal/Qwen2.5-1.5B-Instruct); model differs from the original P4 panel. If Tinker API saturation is observed, the run can be stopped after any completed cell and the partial panel is still usable because each cell is saved independently.
- **Next check:** in ~1 h: `ps -p 1059111`; `tail -30 experiments/tinker-runs/logs/n10_gsm8k_cot_seed_expansion_20260704.out`; verify `experiments/results/n10_seed_expansion/n10_grpo_s42.json` exists and W&B run `n10_grpo_s42_20260704` is active.

### N8 — Pass-rate spectrum / analytic ZVF (Qwen3-8B base, GSM8K test, 78 prompts x K=64)
- **Script:** `/home/claude/tinker-rl-lab/experiments/tinker-runs/scripts/n8_passrate_spectrum_20260704.py`
- **PID:** 1010515 (launched 11:23 local) — no longer running (expected: run completed).
- **Log:** `/home/claude/tinker-rl-lab/experiments/tinker-runs/logs/n8_passrate_spectrum_20260704.out`
- **W&B:** `arvindcr4-pes-university/tinker-new-research/q61zvzt3`
- **Verified status (11:33 local):** COMPLETED SUCCESSFULLY (~6 min wall time, 11:23–11:29).
  - Log shows clean end-to-end finish: 78/78 prompts, 4992/4992 completions, no errors; W&B summary synced and artifacts uploaded.
  - Artifacts on disk (all written 11:29): `experiments/results/n8_passrate_spectrum/passrates.jsonl` (40 KB), `analysis.json`, `zvf_pred_vs_empirical.png`.
  - Key results: `c_baseline_pass_rate=0.7382` (this is the N5 8B-anchor baseline offset), frac_mixed=0.962, frac_all_fail=0.026, frac_all_pass=0.013, ZVF_pred(G=8)=0.296, retention_pred(G=8)=0.704, slope_per_decade=-0.426 vs empirical -0.23 (gap -0.196).
  - W&B API: state=finished, heartbeat 2026-07-04T09:29:51Z.
- **Next check:** none required — run is done. Downstream: use `analysis.json` for N5's c_baseline (8B anchor) and the ZVF-vs-G predictions; figure `zvf_pred_vs_empirical.png` is publication-input ready.

### QP batch — five quick pre-meeting experiments (Qwen3.5-4B, launched ~11:27–11:35, verified 11:38 local 2026-07-04)
- **Runs:** qp12-zvf-dense (`ku25x46q`), qp3-gsweep (`51hz4afs`), qp4-truncation (`a52g04w0`), qp7-adaptive-g (`07gr53c8`), qp8-fraud-sft (`ek1b2cxn`) — all in W&B project `arvindcr4-pes-university/tinker-new-research`; all five runs confirmed via W&B API.
- **Status at 11:38:**
  - qp12-zvf-dense — RUNNING (PID 1014075). Seed 0 done (16/16), seed 1 at 10/16; 27/32 TSV rows + group tensors. ETA ~11:41–11:43.
  - qp3-gsweep — RUNNING (PID 1015693). G=4 phase done (16/16), G=8 at 7/16; 24/32 TSV rows. ETA ~11:42–11:45.
  - qp4-truncation — **COMPLETED** cleanly at 11:31 (132 s, W&B state=finished). Finals (n=200, greedy): acc 0.025@64 / 0.005@128 / 0.005@256 / 0.125@512; mean_len==cap at every cap.
  - qp7-adaptive-g — RUNNING (PID 1014498). Arm A done (16/16, final reward 0.906); arm B at 3/16 with ladder already escalated G 4→6→8 by step 2. ETA ~11:45–11:48.
  - qp8-fraud-sft — RUNNING (PID 1018426). Train 41/63, loss 567→~12; baseline eval acc 0.80 / AUC 0.50. XGBoost comparison wrapper alive (PID 1017459), appends to `qp8_fraud.tsv`. ETA ~11:44–11:48.
- **Outputs:** TSVs + `*_manifest.json` per run in `experiments/results/quick_20260704/`; logs in `experiments/tinker-runs/logs/`. Pre-meeting summary: `experiments/results/quick_20260704/PRE_MEETING_STATUS.md`.
- **Next check (~11:50):** `ps -p 1014075 1015693 1014498 1018426`; expect all four W&B runs state=finished, qp12/qp3 TSVs at 32 rows, qp7 at 32 rows, qp8 final eval rows + xgboost row in `qp8_fraud.tsv`. If a PID is dead with W&B state!=finished, inspect the log tail for a traceback; TSVs are fsync'd per step so partial data is usable.

### A2 — Contrastive-yield re-plot of the scaling null (zero-cost re-analysis)
- **Script:** `/home/claude/tinker-rl-lab/experiments/results/A2_20260704/run_a2.py`
- **Status:** COMPLETED 2026-07-04 (no training; local-file only).
- **Outputs:**
  - Figure: `experiments/results/A2_20260704/a2_contrastive_yield_replot.png`
  - Per-anchor summary: `experiments/results/A2_20260704/a2_ceff_summary.tsv`
  - Scaling fits: `experiments/results/A2_20260704/a2_scaling_fit.tsv`
  - Metadata: `experiments/results/A2_20260704/a2_meta.json`
- **Method:** 3-parameter offset saturation fit on 5 canonical GSM8K anchors; cumulative effective contrastive compute `C_eff = Σ_t G·Y_G(p_x[t])·|loss_t|`, with `p_x` proxied by per-step mean reward and `KL_t` proxied by `|loss|` from step logs.
- **Key result:** Neither raw parameters nor contrastive-yield compute rescues a log-linear scaling law on this evidence base. `R_max ~ log10(params_B)`: slope = +0.008 ± 0.236, ρ = -0.66, p = 0.23; `R_max ~ log10(C_eff)`: slope = -1.09 ± 1.52, ρ = -0.21, p = 0.74. The scaling null persists under the frontier-proposed abscissa.

### A4 — CLMP length-mediation estimator validation (zero-cost re-analysis)
- **Script:** `/home/claude/tinker-rl-lab/experiments/results/A4_20260704/run_a4.py`
- **Status:** COMPLETED 2026-07-04 (no training; local-file only).
- **Outputs:**
  - Estimates: `experiments/results/A4_20260704/mediation_estimates.json`
  - Estimates table: `experiments/results/A4_20260704/mediation_estimates.tsv`
  - Flattened rollout records: `experiments/results/A4_20260704/mediation_records.tsv`
  - Figures: `length_by_group_size.png`, `reward_by_group_size.png`, `mediation_bars.png`, `effect_surface.png` in `experiments/results/A4_20260704/`
- **Method:** Parametric causal mediation (CLMP) on existing `qp3-gsweep` group tensors: treatment T = G=8 vs G=4, mediator M = completion length, outcome Y = binary correctness reward. Models: `length_z ~ 1 + T` and `reward ~ 1 + T + length_z + T:length_z`; counterfactuals integrated by Monte Carlo; 95% CI from 2000 non-parametric bootstrap resamples.
- **Key result:** Group-size effect is small and not significant in this paired-phase run (G=8 mean reward 0.788 vs G=4 0.765, p≈0.50). Point estimates: NDE = +0.005, NIE = +0.009, TE = +0.014, GER = 0.63 (~63% of the tiny positive G=8 effect is mediated through length). Because TE is not distinguishable from zero, GER is unstable (95% CI −5.1 to +6.6). The CLMP estimator runs cleanly on the existing rollout schema, but the data do not support a strong length-mediation claim for group size in this particular run.

### Billing block — 2026-07-04
- N2 (PID 1015998), N10 (PID 1059111), and mega campaign (PID 1059558) were stopped after Tinker API returned 402 billing-block errors.
- Re-analyses N12/A2/A4 completed before the block.
- MiniMax Berkeley autoresearch (PID 1002298) is unaffected and continues.
- Resume Tinker work after adding payment at https://tinker-console.thinkingmachines.ai/billing/balance.
