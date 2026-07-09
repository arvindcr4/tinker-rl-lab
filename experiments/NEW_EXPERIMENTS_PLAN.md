# NEW_EXPERIMENTS_PLAN — dedup against W&B inventory + launch shortlist

Date: 2026-07-04. Entity audited: `arvindcr4-pes-university` (13 projects, 809 runs total).
Inventory TSVs: `/home/claude/tinker-rl-lab/experiments/results/wandb_inventory/*.tsv`.

Dedup method: grepped all 13 TSVs for run-name families, `group_size`, algorithm, dataset,
and instrumentation signatures matching each candidate; cross-checked repo scripts
(`experiments/variance_mitigation_integration.py`, `experiments/tinker_direct_eval.py`,
`experiments/base_instruct_paired.py`, `experiments/group_size_token_normalized.py`,
`tinkerrl/grpo.py`) and the autoresearch state (`minimax_autoresearch/state/`, iters 117–137).

## Candidate table

| id | title | pillar | needs | already-run verdict + evidence | impact | verdict |
|----|-------|--------|-------|--------------------------------|--------|---------|
| A1 | PCD & LARQ vs ZVF head-to-head | P2 | re-analysis | **ALREADY RUN** — backlog itself flags `already_executed: true`; the structural validation (micro-jitter falsification: ZVF 0.158→0.000 while PCD invariant) was completed in the autoresearch line that produced iter130's PCD/LARQ findings. No new runs or analysis needed. | done | SKIP |
| A2 | Contrastive-yield re-plot of scaling null | P1 | re-analysis | NEW as analysis — no C_eff refit exists; anchors' p_x/G/KL traces live in `tinker-rl-scaling` (88 runs) + `tinker-rl-lab-world-class` scale/arch/frontier families. | high, zero cost | NEW-ANALYSIS (no launch needed) |
| A3 | Length-adversarial truncation test | P4 | sampling | NEW — zero hits for truncation/generation-cap eval families across all 822 inventory rows (no `trunc`/`cap`/max-len sweep run names anywhere). **Risk**: needs a converged Dr.GRPO checkpoint; the only Dr.GRPO project (`huggingface`: 3× `dr-grpo-qwen3-8b`) is all-crashed, so checkpoint availability must be confirmed first. | high | NEW-RUN (deferred until Dr.GRPO ckpt confirmed) |
| A4 | CLMP length-mediation on existing rollouts | P4 | re-analysis | NEW as analysis — no NDE/NIE/GER computation exists in repo results. | medium | NEW-ANALYSIS |
| A5 | BEI on matched-stack PPO/GRPO | P3/P1 | grad logging | NEW — `ppo_gsm8k_Qwen3-8B_s42` / `ppo_gsm8k_Llama-3.1-8B-Instruct_s42` exist in `tinker-rl-lab-world-class`, but no run ever logged gradient vectors (no grad-cos metrics in any config). | medium | NEW-RUN (small, later) |
| B1 | 2x2 super-group decisive test | P3 | training | NEW — no 4-arm design anywhere; no `variance-only`/`zvf-only` arm names in any project. Needs a natural large-G arm (G≥16) → violates current G≤8 launch constraint. | high | NEW-RUN (deferred) |
| B2 | Difficulty-stratified G-sweep re-analysis | P3 | re-analysis | NEW as analysis — G-sweeps exist (`campaign_w2_qwen3-8b_G2/G4/G16/G32` in world-class; `gsm8k-qwen3-8b-g4/g32/g64` in structural-ceiling) but per-prompt p_x was **not** logged, so this is blocked on N8's spectrum measurement. | high | NEW-ANALYSIS (unblocked by N8) |
| B3 | Iso-Yield dynamic grouping | P2/P3 | training | NEW — no adaptive-G run families; `group_size_token_normalized.py` exists but only static-G token-normalized cells were run (zvf-audit c0000–c03xx are fixed-G 4/8/16). Needs p_x estimates → also downstream of N8. | high | NEW-RUN (deferred) |
| C1 | Length-confounded sparse-RLVR regime | P4 | training | NEW — no length-spurious train/test split datasets or runs anywhere. Heavy regime construction. | high | NEW-RUN (deferred) |
| C2 | Preregistered curve-collapse scaling law | P1 | training | **PARTIAL** — `tinker-rl-scaling` already holds the N-ladder (Qwen3 0.6B–30B, gsm8k, G=16) with step-count/seed variants; most N×T cells exist, so first do the H(N,T) fit as re-analysis, then fill only missing cells. Full prereg version far exceeds budget. | medium | MOSTLY RE-ANALYSIS / SKIP new training for now |
| C3 | Memorization-vs-generalization ladder | P4 | sampling | NEW — `stratified_heldout.py` exists but no S0–S3 strata eval runs in any project. Eval-engineering-heavy. | medium | NEW-RUN (later) |
| N1 | Prospective CSD early-warning validation (fresh seeds, frozen thresholds) | P2 | training | NEW — zvf-audit's 368 cells are ~5-step runs without dense per-step alarm-channel logging; all CSD/`zvf_risk_max` thresholds (iter126/130) were fitted in-sample on n=5 seeds; no post-freeze runs exist. | high | NEW-RUN (next wave; 10 seeds exceeds top-2 budget) |
| N2 | **Reward-tensor-instrumented short runs (PCD/LARQ + directional ZVF)** | P2 | training | NEW — no project ever logged the per-(prompt×G) reward tensor; `variance_mitigation_integration.py` output schema is aggregate `method,seed,step,zvf,reward_mean,...` and has a `--dry-run` synthetic mode, i.e. real tensor-level GIFT/AREAL/AERO traces do not exist anywhere. | **highest** — unblocks PCD/LARQ, directional ZVF, and de-synthetizes the variance-mitigation table | **NEW-RUN — LAUNCH NOW (#1)** |
| N3 | Closed-loop alarm intervention | P2 | training | NEW — no intervention-gated runs anywhere; purely observational alarms so far. Depends on N1/N2 validating channels out-of-sample first. | high | NEW-RUN (after N1/N2) |
| N4 | Dense-eval-cadence changepoint runs | P1 | training | NEW — anchor traces in `tinker-rl-scaling` have sparse eval cadence by design; no every-step-eval runs exist. Partially satisfied for ≤8B by N2's dense per-step logging. | medium | NEW-RUN (fold ≤8B cells into N2 follow-up) |
| N5 | Direct baseline-offset c measurement | P1 | sampling | NEW — no pre-training pass-rate sampling per anchor; `tinker_direct_eval.py` ran only ad-hoc 50-problem ZVF checks, never a per-anchor c-measurement feeding the saturation refit. Cheap, sampling-only. | high per dollar | NEW-RUN (strong runner-up; N8's K=64 sampling on the 8B anchor **doubles as** its c measurement) |
| N6 | Base-vs-instruct factorial at matched N | P1 | training | **PARTIAL** — `tinker-structural-ceiling` already has a Llama class ladder: `gsm8k-llama3.1-8b-base`, `gsm8k-llama3.1-8b-ladder`, `gsm8k-llama3.1-8b-instruct-ladder`, `gsm8k-llama3.2-3b-ladder`, `gsm8k-llama3.2-1b` (+ `base_instruct_paired.py` in repo). Re-analyze that ladder first; only the Qwen matched-seed factorial cells are genuinely missing. | medium | RE-ANALYZE FIRST, fill Qwen cells later |
| N7 | Native Wu test G=2 vs G=16, off-ceiling | P3 | training | **PARTIAL** — native G2/G16 cells exist: `campaign_w2_qwen3-8b_G2` (wv5ssnmp, 9x0u2bcj) and `campaign_w2_qwen3-8b_G16` (ao38u7bu, 6pwbiixh), gsm8k, 30 steps, lr 1e-5 — but unpaired seeds and only Qwen3-8B/GSM8K; the decisive off-ceiling paired cell (Qwen2.5-1.5B/gsm8k_cot) is unrun. G=16 violates the current G≤8 launch constraint. | high | NEW-RUN (deferred; meanwhile re-analyze the 2×2 campaign_w2 repeats) |
| N8 | **Per-prompt pass-rate spectrum (K=64 rollouts) → predicted ZVF(G)** | P3 | sampling | NEW — no K-rollout-per-prompt sampling runs in any project; per-prompt p_i has never been measured, which is exactly why iter135's "causal driver" claim is untested and why B2/B3 are blocked. | **highest sampling-only** — tests the P3 mechanism analytically, unblocks B2 + B3, doubles as N5's c for the 8B anchor | **NEW-RUN — LAUNCH NOW (#2)** |
| N9 | Retention(T) law replication, 2nd model/task | P3 | training | NEW — no G-sweep-by-budget grid on any non-(Qwen3-8B/GSM8K) pair; needs multiple T budgets → exceeds 40-step constraint. | medium | NEW-RUN (deferred) |
| N10 | gsm8k_cot seed expansion n=3→8-10 | P4 | training | NEW — the GRPO/Dr.GRPO paired traces exist only at n=3 (autoresearch iters 128/132/136); no additional seed-pair runs in W&B. | medium | NEW-RUN (next wave) |
| N11 | Causal length-coupling intervention knob | P4 | training | NEW — no length-penalty/length-normalized-advantage ablation runs anywhere. | high | NEW-RUN (after N10 powers the baseline) |
| N12 | 4th risk channel (rho(dZ,dL)) in max-fusion index | P2 | re-analysis | NEW as analysis — iter130's fusion index uses 3 channels; length traces are already logged in the archived n=52 panel; zero new runs needed. | high, zero cost | NEW-ANALYSIS (do immediately, no launch) |

## Ranked shortlist of genuinely-new experiments

1. **N2 — reward-tensor-instrumented short runs** (training, launch now)
2. **N8 — per-prompt pass-rate spectrum, K=64 sampling** (sampling, launch now)
3. N12 — add length-coupling channel to the risk index (free re-analysis, run alongside)
4. A2 — contrastive-yield re-plot of the scaling null (free re-analysis)
5. N5 — baseline-offset c sampling (cheap; partially covered by N8's 8B cell)
6. A4 — CLMP length-mediation estimator validation (free re-analysis)
7. N1 — prospective EWS validation with frozen thresholds (next training wave)
8. N7 — off-ceiling native Wu paired test (needs G=16 budget exception)
9. N10 — gsm8k_cot seed expansion (powering P4)
10. B1 — 2×2 super-group decisive test (flagship P3, needs large-G arm)
11. N3 — closed-loop alarm intervention (after N1/N2)
12. B3 / C1 / N11 — design-heavy training experiments, sequenced after the above
13. A3 — truncation sweep (blocked on confirming a converged Dr.GRPO checkpoint)
14. C2 / N6 / N7-reanalysis / B2 — re-analyses over existing grids first; fill cells later

## TOP 2 TO LAUNCH NOW

### Launch 1 — N2: reward-tensor-instrumented variance-mitigation runs
- **Why**: nothing in 809 archived runs contains the per-(prompt×G) reward tensor; iter130's
  key claim (magnitude channel anti-discriminates because GIFT/AREAL/ES die at zero contrast
  while GRPO dies saturated) is untestable without it, and the current variance-mitigation
  table rests partly on `--dry-run` synthetic rows.
- **Model**: `Qwen/Qwen3.5-4B` (Tinker-available per commit 6307b8c; ≤8B). LoRA.
- **Data**: GSM8K train, fixed 512-prompt subset (≤600), seed-fixed order.
- **Arms**: 4 methods × 1 seed: `grpo`, `aero`, `gift`, `areal` (hook points already defined
  in `experiments/variance_mitigation_integration.py`); 40 steps, G=8, 16 prompts/step,
  lr 1e-5, temp 0.8.
- **Logged per step**: full G×prompts reward tensor (JSONL artifact + W&B table), ZVF,
  directional decomposition (frac all-zero vs all-one groups), PCD, LARQ, mean/CV length,
  rolling lag-1 reward autocorr (CSD channel), zvf_risk components.
- **Artifact**: `experiments/results/n2_reward_tensor/{method}_s0_tensors.jsonl` + W&B project
  `zvf-tensor-instrumented`; feeds the P2 paper's PCD/LARQ section and replaces synthetic rows.

### Launch 2 — N8: per-prompt pass-rate spectrum (sampling only)
- **Why**: iter135 declares heterogeneous prompt difficulty the CAUSAL DRIVER of the shallow
  −0.230/decade ZVF slope but p_i was never measured; this is the cheapest decisive test in
  the backlog and unblocks B2 (stratified G re-analysis) and B3 (iso-yield grouping).
- **Model**: `Qwen/Qwen3-8B` (matches the iter131 sweep benchmark; ≤8B), untouched base
  checkpoint → the same sweep doubles as N5's direct c measurement for that anchor.
- **Data**: 256 GSM8K test prompts, K=64 rollouts each (16,384 samples), temp 0.8,
  max_tokens 512 — pure Tinker sampling via an extended `experiments/tinker_direct_eval.py`
  (`--num-samples 64`).
- **Analysis**: empirical p_i histogram → analytic ZVF(G)=E[p_i^G+(1−p_i)^G] for
  G∈{2,4,8,16,32,64} → slope vs the empirical −0.230/decade; predicted retention(G) curve;
  baseline pass-rate c for the saturation refit.
- **Artifact**: `experiments/results/n8_passrate_spectrum/passrates.jsonl` + predicted-vs-
  empirical ZVF(G) plot; direct P3-paper figure ("mechanism measured, not asserted").

Constraint check (both launches): Tinker API only, no local GPU; models 4B/8B ≤ 8B;
N2 is 40 steps, G=8, 512 prompts; N8 is sampling-only. One instrumentation-training run +
one sampling sweep, as preferred.
