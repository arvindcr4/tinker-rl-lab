# ZVF Program — Next-Round Experiments (Stage 1: Theory Validation)

Scripts for the experiment slate in `../../gameplan.md` (deployment log) —
the pre-sweep theory validations (E-T1, E-T2, E-T3a) plus the item-8 pass@k
evaluator shared by the later training experiments (E-C, E-R).

## Platform decision: Tinker (with the pool trick)

**Chosen: Tinker sampling client for everything in this directory.** Rationale:

1. **Repo-native.** All existing runners (`platform_hybrid/experiments/
   tinker-runs/`) are Tinker; checkpoints from the audit runs are `tinker://`
   weight paths that `make_sampler()` consumes directly. Any other platform
   (Modal + vLLM, Colab) would need weight export plumbing before it could
   even see the mid-training checkpoints E-T2 needs.
2. **Sampling-only workloads.** E-T1/T2/T3a need zero training steps — just
   rollouts from frozen checkpoints. Tinker prices this as pure sampling; no
   GPU provisioning, no idle burn, no CUDA state to manage.
3. **One pool funds all three analyses.** `build_pool.py` performs a single
   sampling pass (default 512 prompts x 32 rollouts, ~5M output tokens per
   model). E-T1, E-T2, and E-T3a are then *offline resampling analyses* of
   that pool — they never contact an API. Re-running an analysis with
   different M/G/delta grids costs nothing.
4. **Same stack = comparable numbers.** Identical prompt template, tokenizer
   path, and reward parser as `live_zvf_probe.py` / `cell_runner.py`, so pool
   statistics compose with the existing 95 audit runs (this is our own
   MIN-REPORT-RL item-3 discipline applied to ourselves).

Fallbacks, documented for MIN-REPORT-RL item 3: **Modal + vLLM** if Tinker
credits run out (requires exporting LoRA weights; sampler numerics will
differ — do NOT mix pools across backends in one analysis). **Colab** only
for the tiny E-B bandit pilot.

Training experiments (E-T3b, E-C, E-R) do NOT live here — they are arms of
the existing sweep harness (`../sweep/`), which already shells out to the
Tinker cell runner. Extend `matched_compute.py` with the `greso` and
`zvf_ci_gated` arms when Stage 3 starts.

## Pipeline

```
build_pool.py  ──(one Tinker sampling pass per model/checkpoint)──▶  results/pool_*.json
     │
     ├─▶ analyze_t1_ci.py      E-T1  CI coverage (Wald + Wilson, iid vs correlated)
     ├─▶ analyze_t2_floor.py   E-T2  wasted-compute floor vs observed rollouts-to-mixed
     ├─▶ analyze_t3_gstar.py   E-T3a signal-per-rollout vs G, empirical argmax vs analytic
     └─▶ analyze_rollout_quality.py
                               zero-variance groups, active advantages,
                               clustered CIs, length confounding

passk_eval.py  ──(Tinker sampling pass per checkpoint)──▶  results/passk_*.json
                                pass@{1,8,32} + problem-clustered CIs,
                                base AND post-RL, pinned config
                                      │
                                      └─▶ compare_passk_results.py
                                           paired base/post-RL deltas

quality_*.json (3+ evaluation seeds) ──▶ aggregate_seed_audits.py
                                mean ± SD + seed-level bootstrap interval
```

## Run

```bash
cd zvf-program/experiments-next

# 0. plan + cost estimate, contacts nothing
python3 build_pool.py --model Qwen/Qwen3-8B --dry-run

# 1. the one paid sampling pass (~5M output tokens at defaults)
python3 build_pool.py --model Qwen/Qwen3-8B

# 2. offline analyses (free, rerunnable)
python3 analyze_t1_ci.py    --pool results/pool_qwen3-8b_train_n512_r32_s42.json
python3 analyze_t2_floor.py --pool results/pool_qwen3-8b_train_n512_r32_s42.json --group-size 8
python3 analyze_t3_gstar.py --pool results/pool_qwen3-8b_train_n512_r32_s42.json
python3 analyze_rollout_quality.py \
  --pool results/pool_qwen3-8b_train_n512_r32_s42.json

# 3. item-8 baseline for the base model (do this BEFORE any training arm)
python3 passk_eval.py --model Qwen/Qwen3-8B --problems 200

# add clustered CIs to a historical completed result without sampling
python3 passk_eval.py \
  --from-result results/passk_qwen3-8b_base_test_p200_n32_s42.json

# after sampling a post-RL checkpoint with the identical config
python3 compare_passk_results.py \
  --base results/passk_qwen3-8b_base_test_p200_n32_s42.json \
  --post results/passk_qwen3-8b_postrl_test_p200_n32_s42.json

# repeat build_pool + analyses for a mid-training checkpoint:
python3 build_pool.py --model Qwen/Qwen3-8B \
    --sampler-path "tinker://<weights-from-audit-run>" --tag qwen3-8b-step50
```

## Publication-readiness diagnostics

New pools store per-rollout token counts and per-prompt sampling latency. They
also checkpoint retry/failure events and report goodput, output-token
throughput, and MTBF when at least two failures are observed. Historical pools
without token counts remain analyzable, but their length diagnostics are
explicitly marked unavailable rather than imputed.

Collect at least three **evaluation seeds** for each frozen checkpoint:

```bash
for seed in 42 43 44; do
  python3 build_pool.py --model Qwen/Qwen3-8B --seed "$seed" \
    --tag "qwen3-8b-base-s${seed}"
  python3 analyze_rollout_quality.py \
    --pool "results/pool_qwen3-8b-base-s${seed}.json"
done

python3 aggregate_seed_audits.py \
  results/quality_pool_qwen3-8b-base-s42.json \
  results/quality_pool_qwen3-8b-base-s43.json \
  results/quality_pool_qwen3-8b-base-s44.json \
  --out results/quality_qwen3-8b-base_3seed.json
```

These vary prompt selection and stochastic sampling for one frozen checkpoint.
They are not substitutes for independently initialized **training-seed** runs.
Base and post-RL comparisons must use matching seed/config sets.

## Pre-registered decision rules

- **E-T1:** Wald coverage in [0.93, 0.97] at M ≥ 32 under iid batching →
  T1 usable as controller input. Correlated-batch under-coverage →
  clustered-variance correction required before the CI-gated controller arm
  runs. (Gates the 403-cell sweep launch.)
- **E-T2:** bound never violated at the (1−δ)-quantile across strata →
  report as validated floor; tightness ratio recorded either way.
- **E-T3a:** empirical argmax G per stratum within one grid step of the
  closed-form G\* → T3 promoted from "candidate" to "empirically supported";
  otherwise the divergence localizes the failing assumption in S(p,G).

## Provenance (same contract as ../sweep/)

- Every JSON carries `status`; analyses refuse pools with `status != complete`.
- `--dry-run` prints plans and cost estimates and contacts nothing.
- Partial pool progress is checkpointed every 25 prompts; a killed run leaves
  `status: "started"` (visible, resumable with `--resume`, never silently used).
- Resume validates model, split, prompt count, rollout count, seed, sampling
  parameters, and contiguous prompt indices before making another paid call.
- Nothing in this directory invents a run, a metric, or a "win."
