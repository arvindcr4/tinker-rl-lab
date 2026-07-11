# experiments-next — Progress & Resume Notes

Last updated: 2026-07-11 (session: gameplan deployment + first real Tinker runs)

## State at a glance

| Item | Status |
|---|---|
| Tinker account | ✅ credit reloaded 2026-07-11; pool completed |
| pass@k baseline (Qwen3-8B base) | ✅ complete — `results/passk_qwen3-8b_base_test_p200_n32_s42.json` |
| Sampling pool (Qwen3-8B, GSM8K train) | ✅ **512/512 complete** (`results/pool_qwen3-8b_train_n512_r32_s42.json`); p350 kept as truncated-run record |
| E-T1 CI coverage | ✅ run on p350 pool — `results/t1_ci_coverage_..._p350.json` |
| E-T2 wasted-compute floor | ✅ run on p350 pool (G=8) — `results/t2_floor_..._p350_G8.json` |
| E-T3a G\* curve | ✅ run on p350 pool — `results/t3_gstar_..._p350.json` |
| Rollout-quality audit | ✅ p350 complete — zero-variance, advantage, clustered CI |
| pass@k bootstrap audit | ✅ base complete — problem-clustered 95% intervals |
| 403-cell sweep | ⏸ gated — cleared for shuffled batching only (see T1 verdict) |

Sampling hardening is committed through `699cab97` on `main`; nothing pushed
to remote. The rollout-quality/pass@k comparison layer is the next local change.

## Results so far (Qwen3-8B, GSM8K)

- **pass@k baseline** (test split, T=1.0, n=32, unbiased estimator):
  **pass@1 = 30.4%, pass@8 = 79.7%, pass@32 = 91.0%.**
  Problem-clustered 95% intervals are **[27.5, 33.1]%, [75.3, 83.9]%,
  [86.5, 95.0]%**, respectively.
  18/200 problems unsolved at k=32; 0/200 saturated. → Only ~9 pts of k=32
  headroom: GSM8K alone cannot demonstrate capability expansion for this
  model; add MATH-500 + a code task before E-R headline runs.
- **Rollout signal quality** (train split, p350, G=32): pass@1 30.30%
  [27.87, 32.81]%; zero-variance prompts 10.86% [7.43, 14.29]%;
  active-advantage fraction 89.14% [85.71%, 92.29%]. All zero-variance prompts
  are all-incorrect. This is measurable waste, but not the dominant bottleneck
  at the frozen base checkpoint.
- **T1 (calibration): conditional pass.** iid Wald coverage 0.92–0.97
  (passes the pre-registered [0.93, 0.97] rule at M≥64; marginal at M=32).
  Wilson is well-calibrated everywhere (0.95–0.98) → **use Wilson in the
  theory paper and the controller.** Correlated (difficulty-sorted) batching
  collapses coverage to **0.08–0.40** with *narrower* intervals →
  **clustered-variance correction is a required work item before any
  curriculum-ordered training uses the CI.**
- **T2 (floor): clean pass.** Geometric floor `G·⌈ln δ / ln ZVF⌉` holds in
  all 6 difficulty strata, tightness 1.00–1.05. Vivid datum: stratum with
  p̂≈0.01 (ZVF=0.88) needs ≥152 rollouts before one usable gradient (δ=0.1).
- **T3a (G\*): consistent, pending closed form.** Empirical
  signal-per-rollout argmax G = 2–3 per stratum; analytic Bernoulli argmax 2.
  `t3_gstar_prediction` in the JSON is null — plug the closed form from
  `theory/zvf_theory.tex` §T3 and compare.

## How to resume (in order)

```bash
cd ~/Developer/agentic_repos/tinker-rl-lab/zvf-program/experiments-next
V=../../.venv/bin/python          # repo venv (tinker SDK installed via uv)
```

1. **Unblock billing** (human step): add credit at
   https://tinker.thinkingmachines.ai/billing/balance.
   The API key lives in the repo root `.env` (`TINKER_API_KEY=tml-...`,
   gitignored; loaded automatically by the scripts). The key was pasted in a
   chat session on 2026-07-11 — rotate it if that is a concern.

2. **Complete the pool** (~162 prompts, ~20 min, ~1.5M tokens):
   ```bash
   $V build_pool.py --model Qwen/Qwen3-8B --resume
   ```
   Resume is safe: same seed ⇒ deterministic example order; the script
   validates config/sampler-path compatibility and refuses non-contiguous
   prefixes. It also now retries transient failures (`--max-retries`,
   `--retry-backoff-seconds`).

3. **Re-run the three analyses on the full 512 pool** (free, offline):
   ```bash
   POOL=results/pool_qwen3-8b_train_n512_r32_s42.json
   $V analyze_t1_ci.py    --pool $POOL
   $V analyze_t2_floor.py --pool $POOL --group-size 8
   $V analyze_t3_gstar.py --pool $POOL
   $V analyze_rollout_quality.py --pool $POOL
   ```
   The `_p350` artifacts stay as the truncated-run record; full-pool outputs
   get the untruncated tag. Expect verdicts to match (n=350 → 512 tightens
   estimates, shouldn't flip them).

4. **If a future run truncates again**: `finalize_pool.py --pool <file>
   --reason "<why>"` stamps it analyzable with explicit truncation metadata
   (min 100 prompts). Never hand-edit `status`.

## Next work queue (from gameplan.md + verdicts)

1. **Wilson CI swap in `theory/zvf_theory.tex` T1** + record the empirical
   coverage numbers; cite `t1_ci_coverage_*_p350.json`.
2. **Clustered-variance correction** (T1 under correlated batching) — blocks
   the curriculum axis of the sweep, nothing else.
3. **Fill `t3_gstar_prediction`** from the T3 closed form; if it predicts
   G≫3, the divergence is the finding.
4. **Extend `../sweep/matched_compute.py`** with `greso` and `zvf_ci_gated`
   (Wilson, M≥64) arms → launch the 403-cell sweep, shuffled batching only.
5. **MATH-500 + code-task pools/pass@k** (needs billing) — prerequisite for
   the E-R capability-expansion claim per the pass@32=91% headroom finding.
6. **Checkpoint pool** for E-T2 on drifting ZVF: rerun `build_pool.py` with
   `--sampler-path tinker://<audit-run-weights> --tag qwen3-8b-stepNN`.
7. **Matched post-RL pass@k**: use the identical seed/sampling configuration,
   then run `compare_passk_results.py`; new results carry prompt fingerprints
   and the comparison refuses unverified pairing.
8. **Three evaluation seeds** per frozen checkpoint (42/43/44), followed by
   `aggregate_seed_audits.py`. Keep this separate from training-seed evidence.

## Environment notes

- Venv: `tinker-rl-lab/.venv` (Python 3.14; `tinker`, `datasets`,
  tokenizer-only `transformers` — torch deliberately absent, not needed for
  sampling-only scripts).
- All scripts support `--dry-run` (prints plan + token estimate, contacts
  nothing). Progress checkpoints every 25 prompts into the pool JSON.
- Monitoring pattern used: launch via background Bash, then a `tail -f |
  grep --line-buffered` Monitor on milestones + error signatures
  (`402|Error|Traceback|Unauthorized|Killed`). Tinker billing 402 surfaces
  as "The job is paused due to billing status" and the SDK retries forever —
  kill the job rather than letting it spin.

## W&B logging policy (added 2026-07-11)

Every result JSON in `results/` gets mirrored to W&B by the idempotent
backfill: `platform_hybrid/experiments/tinker-runs/wandb_backfill.py`
(projects: `zvf-experiments-next` for this suite, `modal-open-stack` for the
Modal artifacts). **Re-run it after any new experiment completes** — existing
run names are skipped, so it is safe to run repeatedly. Backfilled runs are
tagged `backfill` and carry the source path; W&B created-at is backfill time,
original timing lives in the attached JSON artifact.
