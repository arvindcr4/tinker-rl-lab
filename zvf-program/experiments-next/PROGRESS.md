# experiments-next — Progress & Resume Notes

Last updated: 2026-07-11 (session: gameplan deployment + first real Tinker runs)

## State at a glance

| Item | Status |
|---|---|
| Tinker account | **BLOCKED — billing 402** (add credit: https://tinker.thinkingmachines.ai/billing/balance) |
| pass@k baseline (Qwen3-8B base) | ✅ complete — `results/passk_qwen3-8b_base_test_p200_n32_s42.json` |
| Sampling pool (Qwen3-8B, GSM8K train) | ⚠️ partial **350/512** prompts, billing-truncated; finalized copy at `results/pool_qwen3-8b_train_n512_r32_s42_p350.json` |
| E-T1 CI coverage | ✅ run on p350 pool — `results/t1_ci_coverage_..._p350.json` |
| E-T2 wasted-compute floor | ✅ run on p350 pool (G=8) — `results/t2_floor_..._p350_G8.json` |
| E-T3a G\* curve | ✅ run on p350 pool — `results/t3_gstar_..._p350.json` |
| 403-cell sweep | ⏸ gated — cleared for shuffled batching only (see T1 verdict) |

Committed through `d14659f` (results + tooling) on `main`; nothing pushed to remote.

## Results so far (Qwen3-8B, GSM8K)

- **pass@k baseline** (test split, T=1.0, n=32, unbiased estimator):
  **pass@1 = 30.4%, pass@8 = 79.7%, pass@32 = 91.0%.**
  18/200 problems unsolved at k=32; 0/200 saturated. → Only ~9 pts of k=32
  headroom: GSM8K alone cannot demonstrate capability expansion for this
  model; add MATH-500 + a code task before E-R headline runs.
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
