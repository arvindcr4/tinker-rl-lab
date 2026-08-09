# Lane E1 — SWE-bench Pro

You are lane E1 of 14 in `/Users/arvind/Developer/agentic_repos/tinker-rl-lab`.
Lanes E2–E14 are running concurrently in other sessions. Stay inside your own
files; other lanes own theirs.

**Read first, both binding:**
- `outputs/_setup/LANE_BRIEF.md` — integrity rules, cost boundary, docker mutex, deliverables
- `outputs/_setup/E1_E14_ASSET_INVENTORY.md` — verified local state as of 2026-08-09

## Suite

E1 `swe_bench_pro_eval` — Scale AI SWE-bench Pro.

## Already on disk (verified 2026-08-09)

| Asset | Path / identity |
|---|---|
| Dataset | `outputs/e1_swe_bench_pro/hf_dataset` — `ScaleAI/SWE-bench_Pro` @ `7ab5114912baf22bb098818e604c02fe7ad2c11f`, test split, 731 rows, `data/test-00000-of-00001.parquet` |
| Evaluator | `outputs/e1_swe_bench_pro/evaluator` — `scaleapi/SWE-bench_Pro-os` @ `ca10a60a5fcae51e6948ffe1485d4153d421e6c5` (MIT), with `run_scripts/`, `dockerfiles/`, `swe_bench_pro_eval.py` |
| Local runner | `zvf-program/flagship/swe_bench_pro_eval.py` |
| Local adapter | `zvf-program/flagship/pavlov_swe_bench_pro_eval_adapter.py` |
| Prior receipts | `outputs/e1_swe_bench_pro/*.json` |

Both flagship files above are yours to edit, along with their tests.

## Resolved since the last sprint

The Docker resource blocker is **gone**. Colima now runs 8 CPU / 16.54 GiB, so
the `meets_recommended_resources` check in `swe_bench_pro_eval.py` (~line 394:
`cpus >= 8 and memory_bytes >= 16 * 1024**3`) now returns `True`. Re-run the
preflight to confirm and record it.

## Remaining blockers from the prior receipt

1. Dataset declares no license — HF `cardData.license` is null. A policy gate, not a download.
2. No model patch artifact exists, so the evaluator has nothing to grade.
3. The per-instance image is not pulled:
   `jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5`
   @ `sha256:e49637ebe82a479ca43b4663525955bc9cdd58c457140ee31c20958d621d3cf7`.
   It is linux/amd64 and multi-GB; this host is arm64, so it runs under emulation and will be slow.

## Objective, in order

1. **Re-run the E1 preflight and focused tests.** Capture that the Docker gate now passes.

2. **Prove the harness executes end-to-end without a model.** The selected
   instance is `instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan`.
   The dataset row carries a gold patch; applying it and running the official
   evaluator should produce a resolved verdict.
   - This is **harness validation, not a model score**. Label it
     `harness_validation`, set `is_model_score: false`, and keep the suite
     `score` null.
   - Take the docker mutex (`mkdir outputs/_setup/docker.lock`) before pulling;
     check `df -h /` first and stop if free space is under 15 GiB. Release the
     lock with `rmdir` when done, including on failure.
   - If emulation makes this impractically slow, report it with real timings
     rather than guessing.

   This is the single most valuable thing you can deliver.

3. **Document the prediction format precisely** — the file layout and field
   names the evaluator expects for a model-generated patch, so that crossing
   the last gap is mechanical.

4. **Resolve the license question factually.** Check the HF dataset card, the
   repo, and any LICENSE at the pinned revision. Report what is actually there.
   Do not assume a license.

## Hard rules (from the lane brief — repeated because they matter)

- Never fabricate a score. Blocked is a valid outcome; a fake number is not.
- Never substitute a related benchmark. SWE-bench (classic), SWE-Gym, and the
  historical SWE-agent Pass@K tables are **not** SWE-bench Pro.
- No paid Tinker calls (`TINKER_API_KEY` is in `.env` — do not use it), no W&B
  runs, no HF pushes, no paid model API calls.
- No `git commit` / `git push` / `git checkout` / branch operations.
- Do not change Colima config, do not `docker system prune`, and do not delete
  `ghcr.io/proximal-labs/frontier-swe/revideo-perf-opt:v4` (lane E2's asset).

## Deliverables

- `outputs/e1_swe_bench_pro/lane_status_2026-08-09.md` — at most one page: what
  now runs, the exact commands that work, what remains, the single next action.
- A receipt JSON beside it: `status` ∈ `RUNNING` / `PARTIAL` / `BLOCKED`, a
  null-or-real score, evidence paths, and each blocker with the external receipt
  it needs.

Finish by reporting: status, what executed, what is still missing.
