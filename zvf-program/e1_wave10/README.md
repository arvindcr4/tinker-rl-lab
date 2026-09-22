# E1 wave10 v6 recovery — execution package (built offline, not executed)

Tracked-path build for the sealed E1 v6 recovery: 16 tasks (google__gson x9,
hashicorp__terraform x5, immutable-js x2) from
`outputs/PES_Phase2_Review_2026-09-12/finish/e1_recovery_v6/resource_request_v6.json`
(SHA256 `7adc6e2616f61bc7ef4d08533422a5df83ed1253f2fea507ce831890b2bb972a`).

Nothing in this directory has been executed. No modal dispatch, no network
calls, no model calls, no image builds, no git commits occurred while building
it. All commands below are for the **lead only**.

## Files

| File | Purpose |
| --- | --- |
| `bounded_procgroup.py` | The v6 process-group contract (`run_child`), standalone and importable. |
| `driver.py` | Wave10 driver: sealed-selection load, actor generation via the flagship lane boundary, native evaluation under `bounded_procgroup`. |
| `reservation_wave10.json` | Fresh reservation receipt (`public0919-e1-actor11` 8 USD/3600s, `public0919-e1-runtime11` 8 USD/10800s), chained to the unlimited-mode authorization and the 2026-09-19 root directive. Status `RESERVED_NOT_DISPATCHED`. |
| `BUILD_RECEIPT.json` | Sizes, SHA256s, `py_compile` results, spec-conformance notes for this build. |

## Launch sequence (lead only)

All commands from the repo root
(`/Users/arvind/Developer/agentic_repos/tinker-rl-lab`).

1. **Bind the reservation.** Append both allocations from
   `zvf-program/e1_wave10/reservation_wave10.json` to a new immutable
   continuation ledger (do not edit shared ledgers in place), following the
   pattern of `allocation_request02_ledger11.json` → root ledger append.

2. **Start the actor session** (runtime allocation, GPU hold
   `public0919-e1-actor11`). The sibling member e1-runtime recovers the serve
   CLI (`zvf-program/e1_wave10/recovered/public_colab_runtime_fast.py`); it was
   **not available at build time**, so the driver was coded against the actor05
   facts: OpenAI-compatible chat completions, model
   `pavlov-public-portfolio-bf16`, max_tokens 8192, temperature 0, seed 809,
   top_p 0.95, `chat_template_kwargs {"enable_thinking": false}`, endpoint
   forwarded to `http://127.0.0.1:18015/v1` (container port 8000).
   ADAPTATION POINT: if the recovered runtime's port/tunnel differs, pass
   `--endpoint`.

3. **Validate without dispatch** (no network, no cost):

   ```bash
   PYTHONPATH=zvf-program/flagship python3 zvf-program/e1_wave10/driver.py
   ```

   This verifies the sealed request hash, the embedded 16-task list, the
   preserved source contexts (`contexts_v1.json` hash + per-task
   `source_context.json`), model identity, digest-pinned images against the
   `image_inspect.json` receipts, and prints the execution plan. Exit 0 means
   validation passed; nothing was dispatched.

4. **Execute the wave** (paid; lead only, after the checklist below):

   ```bash
   PYTHONPATH=zvf-program/flagship python3 zvf-program/e1_wave10/driver.py --execute \
     --wandb-run-id <actual actor session W&B run id>
   ```

   Useful overrides: `--endpoint`, `--run-id` (default
   `e1multilingual0919wave10`; must not collide with the failed 0912 run),
   `--output-dir` (default `zvf-program/e1_wave10/run`), `--api-key`.

5. **Read the outputs.** Per-task receipts land next to the preserved contexts
   in `outputs/.../e1_completion/continuation/wave10/attempts/<iid>/`
   (`generation.json`, `generation_request.json`, `generation_response.txt`,
   `generation_http_response.json`, `generation_http_meta.json`,
   `generation_intent.json`) in the same schema the flagship
   `collect_attempts` verifier reads. Wave artifacts land in the output dir:
   `wave10.json`, `wave10.log` (`<iid> GENERATED <ms>` lines then
   `NATIVE_EXIT <rc>` or `NATIVE_UNKNOWN <reason>`), `wave10_predictions.jsonl`,
   `wave10_native_dataset.jsonl`, `wave10_native_execution.json`, and
   `native_group/group_finalization.json` (+ `unknown.json` on UNKNOWN).

6. **Post-run verification.** The native outcome is evidence, not a score: a
   suite score still requires the flagship `ingest` path with its full-300
   attempt coverage. Treat wave results as per-wave evidence in the campaign
   ledger, as waves 01-03 were.

## Safety properties

- **Sealed selection only.** The driver refuses to run unless
  `resource_request_v6.json` hashes to the pinned value AND its 16 task ids
  equal the embedded list. Contexts come from the hash-pinned
  `contexts_v1.json`; actor tasks are cross-checked against the prepared
  300-task projection when present.
- **Lane evidence boundary.** Requests are built exclusively by
  `public_swe_multilingual_native.build_actor_request`; receipts use exactly
  the `collect_attempts` schema. The driver adds no prompt fields and no
  evaluator-visible fields.
- **v6 process-group contract** (see `bounded_procgroup.py` docstring): worker
  runs in its own process group; finalization is unconditional on success AND
  failure; result JSON is read only after `killpg(pgid, 0)` reports ESRCH and
  the finalization receipt is fsync-durable. Success requires worker exit 0 +
  direct-worker reaped + original-group absent + remaining controller
  deadline. Permission/observation failures, surviving groups, or deadline
  overruns produce UNKNOWN with no replay. Nonzero workers stay UNKNOWN even
  when their group is cleaned. Scope: original owned PGID only — no claim over
  descendants that create new sessions, remote RPC cancellation, or provider
  resource deletion.
- **No fabricated failures.** Transport/HTTP errors abort the wave; they never
  become `GENERATION_FAILED` receipts (provider errors stay blocked under the
  lane boundary). `GENERATION_FAILED` is reserved for genuine completed model
  responses without a valid unified diff.
- **Immutable artifacts.** Every receipt is written write-once
  (refuses replacement, allows identical re-write). Logs are append-only.
- **No automatic retry/resume/replay** anywhere in the driver.

## What the lead must verify before dispatch

1. `python3 -m py_compile` on both `.py` files passes locally (recorded in
   `BUILD_RECEIPT.json`; re-run to confirm nothing changed since).
2. The recovered runtime serves the exact model identity
   (`pavlov-public-portfolio-bf16` over the merged Qwen3.6-35B-A3B + adapter,
   commits as in `model_identity.json`) and its endpoint answers a smoke call
   at the configured `--endpoint` before wave dispatch (2 smoke calls are
   budgeted in the actor allocation).
3. The reservation in `reservation_wave10.json` is bound in a continuation
   ledger and not expired (24h window).
4. `--run-id` does not collide with any existing
   `logs/run_evaluation/<run-id>` directory (the 0912 wave10 run id
   `e1multilingual0912wave10` is taken by the cancelled run).
5. `--wandb-run-id` matches the actual online W&B run of the live actor
   session (receipts must not record a placeholder id).
6. Adaptation points in `driver.py` (endpoint, generation params, W&B id)
   were reviewed against the recovered serve CLI flags; adjust `--endpoint`
   if the tunnel differs from 127.0.0.1:18015.
7. Disk floor and CPU image cache per the sealed request
   (`cached_images.actor = im-zlxeznMWhqWsbSBR3KkFWT`,
   `cached_images.cpu = im-pAVtHHKH6EXHe7tlbHI7kb`); no new image builds.
8. On any `NATIVE_UNKNOWN` line in `wave10.log`: stop. Do not re-run, resume,
   or replay; the disposition stays UNKNOWN pending root reconciliation.
