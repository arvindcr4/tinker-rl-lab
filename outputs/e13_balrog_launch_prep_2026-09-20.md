# E13 BALROG Launch Preparation — 2026-09-20 (DRAFT, NO SPEND, NO LAUNCH)

Scope: read-only verification + draft only. No provider/model/billing calls made,
no cloud launch executed. Paid launch remains lead-only authority.

## 1. Technical chain verification (bodies opened)

- Runner: `zvf-program/flagship/public_balrog_native.py` — EXISTS. Pinned header:
  `REVISION = "b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5"`, frozen `ACTOR`
  (Qwen/Qwen3.6-35B-A3B base 995ad96e… + adapter arvindcr4/… commit
  64444133…), passive native bridge, no network instantiation in module.
- Tests: `zvf-program/flagship/test_public_balrog_native.py` — EXISTS
  (pytest, imports runner as module, fixture/contract/observation helpers).
  Prior lane logs record 29/29 incl. E2 tests in pinned env
  (`e13_completion/native_protocol_tests.log` 7 pass,
  `actor_core_tests.log` 27 pass per STATUS.json verification block).
- Actor chain ($8): `outputs/PES_Phase2_Review_2026-09-12/finish/e13_completion/`
  `actor_reservation04.json` (RESERVED, id public0912-e13-actor04, 8 USD /
  3600 s, Modal) + `actor_reservation_ledger04.json` (RESERVED_NOT_DISPATCHED,
  cumulative 97.942396882) + `actor_copy_provenance.json` +
  `offline_actor_validation.json` + `host_connectivity04.json`.
- 21 receipt candidates: confirmed as `selected_episode_ids` in
  `e13_continuation/resource_request_v10.json`: open x1 (episode-09),
  putnext 00–09, pick_up_seq_go_to 00–09; 29 original started exclusions;
  denominator 255; 512 benchmark HTTP + 2 smoke attempt ceiling.
- Amendment: `e13_continuation/amendment_acceptance_2026-09-19.json` =
  ACCEPTED_IN_E13_FORM (D1 of DECISION_PACKAGE D1–D6). Design-only hosted
  supervisor at `zvf-program/e13_balrog/hosted_supervisor.py` (+ offline
  tests) — written, NOT deployed. Referenced
  `e13_continuation/launch_plan_2026-09-19.md` is MISSING from disk.
- Ledger row: `finish/Pending_Experiments.md` E13 =
  AMENDMENT_ACCEPTED_LAUNCH_PENDING; original held-out games stay externally
  blocked.

## 2. Sealed launch request — expired, no fresh request

- Prior $8 reservation EXPIRED: `actor_reservation04.json` expires_epoch
  1789298713 < now 1789882001 (2026-09-20). Connectivity proof also
  stale by design (300 s expiry, STATUS.json next_step: must remeasure).
- No fresh sealed request found under `finish/e13_*` (latest admission
  artifact remains v10, `status: PROPOSED_NOT_ALLOCATED`,
  `launch_authorized: false`,
  `readiness: EXECUTABLE_ADMISSION_BLOCKED_NO_SUPPORTED_STRICT_PROVIDER_LIFECYCLE_ROUTE`).
- Exact format (quoted from `e13_continuation/resource_request_v10.json`,
  schema `e13-consolidated-admission-request-v1array0`):

```json
{"schema": "e13-consolidated-admission-request-v10",
 "status": "PROPOSED_NOT_ALLOCATED", "launch_authorized": false,
 "selected_episodes": 21,
 "selected_episode_ids": ["babyai/BabyAI-MixedTrainLocal-v0/open/episode-09",
  "babyai/BabyAI-MixedTrainLocal-v0/putnext/episode-00", "...(putnext 01-09)",
  "...(pick_up_seq_go_to 00-09)"],
 "allocation": {"provider": "modal", "gpu": {"type": "H200", "count": 1},
  "max_usd_all_in": "8.00",
  "max_wall_seconds_from_first_provider_allocation_including_teardown": 3600,
  "max_benchmark_http_attempts_including_sdk_retries": 512,
  "max_total_generation_http_attempts": 514, "max_sessions": 1,
  "teardown_reserve_seconds": 30},
 "code_binding_sha256": "6c634ba71ea89970c4a812ad8a14bb7622afcf38e4e224ddf3fa2fb628932b3a",
 "contract_sha256": "8600ca2d02927c89…",
 "review_command": "python3 -B .codex-run/finish_20260912/e13_continuation/run_controller_v10.py --request outputs/PES_Phase2_Review_2026-09-12/finish/e13_continuation/resource_request_v10.json --contract outputs/PES_Phase2_Review_2026-09-12/finish/e13_continuation/local_contract_v3final.json"}
```

## 3. Draft fresh sealed launch request content (for lead review, NOT submitted)

New file-to-create on authorization (suggested path
`finish/e13_continuation/resource_request_v11.json`,
schema `e13-consolidated-admission-request-v11`):

- Copy v10 verbatim except: fresh `recorded_at`, new reservation id
  (actor05), fresh expiry (t0-anchored, D = t0+3600, t0 precedes first
  billable op), refreshed connectivity + offline-gate refs, re-measured
  code/contract manifest sha256, `supersedes_request_sha256` = v10 hash,
  `status` stays PROPOSED_NOT_ALLOCATED / `launch_authorized: false`
  until lead signs.
- Must additionally bind (per v10 parent_binding_required + amendment):
  separately-hosted supervisor lease/capability refs (0600, HMAC),
  authenticated account + exact-call cancellation proof, online actor
  provenance (W&B/HF/merge/source), supported_route gate update naming the
  hosted-supervisor route (allowlist currently EMPTY — requires separate
  root-reviewed change), fresh reconciled exclusive-claim namespace.
- Explicitly NOT included: 8xH200/100 USD GCP alternative (closed),
  sampler/retry/prompt/task-config changes (frozen), replayed old
  uncertain attempts.

## 4. Exact preflight + launch commands (DO NOT RUN without lead authorization)

```bash
# 0. Pinned offline gates (free, no spend)
python3 -m pytest zvf-program/flagship/test_public_balrog_native.py -q
python3 -m pytest zvf-program/e13_balrog/test_hosted_supervisor.py -q
# 1. Admission review-only gate (free, no provider calls)
python3 -B .codex-run/finish_20260912/e13_continuation/run_controller_v10.py \
  --request outputs/PES_Phase2_Review_2026-09-12/finish/e13_continuation/resource_request_v10.json \
  --contract outputs/PES_Phase2_Review_2026-09-12/finish/e13_continuation/local_contract_v3final.json
# 2. Refresh 300 s connectivity proof immediately before launch (ephemeral CPU only)
# 3. LAUNCH — LEAD ONLY: sealed v11 execute grant against amended contract; controller
#    owns admission, watchdog owns SDK cancellation, supervisor armed first with immutable D.
```

## 5. Verdict

- Launch-ready: NO.
- Single blocking item: no valid fresh sealed launch authorization exists —
  prior $8 reservation expired, v10 `launch_authorized: false` with empty
  supported_route allowlist, hosted supervisor undeployed, so paid dispatch
  is fail-closed pending lead-issued sealed v11 grant + root-reviewed gate
  change. (Secondary, same gate: native empty-content 5-retry incompatibility
  unresolved per STATUS.json — fresh stochastic attempt is unproven, not a repair.)
