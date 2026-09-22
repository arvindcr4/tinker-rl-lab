# E2 CORE-Bench launch preparation — 2026-09-20 (DRAFT, read-only, no spend)

Status: AMENDMENT_ACCEPTED_LAUNCH_PENDING (ledger:
`outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md` row E2).
No paid launch executed, no provider contacted. Paid launch is lead-only authority.

## 1. Technical-chain verification (bodies opened this session)

- Runner: `zvf-program/flagship/public_core_bench_native.py`
  (`SCHEMA = "core-bench-hard45-native-v1"`, `EXPECTED_TOTAL = 45`,
  `EXPECTED_QUESTIONS = 79`, pinned per-file SHA256 table). Present and reviewed.
- Tests: `zvf-program/flagship/test_public_core_bench_native.py` — 14 tests
  collected here (10 `def test_` incl. parametrized leak cases); this session in
  the current env: 12 passed, 2 failed on the pinned-env gate
  (`ContractError: use pinned native numpy1.26.4/scipy1.13.1 environment`).
  The "29/29 incl. E13" figure from the task brief was NOT reproduced in this
  env — the 2 failures are env-pinning, not logic, but the pinned-env green run
  still needs to be observed before launch.
- Adapter (direct-VM HARD): `finish/e2_completion/direct_vm_v12/` (contract,
  INTERFACE.md, `resource_request_v12.json`, ledger28) and `direct_vm_v13/`
  (`resource_request_v13.json`, `transport_fix.patch`, README). Present.
- Setups: `outputs/public_portfolio_2026-09-05/core_setup/prepared/
  native_ground_truth.json` contains 45 capsules (verified by count this
  session). Present.
- Surviving harness: `finish/e2_completion/native-harness/` (main.py,
  benchmark/, azure_utils/vm_manager.py). Present.
- Rebuilt driver: `zvf-program/e2_core/launch_gate.py` (33 lines) is an
  OFFLINE gate only (checks amendment record + harness files, exit 0/1). The
  full orchestration driver (bounded process-group pattern, VM lifecycle via
  vm_manager adapted to GCP, early DELETE at A+1590/H+2610) does NOT yet exist
  in a tracked path — still to rebuild per `launch_plan_2026-09-19.md`.

Update 2026-09-20 (lead session): pinned-env green run OBSERVED —
`.venv/bin/python -m pytest test_public_core_bench_native.py
test_public_balrog_native.py` → 29 passed (numpy 1.26.4 / scipy 1.13.1
installed into repo `.venv` via uv with UV_CACHE_DIR=/tmp/uv-cache). The §1
test caveat above is superseded for the pinned env.

## 2. Sealed launch request: none fresh exists; priors expired/closed

Search of `outputs/` finds no fresh sealed launch request. The record states
(decision_v19/lifecycle_decision.md): "Sealed E2 v19 is unchanged. No $4
reservation or root IAM authorization exists." Prior requests and their
terminal statuses:

- `direct_vm_v12/resource_request_v12.json` — `"status":
  "REQUEST_ONLY_NOT_ALLOCATED_NOT_LAUNCH_READY"`, ledger28 closed, instance
  independently absent.
- `direct_vm_v13/resource_request_v13.json` — `"status":
  "UNALLOCATED_NEW_ATTEMPT_REQUEST"`, `$3` hold, never allocated/launched.
- `runtime_allocation_request_v6.json` — `"status":
  "REQUESTED_NOT_ALLOCATED_NOT_LAUNCHED"`.
- `resource_allocation_request_v3.json` — `"status": "REQUESTED_NOT_ALLOCATED"`.

Exact sealed-request format (quote from `resource_request_v13.json`,
schema `E2_DIRECT_VM_INFRA_TRANSPORT_FIX_V13` — the template for the fresh request):

```json
{
  "schema": "E2_DIRECT_VM_INFRA_TRANSPORT_FIX_V13",
  "status": "UNALLOCATED_NEW_ATTEMPT_REQUEST",
  "requested_hold_usd": 3,
  "purpose": "Validate direct guest setup, prepared-copy boundary and synthetic actor/report plumbing; no native episode",
  "provider": {"project": "electric-armor-388216", "zone": "us-central1-a",
    "machine": "n1-standard-4", "vcpus": 4, "memory_gib": 15,
    "accelerator": "nvidia-tesla-t4", "accelerator_count": 1, "instances_max": 1},
  "boot": {"project": "deeplearning-platform-release",
    "name": "common-cu129-ubuntu-2204-nvidia-580-v20260909",
    "id": "3612508179781164991", "disk_gib": 50,
    "image_immutable_identity_fresh_check_required": true},
  "scratch": {"disk_gib": 64, "file_bytes_cap": 51539607552,
    "free_floor_bytes": 6442450944, "fresh_mount": "/probe",
    "prepared_copy": "/probe/task/environment",
    "guest_environment_link": "/home/crab/environment", "no_workload_images": true},
  "lease": {"absolute_seconds": 1800, "work_cutoff_seconds": 1620,
    "cleanup_seconds": 180, "independent_owned_deletion_before_create": true,
    "provider_hard_stop_before_boot": true, "retry_count": 0,
    "uncertain_dispatch": "stop and reconcile; no replay"},
  "allocation_requested": "e2-direct-vm-infra-v13",
  "prior_attempt_id": "1139631723467297695",
  "score": null
}
```
(Provider block above is verbatim from the stored v13 file.)

## 3. Draft fresh sealed launch request (for lead to review + seal; NOT sealed)

Target path (lead creates): `outputs/PES_Phase2_Review_2026-09-12/finish/
e2_completion/resource_request_v14.json`, schema
`E2_DIRECT_VM_INFRA_REQUEST_V14`, `requested_hold_usd: 3` (target) plus a
second `requested_hold_usd: 1` helper entry per the $3+$1 envelope in
`amendment_acceptance_2026-09-19.json` (`public0919-e2-target01` +
`public0919-e2-helper01` naming pattern). Draft body: copy the v13 block above
with `status: UNALLOCATED_NEW_ATTEMPT_REQUEST`,
`allocation_requested: e2-direct-vm-infra-v14`, fresh
`prior_attempt_handoff_sha256`, refreshed image-identity check, and the v13
`transport_fix.patch` headers (`User-Agent: e2-read-only-transfer/1.0`,
`Accept: */*`, `Accept-Encoding: identity`). Requires before sealing: new
ledger entry, independently refreshed cleanup absence, fresh pricing/identity
admission.

## 4. Exact preflight + launch commands (draft, lead-only; NOT executed)

```bash
# Preflight (zero spend, any agent may run):
python3 zvf-program/e2_core/launch_gate.py
python3 -m pytest zvf-program/flagship/test_public_core_bench_native.py -q  # pinned env only
python3 -c "import json;d=json.load(open('outputs/public_portfolio_2026-09-05/core_setup/prepared/native_ground_truth.json'));assert len(d)==45"

# Launch (lead-only, after §5 gate clears; each step writes a receipt first):
# 1. Seal outputs/.../e2_completion/resource_request_v14.json (§3).
# 2. Rebuild full orchestration driver under zvf-program/e2_core/ (tracked path).
# 3. Create fresh reservation public0919-e2-target01 ($3) + helper ($1) under
#    authorization_unlimited_v1.json; provider admission checks.
# 4. Run 45-capsule evaluation through finish/e2_completion/native-harness/;
#    ingest per lane evidence boundary; no pooled results.
```

## 5. Verdict

Launch-ready: NO. Single blocking item: the four-permission exact-resource GCP
IAM binding by the account owner (the HARD USER GATE in
`launch_plan_2026-09-19.md` — "NO-GO until the IAM gate clears"; decision_v19
confirms "No $4 reservation or root IAM authorization exists"). Secondary
(non-blocking, locally rebuildable): full orchestration driver rebuild +
pinned-env green test run + fresh sealed v14 request.
