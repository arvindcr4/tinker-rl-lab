# Deep Dive: zvf-program — Preregistered Preflight & Verification

## Overview
This subsystem turns a frozen mathematical-design protocol into a reproducible GPU run, then independently verifies the run before any confirmatory (scientific) execution may start. The pipeline is: `preregistration.json` (status `DESIGN_FROZEN_EXECUTION_AUTHORIZED`) is hash-anchored to its authorization receipt, `verify_design.py` fail-closed checks the contract, a launcher builds a fingerprinted request and spawns a remote Spot VM/Colab via secret-free cloud-init, the VM runs a pinned TRL 1.8 GRPO trainer with a custom contrast-aware sampler and writes receipts to three independent channels (private HF repo, finished W&B run, GCS object), the launcher downloads and re-verifies those receipts, and finally `verify_preflight_matrix.py` gates the whole matrix before confirmatory execution. Every artifact carries `evidence_class: preflight-not-evidence` so a preflight can never be silently promoted into a scientific result.

## Key Components
- `zvf-program/next-submission/verify_design.py:verify_contract` — fail-closed static check of protocol, four amendments (A001-A004), authorization scope, claim ledger, results contract, manuscript blueprint, and every hash-pinned binding.
- `zvf-program/next-submission/run_preflight.py:run_unit` — Colab launcher: builds fingerprinted request, executes a 10-step `colab` CLI plan, stops the session with verified absence, then verifies receipts.
- `zvf-program/next-submission/run_gcp_preflight.py:run_unit` — GCP Spot A100 launcher: builds secret-free startup script, polls instance, downloads GCS receipts, deletes VM with absence verification.
- `zvf-program/next-submission/remote_preflight.py:main` — VM-side runner: enforces CUDA+bf16, pins Qwen3-8B revision, builds `GRPOTrainer` with the contrast rollout, emits `run_manifest.json`, uploads to HF+W&B.
- `zvf-program/next-submission/contrast_sampler.py:assemble_group` — pure contract logic for the G2→G8 early-stop sampler (no torch/TRL importable here).
- `zvf-program/next-submission/trl_sampler_adapter.py:make_rollout_func` — thin TRL adapter that injects the frozen sampler into `GRPOTrainer` via a custom rollout callable.
- `zvf-program/next-submission/verify_preflight_matrix.py:evaluate_matrix` — fail-closed gate between preflight receipts and confirmatory execution; checks every cell, seam, fingerprint, and cleanup.
- `zvf-program/next-submission/preregistration.json` + `execution_authorization.json` — the frozen protocol and the scope-restricted authorization receipt that hash-anchors it.

## Concepts & Decisions

### Frozen, hash-anchored protocol with prospective amendments
- **What**: `preregistration.json` carries status `DESIGN_FROZEN_EXECUTION_AUTHORIZED` and four amendments (A001 MATH parser, A002 Qwen non-thinking decoder, A003 confirmatory hardening, A004 seam window). Each amendment has a `path` + `sha256`; `verify_design.py` re-reads and re-hashes. The authorization receipt anchors the protocol via `canonical_protocol_sha256`, which blanks the two cross-referencing hash fields before hashing to break the circular dependency.
- **Why**: reviewer-facing evidence requires proving no design knob moved after data inspection. Amendments must declare `confirmatory_outcomes_inspected: false`, or the gate fails.
- **Trade-offs**: extremely rigid — bumping a learning rate means a new amendment + receipt + re-verification.
- **Alternatives**: a versioned config file (looser; hash drift invisible) or in-code constants (no temporal provenance).

### Two-tier fingerprints on every request
- **What**: `stack_fingerprint` (runtime packages, hardware flavor, sampler/adapter hashes, decoder contract) and `fingerprint` (the full request including seed/task/arm). Receipts must echo both back through `run_config`. `archive_incompatible_result` preserves older unit receipts under `history/` rather than overwriting.
- **Why**: stack drift is the most common reproducibility failure; isolating it from the unit identity makes burns recoverable.
- **Alternatives**: a single all-in hash (cannot tell stack from cell drift); git commit alone (cannot pin installed binaries or GPU model).

### Secret-free cloud-init on Spot VMs
- **What**: `build_entry_script` embeds only base64-encoded *source* in the startup script; secret values are fetched at boot from the GCP metadata server + Secret Manager using the VM's attached service account. The receipt uploader contains zero credential bytes.
- **Why**: startup scripts land in serial-port logs and stack traces; baking `HF_TOKEN` there is a leak waiting to happen.
- **Trade-offs**: harder local debugging (must run on a VM with the right SA); tighter IAM surface.
- **Alternatives**: env vars in `gcloud --metadata` (visible in describe output), or a sealed-secrets sidecar (overkill for one-shot Spot VMs).

### Spot VM discipline with a frozen cost cap
- **What**: `validate_args` rejects anything other than `max-run-duration=90m`, mandates `--wait`, requires `--provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE --no-restart-on-failure`, and asserts `SPOT_HOURLY_USD * 1.5 < $3.0`. Post-run, `delete_instance` must verify `instance_absent_verified=True` or the receipt is marked failed.
- **Why**: preflight is non-evidence; an unbounded Spot reclaim + retry could burn a confirmatory seed budget silently.
- **Alternatives**: reserved A100 (~3-4x cost), or on-demand with hard quota (less spot-preemption but loses the cost-cap invariant).

### Three independent receipt channels
- **What**: each run writes (1) a private HF model repo with `final/adapter_model.safetensors` + `run_manifest.json` at an exact commit SHA, (2) a W&B run that must reach state `finished`, and (3) a GCS object with `md5_hash`/`crc32c`/`generation` metadata. `validate_manifest` cross-checks all three; `recover_result_from_remote` can rebuild a dropped `NEXT_PREFLIGHT_RESULT` log marker from the HF commit + W&B state alone.
- **Why**: any single channel can fail or be tampered with; agreement across HF + W&B + GCS is far harder to fake.
- **Alternatives**: MLflow-only (single point of failure), or signed receipts with a private key (no temporal visibility for reviewers).

### Three-tier verification (design / runtime / matrix)
- **What**: `verify_design.py` (pre-launch, static, hash-pinned contract), `validate_manifest` inside `verify_remote` (post-run, every field, including `audit_record == manifest.audit_record` and runtime package versions matching the request pins exactly), and `verify_preflight_matrix.py:evaluate_matrix` (multi-cell gate requiring observed mixed-reward optimizer updates and homogeneous early-stop seams per intervention cell).
- **Why**: each tier catches a different failure class — design drift, runtime drift, and scientific-seam absence. `evaluate_matrix` returns `confirmatory_execution_gate: blocked` and exits non-zero if any seam is missing.
- **Alternatives**: a single end-to-end checker (one bug disables everything; harder to attribute the failure mode).

### Pure contract sampler decoupled from TRL
- **What**: `contrast_sampler.py` defines `Rollout`, `FixedSlots`, `assemble_group`, `aggregate_group_telemetry` as plain dataclasses with `__post_init__` invariants (binary rewards, exactly eight aligned slots, inactive slots must be unscorable EOS). The TRL adapter is the only place that imports torch.
- **Why**: the sampler *is* the scientific claim — its logic must be testable on a CPU laptop without GPU or framework. `updated_groups == mixed_groups == fractions[2] * groups` is enforced both in the dataclass and again in `validate_manifest`.
- **Alternatives**: subclassing `GRPOTrainer` directly (couples the claim to framework churn; TRL upgrades silently change behavior).

### A004 seam-verification window
- **What**: two preflight classes — `matrix_infrastructure` (1 optimizer step, 2 rollout groups) and `seam_verification` (up to 24/48 rollout groups by arm, capped under the 60-group confirmatory unit). A `StopOnFirstAppliedUpdate` `TrainerCallback` ends the run the moment a mixed-reward group's update actually lands on GPU.
- **Why**: the infrastructure window proves the VM works; the seam window proves the mixed-reward optimizer-update seam (the actual claim) executes end-to-end before committing 23 more seeds.
- **Trade-offs**: the seam window costs more GPU but is still strictly below one confirmatory unit, so it cannot contaminate the confirmatory seed budget.

## Related Code
- `platform_local.unified` — invoked by `build_unified_entry_script` for non-TRL frameworks (`verl`/`openrlhf`/`skyrl`/`tinker`); the GCP entrypoint clones the repo and runs `python -m platform_local.unified --framework <fw> --backend local`, sharing the same dispatch as the local backend.
- `zvf-program/audit/run_colab_e1_confirmatory.py` — supplies shared helpers (`atomic_json`, `run_logged`, `snapshot_sources`, `load_credentials`) imported by both launchers; the audited E1 campaign and the next-submission preflight share lifecycle primitives.
- `zvf-program/flagship/pilot/` — launch and plans trees use the same source-commit + fingerprint discipline, suggesting the pattern is the canonical spec for any GPU-tracked run in the repo.

## Start Here
1. `zvf-program/next-submission/verify_design.py` — read this first to see what the contract enforces before any GPU fires.
2. `zvf-program/next-submission/run_gcp_preflight.py` — most illustrative launcher: secret-free cloud-init, Spot discipline, receipt download, cleanup verification all in one file.
3. `zvf-program/next-submission/contrast_sampler.py` — the actual scientific kernel; ~190 lines of pure logic that every other component either fingerprint-hashes or re-derives.

---
*Generated by AntiVibe (full-repo pass) · 2026-08-02*
