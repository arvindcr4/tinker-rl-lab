# MIN-REPORT-RL adoption pack

These are submission-ready drafts, not evidence that issues or pull requests
have been filed. External submission requires maintainer review and user
authorization.

## Shared implementation references

- Registry schema: `platform_hybrid/registry/schema.json`
- Manifest generator/verifier:
  `platform_hybrid/registry/provenance/minreport.py`
- Registry query and stackdiff: `platform_hybrid/registry/query.py`
- Author checklist: `zvf-program/position/CHECKLIST.md`
- Controlled-audit contract: `zvf-program/audit/preregistration.json`
- Refusing aggregator: `zvf-program/audit/aggregate_audit.py`

## TRL issue draft

**Title:** Optional machine-readable GRPO run manifest and usable-signal telemetry

GRPO results are difficult to compare when loss form, KL placement, sampler,
precision, group-size schedule, evaluation split, and decontamination checks
are recorded inconsistently. Would maintainers accept an optional callback that
emits a versioned JSON run-start manifest plus per-step reward variance / flat-
group rate? The callback would be off by default and would not change training.

Acceptance criteria:

1. serialize the seven MIN-REPORT-RL run fields without secrets;
2. distinguish `null` (unreported) from explicit disabled values;
3. log group size and flat-group or reward-variance telemetry per step;
4. include held-out split/checkpoint-selection metadata;
5. add a deterministic serialization test and schema-version field.

Reference schema and stdlib verifier are linked above. I can adapt field names
to TRL conventions before opening a PR.

## verl issue draft

**Title:** Export a versioned actor/rollout manifest for reproducible GRPO comparisons

verl already exposes many relevant actor and rollout settings, but a compact
machine-readable artifact would make cross-run comparison safer. Proposed
scope: an opt-in run-start JSON manifest covering loss/normalization, reference
policy and KL, rollout backend/precision, group-size schedule, usable-signal
telemetry, held-out protocol, and decontamination metadata.

Acceptance criteria are the same five items above, with actor and rollout
versions/hashes recorded separately where applicable. This is observability
only; it must not alter execution or add a synchronization point.

## OpenRLHF issue draft

**Title:** Emit an opt-in reproducibility manifest for PPO/GRPO training

Could OpenRLHF expose a versioned JSON summary of the effective PPO/GRPO stack
at run start? The key need is to record the *effective* configuration after
defaults: ratio and clipping granularity, advantage normalization, reference/KL
placement, rollout backend and precision, group schedule, evaluation split,
and contamination/parser checks. Per-step potential/surviving advantage mass
could be an optional extension once the base manifest is stable.

Acceptance criteria are deterministic output, no behavior change, explicit
unknown values, schema versioning, and one integration test.

## Submission checklist

- Confirm each project's issue template and contribution policy.
- Replace repository-relative paths with a public immutable commit URL.
- Attach a minimal example manifest, not private run artifacts.
- Ask maintainers about field naming before proposing a broad adapter.
- Record resulting issue/PR URLs back in this file and in the paper.
