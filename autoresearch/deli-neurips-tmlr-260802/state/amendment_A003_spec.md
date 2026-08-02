# A003_preflight_seam_window_expansion — orchestrator spec (draft for implementation)

Status: ORCHESTRATOR-APPROVED SPEC, not yet implemented or hash-bound.
Author: fable-orchestrator, 2026-08-02.

## Problem

The confirmatory execution gate requires every task-arm cell to observe the
`mixed_reward_optimizer_update` seam live. The preflight window is hash-bound to
`max_steps=1, rollout_groups=2, heldout_n=8`. Qwen3-8B (non-thinking, A002) is
accurate enough on GSM8K and on many drawn MATH-500 prompts that 2-group windows
repeatedly produce homogeneous all-correct groups:

- gsm8k/contrast: seeds 211, 223 (orphan fp 04c441b2492c), 227 — 6/6 groups all-correct.
- math500/grpo_g8: seed 223 — both prompts 8/8 correct.

At per-completion accuracy p≈0.9, P(mixed G=2 group)≈0.18; the 2-group window
observes a seam with P≈0.33 per attempt. Seed-burning is a coin-flip per A100
allocation. This is a structural constraint problem, not a tactics problem.

## Amendment (narrow scope)

For SEAM-VERIFICATION PREFLIGHTS ONLY (evidence_tier `preflight-not-evidence`):

1. Raise `rollout_groups` from 2 to 8. Keep `max_steps=1`, `heldout_n=8`,
   and every other bound parameter unchanged.
2. The run records, per group, the drawn prompt id, per-completion rewards,
   mixed/homogeneous classification, and whether the arm's optimizer-update path
   executed — same receipt schema as today.
3. Confirmatory protocol: ZERO change. Arms, G, seeds, steps, evaluation,
   power analysis, and claim rules are untouched.
4. Cap: at most 2 seam-verification attempts per cell under A003 before
   human-attention flag (deli stale rule >=4 already applies upstream).

Expected seam-observation probability per attempt: contrast arm
1-(1-2p(1-p))^8 ≈ 0.80 at p=0.9; grpo_g8 arm ≥0.86 for p≤0.97.

## Why this cannot bias results

Preflight receipts are constitutionally non-evidence (`preflight-not-evidence`
label, excluded from the results ledger and main table by results_contract.json).
The seam gate validates infrastructure (that a mixed group actually drives an
optimizer update through the live stack); widening its observation window changes
what the gate can SEE, not what confirmatory runs DO. All 0 of 64 confirmatory
rows exist, so the amendment is prospective by construction.

## Implementation contract (for work agent)

- Follow the exact file conventions of `protocol_amendment_001_math_boxed.json`
  and `protocol_amendment_002_qwen_decoder.json` (schema, prospective declaration,
  rationale, evidence enumeration with receipt fingerprints listed above).
- Bind into `preregistration.json` the same way A001/A002 are bound; recompute and
  record hashes the way the prior amendment commits did (see git log for A001/A002).
- Update `remote_preflight.py` / launcher plumbing minimally so seam-verification
  preflights use rollout_groups=8; the launchers' hash bindings in
  preregistration.json must be updated in the same commit that binds A003.
- `verify_design.py`, `verify_preflight_matrix.py`, and the amendment/preflight
  test files must PASS after the change. Existing test assertions may not be
  weakened; tests that enumerate amendments may be extended to include A003.
- Single local commit in repo style: "Amend preflight seam window before
  confirmatory runs (A003)". NO push.
