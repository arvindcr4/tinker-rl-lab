# E1–E14 progress report — submitted 2026-08-17

Campaign run date: 2026-08-16. Status: `PARTIAL`, top-level `score: null`
(schema `pavlov-modal-e1-e14-summary-v1`). E11 was excluded from this campaign.

## Headline

- All 13 non-E11 lane adapters are configured, pass preflight, and can run on Modal.
- Recorded model results exist for E1 and E4 only; E2 has a raw official-verifier observation.
- The remaining lanes are fail-closed on named external inputs (private datasets,
  licenses, verifiers, authorizations). No substitute benchmarks or invented scores.
- 465 Python tests pass across the non-E11 adapters, readiness, E1 validation,
  E2 reward schema, E4 Modal compatibility, Harbor transfer/replay, and bridge gates.

## Lane status

| Lane | Status | Detail |
|---|---|---|
| E1 | RECORDED_MODEL_RESULT | Exact one-task SWE-bench Pro run, seed 1818: score `0.0` (malformed diff failed `git apply --check` — a real failed attempt). One task, not the 731-task suite; full campaign runner and budget absent. Validation now rejects placeholder hunks/prose/fenced output pre-verifier. |
| E2 | Raw verifier evidence | Official Frontier-SWE verifier: reward `0.753697`, correctness 8/8 scenes, geomean speedup `0.7537x` (slower overall). Harbor reward-schema repair is unit-tested but the candidate has not been rerun through Harbor. |
| E3 | Adapter ready | Provider-private 80-task bundle, runtime, split, and grader unavailable. |
| E4 | RECORDED_MODEL_RESULT | Recovery-grade Harbor/Gandalf evaluation: score `0.3115`, 128/128 criteria evaluated, 37 met, 0 errors. Replays the pass-3 workbook without new sampling — recovery-grade, not clean one-pass evidence. |
| E5 | Adapter ready | Archipelago environment, judge credentials, and approved subset absent; projected 480-task Tinker cost ~USD 122 before judge calls vs ~USD 15 remaining. |
| E6 | Adapter ready | Halluminate live environment, authorization, ground truth, and held-out identity unavailable. |
| E7 | Adapter ready | Exact native-amd64 environment receipt exists; license artifact/authorization for the pinned BinaryAudit revision still missing. Bridge-bound Harbor job resolved; launch blocked at USD 0.00 bridge budget. |
| E8 | Adapter ready | Exact dataset revision, manifest, native verifier, disjointness proof, and license receipt unavailable. |
| E9 | Adapter ready | 74 of 75 Kaggle competition agreements still need account-holder acceptance; image, artifacts, disjointness proof absent. |
| E10 | Adapter ready | AISI private held-out files, approved policy grader, and judge authorization unavailable. |
| E12 | Adapter ready | Native AppBench GUI/deployment/verifier, held-out proof, and two independent human graders unavailable. |
| E13 | Adapter ready | Official held-out suite/deployment binding and `OPENREWARD_API_KEY` unavailable. |
| E14 | Adapter ready | Exact evaluation is Epoch-hosted and private; no local substitute can produce an E14 score. |

## Cost (known model/provider subtotal)

USD 1.191367735 total: USD 0.655724655 bridge/Tinker ledger + USD 0.03015018 E1
(estimated) + USD 0.5054929 E4 judge. Modal compute and Gemini billing are not
represented in the Tinker ledger.

## Verification

- Bridge health endpoint reports READY with the pinned Hugging Face commit
  `64444133c55d88c3f1bf0df8a2f5d7ac646125c8`, online W&B, USD 0.655724655 charged,
  no reserved spend.
- Final Harbor Modal app `ap-5P2htcYaicCzra10B6HiCB` is stopped.
- Five Modal/Harbor extension tests and eighteen bridge/launch-gate protocol tests
  pass in their intended runtimes.

## Next actions (unchanged blockers)

1. E1: build the 731-task campaign runner and obtain explicit campaign budget plus dataset-license receipt.
2. E2/E7: authorize a positive bridge total so the resolved Harbor jobs can launch.
3. E4: produce a defensible projected total at or below the authorized USD 1.00 ceiling.
4. E9: account holder to accept the remaining 74 Kaggle competition agreements.
5. E3/E5/E6/E8/E10/E12/E13/E14: external provider inputs (private bundles, licenses, verifiers, credentials) — no local action can unblock.

## Sources

- `CHEAPEST_EXECUTION_RESULTS.md`, `NON_E11_READINESS.md`, `NON_E11_IMPROVEMENTS.md`,
  `preflight_summary.json` in this directory.
- Per-lane receipts are listed in `NON_E11_READINESS.md` and hashed in
  `preflight_summary.json`.
