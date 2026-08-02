# Autoresearch summary

Status: `CEILING`

Goal: improve the evidence-bounded ZVF/GRPO follow-up using the pinned RLHF Book and Harvard CS2824 resources.

## Working metric

- Fixed adversarial mutation score: `12/16` to `16/16` (`+33.3%`).
- Classic iterations: 4; kept: 3; reverted after formatting failure: 1.
- Highest-impact metric changes: controlled failures for missing foundations sections (`+2`), immutable review-bundle enforcement (`+1`), and mandatory stage evidence (`+1`).

## Holdout-driven improvements

- Renamed success to `POSTTRAINING_FOUNDATIONS_CONTRACT_LINT_PASS`; promotion remains false.
- Added strict source repository/commit/file hashes, canonical protocol/ledger/packet digests, audit byte bindings, payload-to-file binding, and duplicate-key rejection.
- Added a seven-claim theorem-transfer ledger with exact source locators, formal domains, assumptions, LLM mappings, falsifiers, and per-claim non-promotional language.
- Added an offline S0-S2 packet with independent `Y_ind`, prompt-clustered splitting/cross-fitting, exact fixed estimator, four-contrast multiplicity, one-class `NOT_IDENTIFIABLE`, power/sample gates, checker-governance receipts, and on-policy prompt/completion separation.
- Added executable runtime adjudication for `PASS`, `FAIL`, `NOT_IDENTIFIABLE`, and `NOT_FEASIBLE`.

## Verification

- Fixed metric: `16/16`.
- Focused tests: `30 passed`.
- Full suite: `125 passed`.
- Ruff check and format check: pass.
- Frozen review bundle: deep verification pass.
- Source/assumption cold-start gate: pass.

## Remaining units

1. Enforce exact numeric types, finite/range checks, observed-total consistency, and checker implementation-hash equality in runtime S2 receipts.
2. Require non-null, schema-validated cross-fit/bootstrap/placebo evidence and substantive S0/S1 values before a receipt may pass.
3. Freeze the remaining runtime details: marginal-versus-joint power interpretation and PRNG/row-order rules.

No GPU/external run, promotion, push, publication, or deployment was authorized or performed.
