# Review 08 registry resolution

Review source: `review-inputs/review_08_registry.md`, supplied from
https://chatgpt.com/c/6ad362f2-56c8-83ee-bd65-f6cb880f6a4d.

Baseline: combined batch-one commit `16cde307`; branch
`fix/pro-review-slice08-work`. Relevant registry-only improvements from PR20,
`fix/pro-review-samestack-v2`, were inspected and reused as a starting point.
No whole-PR cherry-pick or unrelated same-stack training/demo changes were made.

“Fixed” describes code contracts and local reproductions. It does not certify
historical experiments or create missing scientific evidence. Historical entry,
result, key, signature, and audit files were left unchanged.

## Confirmed bugs

| Finding | Status | Resolution and evidence |
|---|---|---|
| A1 / P0 stale-clean strict gate | Fixed | Unique temporary child output; nonzero/startup failure is blocking and stderr visible; required current manifest, schema/sidecar digests, validator version and implementation hash; missing/malformed/mismatched output and concurrent input changes fail. Tests cover prior clean cache, success without output, stale digest/version, changing inputs, and warning verdict parity. |
| A2 / P0 unsigned publication acceptance | Fixed | `audit_file(record_path, trusted_key)` is the sole accepting publication gate. It verifies an externally trusted Ed25519 signature, versioned record schema, full anchors, and evidence consistency. `verify(record)` is explicitly unauthenticated diagnostics and can never accept. CLI publication verification defaults to this gate. File signing also rejects evidence-invalid records. |
| A3 / P0 contradictory/fabricated declarations | Fixed | Missing explicit config raises; unknown claims stay null. Fixed checks validate nested settings, exact distinct integer seed count, completed per-seed artifacts, saved split identifiers/count/disjointness, decontamination artifact, finite results and saved analysis. Means/SD/per-seed results recompute from bound artifacts. Signed internally contradictory fixtures reject. Authentic artifacts still do not prove the historical execution actually happened; see D4. |
| A4 / P1 trust-anchor replacement and secret permissions | Fixed | Signing never generates keys. Provisioning refuses either existing member, including `force`; rotation requires a new directory. Private/public derivation is checked. Secret creation uses exclusive/no-follow opens with mode 0600 inside a 0700 directory, and public keys are archived by fingerprint. Tests preserve the old public anchor after private deletion and reject mismatched pairs. |
| A5 / P1 score offsets failed integrity | Fixed | Acceptance is an explicit conjunction; every required check has a fixed position. Unsigned or failed-integrity records have grade `REJECTED` regardless of reporting percentage. Full SHA-256 code/reward/config hashes, exact embedded resolved config, repository-root full commit, and tracked-worktree identity are verified. Omitted anchors never remove checks. |
| A6 / P1 verification overwrites input | Fixed | Audit output appends `.audit.json` to the full filename, including extensionless files. Atomic writes reject input aliases, hardlinks, symlinks, signatures, trusted keys, source and referenced evidence destinations. Tests cover extensionless names and `.json` in ancestor names. |
| A7 / P1 malformed ingestion and shrinking denominator | Fixed | Every query path uses centralized schema validation before indexing. Unknown documents, duplicate IDs/properties/delta rows and unsupported statuses fail. The two immutable legacy amendments are explicitly classified by filename and full digest in `sidecars.json`. Badge denominator is the fixed substantive schema-leaf catalogue; deleting a null leaf earns no credit. |
| A8 / P1 self-referential freshness | Fixed | Generated projections are rebuilt from specs, including all outcomes. Only explicit manual-annotation namespace survives. Unsupported retained/edited outcomes cannot pass `--check`; derived projections are schema-validated and idempotent. Historical records were not regenerated. |
| A9 / P1 regex anchor disagrees with value | Fixed | Numeric captures are derived or asserted; optimizer keywords/betas, LoRA targets and batch ratios use AST extraction. Captured descriptions cannot supply unrelated hardcoded text. Unobserved defaults are unknown. Semantic annotations supported only by source location are explicitly inferred/conditional, not mechanically extracted code values. Fixtures reproduce changed betas and save interval. |
| A10 / P1 partial E3 reward fingerprint | Fixed | Versioned `python-reward-assignment-and-module-v2` fingerprint covers complete AST reward assignments, parser source and the complete module conservatively, including later transformations/dependencies. Both reward branches, comparison and transformation edits change the fingerprint. |
| A11 / P1 dry-run execution confusion | Fixed | Generated comparison catalogue is complete with explicit unknown cells, with no legacy fallback. Dry-run intentions and synthetic diagnostics have separate namespaces; observed model/task/seeds remain unknown. All top-level fields and owned outcomes are refreshed in every branch. |
| A12 / P1 dropped adaptive rule | Fixed | Canonical `batch.adaptation_rule` is populated by legacy and generated adapters. Opposite adaptive policies yield an R3 difference. |
| A13 / P1/P2 conversion invents/erases knowledge | Fixed | Unsupported decontamination/parser claims are null. Explicit disabled clipping uses `clipping_applied=false` and null inapplicable epsilons; shared scoring excludes epsilons only in that explicit state. Legacy conversion restores `loss.clip='none'`. |
| A14 / P2 lossy numeric identity | Fixed | Integers remain exact; finite configuration floats compare without global rounding; booleans remain distinct. Large seed and tiny epsilon differences no longer collapse to R0. Nonfinite JSON is rejected. |
| A15 / P2 ambiguous signed JSON | Fixed | Signing, verification, registry ingestion and generation use one `json_protocol.py` loader/canonicalizer. Duplicate properties, NaN/Infinity/overflow, invalid Unicode and non-JSON in-memory values reject. Canonicalization is explicitly versioned as Python JSON, not falsely claimed RFC 8785. |
| A16 / P2 CLI/audit defects | Fixed | Framework filtering occurs before both implementation views. Warning verdict/exit decisions agree. External provenance sources reject as structured evidence failures. Report caches require current input/output/producer receipts; `--refresh` recomputes in an isolated temporary copy and prints snapshot identity. Regression smoke tests confirm all three refresh commands leave stored evidence unchanged. |

## Policy/design findings

| Finding | Status | Policy and limitation |
|---|---|---|
| D1 / P1 R0 versus comparability | Fixed | Independent `comparable`, evidence completeness and matching decisions; unilateral delta union contributes unknown R4 evidence and lowers coverage; checkpoint revision is not exempt. Descriptive R0 never establishes acceptance. `--exit-level` returns 6 for unresolved R0, including empty/all-unknown aggregate populations. |
| D2 / P1 evidence classes | Fixed | Disjoint observed/source-defined/assumed/unresolved coverage is reported. Defaults, name inferences and hand-entered values make comparison conditional. Composite H2H subset is unknown when launcher limit is absent; explicit name markers alone support variant inference. Ambiguous names stay unknown. |
| D3 / P1 signing trust policy | Fixed | Ed25519 only, caller-supplied trust and allowed algorithm policy; no automatic HMAC downgrade or embedded-key trust. Authenticated protected header includes protocol, canonicalization, algorithm, key fingerprint and record digest. Tampering any header fails. Historical v1 signatures remain untouched and are not silently upgraded. |
| D4 / P1 execution binding and outcome derivation | Partly fixed | H2H outcomes are recomputed from distinct completed per-seed logs and bound to their full digests. Other extracted outcomes identify the source digest and unreproduced upstream computation. New provenance publication requires hash-bound run/split/analysis artifacts. Historical runs lack independently verifiable immutable model/tokenizer/dependency/execution receipts; generation states execution binding is unknown. Creating those receipts would require unavailable original runtime evidence or new authorized runs, neither fabricated nor performed. |
| D5 / P2 shared schema and metric definitions | Fixed | Provenance and registry share the canonical MIN-REPORT schema, canonical aliases, strict JSON and scoring. Protocol/schema/coverage versions and full schema digest distinguish the new contract. `reported_leaf_coverage_v3` is the unrounded mean of seven substantive leaf fractions; `item_presence_coverage_v3` is item presence, excluding explanation/source metadata. Disabled/inapplicable clipping is explicit. Historical schema remains readable without promoting unknowns. |
| D6 / P2 overstated severity/rigor | Fixed | R-levels are field-based descriptions, not causal-effect guarantees; trainer/reward implementation changes require semantic review. Seed wording is “minimum distinct-seed requirement,” not power. Publication requires every evidence check plus trusted authentication; percentage/grade alone never approves. Saved uncertainty is consistency-checked, with statistical adequacy explicitly outside the authentication guarantee. |

## Verification

Environment: `/tmp/tinker-review-venv/bin/python` (Python 3.12.14).

- `-m pytest -q tests/test_registry_review08.py tests/test_registry_stackdiff.py tests/test_registry_provenance_review.py tests/test_provenance.py`: 167 passed (136 new regressions, 31 existing tests), final run 8.33 seconds
- `query.py validate`: 46 records pass, including read-only recognition of two historical amendment sidecars
- `query.py validate-strict`: PASS, 28 stacks / 18 deltas, zero errors/warnings, 54 informational unknown-item findings
- `query.py coverage --refresh`, `measured-coverage --refresh`, `antiherding --refresh`: successful isolated recomputations with snapshot identities; source registry/results unchanged
- `generate_entries.py --check`: expected exit 1, listing all 12 historical generated entries as stale under v3; those artifacts intentionally remain unchanged pending a separately reviewed migration
- All 12 in-memory regenerated projections satisfy the new schema and are idempotent
- Python parsing, Ruff undefined-name/import checks and `git diff --check` run on changed code

No training, provider calls, paid jobs, stored-result/registry-entry/signature/key
rewrites, push or merge. Missing historical execution receipts and trusted
production private keys were not synthesized or provisioned. Publication on
historical evidence remains blocked unless it independently satisfies the new
contract and an authorized signer supplies approval.

Independent-audit follow-up: the strict producer now resolves delta references
against validated record IDs, never filename stems. A real subprocess regression
rejects a `delta_dapo.json` file declaring `delta_aero` when a stack references the
missing `delta_dapo` ID. Historical files remain untouched.
The authenticated resolved-configuration comparison now validates the candidate
MIN-REPORT block against the shared schema and compares canonical bytes, preventing
Python's `True == 1.0` equality from accepting a changed sampler configuration.
The strict gate pins producer bytes before launch and requires both the receipt
and post-run file to match, rejecting a producer replaced during execution.
