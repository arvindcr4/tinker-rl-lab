# Review 15 resolution

Base: `6407255d`. Protocol changes in this patch are prospective v2 contracts;
no historical results, manuscript artifacts, training runs, or remote state were changed.
No autoresearch agent/service was launched. SDK interactions in regression tests are mocked.

## Priority security follow-up

**Conditional P0: blocked on owner evidence.** The repository documents two historical
Tinker credential exposures. This work neither tested credentials nor revoked them.
An owner must confirm revocation with the provider and retain a non-secret incident
record. Do not interpret this patch or a clean working-tree scan as revocation evidence.

## Finding-by-finding disposition

| # | Status | Resolution and remaining boundary |
|---|---|---|
| 1 | partly fixed | Removed blanket task-output paths, identifier-shaped blanket secrets, and entire-line digest exceptions. Regression checks prevent their return. Executable real-scanner canary regression is included but skipped because the binary is unavailable locally; historical revocation remains blocked above. |
| 2 | fixed | Both W&B inspectors sanitize emitted records recursively, including secret-bearing keys, environment credential values, provider patterns, authorization values, legacy 40-hex keys and authenticated URLs. Config selection uses exact names and scalar types. Arbitrary log excerpts remain opt-in and sanitized. Regression covers nested values and URLs. Sanitization is defense-in-depth, not proof arbitrary text contains no novel secret format. |
| 3 | fixed | Experiment selection/completion instructions now use validation only, prospective freezing, negative/inconclusive completion, descriptive seed variation, and a prespecified inference contract. Removed the operative test-selection and success-dependent completion rules. |
| 4 | fixed | Critic failures remain failed/incomplete; removed stub-success permission. Final v2 gate requires critic coverage, matching run/query/corpus/final hashes, completed envelopes, no unresolved critical findings, and passed lint status after readability edits. Offline tests reject stale/failed/missing reviews and post-review edits. Runtime must invoke the documented gate; it does not itself launch critics or lint. |
| 5 | fixed | Content-addressed successful review envelopes bind PDF/text, prompts, model, generation settings and schema. Schema-invalid output is not cached. Synthesis keys include ordered upstream records; records are atomically replaced. Historical output directory is rejected and deterministic inventory is no longer overwritten by the reviewer. |
| 6 | fixed | HF audit validates nonempty unique unit/repository roster and pinned revisions, emits PASS/FAIL/INCOMPLETE v2 verdicts, and succeeds only for complete mandatory integrity checks. Resume progress is optional. Duplicate repositories are conservatively rejected rather than silently collapsed; a future shared-repository mapping requires an explicit protocol. |
| 7 | fixed | Tracked required-record manifest owns expected record/signature coverage. CI invokes a fail-closed inventory verifier before existing strict record/signature checks. Empty, missing and unexpected records/signatures fail. Identity here is the tracked record filename and existing verifiers; manifest changes require review. |
| 8 | partly fixed | Shared exact five-by-four cell identity/schema predicate, marker order/uniqueness validation, and response_contract_checks v2 naming. Explicitly declares semantic evidence unvalidated. A contradictory prose addition is still outside lexical checking; immutable source-bound numerical validation and human semantic adjudication remain required. Historical summary was preserved. |
| 9 | partly fixed | Quick start now uses the canonical locked development setup and an import-only smoke command; removed unrelated atropos installation. Fresh install and separate validated atroposlib integration profile were not executed/created. |
| 10 | partly fixed | Width sweep waits for all terminal fetch outcomes and freezes source IDs/hashes. Separate preflight/agent gap files must merge with mandatory IDs preserved. These are prospective instruction contracts; executor-level merge/barrier implementation is not supplied by this patch. |
| 11 | fixed | Source-tensions instructions allow zero supported tensions, record searched-corpus rationale, allow unresolved outcomes and condition the report section on evidence. |
| 12 | partly fixed | Source validator resolves each root's bibliography independently and counts include occurrences with recursion-stack cycle rejection. Regression covers another paper's key and repeated labeled include. This lightweight scanner is not a TeX interpreter; actual .aux/.blg/.log validation remains necessary for macros and conditional compilation. |
| 13 | partly fixed | W&B scan uses bounded islice with one lookahead and reports truncation; detail history is explicitly sampled. Tinker sessions report first-page query and incomplete scope. Full pagination and uniform retrieval/endpoint/access metadata remain pending. |
| 14 | fixed | ArithmeticEnv extracted into side-effect-free module; CI uses normal pytest with correct/incorrect reward, termination and deterministic reset assertions. Test skips locally when optional Gymnasium is absent; CI's environment job installs it and pytest explicitly. No training is imported or launched by the test. |
| 15 | partly fixed | Review stages specify run-local paths and run/query/corpus/draft binding; gate rejects stale review envelopes. Enforcement of read-only mounts, output-path allowlists, patch-size validation and migration of all legacy stage paths remains executor work. |
| 16 | partly fixed | Reviewer requires an explicit aggregate request-attempt cap before dispatch, bounded retries, input-size preflight, finish-reason check, and immediate stop on permanent auth/model/contract failures. Token/currency reservations, usage receipts, elapsed-time and campaign scheduling enforcement remain unimplemented; do not use this cap as a monetary guarantee. Historical execution plans were not rewritten. |
| 17 | partly fixed | Locked onboarding and uv dependency-updater ecosystem align with canonical Make targets. Existing lock/export check remains authoritative. Separate pinned optional integration/SB3 profiles and updater export automation remain pending. |
| 18 | partly fixed | Groq import moved into execution-only main, so deterministic utilities import without the SDK. Personal credential-file reads replaced by explicit environment configuration, helper locations by explicit installed-package root. Shared provider-free helper extraction and versioned review extra remain pending. |
| 19 | partly fixed | Renamed misleading pdf_current output to pdf_mtime_not_older_than_tex. This is explicitly only an mtime heuristic; no source-to-PDF correspondence is claimed. Recorder-input/toolchain/build receipts remain to implement; stored historical inventories were not rewritten. |
| 20 | partly fixed | Prospective citation/novelty contracts require claim support passages, unresolved placeholders, inconclusive novelty and bounded counter-search interpretation. Publication-time executable evidence validation remains pending. |
| 21 | skipped | Python 3.11 Ruff target was evaluated but enables 96 B905 findings in unrelated existing files; deferred until a reviewed migration can preserve behavior. Reviewed immutable action SHAs, Gitleaks release digest pin, broad F821/F811 exemption cleanup and duplicate build consolidation also remain pending. No guessed digest or action SHA was introduced. |
| 22 | partly fixed | Active paper-review command now requires a new explicit output directory and refuses its frozen historical tree. Catalogue generation, shared package migration and lifecycle metadata for all dated tools remain pending. |

## Validation

- `python -m pytest tests/test_review15_hygiene.py tests/test_arithmetic_env_contract.py -q`:
  **11 passed, 2 skipped** (optional Gymnasium and real Gitleaks binary absent in the supplied review environment)
- Targeted Ruff checks on modified/new audit utilities and tests: passed
- Regression suite is offline; provider calls, provenance subprocesses and W&B iterator are mocked
- No complete fresh installation, Gitleaks binary scan, remote audit, agent execution, or training performed
- Full `make lint-ruff` exposes 186 pre-existing baseline errors; mypy is unavailable in the supplied review venv. Targeted changed-file lint passes.
- Local targeted tests are not a claim that the entire repository CI or all optional stacks pass

## Migration notes

- Paper audit CLI now needs `--output-dir <new-directory> --max-requests <authorized-attempt-cap>`;
  old bare audit JSON remains historical and is never accepted as a successful v2 cache
- Inspector schemas and review receipts are v2; update consumers of old history/session/freshness fields
- Required provenance manifest v1 is a reviewed inventory contract; deleting a record requires an explicit manifest change
- Do not rename historical evidence or recompute stored results to satisfy new contracts
