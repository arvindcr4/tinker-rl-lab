# Review 34 resolution (partial supplied review)

The supplied review cuts off in P1.6 during patch-repair tests; no unseen continuation or final-five list was inferred. This change performs no training, provider execution, paid jobs, real benchmark run, or stored-result edits.

## Implemented
- P0.1/P1.5: xLAM explicitly labels its test_examples protocol as custom component v2 and the metric as strict_tool_call_reward, never official scoring. Execution requires a JSON ordered content-ID manifest. IDs are SHA-256 of canonical compact sorted prompt/target JSON; task-ID digest is SHA-256 of newline-joined IDs. Actual selected content must match exactly, uniquely, before sampler creation. The historical shuffled-train conversion remains explicitly custom.
- P0.2: SDAB rejects duplicate or out-of-manifest selected IDs before sampler creation, binds each result to its selected task, requires explicit integer usage, and distinguishes subset from full denominator. Receipt v2. No unverified official numeric range is invented.
- P1.1: Frontier verifier timeout is terminal TIMED_OUT; missing, boolean, string and nonfinite scores cannot complete. Numeric-only native contract v2 is not a claim of exact official task schema parity.
- P1.2/P1.3: missing negative control cannot pass MLE validation; rule acceptance preserves unknown and rejects nonboolean declarations.
- P1.4: E1 native results must be an object of booleans. Full/GPU resume validates image manifest schema/hash/coverage/digests and task-to-image bindings before reuse.
- P0.3 mitigation: SDAB checks observed cost after each task before continuing, instead of after the complete batch; this is explicitly not an atomic shared spending cap.

## Remaining / evidence-dependent
P0.2 official score range and aggregation need the pinned provider contract. P0.3 remains open: E1/E9/SDAB lack a single concurrency-safe provider/infrastructure reservation and settlement ledger; projected ceilings and per-task observed checks must not be represented as hard spend enforcement. P1.6 raw/submitted artifacts already exist, but full git-apply differential conformance is not established by these changes. Saved local hashes establish integrity, not independent provider authenticity. Docker timeout cleanup of remote containers remains a separate runtime concern.

## Offline verification
83 passed, 4 optional/archive-dependent skipped across targeted new regressions and xLAM/SDAB/MLE/Frontier tests, then added an explicit rule-acceptance regression. Existing E1 test uses an invalid short digest and expects the later URI diagnostic; corrected independently in review 39 rather than weakening production validation. No native parity claim is made.

### P1.6 follow-up: executed local git differential fixtures

Ten synthetic fixtures execute real `git apply --recount` on raw patches and
ordinary `git apply` on repaired patches in separate temporary repositories.
Count-only fixes compare exact resulting bytes for multiple files, additions,
deletions and missing final newlines. Empty and malformed input is rejected by the production validator;
nonapplicable input fails real git application. The fixtures explicitly expose two
non-equivalences: git --recount accepts context-only hunks that this adapter
drops, and ordinary git may ignore trailing malformed text after counts are
fixed, so the adapter's structural validator remains mandatory. These tests
establish local patch mechanics only, not native SWE scoring conformance.
