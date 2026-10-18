# Review 33 resolution (partial source)

The source cuts off during E9 provenance; unseen continuation is explicitly skipped.
No training, provider calls, paid jobs, benchmark runs, or stored-result edits.

## Implemented
- APEX rejects malformed scoring objects, booleans, strings, missing/non-finite scores before logging success. Raw grades are hashed. Native scale is preserved: the review conditions a [0,1] range on upstream confirmation, so no unverified range is imposed. Contract: finite-numeric-v2
- APEX rejects modified, staged, and untracked checkout files; the separately hashed adapter remains separate
- Wordle-specific split policy rejects disjoint but incorrectly ranged seeds; not generalized to other environments
- E13 local receipt v2 exposes assigned/rejected populations, leaves the official denominator unknown, and retains score prohibition
- Signed E13 intake labels provider assertions and explicitly records external receipts were not independently fetched; existing signed manifest, count, model revision and trust-root checks remain required
- SWE manifests reject non-object records and incomplete SHA-256 digests; CLI requires --evaluation_task_ids_path (JSON string list) and exact inventory equality
- E9 v2 rejects non-finite money/malformed native scores; continuous-to-binary target repair blocks pending explicit benchmark policy. Historical repair identification remains descriptive
- Verilog rejects invalid workers before scheduling and remains reference-only/scoreless

## Remaining evidence-dependent work
A reviewed exact-source/competition repair registry for all legacy E9 repairs, complete effective-program/submission/grader-input provenance, authoritative upstream populations, external receipt retrieval, spending concurrency/billing reconciliation and native differential candidate scoring remain unimplemented here. No credentials or synthetic fixture unlocks provider scoring.

## Verification
Offline pytest with PYTHONPATH=zvf-program/flagship and --import-mode=importlib. Focused E9/E13/E11 and regression run: 138 passed, 4 skipped, 5 deselected before adding three dirty-checkout tests. Five E9 tests require absent archived Freesound/Smartphone outputs; excluded, never fabricated. Toolchain skips do not establish native equivalence.

### Prospective repair registry and exact execution binding (also R38.17/R39.12)

A typed repair policy now binds competition, exact original/effective program
hashes, expected failure signature, scientific classification, review reference,
and an explicit submission contract (columns, count, ordered identifiers, finite
prediction fields). Semantic changes additionally require a benchmark-policy
decision. The runner gates every proposed historical string transformation
through that registry and validates repaired CSV postconditions before grading.
The production registry intentionally has no enabled policies: no supplied
review establishes official approval for those historical transformations.

New executions retain original and effective program artifacts, exact CSV bytes
(including CRLF), native grade bytes, the grader's submission-input hash and
revision/script identity, and the registered repair chain. Native grading checks
that its input bytes did not change. Replay collection recomputes the binding,
rejects rehashed but unauthorized treatment changes, and persists the original
program for idempotent revalidation. Saved-submission regrades check exact bytes
before grading and retain native input evidence. Historical results are untouched.

Synthetic fixtures cover the positive chain plus wrong competition/source/output,
wrong failure signature, missing policy/semantic approval, malformed CSV schema,
row count/order/IDs/nonfinite predictions, newline-only native-input substitution,
and missing/invalid native grades. Local hashes do not authenticate official
policy, datasets, grader installation, or generated-code isolation. No paid path
was enabled, and official native scoring conformance remains unestablished.
