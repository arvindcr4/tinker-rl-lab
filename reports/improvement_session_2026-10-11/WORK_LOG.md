# Improvement work log — 11 October 2026

Authorized session: 13:50 UTC through 23:49:31 UTC. Starting commit
`a5d4c46712b8d86d70b4fcf84f78de65285b2b39`. Scoped local commits only;
no provider/model/GPU jobs, paid work, new credentials, external disclosure or
institutional submission. Original user edits and pre-existing `.coverage`
are preserved. Tests and source inspection do not establish native execution.

## Priority queue

1. Reproduce and repair ingestion/evaluation false positives or silently altered inputs.
2. Independently examine statistics and result verification for false passes and lost data.
3. Examine public evidence checks and reconcile mutable reports with actual source coverage.
4. Review each accepted patch independently and run focused checks; move to the next justified task.
5. At the deadline stop new work, settle checks and run/record final applicable validation.

## Milestones

- 13:50 UTC: baseline preserved; three independent read-only reviews started.
  The already established external OpenRLHF stop/reporting limitation will not
  be repeatedly retried or replaced with an unvalidated backend fork.

## Decisions and boundaries

Frozen scientific artifacts and signed records will not be rewritten to make
checks pass. A suspected bug is not a finding until a bounded reproduction or
source-level execution trace establishes it. Any compatibility tradeoff must
be recorded with the change. Incomplete evidence remains incomplete.

### First reviewed corrections (14:02 UTC)

- Evidence validation now rejects impossible E5/E6 subset counts and E2/E9
  success/status contradictions. Six new negative cases failed before the fix;
  26 focused tests passed afterwards, and stored evidence still passes all21 checks.
- Reward/input review reproduced Boolean/number argument conflation, SWE test
  lists parsed as characters, corrupted patch whitespace accepted as equal, and
  JSON exponents overflowing to infinity. The fixes preserve typed arguments,
  decode test lists, retain patch content whitespace, and reject nonfinite JSON.
  Independent review passed139 tests; the added overflow correction passed143.
  These code corrections do not repair historical scores or establish native
  patch execution. Frozen provenance receipts intentionally retain the old hashes.
- Result identity and statistics patches remain under review; cross-review found
  seed-case handling and ignored alternate step metadata requiring follow-up.
- User requested Muse reviews of authorized ESA draft/public text. Four bounded
  prompts prepared, but no review launch confirmed: executor transport failed.
  No new private source or dataset was included. Extra review launches are held
  while the user clarifies Codex usage concerns. Per-task usage counters unavailable.

### Result-verifier identity correction

The verifier now requires an exact experiment identity (with recognized seed
suffixes), preserves exact seed-suffixed reference names, rejects conflicting
case-insensitive seed tags and duplicate reference identities, and bounds both
reference and observed accuracy to[0,1]. Initial regressions failed11 cases;
independent review found two additional regressions, both reproduced and fixed.
The final focused run passed80 tests (one existing SciPy precision warning).
Unknown unrelated files remain outside the matched-results contract; a passing
summary is not a statement that every file in the directory was validated.
Local commits so far: `e1254ab6` evidence guards; `f2f3af97` ingestion/reward fixes.
