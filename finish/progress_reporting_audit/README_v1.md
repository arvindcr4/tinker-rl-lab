# Receipt-derived progress report generator

`generate_v1.py` produces one JSON report with exactly 14 lane records and a
Markdown report. Each lane carries **original benchmark scope** and **named
portfolio evidence** separately. It reads existing receipts, not cloud APIs,
models, controllers, containers, or an LLM-generated status summary.

From `/Users/arvind/Developer/tinker-rl-lab`:

```sh
python3 -B -m unittest discover -s finish/progress_reporting_audit -p 'test_generate_v1.py' -v
python3 -B finish/progress_reporting_audit/generate_v1.py --out .codex-run/finish_20260912/progress_reporting/my_new_snapshot
```

Choose a new snapshot directory each time. Existing outputs are never overwritten.
Outputs outside a new subdirectory of `.codex-run/finish_20260912/progress_reporting`
or `finish/progress_reporting_audit` are rejected, including the parent's live
reporting paths. Inputs are read-only. Exit 2 means a missing/malformed receipt,
changed contract, hash mismatch, concurrent evidence change, or disk-floor failure;
it is not a score. Do not suppress such failures to publish stale counts.

The generator checks root and workspace disk space before reads and before each
output file against 6 GiB. Unit-test temporary files remain in the owned reporting
directory and are removed by the tests. No dependencies need installation.

## Evidence and version boundaries

- Original suite identifiers are checked against the domain contract. Known original
  denominators come from lane receipts/manifests and are compared with reviewed
  constants. Unknown private denominators stay null. A change fails for review.
- Replacement suite identities and declared denominators are validated against the
  saved portfolio mapping. They do not rename or complete original suites.
- Every consumed file is SHA-256 recorded; explicitly bound child receipt hashes
  are checked where consumed. Historical `agentic_repos/tinker-rl-lab` absolute
  paths are mapped only to the same repository-relative path in this checkout.
- File hashes and discovered membership are checked again before publication.
  Growing run directories can invalidate a snapshot; rerun into a new path.
- E1 selects one latest audit per disjoint batch, checks native report hashes,
  rejects duplicate task identities, and retains native-report vs terminal-error
  counts separately. No repeated audit snapshots are summed.
- Tau3 reads terminal native episode records per run, binds membership to that
  run's native-ready task contract/revision, rejects duplicates, and preserves
  per-run coverage and success without pooling overlapping starts. A continuation
  allocation (e.g. 60 selected tasks) does not redefine the full 97-task scope.
- E13 checks distinct completed episode IDs and native/terminal file hashes.
  E14 uses the saved native scoring recheck with explicit accepted/excluded counts;
  it does not rerun a model or judge.
- E9 legacy coverage is derived from indexed native receipts and unique competition
  IDs. It is not an aggregate competition accuracy and excludes the separate merged
  arm. The historical receipt index bounds discovery; this is not an exhaustive
  search for new unindexed legacy attempts.

This is **receipt-level verification**, not a full transitive artifact audit.
Provider authenticity, every raw model answer, all grader runtimes, cloud job
states, and new artifacts outside the documented discovery patterns are not
revalidated. For newly introduced suites, run layouts or receipt schemas, review
and version the adapters rather than silently guessing. The raw `report.json`
contains source hashes, discovery membership, exact counts, and a code hash so
reviewers can identify the evidence and generator version used.

## Counting rules

Evaluation coverage = native-graded items / the explicitly named scope.
Success/accuracy = passes / the metric's explicit denominator. These denominators
need not match. Unknown stays null; zero is a valid result.

The /14 progress count follows the user's rule that verified partial scores count.
It is computed for the named replacement portfolio only. Complete named scopes
are counted separately, and no aggregate accuracy across suites or runs is computed.
An E1 terminal-attempt success rate with errors-as-zero remains separately labelled.

Examples protected by tests and runtime checks:

- E7 BinaryAudit: **28 primary**, 10 heldout-labelled, 8 training = 46 inventory.
  Prior errored dnsmasq reward 0.0 remains a historical attempt, not accuracy.
- E8: full public LAB-Bench evaluation coverage differs from accuracy, abstention
  coverage, private LAB-Bench, and original LifeSciBench completion.
- E10: benign AgentDojo utility is not AgentHarm private scope or injection security.
- E11: canonical 129/312; 129/311 is retained only in its original sensitivity receipt.
- E13: BabyAI 13/13 successes cover only 13/255 BALROG episodes, not OpenReward Games.
- E14: 2271/4426 accepted-report accuracy; coverage 4426/4428, with two exclusions.
  Strict full-scope score stays null, and FrontierMath remains a separate private suite.

Parent `progress_reporting_state.json` is captured for comparison only. Its string
percentages are not used to derive results, and no parent state is overwritten.
