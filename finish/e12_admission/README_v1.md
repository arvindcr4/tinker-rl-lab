# E12 offline admission, 2026-09-12

Implemented `validate_v1.py` and 10 focused unittest cases. This is a local
consistency gate, not a runner, score, permission grant, or launch allocation.
The observed manifest returns BLOCKED, with `launch_authorized: false` and a null
score. No model calls, downloads, messages, builds, or replay were performed.

Run from the repository root:

```sh
python3 -B -m unittest discover -s finish/e12_admission -p 'test_validate_v1.py' -v
python3 -B finish/e12_admission/validate_v1.py finish/e12_admission/observed_manifest_v1.json
```

The second command exits 2 for blocked inputs; 0 means offline evidence validated.
It prints JSON and never writes files. Disk is checked before every evidence read
and at completion, with a 6 GiB floor. Parent must recheck disk and attempt state
at actual launch; an offline snapshot cannot reserve an attempt or prevent races.

## Reconciliation and exact scope

Original contract: `zvf-program/flagship/pavlovs_domain_contract.json`,
`appbench_eval`, primary evaluation, held-out evaluation prompts, stateful,
artifact or side-effect evidence. The six public tasks cannot replace this
holdout requirement. Public status alone does not prove contamination either;
model-specific evidence is required and currently absent.

The local CSV, split manifest and disjointness proof exactly match the August 22
receipt SHA-256 values. Thus the August 9 `local_receipt.json` statement of zero
downloads is historical, not the current inventory. The six pinned tasks contain
151 rubric items: 24/33/22/25/23/24. The original split manifest contains stale
absolute paths to the older checkout; originals are preserved, and new descriptors
use repository-relative paths. This gate pins the original manifest bytes, avoiding
the older verifier's relocation-sensitive aggregate rebuild.

Official methodology: https://www.afterquery.com/leaderboard/app-bench
Three one-shot attempts per task, best per task, binary functionality grading,
two independent qualified full-stack graders and consensus on disagreements.
The website says 23 Financial Dashboard requirements; pinned CSV has 24. A
provider-backed resolution is required before accepting this exact rubric version.
An automated judge, reconstructed template, BrowserGym, or different public task
portfolio cannot satisfy original admission.

## Required manifest evidence

`observed_manifest_v1.json` intentionally leaves missing assets absent and model
and attempt null. No provider/model/attempt availability has been invented.
All assets use `{path, sha256}` descriptors. Paths must resolve inside the chosen
root, hashes must match, and files must be nonempty and at most 2 MB. Environment
descriptors may point to small immutable manifests describing large images and
bundles; this validator does not download or materialize those bundles.

The fixture in `test_validate_v1.py` documents the full input shape. Its purported
provider evidence is explicitly SYNTHETIC and is only used inside temporary tests.
It is not a reusable readiness package.

- `csv`, `split`, `disjointness`: exact hard-coded SHA-256 pins.
- `permission`: AfterQuery issuer, evaluation and aggregate-publication permission.
- `holdout`: AfterQuery issuer, exact six task IDs and model-specific training holdout.
- `environment`: official exact-environment attestation, image digest, hash-bound
  template, runtime/reset, deployment, artifact verification, side-effect
  verification and credentials policy manifests.
- `grading`: exact rubric counts and best-of-three aggregation, two distinct qualified
  graders, independent grading then consensus, and resolved 23/24 discrepancy.
- `attempt_ledger`: authoritative-complete attestation, all 18 task/attempt slots,
  explicit states, and hash-bound receipts for completed or failed slots.
  Selected attempt must be explicitly not started. Unknown or active states anywhere
  block admission; reused run IDs and repeated slots also block it.
- `parent_review`: reviewer identity, explicit authenticity/history verification,
  exact binding and selected attempt, and SHA-256 values for all five evidence docs.

Every evidence document binds suite, dataset revision, split file hash and immutable
model identity. The parent-review requirement records independently verified evidence;
booleans and issuer strings alone cannot authenticate provider grants. Even a passing
result leaves launch authorization false. Allocation, online tracking/checkpoint
receipts, resource readiness, and atomic attempt reservation are launch-time work.

## Smallest supported next action

No original E12 scored continuation is supported by current inputs. Parent can
populate and independently review exact evidence if received, select an explicitly
unstarted identity, then rerun this offline gate. Existing access request remains:
https://github.com/AfterQuery/appbench.ai-docs/issues/1 (open, zero comments on live
read during this task); pinned HF metadata still has no license card. No follow-up
message was sent. A private replacement bundle needs a separately reviewed contract
version; do not replace these pins silently.

Machine receipt: `.codex-run/finish_20260912/e12_admission/readiness_v1.json`.
