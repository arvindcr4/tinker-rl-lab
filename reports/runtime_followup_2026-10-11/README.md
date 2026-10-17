# Runtime and uncertainty follow-up — 11 October 2026

This is an additive correction note for the public 317-page thesis, not a new
experiment or replacement release. No signed provenance, frozen thesis, release
manifest, historical result, or strict gate was changed. Research completion and
independent final-thesis review remain open.

## Campaign uncertainty: not enough provenance to repair the signed metadata

The signed campaign record already contains `sd: 0.12`; it does **not** lack a
numeric SD. It lacks an identified estimator and aggregation scope, and its
`rigor.reports_ci` remains false. Earlier handoff shorthand saying “missing SD”
was inaccurate.

The original public Git input is
`9b3d256061cfeb03946fb70b84bf7a9dee81abc3:experiments/results/campaign/results.json`.
Its SHA-256 and derived statistics are in [campaign_uncertainty.json](campaign_uncertainty.json).
To reproduce, retrieve that blob with `git show`, select G=4 by mode and seeds
0, 1, 2, then apply Python `statistics.mean`, `pstdev` (ddof=0), and `stdev`
(ddof=1) to `heldout_after - heldout_before` (equal to `heldout_gain`).

| Scope | Mean gain | Population SD | Sample SD |
| --- | ---: | ---: | ---: |
| Baseline, G4, seeds 0–2 | 0.027778 | 0.103935 | 0.127294 |
| Curriculum, G4, seeds 0–2 | -0.027778 | 0.141639 | 0.173472 |
| Paired curriculum minus baseline | -0.055556 | 0.171234 | 0.209718 |
| Six G4 rows pooled across arms | 0.000000 | 0.127294 | 0.139443 |
| All nine rows, mixing G2/G4/G8/G16 | 0.027778 | 0.111111 | 0.117851 |

Both the mean of the two arm population SDs (0.122787) and the sample SD of
all nine mixed-design rows (0.117851) round to 0.12. Neither numerical match
identifies the intended uncertainty quantity. The original config introduced
in `def06729a` provides the literal, not its derivation. The historical findings'
0.10 and 0.14 match the arm population SDs, not the sample SDs. Therefore no
justified correction to the signed record or its `reports_ci` flag is possible
from these artifacts alone.

The current campaign results have 15 rows including six G4 seeds and scores
consistent with 30-item evaluation, whereas the historical signed record and
findings describe three seeds and 12 items. Substituting the current file would
mix evidence versions. The campaign generator also shuffles held-out items per
seed: these are not repeats on one fixed evaluation sample. No power or
no-effect conclusion follows from this reanalysis. The single-seed sweep still
has no additional seed evidence and remains strictly incomplete.

## OpenRLHF runtime correction

The native driver previously hardcoded seed=42, G=8, token caps, batch sizes and
240 samples, ignored the requested optimizer-step horizon, used a reward-service
placeholder, continued after subprocess failure, and fetched metrics by a
nonunique W&B display name. Empty history became numerical zero, and local
execution was labelled `modal-h100`.

The command is now independently CPU-testable. It forwards supported settings,
uses the active interpreter and absolute output paths, requires an explicit
HTTP(S) reward endpoint, and rejects unsupported requests before provider work.
A nonzero subprocess exit raises. Existing result receipts are preserved.
Importing the package no longer starts an emissions tracker. The wrapper no
longer queries ambiguous remote histories, fabricates reward summaries, starts a
second W&B run, or automatically uploads checkpoints. Native CLI W&B
logging now requires explicit `log_to_wandb=True`; an environment key alone does
not enable it. Local TensorBoard logging is the default. The source-checked
reader records finite, contiguous training rewards from one fresh event file
only when the full planned rollout count is present, with a source hash and
`logged_rollout_steps`. Missing or ambiguous logs remain `not_collected`. Every
receipt retains `steps: null`: logged rollout iterations are not optimizer-step
measurements. See the [second-pass review](RUNTIME_REVIEW.md).

The contract targets the official
[OpenRLHF v0.8.4 CLI](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/cli/train_ppo_ray.py).
That parser exposes a dataset `max_samples` budget, not a `max_steps` option.
The wrapper explicitly rejects `max_steps` rather than treating samples as
optimizer steps. Consequently the unified launcher's canonical step-budget
path is **blocked until a tested native step-budget adapter exists**. Direct
native use requires `max_steps=None`, explicit GRPO and reward URL, and deliberate
dataset/batch settings. This change does not claim a tested GPU training path.

These code corrections do not retroactively establish the settings of any
historical run or remove the thesis's managed-default, data provenance,
decontamination, single-seed, and evaluation limitations. The separate historical
Modal experiment script was not altered or executed. Independent final-thesis
review must account for this note; the frozen PDF remains unchanged.

## Validation at commit d47e2b9c (prior run)

The [review continuation](RUNTIME_REVIEW.md) records its own new checks separately.

Validated on macOS with Python 3.12.13, CPU mocks only:

- Focused runtime/launcher tests: 30 passed.
- Full suite: 1,280 passed, one intentional skip, three warnings; 84.53% coverage
  against the repository's 77% gate (92.29 seconds). Coverage scope is the
  repository's configured surface, not all training backends.
- Repository Ruff 0.16.10 lint, policy and formatting: passed; the new native
  command module also passed an explicit lint check.
- Configured mypy surface: 20 files, passed. This is not full-repository typing.
- Aggregate audit suite: nine of nine passed, zero issues.
- Public stdlib tests: 237 passed; frozen publication integrity preserved.
- Offline wheel build and nine-module wheel check: passed.
- Campaign strict verification: rejected at 95.7% (uncertainty metadata).
- Sweep strict verification: rejected at 82.6% (single seed and uncertainty).
- Exact October 11 PDF/receipt check: rejected because independent review is not
  recorded as passing. The default older-revision receipt also remains stale.

No GPU/model job, provider upload, experiment, publication or Git push was run.
The original dirty developer checkout and the existing untracked `.coverage`
file in this integration checkout were preserved. Fresh remote inspection found
main unchanged at `1ae7320981bb3137dba703f612a16056eac83ef2` before this local branch.
The previously completed GitHub workflow 38128942546 had 14/15 passing jobs,
with only strict provenance failing; these local changes have not run on CI.

Priority remaining work:

1. Establish and test native optimizer-step enforcement and exact run-bound
   metrics before claiming the unified OpenRLHF path is operational. Validate
   the explicitly configured reward service and GPU topology in an authorized
   future run; CPU command tests cannot establish training correctness.
2. Recover an authentic campaign uncertainty derivation or create a clearly
   separate, reviewed statistical analysis with a specified estimand. Preserve
   the old signed record. Obtain genuine additional sweep seed evidence if that
   experiment is to satisfy the existing strict gate; this phase did not do so.
3. Complete independent review of the exact 317-page thesis, incorporating this
   correction note. No existing review receipt was upgraded.
4. Resolve the previously documented scientific gaps: training-source and
   decontamination provenance, unresolved manifests and collaborator run
   permission. Runtime fixes and identity receipts do not close these gaps.

Current manuscript/report corrections are enumerated in [MANUSCRIPT_ERRATA.md](MANUSCRIPT_ERRATA.md); the frozen export remains untouched.
