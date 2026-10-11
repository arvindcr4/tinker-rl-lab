# Linux VM validation and fail-closed follow-up — 11 October 2026

This is an additive software-validation note. It does not replace the frozen
thesis, authenticate a training run, supply missing scientific provenance, or
record independent thesis approval.

## Source and environment

The Linux checkout was fetched from the published repository, with both branch
tips verified before validation:

- `fix/provenance-openrlhf-20261011`:
  `ca799b872f8e65f7f76b887b99a3d0f967c2985a` (validation base)
- `review/local-pes-source-20261011`:
  `71c60b6a90643557a8611e6ec1906b07e242c891` (verified, not merged)

Runtime: Linux x86_64, Python 3.12.14, uv 0.12.23. Development dependencies were
installed from the unchanged lock with `uv sync --locked --extra dev`. No Mac
environment or credentials were copied. No GPU training, provider evaluation,
model upload, or frozen-result regeneration was performed.

## Confirmed software corrections

### Duplicate result identities

The verifier previously accepted duplicate records for the same experiment and
canonical seed as two successful experiments. This occurred both within one
trainer log and across JSON/log copies. All observations sharing a duplicated
identity are now `UNVERIFIED`; no arbitrary first observation wins. Distinct
seeds remain distinct even with an explicitly seed-independent reference.

### Single-seed uncertainty and seed ingestion

One score retains its descriptive mean but cannot estimate its across-run
standard error. The bootstrap helper now returns NaN interval bounds for one
score, and standard error returns NaN instead of zero. Tables, CLI output, and
plot legends explicitly say uncertainty is not estimable; a single-seed curve
has no confidence band. CLI exports include an `uncertainty_status` column.

This deliberately changes the old zero-width-interval behavior. Consumers must
handle unavailable uncertainty rather than treating it as zero. Having two or
more values does not itself establish independence or adequate statistical power.
See the [SciPy standard-error contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html).

Malformed `seed_*` entries are now errors rather than silently omitted data.
Seed names must contain a nonnegative decimal integer; canonical aliases still
trigger the existing duplicate-source guard. Empty/missing sources remain errors.
Bootstrap resample counts must be positive integers, not booleans.

### Synthetic OpenRLHF wrapper

The synthetic reward and loss streams now depend on the configured seed.
Repeated identical configurations reproduce the same synthetic traces; different
seeds change them. The wrapper raises rather than silently downgrading an
available real runtime to synthetic execution. Step exceptions propagate rather
than returning a partial trace as successful completion.

These are plumbing checks, not estimates of real model reward or comparisons
between frameworks. Canonical native `max_steps` remains unsupported.

### Test isolation

Mocked online-tracking tests now select their intended fake online mode explicitly.
An inherited `WANDB_MODE=offline` no longer prevents their downstream guards from
being exercised. Provider implementations remain mocked. All 122 tests in that
module pass under an externally supplied offline setting.

## Validation checkpoint

- Baseline CPU suite: **1,413 passed, one expected skip, two LaTeX tests
  deselected; 84.74% configured coverage**.
- Changed-tree CPU suite: **1,451 passed, one expected skip, two LaTeX tests
  deselected; 84.76% configured coverage**, in 131.75 seconds.
- Focused result-identity/OpenRLHF tests: **92 passed**.
- Focused statistics/analysis tests: **131 passed**.
- Repository Ruff lint/format and configured 20-file mypy checks: passed.
- Rebuilt wheel and nine-module content verification: passed.
- All 237 stdlib public-release checks, current selected thesis arithmetic, and
  the seven-test offline demo: passed.

The CPU run used `pytest tests/ -q -m 'not latex' --cov
--cov-config=.coveragerc --cov-report=term`, with CUDA hidden, Hugging Face offline,
and W&B offline. This passes the existing 77% coverage gate; it is not a full
native-document validation result.

## Native-document repair

The installed `pdflatex` executable lacks `pdflatex.fmt`, so its first compilation
fails before processing the paper. The CI-pinned Tectonic 0.17.0 executable was
installed and checked against SHA-256
`8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7`.
Its official default bundle request returned HTTP 403 at
`https://relay.fullyjustified.net/default_bundle_v33.tar.index.gz`.

No access restriction was bypassed. Further inspection found the official TeX
Live source packages already installed, but their generated filename databases,
format, language configuration, and font maps pointed into an absent system
state directory. A workspace-local configuration now searches those installed
sources directly and supplies a newly generated pdfLaTeX format and font maps.
System files and author PDFs were not changed.

Both native LaTeX integration tests now pass on this VM (**2 passed**, 119.14
seconds), including the real repository audit suite. Workspace-local compiler
wrappers and a developer activation script make the repaired configuration
reusable. The optional Tectonic download remains blocked; it is not needed for
the working pdfLaTeX/BibTeX path. Strict provenance and independent-thesis-review
gaps are unchanged.

## Additional checker and metric-ingestion corrections

Four new regression cases reproduced receipt false-passes: E1 prior-wave
successes could exceed that partition's graded count, E6 errors and successes
could overlap beyond the graded total, and either unused McNemar-count alias
could contradict the checked alias. Minimal guards now reject those cases.
All preserved receipts still pass unchanged; 30 focused thesis-evidence tests pass.

Native TensorBoard parsing now reads a private snapshot of the exact bytes whose
SHA-256 it reports. Previously an A-to-B-to-A source replacement during parsing
could attach A's hash to rewards read from B. A regression demonstrates the
old mismatch; the fixed reader preserves the binding and handles source removal
as unavailable metrics. The real TensorBoard 2.20.0 writer/reader also passed a
two-event snapshot/hash integration check using an ephemeral dependency overlay.
No real training was performed, and logged rollout counts remain separate from
optimizer steps. The 38 focused native metric/command tests pass.

## Complete Linux validation after the repair

The full test suite, now including both native LaTeX tests, passed with
**1,459 passed, one expected non-CPU-device skip, no deselections, and 84.91%
configured coverage**, in 241.82 seconds. Lint, formatting, configured mypy,
wheel build/content checks, lock/export freshness, documentation checks, 237
public-release tests, selected thesis arithmetic, and the seven-test demo passed.
The remaining non-test gates were rerun separately after an approval-review
interruption at the package stage; the expensive full suite was not duplicated.

Both strict provenance checks were explicitly rerun and still reject their
unchanged records: campaign 95.7% (unbound uncertainty scope) and sweep 82.6%
(single seed and uncertainty). A high audit score is not a strict pass. These
remaining scientific gaps are not repaired by the software or TeX installation.

The first source/test checkpoint is local commit `39686a2a`; its publication
range passed a redacted Gitleaks 8.30.1 scan. The first push attempt was blocked
by missing VM GitHub authentication, not by a code or secret-scan failure.
Publication status must be checked against the remote; a local commit is not
proof of a successful push or CI run.

## Harness checkpoint: topology and critic-state continuity

The public OpenRLHF v0.8.4 CLI requires actor GPU count to equal
`vllm_num_engines * vllm_tensor_parallel_size` in synchronous colocated mode.
The wrapper fixes one engine but previously accepted two actor GPUs with
tensor parallelism one, failing only after Ray placement-group initialization.
The wrapper now rejects that mismatch before any provider or filesystem work.
CPU regressions verify supported one-, two-, four-, and eight-GPU command
contracts without allocating GPUs. Upstream source: [v0.8.4 train_ppo_ray.py](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/cli/train_ppo_ray.py),
SHA-256 `32e830931965f5a30cff5e3cf601b06827a6b65feae3570783ef3b2437fa3db0`.

The GRPO critic sidecar already recorded its training step, but resume ignored
it and accepted raw model weights with freshly initialized Adam state. A torn
actor/critic write or rewound receipt could silently mix states. Resume now
requires a model-plus-optimizer sidecar with an integer step exactly matching
the actor receipt, before tracking or provider setup. Unversioned legacy
sidecars are intentionally rejected; restore a matching saved pair or start a
fresh run, rather than relabeling weights or inventing optimizer continuity.
The final critic save now precedes the successful completion receipt, so a
failed sidecar write cannot leave an apparent completed run.

The old positive resume test rewrote a final receipt and did not call the actor
state-restoration API. Its replacement captures and restores a genuine periodic
actor/critic pair, verifies actor optimizer-state restoration is selected, and
checks only the remaining optimizer step runs. Tests also cover a real simulated
interruption between the actor and critic writes, stale/future and malformed
step stamps, missing optimizer state, raw legacy state, final-save failure, and
pretraining RNG replay. These are synthetic CPU integration checks, not a paid
provider replay or evidence of model-quality improvement.

Focused validation: 40 OpenRLHF command/budget tests and 125 GRPO module tests
pass. Public-source Muse reviews independently identified the topology and
critic-state mismatch; their suggestions were checked against source and
reproducible regressions before implementation.

Changed-tree CPU suite: **1,465 passed, one expected skip, two native LaTeX
tests deselected; 84.78% coverage**, 126.43 seconds. Repository lint, format and
configured mypy checks passed. Native TeX source and build code are unchanged
from the fully passing native checkpoint above.
