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

## Harness checkpoint: actor receipts and reward boundaries

A nonzero resume step with no actor optimizer-state path previously selected a
fresh actor while skipping earlier training steps. Resume now rejects that
receipt before tracking or provider setup, along with coerced, negative, and
out-of-budget step values and non-object JSON receipts. Completed receipts
retain their read-only result-return behavior. Invalid receipts are preserved.
Dynamic-sampling resume remains unsupported and now fails before tracking.

The portfolio reward router previously split at the last `</think>` anywhere
in the response. Correct JSON argument values and patch source lines containing
that literal were truncated and scored incorrectly. Structured answers are now
preserved; for prefixed reasoning, the first complete structured suffix is
selected. Excessively nested model JSON also returns a failed reward instead
of propagating a recursion exception into training or evaluation.

Math reward comparison previously converted both answers to binary floats,
letting distinct integers above 2^53 compare equal. Decimal comparison now
preserves the existing strict absolute tolerance of 0.01 without that collapse.
The nonnegative difference uses downward rounding against an exactly
representable threshold, preserving the strict comparison even just below the
boundary; the arithmetic context is independent of callers' decimal settings.
Property checks compare against an exact rational oracle. Final-number parsing
now consumes scientific notation and leading-dot decimals as whole numbers,
not an exponent or fractional digit fragment. Numeric zero targets are no
longer replaced by an empty string. These scoring corrections do not regenerate
or regrade frozen experiment results.

Source for decimal semantics: [Python decimal documentation](https://docs.python.org/3/library/decimal.html).
Focused checks: 212 resume/GRPO tests and 198 precision/reward/GRPO tests passed
in overlapping targeted suites. Public-source Muse review identified the literal
reasoning-delimiter issue; all cases were reproduced locally before changes.

Full `make check` passed on this changed tree: **1,510 passed, one expected
non-CPU skip, no deselections, 84.93% coverage**, 240.15 seconds. This includes
both native LaTeX integration tests, repository lint/format/types, package and
lock verification, redacted secret checks, documentation checks, and the 237-test
public-release suite. The independent strict provenance gap remains unchanged.

## Adapter cardinality, finite rewards, and credential-safe native tracking

Training and critic-pretraining groups now require exactly the requested number
of sampled responses. Empty, short, and oversized groups fail before an actor
or critic update instead of silently changing the estimator's group size.
Adapter scores are converted to finite numbers at that boundary. The existing
normalizers already rejected nonfinite ordinary training rewards, but critic
pretraining did not pass through those normalizers; the shared boundary now
protects that path too.

Held-out evaluation now treats nonfinite scores and unexpected sampler response
counts as failed cases, scored zero while retained in the original denominator.
A one-example NaN previously bypassed the multi-example bootstrap guard and
could appear in a successful receipt. The standalone summary also rejects
nonfinite inputs for every sample count. Existing all-finite uncertainty
calculation is unchanged; its result includes failure-as-zero cases under the
existing evaluation policy, not a success-only denominator.

The native OpenRLHF W&B command previously put the API key in `--use_wandb`.
Official v0.8.4 logs that argument namespace to stdout and W&B configuration.
The flag now carries only a non-secret selector, while the SDK uses the already
supplied environment credential. No new authentication flow is introduced.
Regressions verify the key is absent from argv and receipts. The exact official
initializer was also exercised locally with all W&B methods mocked: existing
SDK credentials bypass login, and logged configuration contains only the
selector. No actual credential was read or disclosed in this validation.

Verified upstream [ppo_trainer.py at v0.8.4](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/trainer/ppo_trainer.py),
SHA-256 `2746e6da8e60ed6c5885bd0d4fa0190e4f358d6aecf8d4c11b7f5003b62673c9`.
The installed W&B SDK's API-key resolver gives its environment key precedence.
Focused checks: 246 adapter/GRPO/native-command tests passed. Test subprocess
mocking was narrowed to the provider export command so unrelated local font
inspection is not broken by test ordering.

Changed-tree CPU validation: **1,529 passed, one expected skip, two unchanged
native LaTeX tests deselected; 84.80% coverage**, 131.82 seconds. Repository
lint/format/type checks, wheel/lock validation, documentation/public-release
checks, selected thesis arithmetic, and offline demo all passed. The immediately
preceding full native checkpoint remains recorded above.

## Policy-loss numerical stability

A synthetic GSPO case with new log-probability -1 and behavior log-probability
-1000 produced a finite clipped objective but NaN gradients: the unused
exponential overflowed, and backward multiplied zero by infinity. The loss now
selects the mathematically active clipped branch in log-ratio space before
exponentiation. Positive clipping and zero advantages remain finite with zero
gradient in that case. Negative-advantage overflow is not incorrectly clipped
into a finite objective: it raises before an optimizer update. Behavior-policy
reference tensors are detached and moved to the current log-probability device.
No GPU execution was used to validate this change.

A float32/float64 oracle compares both values and gradients to the direct finite
objective for positive, negative, and zero advantages in every clipping region.
Deep-underflow cases, the disabled lower-bound convention for epsilon >= 1,
nonfinite inputs, reference detachment, and invalid advantage cardinality are
also covered. GRPO now rejects mismatched advantage counts rather than silently
using a prefix, and both kernels reject nonfinite objectives. GSPO clipping
configuration rejects nonfinite values before provider setup. Failed GSPO calls
do not populate the optional reference stash.

The fake-runtime integration uses the actual loss closure and a deliberately
stale sampler: an unbounded negative-advantage objective stops before backward
or any optimizer call, with no success receipt. This is an arithmetic and
failure-handling correction, not measured evidence of improved training reward.
Public-source Muse analysis helped check the signed clipping derivation; its
suggestion to allow nonfinite loss propagation was not adopted.

Full native `make check` on the policy-loss batch passed: **1,554 tests passed,
one expected non-CPU skip, no deselections; 84.94% coverage**, 251.55 seconds.
Both native LaTeX tests and all aggregate quality/package/public gates passed.

## Interrupted-checkpoint recovery

Matched-step validation detects a torn actor/critic pair but cannot recover the
previous pair if either component has already been overwritten. New critic
writes are therefore immutable, content-addressed `critic-<sha256>.pt` siblings.
The writer fully writes and fsyncs a private temporary, then atomically creates
the immutable filename without overwriting an existing artifact. The JSON
receipt, containing `critic_state_sha256`, is committed last. An interruption
before that commit leaves the preceding receipt and its sidecar intact; orphan
sidecars are ignored rather than selected by time or filename order.

Restore derives the filename from a validated digest, verifies a single byte
snapshot, and loads that same snapshot with `weights_only=True`. It checks the
saved step and configuration/seed binding before restoring weights or Adam.
Malformed bindings cannot fall back to legacy paths. Corrupt, symbolic-link,
directory, and FIFO inputs are rejected before deserialization, and corrupted
content-addressed artifacts are never overwritten. No older sidecars or
receipts are automatically migrated or deleted. Disk retention follows the
existing checkpoint cadence, with no new automatic garbage collection.

Matching regular legacy sidecars remain readable with the earlier step and
optimizer checks; new writes do not maintain a legacy flat-file copy, so older
software cannot resume the new format. Completed result receipts retain their
existing read-only return behavior. The guarantee covers process interruption
of these local files, not a cross-provider transaction or a power-loss guarantee
for the entire filesystem.

Fault-injection integrations interrupt both critic publication and JSON receipt
publication, then verify the next run restores the preceding actor/critic pair
and performs only the remaining optimizer step. Other tests verify byte
corruption rejection before `torch.load`, source replacement after snapshot,
configuration/seed mismatch, legacy loading, immutable-file collision handling,
and temporary-file cleanup. All 206 focused checkpoint/GRPO tests passed;
repository lint, formatting and configured type checks also passed.

Aggregate CPU validation: **1,568 passed, 1 skipped, 2 native-LaTeX tests
intentionally deselected**, 84.93% coverage, in 131.60 seconds. Packaging,
lock consistency, documentation, public-demo, and submission checks passed.
Native-LaTeX checks had passed in the preceding full validation; this batch
changes only checkpoint handling and its regressions.

## Configuration fidelity and early budget validation

Two bounded Muse reviews used only published source at `f8371df1` and the
anonymously retrieved official OpenRLHF v0.8.4 source listed above. Their
configuration findings were independently reproduced and narrowed to the
supported contracts; no unpublished working files were sent for review.

GRPO now rejects nonintegral or Boolean counts, invalid budget bounds, and
nonfinite/non-numeric arithmetic settings at construction and again before
execution. The second check covers the existing mutable dataclass API. Invalid
zero seed counts can no longer silently return an empty successful run, and
fractional loop budgets cannot consume provider setup before failing at
`range`. Zero-step runs, single-response groups, rank zero, temperature zero,
frozen critics, zero pretraining/resampling/auxiliary weights, and default
checkpoint cadence remain supported. This does not claim model-specific
provider capability validation or impose new speculative parameter bounds.

Native OpenRLHF now rejects non-default overrides to six historically ignored
fields: `num_workers`, `num_actors`, `actor_num_gpus`, `critic_num_gpus`,
`eval_interval`, and `save_interval`. Old default configurations still load.
The native topology is controlled by `num_gpus` and `tensor_parallel_size`;
the legacy defaults themselves have no operational placement/interval meaning.
Both `save_steps=-1` and `eval_steps=-1` are explicit. The pinned upstream
trainer disables periodic work for these sentinels, while its CLI performs a
separate final actor-model save after training. No interval units, resume
support, or distributed capabilities are invented by mapping these old knobs.
Regression checks prove overrides fail before runtime validation, dataset
preparation, subprocess launch, or output-directory creation.

Validation: **1,576 passed, 1 skipped**, no native-LaTeX deselections,
85.09% coverage, in 254.71 seconds. Full `make check submission-check` passed,
including both native TeX tests, lint, formatting, configured type checks,
packaging, lock consistency, secret scan, docs, and public/demo checks.

## Half-precision policy-loss arithmetic

A public-source Muse review of `aefc66a3` identified precision risks. CPU
reproduction confirmed two: float16 token sums overflow on a finite GRPO
objective, and converting behavior log probabilities to float16/bfloat16 can
erase a policy delta larger than GSPO's configured clip window. For example,
new `-5.0` and behavior `-5.001` incorrectly took the unclipped positive-advantage
gradient; the float32 oracle correctly produced zero gradient. The review's
example had a sign error, which was corrected before writing the regression.

Half-precision inputs now use float32 arithmetic before subtraction, reductions,
and exponentiation. Existing float32/float64 arithmetic stays unchanged, and
autograd still returns gradients to the original input dtype. This cannot
recover precision already lost when a model produced its new log probabilities.
The exact signed GSPO objective and clipping windows are unchanged.

Promotion also checks the analytical per-token gradient range before accepting
an update: a finite float32 objective must not create an infinite gradient when
cast back to float16. A large unclipped ratio is accepted when averaging across
enough tokens makes its gradients representable, rather than being incorrectly
clipped or rejected solely because the loss exceeds the float16 range. Tests
compare values and gradients against float32 references, including auxiliary
NLL, clipped/unclipped signs, first epochs, and response-only prompt masking.
These are CPU kernel regressions; no GPU or provider training was performed.

Aggregate validation: **1,584 passed, 1 skipped, 2 native-LaTeX tests
intentionally deselected**, 84.99% coverage, in 133.19 seconds. Lint,
formatting, configured type checks, packaging, lock consistency, documentation,
public demo, and submission checks passed. The preceding configuration batch
passed both native-LaTeX tests; this batch changes only policy-loss kernels
and their CPU regressions.
