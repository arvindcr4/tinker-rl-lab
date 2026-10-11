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

## Descriptor-pinned checkpoint reads

A further public-source Muse checkpoint review identified a path-check/read
race. A local, synthetic fault injection confirmed that replacing a legacy
sidecar with a symlink between `is_file()` and `read_bytes()` could redirect
the bytes selected for restore. Loading and immutable-artifact collision
checks now open with `O_NOFOLLOW | O_NONBLOCK`, validate the opened descriptor
with `fstat`, and read that same regular-file descriptor. Replacement after
open cannot redirect the snapshot; replacement before open is rejected.
Nonblocking open prevents a raced-in FIFO from hanging the loader. The reader
fails closed on platforms lacking these flags rather than claiming the same
guarantee there; this Linux VM supports both.

Saving without optimizer state now fails before artifact publication, matching
the existing restore contract. Production callers already supply Adam state.
Legacy artifacts remain explicitly unbound compatibility inputs with strict
step/optimizer checks, not cryptographically authenticated history. No receipt
migration, artifact deletion, or guessed recovery from orphan files was added.
The review's suggestion to reject every legacy receipt when any newer artifact
exists was not adopted: a legitimate interrupted new save can leave such an
orphan while the older committed receipt is still the correct recovery point.

Validation: 210 focused tests passed, then **1,588 passed, 1 skipped,
2 native-LaTeX tests intentionally deselected**, 85.00% coverage, in
134.97 seconds. Lint, formatting, configured type checks, packaging,
lock consistency, docs, public demo and submission checks passed.

### Shared cast-back path for auxiliary gradients

An additional adversarial CPU check found that separate promoted casts for
GRPO and NLL could each overflow on cast-back before cancelling at a float16
leaf, despite a finite total derivative. The terms now reuse one promoted
response tensor, so their gradients combine in float32 before the single
cast-back. A `-70000` advantage and `70000` one-token auxiliary coefficient
now produce the expected exact zero loss and zero gradient rather than NaN.
This strengthens the previous half-precision validation; it does not change
normal float32/float64 computations or the loss definition.

The shared-cast regression and aggregate checks passed: **1,589 passed,
1 skipped, 2 native-LaTeX tests intentionally deselected**, 85.00% coverage,
in 132.31 seconds. All static, package, lock, docs, demo and submission gates
also passed. Native TeX was unchanged from the preceding full validation.

## Structured-answer boundaries in exact math scoring

A Muse review of public source at `8d457dd0`, followed by local reproduction,
confirmed that the held-out exact-math scorer awarded full credit to a fraction
denominator, exponent, identifier suffix, or incomplete box. For target `2`,
examples included `\\boxed{1/2}`, `10^2`, `\\boxed{5} for Q2`, and
`\\boxed{2`. Malformed thousands grouping also turned `12,34` into `1234`.

The exact scorer now scans balanced boxes iteratively and only considers a
bare final numeric token after the last boxed span. An incomplete trailing box
cannot leak its numbers, and a wrong final box cannot be rescued by a number
from earlier reasoning. Numeric token boundaries reject adjacent identifier
characters and expression/container operators; grouped commas are normalized
only when their thousands grouping is valid. Nested boxes are retained whole,
with redundant scalar grouping braces handled without recursive parsing.
Ordinary signed decimals, scientific notation, valid thousands grouping,
parenthesized scalars, and inline/display TeX numeric wrappers remain accepted.

The existing distinction is preserved: a wrong box followed by a later bare
correct answer is accepted by `ExactMathReward`, while `MathReward` still gives
its existing partial reward for the wrong box. Decimal tolerance is unchanged;
no symbolic algebra, fraction evaluation, historical re-scoring, or new
benchmark result claim is introduced. Regression cases include deeply nested
and repeated boxes, malformed expressions, identifier digits, and the earlier
wrong-final-box policy. All 215 focused tests and static checks passed.

Full validation passed: **1,597 passed, 1 skipped**, no native-LaTeX
deselections, 85.30% coverage, in 252.25 seconds. `make check submission-check`
passed all static, packaging, locked-dependency, secret, native TeX, docs,
public/demo and submission gates. Frozen benchmark artifacts were unchanged.

## Fail-closed MIN-REPORT metadata assertions

A bounded public-source Muse review of `73995148` and independent local
reproduction found false-PASS cases in the provenance verifier: strings such as
`"false"` satisfied truthiness checks, and a fractional seed count or repeated
seed identifiers could pass the multi-seed check. Malformed JSON section types
could instead crash before producing a useful failure report.

Verification now requires explicit Boolean assertions, an integral count and
matching distinct integral seed identities, and expected top-level metadata
shapes. Malformed records produce failed checks without mutating inputs.
Generator defaults no longer invent held-out disjointness or verified
decontamination; a missing named config fails instead of silently falling back
to empty settings. A CLI integration proves truthy-false uncertainty still
causes strict failure and an audit receipt rather than a traceback.

Documentation now distinguishes declared metadata from independently verified
results, explains that only the live code hash is recomputed here, and corrects
the outdated campaign 100% description. The unchanged signed campaign remains
**95.7%, grade A, strict FAIL** because `rigor.reports_ci` does not explicitly
assert uncertainty reporting. Its literal `results.sd` is not sufficient to
verify scope or provenance. The unchanged sweep remains **82.6%, grade B,
strict FAIL** for a single seed and unasserted uncertainty. No signed record,
measured result, key, signature, or strict-gate threshold was changed.

All 10 focused provenance/API/CLI tests and repository static checks passed.

Aggregate validation passed: **1,603 passed, 1 skipped, 2 native-LaTeX tests
intentionally deselected**, 85.17% coverage, in 139.39 seconds. Packaging,
lock consistency, docs, public/demo and submission gates passed. The preceding
full run passed native TeX; this batch changes metadata validation and its
scope documentation, not thesis source or measured artifacts.

## Native numeric preflight after configuration mutation

CPU reproduction showed that NaN learning rates, Adam coefficients,
temperatures, and clipping values could reach the OpenRLHF launch command.
A public-source Muse review of `fb6d93c9` independently identified nonfinite
floats and assignment-time bypasses of Pydantic construction bounds.

The native command now rejects nonfinite/non-numeric live float settings,
invalid Adam coefficient/learning-rate/weight-decay bounds, and nonintegral
live counts before runtime validation, data preparation, or subprocess launch.
Budget planning rechecks the fixed source's 500-row upper bound even after
mutation. Zero learning rate, temperature, clipping width, weight decay, and
LoRA rank remain supported. Existing Pydantic construction coercion is retained;
this is a launch-time check of the normalized, possibly mutated configuration.
No unverified ranges were imposed on gamma, lambda, or KL coefficients.

The review's proposal to require exactly 500 training rows was rejected:
`max_samples` is deliberately a configurable subset budget, with whole-batch
arithmetic and an upper bound of 500. Requiring equality would contradict that
existing supported contract. All 46 focused native-command/budget tests passed,
including synthetic failure-before-provider checks and preserved zero modes.

Aggregate validation: **1,606 passed, 1 skipped, 2 native-LaTeX tests
intentionally deselected**, 85.17% coverage, in 134.94 seconds. Static,
packaging, lock, docs, public/demo and submission checks passed. No native
provider, GPU, or paid training was launched.

### Parser punctuation and metadata-schema compatibility follow-up

A public-source follow-up Muse review of `fb6d93c9` identified ratio/time and
comparison tails that still looked like bare numeric answers, plus ordinary
parenthesized answers rejected when followed by a sentence period. Independent
reproduction confirmed both. The exact scorer now rejects adjacent ratio,
comparison, and operator components while preserving label colons, assignment
answers, percentage units, and nested numeric wrappers with sentence punctuation.
Wrapper stripping uses indices and remains nonrecursive even with 10,000 pairs.

Cross-checking the audit's accepted shapes against the registry schema also
found that canonical backend/telemetry objects must coexist with the frozen
records' compact string/list forms. Both forms are now explicitly supported;
truthy scalar assertions and malformed section types remain rejected. No
frozen record changes or new scientific claims were needed. The focused math
suite passed all 217 tests before the combined compatibility validation.

Combined compatibility validation passed: **1,609 passed, 1 skipped,
2 native-LaTeX tests intentionally deselected**, 85.21% coverage, in
129.95 seconds. All static, package, lock, docs, public/demo and submission
checks passed. The prior full validation passed native TeX.

## Validated chat-message inputs and native artifact binding

A public-source Muse review of `16af6867` identified unchecked empty/malformed
labels, assumed source row counts, and missing input-byte identity in native
success receipts. The fixed GSM8K source is now validated before any input file
is written: exactly 500 rows, nonempty questions, an explicit final-answer
delimiter, and complete finite numeric labels with valid thousands grouping.
An existing artifact is never overwritten. Smaller configured `max_samples`
budgets remain supported; the corpus itself still contains 500 source rows.

The adapter now emits system/user message objects and enables the pinned
OpenRLHF `--apply_chat_template` path. The previous handwritten `<|system|>`
headers did not match the default Qwen model's actual chat template. Native
rendering now delegates to that model's tokenizer rather than inventing role
markers. Models used through this driver must provide a compatible chat
template; no silent plain-text fallback is introduced.

Primary-source checks, retrieved anonymously with normal TLS:

- OpenRLHF v0.8.4 `openrlhf/datasets/prompts_dataset.py`:
  `https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/v0.8.4/openrlhf/datasets/prompts_dataset.py`
  SHA-256 `97530244a440d0297313f7bfd696e827ddb662f9cbda954d004076307967d82c`
- Official Qwen2.5-1.5B-Instruct tokenizer configuration:
  `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer_config.json`
  SHA-256 `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`

A CPU-only check executed the inspected upstream preprocessing function with
its exact verified source hash and rendered a synthetic system/user pair using
the official Qwen template in a sandboxed Jinja environment. It preserved the
label and produced Qwen's system/user/assistant generation boundaries. This
used no model weights, GPU, provider request, or private dataset.

Before launch, one regular-file descriptor snapshot is parsed and SHA-256
hashed. Malformed/duplicate JSON fields, raw legacy prompt strings, invalid
labels, nonregular files, and wrong row counts fail before subprocess launch.
The result receipt records exact input bytes, row count and schema, and a
post-run hash check refuses success if input bytes changed. These are validated
launch-input messages before tokenizer rendering, not proof of model consumption
or learning quality. The upstream dataset revision remains explicitly unpinned;
`source_revision` is null rather than fabricated. Historical measurements and
framework-confound limitations are unchanged.

Full VM validation passed: **1,618 passed, 1 skipped**, including both native
LaTeX tests, with 85.34% configured supported-surface coverage in 261.27 seconds.
All static, package, lock, docs, public/demo and submission checks passed.
Native adapter tests are outside the configured coverage denominator; this
percentage is not a claim of whole-repository coverage.
