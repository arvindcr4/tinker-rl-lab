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

## Fixed configuration snapshots across long-running work

CPU fault injection demonstrated that changing the caller's native config
inside the subprocess callback launched seed 17 but recorded seed 99. Native
execution now deep-copies the complete nested config before validation so the
command, budget, metrics selection and final receipt describe one snapshot.
A regression mutates seed, optimizer, algorithm, model, data, budget and logging
settings during execution and verifies the receipt retains the launch settings.

A separate multi-seed regression failed before the fix with seeds
`[10, 101, 102]` instead of `[10, 11, 12]` when a caller callback changed the
original config during the first seed. `run_grpo` now constructs one independent
experiment snapshot before iteration; each seed still receives its own copy.
The config constructor normalizes sequence metadata to immutable tuples.
Caller objects remain mutable and their changes are not overwritten. This is
snapshot isolation from later edits, not a claim of atomic copying during a
simultaneous mutation. Both focused config suites passed all 40 tests.

A bounded Muse review of public `8e27a44a` independently identified both live
config issues. Its concern about a missing dataclass decorator was an artifact
of the excerpt and was rejected against the full source. The seed count is
already fixed by Python's range evaluation; the reproduced drift concerned
later seed identities and hyperparameters, not a changing iteration count.
Full validation passed: **1,620 passed, 1 skipped**, including native TeX,
85.34% configured supported-surface coverage, in 253.31 seconds. Static,
package, lock, docs, public/demo and submission checks all passed.

## Native event-file boundaries and real TensorBoard integration

A bounded Muse review of public `8e27a44a` flagged check-then-read file races
and event-writer ambiguity. Two regression tests failed before the fix: replacing
the event file with a symlink after its type check still produced observed
metrics, and adding a second writer during parsing was silently accepted.

Input and metric artifacts now share a descriptor-pinned no-follow/nonblocking
regular-file reader. FIFO replacement fails without waiting for a writer; a
symlink is never followed. Event membership is rechecked after parsing and
parsed-byte hashing remains intact. Optional third-party parser exceptions now
produce `not_collected` with no invented trace; process interruptions still
propagate. The native writer must be quiescent before collection. These checks
cover changes observed during collection, not future modifications, malicious
parent-directory replacement, or proof of model execution. All 28 focused input
and metric boundary checks passed.

A separate workspace-local environment installed official PyPI
`tensorboard==2.20.0` for an actual CPU writer/reader integration. No repository
lock or production dependency was changed. All four synthetic event cases
passed: a complete finite trace was recovered; duplicate steps, NaN and a
missing rollout were rejected. The reproducible integration is now included as
`tests/test_openrlhf_tensorboard_integration.py`; it skips explicitly when the
optional TensorBoard package is absent. A dependency-enabled invocation is:

```sh
uv run --with tensorboard==2.20.0 pytest tests/test_openrlhf_tensorboard_integration.py -q --no-cov
```

The actual VM dependency-enabled run passed **4 tests**. These are synthetic
local event files, not measured training gains or held-out scores.

Full default-environment validation passed: **1,626 passed, 2 skipped**,
including both native TeX tests, with 85.34% configured supported-surface
coverage in 270.31 seconds. One additional skip is the optional TensorBoard
integration module; its four dependency-enabled tests passed separately above.
Static, package, lock, docs, public/demo and submission checks passed. Native
module lint was checked explicitly in addition to the standard repository gate.

## Credential-safe native URL and failure boundaries

A synthetic-only reproduction showed URL userinfo reaching `--remote_rm_url`.
The pinned OpenRLHF v0.8.4 entrypoint prints its argparse namespace
(`train_ppo_ray.py`, line 25 in the previously hash-verified source), so this
was a credential-disclosure path. A bounded Muse review of public `e8581dbb`
confirmed the issue and identified malformed host/port/control handling.
Native preflight now rejects userinfo, missing hosts, invalid ports, literal
whitespace/control characters, backslashes and fragments before runtime or
filesystem work. Normal HTTP(S), IPv6 and nonsecret query parameters retain
their exact configured URLs. Diagnostics do not echo rejected URL contents.
This does not promise to detect arbitrary secrets hidden in paths or query
values: reward-service URLs must still be credential-free because they are
logged by the native framework.

A second synthetic-only reproduction showed the Tinker environment credential
was absent from the shared error redactor's known credential list. It is now
included; longer overlapping values are removed first so masking a short token
cannot expose the suffix of another. No actual credentials were inspected or
transmitted for these tests. The focused suites passed **116 tests**.

Full validation passed: **1,642 passed, 2 skipped**, including native TeX,
85.34% configured supported-surface coverage, in 254.95 seconds. The optional
TensorBoard module remains absent in the default environment; its separate
four-case real integration passed in the preceding batch. Static, package,
lock, docs, public/demo and submission checks passed.

## Immutable public GSM8K source revision for future native inputs

The native loader now explicitly requests public `openai/gsm8k` revision
`740312add88f781978c0658806c59bc2815b9866`, configuration `main`, split
`train[:500]`, with `token=False`. It no longer follows moving `main` or uses
implicit authentication for this public, ungated dataset. New input receipts
record the configured revision alongside the actual launch-input byte hash.
This changes future source selection only; no historical result, dataset or
frozen artifact is rewritten. Model/tokenizer revision pinning remains separate.

Primary-source verification on 2026-10-11:

- Official public metadata: `https://huggingface.co/api/datasets/openai/gsm8k`
- Revision tree: `https://huggingface.co/api/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866/main`
- Training file: `https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet`
- Training file size: 2,306,545 bytes; SHA-256:
  `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d`

The file was fetched anonymously with normal TLS and matched the official
revision's LFS digest. PyArrow confirmed 7,473 training rows. All first 500
actual public rows passed the native formatter and validator, producing JSONL
SHA-256 `4416d0abbb8f31205160171db451cc7e76854d3d0d07ab9a79aaecab78cce795`.
That check used the verified local Parquet rows with the dataset loader mocked;
it is distinct from an end-to-end Hub loader check. Raw dataset files remain
outside the repository. All **57 focused native input/command tests** passed.

A separate live Hub-loader attempt did not complete observably in this VM and
is **not** counted as a passed integration. The first attempt identified missing
HTTPX SOCKS support; the documented optional `socksio==1.0.0` dependency was
installed in a separate workspace directory without changing proxy settings.
The retried run produced no completion receipt. The verified Parquet/formatter
check and mocked revision-argument regressions above are the available evidence.

Full regression validation passed: **1,642 passed, 2 skipped**, including native
TeX, 85.34% configured supported-surface coverage, in 267.54 seconds. Static,
package, lock, docs, public/demo and submission checks passed. The test count
is unchanged: existing input-contract tests now require the immutable revision
and explicit anonymous loading.

## Stable advantages when finite reward moments exceed binary64

CPU reproductions found that group and global normalization raised overflow for
finite rewards `[1e200, -1e200]` and `[1e308, 9e307]`, despite representable
standardized answers. Squared deviations can also underflow to zero, and a
centered value or `std + epsilon` may overflow even when their ratio is finite.
Fourteen new adversarial regressions failed before the fix; the ordinary-input
bit-identity regression already passed.

Ordinary binary64 arithmetic remains unchanged. Only failed/nonfinite moments,
underflowed nonzero variance, or nonfinite outputs enter a high-precision
Decimal fallback. Its private context retains tiny contributions between
cancelling extreme finite floats and is independent of the caller's Decimal
settings. Epsilon placement, tied-group zeros, global exclusion basis and
unbiased centering semantics are preserved. Genuinely unrepresentable output
advantages raise a clear `ValueError`; they are never clamped or propagated as
NaN/Infinity into the update path.

A Muse review of public `a105cb6f` confirmed the intermediate-overflow causes.
Its suggestion to propagate nonfinite centered values was rejected: a centered
value may exceed binary64 while the standardized result remains representable.
The regression using opposite maximum floats covers that distinction. A
high-precision fallback was chosen instead of a hand-written scaled arithmetic
path with additional cancellation/division corner cases.

All **169 focused normalization, advantage-property and GRPO module tests**
passed. This includes exact equality with the previous ordinary-input floating
arithmetic over 50 deterministic vectors, exclusion cases, genuine output
nonrepresentability, subnormal-scale variance, and preserved tiny terms after
extreme cancellation. Reward extraction/scoring policy remains unchanged.

Full validation passed: **1,657 passed, 2 skipped**, including native TeX,
85.42% configured supported-surface coverage, in 251.93 seconds. All static,
package, lock, docs, public/demo and submission checks passed. A small local
smoke timing measured about 2.6 microseconds per ordinary four-reward call and
0.91 milliseconds per exceptional two-reward fallback call in this VM; this
is a local diagnostic, not a general performance claim.

## Reject valid-looking but incorrect native launch corpora

The public-source Muse follow-up on `50370294` identified a remaining provenance
gap: recording a newly computed hash accepted any well-formed 500-row corpus,
including a stale/reordered/substituted one. Native launch now compares that
already-computed hash to the independently verified fixed-profile reference
`4416d0abbb8f31205160171db451cc7e76854d3d0d07ab9a79aaecab78cce795`
before subprocess execution. The success input receipt binds the matching
reference digest as well as the actual digest. Changing source, row order,
system prompt or byte formatting now requires a deliberate reference update;
there is no native bypass flag.

The production gate accepted all 500 rows from the previously verified public
Parquet/formatter artifact with no mocking of its reference constant. Offline
regressions use an explicitly synthetic reference to prove that reordered rows,
a substituted valid numeric label, changed prompt and whitespace-reserialized
JSON all remain schema-valid but fail before launch. A positive byte-identical
control passes, while an ordinary synthetic corpus is rejected by the real
production constant. Existing subprocess-only tests explicitly substitute a
synthetic reference; they do not pretend to contain the official dataset.

All **75 focused input, command and budget tests** passed. This is a launch-byte
identity control, not attestation that a tokenizer or model consumed those
bytes; the previously documented live Hub-loader limitation remains.

Full validation passed: **1,663 passed, 2 skipped**, including native TeX,
85.42% configured supported-surface coverage, in 260.02 seconds. Static,
package, lock, docs, public/demo and submission checks passed. The preceding
source-pin CI completed all engineering jobs successfully; its environment
job's longer runtime was dependency installation, not a failing test.

## Transport-free integration through the actual locked Tinker SDK

The project environment contains Tinker SDK **0.24.1**, distinct from the
separately installed CLI tools. Its official public wheel was anonymously
retrieved from PyPI and verified against the published SHA-256:
`67a6564b2563f970e36f870292eabbbe698b539e84c04ee96cec0ac6af2206a5`
(250,296 bytes; metadata: `https://pypi.org/pypi/tinker/0.24.1/json`). Seven
installed implementation/schema files, including the fast sampling response
classes, matched that wheel byte-for-byte.

New CPU integration checks execute the SDK's actual
`TrainingClient.forward_backward_custom_async` implementation and its gradient
conversion/metrics combination. Only the two transport methods are replaced
with synthetic local fixtures; no `ServiceClient` is constructed, and socket
connections are explicitly blocked during these tests. This is stronger than
replacing the entire SDK with hand-written type stubs, but is not a live
provider/backend or training-quality test.

All **7 focused integration cases** passed. They verify target alignment,
response-only gradients, the SDK's negative-gradient-to-CE-weight convention,
GRPO auxiliary NLL, GSPO unclipped and clipped zero gradients, empty-response
zero gradients, and rejection of nonfinite forward output before backward
transport. The sampling case uses the real request model and NumPy-backed
response classes to check explicit seed/temperature/top-p/token-cap forwarding
and stop/logprob metadata. It proves argument construction, not that a remote
sampler honors those arguments.

A bounded Muse review of the public project loss callbacks and verified public
SDK bridge found no mismatch in those contracts. The integration regressions
make those conclusions executable. No SDK upgrade, model download, provider
request, credential inspection or new learning claim was involved.

Full validation passed: **1,670 passed, 2 skipped**, including native TeX,
85.44% configured supported-surface coverage, in 252.96 seconds. All seven SDK
integration cases ran in the default locked environment. Static, package,
lock, docs, public/demo and submission checks passed; the persisted validation
exit status is zero.

## Finite reward summaries and strict resume JSON (18:59 UTC)

A transport-free CPU run with valid finite custom rewards `1e308` and `9e307`
completed an optimizer step but recorded `Infinity` in its reward trace,
summary and completion receipt. The bootstrap utility likewise returned an
infinite mean and NaN bounds. Reward means and bootstrap resample means now
retain their historical Python/NumPy arithmetic whenever finite, and use a
private high-precision Decimal fallback only when an intermediate sum
fails. Percentiles retain NumPy's ordinary result, with a bounded endpoint
interpolation fallback for overflowing differences. Bootstrap draws, ordering,
confidence convention, and the single-observation uncertainty policy are
unchanged. This does not change the math scorer or regrade historical results.

Regressions exercise ordinary and dynamic two-step training, held-out means,
strictly finite logs and checkpoint JSON, extreme positive/negative means,
opposite-sign percentile interpolation, maximum binary64 values, tiny terms
between cancelling extremes, caller Decimal-context independence, and exact
ordinary bootstrap/RNG equivalence. An additional deterministic 100-case
mixed-scale numerical smoke check passed with a two-ulp endpoint tolerance;
the tolerance preserves existing NumPy rounding rather than introducing a
clamp. A completed Muse review of the public pre-fix functions identified the
same intermediate-overflow and RNG-preservation concerns; two earlier bounded
attempts produced no completed review and are not counted as evidence.

A separate reproduction showed that an existing completion receipt containing
Infinity was reused as successful. Resume parsing now rejects duplicate keys,
nonstandard nonfinite constants and overflowing JSON numbers before tracking
or provider initialization. Checkpoint serialization refuses nonfinite values
before replacing a valid prior receipt. Six new receipt regressions failed
before these guards and pass afterward; invalid files remain untouched.

The combined focused suite passed **86 tests**. The final full run passed
**1,685 tests, 2 skipped**, including native TeX, with **85.52%** configured
supported-surface coverage in 252.17 seconds. Static, package, lock, docs,
public/demo and submission checks passed; persisted aggregate exit status is
zero. An earlier aggregate run exposed two incorrect test assertions about
where the completed receipt nests its trace; those assertions were corrected
and the complete aggregate suite was rerun successfully. No provider job,
model download, new result claim or frozen-artifact rewrite was involved.

## Async loss-input binding (19:06 UTC)

A CPU reproduction showed that mutating a response-length list after constructing
its loss callback reintroduced prompt-token gradients. Mutating nested behavior
logprobs changed an otherwise unclipped GSPO update into a clipped zero-gradient
update. These callbacks are invoked after the SDK awaits its forward request,
so the factory now snapshots masks, response lengths and behavior values.
Tensor behavior values are detached and cloned; nested lists and NumPy arrays
are copied without premature dtype conversion. The `stash` list remains shared
output and keeps its existing first-call, detached response-only semantics.

All eight new aliasing regressions failed before the change and pass afterward.
They cover both loss factories, NumPy masks, nested lists/arrays, leaf and
non-leaf tensors, float64 precision, outer-list mutation and stash ownership.
Two additional transport-free tests run the actual locked Tinker SDK bridge,
mutate caller inputs inside its awaited forward transport, and verify the
resulting response-only cross-entropy weights. Network connections remain
blocked and no provider client is constructed. A completed Muse review of the
public callback implementations confirmed the aliasing risks; suggested scalar
coercions were not adopted because they could weaken existing shape/type checks.

The focused loss/SDK suites passed (111 cases before the two new SDK-await
cases; all 17 snapshot/SDK cases afterward). Full validation passed **1,695 tests,
2 skipped**, including native TeX, at **85.54%** configured supported-surface
coverage in 259.52 seconds. Static, package, lock, docs, public/demo and submission
checks passed, aggregate exit zero. The learning objective, scorer policy and
historical results are unchanged.

## Resume reward-history consistency (19:13 UTC)

A mocked two-step run resumed from a step-one receipt with an empty reward
trace, performed one optimizer update, and declared completion with only one
recorded reward. Resume now validates trace shape, finite numeric entries,
committed-step coverage, configured budget and counters before tracking,
critic loading or provider initialization. It rejects booleans/numeric strings,
unrepresentable integers, missing committed history, impossible or coerced
counters, contradictory zero-reward counts, malformed nested results and
premature completed receipts. It never pads missing history or rewrites an
invalid input receipt.

Legitimate rewinds retain only the committed prefix, reconstruct its zero-reward
count and discard full-run loss counts that cannot be assigned to that prefix.
Legacy receipts with valid traces but no zero-reward counter reconstruct that
counter from observed history. Legacy Hub URL-only receipts remain supported
when their reward history is complete. Three older synthetic fixtures were
updated to supply the step-one history their test scenarios claimed; their
optimizer-resume, missing-run-ID and legacy-URL assertions remain intact.

The first 13 corruption regressions all failed before the guard; the expanded
focused suite passed **268 tests**. A completed Muse review of the public resume
path supported early history validation. Its suggestions to forbid step-zero
partial receipts or require integer-only, unique checkpoint-export steps were
not adopted: those would reject valid existing receipt forms.

Full validation passed **1,714 tests, 2 skipped**, including native TeX, at
**85.63%** configured supported-surface coverage in 245.72 seconds. Static,
package, lock, docs, public/demo and submission checks passed, aggregate exit
zero. No provider training or historical regrading was performed. GitHub's
engineering checks for the preceding published `13f856ad` commit all completed
successfully; the existing scientific `minreport-audit` gate remains red.

## Validate training loss before optimizer updates (19:20 UTC)

A mocked forward/backward response with an empty metrics object still triggered
an optimizer step, logged a NaN loss and wrote a successful completion receipt.
Both GRPO and every GSPO epoch now require a finite real scalar loss before
calling the policy optimizer. Existing `grpo_loss`, `gspo_loss` and `loss`
aliases, including valid signed zero, retain their behavior. Missing/non-mapping
metrics, absent losses, NaN/Infinity, booleans, numeric strings, non-scalars and
unrepresentable integers fail the run without applying that epoch's optimizer
update or emitting successful step/completion records. Previously completed
GSPO epochs are not rolled back; a later invalid epoch aborts further work.
The validated scalar is captured before the optimizer callback so later mutation
of the returned metrics object cannot change the logged loss.

Seventeen failure regressions failed before the gate (three valid-zero controls
already passed). The expanded focused file now passes **23 cases**, and the
broader training/SDK suite passed **248 cases** before the final three additions.
Muse's completed public-source review confirmed the unconditional optimizer
problem. Its suggestions to silently continue/skip or log raw metric payloads
were rejected: this implementation raises a bounded error and does not publish
untrusted payloads or declare a partial run successful.

Full validation passed **1,737 tests, 2 skipped**, including native TeX, at
**85.67%** configured supported-surface coverage in 250.73 seconds. Static,
package, lock, docs, public/demo and submission checks passed, aggregate exit
zero. These are CPU/fake-transport checks, not provider training or evidence of
improved model quality. The two proposed scorer/sampler-refresh default changes
remain unimplemented pending approval.

## CPU critic update guards (19:28 UTC)

Real CPU Adam reproductions showed that target `1e20` produced infinite
float32 half-MSE while still updating, and a NaN target poisoned parameters
and Adam state. Critic training now rejects nonfinite/unrepresentable float32
targets, nonfinite loss before backward, and nonfinite gradients before Adam.
Rejected gradients are cleared. Ordinary arithmetic and empty-batch behavior
are unchanged; no reward clamping, objective change or update reordering was
introduced. Five failing regressions now pass, alongside an exact two-update
comparison against the previous finite-input arithmetic.

An end-to-end mocked training regression confirms that critic overflow cannot
write a successful completion receipt or critic sidecar. The actor update still
precedes critic fitting and is not rolled back; this guard prevents an invalid
critic update and false completion, not every possible earlier actor update.
Seven focused integrity cases passed; the broader critic/training suite passed
224 tests with one hardware skip before the final integration case was added.
Muse's completed public-source review supported finite target/loss/gradient
gates. Its suggested silent skipping/nonfinite diagnostic returns were not
adopted. Its blanket claim that the finite `1e20` example necessarily produces
infinite gradients was not supported by the CPU reproduction: loss overflow
alone is sufficient to reject it.

Full validation passed **1,744 tests, 2 skipped**, including native TeX, at
**85.69%** configured supported-surface coverage in 268.67 seconds. Static,
package, lock, docs, public/demo and submission checks passed, aggregate exit
zero. All critic experiments used synthetic local CPU data and no providers.
A separate follow-up is checking finite model/optimizer checkpoint state;
these pre-update checks alone do not repair pre-existing poisoned Adam state
or guarantee that every finite-gradient optimizer update remains finite.

## Finite critic state at persistence and update boundaries (19:35 UTC)

A synthetic Adam state containing Infinity was successfully serialized,
SHA-256-bound and reloaded by the previous implementation. Byte integrity does
not establish numerical validity. A shared, non-mutating finite-state check
now covers nested model/optimizer tensors and floating scalars before critic
training, after each critic optimizer step, before serialization and after
hash/config/step verification on load. The iterative walker tolerates shared
or cyclic containers without recursion. Invalid saved artifacts are preserved
for diagnosis; invalid new state cannot be published as a critic sidecar.

Eight regressions failed before the change and now pass: model/Adam corruption
at save, bound and legacy load, pre-existing poisoned state, and state poisoned
by an optimizer callback. The last test explicitly establishes post-step
detection: the completed update is not rolled back, but no later iteration
runs. A ninth case covers nested scalar state and container aliasing. All 16
critic integrity cases pass; the broader critic/training suite passed 233 tests
with one hardware skip before that ninth state case was added. A completed
Muse review of public save/load/training code independently identified these
boundaries and the no-rollback limitation.

Full validation passed **1,753 tests, 2 skipped**, including native TeX, at
**85.74%** configured supported-surface coverage in 254.03 seconds. Static,
package, lock, docs, public/demo and submission checks passed, aggregate exit
zero. These guards do not repair poisoned state, prove optimizer quality, or
provide cross-provider rollback; failures require restoring a valid checkpoint.
All experiments were local synthetic CPU checks. The preceding published
`f059d9ad` commit's engineering CI finished successfully; the scientific
`minreport-audit` gate remains the sole known CI failure.

## Necessary optimizer and sampling domains (19:42 UTC)

Local construction reproduced acceptance of negative learning rate/epsilon/
temperature, beta1 equal to one, negative beta2 and top-p greater than one by
both the run config and locked Tinker 0.24.1 request models. No service requests
were made. Configuration preflight now requires nonnegative learning rate,
epsilon and temperature; Adam betas in `[0, 1)`; and top-p in `[0, 1]` before
datasets or tracking are reached. Existing zero-valued controls and defaults
are preserved. These are necessary mathematical bounds, not a guarantee that
a particular remote model/service accepts every boundary value.

The Adam bounds agree with the official versioned implementation inspected at
https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/torch/optim/adam.py .
Current Tinker field documentation was also inspected at
https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/adamparams/
and https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/samplingparams/ ;
those pages describe a newer SDK snapshot and are not substituted for the
locally tested locked schema. A completed Muse public-source review supported
the narrow necessary bounds without model-specific limits or default changes.

The new invalid-domain and mutated-config regressions failed before the guard;
the zero-mode control already passed. The focused config/CLI/training suite
passed **216 tests**. Full validation passed **1,756 tests, 2 skipped**, including
native TeX, at **85.76%** configured supported-surface coverage in 264.44 seconds.
Static, package, lock, docs, public/demo and submission checks passed, aggregate
exit zero. No provider job, learning-policy change or historical regrading was
performed.

## Typed visibility flags and safe endpoint overrides (19:49 UTC)

Synthetic fake-transport reproductions exposed two configuration hazards:
`hf_public="false"` requested public Hub repositories because the string is
truthy, and HTTP userinfo in `base_url` was copied into W&B configuration.
No actual credentials, provider calls or Hub repositories were involved.
Boolean-valued config options now require actual booleans; strings/numbers are
not silently coerced. Optional paid-campaign flags retain their existing
separate fail-closed gate. Validation is repeated at run, tracking/publication
and fingerprint boundaries so mutation after construction cannot bypass it.

Explicit endpoint overrides must be unambiguous HTTP(S) URLs without userinfo,
whitespace/control characters, backslashes, invalid/missing hosts, invalid
ports or fragments. Default `None`, valid HTTPS base paths, local HTTP and
IPv6 endpoints remain supported. Errors never interpolate the rejected URL;
parser exceptions are suppressed to avoid echoing userinfo. This is not a
general secret detector: arbitrary query values are not inspected, and callers
must not place credentials there. No authentication mechanism or access grant
was configured or changed.

Seventeen privacy/invalid-input cases failed before the guards, while five
valid-endpoint controls already passed. All **22 focused cases** and the broader
**238-test** config/CLI/training suite now pass. A completed Muse review of the
public endpoint/tracking path supported early validation and bounded errors.
The existing valid private/public boolean semantics are unchanged.

Full validation passed **1,778 tests, 2 skipped**, including native TeX, at
**85.83%** configured supported-surface coverage in 266.45 seconds. Static,
package, lock, docs, public/demo and submission checks passed, aggregate exit
zero. The preceding published `74f3d646` commit's engineering CI completed
successfully; the scientific `minreport-audit` gate remains red. No scorer or
sampler-refresh default was changed.

## Model-source receipt scope (20:06 UTC)

The locked public Tinker 0.24.1 SDK's training-client creation interface has no
base-weight revision argument. The runner's `model_revision` is a local
tokenizer-load request and provider metadata, not proof of a managed weight
commit. New local `model_source` receipts distinguish configured intent,
caller-supplied versus requested tokenizers, fresh versus optimizer-checkpoint
initialization, and the step from which the record applies. Managed base
weights are explicitly unverified. Completed legacy receipts remain unrecorded;
no historical source provenance is backfilled. Provider arguments and defaults
are unchanged. CLI help now states the revision scope.

Six focused receipt cases failed before the change and now pass; the broader
214-test focused suite passed. A public-source Muse review supported this
scope distinction. The first aggregate found two publication-integrity failures
because a README clarification changed a manifest-frozen file. That paragraph
was removed, leaving README and its manifest untouched. The corrected complete
`make check submission-check` passed **1,784 tests, 2 skipped**, including native
TeX, with **85.84%** configured supported-surface coverage in 252.24 seconds;
static/package/lock/docs/public/demo/submission checks passed, exit zero.
All engineering CI on the preceding `b5c06b6c` passed; the scientific gate remains
red. No training/provider call or scorer/sampler default change was made.

## CLI launch and dataset configuration fidelity (20:12 UTC)

A JSON config with `steps=1000, seed=42` silently ignored explicit CLI
`--steps 1 --seed 17 --no-resume`. The JSON path now uses the same explicit
argument override path as presets. It also rejects duplicate keys at every
object level, non-finite numbers and non-object/malformed JSON with a bounded
error that does not echo config content. Unknown fields and mandatory tracking
and campaign gates remain enforced. A mocked CLI-to-runner integration checks
that the one-step budget, seed and resume choice reach execution; JSON-only
values and file bytes remain unchanged. Nine new cases failed before the fix;
the unknown-field control already passed.

Separately, explicit dataset revisions were ignored by GSM8K and unsupported
adapters. GSM8K now forwards an optional requested revision to both train and
test loads, preserving the historical unpinned default when omitted. The
portfolio accepts only its existing composite revision (or unrecorded None),
and synthetic adapters reject an external revision instead of recording an
unused request. Five new cases failed before; three default/composite controls
already passed. No dataset/model download or provider run was performed.

Two completed public-source Muse reviews supported these findings. The
portfolio review's suggestion to reject every non-null revision was narrowed
to preserve the preset's valid composite revision. **123 focused tests** pass.
Full `make check submission-check` passed **1,802 tests, 2 skipped**, including
native TeX, with **85.86%** configured supported-surface coverage in 258.01
seconds; static/package/lock/docs/public/demo/submission checks passed, exit
zero. Historical results and scorer/sampler defaults are unchanged.

## Fresh-run checkpoint identity protection (20:18 UTC)

`experiment/a` and `experiment a` sanitize to the same checkpoint filename.
Previously `resume=False` bypassed existing-receipt inspection, allowing the
second experiment to overwrite the first in a mocked end-to-end run. Existing
receipts are now parsed even for fresh starts. A fresh restart requires the
original unsanitized name and exact integer seed to match; absent/malformed or
foreign identity fails before tracking/provider creation and preserves bytes.
An explicitly requested fresh restart of the same identity may still change
its configuration. This is an early single-writer identity guard, not a lock or
protection against concurrent filesystem replacement. Existing paths and
compatible resume behavior are unchanged.

Six collision/missing-identity cases failed before the guard; the same-identity
changed-config control already passed. All **238 focused tests** pass. A
public-source Muse review confirmed the collision; its broader suggestion to
forbid all config changes was not adopted because it would defeat an explicit
same-experiment fresh restart. Full `make check submission-check` passed
**1,809 tests, 2 skipped**, including native TeX, at **85.87%** configured
supported-surface coverage in 256.28 seconds. Static/package/lock/docs/public/
demo/submission checks passed, exit zero. No historical file was rewritten.

## Local held-out evaluation evidence (20:24 UTC)

Completion receipts previously retained only the held-out mean; example counts,
failure counts and uncertainty bounds existed only in tracking logs. New
`heldout_evaluation` records snapshot the already-computed aggregate summary,
with disabled/empty/completed/partial-failure/all-failed status, denominator,
failure count and the existing score-zero/keep-denominator policy. Bounds are
explicitly within-run example bootstrap, never across-seed uncertainty. Missing
legacy evidence remains unrecorded. No prompt, response, exception text, extra
sampling call or repeated bootstrap calculation is added. The summary is copied
before the tracking callback so callback mutation cannot alter the local record.
Training success semantics and numerical scoring are unchanged.

Five initial cases failed before the new receipt; seven final cases cover the
outcome states, legacy completion and callback isolation. **107 focused tests**
pass. A public-source Muse review supported snapshot-only local evidence.
Full `make check submission-check` passed **1,816 tests, 2 skipped**, including
native TeX, at **85.88%** configured supported-surface coverage in 257.79 seconds.
Static/package/lock/docs/public/demo/submission checks passed, exit zero.
Engineering CI for preceding published `ee2fc11d` passed; the scientific
`minreport-audit` gate remains red.

## Exclusive JSON checkpoint staging (20:30 UTC)

A synthetic-only local probe demonstrated that a symlink at the predictable
PID-based temporary path redirected a checkpoint write into an unrelated
synthetic file and left the receipt itself as a symlink. JSON is now serialized
before filesystem side effects, then written to an exclusively created unique
UTF-8 temporary, flushed and fsynced before atomic replacement. Cleanup removes
only this write's owned temporary. Existing stale files/links are untouched;
pre-replacement I/O failures preserve the prior receipt. This is not a
multi-writer lock, ancestor-path hardening, cross-provider transaction or full
power-loss durability guarantee.

All four regression cases failed before the fix: stale-link isolation,
replacement failure cleanup, fsync failure preservation, and serialization
failure without directory creation. A completed public-source Muse review
supported the narrow staging change. The combined checkpoint/CPU focused suite
passed **64 tests**. Full `make check submission-check`, including the CPU
optimizer smoke cases documented next, passed **1,824 tests, 2 skipped**, native
TeX included, at **85.90%** configured supported-surface coverage in 248.56
seconds. Static/package/lock/docs/public/demo/submission checks passed, exit zero.

## Real SDK and CPU optimizer/resume smoke (20:30 UTC)

Four offline integration cases now execute the complete one-seed runner, the
actual locked Tinker custom-loss bridge, and Torch Adam on tiny independent
prompt/response categorical parameters. Only network transports, sampling,
tracking and checkpoint publication are faked; socket connections are blocked.
An independent direct objective (not production loss helpers or generated
weights) predicts each gradient and final Adam-updated response parameter for
GRPO, NLL-auxiliary, GSPO, and reversed GRPO rewards. Prompt parameters stay
unchanged. Restoring the captured step-one actor/Adam state and local receipt
then finishing the same budget matches uninterrupted execution.

A separate uncommitted negative-control fixture intentionally discarded Adam
history: NLL and GSPO resume comparisons failed, while constant-gradient GRPO
controls passed. This confirms sensitivity to missing optimizer state where it
changes this fixture's trajectory. The fixed balanced response batch always
has reward 0.5; increasing the rewarded-token probability is a wiring assertion,
not evidence of real-model quality improvement. `save_every=1` is explicit in
this fixture and does not alter the production sampler-refresh default.

A public-source Muse review supported the independent oracle and bounded scope.
All four final smoke cases, the 64-test focused suite and the 1,824-test aggregate
above passed. Run directly with `pytest -q tests/test_cpu_optimizer_smoke.py`.
No provider model, paid job or new authentication grant was used.

## Descriptor-pinned JSON checkpoint reads (20:36 UTC)

JSON resume now shares the existing critic loader's descriptor boundary:
`O_NOFOLLOW | O_NONBLOCK`, regular-file verification, one read from the opened
descriptor, and guaranteed descriptor closure. Existing and dangling symlinks
are refused rather than followed or treated as a missing receipt. FIFOs and
directories fail without blocking; invalid UTF-8 gets a bounded error. Normal
receipts retain strict JSON/config/history validation. This requires the same
OS flags already required for secure critic loading; it does not authenticate
receipt content, pin ancestor directories, or prevent same-inode concurrent
writes. No receipt size limit is newly imposed.

Both symlink cases failed before the change; a normal-receipt control passed.
The six final cases additionally verify FIFO nonblocking flags and descriptor
closure, directory rejection and UTF-8 preservation. The broader 80-test
checkpoint/critic/CPU suite passed before the additional UTF-8 case; all six
final focused cases pass. Full `make check submission-check` passed **1,830
tests, 2 skipped**, including native TeX, at **85.91%** configured
supported-surface coverage in 256.32 seconds. Static/package/lock/docs/public/
demo/submission checks passed, exit zero. No historical artifact was changed.

## Preserve the preflight-resolved checkpoint owner (20:42 UTC)

Publication previously selected the owner again after preflight. A whitespace-
normalized configured owner was replaced by its raw spelling; a tracking
callback changing `HF_REPO_OWNER` redirected later exports to another namespace.
Repository naming now uses only the resolved owner argument. Configuration,
environment and authenticated-account precedence remain in preflight, and
empty explicit owner strings fail before publication. Preflight resolves an
identity; it does not establish organization write permissions.

Two synthetic regressions reproduced the owner drift. Three final cases pass,
along with **198 focused tests**. A former private-helper test that expected an
environment override was updated to assert the resolved-owner contract; the
prefix-slash isolation check remains. Muse reviewed the public paths and
supported the narrow fix. Full `make check submission-check` passed **1,833
tests, 2 skipped**, native TeX included, at **85.92%** configured supported-surface
coverage in 257.93 seconds. All aggregate stages passed, exit zero. Engineering
CI on the preceding published `cfc431f1` passed; the scientific gate remains red.

## Actual public GSM8K/tokenizer CPU check (20:42 UTC)

A separate source-bound input audit used all **7,473** public GSM8K train rows at
revision `740312add88f781978c0658806c59bc2815b9866` and the official Qwen3-8B
tokenizer at `b968826d9c46dd6066d109eabc6255188de91218`. The anonymous tokenizer
fetch was limited to tokenizer JSON/config assets; no weights were downloaded.
The actual Transformers loader ran from those local files with remote code
disabled and network sockets blocked, as did the real GSM8K adapter using the
local pinned Parquet fixture. All rows were retained. Prompt lengths were
48 minimum, 92 median, 137 p95 and 248 maximum; **none exceeded the existing
1,024-token budget**. One synthetic system/user rendering matched the official
default chat template. This does not generalize to other adapters, model
families or tool/multiturn templates, and is not a generation/quality evaluation.
No truncation policy was changed on the basis of a speculative concern.

`GSM8K_TOKEN_BUDGET_AUDIT.json` records source URLs, asset hashes, versions,
function hashes and the exact scope. Raw public data/tokenizer downloads remain
outside the repository. No credentials, provider job or private data were used.

## Offline CLI configuration check (20:53 UTC)

`python -m platform_tinker.tinkerrl.grpo_cli --check-config --preset gsm8k
--steps 1 --seed 17 --no-resume` now runs local configuration and adapter-selection
checks without an API key, dataset load, tokenizer, tracking or provider work.
It reuses strict JSON/override/numeric/tracking/campaign validation and a pure
adapter/revision selector. Output is a configuration SHA-256 plus allowlisted
numeric/boolean settings and selected built-in adapter names. Raw config,
private model identifiers, endpoint/owner strings and filesystem paths are not
printed. Its explicit `not_checked` list prevents treating this as data,
authentication, service, export or scientific-readiness approval. Normal
execution defaults and the existing missing-key behavior are unchanged.

Six initial cases failed before this mode existed; eight final cases cover
offline success, JSON budget overrides, source-revision selection, invalid
selection/budgets, tracking/campaign gates and bounded output. **109 focused
tests** pass. A separate actual CLI invocation with the key absent and sockets
blocked produced the expected one-step configuration receipt. A public-source
Muse review supported the early-exit/pure-selection design. Full
`make check submission-check` passed **1,841 tests, 2 skipped**, native TeX
included, at **85.97%** configured supported-surface coverage in 258.74 seconds;
all aggregate stages passed, exit zero. This also revalidated the newly added
public token-budget audit artifact. Engineering CI on `a19b3fec` passed; the
scientific `minreport-audit` gate remains red.

## Bind newly recorded resumes to training inputs (21:03 UTC)

A synthetic interrupted-run reproduction changed a training prompt under the
same configuration, resumed step one, and previously declared mixed-source
history successful. New local `dataset_source` receipts now fingerprint the
ordered seed-start JSON projection of training prompts, targets and metadata.
Matching recorded inputs retain the original fingerprint scope; changed order,
content or newly unverifiable input fails before tracking, critic or provider
setup. Counts/schema/origin/hash metadata are validated rather than coerced.
Fingerprints are streamed and never sent to tracking/provider metadata.

The projection accepts only exact built-in JSON types and `TrainingExample`
objects. Tuple/custom/non-finite/cyclic/unencodable values remain explicitly
unrecorded, without coercing or changing adapter objects. Completed legacy
receipts remain untouched and do not acquire invented evidence; a legacy
partial resume records only the new suffix's starting step. This is neither
source authentication nor proof that mutable data stayed unchanged after the
seed-start observation. Historical source evidence is not backfilled.

Ten initial cases failed before the change. Thirteen final cases cover changed
inputs, canonical/type-sensitive hashing, malformed receipt metadata, unchanged
and legacy resume, unsupported values, and completed-result reuse. **129 focused
tests** pass, including the real SDK/Adam resume smoke. A public-source Muse
review supported the limited binding and legacy rules. On all 7,473 pinned
public GSM8K train examples (seed 42), fingerprint-only work took 0.491 seconds
with 148,718 bytes of incremental tracemalloc peak allocation; corpus loading
is excluded. The v1 digest was
`6b73cc410d8156c9fdeba59120df196ab0c0bdc0c85f06e8f70a6961636613e3`.

Full `make check submission-check` passed **1,854 tests, 2 skipped**, native TeX
included, at **86.04%** configured supported-surface coverage in 255.30 seconds;
all aggregate stages passed, exit zero. No scorer or sampler-refresh default
was changed, and no paid/provider job was run.

## Local storage checks before avoidable remote work (21:17 UTC)

A synthetic checkpoint-directory collision previously failed only after one
actor optimizer update. New/incomplete runs now probe a unique owned temporary
(write/flush/fsync/cleanup) before tracking or provider setup. Completed cached
receipts do not require writes. At periodic/final save boundaries, local critic
bytes are prepared before remote state/sampler persistence and Hub export, so
a known critic serialization/storage failure does not publish that step first.
Earlier valid exports may remain, and later failures can still orphan external
artifacts or unused immutable sidecars. This is not a future-capacity guarantee
or a cross-provider transaction; the previous local receipt stays authoritative.

Five cases failed before these changes; the completed-read-only control passed.
All six new cases and the 56-test storage/critic/CPU/data-binding focused suite
pass. The training-loop Muse review prompted the publication-order investigation.
Its claim that teardown discards the original exception was rejected: a direct
Python traceback probe retained the original training failure. Duplicate resume
snapshots were not treated as a uniqueness defect or changed without lifecycle
evidence. No new loss-math/statistics defect was substantiated by the two other
independent Muse reviews; these were bounded reasoning reviews, not proof of
all behavior.

Together with the adapter-pin change below, full `make check submission-check`
passed **1,866 tests, 2 skipped**, native TeX included, at **86.08%** configured
supported-surface coverage in 263.06 seconds. All aggregate stages passed,
exit zero. Normal training/scoring/cadence defaults remain unchanged.

## Cross-adapter source-pin preflight (21:17 UTC)

The adapter review identified a reproducible mismatch: selecting `pavlov_xlam`
from the `pavlov_portfolio` preset forwarded the portfolio's composite revision
to the xLAM loader. Cross-source overrides now reject a demonstrably inherited
pin unless the caller explicitly supplies a replacement revision or owns the
revision through a JSON configuration. Same-source aliases retain their pin.
This does not validate remote revision existence or all campaign metadata.
Three new cases failed before the guard; three explicit/alias/JSON controls
passed. The final combined preflight/CLI/CPU focused suite passed **107 tests**,
and the 1,866-test aggregate above passed.

The review's portfolio-JSON example did not silently train on synthetic data:
the existing unsupported-revision guard rejected it locally. CLI help now
clarifies that JSON contains GRPO hyperparameters while adapter selection
follows CLI flags; no adapter is inferred from an experiment name. Dependent
NLL threshold settings remain inactive without enabling the objective, rather
than changing scoring/training defaults based on that review suggestion.

## Preserve configuration identity and metadata shapes (21:26 UTC)

`wandb_tags="experiment"` previously became one tag per character, and
`model=False` passed the local configuration/tracking checks. Identity and
optional source/owner fields now require nonblank strings (or None where
optional). Metadata sequences accept None, lists or tuples of nonblank strings;
scalar strings, mappings, sets, booleans and wrong element types fail with
field-name-only errors. Valid caller lists remain detached as tuples, optional
None remains empty metadata, and pathlib checkpoint directories are normalized
to JSON-compatible filesystem strings. The same guards run at validation
boundaries so later mutation cannot reintroduce ambiguous shapes.

Four initial cases failed; valid sequence copying and canonical W&B-mode
controls already passed. The mutation regression blocks the dataset seam and
network, keeping it independent of authentication state. **244 focused tests**
pass. A public-source Muse review supported the shape guards. Its additional
mode-normalization suggestion was not adopted: the existing runner already
passes the literal `online` to the SDK, as the control test confirms.

Full `make check submission-check` passed **1,872 tests, 2 skipped**, native TeX
included, at **86.11%** configured supported-surface coverage in 260.11 seconds;
all aggregate stages passed, exit zero. Valid experiment defaults, scoring and
sampler cadence remain unchanged.

## Legacy xLAM proxy scope and access limit (21:31 UTC)

The existing xLAM adapter selects `answers[0]` and includes only `tools[:8]`.
An invented two-action fixture received reward 1.0 for its first action alone;
an invented nine-tool fixture omitted the required ninth tool from the prompt.
These reproduce the transformation's limits, not their prevalence in the
pinned corpus. Legacy scores therefore describe a first-call reward proxy,
not verified completion of every requested call.

[Salesforce's public dataset card](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
documents multi-call answer arrays and conditions governing file access. No
gated files were retrieved or conditions accepted. Pinned file contents and
corpus-wide incidence remain unverified. Existing adapter/scorer defaults and
historical metrics are unchanged; a multi-call extension is deferred pending
an explicit definition/approval of its scoring and dataset semantics.

## Rollout scoring and behavior snapshot boundary (21:44 UTC)

User scoring hooks ran before behavior logprobs and stop reasons were copied,
and text decoding reread live response tokens after training tokens were copied.
A callback could therefore change the behavior-policy denominator or truncation
mask, or score a different response than the one used for training. The runner
now detaches token IDs and requested behavior metadata before invoking scoring
hooks, decodes from a fresh copy of the detached IDs, and publishes the captured
metadata after callbacks return. Ordinary scores and sampling cadence are unchanged.

Two regressions failed before the fix: one uses the real Tinker 0.24.1 SDK's
NumPy-backed logprobs; the other changes a later response's stop reason. A
mutating-decoder control already passed. The second regression was then
strengthened to mutate live token IDs and the output metadata dictionary too;
all 16 focused snapshot, real-SDK loss bridge and real Torch CPU optimizer
checks pass with sockets blocked. Public-source Muse review independently
identified both snapshot windows and confirmed the decoder control; speculative
prompt-buffer/concurrent-SDK concerns were not expanded into changes.

Full `make check submission-check` passed **1,875 tests, 2 skipped**, native TeX
included, at **86.11%** configured supported-surface coverage in 252.03 seconds;
all aggregate stages passed, exit zero. The extra token-mutation assertions
were checked in the 16-test focused rerun after that aggregate; production code
was unchanged. This verifies consistency of captured local rollouts, not remote
provider immutability or model-quality gains.

## Closed native event-file integrity (21:54 UTC)

A real TensorBoard 2.20.0 writer/reader probe showed that a valid two-reward
prefix was accepted even with a two-byte partial header, an invalid header CRC,
or a copied record with a flipped payload CRC appended. All three new cases
failed before the fix. TensorBoard's live-stream loader deliberately stops on
`DataLossError` so a future write can complete a record; see the
[official 2.20.0 loader](https://raw.githubusercontent.com/tensorflow/tensorboard/2.20.0/tensorboard/backend/event_processing/event_file_loader.py).
That streaming behavior is insufficient for this wrapper's stopped-writer,
complete-artifact contract.

The native metrics reader now validates complete TFRecord framing and both
masked CRC32C checksums across the exact captured bytes before parsing scalar
rewards. It reuses TensorBoard's checksum implementation, bounds each declared
payload length against remaining bytes before slicing, and rejects corrupt or
partial records as `native_events_unreadable`. Existing reward tag, step,
finiteness, file-membership and byte-stability checks remain in place. This is
an integrity check, not authentication of training or a live-tailing feature.

**29 focused checks passed**, including all seven real TensorBoard cases and
dependency-free reader-boundary tests. A framing test also covers every partial
minimum-record length, an impossible uint64 payload length and both CRC fields
before the scalar parser is called. The dependency-free scalar-parser mocks
use explicitly stubbed CRCs; checksum correctness is exercised by the real
writer/reader cases. Public-source Muse review supported the narrow gate.

Full `make check submission-check` passed **1,876 tests, 2 skipped**, native TeX
included, at **86.11%** configured supported-surface coverage in 258.90 seconds;
all aggregate stages passed, exit zero. Native optional TensorBoard integration
is the separate 29-test check above; native code is outside the configured
coverage denominator. No model training, provider jobs or historical result
rewrites were used.

## Completed-cache identity and arithmetic (22:03 UTC)

A synthetic completed receipt configured for seed 42 returned seed 999 and
`avg_last10=123.0` despite its validated `reward_trace=[0.5]`. The cache path
previously validated history but trusted the separately stored result fields.
It now checks result construction, exact integer seed identity, nonblank run
ID, agreement with an optional top-level run ID, the resumed-step range, and
finite numeric summary/held-out scalars before tracking or provider access.
First-five/last-ten means and peak must agree with the validated trace within
1e-12 relative/absolute floating-point tolerance. Receipts are never repaired
or rewritten by this validation.

Eight rejection cases failed before the fix; the valid cached-return control
already passed. Eleven targeted cases now cover those failures, a contradictory
top-level ID and preserved historical summation roundoff. **126 focused tests**
passed before the final two controls were added; the final 11-case file and the
full aggregate below passed afterward. Existing legacy checkpoint-URL and
unrecorded provenance controls remain valid. Public-source Muse review agreed
with the boundary checks. Its suggestion to grandfather contradictory zero
summaries was not adopted: a missing/default summary is usable only when it
agrees with the stored trace, rather than being silently treated as evidence.

An initial aggregate stopped at a test-parametrization lint rule; after that
syntax correction, full `make check submission-check` passed **1,887 tests,
2 skipped**, native TeX included, at **86.10%** configured supported-surface
coverage in 243.95 seconds; every aggregate stage passed, exit zero. These are
internal consistency checks, not authentication of historical rewards or a
change to reward/scorer versions.

## Joint actor/critic CPU resume smoke (22:12 UTC)

A new end-to-end CPU smoke exercises the actual runner, Tinker 0.24.1 custom
loss bridge, Torch actor Adam, real prompt-value critic/Adam, local digest-bound
critic sidecars, and a mid-run restore. Only provider transport, fixed synthetic
sampling, tracking and export are faked; sockets are blocked. Three synthetic
prompts have different reward patterns. The test checks nontrivial actor motion,
full final actor/critic optimizer-state agreement, retained reward history and
the exact remaining example order after critic-pretraining draw replay.

The initial fixture's seed happened to sample only cancelling reward groups;
its actor-motion assertion correctly failed. The final fixture uses seed 7 to
include an informative group. Two deliberate, process-local negative controls
then demonstrated sensitivity: discarding restored critic Adam moments changed
final critic parameters (largest observed difference about 0.0193), while
omitting pretraining RNG replay changed the checked example order. Neither
negative control altered repository production code or persisted source files.

All **14 focused real-SDK/CPU checks** passed normally. Full `make check
submission-check` passed **1,888 tests, 2 skipped**, native TeX included, at
**86.10%** configured supported-surface coverage in 253.43 seconds; all aggregate
stages passed, exit zero. This adds a regression harness, not a new optimizer,
default sampler cadence, provider run or model-quality result.

## Checkpoint-reference validation and snapshots (22:28 UTC)

An offline provider-double probe returned `save_state(path=None)` and still
completed two optimizer updates while writing two non-resumable periodic actor
references. Another export callback changed a response object's path after
publication, so subsequent sampling and local result references disagreed with
the exported source. Seven initial regressions reproduced these boundaries.

The runner now validates and captures each actor/sampler reference immediately
after its save response, before subsequent save/export/client/receipt use.
References must be nonblank strings without a leading CLI-option dash or ASCII
control characters. Valid strings are preserved verbatim; no URI syntax,
existence or remote-durability attestation is invented. The existing public
export helper's empty-reference error is retained. Snapshot strings are reused
consistently even if a callback later mutates the response objects.

All **8 boundary checks** pass, including initial/periodic/final failures,
actor and sampler mutation, malformed direct export inputs, and retention of
the last valid step-one receipt when the step-two actor save is malformed.
**78 broader focused tests** passed before adding that last-good-receipt case.
Already-applied optimizer updates are not rolled back; the guard prevents
further checkpoint/export work at the invalid boundary. It is not a distributed
transaction or a guarantee that every failed run has a recoverable checkpoint.
Public-source Muse review supported the validation/snapshot boundary; reviewed
source bytes were also matched against anonymous, commit-pinned GitHub fetches.

The final `make check submission-check` passed **1,896 tests, 2 skipped**, native
TeX included, at **86.12%** configured supported-surface coverage in 252.60
seconds; all aggregate stages passed, exit zero. Ordinary references, sampling
cadence, objectives and historical result files remain unchanged.

## Scalar tensor rollout snapshots (22:37 UTC)

Copying a sequence into a tuple did not fully bind loss inputs: iterating a
Torch tensor yields scalar views into the original storage. Mutating tensor
advantages after constructing either loss closure changed the eventual policy
gradient; mutating integer response lengths included prompt-token gradients.
Tensor NLL masks and coefficients likewise changed the auxiliary objective.
Five mutation regressions failed before the fix, alongside two controls that
require fractional response lengths to fail at binding rather than be truncated.

Loss factories now capture advantages and the NLL coefficient as Python floats,
mask elements as bools, and response lengths with the integer-index protocol.
Existing finite checks run before numeric conversion. This detaches scalar
rollout labels, while the explicit GSPO output stash remains shared as designed.
Normal Python input behavior and the objective formulas are unchanged.

All **97 focused checks** passed, covering numerical stability, real Tinker SDK
loss bridging, and CPU actor/critic optimizer-resume integration. The cancelled
Muse review for this issue did not yield a verified result; this batch relies on
local reproductions and validation, not an asserted external review.

Full `make check submission-check` passed **1,903 tests, 2 skipped**, including
native TeX, at **86.1%** displayed configured supported-surface coverage in
249.20 seconds. All aggregate stages passed, exit zero. No default scorer,
sampler cadence, provider job or historical score changed.

## Cached held-out evidence consistency (22:46 UTC)

A synthetic completed receipt was accepted with a held-out result of 1.0 while
its own summary said 0.0, both examples had failed, and its status said
`completed`. Thirteen initial adversarial checks showed that recorded status,
denominators, failure counts, uncertainty metadata and summary values were not
cross-checked before the cached result returned.

Version-one evaluation evidence now requires consistent status/counts, finite
summary values, matching point estimates, and appropriate ordered bootstrap
bounds. All-failed zero-scoring receipts cannot claim a nonzero mean or spread;
a single example cannot claim bootstrap uncertainty. Malformed or unsupported
evidence fails before tracking/provider activity, without repairing the file.
Missing legacy evidence remains unrecorded and accepted; an explicit mean-only
receipt stays mean-only. This verifies internal consistency, not authenticity
or the underlying observations, and does not infer across-seed uncertainty.

A separate cancellation-heavy numeric example exposed two point estimates for
the same 102 scores: the run's canonical mean was about 0.980392, while NumPy's
bootstrap helper returned about 0.823529. The summary now retains the run's
existing canonical mean and uses the bootstrap helper only for its bounds.
No historical result is rewritten or regraded. Ordinary bounded scores and the
scorer's definition are unchanged.

All **228 focused checks** passed. Valid disabled, empty, completed,
partial-failure and all-failed receipts were replayed without a second provider
or evaluation call. The malformed-cache integration check verifies no tracking
starts and no cache bytes change. The public-source Muse review was cancelled
and remains unverified; it is not cited as review evidence.

Full `make check submission-check` passed **1,922 tests, 2 skipped**, native TeX
included, at **86.20%** configured supported-surface coverage in 246.95 seconds.
All aggregate stages passed, exit zero. No provider job or benchmark-default
change was made.

## Fail-closed GSM8K source labels (22:54 UTC)

The Tinker adapter silently omitted rows without an answer marker and accepted
numeric prefixes from malformed labels such as `#### 1/2` or `#### 2 or 3`.
It also truncated exponent answers and omitted explicit plus-signed answers.
Eleven initial regression cases reproduced these behaviors across training and
held-out source rows.

Each source row now requires a nonempty question and a complete finite numeric
label after its final answer delimiter. Properly grouped thousands, signs,
decimals and exponents are handled without prefix matching. Malformed rows or
empty splits stop loading with split/row context, rather than silently changing
the evaluation denominator. Errors do not include question or answer text.
This validates dataset labels; it does not change the response scorer or the
pending final-answer-selection policy.

Both official public splits at revision
`740312add88f781978c0658806c59bc2815b9866` were checked using anonymous,
SHA-256-verified Parquet bytes. With networking blocked and only dataset
transport replaced by local PyArrow rows, the actual adapter retained all
**7,473 training and 1,319 test examples**. Ordered prompt/target/metadata
fingerprints at seed 42 matched exactly before and after the fix. See
`GSM8K_ADAPTER_FIDELITY.json` for source and transformed hashes. This is not a
claim that the live Hugging Face datasets transport was exercised.

All **87 focused checks** passed, including revision forwarding, malformed
labels, empty splits and valid grouped/signed/exponent inputs. Full `make check
submission-check` passed **1,935 tests, 2 skipped**, native TeX included, at
**86.18%** configured supported-surface coverage in 249.53 seconds. All aggregate
stages passed, exit zero. No raw dataset files or historical results are
included in the patch, and no provider job was run.

## Real causal-LM CPU loss-bridge smoke (23:02 UTC)

The CPU harness now exercises an actual Qwen3 causal-LM architecture in
addition to the small categorical actor fixtures. The model is constructed
locally from configuration with random weights: one layer, width 16, vocabulary
64, and 3,392 parameters. No pretrained model, tokenizer download, provider
request, GPU or quality benchmark is involved.

For GRPO, GRPO+NLL and GSPO, the test builds real Tinker datums and executes the
installed SDK's custom-loss bridge. Only its transports are replaced by real
CPU model forward/backward calls. The resulting gradients, updated model
parameters and Adam state must match a direct PyTorch objective on an identical
model. Prompt logprob weights must remain zero, all gradients must be finite,
parameters must actually change, and the rewarded fixture's log-odds relative
to the negative fixture must increase after the update. Network sockets are
blocked throughout the test.

A process-local negative control deliberately reversed the SDK transport
weights and was rejected by the gradient comparison; repository code was not
modified by that control. All **17 related real-SDK/CPU checks** passed normally.
This validates objective wiring on a real autoregressive network, not learned
language ability, sample efficiency or pretrained Qwen3-8B behavior.

Full `make check submission-check` passed **1,938 tests, 2 skipped**, native TeX
included, at **86.18%** configured supported-surface coverage in 253.66 seconds.
All aggregate stages passed, exit zero. This batch adds regression coverage
without changing production behavior.

## Staged-change coverage in the local secret gate (23:09 UTC)

`make secrets` used `gitleaks git --pre-commit` without `--staged`, so it
inspected unstaged tracked diffs while a staged-only addition could pass with
zero bytes scanned. An isolated temporary Git repository and a dedicated
synthetic-marker rule reproduced that false pass. The fixture contains no real
credential. Its clean and unstaged controls already behaved correctly.

The target now runs separate staged and unstaged tracked-diff scans. Newly
added files must be staged to enter this gate; this is not a claim to scan all
untracked files or the complete history. The existing upstream pre-commit hook
and GitHub CI's explicit commit-range scan were already scoped correctly and
are unchanged. No historical findings are suppressed or allowlists expanded.

All three isolated real-Gitleaks checks pass, including redacted output and
nonzero failure status for either changed surface. Full `make check
submission-check` passed **1,941 tests, 2 skipped**, native TeX included, at
**86.18%** configured supported-surface coverage in 246.20 seconds. The aggregate
also exercised the repaired staged scan on this patch's actual staged source
changes; all aggregate stages passed, exit zero. No user credential was read,
used or placed in a fixture.

## Critic restore compatibility before external work (23:24 UTC)

Finite, hash-matched critic checkpoints could still alter execution: loading
Adam state replaced a configured learning rate of 0.001 with 0.5, and a finite
wrong-shaped moment loaded successfully before failing at an update. Eight
initial reproductions reached tracking despite incompatible state. Additional
controls caught boolean step counters and complex model/moment tensors whose
imaginary parts PyTorch would silently discard during loading.

A dedicated restore boundary now checks parameter ordering, the configured
Adam option groups, moment shapes and dtypes/devices, nonnegative second
moments and integer nonboolean update counters. It rejects complex tensors
for this real-valued critic and rechecks finiteness after loading, since a
finite float64 value can overflow when cast to float32. This runs before
tracking or provider construction. Invalid files remain untouched; the local
model/optimizer pair may have been modified before rejection and is discarded.
No optimizer-option mismatch is silently repaired or substituted.

All **11 incompatible-state checks** now fail closed. Ordinary state, an
uninitialized Adam state, and older states missing default `foreach`/`fused`
fields remain supported. The actual joint actor/critic CPU resume smoke still
matches model parameters, Adam moments and replayed example order. All **269
focused tests passed, 1 hardware-dependent test skipped**.

A completed Muse review of anonymously fetched, byte-matched public source
confirmed the option/moment boundary. Its conflicting bound-only suggestion
was not adopted: valid legacy unbound sidecars retain their existing support.
Its broader allocation-limit suggestion is outside this patch. These checks
provide compatibility and numeric integrity, not checkpoint authentication,
allocation bounds or proof of valid training history.

Final `make check submission-check` passed **1,955 tests, 2 skipped**, native
TeX included, at **86.24%** configured supported-surface coverage in 258.00
seconds. All aggregate stages passed, exit zero. No new training defaults or
provider jobs were introduced.

## Closed-loop CPU learning and fresh-object resume (23:40 UTC)

The next smoke test runs the actual GRPO seed runner around a random tiny
causal LM, rather than supplying fixed completions or hand-set advantages.
It has one synthetic prompt and four possible terminal one-token actions;
two separate fixtures reward different actions. Samples are drawn from each
frozen model snapshot, scored by the existing reward adapter, normalized by
the runner, and updated through the real SDK loss bridge and CPU Adam.

Each fixture uses 20 steps, groups of eight, explicit seeded sampling,
temperature/top-p 1, and explicit `save_every=1`. It trains all parameters of
the tiny model, not LoRA adapters or pretrained Qwen3 weights. The rewarded
action's probability must rise from below 0.4 to above 0.9, with a gain over
0.6, and the actual reward trace and completion receipt must agree. This is a
controlled plumbing/learning regression, not benchmark accuracy, generalization
or a language-model-quality claim. The default sampler cadence is unchanged.

The same test then restores the real step-ten model/Adam state into newly
constructed model and optimizer objects through an in-memory provider double.
The remaining ten updates must reproduce the exact sampled reward trace and
match final model/optimizer state within the regression tolerances. Checkpoint
publication/tracking transports remain offline doubles; remote durability is
not being tested. Socket connections are blocked.

Two process-local negative controls verify sensitivity: disabling optimizer
updates fails the learning assertion, and discarding saved Adam moments breaks
resume equivalence. A Muse review of the published one-update harness clarified
why it alone was insufficient and informed the fresh-object restore control.
No uncommitted source was sent for that review. All **19 related real-SDK/CPU
checks** passed normally.

Final `make check submission-check` passed **1,957 tests, 2 skipped**, native
TeX included, at **86.28%** configured supported-surface coverage in 262.44
seconds. All aggregate stages passed, exit zero. This batch adds two substantive
regressions without changing production behavior or historical results.

## 2026-10-11 23:58 UTC — portfolio evaluation input/output evidence

The standalone portfolio source evaluator now rejects nonfinite adapter rewards and provider response counts other than the requested one before creating a SCORED receipt. Invalid token budgets and nonfinite/negative temperatures fail before dataset, tracking, or provider work. Strict JSON serialization supplies a second finite-value guard.

The existing 2,048-token prefix policy is unchanged. New receipt fields record the original token count, truncation flag, canonical compact-JSON SHA-256 of the actual prompt token IDs, per-example sampling seed, top-p and prompt-budget policy. Response lengths are captured before decoder/reward callbacks. This does not change reward semantics, turn these source-validation results into E1–E14 scores, or attest managed-model weights.

Nine adversarial integration cases failed before the patch. The final offline integration additionally exercises exact-budget nontruncation and callback-mutated response buffers; 42 related checks passed. W&B, tokenization and sampling are test doubles with socket connections forbidden; no provider evaluation was launched. The test restores its tracking-mode environment after each invocation. Muse reviewed anonymously fetched bytes of the public evaluator at `9732b9a59789de97a70ce40d456c8e7938675c16`, independently identifying these boundaries. Suggestions that would change failure scoring or claim checkpoint attestation were not adopted.

Final `make check submission-check` completed successfully: **1,968 passed, 2 skipped**, **87.31%** configured supported-surface coverage, 263.26 seconds for pytest. Lint, formatting, types, packaging, locked export, staged/unstaged secret scans, public-document checks, native TeX tests and submission self-tests passed. Scientific minreport and independent thesis-review blockers remain separate and unresolved.

## 2026-10-12 00:10 UTC — reject unsupported canonical execution claims

The local/Colab unified Tinker and SkyRL execution routes now fail before a subprocess/provider launch instead of implying canonical training. Intercepted execution showed Tinker silently dropping requested accumulation, KL, clipping and LoRA-contract fields, and SkyRL launching only its API server; both previously reported the requested 30 optimizer steps without a verified step receipt. Three regressions failed before the guard, while two read-only dry-run controls already passed. The independent Tinker baseline CLI and its offline `--check-config` path remain available. Frozen protocol values, reward policies and sampler-refresh defaults are unchanged.

Results with `final_step=None` now explicitly report unverified optimizer-step completion rather than printing “Training Complete!”. The matrix tests/documentation distinguish planned routing from executable canonical equivalence. See `UNIFIED_EXECUTION_LIMITS.md` for migration and scope; other backend-specific drivers are not newly certified by this patch. Muse's completed review used anonymously fetched public launcher/canonical bytes from `9732b9a59789de97a70ce40d456c8e7938675c16` and independently identified the omissions and false-completion paths.

**92 focused launch/migration/native-contract checks passed.** One initial full run was intentionally interrupted to correct a test assertion about the config-inspection JSON shape. Its replacement session was lost during an executor-key/transport reset and had no completion receipt, so neither is counted as a pass. The recovered final `make check submission-check` completed: **1,973 passed, 2 skipped**, **87.31%** supported-surface coverage, 254.44 seconds for pytest; all aggregate stages passed. The native launcher is outside that configured coverage surface. No provider, GPU or paid job was started.

## 2026-10-12 00:17 UTC — representable GSPO scaled exponentials

The selected GSPO branch now includes the log advantage magnitude before exponentiating, preserving sign separately. Previously, `exp(log_ratio) * advantage` could overflow and reject a representable objective, or underflow to a false zero loss/gradient. For example, float32 drift 90 with advantage −1e−5 should yield a finite loss about 1.2204e34, while drift −110 with advantage 1e30 should yield about −1.6889e−18 rather than zero. This changes numerical evaluation, not the mathematical clipping objective or its defaults.

Six red-before regressions use independent 90-digit Decimal value/gradient oracles across float32, float64, float16 and bfloat16. The repaired kernel meets the recorded 2e−5 relative tolerance, including cast-back gradients. Existing signed clipping, zero-advantage, unrepresentable-gradient rejection, SDK bridge, input-snapshot and actual tiny causal-LM checks remain passing: **94 focused checks**. The attempted public-source Muse review stopped with missing authentication and is **not** counted as a completed review; no credentials were copied or configured.

Final `make check submission-check` passed all stages: **1,979 passed, 2 skipped**, **87.31%** supported-surface coverage, 260.99 seconds for pytest. Secret scans include the staged patch and the resulting commit. These are numerical correctness tests, not new model-quality or benchmark results.

## 2026-10-12 00:25 UTC — bind cached held-out scores to their input cohort

A completed result previously checked its training-input fingerprint but accepted changed held-out targets while replaying the old held-out score. This is now guarded for newly recorded evidence: evaluation records a separate domain-tagged hash of ordered strict-JSON prompt/target/metadata inputs, and completed-cache replay verifies that hash and count before tracking/provider work. Changed prompts, targets, metadata, ordering, counts or lost verifiability fail without rewriting the cache. Empty evaluation cohorts are bound too, preventing a cached empty evaluation from hiding newly supplied examples.

Historical receipts without this evidence remain explicitly unbound and are not rewritten or assigned invented fingerprints. Custom/non-JSON inputs are recorded as unrecorded, not falsely certified. Disabled evaluation does not read the test adapter. The source record contains only digest/count/scope; it does not authenticate data, identify the scorer, or prove callbacks could not mutate inputs during evaluation. Failure scoring and bootstrap/denominator policies are unchanged.

The changed-target replay regression failed before the fix. **57 focused cache/input-integrity checks passed**, including malformed binding fields, exact unchanged replay, empty/legacy/unverifiable cohorts, and a golden training hash independently computed with public `39440af1` code, proving the training hash domain/bytes remain compatible. Final `make check submission-check` passed every stage: **1,993 passed, 2 skipped**, **87.37%** supported-surface coverage, 268.35 seconds for pytest. No provider evaluation or external private-data review was run.

## 2026-10-12 00:33 UTC — sampled two-token, multi-epoch GSPO CPU integration

A new offline integration samples eight real two-token trajectories from a random tiny Qwen3 CPU model, scores their terminal action, and performs three GSPO optimizer epochs through the actual Tinker SDK loss bridge. An independently written direct ratio/min/clamp objective on a separate model computes the sequence-mean log ratio against the original behavior policy. It checks the final model and Adam state, and verifies that a selected clipping branch is active after the first epoch. The oracle does not call the project's loss factory.

A process-local negative control discarding the original behavior logprobs was invoked exactly three times and failed the model-state comparison (maximum reported absolute mismatch about 0.02256, versus 2e−6 absolute tolerance). The normal check passes. Related SDK/CPU/numerical checks: **21 passed**. This is a full-parameter random toy model with in-memory persistence/provider doubles, not LoRA, pretrained-model improvement, remote durability, or a benchmark result.

Final `make check submission-check` passed all stages: **1,994 passed, 2 skipped**, **87.37%** supported-surface coverage, 266.97 seconds for pytest. No real provider or GPU job was run. The added test leaves algorithm defaults and scientific-score policy unchanged.

## 2026-10-12 00:47 UTC — explicit evaluator source selection and fresh atomic receipts

Portfolio evaluation now rejects empty/blank trained-sampler selectors instead of falling back to the base model, and rejects unused Hub-reference flags on a base-model request. Receipts and tracking configuration explicitly distinguish the tokenizer revision request from unverified managed weights and unverified caller-supplied Hub/sampler binding. The recorded sampler/tokenizer requests are checked against intercepted calls in offline tests. The public Tinker 0.24.1 `create_sampling_client` interface accepts a base-model name or model path, not the tokenizer revision as a weight pin; no weight attestation is fabricated.

Each invocation now requires a **fresh `--out` path**. Existing files and dangling symlinks fail before provider work. A writable-storage/atomic-link capability probe also runs before sampling. The completed JSON is fsynced in an owned unique temporary and published using an exclusive same-filesystem link, so a competing receipt created during evaluation cannot be overwritten. Unsupported links fail closed; there is no clobbering fallback. Owned probe/staging files are cleaned on tested failure paths. This is atomic no-overwrite publication, not a future-space, power-loss durability or distributed-transaction guarantee. Existing training checkpoint replacement behavior remains the default and is unchanged.

Five source-selection/provenance regressions and four output-safety regressions failed before these changes; an additional link-capability regression exposed a late failure after the first atomic-writer patch and drove the preflight check. **38 focused checks passed**, including the existing training checkpoint/storage regressions. Final `make check submission-check`: **2,004 passed, 2 skipped**, **87.45%** supported-surface coverage, 249.82 seconds for pytest, all aggregate stages successful. Historical outputs, scoring policies and sampling defaults were not rewritten; all integration transports were offline doubles.

## 2026-10-12 01:03 UTC — cached checkpoint-reference consistency

Completed-cache replay now verifies recorded checkpoint relationships before tracking/provider work: duplicated top-level lists must agree with the result, structured receipts must have valid step/source fields, recorded URL/commit lists must match their structured receipts, and a unique final export must reference the reported sampler. Previously, changing the cached sampler URI, export URL list or commit list could replay inconsistent evidence; all three reproductions failed before the fix. This is internal consistency, not remote checkpoint authenticity.

Legacy receipts lacking parallel lists remain accepted without inventing metadata. Existing URLs-only legacy evidence remains unbound to absent structured receipts. One synthetic coverage fixture was corrected to describe the same final sampler in its result and receipt; no historical result was rewritten. **125 focused checks passed**, including unchanged byte-preserving replay, legacy shapes and malformed receipt fields, with networking forbidden in the new regressions.

Final `make check submission-check` completed successfully: **2,011 passed, 2 skipped**, **87.52%** supported-surface coverage, 264.97 seconds for pytest; all aggregate stages passed. The completion exit receipt was recovered after a truncated poll output, so no interrupted run is being counted. Scientific-score policy and sampler-refresh defaults remain unchanged.
