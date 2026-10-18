# Cross-framework prospective protocol v3

This version changes scientific treatment. Do not merge its observations with historical runs, relabel old objectives as repaired, or regenerate old results in place.

## Objective and rollout semantics

- N10 is explicitly `n10_unclipped_is_response_normalized_v3`. Its built-in importance-sampling loss receives behavior log probabilities. Each response token receives `A / T / (B*G)` for GRPO or `A / Tmax / (B*G)` for Dr.GRPO. GRPO standardizes group rewards; Dr.GRPO only centers them. The rollout batch, including zero-advantage samples, defines the batch denominator. Empty completions and incomplete groups fail. The unused clipping option is removed, not silently ignored
- N10 records every rollout, including entirely zero-advantage batches, and separates rollout counts, successful optimizer updates, update decisions, skip reasons, and completion receipts. Requested optimizer updates unmet because of starvation remain partial completion, with `rollout_complete` recorded separately
- Legacy custom policy-gradient arms mask prompt positions and train only actual sampled response targets. The final unsampled target is removed. Their sampling clients refresh after every successful optimizer update; persistence cadence does not determine the generating policy
- Cell-runner labels describe implemented mathematics, not canonical DAPO/GSPO/GRPO equivalence. DAPO is an advantage-clipped surrogate; GSPO is a response-mean surrogate; no behavior-ratio correction or dynamic sampling is claimed. Modal's best-of-four successful-response update is `success_filtered_sft_best_of_4`

## Shared grading and metrics

`gsm8k-final-visible-box-or-number-v3` accepts plain text or one conversational assistant-message representation. Batches require exactly one gold answer per completion. It removes closed thinking spans; unterminated thinking cannot supply an answer. The last recognized box is authoritative, including malformed boxes. If no box exists, the final `####` segment / last numeric literal is used. Commas are accepted; numeric equality uses relative and absolute tolerance 1e-9. Non-numeric GSM8K answers are rejected. A wrong box cannot be rescued by an incidental number.

ZVF is the fraction of prompt groups having zero population reward variance. A homogeneous success group and a homogeneous failure group are both starved. N2's mean population variance and Mega's scaled plug-in Bernoulli variance have different field names. Filtered and pre-filter populations must have different metric-definition IDs.

## Immutable specifications and evidence

`framework_contracts.experiment_spec` makes a detached, deterministic JSON specification and SHA-256 hash. Live probe, N8, Wave 6 and parallel-run resumption validate identity before reuse. Mega IDs include consequential generation and scorer settings, and old family-level inventory coverage never skips a new cell. Resumptions preserve prior traces. New default N10/Mega/Wave 6/parallel output locations are versioned; live cells refuse to overwrite existing attempts without explicit compatible resume.

Three records are distinct:

1. Requested specification: desired treatment and its content hash
2. Resolved configuration: actual initialized framework/environment values; managed or missing values are explicitly unavailable
3. Completion receipt: requested/observed updates, exact trainer ID, final checkpoint, status and evidence errors

`validate_comparison` additionally requires equal resolved model/tokenizer/data revisions, rendering/template, train/dev/test IDs, scorer, implemented objective, adaptation scope, reduction, KL, sampling, optimizer, budget, selection, environment and source revision, disjoint exposure, and observed prompt/completion/token counts. The current heterogeneous wrappers are **native-recipe comparisons**, with `comparison_contract_satisfied=False`. A successful subprocess or plausible algorithm label is insufficient. This offline patch does not establish runtime equivalence.

## Identity, export and reports

Exact references are not training ownership. Registry matching requires an explicit training-owner ID and unambiguous full model identity; competing/missing/conflicting identities remain quarantined without borrowed metrics. Bare UUIDs never gain invented training suffixes. HF evidence remains file- and immutable-revision-local.

Adapter exports require verified completion, matching checkpoint identity, model revision and config hash. Branch identities include source treatment/checkpoint; idempotency requires a matching receipt and destination file hashes. Export commands must succeed. Indexes link only verified immutable commits. Results-only repositories are labelled as metadata, not adapters.

H2H summaries require an explicit campaign manifest, reject incompatible duplicate records and preserve missing metrics. Claim-table validation is `SCHEMA_PASS_ONLY`; it is not proof that a claim is supported. Audit aggregation requires compatible completion states, comparison identities and per-metric definitions, and reports exclusions and per-metric counts.

## Boundaries not established offline

Historical effective configs, source/provider image digests, immutable model/tokenizer/data revisions, managed training/sampling hardware, provider determinism, historical initialized Tinker/SkyRL settings, live export byte provenance and controlled final evaluation were not obtained. Do not substitute local hardware or package assumptions. HumanEval program construction is corrected and compile-checked, but execution fails explicitly until a trusted isolated evaluator is supplied; a plain subprocess is not an isolation boundary. No training, provider invocation, paid job, push, or historical regrade is part of this patch.

## Source artifacts found in the full checkout

The review bundle omitted files that are present in the full checkout. `framework_config_dumps/` contains nominal YAMLs with placeholder model hashes and managed nulls; these are not linked initialized-runtime receipts. The delegated Tinker GSM8K preset requests G4, batch 2 and LR 3e-5 before the wrapper's model/steps/seed overrides; its source hash and requested values are now retained. SkyRL's source YAML has duplicate trainer keys and declares Qwen2.5-1.5B-Instruct while the wrapper is labelled Qwen3-8B. The wrapper now rejects duplicate keys or model mismatch before launching. Existing YAML artifacts are preserved unchanged.

## Independent runtime-audit amendment

These guards enforce v3 validity; they do not change the objective, scorer, requested population or reinterpret stored results.

- All populated explicit W&B owner aliases must contain the same full training ID. Conflicting aliases cannot own either run or transfer metrics to them
- Wave 6 failures and process crashes carry the same v3 requested specification as successful receipts, so compatible attempts remain retryable. Observed in-worker progress is retained; process-crash progress is unavailable, not invented as zero
- Live `--resume` without a saved training-state URI fails before initialization or artifact mutation. A fresh start requires a new attempt tag; old traces cannot be mixed into a restart
- N8 binds each output stream to its own `passrates.spec.json` or `passrates_smoke.spec.json`. Existing shared manifests are read-only legacy evidence, accepted only after exact specification validation. Even an empty stream's existing manifest cannot be replaced with a different treatment
- Campaign, Wave 6, N8 and Mega require exactly the requested number of nonempty completions per sampling response. Malformed responses are failures, never zero-valued completed observations. N8 and Mega may use their existing bounded retries; partial response contents never accumulate toward a valid group. Mega also requires the fixed prompt count and aligned reward/length vectors
- HumanEval assembly retains body-only completion indentation, including fenced blocks and nested helper definitions. Both full-function and body-only fixtures are compile-checked only; execution remains blocked pending a trusted isolated evaluator
