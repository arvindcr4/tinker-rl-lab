# Review 11: candidate publication/audit protocol

Status: prospective candidate, not deployed; no historical result is regraded in place.
Base: `16cde307`. Scope: the validation/publication scripts in review 11.

## Same-stack publication v3

The CLI now requires an explicit immutable selection manifest:

`python -I v2_analysis.py SCRIPTS DATA GOLD --protocol PLAN.json`

The plan must contain:

- `schema_version: samestack-publication-protocol-v3-candidate`
- `primary_contrasts: [g8-ppo, g8-ppo_8x8, g8-g2]`, exactly in this order
- Ordered, distinct integer `expected_seeds`, at least two
- `endpoint: post_accuracy`, `stopping_rule: fixed_expected_seeds`
- `equivalence_margin: 0.02`, a sensitivity margin, not a newly justified scientific equivalence threshold
- `dataset: {revision, sha256}`, binding the complete gold JSON bytes
- `environment`: nonempty `provider`, full `hardware`, `software`, `precision`, `checkpoint_revision`, `tokenizer_revision`, `evaluation_protocol`; `environment_fingerprint` is the SHA256 of its canonical sorted compact JSON; also `run_protocol_version`
- `selected_runs`: one descriptor per file, with `file`, `sha256`, `texts_sha256`, `run_id`, `attempt_id`, `protocol_version`, `environment_fingerprint`

Every run and completion artifact must have the dataset revision and the same ordered `eval_item_ids`. Gold IDs are explicit item IDs when present; otherwise they are `test:<index>:<sha256(canonical question/answer JSON)>`. Both phases' texts, stored binary correctness, and binary truncation arrays must equal the gold length. Counts are derived from those arrays. The selected directory must exactly match the selected-file inventory. Duplicate arm/seed, execution, or attempt identities fail closed. A caller must prepare a selected directory rather than letting filename order choose among attempts.

All selected runs must match the plan's complete environment fingerprint and training protocol version. The fingerprint must bind the complete environment descriptor; an arbitrary GPU family string does not pass the candidate contract. The current implementation checks descriptor/hash equality and immutable run hashes; it does not authenticate provider claims or recover missing historical environment metadata. No platform calibration or transport inference is implemented here.

The analysis emits `samestack-publication-v3-candidate` and `independent-boxed-rational-v2-candidate`. The independent grader uses exact Decimal/Fraction values, real command boundaries, balanced braces and approved numerical wrappers; malformed fractions and unclosed final boxes receive structured failure reasons. It never imports the training grader.

The primary Holm and equivalence sensitivity families never change with available files. Missing seeds or contrasts produce `INCOMPLETE_PRIMARY_FAMILY`; unavailable p-values conservatively occupy family slots at one. `g2-ppo` remains exploratory. Insufficient pairs and zero variance produce null parametric statistics plus reasons. Exact sign flips reject nonfinite or non-vector data; a zero-discordance McNemar observation has p=1 and a stated status. JSON is strict.

Power output is unadjusted paired-t sensitivity plus a conservative alpha/3 family bound and 0.5x/1x/2x pilot-SD sensitivity. It includes both noncentral-t tails and adaptive bracketing. Five-pair sign flips cannot reach .05: the design lower bound is .0625, or .1875 for the first three-hypothesis Holm comparison. Actual joint-family power and equivalence power remain explicitly unestablished; this code does not invent those assumptions.

## Archive replay gate v2

The gate compares the complete candidate publication object recursively: exact keys, types, counts, IDs and ordered sequences; finite floating values use absolute tolerance 1e-12 and no relative tolerance. The required schema includes all four contrasts, both grading variants, inference, power, selected inputs and family completeness. Both pre/post regrading disagreements must be zero. Python optimization cannot disable any acceptance condition or remote payload-integrity predicate.

Platform-calibrated fields, if present, are compared recursively, but this same-stack gate never certifies calibration semantics or transport validity. `platform_calibration_verified` remains false. The missing/excluded calibration module must be reviewed separately.

`accept_archive.py --receipt NEW.json` refuses to replace an existing receipt. A candidate archive needs its bound `publication_protocol_v3.json` and newly produced matching candidate analysis. Old shipped analysis JSON is retained and cannot silently pass as the new schema.

## Reporting evidence audit v4

Discovery reads remain distinct from mandatory `ref`/`raw_ref` reads. References require a lower-case 64-hex SHA256, and the evidence JSON parser rejects duplicate keys/nonfinite tokens.

SWE candidate audits require a hash-bound `task_manifest` with schema `swe-canonical-task-manifest-v1`, the exact 300 canonical IDs, suite and dataset revision. Each audit has a binding containing suite, revision, model, run and evaluator version. Each native report uses `swe-native-task-report-v1` with that binding, one instance ID and a boolean resolution. Reports and native outcomes must correspond one-to-one; native resolution is read from the child and reconciled against the summary. Foreign IDs, reused children and mixed model/evaluator/dataset batches fail. This is a candidate input contract, not a fabricated migration or authentication of absent historical reports.

Both original and replacement measurements are validated. Every score has either `completed_only` or `full_scope_errors_as_zero` denominator policy; passes cannot exceed graded coverage. All ratio fields, exactly 14 lanes and the derived completion/progress lists are validated. Recommendations use finalized measurements. Each published measurement has a claim receipt identifying its result path, evidence depth, known run/grader/protocol identifiers, full consumed-input hash inventory, dependency version, completion and exclusion information. Unknown grader identities remain explicitly unestablished. Summary-only LAB/AgentDojo/BALROG/Omni evidence is not promoted into an independent per-item replay.

Historical Tau snapshots require an explicit `snapshot_mode: historical` and timestamp-based exact cohort equality at their cutoff; later grades are listed separately. The default current mode requires exact equality with raw graded task IDs. Terminal v3 retains its separate terminal identity reconciliation.

Historical E12 completed/failed receipts use `e12-attempt-receipt-v2-candidate`, with complete task/ordinal/run/state/model/dataset binding. Execution IDs must be nonempty and unique; ordinals must be actual integers. Admission remains offline local consistency and never authorizes launch.

## Parser and exact-comparator versions

Historical parser v2/v3 and the float-based reference comparator remain unchanged. `parser_v4.py` explicitly revises wrapper-closure policy; it does not promise to preserve malformed objects previously accepted by v3. `reference_grader_v2.py` is a whole-candidate exact comparator with conflict detection and item-indexed disagreement output. Neither candidate is deployed into a primary evaluation.

The original parser-review and rescue-statistics source spans are retained as `*_historical_v1.py`; `PROVENANCE.json` moves only their filenames, preserving every historical function digest. The public entry points implement candidate validation. Parser review rejects duplicate request IDs at every entry, binds both reviews to the reconstructed canonical packet digest and provides `review_and_decide` for complete packet/annotation validation.

## ZVF and registry candidates v2

The per-group algebraic result is retained. The empirical per-step residual has a sampling plug-in term minus a latent heterogeneity gap; there is no guaranteed sign. Raw within-step prompt structure is unavailable, so the historical arithmetic residual is descriptive. Bootstrap resampling keeps whole seed trajectories, including all repeated steps and group sizes; group-size-specific estimates are emitted. The retained three-seed sample remains weak population evidence.

Beta-binomial output includes optimizer convergence and descriptive fit discrepancy, but no unique mechanism identification. ZVF regression uses mean reward and mean ZVF from the same complete observed history window. Incomplete runs are excluded explicitly. The joint incremental reward test is an HC3 Wald F on the exact same full/reduced cohort, with run identities, rank, restrictions and degrees of freedom. The model-mean count is derived from fitted observations.

Registry dedup uses identical normalized signatures including known/unknown states and delta declarations, never transitive closure of pairwise R0 compatibility. Classes and deterministic representatives are emitted. Coverage median uses the ordinary median and rejects an empty population.

The history exporter remains explicitly limited sampledHistory, not a complete paginated export. It obtains config/summary/history together, rejects registry/config identity drift and missing/duplicate expected steps, includes snapshot limitations and exits nonzero for failed/incomplete records. No credentials were used and no exporter request was executed during this review.
