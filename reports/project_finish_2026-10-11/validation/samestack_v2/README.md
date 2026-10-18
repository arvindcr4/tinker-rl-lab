# Same-stack analysis: historical evidence and retrospective repair

The checked-in `v1_recompute.json` and `v2_analysis.json` are **unchanged historical
outputs from the October 2026 analysis**, not outputs of the repaired code. They
must not be described as a verified final analysis. In particular, the old v2
Holm family substituted G2−PPO while the exposure-matched arm was missing, the
old audit used approximate floating-point equality, and the historical runs lack
the immutable provenance now required for a canonical cohort. The previous
"zero disagreements" audit claim applied to the old parser and is not evidence
about a changed grading specification by itself. A separate read-only Python
cross-check of the repaired, documented coefficient grammar found 0 disagreements
among 71,226 pre/post scores from all 27 stored runs. An earlier overly narrow
formatting draft rejected 139 currency/percent/unit answers; preserving the
explicit coefficient grammar removed those representational differences. That
cross-check recorded 210 legacy NaN diagnostic tokens and normalized them only
in temporary audit input, without rewriting original artifacts or inferring
missing provenance.

The repaired training code has a **new v4 candidate protocol**, following the v3
policy, stopping/masking, prompt-schedule and exact-grading repair. V4 rejects
malformed numeric grouping that v3 accepted, uses `boxed-exact-rational-v2`, and
has a distinct protocol ID/hash and therefore distinct seeded prompt streams.
It cannot retrospectively make v1, v2 or v3 runs instances of that protocol. Keep
original result arrays and previous analysis JSON intact; any newly requested
rescore is a separate audit artifact.
No training is launched by these analysis modules.

## Canonical artifact

`analysis_plan.analyze_runs(records, plan=None)` is the pure, shared producer of
`DATA.analysis` (schema version 1). The exporter embeds its output unchanged and
the dashboard consumes that artifact rather than independently pairing rows.
`v2_analysis.analyze(inputs, plan)` returns the same artifact under `analysis`,
plus an independent grading audit, explicitly secondary item diagnostics and
pilot power sensitivity. `load_inputs` performs explicit file I/O; imports have
no CLI, file-data loading, path modification or analysis side effects.

The artifact records:

- The explicit plan and its deterministic hash
- Every selected run ID, arm, seed, environment label and actual environment
  hash, shared-stack fingerprint, protocol hash and ordered test item hash;
  each contrast also carries paired run IDs and exact differences
- Missing planned runs and excluded records with reasons
- Planned/selected/missing counts, endpoint status and interim reasons
- The fixed primary family: `g8-ppo`, `g8-ppo_8x8`, `g8-g2`
- G2−PPO in a separate exploratory family, with no substitution

Duplicate run IDs or semantic identities are errors. Invalid, unverified,
smoke, out-of-plan and incompatible records cannot enter the selected cohort.
The shared run contract validates manifests and recomputed metrics. The plan
must bind `environment`, `environment_hash`, `shared_stack_hash`, `protocol_hash`
and `ordered_test_item_hash`, and its `plan_id` must match each selected manifest's `analysis_plan_id`. Missing
contrasts retain p=1 in **all three** fixed-family Holm calculations: paired t,
sign-flip sensitivity and combined TOST p (maximum of its two one-sided p's).

The actual environment hash covers recorded hardware and dependency builds,
not merely the Modal/Colab label. `shared_stack_fingerprint(record)` computes
`samestack-shared-stack-v1` from source commit, dirty state/hash and source-file
hashes; immutable model/tokenizer/dataset revision records; chat-template hash;
full train/eval generation configs; ordered train/test item IDs; and shared
effective config. Only intentional treatment/replicate fields `arm`, `seed`,
`group`, `prompts_per_step`, `critic` and `advantage_normalization` are excluded
from that config comparison. Missing components produce a null fingerprint and
an unverified exclusion. The fingerprint is derived from recorded provenance,
not accepted from a self-asserted hash; it does not independently attest facts.

File-based v2 loading verifies the complete artifact bundle before accepting a
manifested input. A missing, corrupt or mismatched completion marker raises a
clear error. The filesystem exporter likewise validates completion bundles and
passes failed-loader exclusions into the canonical producer. The pure API stays
filesystem-free: callers pass `completion_validated_run_ids` only after actual
bundle checks and can pass structured `input_artifact_exclusions`. Selected
rows report `completion_bundle_validated`; the artifact's `artifact_validation`
section distinguishes checked and unchecked selected IDs. Unchecked materialized
inputs are not silently presented as preserved, complete filesystem bundles.

`default_plan()` uses method ID `samestack-v3-seed-paired-analysis-v1`, matching
the new runner. This ID names an analysis method; it is **not evidence of a
prospectively frozen plan**. Its default provenance fields are unbound, the
endpoint is undeclared, the planned seeds are the ten listed in the runner, and
`prespecification_status` is `retrospective_repair`. Nothing is selected by
silently inferring a cohort from the available results.

To evaluate a new cohort, supply a plan JSON with the intended seeds,
environment label/hash and shared-stack fingerprint, verified protocol and
item hashes, the unchanged three-member
family, matching method ID and explicit endpoint declaration. The endpoint is
complete only once every planned arm/seed is selected and the endpoint has been
declared. This software only supports retrospective-repair plans:
`final_eligible` and `equivalence_claim_eligible` remain false even at
`endpoint_complete`. Genuine prospective use still requires a separately
reviewed, timestamped pre-outcome plan, fixed selection/stopping rule, scientific
margin justification and appropriate design/power work. Relabeling this plan
cannot establish those facts.

## Statistical interpretation

The primary replication unit is the independently trained seed. Paired t
inference assumes independent pairs and approximately normal paired differences.
The exact sign-flip calculation is a sensitivity analysis requiring symmetric,
exchangeable signs under its null; the training comparison was not a randomized
treatment assignment. Five nonzero pairs have minimum two-sided p=0.0625.
Exact enumeration is limited to 20 pairs and fails explicitly above that limit.

Every paired helper requires at least two finite differences. Constant zero
differences have difference-test p=1, constant nonzero differences p=0. TOST at
an exact boundary has p=0.5; only strict interior intervals support the
unadjusted numeric margin check. Output displays the **95% difference CI** and
**90% TOST CI** separately. The +/-2 percentage-point margin has no supplied
scientific justification and is a sensitivity parameter, not a basis for a
final equivalence claim. The directional 5pp test explicitly tests
H0: mean >= +0.05 versus H1: mean < +0.05 and reports its one-sided 95% upper
bound. It cannot exclude a large negative difference.

Item-only bootstrap/Wilcoxon results are conditional on the observed training
seeds. Pooled McNemar is a nonprimary diagnostic because repeated seed-item
observations are dependent; zero discordance yields p=1. Two-way bootstrap is
an additional sensitivity analysis, not a substitute for seed replication.

MDE/sample-size calculations include both noncentral-t tails and adaptive
bracketing, including n=2. A numerical normal/chi-mixture integration fallback
handles a nonfinite SciPy tail rather than silently dropping it. These are
**pilot-SD, single unadjusted t-test sensitivity calculations**, not Holm-family
or equivalence design power. Actual-design joint seed/item simulation remains
**unmet**: scientifically justified effects, variances/correlations and an
acceptable equivalence margin have not been supplied.

## Independent grading and JSON

The audit implements `boxed-exact-rational-v2` independently of the training
parser. Only an exact `\\boxed` command, optional whitespace and an immediate
balanced brace are accepted. It selects the last complete box and compares
finite Decimal/Fraction values exactly, rejecting malformed commands, zero
denominators, nonfinite values, broad unit stripping and six-digit rounding
collisions. It preserves a narrowly documented full-string GSM8K coefficient
grammar: one currency prefix/wrapper, percent suffix (not divided by 100), or
second/seconds suffix. Decorations inside fraction operands, multiple kinds of
decoration, numeric content in unit text and unrelated prose are rejected.
Thousands separators are validated over each complete numeric atom, after
fraction parsing: commas can separate three-digit integer groups but cannot
appear in fractional digits or exponents. Thus `1,234.5` and
`\\frac{1,000}{2,000}` remain valid, while `42.0,000` and `1e0,000` fail closed.
Training, audit and JS consume
`tests/fixtures/samestack_grading_v2.json`. For manifested runs the audit checks
ordered gold IDs and completion IDs against the manifest before regrading.
The v1 fixture and historical outputs remain unchanged. Demo exports and the
historical audit preserve gold-answer commas for the same validation rather
than silently deleting them before grading.

JSON input rejects NaN/Infinity, overflowed numbers and duplicate object keys;
output uses `allow_nan=False`, with undefined diagnostics represented as null.
Some legacy files contain nonstandard NaN diagnostic tokens: strict loading
rejects them. Do not edit them in place or silently normalize them into verified
artifacts. A separately authorized legacy audit may record that limitation.

## CLI and regression tests

From the repository root, emit a **new** audit to a separately chosen path:

```sh
D=reports/project_finish_2026-10-11/validation/samestack_v2
python3 -I "$D/v2_analysis.py" NEW_DATA "$D/gsm8k_test_gold.json" --plan PLAN.json > NEW_AUDIT.json
python3 -I "$D/v1_recompute.py" platform_hybrid/experiments/results > NEW_V1_AUDIT.json
python3 -m pytest tests/test_samestack_analysis_review.py -q
```

The legacy three-positional-argument v2 CLI remains accepted, but its first
scripts-directory argument is no longer used as an arbitrary import path.
No command here should overwrite the checked-in historical outputs.

## Updating the pending exposure-matched comparison

The frozen pack records 7/10 PPO seeds and a pending exposure-matched PPO arm;
these are historical status statements, not a claim about newly available runs.
Use the canonical selector's completion and exclusion lists for current status.
The `g8-ppo_8x8` contrast is already in the fixed primary dictionaries in
`analysis_plan.py` and `v2_analysis.py`; do not change the family after seeing results.
After independently verifying new completion bundles, rerun the documented CLI
into a new, explicitly chosen output path and inspect the planned/missing counts.
An approved publication refresh must then reconcile thesis §6.5.5/Table 6.2,
Table 1.1, §9.3, abstract, appendices A/E, deck slides 4/5/6/10/12 and their notes,
the brief, viva answers, and reviewer guide. Do not overwrite the frozen pack
or historical outputs as part of a code review correction.

### Run identity version 2

New bundles identify the runtime configuration, source and environment plus the
chat-template hash and both ordered train/test item hashes. `run_identity_version`
is 2; pre-v2 run IDs are not silently upgraded or reused. This versions the
artifact/resume identity rather than changing the training or grading protocol.
Training log-probability and value-head precision remain declared recipe fields
under `effective_config.precision`; hardware capture does not present those
constants as measured tensor dtypes. The model dtype/backend is still compared
with the loaded model before execution.
