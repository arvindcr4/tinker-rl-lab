# MIN-REPORT-RL / 2.0: authenticated provenance publication policy

This protocol separates reporting completeness, authenticated approval, and
artifact consistency. It does **not** establish that an experiment actually ran,
that a signer is honest, that a dataset has no semantic contamination, or that an
experiment has adequate statistical power.

The publication rule is exact: `accepted = schema_valid AND authenticated AND
all_required_evidence_checks_pass`. There is no grade-B exception, score-based
integrity exemption, unsigned publication mode, or automatic algorithm downgrade.

## Entry points and backward compatibility

- `minreport.audit_file(record_path, trusted_key)` is the publication gate. Trust
  is a caller-supplied Ed25519 public-key file, never a key inside the record.
- `minreport.verify(record, strict=False)` is an in-memory, **unauthenticated
  diagnostic**. It always returns `accepted: false`. The old `strict` argument is
  accepted for compatibility, but cannot supply authentication.
- `minreport.py verify RECORD --trusted-key PUBLIC_KEY` is the publication CLI.
  Missing/invalid signatures or any evidence gap return exit status 1.
  `--strict` is a compatibility alias for this already-strict policy.
- `--diagnostic` permits an unsigned diagnostic command to return status 0; its
  JSON and terminal output still say `accepted: false` / `PUBLICATION: REJECT`.
  `--strict --diagnostic` is rejected.
- `sign.verify_file` verifies authentication **only**. It must not be substituted
  for the publication gate. `sign.sign_file` first requires publication-ready
  evidence, then writes a new detached signature without replacing an old one.
  `sign.sign_record` is a lower-level strict-JSON authentication primitive; it
  deliberately makes no evidence-validity or publication claim.

Historical `MIN-REPORT-RL/1.0` records and signature-v1 envelopes are diagnostic
inputs only. Their previous letter grades are not current acceptance evidence.
They are not rewritten, upgraded, re-signed, or supplemented with invented
artifacts. A new v2 record requires actual retained evidence and explicit approval
by an appropriately trusted signer.

## Record and shared schema

The envelope uses `spec_version: "MIN-REPORT-RL/2.0"`. Required top-level fields:

- `run_id`: nonempty identity shared by its completed run artifacts
- `min_report_schema`: `registry-v3#/$defs/min_report`
- `min_report_schema_hash`: full SHA-256 of the shared MIN-REPORT definition and
  its referenced nullable definitions
- `min_report_rl`: the canonical seven-item block from
  `../schema.json#/$defs/min_report`, also used by registry generation/ingestion
- `provenance`, `rigor`, `results`, and `evidence`: objects described below

Canonical MIN-REPORT names are shared, including sampler/telemetry **objects**,
`reference_kl.kl_beta`, `group_size_schedule.initial_g`,
`heldout_split.disjoint_from_reward_env`, and `heldout_split.n`.
`loss_form.clipping_applied: false` means clipping is disabled and both epsilons
are inapplicable/null. This must not be represented as epsilon zero.

Generation fills unsupported fields with `null`. It does not default held-out
separation, decontamination, sampler selection, telemetry, or loss settings to
favorable values. Explicit missing config paths fail. Known historical aliases
such as `beta`, `G`, `constant`, and `disjoint` are mapped to canonical field
names; a telemetry-name list cannot prove any telemetry was saved and is not
converted into positive reporting flags. The original supplied runtime config
is retained in the record. Existing output records are never overwritten.

### Coverage has named, fixed denominators

The registry and provenance auditor call the same `query.badge` implementation.
`coverage_version` is `reported-substantive-leaf-mean-v3`:

- `reported_leaf_coverage_v3`: the registry badge divided by 100. The badge is the
  rounded percentage of the equal-weight mean of seven item leaf-completeness
  fractions. Each item uses the fixed substantive fields in the shared schema;
  `source` and `description` do not earn credit. Explicit disabled clipping
  removes only the two inapplicable epsilon leaves from that item's denominator.
- `item_presence_coverage_v3`: the fraction of the seven items with at least one
  reported substantive leaf, rounded to four decimal places

`score_pct` is reporting completeness only. Every rejected or unauthenticated
record has `grade: "REJECTED"`, even when reporting completeness is 100%.
Acceptance requires every item to be complete and every other check to pass.
The checklist is fixed; missing inputs create failed checks rather than removing
checks from a denominator. `checks[].weight` is retained as diagnostic metadata,
not an acceptance formula.

## Immutable anchors and repository scope

`provenance` contains:

- Full `git_sha`, obtained with `git -C REPO_ROOT rev-parse HEAD`. Verification
  requires the same full repository HEAD, independent of the caller's cwd
- `code_file` and full `code_hash` (`sha256:` plus 64 hexadecimal characters)
- Full `reward_fn_hash` of the sole recognized reward function's entire source
  segment; ambiguous/missing functions fail. The whole source hash separately
  detects changes outside that function; this does not prove transitive runtime
  dependencies or the actually executed implementation
- `resolved_config` and full `config_hash` of its strict canonical bytes. This
  retains exact runtime configuration content; MIN-REPORT, source and dataset
  declarations must agree with it. Generation excludes the reporting fields
  `rigor`, `results`, and `evidence` from this runtime-config digest to avoid a
  cyclic digest dependency through run artifacts. Those fields are covered by
  the complete record signature instead
- `worktree`: format `git-tracked-worktree/1.0`, full Git tree identity,
  full SHA-256 of the binary tracked diff against HEAD, and its `dirty` flag.
  The current tracked worktree must match. Untracked dependencies are outside
  this scope; the selected source is separately content-hashed even if untracked
- `data_ref`: a dataset label, explicitly **not** a dataset-content digest

Source/evidence paths must resolve inside the repository, including after
symlink resolution. The historical `experiments/...` to
`platform_hybrid/experiments/...` alias remains supported for diagnostics.
External absolute paths and missing sources produce structured failures.

This policy checks the current checkout against the anchored record. Verification
in an unrelated or subsequently modified checkout rejects; it does not silently
reinterpret an old commit or accept source drift. Reconstruct the recorded
checkout and retain evidence separately to audit historical work.

## Required artifact evidence and semantic checks

Each `evidence` reference is `{"path": "repo/relative.json", "sha256":
"sha256:<full digest>"}`. Every referenced file is read with the shared strict JSON
loader and its raw bytes are hashed. This detects even whitespace changes to
artifacts. No verifier creates or repairs experiment evidence.

### Runs and minimum replication

`rigor.n_seeds` must be an integer, not a Boolean; `rigor.seeds` must contain
exactly that many distinct integer seeds, with at least three. This is called the
**minimum distinct-seed requirement**, never a statistical-power result.

`evidence.runs` contains one distinct completed run artifact per seed. Each has:

- `run_id`, integer `seed`, and `status: "completed"`
- `code_hash` and `config_hash` equal to the record's full anchors
- `split_sha256` equal to the saved split reference
- `heldout_metrics`: finite numeric metrics including `results.metric`

The seeds, count, artifact identities, and values must agree. Flags such as
`rigor.reports_heldout` cannot substitute for those artifacts.

### Split and decontamination

`evidence.split` points to `{data_ref, train_ids, heldout_ids}`. Both identifier
lists must be nonempty and internally unique, and their intersection must be
empty. Their held-out count must equal the positive integer
`min_report_rl.heldout_split.n`. The data label and both declared disjointness
flags must agree with the artifact.

`evidence.decontamination` is an independently hashed check report with
`method: "split-id-disjoint/1.0"`, `performed: true`, `verified: true`,
`split_sha256`, integer `overlap_count`, and a Boolean
`parser_robustness_probe`. Its overlap count is recomputed from the saved IDs;
performed/probe claims must agree with the MIN-REPORT block. An explicit
`verified: false` fails. **This verifies identifier disjointness only.** Content-
level contamination and adequacy of any declared parser probe require separate
scientific review; this protocol does not claim to have established either.

### Results and uncertainty

`results` contains `metric`, ordered `per_seed` entries `{seed, value}`, `mean`,
`sd`, two finite ordered `ci` endpoints, and `confidence` strictly between 0 and 1.
`evidence.analysis` contains those summary fields, integer `n`, a nonempty
`method`, and `input_run_sha256` matching the exact ordered completed-run digests.

The verifier recomputes the sample mean and sample standard deviation from the
completed run metrics, checks the per-seed reported results, verifies the sample
count and the saved interval/confidence against the reported interval/confidence,
and checks that the interval contains the mean. `rigor.reports_ci` must be true.
The numerical mean/SD recomputation uses relative tolerance 1e-12 and absolute
tolerance 1e-15. All other configuration/identity comparisons are exact.

This checks that uncertainty was actually saved consistently; it does not
recompute arbitrary analysis methods or certify CI coverage, precision, power,
causal interpretation, or a scientific conclusion. Those remain analysis-review
questions, even for an accepted record.

## Signature-v2 wire protocol and trust

`../json_protocol.py` is the single strict JSON loader/canonicalizer used by
signing, verification, and registry ingestion. Duplicate object names,
NaN/Infinity, overflow-to-infinity numbers, non-string object keys, non-JSON
in-memory values, and unpaired Unicode surrogates are rejected before signing or
verification. Encoding uses `allow_nan=False`.

A detached `RECORD.sig` has exactly two members: `protected` and `signature_b64`.
The protected header has exactly:

- `spec: "P5-PROVENANCE-SIG/2.0"`
- `canonicalization: "python-json-sortkeys-ascii/1.0"`
- `algorithm: "ed25519"`
- `key_id: "sha256:<SHA-256 of raw public-key bytes>"`
- `digest_sha256`: full hex SHA-256 of the canonical record bytes

The signed bytes are the ASCII domain `P5-PROVENANCE-SIG/2.0` followed by one NUL
byte, followed by canonical JSON of `{"protected": HEADER, "record": RECORD}`.
Thus protocol, algorithm, canonicalization, key fingerprint, digest and every
record field are authenticated. Adding unprotected assurance fields is rejected.

Canonical bytes are Python `json.dumps` with sorted keys, compact separators,
`ensure_ascii=True`, `allow_nan=False`, encoded as ASCII. Version 1 is explicitly a
Python serialization contract, **not RFC 8785/JCS or a cross-language promise**.
Any future canonicalization change requires a new identifier/protocol policy.

The verifier requires caller-supplied trust and an allowed algorithm set. This
version only implements Ed25519; an envelope cannot select HMAC, supply its own
trust anchor, or override verifier policy. Missing `cryptography` is a hard
failure, never a fallback. Assurance is derived from the successfully verified
algorithm, not a document's self-reported public-key flag. Trust distribution and
revocation remain the caller/operator's responsibility.

### Key provisioning and rotation

`sign.py keygen --key-dir NEW_DIRECTORY` creates a private directory with mode
0700. The private key is opened with `O_CREAT | O_EXCL`, mode 0600 **at creation**;
there is no world-readable write-before-chmod interval. Keys use strict base64 of
raw Ed25519 bytes. The public key is also retained under its full fingerprint.

Key generation refuses if either configured key exists. `force=True` does not
permit in-place replacement. Provision a distinct directory for rotation and
retain historical public keys/fingerprints. Signing requires an existing private
key and matching configured public key, secure directory/secret permissions, and
no symlinked key paths; it never initializes or replaces an identity.

## Non-destructive output and example workflow

Audit output defaults to `RECORD_NAME.audit.json` appended to the complete
filename, so extensionless inputs and `.json` ancestor directory names are safe.
An explicit `--out` resolving or hardlinking to the input, signature, selected
source, any evidence artifact, or trusted public key is rejected. Output symlinks
are rejected. Audits are written to a temporary file and atomically replaced.
Auditing never updates records, keys, signatures, or artifacts.

For a newly collected run with all required artifacts already retained:

```sh
python platform_hybrid/registry/provenance/minreport.py generate \
  --run RUN_ID --config runtime-and-evidence.json --out new.provenance.json
python platform_hybrid/registry/provenance/minreport.py verify \
  new.provenance.json --diagnostic
python platform_hybrid/registry/provenance/sign.py keygen --key-dir /private/new-identity
python platform_hybrid/registry/provenance/sign.py sign \
  new.provenance.json --key-dir /private/new-identity
python platform_hybrid/registry/provenance/minreport.py verify \
  new.provenance.json --trusted-key /trusted/provenance_ed25519.pub
```

The last public key must already be trusted through an independent distribution
process; copying a key from an untrusted record is not sufficient. A diagnostic
run or a valid signature alone is never publication approval.

## Integration with native-contract input guards

The native-contract branch's typed-section, nonempty-hash, distinct-integer-seed,
and explicit-Boolean guards remain enforced by the v2 schema and artifact checks.
Full reward-function hashes and `data_ref` remain mandatory; v1 partial hashes,
legacy compact reporting forms, or unsigned diagnostics cannot bypass the v2
publication gate. Historical records are inspected without migration or edits.
