# Versioned Colab artifact preservation

These wrappers now run `samestack-gsm8k-cot-v3-policy-exact-eos`. The historical
v2 filenames in this directory are entrypoint compatibility names, not permission
to reuse legacy results as evidence for the corrected protocol. No GPU training
or real Colab session was executed to validate this change.

## Launch contract

From the repository root, with an explicitly authorized Colab session:

```sh
python platform_colab/samestack_v2/drive_sessions.py SESSION grpo_g8:42 ppo_8x8:42
# Add --new only when creation of a new L4 session is intended.
```

The driver captures the observed Git commit, dirty-tree hash and source-file
hashes; uploads a checked source archive; and calls the platform-neutral
`run_arm_local`. It never impersonates the Modal package. Both platforms use
`samestack_dependencies.py` for exact package versions. Its status is explicitly
`candidate-not-gpu-validated`: matching versions is not evidence of matched GPU
numerics or a verified GPU installation. Missing or mismatched packages are
reconciled together and verified, with a restart required if incompatible
packages/source have already been imported into the persistent kernel.

The core resolves model, tokenizer and dataset commit revisions before use and
records effective configuration, ordered data IDs, source, dependency versions,
hardware, generation settings and schedules. Protocol/config/source/environment
hashes bind the completion identity. An incompatible existing artifact fails
closed and is retained. This is reuse of a **finished artifact bundle**, not
optimizer/RNG recovery or continuation of interrupted training.

## Preservation and verification

Each run must contain:

- `<arm>_s<seed>.json` with its complete immutable manifest
- `<arm>_s<seed>_texts.json.gz` with both pre/post completion arrays and ordered IDs
- Adapter weights and `adapter_config.json`
- Tokenizer configuration and tokenizer vocabulary/model
- Saved train/eval generation configurations matching the effective manifest
- An atomic `<arm>_s<seed>.complete.json` inventory with file sizes and SHA-256 hashes

Downloads go to a hidden temporary per-run directory. The shared validator checks
all files and hashes before atomic promotion to the visible run directory.
A lone downloaded JSON never causes a skip. Retrying an interrupted transfer
asks the core to validate the remote completed run again, then retries the bundle
download. Previous and partial local evidence is retained, and the Colab session
is stopped only after all requested artifacts have been independently preserved.

```sh
python platform_colab/samestack_v2/verify_and_merge.py
python platform_colab/samestack_v2/verify_and_merge.py --merge
```

Default staging/canonical directories are under the new protocol namespace.
Use explicit `--stage`, `--canonical` and `--report` paths for offline inspection
of other directories. The verifier checks filename identity, allowed arms/seeds,
recipe/config hashes, binary arrays and exact lengths, ordered item identities,
recomputed summary metrics, schedules, environment hashes and duplicate
identities. Empty staging is unsuccessful. Missing legacy provenance is reported
unverified, never fabricated, and unverified artifacts cannot merge. Existing
historical files are not rewritten.

## Offline regression checks

```sh
python -m pytest tests/test_samestack_artifact_contract.py
```

All fixtures are explicitly synthetic. Tests cover the contradictory C5 review
record, malformed/non-finite JSON, unchanged `NaN` inside text strings, missing
artifacts, source changes, duplicate identities, incomplete downloads, retry,
and session-stop safety. They do not establish Colab CLI/network behavior or GPU
training compatibility.
