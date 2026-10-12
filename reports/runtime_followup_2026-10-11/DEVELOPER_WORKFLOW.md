# Current developer workflow (2026-10-12)

This guide describes the current engineering surface without revising frozen thesis results or claiming canonical backend equivalence.

## Start here: one offline CPU command

From the repository root, with the locked development environment already installed:

```sh
.venv/bin/python tools/run_offline_harness.py --out /tmp/tinker-offline-harness-001.json
```

Use a **new output path for each invocation**. Expected final status is `OFFLINE HARNESS: PASS (5 checks; 0 skipped)`. The command takes roughly tens of seconds on the development VM, uses random tiny CPU models, downloads no weights/data, and starts no provider job. The tests use in-memory provider, tracking and persistence doubles and forbid socket connections. This is a developer integration harness, not a pretrained-model, LoRA, benchmark-quality or remote-durability demonstration.

If dependencies are missing, the command exits with an actionable error before running checks. To provision a new authorized development environment, use the repository lock:

```sh
uv sync --frozen --extra dev --python 3.12
```

Select Python 3.12 explicitly for the VM-validated setup. The repository has no `.python-version` pin; [uv otherwise selects a suitable interpreter according to project requirements](https://docs.astral.sh/uv/reference/cli/#uv-sync--python), which is broader than this tested environment. An offline dry-run of the command above checked 141 packages and required no changes in the current VM. This is not a claim that every supported Python/platform combination was freshly installed.

Installation can download substantial dependencies; the offline command itself never installs anything. Run it with the resulting `.venv/bin/python`, rather than an unrelated system Python. No service credentials are needed.

### What the command actually exercises

1. A real JSON configuration passes through the direct baseline CLI's `--check-config` path; its fingerprint is compared with the configuration used for the run.
2. The CLI runs a 20-step random tiny Qwen3 CPU policy against a synthetic one-action task. The dataset/tokenizer and external transports are offline test adapters; the training loop, SDK loss bridge and Torch/Adam computations are real.
3. Evaluation uses the final saved sampler snapshot. Three sampled successes and one deliberately injected transport failure produce a held-out mean of 0.75 with a denominator of four. This is a failure-accounting oracle, not measured benchmark accuracy. The local categorical sampler implements the toy test's temperature/top-p transformation; it does not attest a live provider's sampling implementation.
4. A completed JSON receipt is checked against its configuration and evaluation evidence. Replay returns the same result without a provider call or rewriting the receipt.
5. Companion integration tests check sampled learning for two targets, fresh-object actor/Adam resume, recovery after a lost optimizer response using the last durable checkpoint, critic resume, and a multi-epoch GSPO update against an independently written clipped-ratio objective.

The five pytest cases jointly cover these stages. A skipped test, missing/invalid test report, nonzero exit, timeout, changed selected source bytes, or unexpected test count cannot produce a passing receipt.

### Receipt and failure behavior

The fresh JSON receipt records pass/fail, check counts, dependency versions, selected source hashes, elapsed time and a bounded diagnostic tail. Hashes identify the selected local source bytes; they are not a signature, full dependency lock, or remote authenticity proof. Final publication is an atomic same-filesystem link with no overwrite fallback. Existing files, dangling symlinks and competing new receipts are preserved. Unsupported link capability fails before the CPU checks.

- Exit 0: all five required cases passed without skips and source bytes stayed stable
- Exit 1: checks failed/timed out, or local execution/publication failed
- Exit 2: invalid arguments, existing output path, or missing dependencies

A test failure normally creates a `failed` receipt. A preflight/publication failure may produce no receipt; the terminal error explains why. The diagnostic tail is local; review it before sharing. Generated model/checkpoint artifacts live in pytest temporary storage. The durable deliverable is the local harness receipt, not a downloadable trained model.

### Clean-source portability check

The five-case command also passed in 14.507 seconds from a separate 20-file tracked-source archive of commit `412d9293047ae272cae072f3f49dede066070c00`. Every copied file matched its Git blob, and the selected project modules resolved inside that archive. The existing Python 3.12.14 VM dependency environment was reused; this checks source portability, not a hermetic environment or fresh dependency installation. No untracked model/data assets were copied. The harness source manifest now additionally includes its CPU optimizer helper, test configuration and project configuration so their changes are visible in later receipts.

## Inspect a real baseline request without launching it

```sh
.venv/bin/python -m platform_tinker.tinkerrl.grpo_cli --preset gsm8k --check-config
```

This checks local configuration and adapter selection only. It does not fetch datasets/tokenizers, check credentials or provider availability, attest tracking/export, or validate scientific claims. The output whitelists numeric/boolean settings and a configuration digest rather than printing endpoints or private identifiers.

The documented module command reports common invalid-value, unknown-config-field and missing-file errors concisely with exit code 1 and the correct module help command. Argument-parser usage errors retain exit code 2. Programmatic `main()` callers still receive their original exceptions; unexpected runtime failures are not turned into successful or configuration-check receipts.

Removing `--check-config` changes the action: direct execution can contact tracking, Hub and training services, publish checkpoints and consume provider resources. It requires a separately authorized run and appropriate user-managed credentials. The offline developer command does not grant or test those capabilities.

## Inspect a source-validation evaluation request

```sh
.venv/bin/python -m platform_tinker.tinkerrl.pavlov_portfolio_eval --base-model --per-suite 2 --out /tmp/portfolio-inspection-001.json --check-config
```

This prints a local plan for four examples (two per source suite), a request fingerprint and explicit unchecked capabilities. It does not create the output or its parent directory, inspect credentials, load data/tokenizers, or contact a service. Existing output paths still fail so a subsequent authorized run cannot casually overwrite prior evidence. For a trained sampler, the existing sampler/Hub-reference requirements apply; inspection does not attest that those references point to the same weights. Private sampler, Hub and output identifiers are hashed rather than echoed. The request digest includes the tokenizer/source revision requests and fixed sampling policy, and is recorded in subsequent execution/tracking receipts so the same request can be matched to its inspection. It is not a weight or dataset-content attestation.

Removing `--check-config` invokes the existing external evaluation workflow and requires separate authorization and user-managed credentials. A successful plan does not establish example availability, writable storage, model identity, provider cost, scientific validity or a completed evaluation. This is portfolio source validation, not an E1–E14 benchmark run.

## Avoid duplicate local training launches

Active direct-baseline runs take a nonblocking advisory lock beside their seed checkpoint (`.json.lock`). A second cooperating local invocation using the same checkpoint fails before tracking/provider setup or changing the active run's Torch RNG. Wait for that run, or choose a genuinely separate experiment name/directory; different seed checkpoint paths are independent.

The lock marker intentionally remains after release. Its existence does not mean a run is active, and deleting it is not a safe way to unlock a run. The operating system releases ownership when the descriptor or owning process closes. Lock files must be nonsymlink regular files on storage supporting Unix advisory locks. This is local exclusion, not a distributed lock or a provider transaction.

Completed-cache replay uses one committed checkpoint snapshot without acquiring a write lock or rewriting that checkpoint. It still performs the existing validation and configured tracking checks. This protection does not authorize external training/evaluation or establish that provider-side resume is exact.

## Audit base/trained receipts before comparing them

```sh
.venv/bin/python tools/audit_portfolio_comparability.py --base BASE.json --trained TRAINED.json
```

This stdlib-only command reads two existing portfolio source-validation receipts without rewriting them. It checks strict JSON, recorded row/aggregate arithmetic, suite accounting, ordered prompt/token hashes, source/tokenizer revision requests and sampling controls. Output identifies the exact input bytes by SHA-256 without printing prompts or new score differences.

The current v1 format does **not** bind target contents or scorer implementation. Therefore even matching recorded settings yield overall `status: unknown`, `recorded_configuration: compatible` and **exit code 3**, never a readiness certificate. Missing historical metadata also remains unknown, without backfilling invented values. The audit deliberately does not provide a passing overall-comparability result for this incomplete schema.

Known cohort/policy mismatches return `incompatible` with exit 1; malformed/contradictory evidence returns `invalid` with exit 1; argument-usage errors return exit 2. A matching configuration does not establish weight identity, decontamination, representativeness, causal improvement or statistical significance. Do not suppress the nonzero status to label a comparison scientifically ready. Existing source scores and historical results remain unchanged.

Inputs must be nonsymlink regular files of at most 8 MiB. The tool does not load datasets, inspect credentials, contact a provider, choose a new scorer, regrade outputs or compute a new metric. Review its local diagnostic output before sharing it.

## Backend and demonstration matrix

| Entry point | Current supported purpose | Important limit |
| --- | --- | --- |
| `tools/run_offline_harness.py` | Fixed offline CPU CLI/training/evaluation/resume checks and local receipt | Random toy models; in-memory service doubles |
| `submission/demo/demo.sh` | Existing stdlib offline group-math and historical-artifact integrity demo | Does not train a model or prove benchmark improvement |
| `platform_tinker.tinkerrl.grpo_cli --check-config` | Direct baseline configuration inspection | No provider/data/weight/scientific verification |
| `tools/audit_portfolio_comparability.py` | Read-only arithmetic and recorded-input comparison | Current v1 target/scorer identity gaps force overall UNKNOWN |
| Portfolio evaluator with `--check-config` | Read-only sampling-request and fresh-path inspection | No source availability, weight binding, cost or evaluation attestation |
| Direct Tinker baseline without `--check-config` | Independently configured baseline execution when separately authorized | Not canonical clipped-PPO equivalence; live operation was not run in this session |
| `platform_local.unified --framework tinker --dry-run` | Read-only route/configuration inspection | Execution is guarded because canonical fields are not faithfully implemented |
| `platform_local.unified --framework skyrl --dry-run` | Read-only route/configuration inspection | Server startup is not a completed training recipe; execution is guarded |
| Other native/unified backends | Backend-specific drivers and inspection where implemented | This guide does not certify dependency availability, hardware support or algorithm equivalence |

See [UNIFIED_EXECUTION_LIMITS.md](UNIFIED_EXECUTION_LIMITS.md) for the exact migration boundary. A successful process with no observed optimizer-step receipt must not be reported as verified training completion.

## Broader checks before publishing a change

```sh
make check submission-check
```

This is broader and slower than the offline harness: lint, formatting, typing, coverage, packaging, lock, secret, documentation/public checks and submission checks. Native TeX checks additionally require a working TeX installation. The staged and unstaged tracked patch must pass the secret gate; new files need to be staged before that scan.

Engineering checks passing does not clear the standing scientific `minreport-audit` findings. Historical scores remain unchanged. The scorer-default choice, sampler-refresh cadence, multi-call tool adapter extension, canonical execution fidelity and live-provider validation remain separate open decisions or limitations. No quality conclusion follows from the CPU demo's synthetic score.
