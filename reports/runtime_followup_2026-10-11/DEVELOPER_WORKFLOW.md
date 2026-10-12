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
uv sync --frozen --extra dev
```

Installation can download substantial dependencies; the offline command itself never installs anything. Run it with the resulting `.venv/bin/python`, rather than an unrelated system Python. No service credentials are needed.

### What the command actually exercises

1. A real JSON configuration passes through the direct baseline CLI's `--check-config` path; its fingerprint is compared with the configuration used for the run.
2. The CLI runs a 20-step random tiny Qwen3 CPU policy against a synthetic one-action task. The dataset/tokenizer and external transports are offline test adapters; the training loop, SDK loss bridge and Torch/Adam computations are real.
3. Evaluation uses the final saved sampler snapshot. Three sampled successes and one deliberately injected transport failure produce a held-out mean of 0.75 with a denominator of four. This is a failure-accounting oracle, not measured benchmark accuracy. The local categorical sampler implements the toy test's temperature/top-p transformation; it does not attest a live provider's sampling implementation.
4. A completed JSON receipt is checked against its configuration and evaluation evidence. Replay returns the same result without a provider call or rewriting the receipt.
5. Companion integration tests check sampled learning for two targets, fresh-object actor/Adam resume, critic resume, and a multi-epoch GSPO update against an independently written clipped-ratio objective.

The five pytest cases jointly cover these stages. A skipped test, missing/invalid test report, nonzero exit, timeout, changed selected source bytes, or unexpected test count cannot produce a passing receipt.

### Receipt and failure behavior

The fresh JSON receipt records pass/fail, check counts, dependency versions, selected source hashes, elapsed time and a bounded diagnostic tail. Hashes identify the selected local source bytes; they are not a signature, full dependency lock, or remote authenticity proof. Final publication is an atomic same-filesystem link with no overwrite fallback. Existing files, dangling symlinks and competing new receipts are preserved. Unsupported link capability fails before the CPU checks.

- Exit 0: all five required cases passed without skips and source bytes stayed stable
- Exit 1: checks failed/timed out, or local execution/publication failed
- Exit 2: invalid arguments, existing output path, or missing dependencies

A test failure normally creates a `failed` receipt. A preflight/publication failure may produce no receipt; the terminal error explains why. The diagnostic tail is local; review it before sharing. Generated model/checkpoint artifacts live in pytest temporary storage. The durable deliverable is the local harness receipt, not a downloadable trained model.

## Inspect a real baseline request without launching it

```sh
.venv/bin/python -m platform_tinker.tinkerrl.grpo_cli --preset gsm8k --check-config
```

This checks local configuration and adapter selection only. It does not fetch datasets/tokenizers, check credentials or provider availability, attest tracking/export, or validate scientific claims. The output whitelists numeric/boolean settings and a configuration digest rather than printing endpoints or private identifiers.

Removing `--check-config` changes the action: direct execution can contact tracking, Hub and training services, publish checkpoints and consume provider resources. It requires a separately authorized run and appropriate user-managed credentials. The offline developer command does not grant or test those capabilities.

## Backend and demonstration matrix

| Entry point | Current supported purpose | Important limit |
| --- | --- | --- |
| `tools/run_offline_harness.py` | Fixed offline CPU CLI/training/evaluation/resume checks and local receipt | Random toy models; in-memory service doubles |
| `submission/demo/demo.sh` | Existing stdlib offline group-math and historical-artifact integrity demo | Does not train a model or prove benchmark improvement |
| `platform_tinker.tinkerrl.grpo_cli --check-config` | Direct baseline configuration inspection | No provider/data/weight/scientific verification |
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
