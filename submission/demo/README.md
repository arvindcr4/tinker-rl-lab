# TinkerRL MTech Defense Demo

This is the evaluator-friendly demo entry point for the MTech defense. The default run is deterministic, offline, CPU-only, and uses only the Python standard library.

It demonstrates two narrow, auditable things:

1. how rewards inside a completion group become group-relative normalized advantages, including the equal-reward case; and
2. whether the exact bytes and recorded aggregates in `experiments/results/tinker_direct_eval.json` are internally consistent.

It does **not** train a model, reproduce a headline benchmark, establish causal improvement, or treat online reward as held-out accuracy.

## One-command quick start

From the repository root:

```bash
./submission/demo/demo.sh
```

Expected final line:

```text
DEMO STATUS: PASS
```

The command creates deterministic outputs at:

- `submission/demo/output/demo_report.json`
- `submission/demo/output/demo_report.html`

Run the visual defense dashboard locally:

```bash
./submission/demo/demo.sh --serve
```

Then open `http://127.0.0.1:8765/demo_report.html`. The server binds only to localhost and requires no internet connection.

## Evaluator smoke test

```bash
./submission/demo/demo.sh --self-test
```

The tests use `unittest` from the standard library. They check the advantage calculation, equal-reward behavior, fixture contract, artifact SHA-256, aggregate recomputation, JSON extraction, and byte-for-byte deterministic report generation.

## Optional live endpoint mode

The repository instructions require Groq with `kimi-k2-0905-preview` for Python LLM processing. The live mode follows that requirement:

```bash
GROQ_API_KEY=... ./submission/demo/demo.sh --mode live --serve
```

Live mode runs three fixed toy arithmetic/schema checks at temperature zero. It is only a Groq endpoint and JSON-contract connectivity smoke. It is **not** a model-quality result, thesis benchmark, comparison against the offline artifact, or evidence that GRPO caused an improvement. The key is read only from the environment and is never written to the report.

If live mode is unavailable during the defense, use the default offline command. No scientific conclusion depends on the live service.

## Inputs and integrity contract

`fixtures/offline_demo.json` contains:

- four explicitly synthetic reward groups and their expected ZVF/GU contract;
- the exact SHA-256 and expected aggregate contract for `experiments/results/tinker_direct_eval.json`; and
- the three optional live smoke prompts.

The artifact audit recomputes all 10 per-problem reward means, all 10 zero-variance indicators, the mean of all 80 binary rewards, and overall ZVF. A changed artifact, row mismatch, group-size mismatch, or aggregate mismatch fails closed with exit code 1.

Passing the audit proves only exact-byte provenance and internal arithmetic. It does not prove that the recorded completions are correct, that the sample is representative, or that the setup generalizes.

## Exit codes

- `0`: all requested checks passed.
- `1`: fixture, integrity, schema, live-smoke, or I/O check failed.
- `2`: invalid command-line usage (from `argparse`).

## Files

```text
submission/demo/
├── demo.sh                         # single entry command
├── run_demo.py                     # stdlib computation, audit, report, server
├── fixtures/offline_demo.json      # synthetic fixture + artifact contract
├── tests/test_demo.py              # stdlib smoke tests
├── DEFENSE_RUNBOOK.md              # 90-second talk and fallback procedure
└── output/                          # generated, git-ignored
```

