# E3 — SDAB (`sdab_eval`) — lane status 2026-08-09

**Status: BLOCKED. Score: null.** No SDAB task data was obtained, no environment
was launched, no grader was run. The block is provider access, not local capability.

## What now runs

| Thing | State |
|---|---|
| Focused test suite | **50 tests pass** (was 27). `flagship.test_pavlov_sdab_eval_adapter` + `flagship.test_eval_pavlov_sdab` |
| Preflight gates | **4 of 5 PASS.** Only `exact_native_sdab_runtime` is BLOCKED |
| Isolated environment | **Built.** `outputs/e3_sdab/venv-sdab-e3`, Python 3.12.13, `tinker` 0.24.1 + `wandb` 0.21.0 |
| 80-task bundle ingestion | **Implemented and tested** against a synthetic fixture: schema validation, task-ID hashing (both hash schemes), split-manifest construction, disjointness proof, runtime-manifest emission, receipt emission |
| Access request | **Written and sendable as-is:** `ACCESS_REQUEST_SDAB_2026-08-09.md` |

### Commands that work

```bash
cd /Users/arvind/Developer/agentic_repos/tinker-rl-lab
V=outputs/e3_sdab/venv-sdab-e3/bin/python

# 1. tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=zvf-program $V -m unittest \
  flagship.test_pavlov_sdab_eval_adapter flagship.test_eval_pavlov_sdab

# 2. preflight (zero side effects, zero Tinker calls)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=zvf-program $V -m flagship.eval_pavlov_sdab \
  --preflight --out outputs/e3_sdab/lane_e3_20260809/preflight_receipt_isolated.json

# 3. bundle ingestion, exercised on the synthetic fixture
PYTHONPATH=zvf-program $V -m flagship.pavlov_sdab_eval_adapter \
  --bundle outputs/e3_sdab/synthetic_fixture/SYNTHETIC_bundle_NOT_SDAB_DATA.json \
  --train-task-ids outputs/e3_sdab/synthetic_fixture/SYNTHETIC_train_task_ids_NOT_SDAB_DATA.json \
  --mode harness_validation \
  --out outputs/e3_sdab/lane_e3_20260809/synthetic_ingest_receipt.json

# 4. the moment the real bundle arrives, same command without --mode:
#    --bundle <provider bundle> --train-task-ids <provider train ids> \
#    --source-revision-digest sha256:<...> --container-digest sha256:<...>
```

## What changed

1. **`eval_pavlov_sdab.py` — real bug fixed.** `REPO_ROOT` was `HERE.parents[2]`,
   which resolves to the directory *above* the checkout. The default isolated
   environment therefore pointed at `/Users/arvind/Developer/agentic_repos/.venv-sdab-e3`
   and the isolation gate could never be satisfied. Now `parents[1]`, with the
   default environment root at `outputs/e3_sdab/venv-sdab-e3`.
2. **`pavlov_sdab_eval_adapter.py` — bundle ingestion added.**
   `validate_task_bundle` (80-task schema, unique case-folded IDs, official
   categories, raw content stripped), `newline_task_id_sha256`,
   `build_split_manifest`, `prove_split_disjointness`,
   `build_boundary_spec_from_bundle`, `build_runtime_manifest`,
   `ingest_task_bundle`, `build_ingest_receipt`, plus a `--bundle` CLI mode.
   `build_sdab_boundary` now rejects anything carrying a synthetic marker.
3. **Two hash schemes reconciled.** The runner hashes `"\n".join(task_ids)` and
   returns bare hex; the boundary hashes the canonical JSON array and returns a
   `sha256:` digest. Both are now emitted from one place and a test asserts
   `newline_task_id_sha256(...) == eval_pavlov_sdab.task_ids_sha256(...)`, so the
   two receipt families cannot silently disagree.
4. **`tinker` gap closed — it was a packaging problem.** `tinker` is on public
   PyPI (0.24.1, matching the `>=0.24.0,<0.25.0` pin in `pyproject.toml`). No
   private index, no credential. The prior blocker was that the earlier sprint
   built its isolated venv on Python 3.14 with no installs. 323 MB, no torch.

## The synthetic fixture

`outputs/e3_sdab/synthetic_fixture/` — **not SDAB data, cannot produce a score.**

- Every task ID is `SYNTHETIC-NOT-SDAB-NNNN`; the bundle sets `synthetic: true`;
  the adapter raises if the flag and the ID markers disagree.
- `build_sdab_boundary` refuses any spec carrying a synthetic marker, and
  `ingest_task_bundle` *verifies that refusal* before returning — if the guard
  ever regresses, ingestion fails closed.
- `build_runtime_manifest` refuses a synthetic ingest, so the fixture can never
  reach the runner.
- Every ingest and receipt emits `score: null` and `is_model_score: false`.

Observed: authoritative ingest of the fixture exits 1 with
`bundle is marked synthetic; ingest it with mode='harness_validation'`.

## What remains — all four are provider access

| Blocker | External receipt needed |
|---|---|
| 80-task evaluation bundle | Immutable 40/64-hex revision, the 80 task IDs, category + difficulty-variant labels, license identifier or agreement reference |
| Train/eval disjointness | Provider statement of training/development/public task IDs (or an explicit "none exists"), with an immutable reference |
| Native runtime | Container digest, environment/seed digest, cloudbox provisioning path, `module:factory` adapter entrypoint (`list_tasks`/`evaluate_task`/`verify_result`), and the leaderboard agent scaffold + action budget |
| Native grader | Verifier identity/revision/sha256, behavioral-test + rubric + state-validation digests, judge model and config, composite weighting |

Evidence that this is the *only* thing left: running preflight against a
shape-only runtime manifest passes every schema, hash, digest, and disjointness
check and fails on exactly one line —
`cannot import provider-native SDAB adapter emulated_sdab.runtime:create_runtime: No module named 'emulated_sdab'`
(`lane_e3_20260809/shape_only_manifest_preflight_summary.txt`). That manifest was
written to the session scratchpad, never to `outputs/`, so nothing runner-loadable
and non-authentic exists on disk.

## Cost note worth escalating

The E3 budget model does not survive contact with the real benchmark. At the
configured ceilings (`max_actions=8`, 4096 prompt / 256 response tokens) a 1-task
slice projects to $0.0204 and a full 80-task run to **$1.63 — already over the
$0.50 E3 cap**. And those ceilings are not realistic: the provider states tasks
run **up to 12 hours** against live infrastructure. The budget must be re-derived
from the provider's actual scaffold and action budget before any paid run.

## Single next action

Send `outputs/e3_sdab/ACCESS_REQUEST_SDAB_2026-08-09.md` to `founders@emulated.so`.
Nothing local unblocks this lane.
