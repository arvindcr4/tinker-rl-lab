# E10 — `agentharm_eval` lane status, 2026-08-09

**Status: PARTIAL. Score: `null`.** The harness now runs end-to-end on the public
split at zero cost. The AgentHarm score is defined on the held-out
(`test_private`) split, whose three files are absent from the public dataset
revision and gated by the UK AI Security Institute. No score is claimed.

## What now runs

`inspect_evals/agentharm` executes the **entire 176-behaviour public harmful
split**, twice, in ~5 s, with every model set to `mockllm/model` — dataset load,
tool injection, agent scaffold, grading functions, refusal judge, metrics, eval
log. **Zero paid calls, zero network** (`HF_HUB_OFFLINE=1`).

```bash
outputs/_setup/venvs/e10/bin/python outputs/e10_agentharm/harness_validation_run.py
outputs/_setup/venvs/e10/bin/python outputs/e10_agentharm/build_receipt.py
cd zvf-program && ../outputs/_setup/venvs/e10/bin/python -m unittest \
  flagship.test_pavlov_agentharm_frontiermath_adapter -v      # 43 tests, OK
```

| Config | Agent | Samples | Status | avg_refusals | avg_score |
|---|---|---|---|---|---|
| `refusal_agent` | `refusal` | 176/176 | success | 1.0 | 0.0 |
| `default_agent_mockllm` | `default` | 176/176 | success | 1.0 | 0.0 |

Labelled `harness_validation`, `is_model_score: false`, suite `score: null`.
**These numbers are a property of the mock, not of any model.** 8 harm categories
scored, 76 distinct AgentHarm tools instantiated across 652 target-tool slots,
176/176 samples carried a real grading breakdown.

## Receipt binding (built, tested, fail-closed)

`zvf-program/flagship/pavlov_agentharm_frontiermath_adapter.py` gained the E10
split binding: `agentharm_task_id_hash`, `build_agentharm_split_manifest`,
`agentharm_verifier_identity`, `check_heldout_split_available`,
`emit_agentharm_score`. 22 new unit tests.

- Dataset `ai-safety-institute/AgentHarm` @ `e23b3fe6…` (immutable; mutable refs
  like `main` are rejected)
- Public split: 396 task-ID hashes (176 harmful + 176 benign + 44 chat), each
  domain-separated over revision × split × dataset × behaviour ID;
  `split_task_id_hash` `9b09bb35…`, `split_manifest_hash` `304d40b5…`
  (path-independent, so it reproduces on another machine)
- Verifier identity: SHA-256 over 11 grading sources at `inspect_evals`
  `b935c0e5…` → `verifier_hash` `3c158214…`
- **Fail-closed:** `emit_agentharm_score` holds `score` at `null` unless the run
  is a model score on `split="test_private"` with all three gated files present,
  a complete manifest and verifier identity, and an approved policy-grader
  artifact. A `raw_score` from a blocked run is discarded, never carried through.
  Run against real local state it returns `BLOCKED` with 5 blockers.

## What remains

| Blocker | Needs |
|---|---|
| `E10-HELDOUT-SPLIT` | `harmful_behaviors_test_private.json`, `benign_behaviors_test_private.json`, `chat_private_test.json` via `split="test_private"`. AISI grant. |
| `E10-POLICY-GRADER-ARTIFACT` | AISI-approved judge identity + grading revision, with a pinned hash. |
| `E10-JUDGE-MODEL-KEY` | Grader defaults to `openai/gpt-4o-2024-08-06` for both judges — paid, and the paper implies a non-logging deployment is expected for the held-out split. |
| `E10-MODEL-ARTIFACT` | No agent artifact for `Qwen/Qwen3.6-35B-A3B@995ad96e`; that is a paid run this lane may not make. |

Full spec in `PRIVATE_SPLIT_RUN_SPEC.md` (file paths, loader flag, the cache
directory the files must land in, the 12 receipt requirements and the test that
enforces each).

## Single next action

Send `AISI_ACCESS_REQUEST.md` — fill the three `[[ ]]` identity fields, open a
discussion at `huggingface.co/datasets/ai-safety-institute/AgentHarm/discussions`,
and record the thread URL under blocker `E10-HELDOUT-SPLIT` in the receipt. The
request also offers the alternative that AISI run the eval themselves, which may
be the faster path given the no-logging condition.

## Artifacts

`receipt_2026-08-09.json` · `AISI_ACCESS_REQUEST.md` · `PRIVATE_SPLIT_RUN_SPEC.md` ·
`harness_validation_run.py` · `build_receipt.py` · `evidence/` (inspect help,
registry entry, validation result, log probe, 396 task-ID hashes, unit-test log) ·
`logs/harness_validation/` (2 `.eval` logs)
