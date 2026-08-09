# E10 — what the held-out AgentHarm run needs

Written 2026-08-09. Everything here is read off the installed harness
(`inspect_evals` @ `b935c0e5cfa04710f016f925db75d8e81413e2cf`), not from memory.

## 1. The three files

`inspect_evals/agentharm/utils.py::load_dataset` maps split names to paths. The
held-out column is:

| Loader dataset | Path relative to the dataset root | Present locally |
|---|---|---|
| `harmful` | `benchmark/harmful_behaviors_test_private.json` | no |
| `benign` | `benchmark/benign_behaviors_test_private.json` | no |
| `chat` | `benchmark/chat_private_test.json` | no |

Note the chat file breaks the naming pattern — it is `chat_private_test.json`,
not `chat_test_private.json`. Getting this wrong produces a `FileNotFoundError`
that looks like a missing grant.

All three are absent from the public dataset revision
`e23b3fe60a0da9037314b88e5ee3a0c054970dad`. The paper puts the held-out set at
30% of behaviours (132 augmented), never exposed to public LLM APIs.

## 2. The loader flag

`split="test_private"`, on both tasks:

```python
agentharm(split="test_private", ...)          # harmful (default) or chat via chat_dataset=True
agentharm_benign(split="test_private", ...)   # benign counterpart
```

From the CLI:

```bash
outputs/_setup/venvs/e10/bin/inspect eval inspect_evals/agentharm \
  --model <agent-model> -T split=test_private
outputs/_setup/venvs/e10/bin/inspect eval inspect_evals/agentharm \
  --model <agent-model> -T split=test_private -T chat_dataset=True
outputs/_setup/venvs/e10/bin/inspect eval inspect_evals/agentharm_benign \
  --model <agent-model> -T split=test_private
```

## 3. Where the files have to land

`load_dataset` resolves paths against
`INSPECT_EVALS_CACHE_PATH / "agentharm_dataset" / "AgentHarm"` —
on this host, `~/Library/Caches/inspect_evals/agentharm_dataset/AgentHarm`.

It only calls `snapshot_download` when the `benchmark/` directory does not
exist. That directory already exists with the six public files, so the loader
will **not** attempt a download and will **not** notice a missing private file
until it raises `FileNotFoundError`. The granted files therefore have to be
copied into that `benchmark/` directory by hand (keep the lane copy under
`outputs/e10_agentharm/hf_dataset/benchmark/` in sync so the manifest builder
sees the same bytes).

Verified failure mode, from `private_split_preflight.txt`:

```
private_split_load=BLOCKED:FileNotFoundError:File not found:
  .../agentharm_dataset/AgentHarm/benchmark/harmful_behaviors_test_private.json
```

## 4. The grader

Both judges default to `openai/gpt-4o-2024-08-06`:

- `refusal_judge` — decides whether a response is a refusal (`scorer.py`,
  `RefusalJudgeLLM`, prompt `DEFAULT_REFUSAL_PROMPT`).
- `semantic_judge` — called from individual grading functions via
  `benchmark/grading_utils.py::semantic_judge`, must answer `pass` or `fail`.

Two consequences for a held-out run. First, it is a paid call per graded message,
so the run needs budget authorization. Second, the paper states AISI scores the
private set on a non-logging Azure deployment; a run through the public OpenAI
API may not satisfy the contamination condition the held-out split exists to
protect. Both points are in the access request.

Held-out behaviours may also reference `grading_function` names that are absent
from the public `harmful_grading_functions.py`. If so the scorer raises on
`getattr`, and the matching grading-function revision has to come with the grant.

## 5. Evidence the receipt must carry

The fail-closed emitter
(`zvf-program/flagship/pavlov_agentharm_frontiermath_adapter.py::emit_agentharm_score`)
holds `score` at `null` until every one of these is satisfied. Each line is a
test in `test_pavlov_agentharm_frontiermath_adapter.py::AgentharmSplitBindingTests`.

| # | Requirement | Enforced by |
|---|---|---|
| 1 | `is_model_score` is true — a harness validation can never be promoted | `test_emitter_never_promotes_harness_validation` |
| 2 | Split is `test_private` — a public-split number is never the score | `test_emitter_never_promotes_a_public_split_run` |
| 3 | All three held-out files present | `test_emitter_blocks_score_when_private_files_absent` |
| 4 | Split manifest complete: per-file SHA-256 + behaviour count | `test_private_split_manifest_reports_all_three_missing_files` |
| 5 | Dataset revision immutable (40-hex or sha256, never `main`) | `test_task_id_hash_rejects_mutable_revision` |
| 6 | Per-task ID hashes, unique, aggregating to `split_task_id_hash` | `test_emitter_blocks_on_tampered_split_task_id_hash` |
| 7 | `split_manifest_hash` reproducible across machines (path-independent) | `test_split_manifest_hash_is_path_independent` |
| 8 | Verifier identity complete — all 11 grading sources hashed | `test_emitter_blocks_when_verifier_sources_are_missing` |
| 9 | `verifier_hash` changes if any grading source changes | `test_verifier_hash_changes_when_a_grading_source_changes` |
| 10 | Harness revision immutable | `agentharm_verifier_identity` guard |
| 11 | Approved policy-grader artifact receipt with an identity | `test_emitter_blocks_without_approved_policy_grader_artifact` |
| 12 | Model identity pinned to an immutable revision | `_score_blockers` |

Plus, from AISI rather than from local state: the grant identifier and date, the
revision the held-out files were served at, their SHA-256s and behaviour counts,
the approved judge model ID, and any embargo conditions. Those are enumerated in
`AISI_ACCESS_REQUEST.md` section 4 — without them requirement 11 cannot be met
and the emitter stays blocked.

## 6. Order of operations once access lands

1. Copy the three files into the cache `benchmark/` directory and the lane copy.
2. `outputs/_setup/venvs/e10/bin/python outputs/e10_agentharm/build_receipt.py` —
   the `test_private` manifest should now report `complete: true` and a non-null
   `split_task_id_hash`; check the file SHA-256s against what AISI published.
3. Record the grant ID, the AISI-published checksums, and the approved judge in
   the run payload as `policy_grader_artifact`.
4. Only then run the eval with a real agent model and a real judge key, under an
   authorized budget.
5. Feed the run into `emit_agentharm_score`. If `blockers` is empty it emits the
   score; if not, fix the blocker rather than the emitter.
