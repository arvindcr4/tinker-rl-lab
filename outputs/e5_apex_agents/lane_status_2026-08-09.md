# E5 — APEX Agents (`apex_agents_eval`) — lane status 2026-08-09

**Status: PARTIAL.** Suite score `null`. Dataset unblocked and fully validated;
8 of 11 preflight gates now PASS. No model was evaluated. Zero Tinker calls,
zero paid API calls, zero W&B runs.

## 1. Gates that flipped

The owner ran `hf auth login` and accepted the terms. The `gated: auto` reading
held — no human review. Anonymous probes returned **401 `GatedRepo`**;
authenticated probes at the same pinned revision return **200** with
`X-Repo-Commit: 92c86856cf1b11f9833a8a076b3a45a63afa3929` and content lengths
matching the bytes on disk exactly (1,080,670 / 45,846 / 522).

| Gate | Before | Now |
|---|---|---|
| `benchmark_metadata` | PASS | PASS |
| `benchmark_access` | BLOCKED | **PASS** |
| `dataset_schema` | BLOCKED | **PASS** |
| `task_split` | BLOCKED | **PASS** |
| `native_verifier` | PASS | PASS |
| `isolated_runtime` | PASS | PASS |
| `model_identity` | PASS | PASS |
| `budget` | PASS | PASS |
| `wandb_online_before_tinker` | BLOCKED | BLOCKED — `wandb` not installed (credential itself is present in `~/.netrc`) |
| `tinker_access` | BLOCKED | BLOCKED — `tinker` not importable |
| `native_grader_credentials` | BLOCKED | BLOCKED — no judge provider key |

The three remaining blockers are all deliberately out of bounds for this lane.

The preflight now accepts an already-downloaded copy via `--dataset-dir`, but
only on proof of revision: it reads the commit hash Hugging Face writes into
`.cache/huggingface/download/<file>.metadata` and rejects the directory unless
it matches the pin. A hand-assembled folder, or a copy pulled from `main`,
cannot masquerade as the pinned revision. Four regression tests cover that.

```bash
PYTHONPATH=zvf-program python3 zvf-program/flagship/eval_apex_agents.py \
  --archipelago-dir outputs/e5_apex_agents/sprint.8cEFaN/archipelago \
  --dataset-dir outputs/e5_apex_agents/hf_dataset \
  --task-index 0 \
  --out outputs/e5_apex_agents/preflight_receipt_2026-08-09.json
```

## 2. Schema validation against the real file

The validator was written against a synthetic fixture before the data existed.
Run unchanged against the real `tasks_and_rubrics.json`, first attempt:

```
valid            : True
errors           : 0
count_warnings   : []
task_count       : 480      world_count : 33
task_id_sha256   : 0467d071bd800a9875e787a958c2d125f3a0748d7c058703035d57c6cb78f8c4
world_id_sha256  : 4dda01d4cf5116f939e27714ea58b6657f0d08f803b323cd5425b2dc5a313558
order-independent hashing: confirmed on real data (shuffled input, identical hash)
```

Referential integrity — checked against the 319-entry Hugging Face **file
listing**, so no bulk asset was downloaded. All four are **exact set matches**,
not merely equal counts:

| Check | Result |
|---|---|
| 33 `world_id` refs → `world_descriptions.json` | no dangling refs |
| 33 `world_id` → `world_files_zipped/*.zip` | set equality, no orphans |
| 141 tasks with `task_input_files` → `task_files/<task_id>/` | **set equality** |
| 58 tasks with `gold_response_type == "file"` → `gold_files/<task_id>/` | **set equality** |

Schema correction worth recording: **`task_input_files` is `str \| None`**, a
snapshot id like `snap_<32 hex>` — not a list of filenames. Only its truthiness
is meaningful, and the per-task asset directory is keyed by `task_id`. My
contract used truthiness and `task_id`, which the set equality above confirms as
correct. The revision also carries three fields Mercor's own example never
reads: `expected_output`, `gold_response`, `gold_response_type`.

Rubric criteria carry **only** `criteria` and `verifier_id` — no weights, no
`is_primary_objective`, no negative criteria. The upstream example therefore
picks the primary objective arbitrarily (`is_primary_objective = (i == 0)`).

## 3. The 480 reconciliation — everything agrees

I could not find a discrepancy. Four independent sources concur:

| Claim | Dataset card | `metadata.json` | Measured from the file |
|---|---|---|---|
| Tasks | 480 (160 per job category) | 480 | **480** (160 / 160 / 160) |
| Worlds | 33 (10 banking, 11 consulting, 12 law) | 33 | **33** |
| Mean criteria/task | 4.06 | — | **4.058** (1,948 total) |
| Criteria range | "between 1 and 10" | — | **min 1, max 10** |
| IB / Law / MC mean criteria | 2.93 / 4.57 / 4.68 | — | **2.925 / 4.569 / 4.681** |
| Tasks with file outputs | 58 (12.1%) | — | **58** |

`metadata.json` also records 30,720 trajectories over 8 models — exactly
480 × 8 models × 8 rollouts, consistent with the card's Pass@8 leaderboard.

Three genuine findings fell out of the reconciliation instead:

1. **Training is contractually forbidden.** The card: "APEX-Agents is intended
   exclusively for model evaluation. Any use of this dataset for training,
   fine-tuning, or parameter fitting is forbidden." The card also carries a
   robots-exclusion statement asking that the dataset not be crawled or scraped.
   Access here was via the sanctioned `hf download` path after accepting terms,
   and only the ~1.1 MB of JSON/YAML/MD was fetched. The sealed manifest pins
   `training_task_ids: []` and `training_prohibited: true`. **This constraint
   must propagate to any lane that might otherwise reach for APEX data.**
2. **The HF config exposes the split as `train`.** The card frontmatter maps
   `tasks_and_rubrics.json` to `split: train`, so a naive
   `load_dataset("mercor/apex-agents")["train"]` reads as training data on a
   dataset where training is prohibited. Footgun, upstream.
3. **`task_files/.DS_Store` is committed to the dataset repo.** Harmless, but it
   inflates a naive directory count from 141 to 142. The manifest builder
   filters dot-entries and records what it ignored.

## 4. Gold responses through the native verifier

`harness_validation` · `is_model_score: false` · **suite score stays `null`**.

Real expert-authored `gold_response` text from real APEX tasks, carrying real
`task_id` / `world_id` / `verifier_id` values, fed through the pinned
Archipelago runner and its `final_answer` helper.

```bash
python3 outputs/e5_apex_agents/harness_validation/run_gold_response_validation.py
```

```
selected 12 real APEX tasks across 3 domains (4 per domain, deterministic)
tasks graded: 12 | fully matching expectation: 12
  positive  12/12   pattern from THIS task's gold response      -> 1.0
  swap      12/12   pattern from a DIFFERENT task's gold        -> 0.0
  sentinel  12/12   token that cannot occur                     -> 0.0
```

The **swap control** is the point. A verifier that returned 1.0 unconditionally
would pass the positive check and look identical; it fails the swap. 36/36
controls landed as predicted, so the verifier demonstrably reads the specific
answer under test.

Only `pattern_match_check` was used — `eval_types=[EvalType.PROGRAMMATIC]`. No
LLM judge, provider keys stripped from the subprocess environment.

Two honest limits, stated rather than buried:

- This is **not** grading against the real APEX rubric. The real rubric is
  LLM-judged (`ec_output_llm`), and invoking a judge is out of bounds. What is
  proven is the ingestion-and-verdict pipeline, not rubric semantics.
- The snapshot zips are the inert synthetic pair from the earlier validation;
  `world_files_zipped/` was deliberately not downloaded. The gold response text
  is real; the snapshots are scaffolding.

The earlier fully-synthetic validation still passes unchanged
(`run_harness_validation.py`: 3/3 verifiers, final 0.667).

## 5. Sealed 480-task split manifest

`outputs/e5_apex_agents/split_manifest_480.json` (350 KB) pins all 480 task IDs,
33 world IDs, every `verifier_id`, per-task rubric size, `expected_output`,
`gold_response_type`, a SHA-256 of each gold response, file digests, the
per-domain breakdown, and the four referential-integrity results — so any later
subset is provably a subset of *this* revision and upstream drift is detectable
without re-downloading.

```bash
python3 outputs/e5_apex_agents/build_split_manifest.py
```

It refuses to seal a manifest over a dataset that fails schema validation.

## 6. Cost of a real judged run — it does not fit

```bash
python3 outputs/e5_apex_agents/project_full_run_cost.py
```

Counts are measured, not assumed. Judge calls per criterion come from reading
the pinned source: 1 grading call always, plus 1 artifact-selection call when
the snapshot diff is non-empty; `NEGATIVE_CRITERIA_ENABLED` is `False` and the
APEX rubric has no negative criteria, so that path contributes nothing.

**Tinker (agent side), at the existing E5 ceiling** — 50 steps × 8,192 prompt +
50 × 512 sampled, at $0.54/M prefill and $1.335/M sample:

| Scope | Tinker ceiling |
|---|---|
| 1 task | $0.26 |
| 480 tasks, pass@1 | **$122.57 — 8.2× the ~$15.00 remaining** |
| 480 tasks, pass@8 (leaderboard parity) | $980.58 |
| What $15.00 buys | **58 tasks (12% of the suite)**, spending the entire remaining budget |

At `max_steps=100` — which Mercor's own example suggests for complex tasks — the
per-task ceiling doubles and $15 buys 29 tasks.

**Judge (separate provider wallet, not Tinker):** 1,948 grading calls fixed, plus
322 (file-output tasks only) to 1,948 (every task leaves artifacts)
artifact-selection calls → **2,270–3,896 judge calls** for one pass@1 sweep. At
a 2,000-token agent answer, using placeholder rate tiers that must be replaced
with the real rate card:

| Rate tier (placeholder) | pass@1 judge cost |
|---|---|
| cheap / flash-class | $0.92 – $1.21 |
| mid | $9.79 – $12.94 |
| frontier | $29.38 – $38.83 |

**Wall clock may bind before money does.** Each task boots a Docker environment,
populates a world, runs a 9-server MCP agent loop, snapshots, and grades —
roughly 5–15 minutes serially, so 40–120 hours for the full suite.

**Recommendation.** Do not attempt the full suite. If an APEX number is wanted,
pre-register a stratified subset, seal its task IDs in the split manifest
*before* the run, and report it as a subset estimate with confidence intervals —
never as a suite score comparable to the published leaderboard, which is pass@8
over all 480. On the current budget, an APEX number and the E11 eval plus a
possible E13 training run cannot all be bought.

## 7. What still blocks a real run

| Requirement | State |
|---|---|
| Dataset | **resolved** — scoped JSON/YAML/MD at the pinned revision, validated |
| W&B credential | present (`~/.netrc`, `api.wandb.ai`); the gate is now netrc-aware |
| `wandb` package | **missing.** Trap: a leftover `wandb/` directory at the repo root makes `import wandb` succeed as an empty namespace package for anything with cwd on `sys.path`. Flagged separately. |
| `tinker` package | **missing** (`TINKER_API_KEY` present in `.env`, unused) |
| Agent model key | **missing** — one of `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` |
| Judge model key | **missing** — separate cost centre, see §6 |
| Environment container | not built; >2 GB build needing the shared `outputs/_setup/docker.lock` mutex |
| World assets | not downloaded by design — 20.09 GB; fetch per-task via `allow_patterns` |

Upstream reproducibility gap, unchanged: Mercor's
`examples/hugging_face_task/main.py` calls `hf_hub_download` **without a
revision**, silently tracking `main`. Our runner pins the commit. Also unchanged:
`grading/mise.toml`'s `start` task references two files absent at the pinned
revision, and `testpaths = ["tests"]` points at a directory the grading package
does not ship — the documented `runner.main` CLI path is the one that works.

## 8. Single next action

Decide the scope question before spending anything: a stratified pre-registered
subset of roughly 40–58 tasks is the largest APEX evaluation the remaining
budget supports, and it consumes the budget. Everything upstream of the model
call is now built, validated, and sealed.

## Tests

```bash
PYTHONPATH=zvf-program python3 -m unittest \
  flagship.test_eval_apex_agents_ingestion \
  flagship.test_pavlov_apex_agents_eval_adapter
# Ran 43 tests — OK  (33 ingestion incl. 4 new local-dataset-gate, 10 adapter)
```

## Files

| Path | What |
|---|---|
| `lane_status_2026-08-09.md` | this file |
| `lane_receipt_2026-08-09.json` | lane receipt (`PARTIAL`, score `null`) |
| `preflight_receipt_2026-08-09.json` | gate board — 8/11 PASS |
| `split_manifest_480.json` | sealed 480-task manifest |
| `full_run_cost_projection.json` | measured cost model |
| `build_split_manifest.py`, `project_full_run_cost.py` | the two generators |
| `hf_dataset/` | scoped dataset at the pinned revision (~1.1 MB) |
| `evidence/hf_dataset_meta.json` | HF API response proving `gated: auto` |
| `evidence/authenticated_access_probe.json` | 401 → 200 access receipt |
| `harness_validation/run_gold_response_validation.py` | gold-response validation + controls |
| `harness_validation/run_harness_validation.py` | original synthetic validation |
| `harness_validation/gold_out/` | per-task grades, logs, receipt |
| `zvf-program/flagship/eval_apex_agents.py` | schema gate, `--dataset-dir`, gating facts, netrc-aware W&B gate |
| `zvf-program/flagship/test_eval_apex_agents_ingestion.py` | 33 tests |
