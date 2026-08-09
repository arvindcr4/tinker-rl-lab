# E5 — APEX Agents (`apex_agents_eval`) — lane status 2026-08-09

**Status: PARTIAL.** Suite score `null`. No model was evaluated, no Tinker call,
no paid API call. The native verifier now installs and runs; the dataset is one
self-service login away.

## 1. Gating status — verified, and it is not a denial

The prior sprint note recorded "exact HF task files return 403 gated access".
That reads as a refusal. It is not one.

| Fact | Value | Source |
|---|---|---|
| `gated` | `"auto"` | `GET https://huggingface.co/api/datasets/mercor/apex-agents` → 200 |
| `private` | `false` | same |
| `disabled` | `false` | same |
| `sha` | `92c86856cf1b11f9833a8a076b3a45a63afa3929` | same — matches our pin |
| `cardData.license` | `cc-by-4.0` | same |
| file count | 319 | same |
| `usedStorage` | 20,093,539,877 B (**20.09 GB**) | same |
| file probe | **401**, `x-error-code: GatedRepo` | `HEAD .../resolve/92c8685.../tasks_and_rubrics.json` |
| `auth-check` | **401**, "You must … be authenticated to access it. Please log in." | `GET /api/datasets/mercor/apex-agents/auth-check` |

Two corrections to the prior note: the status is **401 Unauthorized**, not 403,
and the error code is `GatedRepo` — an *authentication* gate, not a rejection.

Per the Hugging Face Hub docs on gated datasets, `gated: "auto"` is **automatic
approval**: "By default, access to the dataset is automatically granted to the
user when requesting it… any user can access your dataset once they've shared
their personal information with you." There is **no human review step** and no
owner queue. The only requirement is a logged-in account that has accepted the
terms once.

Raw evidence: `evidence/hf_dataset_meta.json`.

### Exact unblock sequence

```bash
# 1. Authenticate (writes ~/.cache/huggingface/token). Needs a HF account.
hf auth login

# 2. Accept the terms ONCE, in the browser — this is the step the token alone
#    does not cover. Open the dataset page and click "Agree and access repository".
open https://huggingface.co/datasets/mercor/apex-agents

# 3. Verify: this must return 200 instead of 401.
hf download mercor/apex-agents --repo-type dataset \
  --revision 92c86856cf1b11f9833a8a076b3a45a63afa3929 \
  tasks_and_rubrics.json world_descriptions.json eval.yaml

# 4. Re-run preflight; benchmark_access / dataset_schema / task_split flip to PASS.
PYTHONPATH=zvf-program python3 zvf-program/flagship/eval_apex_agents.py \
  --archipelago-dir outputs/e5_apex_agents/sprint.8cEFaN/archipelago \
  --cache-dir outputs/e5_apex_agents/cache \
  --out outputs/e5_apex_agents/preflight_receipt_2026-08-09.json
```

Step 2 is not optional and cannot be automated from here. No bypass was
attempted.

**Disk note:** a full `snapshot_download` is 20.09 GB. A single-task run needs
only `tasks_and_rubrics.json`, `world_descriptions.json`, one of the 33
`world_files_zipped/<world_id>.zip`, and — for 142 of 480 tasks —
`task_files/<task_id>/**`. Fetch by `allow_patterns`, not the whole repo.

## 2. What now runs — native verifier, `harness_validation`

`harness_validation` · `is_model_score: false` · **suite score stays `null`**.

The pinned Archipelago grading verifier
(`Mercor-Intelligence/archipelago @ 1c3dcd4694b313020cd626699c9c7cc1c0a2fc58`)
is installed in an isolated Python 3.13 venv and executes end to end:
snapshot diff → helper execution → programmatic verifier → scoring method →
`grades.json`.

```bash
# install (once) — 909 MB, isolated, does not touch the pinned checkout
cd outputs/e5_apex_agents/sprint.8cEFaN/archipelago/grading
UV_PROJECT_ENVIRONMENT=$PWD/../../../venv-grading uv sync --locked --python 3.13

# run the validation (re-runnable, self-asserting)
python3 outputs/e5_apex_agents/harness_validation/run_harness_validation.py
```

Observed:

```
grading_run_status = completed
ver_pass_at_least : 1.0   (Word Count >= 5      -> pass)
ver_fail_at_least : 0.0   (Word Count >= 10000  -> fail)
ver_pass_exact    : 1.0   (Word Count == 12     -> pass)
final_score       = 0.6666666666666666
```

Registry loads 69 evals (42 programmatic-only) and 9 scoring methods.

Design points that make this evidence rather than decoration:

- The fixture is **synthetic and labelled as such** in every artifact. Snapshots
  derive from the repo's own `examples/simple_task/original_snapshot.zip` plus
  one file the fixture builder writes. Nothing from `mercor/apex-agents` is
  involved.
- One verifier is **built to fail**. A validation that can only pass proves
  nothing; this one shows the harness discriminates.
- The `Exactly 12 words` verifier passes only if the snapshot diff surfaced the
  created file's **real content**, so the diff path is genuinely exercised — not
  stubbed.
- Only `content_length_check` is used, registered
  `eval_types=[EvalType.PROGRAMMATIC]`. **No LLM judge, zero paid API calls.**
  Provider keys are stripped from the subprocess env and `llm_judge_model` is
  set to a sentinel so an accidental judge call would fail loudly.

Artifacts: `harness_validation/out/harness_validation_receipt.json`,
`harness_validation/out/grades.json`, `harness_validation/out/grading_run.log`.

## 3. Ingestion path — the dataset drops straight in

The field contract is taken from Mercor's **own** loader
(`archipelago/examples/hugging_face_task/main.py` at the pinned revision), not
guessed: `task_id`, `world_id`, `task_name`, `domain`, `prompt`,
`rubric[].verifier_id`, `rubric[].criteria`, `world_id`/`world_name`, plus
`world_files_zipped/<world_id>.zip` and `task_files/<task_id>/**`.

Added to `zvf-program/flagship/eval_apex_agents.py`:

- `validate_task_records` / `validate_world_records` — per-field schema checks,
  `task_<32 lowercase hex>` id pattern, duplicate id/verifier detection.
- `validate_dataset_references` — every `task.world_id` must resolve in
  `world_descriptions.json`.
- `dataset_ingestion_report` — counts, order-independent `task_id_sha256` /
  `world_id_sha256`, errors, and advisory warnings when counts drift from the
  documented 480 tasks / 33 worlds.
- `required_task_assets` — the exact repo-relative paths one task needs, so the
  download is scoped instead of pulling 20 GB.
- New **`dataset_schema` gate**, wired into `run_preflight`, into the receipt,
  and into `_all_launch_gates_pass`. A schema mismatch forces `task_split` to
  BLOCKED too, so selection cannot succeed on the one well-formed record.

Split manifest and task-ID hashing already existed (`_task_split_gate`) and are
now downstream of schema validation.

```bash
PYTHONPATH=zvf-program python3 -m unittest \
  flagship.test_eval_apex_agents_ingestion \
  flagship.test_pavlov_apex_agents_eval_adapter
# Ran 39 tests — OK  (29 new ingestion tests + 10 pre-existing adapter tests)
```

`zvf-program/flagship/test_eval_apex_agents_ingestion.py` is new; every fixture
in it is marked `SYNTHETIC FIXTURE - NOT mercor/apex-agents CONTENT`.

## 4. Preflight gate board

`outputs/e5_apex_agents/preflight_receipt_2026-08-09.json` — `status: BLOCKED`,
`score: null`, `tinker_calls: 0`.

| Gate | Status | Change vs. prior receipt |
|---|---|---|
| `benchmark_metadata` | PASS | now records `gated`, `gating_mode`, `gating_unblock`, `file_count` |
| `benchmark_access` | BLOCKED | 401 `GatedRepo`; now carries the 3-step unblock commands |
| `dataset_schema` | BLOCKED | **new gate** |
| `task_split` | BLOCKED | unchanged (downstream of the dataset) |
| `native_verifier` | **PASS** | was BLOCKED — checkout present, tree sha `5b5f627d…` |
| `isolated_runtime` | **PASS** | was BLOCKED — docker/uv/git/py3.13 all resolve |
| `model_identity` | PASS | — |
| `budget` | PASS | projected max $0.25536 vs $0.50 ceiling |
| `wandb_online_before_tinker` | BLOCKED | credential now correctly found (`netrc:api.wandb.ai`); only the missing package remains |
| `tinker_access` | BLOCKED | `tinker` not importable |
| `native_grader_credentials` | BLOCKED | no provider key in this session's env |

## 5. What a real run still needs beyond the dataset

Nothing below was used, started, or billed.

| Requirement | State | Note |
|---|---|---|
| **W&B credential** | present | `~/.netrc` machine `api.wandb.ai`. The gate previously demanded `WANDB_API_KEY` and mis-reported this as missing; it now detects the netrc fallback that `wandb.login()` itself uses. Value never read. |
| **`wandb` package** | **missing** | not installed. Trap: a leftover `wandb/` directory at the repo root makes `import wandb` succeed as an *empty namespace package* (`__file__ is None`) for anything with cwd on `sys.path`. Flagged separately. |
| **`tinker` package** | **missing** | `ModuleNotFoundError`. `TINKER_API_KEY` is present in `.env` and was not used. |
| **Agent model key** | **missing** | Archipelago's agent runner needs one of `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY`. |
| **Judge model key** | **missing** | Separate cost centre. The official APEX path maps *every* rubric criterion to `eval_config_id: "ec_output_llm"` → one LLM judge call per criterion per task. Rubrics are multi-criterion, so judge spend scales with criteria, not tasks. |
| **Docker environment** | available, not built | Colima 28.4.0, 8 CPU / 16.5 GiB. A run needs `docker compose up --build` in `archipelago/environment` and all 9 MCP servers. Image not built — that is a >2 GB build and would need the shared mutex. |
| **Disk** | 37 GiB free at time of writing (was 63) | scope the HF download with `allow_patterns`; do not `snapshot_download` the full 20.09 GB. |

Upstream reproducibility gap worth knowing: Mercor's own
`examples/hugging_face_task/main.py` calls `hf_hub_download` **without a
`revision`**, so it silently tracks `main`. Our runner pins
`92c86856cf1b11f9833a8a076b3a45a63afa3929`. Do not adopt the upstream example
verbatim.

Upstream defect: `grading/mise.toml`'s `start` task runs `validate_config.py`
and `test_local.py`, neither of which exists at the pinned revision, and
`[tool.pytest.ini_options] testpaths = ["tests"]` points at a directory the
grading package does not ship. `mise run start` and `pytest` both fail there.
The documented `runner.main` CLI path — the one used above — works.

## 6. Single next action

Run `hf auth login`, then accept the terms at
`https://huggingface.co/datasets/mercor/apex-agents`. Access is auto-granted;
re-running preflight then flips `benchmark_access`, `dataset_schema`, and
`task_split` to PASS in one pass. Everything downstream of the dataset in this
lane is already built and tested.

## Files

| Path | What |
|---|---|
| `lane_status_2026-08-09.md` | this file |
| `lane_receipt_2026-08-09.json` | lane receipt (`PARTIAL`, score `null`) |
| `preflight_receipt_2026-08-09.json` | full gate board from the runner |
| `evidence/hf_dataset_meta.json` | raw HF API response proving `gated: auto` |
| `harness_validation/run_harness_validation.py` | re-runnable, self-asserting validation |
| `harness_validation/build_fixture.py` | synthetic fixture builder |
| `harness_validation/fixtures/` | synthetic snapshots + configs (`fixture_manifest.json`) |
| `harness_validation/out/` | `grades.json`, `grading_run.log`, validation receipt |
| `venv-grading/` | isolated Python 3.13 env, 909 MB |
| `sprint.8cEFaN/archipelago/` | pinned verifier checkout, left unmodified |
| `zvf-program/flagship/eval_apex_agents.py` | schema validation + `dataset_schema` gate + gating facts + netrc-aware W&B gate |
| `zvf-program/flagship/test_eval_apex_agents_ingestion.py` | 29 new tests (synthetic fixtures) |
