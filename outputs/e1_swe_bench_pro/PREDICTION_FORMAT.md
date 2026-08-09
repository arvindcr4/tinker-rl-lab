# SWE-bench Pro prediction format (exact evaluator contract)

Pinned evaluator: `scaleapi/SWE-bench_Pro-os@ca10a60a5fcae51e6948ffe1485d4153d421e6c5`
Pinned dataset: `ScaleAI/SWE-bench_Pro@7ab5114912baf22bb098818e604c02fe7ad2c11f` (test, 731 rows)

## 1. Patch JSON (`--patch_path`)

A JSON **array** of objects. Each object:

| Field | Required | Notes |
|---|---|---|
| `instance_id` | yes | Exact dataset `instance_id` string |
| `patch` | yes* | Unified git diff string applied with `git apply -v /workspace/patch.diff` |
| `model_patch` | yes* | Alias accepted by official evaluator (`patch_sample.get("model_patch", patch_sample.get("patch", ""))`) |
| `prefix` | no | Used only for output filenames (`{prefix}_output.json`, logs). Gold uses `"gold"`. |

\* One of `patch` or `model_patch` must be a non-empty string.

Example (gold, one task):

```json
[
  {
    "instance_id": "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan",
    "patch": "diff --git a/...\n...",
    "prefix": "gold"
  }
]
```

Local flagship validator (`validate_patch_artifact`) additionally requires the set of `instance_id`s to **exactly equal** the selected evaluation task IDs (no extras, no missing).

SWE-agent path: produce per-instance `.pred` files, then:

```bash
python helper_code/gather_patches.py \
  --directory <pred_dir> \
  --prefix <model_run_id> \
  --output <patches>.json
```

## 2. Raw samples (`--raw_sample_path`)

CSV **or** JSONL. Required columns used by the grader:

- `instance_id` (index)
- `repo` (feeds Docker Hub tag construction)
- `base_commit`
- `before_repo_set_cmd`
- `selected_test_files_to_run` — string form of a Python list, e.g. `"['path/to/test.js']"`
- `fail_to_pass` — string form of a Python list of test names
- `pass_to_pass` — string form of a Python list of test names

Resolved iff `(fail_to_pass ∪ pass_to_pass) ⊆ {tests with status PASSED}` in container `output.json`.

Instance also needs local evaluator assets:

- `run_scripts/{instance_id}/run_script.sh`
- `run_scripts/{instance_id}/parser.py`
- `dockerfiles/base_dockerfile/{instance_id}/Dockerfile`
- `dockerfiles/instance_dockerfile/{instance_id}/Dockerfile`

## 3. Docker image

```
jefzda/sweap-images:{dockerhub_tag}
```

For the selected NodeBB instance:

```
jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5
@ sha256:e49637ebe82a479ca43b4663525955bc9cdd58c457140ee31c20958d621d3cf7
```

`linux/amd64` only. On arm64 hosts pass `--docker_platform=linux/amd64` (official script auto-detects this).

Hub reported compressed size ≈ 846 MB for this tag.

## 4. Official eval command (local Docker)

```bash
python swe_bench_pro_eval.py \
  --raw_sample_path=<samples>.jsonl \
  --patch_path=<patches>.json \
  --output_dir=<out>/ \
  --scripts_dir=run_scripts \
  --num_workers=1 \
  --dockerhub_username=jefzda \
  --use_local_docker \
  --docker_platform=linux/amd64
```

Success artifact: `<out>/eval_results.json` mapping `instance_id → bool`.

## 5. Model-score artifact fields (flagship external receipt)

Beyond the evaluator JSON, the local preflight gate for a **model** run also wants provenance:

- `instance_id`, `patch`
- `model_revision` (immutable 40-char HF commit of the evaluated model)
- `generation_run_id` (W&B/run id or equivalent)

A gold/reference patch proves plumbing only — never promote it to `score`.
