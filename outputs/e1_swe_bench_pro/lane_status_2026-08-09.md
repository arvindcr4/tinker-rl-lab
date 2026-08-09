# E1 lane status — 2026-08-09

**Status: PARTIAL** · suite `score: null` · `is_model_score: false`

## What now runs

- **Docker gate passes.** Colima reports 8 CPU / 16.544 GiB → `meets_recommended_resources: true` (was 2 CPU / 1.9 GiB).
- **Isolated venv** at `outputs/e1_swe_bench_pro/runtime/venv` with `datasets`, `pandas`, `tqdm`, `docker`, `huggingface_hub`.
- **Adapter unit tests:** 14/14 pass (`PYTHONPATH=zvf-program pytest zvf-program/flagship/test_pavlov_swe_bench_pro_eval_adapter.py -q`).
- **Per-instance image pulled locally:**
  `jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5`
  @ `sha256:e49637ebe82a479ca43b4663525955bc9cdd58c457140ee31c20958d621d3cf7` (amd64, ~846 MB, pull 316 s).
- **Harness validation (gold, not a model score):**
  - Official scripts (`timeout 120` around mocha): **resolved=false** in 521 s under QEMU — empty tests JSON.
  - QEMU-tolerant scripts (`timeout 1200`): **resolved=true** in 131 s — 300 PASSED, f2p 3/3, p2p 288/288.
- **Prediction format** documented: `outputs/e1_swe_bench_pro/PREDICTION_FORMAT.md`.

## Commands that work

```bash
# preflight (Docker gate)
python3 zvf-program/flagship/swe_bench_pro_eval.py preflight \
  --runtime-dir outputs/e1_swe_bench_pro/runtime \
  --evaluator-dir outputs/e1_swe_bench_pro/evaluator \
  --out outputs/e1_swe_bench_pro/preflight_2026-08-09.json \
  --task-id instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan

# gold harness (must run with CWD = evaluator; image must be local)
cd outputs/e1_swe_bench_pro/evaluator && \
  ../../e1_swe_bench_pro/runtime/venv/bin/python swe_bench_pro_eval.py \
  --raw_sample_path=../selected_sample_nodebb.jsonl \
  --patch_path=../gold_patch_nodebb.json \
  --output_dir=../harness_validation_qemu \
  --scripts_dir=../run_scripts_qemu \
  --num_workers=1 --dockerhub_username=jefzda \
  --use_local_docker --docker_platform=linux/amd64 --redo
```

Take `mkdir outputs/_setup/docker.lock` before pull/run; `rmdir` when done. Stop if `df` free space &lt; 15 GiB.

## License (factual)

| Source | Finding |
|---|---|
| HF `cardData.license` @ `7ab5114…` | `null` |
| README YAML at pinned rev | no `license:` field |
| `LICENSE` / `LICENSE.md` / `LICENSE.txt` / `COPYING` at pinned rev | all HTTP 404 |
| Evaluator `scaleapi/SWE-bench_Pro-os@ca10a60` | **MIT** (`LICENSE` present) |

Dataset data license remains **undeclared** — policy gate, not a download.

## What remains

1. **Model patch artifact** for the selected instance (`instance_id` + `patch` + `model_revision` + `generation_run_id`).
2. **Immutable dataset-license receipt** with `approved=true` once Scale publishes one (or an explicit policy exception).
3. **Native amd64** (or approved timeout override) so the *official* 120 s mocha timeout is not QEMU-starved.

## Single next action

Generate a model patch JSON for
`instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan`
in the format of `PREDICTION_FORMAT.md`, then re-run the official evaluator (prefer amd64 host). Do not promote gold/`harness_validation` to a suite score.

Receipt: `outputs/e1_swe_bench_pro/lane_receipt_2026-08-09.json`
