# E4 BankerToolBench: small-scale base-model arm (2026-09-26)

**Result:** mean native reward **0.0681** (0.4087 / 6 tasks). Base Qwen3.6-35B-A3B, no adapter, run on Tinker.

**What ran:**
- The official BTB Harbor tasks (`outputs/e4_banker_toolbench/official_repo_ff6db552`), using the opencode agent in local Docker (colima).
- The model was served by `code/tinker_shim.py`, a local OpenAI-compatible shim. It sets `enable_thinking=False` and `temperature=0`, and stops on `<|im_end|>`.
- Grading used the native gandalf rubric verifier with the judge `gemini/gemini-3-flash-preview`.
- Tasks: the first 6 of `random.Random(20260926).sample(sorted(btb-*), 10)`, with a 500k-token Tinker cap per task.
- Per-task rewards are in `result.json`. Only btb-19b3361c scored above 0 (0.4087). The rest are cap hits, an external cancellation, or no deliverables.

**Prior base run (2026-09-21):** 100 trials with reward 0.0, listed as `prior_base_run`. That run went through the Modal bridge, which set no stop tokens and had a lossy Responses converter. It is not comparable to this run.

**Rerun:**
```bash
# 1. Shim. Needs TINKER_API_KEY from repo .env and SHIM_KEY, any random bearer.
echo '{"cap_total_tokens": 500000}' > raw/cap.json
code/shim_loop.sh E4 18741            # uses a venv with tinker==0.30.4, transformers, fastapi, uvicorn
# 2. Tasks, one at a time. Each call resets the cap to used+500k.
#    Needs DOCKER_HOST=colima and GEMINI_API_KEY from the keychain.
code/e4_loop.sh btb-19b3361c btb-11e08646 btb-07727295 btb-bc55d3e8 btb-507a3d72 btb-990cf5d7
# 3. Recompute
python code/compute_result.py btb-19b3361c btb-11e08646 btb-07727295 btb-bc55d3e8 btb-507a3d72 btb-990cf5d7
```

**Raw files:**
- `raw/jobs/*/btb-*/verifier/reward.json` holds the per-task reward.
- `raw/shim_ledger.jsonl` logs every sampling call with token counts and output text.
- `raw/infra_failed_attempts/` holds runs killed by Docker daemon loss or a full disk. They are not scored.

**Paired vLLM arm:** results are in `vllm_trained/` and `vllm_base/`, with the comparison in `paired.json`.
- Setup: same 6 tasks, harness, grader and greedy decoding as above. The shim runs with `--backend vllm`, which sends the identical prompt ids to `/v1/completions`. Context is 32768 in both vLLM arms.
- Result: trained 0.0662 vs base 0.0928. The paired difference is −0.0266, with a bootstrap 95% CI of [−0.080, 0.000].
- Rerun: `code/e4_loop_arm.sh <arm> <port> <tasks>`, then `code/compute_result.py --raw`, then `code/paired.py E4 continuous`.
- Artifact note: agent-created workspace `.venv` directories were removed from raw/. They are listed in `REMOVED_ARTIFACTS.md`.
