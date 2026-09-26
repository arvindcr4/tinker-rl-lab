# E5: small-scale base-model arm, run on the τ²-bench substitute (2026-09-26)

**Result:** τ²-bench pass^1 = **0.30 (3/10)**. Airline scored 3/5 and retail 0/5; the retail total includes 1 infrastructure error, which counts as a fail. The actor was base Qwen3.6-35B-A3B on Tinker, with no adapter.

**What ran**
- Benchmark: `sierra-research/tau2-bench` at commit a2c02472, on the test split. Task IDs are in `raw/selection.json`: 10 per domain, seed 20260926. Only the first 5 per domain were run, because the Tinker-token cap ran out.
- Model access: the actor and the user simulator both call the Tinker base model through `../E4/code/tinker_shim.py` (thinking off, temperature 0, `max_tokens` 4096).
- Scoring: the native evaluator. The NL-assertion judge was switched to `gemini/gemini-2.5-flash` because the OpenAI account has no credits. That is a scratch-only edit to `config.py`.

**Why a substitute:** the lead chose to stop the APEX-Agents native run. Its artifacts are kept, unscored, in `raw/apex_attempt_unscored/`. That run produced 1 natively graded task at 0.667; the other tasks hit the token cap, an argparse bug in the auth token, or a world-upload timeout. It is not a result. The two gaps from the original benchmark are recorded in `result.json` as `substitute_gap`: the task domain is different, and the user simulator is not the official one.

**How to rerun**
```bash
E4/code/shim_run.sh e5tinker 18743 E5/raw/shim_ledger_tinker.jsonl E5/raw/cap_tinker.json   # Tinker base shim
E5/code/run_tau2.sh base 18743 airline 8 6 2 22 18
E5/code/run_tau2.sh base 18743 retail 18 12 9 45 90
python E5/code/compute_tau2.py E5/raw e5_base E5/raw/shim_ledger_tinker.jsonl
```
Setup notes:
- `run_tau2.sh` expects a checkout at `$SCRATCH/tau2` (commit a2c02472, `uv sync --frozen --no-dev`).
- It also needs two edits to `config.py`: `DEFAULT_LLM_NL_ASSERTIONS = "gemini/gemini-2.5-flash"`, and `response_format` set to `json_object` for that judge.

**Paired vLLM arm:** see `vllm_trained/`, `vllm_base/` and `paired.json`. The user simulator in both vLLM arms is the vLLM **base** endpoint, not Tinker, because the Tinker cap was exhausted. The simulator is the same in both arms, so the trained-vs-base difference stays paired.
