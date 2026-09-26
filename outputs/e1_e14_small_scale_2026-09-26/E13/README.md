# E13 — BALROG small-scale, base Qwen3.6-35B-A3B (substitute for the OpenReward games)

**Result:** the BALROG native progression is **23.10%** (native stderr 5.36) over 26 episodes in 5 environments. Per environment: BabyAI 70.0 (n=10), MiniHack 25.0 (n=4), Crafter 13.6 (n=2, capped at 150 steps), TextWorld 6.9 (n=6), BabaIsAI 0.0 (n=4). 8/26 episodes fully solved; Wilson 95% interval [0.165, 0.500].

**What ran:** the BALROG source `balrog-ai/BALROG@b7afe79` with its native `Evaluator.run_episode`, `NaiveAgent`, the native `vllm` OpenAI client and `collect_and_summarize_results`. The client points at a local OpenAI-compatible shim (`code/tinker_shim.py`). The shim samples base Qwen/Qwen3.6-35B-A3B on Tinker with no adapter, the non-thinking template, temperature 0 and max_tokens 128. The games ran locally on macOS arm64 in a Python 3.10 venv with the pinned forks of Minigrid, baba-is-ai, TextWorld and minihack, plus balrog-nle 0.9.0. The native assets came from `public_portfolio_2026-09-05/balrog_setup/linux_install01/assets/`. NetHack was not run.

**Items:** `raw/plan.json` (seed 20260926). The environment seed is 20260926+episode_idx. The per-episode native JSON/CSV files are in `raw/<env>/<task>/`. Every Tinker call is in `raw/shim_calls.jsonl`. `raw/native_summary.json` holds the native aggregation, and `raw/per_episode.json` the per-episode table.

**Cost:** 4,184,658 Tinker tokens across 1,947 calls. That is under the 6M cap; almost all of it is prefill, with about 5k output tokens. No Colab or Modal was used. Sampler speed was 1.39 s per call and 2.55 output tokens per call, with 5 drivers running concurrently (about 3.7 calls/s in aggregate).

**Rerun:**
```sh
# venv: uv venv -p 3.10; install gym==0.23 numpy<2 scipy==1.13.1 setuptools<70 openai hydra-core crafter tatsu==5.8.3
#   + the 4 pinned git forks + balrog-nle==0.9.0; pip install -e <balrog src> --no-deps
# unzip textworld.zip into cwd; unzip boxoban.zip into minihack/dat as boxoban-levels-master
set -a; source .env; set +a
SHIM_LOG=E13/raw/shim_calls.jsonl SHIM_BUDGET=5800000 SHIM_MAX_TOKENS=128 PORT=8773 \
  /Users/arvind/.local/share/uv/tools/tinker/bin/python E13/code/tinker_shim.py &
export BALROG_SRC=<balrog src> E13_OUT=E13/raw
python E13/code/run_e13.py --plan
for e in babyai textworld babaisai minihack crafter; do python E13/code/run_e13.py --env $e & done; wait
python E13/code/run_e13.py --summarize && python3 E13/code/make_result.py
```

**Caveats:** this is a new arm and must not be pooled with the seed809 LoRA plan. The run deviates from native BALROG in temperature, max_tokens, seeds, the Crafter step cap and the missing NetHack environment. The overall figure is a 5-environment mean, not the native 6-environment one. Per-environment n is small. The full list is in `result.json`.
