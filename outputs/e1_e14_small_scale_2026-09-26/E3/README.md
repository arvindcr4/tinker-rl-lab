# E3 (SDAB) small-scale substitute: Terminal-Bench v2.0, 8 tasks. Result: 4/8 = 0.50 (Wilson 95% [0.22, 0.78])

**What ran:** the `Qwen/Qwen3.6-35B-A3B` base model (no adapter), sampled on Tinker with thinking off and temperature 0, drove Harbor's `terminus-2` agent through a local OpenAI-compatible shim (`code/tinker_shim.py`). Each task ran in its native TB2 container on Modal, because every TB2 image is an amd64-only single manifest and this host is aarch64. The scores come from the native TB2 verifiers (`reward.txt`).

**Why a substitute:** SDAB data is private and blocked externally. Terminal-Bench 2 is generic terminal work, not SDAB's DevOps categories or stateful grading.

**Selection:** `raw/eligibility.json` records all 89 tasks. The filter kept tasks with difficulty below hard, an agent timeout of at most 900 s, 1 CPU and 2 GB (35 eligible). From that sorted list, `random.Random(20260926).sample(eligible, 8)` picked the subset → `raw/selection.json`.
Passed: build-pmars, cobol-modernization, hf-model-inference, pytorch-model-recovery. Failed: build-cython-ext, count-dataset-tokens, largest-eigenval, query-optimize. There were no errors.

**Caps and cost:** Tinker used 1,455,114 tokens (prefill + sample) over 132 calls. The ledger is `raw/shim_calls.jsonl`, and the shim's hard budget was 2.6M. The Modal cost bills to the shared `__harbor__` app (see E7/README.md).

**Rerun** (from this directory):
```
set -a; source ../../../.env; set +a
SHIM_LOG=raw/shim_calls.jsonl SHIM_BUDGET=2600000 PORT=8763 /Users/arvind/.local/share/uv/tools/tinker/bin/python code/tinker_shim.py &
harbor datasets download terminal-bench@2.0 -o code/tb2
./code/run_e3.sh          # terminus-2, max_turns=25, 32k ctx, Modal, -n 4
python3 code/finalize.py E3 '<overrides as in result.json>'
```
**Caveats:** this is a new arm and must not be pooled with the original lane. The filter makes the subset easier than the full TB2 set. max_turns=25 is far below TB2's native agent budget. With n=8 the CI is wide.
