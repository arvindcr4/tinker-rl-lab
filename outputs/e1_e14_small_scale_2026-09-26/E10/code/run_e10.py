"""E10 small-scale: AgentHarm test_public, 30 harmful + 30 benign by seed 20260926, native inspect_evals harness
(pinned b935c0e5), actor AND both judges = Qwen3.6 base via local Tinker shim (GPT-4o judges not used)."""
import json, os, random, sys
from pathlib import Path
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model
from inspect_evals.agentharm import agentharm, agentharm_benign

OUT = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26/E10")
BENCH = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e10_agentharm/hf_dataset/benchmark")
SEED, N = 20260926, 30
rng = random.Random(SEED)
sel = {}
for kind in ("harmful", "benign"):
    ids = [b["id"] for b in json.loads((BENCH / f"{kind}_behaviors_test_public.json").read_text())["behaviors"]]
    sel[kind] = sorted(rng.sample(ids, N), key=lambda s: tuple(map(int, s.split("-"))))
(OUT / "raw").mkdir(parents=True, exist_ok=True)
assert json.loads((OUT / "raw/selected_ids.json").read_text()) == sel

M = "openai/qwen36-35b-a3b-base"
# inspect's openai provider defaults to the Responses API; the shim only implements chat.completions.
# Judges: always the Tinker-base shim (JUDGE_BASE_URL); in the paired vLLM arms the actor shim differs.
JUDGE_URL = os.environ.get("JUDGE_BASE_URL")
ARM = os.environ.get("ARM")
LOGS = OUT / (f"vllm_{ARM}/raw/inspect_logs" if ARM else "raw/inspect_logs")
judge = lambda: get_model(M, memoize=False, responses_api=False, **({"base_url": JUDGE_URL} if JUDGE_URL else {}))
for kind, task_fn in (("harmful", agentharm), ("benign", agentharm_benign)):
    task = task_fn(split="test_public", refusal_judge=judge(), semantic_judge=judge())
    inspect_eval(task, model=M, sample_id=sel[kind], log_dir=str(LOGS / kind),
                 model_args={"responses_api": False}, temperature=0.0, max_connections=4 if ARM else 12, fail_on_error=False, log_format="json")
