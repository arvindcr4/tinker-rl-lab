"""E14 small-scale actor: Omni-MATH test, 100 rows by seed 20260926, native actor prompt (system + problem), base Qwen3.6 on Tinker."""
import hashlib, json, random, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
if os.environ.get('ARM'):  # paired vLLM arm: same items/prompts/caps, different endpoint
    import vc as tk
else:
    import tk
WORKERS = 4 if os.environ.get('ARM') else 16

REPO = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab")
SETUP = REPO / "outputs/public_portfolio_2026-09-05/omni_setup"
RAW = REPO / "outputs/e1_e14_small_scale_2026-09-26/E14" / (f"vllm_{os.environ['ARM']}/raw" if os.environ.get("ARM") else "raw"); RAW.mkdir(parents=True, exist_ok=True)
SYSTEM_PROMPT = "You are an experienced educator in the field of MATHEMATICS."  # native, per public_omni_math_native.py
N, SEED, MAX_TOKENS = 100, 20260926, 2048

data = (SETUP / "test.jsonl").read_bytes()
assert hashlib.sha256(data).hexdigest() == "7c87be8ee41ac7c7a597ef5a5500e84bd2b639a85a06db3da7f69bf9a32ef168"
rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
assert len(rows) == 4428
idx = sorted(random.Random(SEED).sample(range(len(rows)), N))


def run(i):
    p = RAW / "actor" / f"row{i:05d}.json"; p.parent.mkdir(exist_ok=True)
    if p.exists():
        return json.loads(p.read_text())
    r = rows[i]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["problem"]}]
    rec = {"row_index": i, "domain": r["domain"], "difficulty": r["difficulty"], "source": r["source"],
           "problem": r["problem"], "answer": r["answer"]}
    try:
        o = tk.sample(msgs, MAX_TOKENS)
        rec.update(model_generation=o["text"], prompt_tokens=o["prompt_tokens"],
                   completion_tokens=o["completion_tokens"], stop_reason=str(o["stop_reason"]), error=None)
    except Exception as e:
        rec.update(model_generation="", error=f"{type(e).__name__}: {e}")
    p.write_text(json.dumps(rec, indent=1))
    return rec


t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with ThreadPoolExecutor(WORKERS) as ex:
    recs = list(ex.map(run, idx))
with open(RAW / "actor_generations.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
json.dump({"started_utc": t0, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "row_indices": idx,
           "errors": sum(1 for r in recs if r["error"]),
           "truncated": sum(1 for r in recs if r.get("stop_reason") == "length"),
           "prefill": sum(r.get("prompt_tokens", 0) for r in recs),
           "sample": sum(r.get("completion_tokens", 0) for r in recs)}, open(RAW / "actor_summary.json", "w"), indent=1)
print(open(RAW / "actor_summary.json").read()[-300:])
