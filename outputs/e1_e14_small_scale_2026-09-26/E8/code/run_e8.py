"""E8 small-scale: LAB-Bench public, 10 per category (80), base Qwen3.6 on Tinker, native LAB-Bench prompt/parser/scorer."""
import asyncio, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab")
sys.path.insert(0, str(REPO / "zvf-program/flagship"))
import public_portfolio_native as ppn  # native loader: pinned LAB-Bench + chembench parser
import os
if os.environ.get('ARM'):  # paired vLLM arm: same items/prompts/caps, different endpoint
    import vc as tk
else:
    import tk
WORKERS = 4 if os.environ.get('ARM') else 16

OUT = REPO / "outputs/e1_e14_small_scale_2026-09-26/E8"
RAW = (OUT / f"vllm_{os.environ['ARM']}" / "raw") if os.environ.get("ARM") else OUT / "raw"; RAW.mkdir(parents=True, exist_ok=True)
SETUP = ppn.DEFAULT_SETUP
PER_CAT, SEED, MAX_TOKENS = 10, 20260926, 4096

manifest = json.loads((SETUP / "prepared/manifest.json").read_text())
keys = json.loads((SETUP / "prepared/answer_key.json").read_text())["answers"]
native = ppn.load_native(SETUP)
root = SETUP / f"LAB-Bench-{ppn.SOURCE_REVISION}"

rng = random.Random(SEED)
selected = []
for cat in sorted(ppn.EXPECTED_COUNTS):
    pool = [t for t in manifest["tasks"] if t["category"] == cat]  # canonical order = sorted task_id
    selected += rng.sample(pool, PER_CAT)


def run(task):
    rec_path = RAW / (task["task_id"].replace("/", "__") + ".json")
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    imgs = [(root / im["path"]).read_bytes() for im in task["images"]]
    content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": task["prompt"]}]
    msg = [{"role": "user", "content": content if imgs else task["prompt"]}]
    rec = {"task_id": task["task_id"], "category": task["category"], "prompt_sha256": task["prompt_sha256"]}
    try:
        r = tk.sample(msg, MAX_TOKENS, images=imgs)
        rec.update(raw_text=r["text"], prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
                   stop_reason=str(r["stop_reason"]), error=None)
    except Exception as e:
        rec.update(raw_text=None, error=f"{type(e).__name__}: {e}")
    rec_path.write_text(json.dumps(rec, indent=1))
    return rec


t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with ThreadPoolExecutor(WORKERS) as ex:
    recs = list(ex.map(run, selected))

# Native grading (same logic as public_portfolio_native.grade_response, minus the HTTP envelope).
graded = []
for task, rec in zip(selected, recs):
    k = keys[task["task_id"]]
    g = {"task_id": task["task_id"], "category": task["category"], "target_choice": k["target_choice"],
         "unsure_choice": k["unsure_choice"], "error": rec.get("error")}
    if rec.get("raw_text") is None:
        g.update(agent_output=None, unanswerable=True, correct=False, sure=False)
    else:
        try:
            p = asyncio.run(ppn.parse_native(native, task, rec["raw_text"], True))
        except Exception as e:  # parser exception counted as failure
            p = {"agent_output": None, "unanswerable": True, "parser_error": str(e)}
        g.update(p, correct=not p["unanswerable"] and p["agent_output"] == k["target_choice"],
                 sure=not p["unanswerable"] and p["agent_output"] != k["unsure_choice"])
    graded.append(g)
(RAW / "graded.jsonl").write_text("".join(json.dumps(g) + "\n" for g in graded))
metrics = native.Evaluator.compute_metrics(graded)
per_cat = {c: sum(g["correct"] for g in graded if g["category"] == c) for c in sorted(ppn.EXPECTED_COUNTS)}
summary = {"started_utc": t0, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n": len(graded), "correct": sum(g["correct"] for g in graded), "native_metrics": metrics,
           "per_category_correct_of_10": per_cat, "errors": sum(1 for g in graded if g["error"]),
           "tokens": {"prefill": sum(r.get("prompt_tokens", 0) for r in recs),
                      "sample": sum(r.get("completion_tokens", 0) for r in recs)},
           "item_ids": [t["task_id"] for t in selected]}
(RAW / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "item_ids"}, indent=1))
