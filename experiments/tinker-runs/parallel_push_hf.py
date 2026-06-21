"""Parallel consolidation: push all completed h2h adapters to one HF repo, ~5 at a
time, each to its own branch. Idempotent (skips branches already uploaded),
retries on the Tinker->HF timeout. Requires env HF_TOKEN, TINKER_API_KEY.
"""
import json, glob, re, subprocess, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi

REPO = "arvindcr4/zvf-h2h-qwen3.5-4b"
TINKER = ".venv/bin/tinker"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, private=True, exist_ok=True, repo_type="model")

def branch_of(tag):
    m = re.search(r"h2h2_([a-z]+)_.*_(s\d+)_", tag)
    return f"{m.group(1)}-{m.group(2)}"

cells = {}
for f in sorted(glob.glob("experiments/tinker-runs/results/week_h2h/*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    if d.get("status") == "completed" and d.get("checkpoint"):
        cells.setdefault(d["tag"], d)

def push(tag, d):
    br, ck = branch_of(tag), d["checkpoint"]
    try:
        if "adapter_model.safetensors" in api.list_repo_files(REPO, revision=br):
            return (br, "skip")
    except Exception:
        pass
    api.create_branch(REPO, branch=br, exist_ok=True)
    for _ in range(3):
        subprocess.run([TINKER, "checkpoint", "push-hf", ck, "--repo", REPO,
                        "--revision", br, "--no-model-card", "--commit-message", tag],
                       env=os.environ, capture_output=True, text=True)
        try:
            if "adapter_model.safetensors" in api.list_repo_files(REPO, revision=br):
                return (br, "ok")
        except Exception:
            pass
        time.sleep(3)
    return (br, "FAIL")

print(f"[parallel] {len(cells)} cells -> {REPO}", flush=True)
done = []
with ThreadPoolExecutor(max_workers=5) as ex:
    futs = {ex.submit(push, t, d): t for t, d in cells.items()}
    for fu in as_completed(futs):
        r = fu.result(); done.append(r); print("[push]", r, flush=True)

rows = "\n".join(f"| `{branch_of(t)}` | {d.get('tag')} | {d.get('last10_avg')} |"
                 for t, d in sorted(cells.items()))
readme = ("---\nbase_model: Qwen/Qwen3.5-4B\nlibrary_name: peft\n"
          "tags: [lora, grpo, drgrpo, dapo, gspo, gsm8k, zvf]\n---\n"
          "# ZVF head-to-head LoRA adapters (Qwen3.5-4B, GSM8K)\n\n"
          "GRPO / Dr.GRPO / DAPO / GSPO LoRA rank-16 adapters, one per **branch**. "
          "Load base model + `peft`, select branch via `revision=`.\n\n"
          f"| branch | run | last10 reward |\n|---|---|---|\n{rows}\n")
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                repo_id=REPO, revision="main", commit_message="index of run branches")
ok = [b for b, s in done if s == "ok"]; sk = [b for b, s in done if s == "skip"]; fa = [b for b, s in done if s == "FAIL"]
print(f"OK={len(ok)} SKIP={len(sk)} FAIL={fa}", flush=True)
print("REPO: https://huggingface.co/" + REPO, flush=True)
