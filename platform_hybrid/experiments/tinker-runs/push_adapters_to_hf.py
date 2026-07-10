"""Consolidate all completed h2h LoRA adapters into ONE HF repo.

Each run goes to its own branch/revision (e.g. grpo-s42, drgrpo-s123) of a single
repo, instead of N separate repos. Idempotent: skips runs already uploaded, so it
can be re-run as more cells finish. Retries on the Tinker->HF transfer timeout.

Requires env: HF_TOKEN, TINKER_API_KEY.
Usage:  HF_TOKEN=... TINKER_API_KEY=... .venv/bin/python platform_hybrid/experiments/tinker-runs/push_adapters_to_hf.py
"""
import json, glob, re, subprocess, os, time
from huggingface_hub import HfApi

REPO = "arvindcr4/zvf-h2h-qwen3.5-4b"
TINKER = ".venv/bin/tinker"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, private=True, exist_ok=True, repo_type="model")

def branch_of(tag):
    m = re.search(r"h2h2_([a-z]+)_.*_(s\d+)_", tag)
    return f"{m.group(1)}-{m.group(2)}" if m else re.sub(r"[^a-zA-Z0-9._-]", "-", tag)

# completed cells (dedup by tag, prefer week_h2h copy)
cells = {}
for f in sorted(glob.glob("platform_hybrid/experiments/tinker-runs/results/week_h2h/*.json")) + \
         sorted(glob.glob("platform_hybrid/experiments/tinker-runs/results/h2h2_*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    if d.get("status") == "completed" and d.get("checkpoint"):
        cells.setdefault(d["tag"], d)

print(f"[consolidate] {len(cells)} completed cells -> {REPO}")
pushed, skipped, failed = [], [], []
for tag, d in sorted(cells.items()):
    br, ck = branch_of(tag), d["checkpoint"]
    api.create_branch(REPO, branch=br, exist_ok=True)
    try:
        if "adapter_model.safetensors" in api.list_repo_files(REPO, revision=br):
            skipped.append(br); print(f"[skip] {br} already uploaded"); continue
    except Exception:
        pass
    ok = False
    for attempt in range(1, 4):
        r = subprocess.run(
            [TINKER, "checkpoint", "push-hf", ck, "--repo", REPO, "--revision", br,
             "--commit-message", tag, "--no-model-card"],
            env=os.environ, capture_output=True, text=True)
        try:
            if "adapter_model.safetensors" in api.list_repo_files(REPO, revision=br):
                ok = True; break
        except Exception:
            pass
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        print(f"[retry] {br} attempt {attempt}: {tail[0][-100:]}")
        time.sleep(4)
    (pushed if ok else failed).append(br)
    print(f"[{'OK' if ok else 'FAIL'}] {br}  reward(last10)={d.get('last10_avg')}")

# index README on main
rows = "\n".join(
    f"| `{branch_of(t)}` | {d.get('tag')} | {d.get('last10_avg')} |"
    for t, d in sorted(cells.items()))
readme = f"""---
base_model: Qwen/Qwen3.5-4B
library_name: peft
tags: [lora, grpo, drgrpo, dapo, gspo, gsm8k, zvf]
---
# ZVF head-to-head LoRA adapters (Qwen3.5-4B, GSM8K)

GRPO / Dr.GRPO / DAPO / GSPO LoRA rank-16 adapters, one per **branch/revision**.
Load with the base model + `peft`, selecting the branch via `revision=`.

| branch | run | last10 reward |
|---|---|---|
{rows}
"""
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                repo_id=REPO, revision="main", commit_message="index of run branches")
print("PUSHED:", pushed)
print("SKIPPED:", skipped)
print("FAILED:", failed)
print("REPO: https://huggingface.co/" + REPO)
