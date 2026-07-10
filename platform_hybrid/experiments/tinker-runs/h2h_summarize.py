"""Aggregate the Tinker h2h results and log to Weights & Biases.

Reads the completed h2h2_* result JSONs, writes week_h2h/SUMMARY.json, and logs
a per-arm table to W&B (auth via ~/.netrc).
"""
import json, glob, re, statistics, pathlib
from collections import defaultdict

ROOT = pathlib.Path("experiments/tinker-runs/results")
cells = {}
for f in sorted((ROOT / "week_h2h").glob("*.json")) + sorted(ROOT.glob("h2h2_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get("status") == "completed":
        cells.setdefault(d["tag"], d)

arms = defaultdict(list)
for tag, d in cells.items():
    arm = re.search(r"h2h2_([a-z]+)_", tag).group(1)
    sl = d.get("step_log", [])
    arms[arm].append({"seed": d.get("seed"), "last10": d.get("last10_avg"),
                      "peak": d.get("peak_reward"),
                      "zvf": statistics.mean([s.get("zvf", 0) for s in sl]) if sl else None})

summary = {"experiment": "tinker_h2h_qwen3.5-4b_gsm8k", "model": "Qwen/Qwen3.5-4B",
           "config": "G8 Dmedium R16 S15, seeds 42/123/456", "cells": len(cells),
           "cross_stack_note": "DAPO ZVF=0.58 here (closed Tinker, grpo+asym-clip surrogate) "
                               "vs ~0.00 in open Colab E3 (true dynamic sampling) -- same label, "
                               "different behavior; the stack determines the result.",
           "by_arm": {}}
for arm in ["grpo", "drgrpo", "dapo", "gspo"]:
    rs = arms.get(arm, [])
    if not rs:
        continue
    mean = lambda k: round(statistics.mean([r[k] for r in rs if r[k] is not None]), 3)
    summary["by_arm"][arm] = {"n": len(rs), "mean_last10_reward": mean("last10"),
                              "mean_peak": mean("peak"), "mean_zvf": mean("zvf")}

(ROOT / "week_h2h" / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print("[h2h] wrote week_h2h/SUMMARY.json")

try:
    import wandb
    run = wandb.init(project="zvf-colab-experiments", name="tinker-h2h-qwen3.5-4b", reinit=True,
                     config={"config": summary["config"], "model": summary["model"]},
                     tags=["tinker", "h2h", "closed-stack", "qwen3.5-4b"])
    t = wandb.Table(columns=["arm", "n", "mean_last10_reward", "mean_peak", "mean_zvf"])
    for arm, v in summary["by_arm"].items():
        t.add_data(arm, v["n"], v["mean_last10_reward"], v["mean_peak"], v["mean_zvf"])
    wandb.log({"h2h_by_arm": t})
    wandb.summary.update({f"{a}_last10": v["mean_last10_reward"] for a, v in summary["by_arm"].items()})
    print("[h2h] wandb:", run.url)
    wandb.finish()
except Exception as e:
    print(f"[h2h] wandb skipped: {e}")
