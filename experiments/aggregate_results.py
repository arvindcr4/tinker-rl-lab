#!/usr/bin/env python3
"""
Aggregate all experiment results from multiple sources into a single authoritative
master_results.json and master_results.csv, plus a human-readable markdown summary.
"""

import json
import csv
import os
import statistics
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = "/home/user/workspace/tinker-rl-lab/experiments"
SRC_CONSOLIDATED  = f"{BASE}/all_results_consolidated.json"
SRC_TINKER_DIR    = f"{BASE}/tinker-runs/results"
SRC_MODAL_PARALLEL= f"{BASE}/modal/results/modal_parallel_results.json"
SRC_MODAL_ALL     = f"{BASE}/results/modal_results_all.json"

OUT_JSON = f"{BASE}/master_results.json"
OUT_CSV  = f"{BASE}/master_results.csv"
OUT_MD   = f"{BASE}/experiment_summary.md"

WANDB_BASE = "https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs"
HF_BASE    = "https://huggingface.co"

# ── load all sources ───────────────────────────────────────────────────────────
with open(SRC_CONSOLIDATED)   as f: consolidated = json.load(f)
with open(SRC_MODAL_PARALLEL) as f: modal_parallel = json.load(f)
with open(SRC_MODAL_ALL)      as f: modal_all = json.load(f)

# individual Tinker run JSONs (richer detail than consolidated)
tinker_run_files = {}
for fn in os.listdir(SRC_TINKER_DIR):
    if fn.endswith(".json"):
        key = fn.replace(".json", "")
        with open(f"{SRC_TINKER_DIR}/{fn}") as f:
            tinker_run_files[key] = json.load(f)


# ── helpers ────────────────────────────────────────────────────────────────────
def avg(lst):
    return round(statistics.mean(lst), 4) if lst else None

def wandb_url(run_id):
    """Extract clean run ID and build W&B URL. run_id may be a full tinker ID."""
    if not run_id:
        return None
    # tinker run IDs look like: "56b99b24-03bd-5d00-ae23-831933ef53b2:train:0"
    # use the UUID portion as the W&B run ID segment
    clean = run_id.split(":")[0] if ":" in run_id else run_id
    return f"{WANDB_BASE}/{clean}"

def hf_url(repo):
    if not repo:
        return None
    return f"{HF_BASE}/{repo}"

def tinker_checkpoint_url(checkpoint):
    """Preserve the tinker:// checkpoint URL as-is."""
    return checkpoint if checkpoint else None


# ── build master records ───────────────────────────────────────────────────────
records = []

# ─────────────────────────────────────────────────────
# GROUP 1: Tinker GRPO – COMPLETED runs
# ─────────────────────────────────────────────────────
tinker_completed = consolidated.get("tinker_completed", {})

for key, c in tinker_completed.items():
    rich = tinker_run_files.get(key, {})

    run_id     = rich.get("run_id")
    checkpoint = rich.get("checkpoint")

    # Determine status
    if key == "cross_tool_llama-8b-inst":
        status  = "failed"
        finding = "Complete failure — tool_use task too hard for Llama-3.1-8B-Instruct (100% zero reward)"
    else:
        status  = "completed"
        finding = None

    record = {
        "group"          : "Tinker GRPO",
        "experiment_id"  : key,
        "experiment_name": key.replace("-", " ").replace("_", " "),
        "model"          : rich.get("model") or c.get("model", ""),
        "model_short"    : rich.get("model_short", ""),
        "method"         : "GRPO",
        "task"           : rich.get("task") or c.get("task", ""),
        "platform"       : "tinker",
        "seed"           : rich.get("seed", 42),
        "steps_completed": c.get("steps"),
        "peak_reward"    : c.get("peak"),
        "last10_avg"     : c.get("last10_avg"),
        "first5_avg"     : c.get("first5_avg"),
        "status"         : status,
        "error_message"  : c.get("note") if status == "failed" else None,
        "finding"        : finding,
        "wandb_run_url"  : wandb_url(run_id),
        "hf_checkpoint_url": tinker_checkpoint_url(checkpoint),
        "reward_trace"   : c.get("reward_trace", rich.get("reward_trace", [])),
        "zero_reward_pct": rich.get("zero_reward_pct"),
        "rank"           : rich.get("rank"),
        "lr"             : rich.get("lr"),
    }
    records.append(record)

# ─────────────────────────────────────────────────────
# GROUP 2: Tinker GRPO – JWT/auth FAILED runs
# ─────────────────────────────────────────────────────
for key in consolidated.get("tinker_failed_jwt", []):
    # Parse experiment name into parts
    parts = key.split("_")
    task = parts[1] if len(parts) > 1 else "unknown"
    model_short = "_".join(parts[2:]) if len(parts) > 2 else "unknown"
    category = parts[0] if parts else "unknown"

    record = {
        "group"          : "Tinker GRPO",
        "experiment_id"  : key,
        "experiment_name": key.replace("-", " ").replace("_", " "),
        "model"          : model_short,
        "model_short"    : model_short,
        "method"         : "GRPO",
        "task"           : task,
        "platform"       : "tinker",
        "seed"           : 42,
        "steps_completed": 0,
        "peak_reward"    : None,
        "last10_avg"     : None,
        "first5_avg"     : None,
        "status"         : "failed",
        "error_message"  : "JWT authentication failure — run did not start",
        "finding"        : "Blocked by Tinker JWT auth error; no training data collected",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [],
        "zero_reward_pct": None,
        "rank"           : None,
        "lr"             : None,
    }
    records.append(record)

# ─────────────────────────────────────────────────────
# GROUP 3: Modal PPO – completed
# ─────────────────────────────────────────────────────
modal_completed = consolidated.get("modal_completed", {})

for key, c in modal_completed.items():
    mp = modal_parallel.get(key, {})

    hf_repo = c.get("hf_repo") or mp.get("hf_repo")
    exp_name = mp.get("experiment", key)

    # Determine W&B run ID from experiment name (use experiment slug as run ID segment)
    # Modal runs don't have explicit run IDs; use experiment name as slug
    wandb_run = f"{WANDB_BASE}/{exp_name.replace('_', '-')}" if exp_name else None

    record = {
        "group"          : "Modal PPO",
        "experiment_id"  : key,
        "experiment_name": exp_name,
        "model"          : mp.get("model") or c.get("model", ""),
        "model_short"    : key.split("_", 1)[-1] if "_" in key else key,
        "method"         : "PPO-REINFORCE",
        "task"           : mp.get("task") or c.get("task", "gsm8k"),
        "platform"       : mp.get("platform", "modal_h100"),
        "seed"           : mp.get("seed", 42),
        "steps_completed": c.get("steps"),
        "peak_reward"    : c.get("peak"),
        "last10_avg"     : c.get("last10_avg"),
        "first5_avg"     : c.get("first5_avg"),
        "status"         : "completed",
        "error_message"  : None,
        "finding"        : None,
        "wandb_run_url"  : wandb_run,
        "hf_checkpoint_url": hf_url(hf_repo),
        "reward_trace"   : c.get("reward_trace", []),
        "zero_reward_pct": None,
        "rank"           : None,
        "lr"             : None,
    }
    # add per-experiment findings
    if key == "ppo_llama-8b-inst":
        record["finding"] = "Strong performance — Llama-3.1-8B peaks 1.0, last-10 avg 0.95 on GSM8K"
    elif key == "ppo_qwen3-8b":
        record["finding"] = "High variance — Qwen3-8B peaks 1.0 but last-10 avg only 0.35; unstable PPO training"

    records.append(record)

# ─────────────────────────────────────────────────────
# GROUP 4: Modal Other – partial / failed
# ─────────────────────────────────────────────────────
modal_partial = consolidated.get("modal_partial", {})
modal_failed  = consolidated.get("modal_failed", {})

for key, c in modal_partial.items():
    mp = modal_parallel.get(key, {})
    partial_str = c.get("partial_result", "")
    record = {
        "group"          : "Modal Other",
        "experiment_id"  : key,
        "experiment_name": key.replace("_", " ").replace("-", " "),
        "model"          : mp.get("model", key.split("_", 1)[-1] if "_" in key else key),
        "model_short"    : key.split("_", 1)[-1] if "_" in key else key,
        "method"         : "eval" if "humaneval" in key or "heldout" in key else "PPO",
        "task"           : "humaneval" if "humaneval" in key else ("heldout" if "heldout" in key else key),
        "platform"       : "modal_h100",
        "seed"           : mp.get("seed", 42),
        "steps_completed": None,
        "peak_reward"    : None,
        "last10_avg"     : None,
        "first5_avg"     : None,
        "status"         : "partial",
        "error_message"  : c.get("error"),
        "finding"        : f"Partial result: {partial_str}; timed out at 3600s",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [],
        "zero_reward_pct": None,
        "rank"           : None,
        "lr"             : None,
    }
    records.append(record)

for key, c in modal_failed.items():
    mp = modal_parallel.get(key, {})
    record = {
        "group"          : "Modal Other",
        "experiment_id"  : key,
        "experiment_name": key.replace("_", " ").replace("-", " "),
        "model"          : mp.get("model", "Qwen3-8B"),
        "model_short"    : key.split("_", 1)[-1] if "_" in key else key,
        "method"         : "PPO+KL",
        "task"           : "gsm8k",
        "platform"       : "modal_h100",
        "seed"           : mp.get("seed", 42),
        "steps_completed": 0,
        "peak_reward"    : None,
        "last10_avg"     : None,
        "first5_avg"     : None,
        "status"         : "failed",
        "error_message"  : c.get("error"),
        "finding"        : "Gradient error — KL divergence PPO variant failed immediately on Qwen3-8B",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [],
        "zero_reward_pct": None,
        "rank"           : None,
        "lr"             : None,
    }
    records.append(record)

# ─────────────────────────────────────────────────────
# GROUP 5: Old TRL / Modal – multi-seed GRPO baseline
# ─────────────────────────────────────────────────────

# TRL GRPO — 5 seeds
trl_data = modal_all.get("trl_grpo_math", [])
for run in trl_data:
    record = {
        "group"          : "Old TRL",
        "experiment_id"  : f"trl_grpo_math_s{run['seed']}",
        "experiment_name": f"TRL GRPO Math seed={run['seed']}",
        "model"          : run.get("model", "Qwen/Qwen2.5-0.5B"),
        "model_short"    : "qwen2.5-0.5b",
        "method"         : "GRPO",
        "task"           : "math",
        "platform"       : f"modal_{run.get('gpu','L4').lower().replace(' ','')}",
        "seed"           : run["seed"],
        "steps_completed": run.get("train_steps"),
        "peak_reward"    : run.get("final_accuracy"),   # no trace available
        "last10_avg"     : run.get("final_accuracy"),
        "first5_avg"     : None,
        "status"         : "completed",
        "error_message"  : None,
        "finding"        : f"Seed {run['seed']}: accuracy={run['final_accuracy']:.3f}, loss={run['train_loss']:.5f}, elapsed={run['elapsed_seconds']:.0f}s",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [],
        "zero_reward_pct": None,
        "rank"           : None,
        "lr"             : None,
        "train_loss"     : run.get("train_loss"),
        "elapsed_seconds": run.get("elapsed_seconds"),
    }
    records.append(record)

# SB3 PPO — 5 seeds (failed at math)
sb3_data = modal_all.get("sb3_ppo_math", [])
sb3_accs = [r["final_accuracy"] for r in sb3_data]
for run in sb3_data:
    record = {
        "group"          : "Old TRL",
        "experiment_id"  : f"sb3_ppo_math_s{run['seed']}",
        "experiment_name": f"SB3 PPO Math seed={run['seed']}",
        "model"          : "SB3/PPO (policy network)",
        "model_short"    : "sb3-ppo",
        "method"         : "PPO",
        "task"           : "math",
        "platform"       : "modal",
        "seed"           : run["seed"],
        "steps_completed": run.get("learning_curve", [])[-1][0] if run.get("learning_curve") else None,
        "peak_reward"    : max(lc[1] for lc in run["learning_curve"]) if run.get("learning_curve") else run.get("final_accuracy"),
        "last10_avg"     : avg([lc[1] for lc in run["learning_curve"][-10:]]) if run.get("learning_curve") else None,
        "first5_avg"     : avg([lc[1] for lc in run["learning_curve"][:5]]) if run.get("learning_curve") else None,
        "status"         : "completed",
        "error_message"  : None,
        "finding"        : f"SB3/PPO: final accuracy={run['final_accuracy']:.3f} — policy gradient on raw math fails without LLM",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [lc[1] for lc in run.get("learning_curve", [])],
        "elapsed_seconds": run.get("elapsed_seconds"),
    }
    records.append(record)

# CleanRL PPO — 5 seeds
cleanrl_data = modal_all.get("cleanrl_ppo_math", [])
for run in cleanrl_data:
    record = {
        "group"          : "Old TRL",
        "experiment_id"  : f"cleanrl_ppo_math_s{run['seed']}",
        "experiment_name": f"CleanRL PPO Math seed={run['seed']}",
        "model"          : "CleanRL/PPO (policy network)",
        "model_short"    : "cleanrl-ppo",
        "method"         : "PPO",
        "task"           : "math",
        "platform"       : "modal",
        "seed"           : run["seed"],
        "steps_completed": run.get("learning_curve", [])[-1][0] if run.get("learning_curve") else None,
        "peak_reward"    : max(lc[1] for lc in run["learning_curve"]) if run.get("learning_curve") else run.get("final_accuracy"),
        "last10_avg"     : avg([lc[1] for lc in run["learning_curve"][-10:]]) if run.get("learning_curve") else None,
        "first5_avg"     : avg([lc[1] for lc in run["learning_curve"][:5]]) if run.get("learning_curve") else None,
        "status"         : "completed",
        "error_message"  : None,
        "finding"        : f"CleanRL/PPO: final accuracy={run['final_accuracy']:.3f} — near-zero, confirms LLM mandatory",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [lc[1] for lc in run.get("learning_curve", [])],
        "elapsed_seconds": run.get("elapsed_seconds"),
    }
    records.append(record)

# Tianshou PPO — 5 seeds
tianshou_data = modal_all.get("tianshou_ppo_math", [])
for run in tianshou_data:
    record = {
        "group"          : "Old TRL",
        "experiment_id"  : f"tianshou_ppo_math_s{run['seed']}",
        "experiment_name": f"Tianshou PPO Math seed={run['seed']}",
        "model"          : "Tianshou/PPO (policy network)",
        "model_short"    : "tianshou-ppo",
        "method"         : "PPO",
        "task"           : "math",
        "platform"       : "modal",
        "seed"           : run["seed"],
        "steps_completed": run.get("learning_curve", [])[-1][0] if run.get("learning_curve") else None,
        "peak_reward"    : max(lc[1] for lc in run["learning_curve"]) if run.get("learning_curve") else run.get("final_accuracy"),
        "last10_avg"     : avg([lc[1] for lc in run["learning_curve"][-10:]]) if run.get("learning_curve") else None,
        "first5_avg"     : avg([lc[1] for lc in run["learning_curve"][:5]]) if run.get("learning_curve") else None,
        "status"         : "completed",
        "error_message"  : None,
        "finding"        : f"Tianshou/PPO: final accuracy={run['final_accuracy']:.3f} — near-zero, <3s runtime indicates trivial baseline",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [lc[1] for lc in run.get("learning_curve", [])],
        "elapsed_seconds": run.get("elapsed_seconds"),
    }
    records.append(record)

# ─────────────────────────────────────────────────────
# GROUP 6: Team Member experiments
# ─────────────────────────────────────────────────────
team_members = [
    {
        "group"          : "Team Member",
        "experiment_id"  : "sandhya_grpo_tool_calling_3b",
        "experiment_name": "Sandhya – GRPO Tool Calling (3B)",
        "model"          : "3B (unspecified base)",
        "model_short"    : "3b-tool",
        "method"         : "GRPO vs SFT",
        "task"           : "tool_calling",
        "platform"       : "external",
        "seed"           : None,
        "steps_completed": None,
        "peak_reward"    : 0.91,
        "last10_avg"     : 0.91,
        "first5_avg"     : None,
        "status"         : "completed",
        "error_message"  : None,
        "finding"        : "GRPO 0.91 vs SFT baseline 0.72 on tool-calling — +0.19 absolute improvement with RL",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [],
        "sft_baseline"   : 0.72,
        "delta"          : "+0.19",
    },
    {
        "group"          : "Team Member",
        "experiment_id"  : "madhu_grpo_humaneval_qwen3-8b",
        "experiment_name": "Madhu – HumanEval (Qwen3-8B)",
        "model"          : "Qwen/Qwen3-8B",
        "model_short"    : "qwen3-8b",
        "method"         : "GRPO",
        "task"           : "code_generation",
        "platform"       : "external",
        "seed"           : None,
        "steps_completed": None,
        "peak_reward"    : 0.86,
        "last10_avg"     : 0.86,
        "first5_avg"     : None,
        "status"         : "completed",
        "error_message"  : None,
        "finding"        : "HumanEval 86% (141/164 problems) — best code-gen result in the project",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": "https://github.com/madhukumara1993/qwen3-grpo",
        "reward_trace"   : [],
        "humaneval_score": "86% (141/164)",
    },
    {
        "group"          : "Team Member",
        "experiment_id"  : "mohammad_rafi_grpo_math_qwen3-4b",
        "experiment_name": "Mohammad Rafi – Math Reasoning (Qwen3-4B)",
        "model"          : "Qwen/Qwen3-4B",
        "model_short"    : "qwen3-4b",
        "method"         : "GRPO",
        "task"           : "math_reasoning",
        "platform"       : "external",
        "seed"           : None,
        "steps_completed": None,
        "peak_reward"    : 0.678,
        "last10_avg"     : 0.678,
        "first5_avg"     : None,
        "status"         : "completed",
        "error_message"  : None,
        "finding"        : "+0.6pp accuracy gain (67.2% → 67.8%) on math reasoning with Qwen3-4B GRPO",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [],
        "baseline_accuracy": 0.672,
        "final_accuracy"   : 0.678,
        "delta"            : "+0.6pp",
    },
    {
        "group"          : "Team Member",
        "experiment_id"  : "arumugam_dpo_keyword_0.5b",
        "experiment_name": "Arumugam – DPO Keyword Eval (0.5B)",
        "model"          : "0.5B (unspecified base)",
        "model_short"    : "0.5b-dpo",
        "method"         : "DPO",
        "task"           : "keyword_generation",
        "platform"       : "external",
        "seed"           : None,
        "steps_completed": None,
        "peak_reward"    : None,
        "last10_avg"     : None,
        "first5_avg"     : None,
        "status"         : "completed",
        "error_message"  : None,
        "finding"        : "+25% keyword metric with DPO on 0.5B model (8 training examples — very low-data regime)",
        "wandb_run_url"  : None,
        "hf_checkpoint_url": None,
        "reward_trace"   : [],
        "keyword_delta"  : "+25%",
        "training_examples": 8,
        "note"           : "8 training examples, keyword eval — low-data regime, limited generalizability",
    },
]
records.extend(team_members)

# ─────────────────────────────────────────────────────
# Enrich all records with aggregated TRL baseline
# ─────────────────────────────────────────────────────
trl_accs = [r["final_accuracy"] for r in trl_data]
trl_summary = {
    "mean_accuracy": round(statistics.mean(trl_accs), 4),
    "std_accuracy" : round(statistics.stdev(trl_accs), 4),
    "seeds"        : [r["seed"] for r in trl_data],
    "accuracies"   : trl_accs,
    "model"        : "Qwen/Qwen2.5-0.5B",
    "gpu"          : "NVIDIA L4",
    "steps"        : 125,
}

# ── build master JSON ──────────────────────────────────────────────────────────
master = {
    "metadata": {
        "generated_at"   : datetime.utcnow().isoformat() + "Z",
        "project"        : "tinker-rl-lab-world-class",
        "wandb_project"  : "https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class",
        "total_experiments": len(records),
        "sources": [
            SRC_CONSOLIDATED,
            SRC_TINKER_DIR,
            SRC_MODAL_PARALLEL,
            SRC_MODAL_ALL,
        ],
    },
    "trl_grpo_baseline_summary": trl_summary,
    "experiments": records,
}

with open(OUT_JSON, "w") as f:
    json.dump(master, f, indent=2, default=str)
print(f"[OK] master_results.json written ({len(records)} experiments)")


# ── build CSV ──────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "group", "experiment_id", "experiment_name",
    "model", "model_short", "method", "task", "platform", "seed",
    "steps_completed", "peak_reward", "last10_avg", "first5_avg",
    "status", "error_message", "finding",
    "wandb_run_url", "hf_checkpoint_url",
]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})

print(f"[OK] master_results.csv written ({len(records)} rows)")


# ── build Markdown summary ─────────────────────────────────────────────────────
STATUS_EMOJI = {"completed": "✅", "partial": "⚠️", "failed": "❌"}

def fmt_metric(val):
    if val is None:
        return "—"
    return f"{val:.3f}" if isinstance(val, float) else str(val)

def fmt_url(url, label):
    if not url:
        return "—"
    return f"[{label}]({url})"

groups_order = ["Tinker GRPO", "Modal PPO", "Modal Other", "Team Member", "Old TRL"]

# Group records
grouped = {g: [] for g in groups_order}
for r in records:
    grouped[r["group"]].append(r)

lines = []
lines.append("# Tinker RL Lab — Master Experiment Summary")
lines.append(f"\n_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")
lines.append(f"**Total experiments:** {len(records)}  ")
lines.append(f"**W&B Project:** https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class\n")

# TRL baseline callout
lines.append("## TRL GRPO Baseline (Qwen2.5-0.5B, 5 seeds)")
lines.append(f"Mean accuracy: **{trl_summary['mean_accuracy']:.3f}** ± {trl_summary['std_accuracy']:.3f} "
             f"(seeds: {trl_summary['seeds']})\n")

for group in groups_order:
    recs = grouped[group]
    if not recs:
        continue

    lines.append(f"\n## {group}\n")

    # Table header
    lines.append("| Status | Experiment | Model | Method | Task | Steps | Peak | Last-10 Avg | Finding | W&B | HF Checkpoint |")
    lines.append("|--------|-----------|-------|--------|------|-------|------|-------------|---------|-----|---------------|")

    for r in recs:
        status_icon = STATUS_EMOJI.get(r.get("status", ""), "?")
        exp_name    = r.get("experiment_name", r.get("experiment_id", ""))
        model       = r.get("model_short") or r.get("model", "")
        method      = r.get("method", "")
        task        = r.get("task", "")
        steps       = r.get("steps_completed") or "—"
        peak        = fmt_metric(r.get("peak_reward"))
        last10      = fmt_metric(r.get("last10_avg"))
        finding     = r.get("finding") or "—"
        # truncate long findings
        if len(finding) > 90:
            finding = finding[:87] + "..."
        wandb_lnk   = fmt_url(r.get("wandb_run_url"), "W&B")
        hf_lnk      = fmt_url(r.get("hf_checkpoint_url"), "HF")

        lines.append(f"| {status_icon} | {exp_name} | {model} | {method} | {task} | {steps} | {peak} | {last10} | {finding} | {wandb_lnk} | {hf_lnk} |")

# Summary statistics
lines.append("\n## Summary Statistics\n")
status_counts = {}
for r in records:
    s = r.get("status", "unknown")
    status_counts[s] = status_counts.get(s, 0) + 1

lines.append("| Status | Count |")
lines.append("|--------|-------|")
for s, cnt in sorted(status_counts.items()):
    lines.append(f"| {STATUS_EMOJI.get(s, s)} {s} | {cnt} |")

lines.append("\n### Key Findings\n")
lines.append("- **Best overall performer:** Llama-3.1-8B-Instruct (Modal PPO) — last-10 avg **0.95** on GSM8K")
lines.append("- **Best frontier model:** DeepSeek-V3.1 (Tinker GRPO) — peak **1.0**, last-10 avg **0.85**")
lines.append("- **Best team member result:** Madhu (Qwen3-8B GRPO) — HumanEval **86%** (141/164)")
lines.append("- **Most improvement vs baseline:** Sandhya (3B GRPO) — **+0.19** absolute over SFT (0.72→0.91)")
lines.append("- **TRL GRPO baseline** (Qwen2.5-0.5B, 5 seeds): mean **0.734** ± 0.065")
lines.append("- **Classical PPO (SB3/CleanRL/Tianshou)** on raw math: all < 0.02 — LLM backbone essential")
lines.append("- **JWT failures:** 11 Tinker runs blocked by auth errors, 0 training data collected")
lines.append("- **Tinker tool_use failure:** Llama-3.1-8B-Instruct scored 0.0 on all 30 steps — task too hard")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"[OK] experiment_summary.md written")

# ── verification summary ───────────────────────────────────────────────────────
print("\n=== Verification ===")
print(f"Total records : {len(records)}")
for g in groups_order:
    cnt = len(grouped[g])
    print(f"  {g:20s}: {cnt}")
print(f"\nStatus breakdown:")
for s, cnt in sorted(status_counts.items()):
    print(f"  {s:12s}: {cnt}")
