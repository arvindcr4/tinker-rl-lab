"""P9 Deep Review (2026-10-18): recompute every row of tab:main_results from run JSON.

Peak = max(reward_trace); Last-10 = mean of the last 10 reward_trace entries.
Rows whose only source is a 6-sample log recovery, a console reconstruction
("partial-from-console"), or a hand edit are not reproduced here; they were
removed from the table (see reports/deepreview/dispositions/P9_dnb.jsonl).
Output: platform_hybrid/experiments/results/review_2026-10-18/P9_dnb/table6_recomputed.tsv
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "platform_hybrid/experiments"
TR = EXP / "tinker-runs/results"
OUT = EXP / "results/review_2026-10-18/P9_dnb"


def stats(trace):
    trace = [float(x) for x in trace]
    return len(trace), max(trace), sum(trace[-10:]) / len(trace[-10:])


rows = []


def add(block, label, src, trace, extra=""):
    n, pk, l10 = stats(trace)
    rows.append((block, label, src, n, round(pk, 5), round(l10, 5), extra))


# Single-run Tinker JSONs with full traces
for f, label in [("scale_gsm8k_qwen3-8b.json", "Qwen3-8B"), ("scale_gsm8k_qwen3.5-4b.json", "Qwen3.5-4B"),
                 ("scale_gsm8k_llama-8b-inst.json", "Llama-3.1-8B-Inst"),
                 ("frontier_gsm8k_deepseek-v3.1.json", "DeepSeek-V3.1"),
                 ("frontier_gsm8k_nemotron-120b.json", "Nemotron-120B"),
                 ("arch_gsm8k_kimi-k2.json", "Kimi-K2-Thinking (arch run)"),
                 ("arch_gsm8k_gpt-oss-20b.json", "GPT-OSS-20B"),
                 ("cross_tool_qwen3-32b.json", "Qwen3-32B tool-use"),
                 ("cross_tool_llama-8b-inst.json", "Llama-3.1-8B-Inst tool-use")]:
    d = json.loads((TR / f).read_text())
    assert d.get("run_id") != "partial-from-console", f
    add("tinker_single", label, f"{f}:reward_trace (model={d['model']}, run_id={d['run_id']})", d["reward_trace"])

# Completed 30-step campaign batch (named as source in run_manifest header)
camp = json.loads((TR / "campaign_v2_20260419_111623.json").read_text())
for r in camp["results"]:
    if r.get("status") != "completed":
        continue
    add("campaign_111623", r["tag"], f"campaign_v2_20260419_111623.json:{r['tag']} (model={r['model']}, G={r.get('group_size')}, run_id={r.get('run_id')})", r["reward_trace"])

# Matched launcher comparison
fc = json.loads((EXP / "results/framework_comparison.json").read_text())
for fr in fc["frameworks"]:
    if fr.get("mode") != "real":
        continue
    add("framework_comparison", fr["framework"], f"results/framework_comparison.json ({fr['model']}, source={fr.get('source')})", fr["reward_trace"])

# TRL-GRPO on Modal H100 (no per-run JSON; traces in master_results.json, match W&B final_peak/final_last10)
mr = json.loads((EXP / "master_results.json").read_text())["experiments"]
for i in (60, 61, 62):
    e = mr[i]
    add("trl_modal", e["model"], f"master_results.json:experiments[{i}].reward_trace (platform={e.get('platform')})", e["reward_trace"])

# Tier-A Tinker Llama-3.3-70B-Inst, 4 seeds
ll = json.loads((TR / "llama33_70b_seeds.json").read_text())
for r in ll["results"]:
    if r.get("reward_trace"):
        add("llama33_70b_seeds", f"seed {r.get('seed')}", f"llama33_70b_seeds.json (run_id={r.get('run_id')})", r["reward_trace"])

# Two-seed matched-budget follow-up (canonical files, not .orig1)
for g in (2, 16):
    for s in (123, 456):
        d = json.loads((TR / f"er2b_g{g}_s{s}.json").read_text())
        zvf = [x.get("zvf") for x in d.get("step_log", []) if x.get("zvf") is not None]
        extra = f"status={d['status']}; G={d['group_size']}; batch={d['batch']}; steps={len(d['reward_trace'])}; zvf_max={max(zvf):.3f}; zvf_final={zvf[-1]:.3f}; zvf_last10={sum(zvf[-10:]) / len(zvf[-10:]):.3f}"
        add("er2b", f"G={g} s{s}", f"er2b_g{g}_s{s}.json", d["reward_trace"], extra)

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "table6_recomputed.tsv", "w") as fh:
    fh.write("block\tlabel\tsource\tn_steps\tpeak\tlast10\textra\n")
    for r in rows:
        fh.write("\t".join(map(str, r)) + "\n")
for r in rows:
    print(*r, sep=" | ")
