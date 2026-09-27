#!/usr/bin/env python3
"""Build E13/result.json from raw/ (native episode JSONs + plan + attempts + proxy log).

Scoring = BALROG native: per-episode `progression` written by the native env wrapper; env score = mean over the
env's planned episodes (x100); overall = mean of the 6 env scores; SE per env = native population-sd/sqrt(n);
overall SE = sqrt(sum SE_env^2)/6 (BALROG leaderboard convention). Planned episodes without a native JSON
(errors / not run) are scored 0 in the denominator. The native collect_and_summarize_results output (JSONs
only) is stored alongside for cross-check.
"""
import json, math, os, sys
from collections import defaultdict
from pathlib import Path

E = Path(__file__).resolve().parents[1]
RAW = E / "raw"
sys.path.insert(0, os.environ["BALROG_SRC"])
from balrog.utils import collect_and_summarize_results  # noqa: E402

items = json.loads((RAW / "plan.json").read_text())["items"]


def jpath(it):
    return RAW / it["env"] / it["task"] / f"{it['task']}_run_{it['episode_idx']:02d}.json"


per_env, per_task, missing, steps, toks = defaultdict(list), defaultdict(list), [], defaultdict(list), defaultdict(int)
truncated = []  # NLE episodes that hit a step cap without done=True (labelled; scored at native progression-at-truncation)
for it in items:
    p = jpath(it)
    if p.exists():
        d = json.loads(p.read_text())
        v = float(d.get("progression", 0.0))
        if not d.get("done"):
            truncated.append({"id": it["id"], "steps": d.get("num_steps"), "progression": v})
        steps[it["env"]].append(d.get("num_steps", 0))
        toks["in"] += d.get("input_tokens", 0); toks["out"] += d.get("output_tokens", 0)
    else:
        v = 0.0
        missing.append(it["id"])
    per_env[it["env"]].append(v)
    per_task[(it["env"], it["task"])].append(v)


def stats(xs):
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n) if n > 1 else 0.0
    return 100 * m, 100 * sd / math.sqrt(n) if n > 1 else 0.0


envs = {}
for env, xs in per_env.items():
    m, se = stats(xs)
    envs[env] = {"progression_pct": round(m, 2), "se_pct": round(se, 2), "ci95_pct": [round(m - 1.96 * se, 2), round(m + 1.96 * se, 2)],
                 "n_planned": len(xs), "n_graded": len(steps[env]), "n_missing_scored_0": len(xs) - len(steps[env]),
                 "mean_steps_graded": round(sum(steps[env]) / len(steps[env]), 1) if steps[env] else None,
                 "tasks": {t: round(stats(v)[0], 1) for (e, t), v in per_task.items() if e == env}}
overall = sum(v["progression_pct"] for v in envs.values()) / len(envs)
overall_se = math.sqrt(sum(v["se_pct"] ** 2 for v in envs.values())) / len(envs)

# Percentile bootstrap over episodes (B=10000, fixed seed). Per env: resample that env's episodes with replacement.
# Overall: stratified bootstrap (resample within each env, then mean of env means), matching the overall estimator.
import random
B, rng = 10000, random.Random(20260927)
boot_env = {e: sorted(100 * sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(B)) for e, xs in per_env.items()}
rng = random.Random(20260927)
boot_all = sorted(sum(100 * sum(rng.choice(xs) for _ in xs) / len(xs) for xs in per_env.values()) / len(per_env)
                  for _ in range(B))
pct = lambda a, q: round(a[min(len(a) - 1, int(q * len(a)))], 2)
for e in envs:
    envs[e]["ci95_bootstrap_pct"] = [pct(boot_env[e], 0.025), pct(boot_env[e], 0.975)]
    envs[e]["n_truncated_step_cap"] = sum(1 for t in truncated if t["id"].startswith(e + "/"))

attempts = [json.loads(l) for l in open(RAW / "attempts.jsonl")] if (RAW / "attempts.jsonl").exists() else []
calls = [json.loads(l) for l in open(RAW / "proxy_calls.jsonl")] if (RAW / "proxy_calls.jsonl").exists() else []
native = collect_and_summarize_results(str(RAW))
(RAW / "native_summary.json").write_text(json.dumps(native, indent=1, default=str))

res = {
    "lane": "E13",
    "benchmark": "BALROG (balrog-ai/BALROG@b7afe79), native Evaluator.run_episode + NaiveAgent + native vllm client",
    "scope": "replacement (full native BALROG suite: 255 episodes, 58 task configs, 6 envs incl. NetHack); "
             "never pooled with the original-contract E13-native-20260912-01 13-episode receipt",
    "actor": "pavlov-public-portfolio-bf16 (Qwen3.6-35B-A3B + seed809 LoRA merged, bf16) via shared Modal vLLM endpoint",
    "protocol": {"temperature": 1.0, "max_tokens": 8192, "agent": "naive, max_text_history 16", "seeds": "native get_unique_seed (logged per episode)",
                 "enable_thinking": False, "step_caps": "native config.yaml (nle native 100000 + no_progress_timeout 150 BUT deviation cap 2000 agent steps via eval.max_steps_per_episode for the 2 NLE episodes run after 01:40 UTC, crafter 2000, minihack 100, textworld 80, babyai/babaisai native)"},
    "n_planned": len(items), "n_attempted": len({a["id"] for a in attempts}), "n_graded": len(items) - len(missing),
    "n_missing_scored_0": len(missing), "missing_ids": missing,
    "truncated_episodes": truncated,
    "truncation_note": "Episodes with no done=True hit the deviation step cap (NLE 2000 agent steps via native "
                       "eval.max_steps_per_episode); scored at native progression-at-truncation and labelled here. "
                       "Native env time limits set done=True and are not listed.",
    "metric": "BALROG native progression % (mean of per-env means)",
    "score": round(overall, 2), "se": round(overall_se, 2), "ci95": [round(overall - 1.96 * overall_se, 2), round(overall + 1.96 * overall_se, 2)],
    "ci95_bootstrap": [pct(boot_all, 0.025), pct(boot_all, 0.975)],
    "ci_method": "ci95 = normal approx with BALROG-convention SE (per-env population sd/sqrt(n); overall sqrt(sum SE^2)/6); "
                 "ci95_bootstrap = percentile bootstrap over episodes, B=10000, seed 20260927, stratified by env for overall. "
                 "Missing/errored episodes enter as 0 in both.",
    "per_env": envs,
    "tokens": {"episode_json_input": toks["in"], "episode_json_output": toks["out"], "proxy_calls": len(calls),
               "proxy_errors": sum(1 for c in calls if "error" in c),
               "proxy_prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
               "proxy_completion_tokens": sum(c.get("completion_tokens", 0) for c in calls)},
    "native_summary_ref": "raw/native_summary.json (JSON-only; excludes missing episodes)",
}
extra = E / "code" / "result_extra.json"
if extra.exists():
    res.update(json.loads(extra.read_text()))
(E / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("n_planned", "n_graded", "n_missing_scored_0", "score", "se")}),
      {k: v["progression_pct"] for k, v in envs.items()})
