#!/usr/bin/env python3
"""Build E13/result.json from raw/ (plan.json, per-episode native JSONs, shim_calls.jsonl, native_summary.json).
Missing/errored episodes count as progression 0. Primary metric = native BALROG aggregation (equal-weight mean
of per-env mean progression, x100) over the envs run. Wilson CI is over binary episode success (progression==1)."""
import json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

E = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]  # arm dir
ACTOR = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.6-35B-A3B base (no adapter), Tinker"
ROUTE = sys.argv[3] if len(sys.argv) > 3 else "local macOS arm64 (BALROG games) + Tinker sampling via local OpenAI-compatible shim"
R = E / "raw"
plan = json.loads((R / "plan.json").read_text())
per_env, rows = {}, []
for it in plan["items"]:
    jf = R / it["env"] / it["task"] / f"{it['task']}_run_{it['episode_idx']:02d}.json"
    if jf.exists():
        d = json.loads(jf.read_text())
        p, st, err = float(d.get("progression", 0.0)), d.get("num_steps"), None
    else:
        p, st, err = 0.0, None, "missing_or_error"
    rows.append({"id": it["id"], "progression": p, "steps": st, "error": err})
    per_env.setdefault(it["env"], []).append(p)

env_means = {k: 100 * sum(v) / len(v) for k, v in per_env.items()}
overall = sum(env_means.values()) / len(env_means)
succ = sum(1 for r in rows if r["progression"] >= 1.0)
n = len(rows)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


calls = [json.loads(l) for l in open(R / "shim_calls.jsonl")]
tok = sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls)
dts = [c["dt"] for c in calls if "dt" in c]
out_tok = sum(c["completion_tokens"] for c in calls)
(R / "per_episode.json").write_text(json.dumps(rows, indent=1))
res = {
    "lane": "E13", "original_benchmark": "OpenReward held-out games (externally blocked)",
    "benchmark_run": "BALROG (balrog-ai/BALROG@b7afe79) small subset: BabyAI, TextWorld, BabaIsAI, MiniHack, Crafter(capped); NetHack skipped",
    "scope": "substitute",
    "substitute_gap": "OpenReward held-out games are inaccessible; BALROG is the accepted public replacement, run here on 26/255 episodes across 5 of 6 envs (no NetHack), so it is not the full-suite BALROG score either.",
    "actor": ACTOR,
    "thinking": False, "temperature": 0, "max_tokens": 128,
    "n_selected": n, "n_scored": n, "n_errors": sum(1 for r in rows if r["error"]),
    "item_ids": [r["id"] for r in rows],
    "metric": "BALROG native progression %: mean over envs of per-env mean episode progression (x100), 5 envs equal-weighted",
    "value": round(overall, 2),
    "numerator": succ, "denominator": n,
    "wilson95": wilson(succ, n),
    "per_env_progression_pct": {k: round(v, 2) for k, v in env_means.items()},
    "per_env_n": {k: len(v) for k, v in per_env.items()},
    "episode_full_success": {"k": succ, "n": n, "note": "numerator/denominator/wilson95 refer to episodes with progression==1.0, not to `value`"},
    "grader": "native", "compute_route": ROUTE,
    "cost": {"tinker_tokens": tok, "colab_units": 0.0, "modal_usd": 0.0},
    "sampler_speed": {"calls": len(calls), "mean_latency_s": round(sum(dts) / len(dts), 3),
                      "mean_output_tokens": round(out_tok / len(calls), 2),
                      "note": "throughput from shim_calls.jsonl; calls ran concurrently from 5 env drivers"},
    "started_utc": (R / "started_utc.txt").read_text().strip(),
    "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "commands": [
        "SHIM_LOG=raw/shim_calls.jsonl SHIM_BUDGET=5800000 SHIM_MAX_TOKENS=128 PORT=8773 <tinker-python> code/tinker_shim.py",
        "BALROG_SRC=<balrog@b7afe79> E13_OUT=raw python code/run_e13.py --plan",
        "for e in babyai textworld babaisai minihack crafter: python code/run_e13.py --env $e  (cwd has native tw_games/)",
        "python code/run_e13.py --summarize ; python3 code/make_result.py"],
    "raw_dir": str(R.relative_to(E.parent.parent)) + "/" if E.name != "E13" else "E13/raw/",
    "caveats": [
        "New base-model arm; not comparable to or pooled with the original seed809-LoRA E13 plan (21-episode BabyAI admission set).",
        "Protocol deviations from BALROG native: temperature 0 (native 1.0), max_tokens 128 (native 8192), non-thinking template, 1 worker; fixed env seeds 20260926+episode_idx instead of native random seeds.",
        "Crafter capped at 150 steps (native 2000), so its progression is a lower bound vs native; NetHack not run.",
        "Subset: BabyAI 5 tasks x2, TextWorld 3x2, BabaIsAI 4/40 tasks x1, MiniHack 4/8 tasks x1 (seed 20260926); per-env n is 2-10, so env means are very noisy.",
        "Run on macOS arm64 (balrog-nle 0.9.0 mac wheel, scipy 1.13.1 downgrade for dyld compat), not the Linux x86 image; game dynamics expected identical but not verified bit-for-bit.",
        "Overall value averages 5 envs, not the native 6 (NetHack excluded).",
    ],
}
(E / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("value", "per_env_progression_pct", "numerator", "denominator", "cost", "sampler_speed")}, indent=1))
