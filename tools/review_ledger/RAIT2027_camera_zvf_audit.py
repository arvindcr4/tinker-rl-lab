"""Recompute the RAIT2027 '368-run audit' summaries from the W&B export.

Input: a W&B runs dump (list of {project, config, summary, tags, state, ...});
default path is the 2026-10-18 pull. Only project == 'zvf-audit' is used.
Unit of analysis: one run (its summary mean_zvf / final_reward). Runs are
descriptive cells (mostly single seed per config); nothing here is a test.
"""
import json, sys, collections, statistics, pathlib

src = sys.argv[1] if len(sys.argv) > 1 else \
    "/private/tmp/claude-501/-Users-arvind/a00aabd7-5f45-4968-9d97-3fdddd937b51/scratchpad/tinker/wandb_runs.json"
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                   "platform_hybrid/experiments/results/review_2026-10-18/RAIT2027_camera")
out.mkdir(parents=True, exist_ok=True)

runs = [r for r in json.load(open(src)) if r.get("project") == "zvf-audit"]
rows = []
for r in runs:
    c, s = r.get("config", {}), r.get("summary", {})
    if "mean_zvf" not in s:
        continue
    rows.append(dict(model=c.get("model"), G=c.get("group"), steps=c.get("steps"),
                     seed=c.get("seed"), state=r.get("state"),
                     backfill="backfill" in (r.get("tags") or []),
                     zvf=s["mean_zvf"], final_reward=s.get("final_reward"),
                     mean_reward=s.get("reward")))

res = {"n_runs_project": len(runs), "n_with_mean_zvf": len(rows),
       "n_models": len({x["model"] for x in rows}),
       "models": sorted({str(x["model"]) for x in rows}),
       "steps_distribution": dict(collections.Counter(x["steps"] for x in rows)),
       "n_backfill_tagged": sum(x["backfill"] for x in rows)}

# reward-binned ZVF (bin on final_reward)
bins = [(0, .05), (.05, .2), (.2, .35), (.35, .5), (.5, .65), (.65, .8), (.8, .95), (.95, 1.01)]
rb = []
for lo, hi in bins:
    z = [x["zvf"] for x in rows if x["final_reward"] is not None and lo <= x["final_reward"] < hi]
    rb.append(dict(lo=lo, hi=hi, n=len(z), mean_zvf=round(statistics.mean(z), 4) if z else None))
res["zvf_by_final_reward_bin"] = rb

# per model x G cell means
cells = collections.defaultdict(list)
for x in rows:
    cells[(x["model"], x["G"])].append(x["zvf"])
res["model_G_cells"] = [dict(model=m, G=g, n=len(v), mean_zvf=round(statistics.mean(v), 4))
                        for (m, g), v in sorted(cells.items(), key=lambda k: (str(k[0][0]), k[0][1] or 0))]
json.dump(res, open(out / "zvf_audit_recompute.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "model_G_cells"}, indent=1))
for c in res["model_G_cells"]:
    print(c)
