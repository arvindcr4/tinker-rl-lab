"""E6: aggregate raw/<split>/results.jsonl into result.json fields. Errors/timeouts = failures (score 0).
judge_pending records (no judge credit) are reported separately: bounds + score over judged."""
import json, math, sys, glob, collections
from pathlib import Path

def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]

raw = Path(sys.argv[1])
recs = {}
for f in sorted(raw.glob("*/results.jsonl")):
    for l in f.read_text().splitlines():
        if l.strip():
            r = json.loads(l); r["split"] = f.parent.name; recs[r["task_id"]] = r
R = list(recs.values())
n = len(R)
pend = [r for r in R if r.get("judge_pending")]
graded = [r for r in R if not r.get("judge_pending")]
k = sum(1 for r in graded if (r["score"] or 0) >= 1.0)
errs = sum(1 for r in R if r.get("error"))
tmo = sum(1 for r in R if r.get("error") and "timeout" in r["error"].lower())
judge_cost = sum((r.get("usage") or {}).get("judge_cost", 0) for r in R)
ap = sum((r.get("usage") or {}).get("actor_prompt", 0) for r in R)
ac = sum((r.get("usage") or {}).get("actor_completion", 0) for r in R)
by = collections.defaultdict(lambda: [0, 0, 0])
for r in R:
    b = by[r["split"]]; b[0] += 1
    if r.get("judge_pending"): b[2] += 1
    elif (r["score"] or 0) >= 1: b[1] += 1
out = {
    "n_attempted": n, "n_graded": len(graded), "n_judge_pending": len(pend), "n_success": k,
    "n_errors": errs, "n_timeouts": tmo,
    "score_over_graded": round(k / len(graded), 4) if graded else None, "ci95_wilson_over_graded": wilson(k, len(graded)),
    "score_lower_bound_pending_as_fail": round(k / n, 4) if n else None, "ci95_wilson_lower_bound": wilson(k, n),
    "score_upper_bound_pending_as_pass": round((k + len(pend)) / n, 4) if n else None,
    "per_split": {s: {"n": v[0], "success": v[1], "judge_pending": v[2]} for s, v in sorted(by.items())},
    "judge_cost_usd": round(judge_cost, 4), "actor_prompt_tokens": ap, "actor_completion_tokens": ac,
}
print(json.dumps(out, indent=1))
