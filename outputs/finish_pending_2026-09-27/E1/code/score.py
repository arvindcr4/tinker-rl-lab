"""Aggregate native swebench reports: waves01-09 (2026-09-12, recorded) + wave10 + remaining (this lane)."""
import glob
import json
import math
from pathlib import Path

REPO = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab")
E1 = REPO / "outputs/finish_pending_2026-09-27/E1"
FIN = REPO / "outputs/PES_Phase2_Review_2026-09-12/finish/e1_completion"
ALL = [json.loads(l)["instance_id"] for l in open(
    REPO / "outputs/public_portfolio_2026-09-05/swe_multilingual_setup/prepared/native_dataset.jsonl")]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def load(paths):
    out = {}
    for p in paths:
        d = json.load(open(p))
        for k in ("submitted_ids", "completed_ids", "resolved_ids", "error_ids", "empty_patch_ids"):
            d.setdefault(k, [])
        for i in d["submitted_ids"]:
            out[i] = ("resolved" if i in d["resolved_ids"] else "unresolved" if i in d["completed_ids"]
                      else "empty_patch" if i in d["empty_patch_ids"] else "error")
    return out


def summary(outcomes, denom=None):
    n = len(outcomes)
    graded = sum(v in ("resolved", "unresolved") for v in outcomes.values())
    res = sum(v == "resolved" for v in outcomes.values())
    s = {"attempted": n, "graded": graded, "resolved": res,
         "errors_patch_apply_or_eval": sum(v == "error" for v in outcomes.values()),
         "empty_patch": sum(v == "empty_patch" for v in outcomes.values()),
         "resolved_over_graded": round(res / graded, 4) if graded else None,
         "resolved_over_graded_ci95": wilson(res, graded),
         "resolved_over_attempted": round(res / n, 4) if n else None,
         "resolved_over_attempted_ci95": wilson(res, n),
         "resolved_ids": sorted(k for k, v in outcomes.items() if v == "resolved")}
    if denom:
        s[f"resolved_over_{denom}"] = round(res / denom, 4)
        s[f"resolved_over_{denom}_ci95"] = wilson(res, denom)
    return s


prior = load(glob.glob(str(FIN / "**/reports/*.json"), recursive=True))
w10 = load(glob.glob(str(E1 / "wave10/run/reports/*.json")))
rem = load(glob.glob(str(E1 / "remaining/*/reports/*.json")))
assert not (set(prior) & set(w10)) and not (set(prior) & set(rem)) and not (set(w10) & set(rem))
lane = {**w10, **rem}
combined = {**prior, **lane}
missing = [i for i in ALL if i not in combined]

# context-overflow / pre-eval failures recorded by run_remaining
pre = {}
for sp in glob.glob(str(E1 / "remaining/*/*_status.json")):
    for iid, st in json.load(open(sp))["tasks"].items():
        if st.get("generation") not in ("GENERATED",):
            pre[iid] = st.get("generation") or st.get("error", "")[:160]

out = {"prior_waves01_09_2026_09_12": summary(prior),
       "this_lane_wave10": summary(w10),
       "this_lane_remaining": summary(rem),
       "this_lane_total": summary(lane),
       "combined_300": summary(combined, denom=300),
       "not_attempted": missing,
       "this_lane_non_generated": pre}
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "resolved_ids"} if isinstance(v, dict) and k != "this_lane_non_generated" else v
                  for k, v in out.items()}, indent=1))
json.dump(out, open(E1 / "score_breakdown.json", "w"), indent=2)
