"""Score the E5 tau3 run with the native metric, apply the lane rule (errors = failures over all 97),
and copy redacted raw outputs into the lane dir.  Run with the tau2 venv python.
usage: score_and_redact.py <native_save_dir> <lane_dir> <relay_spend_json>
"""
import json
import math
import re
import shutil
import sys
from pathlib import Path

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics

src, lane = Path(sys.argv[1]), Path(sys.argv[2])
relay = json.loads(Path(sys.argv[3]).read_text())
REPO = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab")
inv = json.loads((REPO / "outputs/PES_Phase2_Review_2026-09-12/finish/e5_runtime/successor27_v6_27/inventory.json").read_text())
prior_graded20 = set(inv["native10_exclusion_categories"]["graded20"])

res = Results.load(src / "results.json")
m = compute_metrics(res)
all_ids = sorted(t.id for t in res.tasks)
per = {}
for s in res.simulations:
    ri = s.reward_info
    per[s.task_id] = {"termination": str(s.termination_reason.value if hasattr(s.termination_reason, "value") else s.termination_reason),
                      "reward": None if ri is None else ri.reward,
                      "reward_basis": None if ri is None or ri.reward_basis is None else [str(b.value if hasattr(b, "value") else b) for b in ri.reward_basis]}


def succ(t):
    r = per.get(t, {}).get("reward")
    return r is not None and abs(r - 1.0) < 1e-6  # native is_successful


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def block(ids):
    k = sum(succ(t) for t in ids)
    return {"n": len(ids), "passes": k, "pass1": round(k / len(ids), 4) if ids else None, "wilson95": wilson(k, len(ids))}


infra = [t for t in all_ids if per.get(t, {}).get("termination") == "infrastructure_error" or t not in per]
out = {
    "benchmark": "tau3 = sierra-research/tau2-bench@a2c024725189473d2d7cea3a5cfdbcc67478e41f, domain banking_knowledge, alltools retrieval, 97 tasks",
    "scope": "replacement scope (E5 Tau3); fresh single run of all 97 tasks under one protocol; never pooled with original-contract numbers",
    "n_attempted": len(all_ids), "n_graded": sum(1 for t in all_ids if per.get(t, {}).get("reward") is not None),
    "n_infrastructure_error": len(infra), "infrastructure_error_tasks": infra,
    "metric": "pass^1 (native compute_metrics(...).pass_hat_ks[1]; success = reward 1.0)",
    "native_pass1_excluding_infra": m.pass_hat_ks.get(1),
    "score": block(all_ids)["pass1"],
    "score_rule": "lane rule: infrastructure errors/timeouts count as failures over all 97",
    "all97": block(all_ids),
    "remaining77_excluding_native10_graded20": block([t for t in all_ids if t not in prior_graded20]),
    "native10_graded20_rerun": block(sorted(prior_graded20)),
    "per_task": per,
}
lane.mkdir(parents=True, exist_ok=True)
(lane / "raw").mkdir(exist_ok=True)
KEYPAT = re.compile(r'("api_key"\s*:\s*")[^"]*(")')
SKPAT = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
secrets = [s for s in sys.argv[4:] if s]


def redact(text):
    text = KEYPAT.sub(r"\1<REDACTED>\2", text)
    text = SKPAT.sub("<REDACTED>", text)
    for s in secrets:
        text = text.replace(s, "<REDACTED>")
    return text


dst = lane / "raw" / src.name
if dst.exists():
    shutil.rmtree(dst)
for f in src.rglob("*"):
    if f.is_file():
        t = dst / f.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        try:
            t.write_text(redact(f.read_text()))
        except UnicodeDecodeError:
            shutil.copy2(f, t)
out["relay_openai_spend"] = relay
(lane / "raw" / "scores_native.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "per_task"}, indent=1, default=str))
