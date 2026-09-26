"""Recompute E5 result.json fields from raw/ (native Archipelago grades.json per task + shim ledger)."""
import json
import math
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "raw"
ITEMS = sys.argv[1:]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


per, status, errors = {}, {}, 0
for t in ITEMS:
    out = RAW / "archipelago_output" / t
    traj = out / "trajectory.json"
    status[t] = json.loads(traj.read_text()).get("status") if traj.exists() else None
    g = out / "grades.json"
    score = json.loads(g.read_text()).get("scoring_results", {}).get("final_score") if g.exists() else None
    if score is None:
        errors += 1
        score = 0.0
    per[t] = float(score)
led = [json.loads(l) for l in (RAW / "shim_ledger.jsonl").read_text().splitlines()]
num = sum(per.values())
print(json.dumps({"per_item": per, "agent_status": status, "numerator": round(num, 4),
                  "denominator": len(ITEMS), "value": round(num / len(ITEMS), 4), "n_errors": errors,
                  "wilson95": wilson(num, len(ITEMS)),
                  "tinker_tokens_total_incl_superseded": sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in led),
                  "n_cap_refusals": sum(1 for r in led if r.get("refused_cap"))}, indent=2))
