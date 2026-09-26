"""Recompute E4 result.json fields from raw/ (verifier reward.json per trial + shim ledger)."""
import json
import math
import sys
from pathlib import Path

# usage: compute_result.py [--raw DIR --prefix e4-vllm_trained] task_ids...
args = sys.argv[1:]
RAW, PREFIX = Path(__file__).resolve().parents[1] / "raw", "e4-small"
if args and args[0] == "--raw":
    RAW, PREFIX, args = Path(args[1]), args[3], args[4:]
ITEMS = args  # task ids in seeded order


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


per, errors = {}, 0
for t in ITEMS:
    trials = sorted((RAW / "jobs" / f"{PREFIX}-{t}").glob(f"{t}__*"))
    rj = trials[0] / "verifier" / "reward.json" if trials else None
    if rj and rj.exists():
        per[t] = float(json.loads(rj.read_text())["reward"])
    else:
        per[t] = 0.0
        errors += 1
led = [json.loads(l) for l in (RAW / "shim_ledger.jsonl").read_text().splitlines()]
num = sum(per.values())
out = {"per_item": per, "numerator": round(num, 4), "denominator": len(ITEMS),
       "value": round(num / len(ITEMS), 4), "n_errors": errors,
       "wilson95": wilson(num, len(ITEMS)),
       "tinker_tokens": sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in led),
       "n_requests": sum(1 for r in led if not r.get("refused_cap")),
       "n_cap_refusals": sum(1 for r in led if r.get("refused_cap"))}
print(json.dumps(out, indent=2))
