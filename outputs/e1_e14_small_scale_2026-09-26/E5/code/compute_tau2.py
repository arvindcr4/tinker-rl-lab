"""Recompute a tau2 arm's pass^1 from raw results.json files (+ shim ledger token totals).
usage: python compute_tau2.py <raw_dir> <arm_prefix e5_base|e5_vllm_trained|e5_vllm_base> [ledger.jsonl ...]"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

raw, prefix, ledgers = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
SEL = json.loads((Path(__file__).resolve().parents[1] / "raw" / "selection.json").read_text())
ITEMS = [("airline", t) for t in SEL["airline"][:5]] + [("retail", t) for t in SEL["retail"][:5]]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


per, term, errors = {}, {}, 0
for dom, tid in ITEMS:
    sims = {s["task_id"]: s for s in json.loads((raw / f"{prefix}_{dom}" / "results.json").read_text())["simulations"]}
    s = sims.get(tid)
    reward = (s.get("reward_info") or {}).get("reward") if s else None
    term[f"{dom}/{tid}"] = s.get("termination_reason") if s else "missing"
    if reward is None:
        errors += 1
        reward = 0.0
    per[f"{dom}/{tid}"] = 1 if float(reward) >= 1.0 - 1e-6 else 0  # pass^1 at one trial: reward==1
k = sum(per.values())
tok = Counter()
for lp in ledgers:
    for line in Path(lp).read_text().splitlines():
        r = json.loads(line)
        tok[r.get("tag") or "untagged"] += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
print(json.dumps({"per_item": per, "termination": term, "numerator": k, "denominator": len(ITEMS),
                  "value": round(k / len(ITEMS), 4), "n_errors": errors, "wilson95": wilson(k, len(ITEMS)),
                  "tokens_by_tag": dict(tok)}, indent=2))
