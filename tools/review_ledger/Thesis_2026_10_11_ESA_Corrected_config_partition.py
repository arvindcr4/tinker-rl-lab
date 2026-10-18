"""Recompute the 44-field partition of platform_hybrid/experiments/framework_config_dumps/*.yaml (thesis §5.2, finding 984).

Run from repo root: python3 -I tools/review_ledger/Thesis_2026_10_11_ESA_Corrected_config_partition.py
"""

import json
import os

import yaml

D = "platform_hybrid/experiments/framework_config_dumps/"
FW = ["tinker", "trl", "verl", "openrlhf"]


def flat(x, p=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
        return out
    return {p: x}


cfg = {f: flat(yaml.safe_load(open(f"{D}{f}_qwen3_8b_gsm8k.yaml"))) for f in FW}
keys = sorted(set().union(*cfg.values()))
MISSING = object()
part: dict[str, list[str]] = {"identical_all_four": [], "null_in_tinker": [], "other": []}
detail = {}
for k in keys:
    vals = [cfg[f].get(k, MISSING) for f in FW]
    if (
        all(v is not MISSING for v in vals)
        and len({json.dumps(v, sort_keys=True, default=str) for v in vals}) == 1
    ):
        part["identical_all_four"].append(k)
    elif cfg["tinker"].get(k, MISSING) is None:
        part["null_in_tinker"].append(k)
        detail[k] = {f: (None if v is MISSING else v) for f, v in zip(FW, vals)}
    else:
        part["other"].append(k)
        detail[k] = {f: ("<absent>" if v is MISSING else v) for f, v in zip(FW, vals)}
out = {
    "n_leaf_fields": len(keys),
    "counts": {k: len(v) for k, v in part.items()},
    "partition": part,
    "detail": detail,
}
os.makedirs(
    "platform_hybrid/experiments/results/review_2026-10-18/Thesis_2026_10_11_ESA_Corrected",
    exist_ok=True,
)
json.dump(
    out,
    open(
        "platform_hybrid/experiments/results/review_2026-10-18/Thesis_2026_10_11_ESA_Corrected/config_partition.json",
        "w",
    ),
    indent=1,
    default=str,
)
print(json.dumps(out, indent=1, default=str))
