#!/usr/bin/env python3
"""Propose review17-v2 mag_mean repairs only from an explicit mag_mean source.

The historical risk-index mean_zvf column is NOT interchangeable with mag_mean.
No original registry entry is overwritten; missing source semantics fail closed.
"""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from review17_contract import validate_proposals
from _paths import REPO_ROOT
from review17_contract import load_seed_metric, paired_result, source_hash

ROOT = REPO_ROOT
ENTRIES = ROOT / "platform_hybrid/registry/entries"
RES_OUT = ROOT / "platform_hybrid/experiments/results/p5p8/review17_v2"
SOURCE = ROOT / "platform_hybrid/experiments/results/zvf_iter130_risk_index.tsv"
SEED = 20260705
N_BOOT = 2000
TARGETS = ["delta_cppo", "delta_es", "delta_mcgrpo", "delta_ngrpo", "delta_scafgrpo"]


def load_per_seed():
    return load_seed_metric(SOURCE, "mag_mean")


def paired_boot_ci(grpo_vals, meth_vals, B, seed):
    result = paired_result(grpo_vals, meth_vals, B, seed)
    return result["delta"], result["ci_low"], result["ci_high"]


def patch_entry(entry_id, per_seed):
    rec = json.loads((ENTRIES / f"{entry_id}.json").read_text())
    method = rec["name"].lower()
    result = paired_result(per_seed.get("grpo", {}), per_seed.get(method, {}), N_BOOT, SEED)
    result.update(
        metric="mag_mean",
        panel="zvf130_5seed",
        base="grpo",
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=source_hash(SOURCE),
    )
    rows = rec.setdefault("measured", [])
    indices = [
        i
        for i, row in enumerate(rows)
        if (row.get("metric"), row.get("panel"), row.get("base", "grpo"))
        == ("mag_mean", "zvf130_5seed", "grpo")
    ]
    if len(indices) > 1:
        raise ValueError("Duplicate measurement identity")
    if indices:
        rows[indices[0]] = result
    else:
        rows.append(result)
    return rec


def main():
    data = load_per_seed()
    proposed = {eid: patch_entry(eid, data) for eid in TARGETS}
    validate_proposals(proposed, ROOT / "platform_hybrid/registry/schema.json")
    RES_OUT.mkdir(parents=True, exist_ok=True)
    (RES_OUT / "p6_iter130_proposed_entries.json").write_text(
        json.dumps(proposed, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
