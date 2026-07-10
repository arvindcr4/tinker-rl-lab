#!/usr/bin/env python3
"""P6 iter-130 patcher: recompute the 5 remaining stale `mag_mean` point
rows (delta_cppo, delta_es, delta_mcgrpo, delta_ngrpo, delta_scafgrpo) from
per-seed `mean_zvf` in experiments/results/zvf_iter130_risk_index.tsv.

For each method, computes:
  baseline_mean = mean(per-seed grpo mean_zvf)
  method_mean   = mean(per-seed method mean_zvf)
  delta         = method_mean - baseline_mean
  ci_low/high   = paired-seed bootstrap 95% CI (B=2000, seed=20260705)

Patches the registry entries in place (writes a new measured[] row with
ci_method=bootstrap_paired_5seed, replaces the point_no_perseed_sd row).
Writes experiments/results/p5p8/p6_iter130_patch_log.tsv with diff per entry.

Stdlib only. <= 300 lines.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
ENTRIES = ROOT / "registry" / "entries"
RES_OUT = ROOT / "experiments" / "results" / "p5p8"
SEED = 20260705
N_BOOT = 2000
TARGETS = ["delta_cppo", "delta_es", "delta_mcgrpo", "delta_ngrpo", "delta_scafgrpo"]


def load_per_seed():
    out = {}  # method -> [mean_zvf per seed]
    with (ROOT / "experiments" / "results" / "zvf_iter130_risk_index.tsv").open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            m = row["method"]
            try:
                v = float(row["mean_zvf"])
            except (ValueError, KeyError):
                continue
            out.setdefault(m, []).append(v)
    return out


def paired_boot_ci(grpo_vals, meth_vals, B, seed):
    n = min(len(grpo_vals), len(meth_vals))
    if n == 0:
        return 0.0, 0.0, 0.0
    g = grpo_vals[:n]
    m = meth_vals[:n]
    obs_delta = sum(m[i] - g[i] for i in range(n)) / n
    # LCG for reproducibility (matches iter-128 / iter-111)
    state = [seed & 0xFFFFFFFF]
    def lcg():
        state[0] = (state[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return state[0]
    deltas = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            idx = lcg() % n
            s += m[idx] - g[idx]
        deltas.append(s / n)
    deltas.sort()
    lo = deltas[int(0.025 * B)]
    hi = deltas[int(0.975 * B) - 1]
    return obs_delta, lo, hi


def patch_entry(entry_id, per_seed):
    path = ENTRIES / f"{entry_id}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    method = rec.get("name", "").lower()  # e.g. "CPPO" -> "cppo"
    grpo = per_seed.get("grpo", [])
    meth = per_seed.get(method, [])
    if not grpo or not meth:
        return {"entry_id": entry_id, "skipped": True, "reason": "missing per-seed"}
    delta, lo, hi = paired_boot_ci(grpo, meth, N_BOOT, SEED)
    n_seeds = min(len(grpo), len(meth))
    # Find & replace the stale mag_mean point row
    rows = rec.get("measured") or []
    target_idx = None
    for i, r in enumerate(rows):
        if (r.get("metric") == "mag_mean" and r.get("panel") == "zvf130_5seed"
                and ((r.get("ci_method") or {}).get("method") == "point_no_perseed_sd")):
            target_idx = i
            break
    if target_idx is None:
        # insert if missing
        rows.append({})
        target_idx = len(rows) - 1
    old = dict(rows[target_idx])
    rows[target_idx] = {
        "metric": "mag_mean",
        "panel": "zvf130_5seed",
        "base": "grpo",
        "delta": round(delta, 6),
        "ci_low": round(lo, 6),
        "ci_high": round(hi, 6),
        "n": n_seeds,
        "significant": (lo > 0) or (hi < 0),
        "ci_method": {
            "method": "bootstrap_paired_5seed",
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci_level": 0.95,
            "source": "scripts/p5p8/p6_iter130_patch_stale_mag.py",
        },
        "source": "experiments/results/zvf_iter130_risk_index.tsv",
        "note": f"recomputed iter-130 paired-seed bootstrap on mean_zvf; "
                f"method={method}, n_seeds={n_seeds}, B={N_BOOT}, seed={SEED}",
        "synth_from_agg": False,
    }
    rec["measured"] = rows
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    return {
        "entry_id": entry_id,
        "method": method,
        "n_seeds": n_seeds,
        "old_delta": old.get("delta"),
        "old_ci": (old.get("ci_low"), old.get("ci_high")),
        "new_delta": round(delta, 6),
        "new_ci": (round(lo, 6), round(hi, 6)),
        "significant": (lo > 0) or (hi < 0),
        "skipped": False,
    }


def main():
    per_seed = load_per_seed()
    log = []
    for eid in TARGETS:
        r = patch_entry(eid, per_seed)
        if r is not None:
            log.append(r)
    # write log
    cols = ["entry_id", "method", "n_seeds", "old_delta", "old_ci",
            "new_delta", "new_ci", "significant", "skipped"]
    with (RES_OUT / "p6_iter130_patch_log.tsv").open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in log:
            line = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, tuple):
                    line.append(str(v))
                else:
                    line.append("" if v is None else str(v))
            f.write("\t".join(line) + "\n")
    # summary json
    summary = {
        "n_targeted": len(TARGETS),
        "n_patched": sum(1 for r in log if not r.get("skipped")),
        "n_skipped": sum(1 for r in log if r.get("skipped")),
        "patch_results": log,
    }
    (RES_OUT / "p6_iter130_patch_log.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"iter-130 patch: targeted={len(TARGETS)}, patched={summary['n_patched']}, "
          f"skipped={summary['n_skipped']}")


if __name__ == "__main__":
    main()
