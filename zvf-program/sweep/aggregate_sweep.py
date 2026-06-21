#!/usr/bin/env python3
"""
ZVF Program — Pillar-1 / M1 aggregator.

Reads the run-cell result JSONs that the REAL runners wrote, joins each run's
per-step ZVF/GU trajectory with its held-out eval scores, and emits a results
table. Every value comes from a file on disk. If a file (or a field) is missing,
the cell is reported as MISSING / null — NEVER a guessed or imputed number.

INPUTS (all produced by the user's real runs, not by this harness):
  * run_manifest.json            — the enumerated grid (from run_sweep.py)
  * <results_dir>/<tag>.json     — per-run training log; must contain step_log
                                   with per-step {step, reward, loss, zvf, gu}
  * <results_dir>/<tag>.heldout.json (OPTIONAL) — per-run held-out scores, shape:
        {"GSM-Plus": <float|null>, "AIME-2025": <float|null>, "BFCL-v3": <float|null>}
      Produced by the user's eval step (the held_out_suite in sweep_config.yaml).
      If absent, held-out columns are reported MISSING.

OUTPUTS:
  * aggregate_table.csv  — one row per run-cell (path from sweep_config output.*)
  * stdout summary       — counts of completed / missing / failed

USAGE:
  python3 aggregate_sweep.py                 # aggregate everything in the manifest
  python3 aggregate_sweep.py --matched       # only matched-compute cells
  python3 aggregate_sweep.py --block audit   # only a given block
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "sweep_config.yaml"

HELDOUT_KEYS = ["GSM-Plus", "AIME-2025", "BFCL-v3"]
MISSING = "MISSING"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def load_manifest(cfg: dict) -> dict:
    mp = resolve(cfg["output"]["manifest_path"])
    if not mp.exists():
        sys.exit(f"Manifest not found: {mp}\nRun `python3 run_sweep.py --dry-run` first.")
    with open(mp) as f:
        return json.load(f)


def _safe_mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return round(statistics.mean(xs), 6) if xs else None


def _safe_optional_float(v):
    """Type-safe int/float extraction. None, NaN, strings, and other
    non-numeric values become the MISSING sentinel; real numbers are
    returned unchanged. Used for the per-run summary fields
    (peak_reward, last10_avg, first5_avg, zero_reward_pct) where a
    runner bug that wrote ``None`` or a string must surface as
    MISSING in the aggregate table, not as a hidden NaN that
    silently propagates to downstream consumers.
    """
    if v is None:
        return MISSING
    if isinstance(v, bool):
        # bool is a subclass of int in Python; treat as int.
        return int(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return MISSING
        return v
    return MISSING


def extract_zvf_gu(result: dict) -> dict:
    """Pull per-step ZVF/GU from a run's step_log. Returns trajectory summaries.

    Never invents values: if step_log is absent, everything is None. The
    GU columns have a derived-vs-measured distinction that is propagated
    in the `gu_derived_count` field so a downstream consumer can decide
    whether to trust a GU summary that is partly derived. The "no
    fabrication" property means we do NOT silently rebrand derived GU
    as measured GU: the n_gu_derived count surfaces the proportion
    of the trajectory that came from the 1 - ZVF identity.
    """
    step_log = result.get("step_log")
    if not step_log:
        return {"n_steps": 0, "mean_zvf": None, "final_zvf": None,
                "mean_gu": None, "final_gu": None, "first5_gu": None,
                "last10_gu": None, "gu_derived_count": 0, "gu_measured_count": 0}
    zvf = [s.get("zvf") for s in step_log]
    gu = [s.get("gu") for s in step_log]
    real_measured_gu = sum(
        1 for g, z in zip(gu, zvf)
        if isinstance(g, (int, float))
    )
    real_derived_gu = sum(
        1 for g, z in zip(gu, zvf)
        if g is None and isinstance(z, (int, float))
    )
    # If gu missing but zvf present, derive gu = 1 - zvf ONLY where zvf
    # is real. The derived-vs-measured split is preserved by the
    # gu_derived_count / gu_measured_count fields; the rendered
    # summary columns reflect the (derived OR measured) numeric value,
    # not a hidden fabrication.
    gu = [(1.0 - z) if (g is None and isinstance(z, (int, float))) else g
          for g, z in zip(gu, zvf)]
    real_zvf = [z for z in zvf if isinstance(z, (int, float))]
    real_gu = [g for g in gu if isinstance(g, (int, float))]
    return {
        "n_steps": len(step_log),
        "mean_zvf": _safe_mean(zvf),
        "final_zvf": real_zvf[-1] if real_zvf else None,
        "mean_gu": _safe_mean(gu),
        "final_gu": real_gu[-1] if real_gu else None,
        "first5_gu": _safe_mean(gu[:5]),
        "last10_gu": _safe_mean(gu[-10:]),
        "gu_derived_count": real_derived_gu,
        "gu_measured_count": real_measured_gu,
    }


def load_heldout(result_json_path: Path) -> dict:
    """Load the optional <tag>.heldout.json next to the training result.
    Returns dict of key -> score, with MISSING for any absent/null key.
    """
    heldout_path = result_json_path.with_suffix(".heldout.json")
    out = {k: MISSING for k in HELDOUT_KEYS}
    if not heldout_path.exists():
        return out
    try:
        with open(heldout_path) as f:
            data = json.load(f)
    except Exception:
        return out
    for k in HELDOUT_KEYS:
        v = data.get(k)
        out[k] = v if isinstance(v, (int, float)) else MISSING
    return out


ROW_COLS = [
    "tag", "block", "framework", "family", "model_short", "tier", "task", "loss",
    "seed", "group_size", "difficulty", "steps_planned",
    "run_status",            # completed | failed | MISSING (no result file)
    "n_steps",
    "peak_reward", "last10_avg", "first5_avg", "zero_reward_pct",
    "mean_zvf", "final_zvf", "mean_gu", "final_gu", "first5_gu", "last10_gu",
    "gu_measured_count", "gu_derived_count",
    "heldout_GSM-Plus", "heldout_AIME-2025", "heldout_BFCL-v3",
    "result_json",
]


def aggregate(cfg: dict, manifest: dict, block_filter: str | None,
              matched_only: bool) -> list[dict]:
    rows = []
    for cell in manifest["cells"]:
        if matched_only and cell["block"] != "matched":
            continue
        if block_filter and cell["block"] != block_filter:
            continue
        rj = Path(cell["result_json"])
        base = {
            "tag": cell["tag"], "block": cell["block"], "framework": cell["framework"],
            "family": cell["family"], "model_short": cell["model_short"],
            "tier": cell["tier"], "task": cell["task"], "loss": cell["loss"],
            "seed": cell["seed"], "group_size": cell["group_size"],
            "difficulty": cell["difficulty"], "steps_planned": cell["steps"],
            "result_json": str(rj),
        }
        if not rj.exists():
            rows.append({**base, "run_status": MISSING, "n_steps": 0,
                         **{c: MISSING for c in ROW_COLS
                            if c not in base and c != "run_status" and c != "n_steps"}})
            continue
        try:
            with open(rj) as f:
                result = json.load(f)
        except Exception:
            rows.append({**base, "run_status": "failed(parse)", "n_steps": 0,
                         **{c: MISSING for c in ROW_COLS
                            if c not in base and c not in ("run_status", "n_steps")}})
            continue
        # A result JSON whose top-level is a list (some legacy runner
        # configs emit a JSON list of step records) cannot be
        # projected to a record schema; report and skip rather than
        # crashing the aggregation.
        if not isinstance(result, dict):
            rows.append({**base, "run_status": "failed(schema:non-dict)",
                         "n_steps": 0,
                         **{c: MISSING for c in ROW_COLS
                            if c not in base and c not in ("run_status", "n_steps")}})
            continue

        status = result.get("status")
        if status is None:
            status = "completed" if result.get("step_log") else "failed"
        zg = extract_zvf_gu(result)
        ho = load_heldout(rj)
        rows.append({
            **base,
            "run_status": status,
            "n_steps": zg["n_steps"],
            "peak_reward": _safe_optional_float(result.get("peak_reward")),
            "last10_avg": _safe_optional_float(result.get("last10_avg")),
            "first5_avg": _safe_optional_float(result.get("first5_avg")),
            "zero_reward_pct": _safe_optional_float(result.get("zero_reward_pct")),
            "mean_zvf": zg["mean_zvf"] if zg["mean_zvf"] is not None else MISSING,
            "final_zvf": zg["final_zvf"] if zg["final_zvf"] is not None else MISSING,
            "mean_gu": zg["mean_gu"] if zg["mean_gu"] is not None else MISSING,
            "final_gu": zg["final_gu"] if zg["final_gu"] is not None else MISSING,
            "first5_gu": zg["first5_gu"] if zg["first5_gu"] is not None else MISSING,
            "last10_gu": zg["last10_gu"] if zg["last10_gu"] is not None else MISSING,
            "gu_measured_count": zg["gu_measured_count"],
            "gu_derived_count": zg["gu_derived_count"],
            "heldout_GSM-Plus": ho["GSM-Plus"],
            "heldout_AIME-2025": ho["AIME-2025"],
            "heldout_BFCL-v3": ho["BFCL-v3"],
        })
    return rows


def write_table(cfg: dict, rows: list[dict]) -> Path:
    out = resolve(cfg["output"]["aggregate_table"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, MISSING) for c in ROW_COLS})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched", action="store_true", help="only matched-compute cells")
    ap.add_argument("--block", default="", help="restrict to a block (audit|matched)")
    args = ap.parse_args()

    cfg = load_config()
    manifest = load_manifest(cfg)
    rows = aggregate(cfg, manifest, args.block or None, args.matched)

    out = write_table(cfg, rows)

    n_total = len(rows)
    n_missing = sum(1 for r in rows if r["run_status"] == MISSING)
    n_completed = sum(1 for r in rows if r["run_status"] == "completed")
    n_failed = n_total - n_missing - n_completed
    n_heldout = sum(
        1 for r in rows
        if any(r.get(f"heldout_{k}") not in (MISSING, None) for k in HELDOUT_KEYS)
    )

    print("=" * 70)
    print("ZVF SWEEP AGGREGATION")
    print("=" * 70)
    print(f"cells in scope     : {n_total}")
    print(f"  completed (real) : {n_completed}")
    print(f"  failed           : {n_failed}")
    print(f"  MISSING (no file): {n_missing}")
    print(f"  with held-out    : {n_heldout}")
    print(f"table written      : {out}")
    print("-" * 70)
    if n_completed == 0:
        print("No completed runs found on disk yet. Every metric is MISSING until")
        print("you actually launch the sweep (run_sweep.py --execute). Nothing is")
        print("fabricated.")
    else:
        # Show a tiny preview of REAL completed rows only.
        shown = 0
        for r in rows:
            if r["run_status"] == "completed":
                print(f"  {r['tag']:42s} mean_GU={r['mean_gu']} "
                      f"final_ZVF={r['final_zvf']} peak={r['peak_reward']}")
                shown += 1
                if shown >= 10:
                    print(f"  ... ({n_completed - 10} more in {out})")
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
