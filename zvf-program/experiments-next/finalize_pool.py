#!/usr/bin/env python3
"""finalize_pool.py — deliberately mark a TRUNCATED pool as analyzable.

load_pool() refuses pools with status != "complete" so that a crashed run is
never silently analyzed. When a run was cut short for a known, recorded
reason (e.g. a billing block) and the banked prompts are sufficient, this
script performs the explicit, logged act of finalizing it:

  - status -> "complete"
  - truncated -> true, with planned vs actual counts and the stated reason
  - tag -> "<tag>_p<actual>" so downstream artifacts are visibly partial

The original file is preserved; a new pool_<newtag>.json is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import RESULTS_DIR, utc_now, write_result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--reason", required=True,
                    help="why the run stopped (recorded verbatim)")
    ap.add_argument("--min-prompts", type=int, default=100,
                    help="refuse to finalize below this count")
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text())
    if pool.get("status") == "complete":
        raise SystemExit("Pool is already complete; nothing to finalize.")
    banked = len(pool.get("prompts", []))
    if banked < args.min_prompts:
        raise SystemExit(f"Only {banked} prompts banked (< {args.min_prompts}); "
                         "refusing to finalize.")

    planned = pool.get("n_prompts", banked)
    new_tag = f"{pool['tag']}_p{banked}"
    pool.update({
        "status": "complete",
        "tag": new_tag,
        "truncated": True,
        "planned_prompts": planned,
        "n_prompts": banked,
        "truncation_reason": args.reason,
        "finalized_at": utc_now(),
    })
    out = RESULTS_DIR / f"pool_{new_tag}.json"
    write_result(out, pool)
    print(f"finalized {banked}/{planned} prompts -> {out}")


if __name__ == "__main__":
    main()
