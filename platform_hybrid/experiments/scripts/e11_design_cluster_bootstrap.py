#!/usr/bin/env python3
"""E11 VerilogEval design-cluster bootstrap for pass@1 = 129/312.

The 312 E11 prompts are 156 designs, each evaluated in two framings
(code-complete-iccad2023 and spec-to-rtl), so they are not 312 independent
observations. This script pairs the two per-problem verdicts by design ID
from the full receipt, then resamples the 156 design pairs with replacement
and reports the percentile interval of (summed passes) / 312.

Method (matches thesis Appendix D, "E11 design-cluster uncertainty"):
  - B = 200,000 replicates, RNG = Python ``random.Random(20261002)``
  - each replicate draws 156 pairs via ``rng.choices(pairs, k=156)``
    (i.e. index = floor(rng.random() * 156))
  - quantiles: sort, linear interpolation at index (B - 1) * p, p = 0.025 / 0.975

The 97.5% quantile sits on a knife-edge between 151/312 and 152/312
(exact P(S <= 151) ~ 0.97495), so the draw primitive matters: ``choices``
gives the reported [0.3429487, 0.4871795]; ``randrange`` with the same seed
gives [0.3429487, 0.4839744].

Input: the full receipt (gitignored: it carries provider run/checkpoint
identifiers) when present; otherwise the committed paired verdicts in
``e11_design_cluster_bootstrap.json`` next to this script, which records the
receipt's SHA-256.

Standard library only. Run from the repository root:

    python platform_hybrid/experiments/scripts/e11_design_cluster_bootstrap.py
    python platform_hybrid/experiments/scripts/e11_design_cluster_bootstrap.py \\
        --out platform_hybrid/experiments/scripts/e11_design_cluster_bootstrap.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

RECEIPT = Path("outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json")
PAIRS_JSON = Path(__file__).with_suffix(".json")
FRAMINGS = ("code-complete-iccad2023", "spec-to-rtl")
SEED = 20261002
B = 200_000


def _quantile(sorted_vals: list, p: float) -> float:
    pos = (len(sorted_vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def load_pairs(receipt: Path) -> dict:
    data = json.loads(receipt.read_text())
    logs = [data["verifier"]["receipts"][f]["log_receipts"] for f in FRAMINGS]
    if set(logs[0]) != set(logs[1]):
        raise SystemExit("framings do not share the same design IDs")
    return {
        design: tuple(int(fr[design]["verdict"] == "PASS") for fr in logs)
        for design in sorted(logs[0])
    }


def bootstrap(pairs: dict, b: int = B, seed: int = SEED) -> tuple:
    sums = [a + c for a, c in pairs.values()]
    n = len(sums)
    denom = 2 * n
    rng = random.Random(seed)
    stats = sorted(sum(rng.choices(sums, k=n)) / denom for _ in range(b))
    return _quantile(stats, 0.025), _quantile(stats, 0.975)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receipt", type=Path, default=RECEIPT)
    ap.add_argument("--replicates", type=int, default=B)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=None, help="Write method, counts and paired verdicts as JSON")
    args = ap.parse_args()

    if args.receipt.exists():
        pairs = load_pairs(args.receipt)
        source = args.receipt
        source_sha = hashlib.sha256(args.receipt.read_bytes()).hexdigest()
    elif PAIRS_JSON.exists():
        saved = json.loads(PAIRS_JSON.read_text())
        pairs = {k: tuple(v) for k, v in saved["paired_verdicts"].items()}
        source, source_sha = Path(saved["source"]), saved["source_sha256"]
        print(f"receipt not found; using paired verdicts from {PAIRS_JSON}")
    else:
        print(f"error: neither {args.receipt} nor {PAIRS_JSON} found", file=sys.stderr)
        return 2
    counts = {
        "both_pass": sum(1 for v in pairs.values() if v == (1, 1)),
        "both_fail": sum(1 for v in pairs.values() if v == (0, 0)),
        "code_complete_only": sum(1 for v in pairs.values() if v == (1, 0)),
        "spec_to_rtl_only": sum(1 for v in pairs.values() if v == (0, 1)),
    }
    passes = sum(a + c for a, c in pairs.values())
    lo, hi = bootstrap(pairs, args.replicates, args.seed)

    print(f"designs={len(pairs)} prompts={2 * len(pairs)} passes={passes} pass@1={passes / (2 * len(pairs)):.7f}")
    print(f"pairs: {counts}")
    print(f"B={args.replicates} seed={args.seed} 95% cluster-bootstrap CI = [{lo:.7f}, {hi:.7f}]")

    if args.out:
        payload = {
            "method": "Resample 156 design pairs with replacement (random.Random(seed).choices(k=156)), "
            "statistic = summed passes / 312, percentile CI with linear interpolation at (B-1)p.",
            "seed": args.seed,
            "replicates": args.replicates,
            "source": str(source),
            "source_sha256": source_sha,
            "framings": list(FRAMINGS),
            "pair_counts": counts,
            "passes": passes,
            "denominator": 2 * len(pairs),
            "ci95": [lo, hi],
            "paired_verdicts": {k: list(v) for k, v in pairs.items()},
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
