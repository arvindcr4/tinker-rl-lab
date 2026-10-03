"""
utils/verify_results.py
=======================

Compare a directory of experiment result JSONs / logs against the expected
headline numbers reported in the paper (`platform_hybrid/paper/expected_results.json`), within
documented tolerances.

Designed to be run by NeurIPS artifact reviewers after a reproduction run:

    python utils/verify_results.py \\
        --results-dir results/ \\
        --expected-results platform_hybrid/paper/expected_results.json \\
        --last10-tolerance 0.05 \\
        --peak-tolerance   0.10

The default tolerances (±5 pts on last-10, ±10 pts on peak) are justified in
`ARTIFACT.md §6` and `REPRODUCE.md §8`.

Result file formats accepted:

1. JSON (one object per file)::

    {
        "experiment": "gsm8k_qwen3_8b_s42",
        "model": "Qwen/Qwen3-8B",
        "seed": 42,
        "last10_avg": 0.344,
        "peak":       0.625
    }

2. A ``.log`` captured from ``platform_tinker/grpo_gsm8k_base.py`` (which
   routes through ``tinkerrl.grpo_cli``). One block per seed is printed::

    [grpo_cli] Seed 42 done.
      avg_last10    : 0.344
      peak_reward   : 0.625

   The experiment tag is the log's file stem (plus ``_s<seed>`` when one log
   holds several seeds).

3. The legacy final-report lines ``Last-10 avg accuracy: 34.4%`` /
   ``Peak accuracy: 62.5%``.

Exit codes:
    0  every matched experiment within tolerance
    1  an experiment outside tolerance, or no result file matched an expected key
    2  usage / IO error (missing results dir or expectations file, bad JSON)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default expected results — overridden by --expected-results file if present.
# Values from platform_hybrid/paper/main.tex Table 2 ("Atropos GSM8K, Tinker, GRPO" block).
DEFAULT_EXPECTED: Dict[str, Dict[str, float]] = {
    "gsm8k_qwen3_8b": {"last10": 0.344, "peak": 0.625},  # headline
    "gsm8k_qwen3_8b_base": {"last10": 0.844, "peak": 1.000},
    "gsm8k_qwen3_5_4b": {"last10": 0.850, "peak": 1.000},
    "gsm8k_qwen3_5_27b": {"last10": 0.437, "peak": 0.750},
    "gsm8k_qwen3_8b_g2": {"last10": 0.375, "peak": 0.500},
    "gsm8k_qwen3_8b_g4": {"last10": 0.521, "peak": 0.750},
    "gsm8k_qwen3_8b_g8": {"last10": 0.844, "peak": 1.000},
    "gsm8k_qwen3_8b_g16": {"last10": 0.380, "peak": 0.719},
}

_LOG_LAST10_RE = re.compile(r"Last-10 avg accuracy:\s*([0-9.]+)%")
_LOG_PEAK_RE = re.compile(r"Peak accuracy:\s*([0-9.]+)%")
# grpo_cli.main() per-seed block (fractions, not percentages).
_CLI_SEED_RE = re.compile(r"\[grpo_cli\] Seed (\S+) done\.")
_CLI_LAST10_RE = re.compile(r"avg_last10\s*:\s*([0-9.]+)")
_CLI_PEAK_RE = re.compile(r"peak_reward\s*:\s*([0-9.]+)")


def _parse_result_file(path: Path) -> List[Dict]:
    """Parse a JSON result file or a trainer log into zero or more result dicts."""
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  ! could not parse {path}: {exc}", file=sys.stderr)
            return []
        return [data] if isinstance(data, dict) else []
    text = path.read_text(errors="replace")
    exp = path.stem
    # grpo_cli format: one block per seed, split on the "Seed N done." header.
    headers = list(_CLI_SEED_RE.finditer(text))
    rows: List[Dict] = []
    for i, h in enumerate(headers):
        block = text[h.end() : headers[i + 1].start() if i + 1 < len(headers) else len(text)]
        last10 = _CLI_LAST10_RE.search(block)
        peak = _CLI_PEAK_RE.search(block)
        if not last10 or not peak:
            continue
        rows.append(
            {
                "experiment": exp if len(headers) == 1 else f"{exp}_s{h.group(1)}",
                "last10_avg": float(last10.group(1)),
                "peak": float(peak.group(1)),
            }
        )
    if rows:
        return rows
    # Legacy final-report block (percentages)
    last10 = _LOG_LAST10_RE.search(text)
    peak = _LOG_PEAK_RE.search(text)
    if not last10 or not peak:
        return []
    return [
        {
            "experiment": exp,
            "last10_avg": float(last10.group(1)) / 100.0,
            "peak": float(peak.group(1)) / 100.0,
        }
    ]


def _match_key(experiment: str, expected: Dict[str, Dict[str, float]]) -> Optional[str]:
    """Map a result's 'experiment' tag to an expected-results key.

    Matches by substring, preferring the *longest* matching key so that
    ``gsm8k_qwen3_8b_base`` doesn't accidentally bind to ``gsm8k_qwen3_8b``.
    """
    e = experiment.lower()
    best: Optional[str] = None
    for key in expected:
        if key.startswith("_"):
            continue
        if key in e and (best is None or len(key) > len(best)):
            best = key
    return best


def verify(
    results_dir: Path,
    expected: Dict[str, Dict[str, float]],
    last10_tol: float,
    peak_tol: float,
) -> Tuple[List[Tuple[str, str, float, float, float, float, bool]], int]:
    """Return (rows, num_failed)."""
    rows: List[Tuple[str, str, float, float, float, float, bool]] = []
    failed = 0
    files = sorted(list(results_dir.rglob("*.json")) + list(results_dir.rglob("*.log")))
    if not files:
        print(f"  ! no .json or .log files found under {results_dir}", file=sys.stderr)
        return rows, 1

    for path in files:
        for parsed in _parse_result_file(path):
            exp = str(parsed.get("experiment", path.stem))
            key = _match_key(exp, expected)
            if not key:
                continue
            exp_vals = expected[key]
            got_l10 = float(parsed.get("last10_avg", parsed.get("last10", float("nan"))))
            got_peak = float(parsed.get("peak", parsed.get("peak_reward", float("nan"))))
            # NaN (missing field) compares False, so it counts as a failure.
            within = (
                abs(got_l10 - exp_vals["last10"]) <= last10_tol
                and abs(got_peak - exp_vals["peak"]) <= peak_tol
            )
            rows.append((exp, key, exp_vals["last10"], got_l10, exp_vals["peak"], got_peak, within))
            if not within:
                failed += 1
    return rows, failed


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--expected-results", type=Path, default=None)
    p.add_argument("--last10-tolerance", type=float, default=0.05)
    p.add_argument("--peak-tolerance", type=float, default=0.10)
    p.add_argument(
        "--strict", action="store_true", help="Fail if any expected result is missing a match."
    )
    args = p.parse_args()

    if not args.results_dir.exists():
        print(f"error: {args.results_dir} does not exist", file=sys.stderr)
        return 2

    expected = DEFAULT_EXPECTED
    if args.expected_results is not None:
        if not args.expected_results.exists():
            print(
                f"error: expectations file {args.expected_results} does not exist", file=sys.stderr
            )
            return 2
        try:
            expected = json.loads(args.expected_results.read_text())
        except json.JSONDecodeError as exc:
            print(f"error: bad JSON in {args.expected_results}: {exc}", file=sys.stderr)
            return 2

    rows, failed = verify(args.results_dir, expected, args.last10_tolerance, args.peak_tolerance)

    if not rows:
        print(
            f"FAIL: no result under {args.results_dir} matched an expected experiment "
            f"({', '.join(k for k in expected if not k.startswith('_'))}).",
            file=sys.stderr,
        )
        return 1

    hdr = f"{'experiment':35s} {'key':22s} {'last10_exp':>10s} {'last10_got':>10s} {'peak_exp':>9s} {'peak_got':>9s}  ok?"
    print(hdr)
    print("-" * len(hdr))
    for exp, key, l10_e, l10_g, pk_e, pk_g, ok in rows:
        print(
            f"{exp[:34]:35s} {key[:21]:22s} {l10_e:10.3f} {l10_g:10.3f} {pk_e:9.3f} {pk_g:9.3f}  {'Y' if ok else 'N'}"
        )

    print()
    print(
        f"summary: {len(rows) - failed}/{len(rows)} experiments within tolerance "
        f"(last10±{args.last10_tolerance:.2f}, peak±{args.peak_tolerance:.2f})"
    )

    if args.strict:
        missing = [
            k for k in expected if not k.startswith("_") and not any(r[1] == k for r in rows)
        ]
        if missing:
            print(f"  ! strict mode: missing expected experiments: {missing}")
            return 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
