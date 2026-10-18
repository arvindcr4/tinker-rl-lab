#!/usr/bin/env python3
"""
Offline claim verification script for TinkerRL Lab submission.

This script verifies every numeric claim used in the paper from shipped files.
Run from the extracted bundle root:
    python3 platform_modal/scripts/verify_claims_offline.py

Expected output:
    PASS checksum manifest
    PASS qwen3_8b_headline_reward
    PASS gsm8k_heldout_nonsignificant
    ...

Usage:
    python3 platform_modal/scripts/verify_claims_offline.py              # Run all checks
    python3 platform_modal/scripts/verify_claims_offline.py --claim qwen3_8b  # Run specific check
    python3 platform_modal/scripts/verify_claims_offline.py --list     # List all checks
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math
import hashlib
import re

ANALYSIS_VERSION = "review27-v3"

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Base directory (script location)
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

# Paths relative to BASE_DIR
EXPERIMENTS_DIR = BASE_DIR / "experiments"
# Try multiple possible locations for reports
REPORTS_DIR = BASE_DIR / "reports" / "final"
if not REPORTS_DIR.exists():
    REPORTS_DIR = BASE_DIR / "supporting_data" / "reports" / "final"
PAPER_DIR = BASE_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figures" / "v2"


def pass_check(name: str, details: str = "") -> str:
    """Format a passing check."""
    return f"{GREEN}PASS{RESET} {name}" + (f": {details}" if details else "")


def fail_check(name: str, details: str = "") -> str:
    """Format a failing check."""
    return f"{RED}FAIL{RESET} {name}" + (f": {details}" if details else "")


def warn_check(name: str, details: str = "") -> str:
    """Format a warning (partial pass)."""
    return f"{YELLOW}WARN{RESET} {name}" + (f": {details}" if details else "")


def load_json(path: Path) -> Dict:
    """Load and parse a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def safe_get(d: Dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict, handling None."""
    value = d.get(key, default)
    if value is None:
        return default
    return value


def verify_checksum_manifest() -> Tuple[bool, str]:
    """Hash every manifest member; reject missing, escaped, or ambiguous entries."""
    candidates = [BASE_DIR / "SHA256SUMS.txt", BASE_DIR / "checksums.sha256", BASE_DIR / "submission/contents/checksums.sha256"]
    manifest = next((p for p in candidates if p.is_file()), None)
    if manifest is None:
        return False, "UNVERIFIED: checksum manifest missing"
    try:
        lines = [line for line in manifest.read_text().splitlines() if line.strip()]
        if not lines:
            return False, "empty checksum manifest"
        seen=set()
        root=BASE_DIR.resolve()
        for line in lines:
            match=re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)",line)
            if not match:
                return False, "malformed checksum entry"
            digest, member=match.groups()
            target=(root/member).resolve()
            if not target.is_relative_to(root) or target in seen:
                return False, "escaped or duplicate manifest member"
            seen.add(target)
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest.lower():
                return False, f"missing or mismatched checksum: {member}"
        return True, f"{len(seen)} SHA256 digests verified ({ANALYSIS_VERSION})"
    except (OSError,ValueError) as exc:
        return False, str(exc)


def verify_qwen3_8b_headline_reward() -> Tuple[bool, str]:
    """Exact model/algorithm identity; absent and zero values are distinct."""
    try:
        data=load_json(EXPERIMENTS_DIR / "master_results.json")
        rows=data if isinstance(data,list) else data.get("experiments",[])
        values={}
        for algorithm in ("grpo","ppo"):
            matches=[r for r in rows if str(r.get("model","")).lower().split("/")[-1] == "qwen3-8b"
                     and str(r.get("method",r.get("algorithm",""))).lower() == algorithm]
            if len(matches)!=1:
                return False, f"UNVERIFIED: expected one exact Qwen3-8B/{algorithm} run, found {len(matches)}; select explicit run scope"
            value=matches[0].get("last10_avg",matches[0].get("last_10_avg"))
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                return False, f"missing/nonfinite {algorithm} reward"
            values[algorithm]=value
        passed=abs(values['grpo']-.344)<=.05 and abs(values['ppo']-.225)<=.05
        return passed, f"exact Qwen3-8B GRPO={values['grpo']:.4f}, PPO={values['ppo']:.4f}; tolerance=.05"
    except (OSError,ValueError,TypeError) as exc:
        return False,str(exc)


def verify_gsm8k_heldout() -> Tuple[bool, str]:
    """Recalculate a seed-level t comparison; no paper-value fallback.

    This is the historical summary-level approximate test, not example-paired
    inference. Exact test-protocol reproduction requires raw paired outcomes.
    """
    try:
        roots=list(dict.fromkeys(p.resolve() for p in [REPORTS_DIR,BASE_DIR / "reports/final"]))
        bases=[p/name for p in roots for name in ("gsm8k_base_control_200.json","gsm8k_base_results.json") if (p/name).is_file()]
        if not bases:
            return False,"UNVERIFIED: measured base evaluation missing"
        def counts(path):
            data=load_json(path); summary=data.get("summary",data)
            c=summary.get("correct"); n=summary.get("total",summary.get("attempted"))
            if type(c) is not int or type(n) is not int or n<=0 or not 0<=c<=n:
                raise ValueError(f"invalid evaluation counts: {path.name}")
            return c,n
        base_counts={counts(p) for p in bases}
        if len(base_counts)!=1:
            return False,"conflicting base evaluations"
        bc,bn=next(iter(base_counts)); base_rate=bc/bn
        paths=set(); seeds={}
        for root in roots:
            for path in sorted(root.glob("gsm8k_heldout_seed*.json")):
                resolved=path.resolve()
                if resolved in paths: continue
                paths.add(resolved)
                match=re.fullmatch(r"gsm8k_heldout_seed(\d+)\.json",path.name)
                if not match: continue
                seed=int(match[1])
                if seed in seeds:
                    return False,f"duplicate seed identity {seed} across different files"
                data=load_json(path)
                if 'seed' in data and data['seed']!=seed:
                    return False,f"filename/record seed mismatch {seed}"
                c,n=counts(path); seeds[seed]=c/n
        if len(seeds)<2:
            return False,f"UNVERIFIED: need at least two distinct measured seeds; n={len(seeds)}"
        from scipy import stats
        import statistics
        rates=list(seeds.values()); mean=statistics.mean(rates)
        se=statistics.stdev(rates)/math.sqrt(len(rates))
        if se==0:
            return False,"UNVERIFIED: degenerate zero-variance t statistic"
        p=float(2*stats.t.sf(abs((mean-base_rate)/se),len(rates)-1))
        success=abs(base_rate-.820)<.01 and abs(mean-.833)<.05 and .05<p<.40
        return success,f"base={bc}/{bn}, GRPO mean={mean:.6f}, n_unique_seeds={len(seeds)}, approximate seed-t p={p:.8f}"
    except (OSError,ValueError,TypeError,ImportError) as exc:
        return False,str(exc)


# review27-v3 evidence contract. These files are deliberately not synthesized
# from historical summary tables: cohort membership requires a scientific choice.
# Each claim file contains schema_version, protocol_id, expected_run_ids, rows.
# Each row has run_id, model, task, split, stack, algorithm, seed, protocol_id.
# Values below are raw observations/predictions, not precomputed claim statistics.
IDENTITY_FIELDS = ("run_id", "model", "task", "split", "stack", "algorithm", "protocol_id")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite numeric observation")
    return float(value)


def _claim_rows(claim):
    path = EXPERIMENTS_DIR / "verification_v3" / (claim + ".json")
    if not path.is_file():
        raise FileNotFoundError(f"UNVERIFIED: explicit {claim} cohort contract unavailable: {path}")
    document = load_json(path)
    if document.get("schema_version") != "review27-v3" or not document.get("protocol_id"):
        raise ValueError("unsupported evidence schema or missing protocol identity")
    rows = document.get("rows")
    expected = document.get("expected_run_ids")
    if not isinstance(rows, list) or not rows or not isinstance(expected, list) or not expected:
        raise ValueError("empty or missing explicit cohort")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in IDENTITY_FIELDS):
            raise ValueError("missing run/model/task/split/stack/algorithm/protocol identity")
        if type(row.get("seed")) is not int or row["protocol_id"] != document["protocol_id"]:
            raise ValueError("invalid seed or mismatched protocol")
        ids.append(row["run_id"])
    if any(not isinstance(i, str) for i in expected) or len(set(expected)) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("duplicate, missing, or unexpected cohort run")
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_evidence(claim, calculator):
    try:
        rows, digest = _claim_rows(claim)
        passed, detail = calculator(rows)
        return passed, f"{detail}; n_runs={len(rows)}; evidence_sha256={digest}; review27-v3"
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        return False, str(exc)


def verify_zvf_triage():
    def calculate(rows):
        counts = dict(TP=0, FP=0, TN=0, FN=0)
        for row in rows:
            # Predictions must be recorded independently of outcome labels.
            predicted, observed = row["predicted_collapse"], row["observed_collapse"]
            if type(predicted) is not bool or type(observed) is not bool:
                raise ValueError("triage requires Boolean rule predictions and observed labels")
            counts[("T" if predicted == observed else "F") + ("P" if predicted else "N")] += 1
        return counts["TP"] == 2 and counts["FN"] == 0 and counts["FP"] == 0 and counts["TN"] > 0, json.dumps(counts, sort_keys=True)
    return _verify_evidence("zvf_triage", calculate)


def verify_zvf_correlation():
    def calculate(rows):
        from scipy.stats import pearsonr
        # Reject incomplete cohorts rather than silently selecting favorable cases.
        x = [_number(r["mean_zvf"]) for r in rows]
        y = [_number(r["last10_avg"]) for r in rows]
        units = [r["independent_unit_id"] for r in rows]
        if len(rows) < 3 or any(not isinstance(u,str) or not u for u in units) or len(set(units)) != len(units) or len(set(x)) < 2 or len(set(y)) < 2:
            raise ValueError("correlation needs complete nonconstant independent units")
        rho, p = map(float, pearsonr(x, y))
        return abs(rho - (-.769)) <= .005 and abs(p - .0008) <= .0005, f"Pearson r={rho:.8g}, two-sided p={p:.8g}, complete_cases={len(rows)}"
    return _verify_evidence("zvf_correlation", calculate)


def verify_saturation_fit():
    def calculate(rows):
        scores = []
        for row in rows:
            if row.get("trace_provenance") != "measured" or row.get("fit_family") != "exponential_saturation":
                raise ValueError("actual measured traces and exponential-saturation parameters required")
            t = [_number(v) for v in row["steps"]]
            y = [_number(v) for v in row["rewards"]]
            if len(t) != len(y) or len(t) < 3 or any(b <= a for a,b in zip(t,t[1:])):
                raise ValueError("invalid ordered measured trajectory")
            initial, asymptote, rate = [_number(row[k]) for k in ("initial", "asymptote", "rate")]
            if rate <= 0:
                raise ValueError("positive saturation rate required")
            predicted = [asymptote + (initial-asymptote)*math.exp(-rate*(v-t[0])) for v in t]
            mean = sum(y)/len(y)
            total = sum((v-mean)**2 for v in y)
            if total == 0:
                raise ValueError("R2 undefined for constant trace")
            scores.append(1-sum((a-b)**2 for a,b in zip(y,predicted))/total)
        score = sum(scores)/len(scores)
        return abs(score-.21) <= .02, f"equal-run mean measured-trace R2={score:.8g}; individual_R2={scores}"
    return _verify_evidence("saturation_fit", calculate)


def _matched_cells(rows, varying):
    keys = ("model", "task", "split", "stack", "seed", "protocol_id")
    keys = tuple(k for k in keys if k != varying)
    cells = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        cell = cells.setdefault(key, {})
        value = row[varying]
        if value in cell:
            raise ValueError("duplicate treatment in matched cell")
        cell[value] = _number(row["last10_avg"])
    return cells


def verify_stack_sensitivity():
    def calculate(rows):
        cells = _matched_cells(rows, "algorithm")
        differences = {}
        for key, cell in cells.items():
            if set(cell) != {"grpo", "ppo"}:
                raise ValueError("every matched cell requires exactly GRPO and PPO")
            differences.setdefault(key[0], []).append(cell["grpo"]-cell["ppo"])
        means = {k: sum(v)/len(v) for k,v in differences.items()}
        return len(means) >= 2 and min(means.values()) < 0 < max(means.values()), f"descriptive matched GRPO-minus-PPO by model={means}; no causal stack attribution"
    return _verify_evidence("stack_sensitivity", calculate)


def verify_trl_baseline():
    def calculate(rows):
        if any(r["stack"] != "trl" or r["algorithm"] != "grpo" for r in rows):
            raise ValueError("exact TRL/GRPO cohort required")
        scope = {(r["model"], r["task"], r["split"]) for r in rows}
        if len(scope) != 1 or len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("one matched task/model/split with distinct seeds required")
        values = [_number(r["last10_avg"]) for r in rows]
        mean = sum(values)/len(values)
        return abs(mean-.734) <= .01, f"equal-seed TRL/GRPO last10 mean={mean:.8g}"
    return _verify_evidence("trl_baseline", calculate)


def verify_group_size():
    def calculate(rows):
        if any(type(r.get("group_size")) is not int or r["group_size"] < 2 for r in rows):
            raise ValueError("explicit integer group sizes >=2 required")
        # Algorithm is a matching dimension here as well as stack/task/model.
        if len({r["algorithm"] for r in rows}) != 1:
            raise ValueError("algorithm must be held fixed in group-size ablation")
        cells = _matched_cells(rows, "group_size")
        if any(8 not in c or len(c) < 2 for c in cells.values()) or len({tuple(sorted(c)) for c in cells.values()}) != 1:
            raise ValueError("matched seeds require G=8 plus identical comparison group-size grid")
        values = [c[8] for c in cells.values()]
        mean = sum(values)/len(values)
        return abs(mean-.844) <= .01, f"matched G=8 equal-cell last10 mean={mean:.8g}; n_cells={len(cells)}"
    return _verify_evidence("group_size", calculate)


def verify_tool_proxy():
    return False, "UNVERIFIED: documentation scope statement is not an executable numeric claim"


def verify_proxy_harnesses():
    return False, "UNVERIFIED: documentation scope statement is not an executable numeric claim"

def verify_figure_inputs() -> Tuple[bool, str]:
    """Verify figure input files exist."""
    required_figures = [
        FIGURES_DIR / "learning_curves.pdf",
        FIGURES_DIR / "performance_profiles.pdf",
        FIGURES_DIR / "sensitivity_heatmap.pdf",
        FIGURES_DIR / "framework_comparison.pdf",
        FIGURES_DIR / "group_size_ablation.pdf",
        FIGURES_DIR / "ppo_vs_grpo.pdf",
        FIGURES_DIR / "scaling.pdf",
        FIGURES_DIR / "kl_proxy.pdf",
    ]
    
    missing = []
    for fig in required_figures:
        if not fig.exists():
            missing.append(fig.name)
    
    if missing:
        return False, f"Missing figures: {', '.join(missing[:3])}"
    
    return True, f"All {len(required_figures)} figures present"


def list_checks() -> List[str]:
    """List all available checks."""
    return [
        "checksum_manifest",
        "qwen3_8b_headline_reward",
        "gsm8k_heldout_nonsignificant",
        "zvf_triage_claim",
        "stack_sensitivity_claim",
        "tool_proxy_scope",
        "proxy_harnesses",
        "zvf_correlation",
        "saturation_fit_quality",
        "trl_baseline",
        "group_size_ablation",
        "figure_inputs_present",
    ]


# Map check names to functions
CHECKS = {
    "checksum_manifest": ("Verify SHA256 checksums", verify_checksum_manifest),
    "qwen3_8b_headline_reward": ("Qwen3-8B GRPO/PPO last-10 values", verify_qwen3_8b_headline_reward),
    "gsm8k_heldout_nonsignificant": ("Held-out GSM8K 82.0%→83.3%, p≈0.26", verify_gsm8k_heldout),
    "zvf_triage_claim": ("ZVF triage catches 2/2 collapse, 0 FP", verify_zvf_triage),
    "stack_sensitivity_claim": ("PPO/GRPO reversal across model families", verify_stack_sensitivity),
    "tool_proxy_scope": ("Tool-use rewards are schema/proxy", verify_tool_proxy),
    "proxy_harnesses": ("HumanEval/MATH are reward probes", verify_proxy_harnesses),
    "zvf_correlation": ("ZVF correlation r=-0.769, p=0.0008", verify_zvf_correlation),
    "saturation_fit_quality": ("Exponential R²≈0.21", verify_saturation_fit),
    "trl_baseline": ("TRL GRPO baseline mean≈73.4%", verify_trl_baseline),
    "group_size_ablation": ("Group size G=8 last-10≈84.4%", verify_group_size),
    "figure_inputs_present": ("Figure input files exist", verify_figure_inputs),
}


def run_check(check_name: str, verbose: bool = False) -> Tuple[bool, str, str]:
    """Run a single check."""
    if check_name not in CHECKS:
        return False, "", f"Unknown check: {check_name}"
    
    description, func = CHECKS[check_name]
    
    try:
        success, details = func()
        return success, description, details
    except Exception as e:
        return False, description, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Offline claim verification for TinkerRL Lab"
    )
    parser.add_argument(
        "--claim", "-c",
        help="Run specific claim check (e.g., qwen3_8b_headline_reward)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available claim checks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR, help="Explicit evidence root containing experiments, reports, and checksum manifest")
    args = parser.parse_args()
    globals()["BASE_DIR"] = args.base_dir.resolve()
    globals()["EXPERIMENTS_DIR"] = BASE_DIR / "experiments"
    globals()["REPORTS_DIR"] = BASE_DIR / "reports" / "final"
    globals()["FIGURES_DIR"] = BASE_DIR / "paper" / "figures" / "v2"
    
    if args.list:
        print("Available checks:")
        for name, (desc, _) in CHECKS.items():
            print(f"  {name}: {desc}")
        return 0
    
    # Determine which checks to run
    if args.claim:
        checks_to_run = [args.claim]
    else:
        checks_to_run = list(CHECKS.keys())
    
    # Run checks
    results = []
    for check_name in checks_to_run:
        success, description, details = run_check(check_name, args.verbose)
        status = pass_check(check_name, details) if success else fail_check(check_name, details)
        results.append((success, check_name, description, details))
        print(status)
    
    # Summary
    n_pass = sum(1 for r in results if r[0])
    n_fail = len(results) - n_pass
    
    print()
    print(f"Summary: {n_pass}/{len(results)} checks passed")
    
    if n_fail > 0:
        print(f"{RED}{n_fail} checks failed{RESET}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
