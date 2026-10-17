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

ANALYSIS_VERSION = "review27-v2"

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


def verify_zvf_triage() -> Tuple[bool, str]:
    return False, "UNVERIFIED (verify_zvf_triage): requires run-level rule predictions and observed collapse labels"


def verify_stack_sensitivity() -> Tuple[bool, str]:
    return False, "UNVERIFIED (verify_stack_sensitivity): requires exact matched run and stack identities, not first model-family match"


def verify_tool_proxy() -> Tuple[bool, str]:
    return False, "UNVERIFIED (verify_tool_proxy): scope statement is documentation, not an executable evidence check"


def verify_proxy_harnesses() -> Tuple[bool, str]:
    return False, "UNVERIFIED (verify_proxy_harnesses): scope statement is documentation, not an executable evidence check"


def verify_zvf_correlation() -> Tuple[bool, str]:
    return False, "UNVERIFIED (verify_zvf_correlation): requires defined complete-case observations and correlation recomputation"


def verify_saturation_fit() -> Tuple[bool, str]:
    return False, "UNVERIFIED (verify_saturation_fit): requires actual traces, fitted predictions, and stated fit aggregation"


def verify_trl_baseline() -> Tuple[bool, str]:
    return False, "UNVERIFIED (verify_trl_baseline): requires exact TRL/GRPO run cohort and protocol identities"


def verify_group_size() -> Tuple[bool, str]:
    return False, "UNVERIFIED (verify_group_size): requires matched group-size ablation cohort, not first model match"


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
    
    args = parser.parse_args()
    
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
