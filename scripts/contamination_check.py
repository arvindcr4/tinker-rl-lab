"""
Data Contamination Detection for TinkerRL
==========================================
Checks for train/test overlap and data leakage in benchmark tasks.

References:
    Dodge et al., "Documenting Large Webtext Corpora" (EMNLP 2021)
    Jacovi et al., "Stop Uploading Test Data in Plain Text" (EMNLP 2023)

Usage:
    python scripts/contamination_check.py --results-dir experiments/results/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_string(text: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8-encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Return the set of n-gram tuples for a token list."""
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# 1. Arithmetic task contamination
# ---------------------------------------------------------------------------

def check_arithmetic_contamination(seed_list: List[int]) -> Dict:
    """
    Verify that generated arithmetic problems are uniformly sampled and not
    memorised across seeds.

    For each seed, generate a batch of (num1, num2) pairs and check:
    - Marginal distributions of num1 and num2 are approximately Uniform[1,99].
    - Chi-squared statistic is below a liberal threshold.
    - Different seeds produce overlapping but not identical problem sets.

    Parameters
    ----------
    seed_list : list of int seeds to check

    Returns
    -------
    dict with keys: 'uniform_ok', 'per_seed_stats', 'flag'
    """
    SAMPLE_SIZE = 500

    per_seed_stats = {}
    all_problem_sets: List[Set[Tuple[int, int]]] = []

    for seed in seed_list:
        rng = np.random.default_rng(seed)
        num1s = rng.integers(1, 100, size=SAMPLE_SIZE)
        num2s = rng.integers(1, 100, size=SAMPLE_SIZE)
        problems = set(zip(num1s.tolist(), num2s.tolist()))

        # Chi-squared uniformity test on num1 (99 bins)
        counts = Counter(num1s.tolist())
        observed = np.array([counts.get(v, 0) for v in range(1, 100)])
        expected = np.full(99, SAMPLE_SIZE / 99)
        chi2 = float(np.sum((observed - expected) ** 2 / expected))

        per_seed_stats[seed] = {
            "unique_problems": len(problems),
            "chi2_num1": round(chi2, 2),
            "chi2_ok": chi2 < 200,   # liberal threshold for 98 df, alpha=0.001
        }
        all_problem_sets.append(problems)

    # Check seed independence: no two seeds should produce the exact same set
    identical_pairs = []
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            if all_problem_sets[i] == all_problem_sets[j]:
                identical_pairs.append((seed_list[i], seed_list[j]))

    uniform_ok = all(s["chi2_ok"] for s in per_seed_stats.values())
    flag = "PASS" if (uniform_ok and not identical_pairs) else "FAIL"

    return {
        "check": "arithmetic_contamination",
        "uniform_ok": uniform_ok,
        "identical_seed_pairs": identical_pairs,
        "per_seed_stats": per_seed_stats,
        "flag": flag,
    }


# ---------------------------------------------------------------------------
# 2. N-gram overlap (GSM8K-style)
# ---------------------------------------------------------------------------

def check_ngram_overlap(
    train_texts: List[str],
    test_texts: List[str],
    n: int = 8,
) -> Dict:
    """
    Compute the fraction of test n-grams that appear in the training set.
    Standard methodology from Brown et al. (2020) and Jacovi et al. (2023).

    Parameters
    ----------
    train_texts : list of training document strings
    test_texts  : list of test document strings
    n           : n-gram size (8 or 13 are standard)

    Returns
    -------
    dict with 'overlap_fraction', 'contaminated_test_count', 'flag'
    """
    # Build training n-gram set
    train_ngram_set: Set[Tuple[str, ...]] = set()
    for text in train_texts:
        tokens = text.lower().split()
        train_ngram_set.update(_ngrams(tokens, n))

    contaminated_count = 0
    overlap_fractions  = []

    for text in test_texts:
        tokens     = text.lower().split()
        test_grams = _ngrams(tokens, n)
        if not test_grams:
            continue
        overlap = len(test_grams & train_ngram_set)
        frac    = overlap / len(test_grams)
        overlap_fractions.append(frac)
        if frac > 0.0:
            contaminated_count += 1

    mean_overlap = float(np.mean(overlap_fractions)) if overlap_fractions else 0.0
    flag = "FAIL" if mean_overlap > 0.05 else "PASS"   # >5% overlap triggers warning

    return {
        "check": f"ngram_overlap_n{n}",
        "n": n,
        "train_docs": len(train_texts),
        "test_docs": len(test_texts),
        "mean_overlap_fraction": round(mean_overlap, 4),
        "contaminated_test_docs": contaminated_count,
        "flag": flag,
    }


# ---------------------------------------------------------------------------
# 3. Exact duplicate detection
# ---------------------------------------------------------------------------

def check_exact_duplicates(
    dataset1: List[str],
    dataset2: List[str],
    label1: str = "train",
    label2: str = "test",
) -> Dict:
    """
    Hash every example in both datasets and report exact cross-set duplicates.

    Parameters
    ----------
    dataset1, dataset2 : lists of string examples
    label1, label2     : human-readable names for the two splits

    Returns
    -------
    dict with 'duplicate_count', 'duplicate_hashes', 'flag'
    """
    hashes1 = {_hash_string(ex) for ex in dataset1}
    hashes2 = {_hash_string(ex) for ex in dataset2}
    common  = hashes1 & hashes2

    flag = "FAIL" if common else "PASS"
    return {
        "check": "exact_duplicates",
        f"{label1}_size": len(dataset1),
        f"{label2}_size": len(dataset2),
        "duplicate_count": len(common),
        "duplicate_hashes": sorted(common)[:20],   # cap at 20 for readability
        "flag": flag,
    }


# ---------------------------------------------------------------------------
# 4. Seed independence
# ---------------------------------------------------------------------------

def check_seed_independence(seeds: List[int], sample_size: int = 200) -> Dict:
    """
    Verify that different seeds yield different problem orderings for the
    arithmetic task. All pairwise orderings should differ.

    Parameters
    ----------
    seeds       : list of seeds to compare
    sample_size : number of problems to generate per seed

    Returns
    -------
    dict with 'all_unique', 'pairwise_matches', 'flag'
    """
    orderings: Dict[int, List[Tuple[int, int]]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        num1s = rng.integers(1, 100, size=sample_size).tolist()
        num2s = rng.integers(1, 100, size=sample_size).tolist()
        orderings[seed] = list(zip(num1s, num2s))

    pairwise_matches = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            matches = sum(
                1 for a, b in zip(orderings[seeds[i]], orderings[seeds[j]]) if a == b
            )
            pairwise_matches.append({
                "seed_a": seeds[i],
                "seed_b": seeds[j],
                "matching_positions": matches,
                "fraction": round(matches / sample_size, 4),
            })

    # Expect at most ~1/99^2 ≈ 0.01% positional agreement by chance
    max_fraction = max((m["fraction"] for m in pairwise_matches), default=0.0)
    all_unique   = max_fraction < 0.10   # generous threshold
    flag = "PASS" if all_unique else "FAIL"

    return {
        "check": "seed_independence",
        "seeds": seeds,
        "pairwise_matches": pairwise_matches,
        "max_positional_overlap_fraction": round(max_fraction, 4),
        "all_unique": all_unique,
        "flag": flag,
    }


# ---------------------------------------------------------------------------
# 5. Results file integrity (checksum validation)
# ---------------------------------------------------------------------------

def check_results_integrity(results_dir: str) -> Dict:
    """
    Walk a results directory looking for JSON/CSV result files and verify
    their stored checksum (if any) matches the file content.

    Convention: each results file may include a sidecar <filename>.sha256
    containing the expected SHA-256 digest.  Files without a sidecar are
    flagged as "unverified" (not necessarily tampered).

    Parameters
    ----------
    results_dir : path to directory containing experiment result files

    Returns
    -------
    dict with per-file status and overall flag
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return {
            "check": "results_integrity",
            "error": f"Directory not found: {results_dir}",
            "flag": "SKIP",
        }

    file_status = []
    for candidate in sorted(results_path.rglob("*.json")) + sorted(results_path.rglob("*.csv")):
        sidecar = candidate.with_suffix(candidate.suffix + ".sha256")
        actual_hash = _hash_file(str(candidate))

        if sidecar.exists():
            expected_hash = sidecar.read_text().strip()
            match = actual_hash == expected_hash
            status = "OK" if match else "TAMPERED"
        else:
            status = "UNVERIFIED"

        file_status.append({
            "file": str(candidate.relative_to(results_path)),
            "sha256": actual_hash[:16] + "…",
            "status": status,
        })

    tampered = [f for f in file_status if f["status"] == "TAMPERED"]
    flag = "FAIL" if tampered else ("PASS" if file_status else "SKIP")

    return {
        "check": "results_integrity",
        "files_checked": len(file_status),
        "tampered_files": tampered,
        "file_status": file_status,
        "flag": flag,
    }


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(check_results: List[Dict], output_path: str) -> None:
    """
    Write a contamination report (JSON) to output_path.

    Parameters
    ----------
    check_results : list of result dicts from each check function
    output_path   : file path for the JSON report
    """
    overall_pass = all(r.get("flag") in ("PASS", "SKIP") for r in check_results)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_flag": "PASS" if overall_pass else "FAIL",
        "checks": check_results,
        "methodology": {
            "ngram_sizes": [8, 13],
            "references": [
                "Dodge et al. (EMNLP 2021) – Documenting Large Webtext Corpora",
                "Jacovi et al. (EMNLP 2023) – Stop Uploading Test Data in Plain Text",
                "Brown et al. (NeurIPS 2020) – Language Models are Few-Shot Learners",
            ],
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nContamination report written to: {output_path}")


# ---------------------------------------------------------------------------
# Synthetic GSM8K-like data for demonstration
# ---------------------------------------------------------------------------

def _generate_synthetic_gsm8k(n: int, rng: np.random.Generator) -> List[str]:
    """Generate n synthetic arithmetic word-problem strings (GSM8K-like)."""
    templates = [
        "Janet has {a} apples. She gives {b} to her friend. How many does she have left?",
        "A store sells {a} items in the morning and {b} items in the afternoon. Total?",
        "Tom walks {a} km on Monday and {b} km on Tuesday. How far in total?",
        "There are {a} red balls and {b} blue balls. How many balls altogether?",
        "Maria earns ${a} per hour and works {b} hours. What is her total pay?",
    ]
    problems = []
    for _ in range(n):
        tmpl = templates[int(rng.integers(0, len(templates)))]
        a = int(rng.integers(1, 100))
        b = int(rng.integers(1, 100))
        problems.append(tmpl.format(a=a, b=b))
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Data contamination detection for TinkerRL benchmarks."
    )
    parser.add_argument(
        "--results-dir", type=str, default="experiments/results/",
        help="Directory containing experiment result files for integrity check"
    )
    parser.add_argument(
        "--output", type=str, default="results/contamination/contamination_report.json",
        help="Path to write the contamination report JSON"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 42, 123],
        help="Seeds to use for arithmetic contamination and seed-independence checks"
    )
    args = parser.parse_args()

    all_results = []

    # ------------------------------------------------------------------
    print("\n[1/5] Checking arithmetic task contamination...")
    r1 = check_arithmetic_contamination(args.seeds)
    all_results.append(r1)
    print(f"  Uniform sampling: {'OK' if r1['uniform_ok'] else 'FAIL'}")
    print(f"  Identical seed pairs: {r1['identical_seed_pairs'] or 'none'}")
    print(f"  Flag: {r1['flag']}")

    # ------------------------------------------------------------------
    print("\n[2/5] Checking GSM8K n-gram overlap (synthetic demo)...")
    rng_train = np.random.default_rng(0)
    rng_test  = np.random.default_rng(999)
    train_texts = _generate_synthetic_gsm8k(1000, rng_train)
    test_texts  = _generate_synthetic_gsm8k(200,  rng_test)

    for n in (8, 13):
        r2 = check_ngram_overlap(train_texts, test_texts, n=n)
        all_results.append(r2)
        print(f"  {n}-gram overlap: {r2['mean_overlap_fraction']:.4f}  Flag: {r2['flag']}")

    # ------------------------------------------------------------------
    print("\n[3/5] Checking exact duplicates...")
    # For arithmetic: generate string representations of problems
    train_problems = [f"{a}+{b}" for a, b in zip(
        np.random.default_rng(0).integers(1, 100, 1000),
        np.random.default_rng(1).integers(1, 100, 1000),
    )]
    test_problems = [f"{a}+{b}" for a, b in zip(
        np.random.default_rng(99).integers(1, 100, 200),
        np.random.default_rng(100).integers(1, 100, 200),
    )]
    r3 = check_exact_duplicates(train_problems, test_problems)
    all_results.append(r3)
    print(f"  Exact cross-split duplicates: {r3['duplicate_count']}  Flag: {r3['flag']}")

    # ------------------------------------------------------------------
    print("\n[4/5] Checking seed independence...")
    r4 = check_seed_independence(args.seeds)
    all_results.append(r4)
    print(f"  Max positional overlap: {r4['max_positional_overlap_fraction']:.4f}  Flag: {r4['flag']}")

    # ------------------------------------------------------------------
    print("\n[5/5] Checking results file integrity...")
    r5 = check_results_integrity(args.results_dir)
    all_results.append(r5)
    if r5["flag"] == "SKIP":
        print(f"  Skipped (directory not found or empty): {args.results_dir}")
    else:
        print(f"  Files checked: {r5['files_checked']}  Tampered: {len(r5['tampered_files'])}  Flag: {r5['flag']}")

    # ------------------------------------------------------------------
    generate_report(all_results, args.output)

    overall = "PASS" if all(r.get("flag") in ("PASS", "SKIP") for r in all_results) else "FAIL"
    print(f"\n{'='*50}")
    print(f"  OVERALL CONTAMINATION CHECK: {overall}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
