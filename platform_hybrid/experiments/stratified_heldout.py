#!/usr/bin/env python3
"""Checkpoint-selection audit, protocol v2. Sampling frames contain saved checkpoint identities or one already measured final endpoint. Missing held-out observations remain missing. Descriptive checkpoint means have no independent-run confidence intervals; historical records need provenance audit."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir())))
from platform_hybrid.experiments.research_contracts import require_empirical, inherit_evidence, EvidenceUnavailable


import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "experiments" / "master_results.json"
HELDOUT = REPO / "experiments" / "results" / "heldout_gsm8k.json"
OUT_TSV = REPO / "experiments" / "results" / "heldout_stratified.tsv"

N_BOOT = 1000
SEED_DEFAULT = 42

# Conditions we explicitly report in the paper table. Each entry is a
# (display_name, predicate) pair; the predicate runs over an experiments row.
CONDITIONS = [
    ("Qwen3-8B-Base (W1)",
     lambda e: e.get("model") == "Qwen/Qwen3-8B-Base"
               or "qwen3-8b-base" in str(e.get("experiment_id", "")).lower()),
    ("DeepSeek-V3.1 (W3)",
     lambda e: "deepseek" in str(e.get("model", "")).lower()
               and str(e.get("method") or "").upper().startswith("GRPO")),
    ("gpt-oss-20b (arch)",
     lambda e: "gpt-oss-20b" in str(e.get("model", "")).lower()
               or "gpt-oss-20b" in str(e.get("experiment_id", "")).lower()),
    ("Qwen3.5-4B (scale)",
     lambda e: "qwen3.5-4b" in str(e.get("model", "")).lower()
               or "qwen3.5-4b" in str(e.get("experiment_id", "")).lower()),
    ("gpt-oss-120b (W1)",
     lambda e: "gpt-oss-120b" in str(e.get("model", "")).lower()
               or "gpt-oss-120b" in str(e.get("experiment_id", "")).lower()),
    ("Qwen3-8B GRPO (campaign-v2)",
     lambda e: e.get("model") == "Qwen/Qwen3-8B"
               and str(e.get("method") or "").upper().startswith("GRPO")
               and "campaign_v2" in str(e.get("experiment_id", ""))),
    ("Qwen3-8B Wave6 temp-0.4",
     lambda e: e.get("group") == "Wave 6 Sensitivity"
               and "temp0.4" in str(e.get("experiment_id", ""))),
    ("Qwen3-8B Wave6 batch-1",
     lambda e: e.get("group") == "Wave 6 Sensitivity"
               and "batch1" in str(e.get("experiment_id", ""))),
    ("TRL-GRPO Qwen2.5-0.5B",
     lambda e: e.get("model") == "Qwen/Qwen2.5-0.5B"
               and str(e.get("method") or "").upper().startswith("GRPO")),
    ("SB3/CleanRL/Tianshou PPO",
     lambda e: "PPO" in str(e.get("method") or "")
               and str(e.get("model", "")).lower().endswith("policy network)")),
]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_master():
    with MASTER.open() as f:
        data = json.load(f)
    return {**data, "experiments": [inherit_evidence(r, data) for r in data.get("experiments", [])]}


def load_heldout_map():
    """Retain evaluation receipts verbatim; container exclusions are inherited."""
    if not HELDOUT.exists():
        return {}
    with HELDOUT.open() as f:
        data = json.load(f)
    out = {}
    for r in data.get("results", []):
        row = inherit_evidence(r, data)
        key = row.get("checkpoint_id") or row.get("experiment_id")
        if not key:
            raise EvidenceUnavailable("Held-out evaluation identity required")
        for identity in {key, row.get("experiment_id")} - {None}:
            if identity in out and out[identity] != row:
                raise EvidenceUnavailable(f"Ambiguous held-out evaluation: {identity}")
            out[identity] = row
    return out


def group_checkpoints(experiments, heldout_map):
    """Use persisted checkpoints only; measured run endpoints are never expanded."""
    grouped = defaultdict(list)
    for exp in experiments:
        condition = next((name for name, pred in CONDITIONS if pred(exp)), None)
        if condition is None:
            continue
        exp_id = exp.get("experiment_id")
        checkpoints = exp.get("checkpoints") or []
        # A measured final endpoint is one observation, not one per trace step.
        if not checkpoints and exp_id in heldout_map:
            checkpoints = [{"checkpoint_id": heldout_map[exp_id].get("checkpoint_id"),
                            "last10_avg": exp.get("last10_avg", 0), "evaluation": heldout_map[exp_id]}]
        for checkpoint in checkpoints:
            cid = checkpoint.get("checkpoint_id") or checkpoint.get("checkpoint_path")
            # Training provenance can veto an evaluation, but cannot supply a
            # missing evaluation receipt or promote a bare endpoint to measured.
            evaluation = inherit_evidence(checkpoint.get("evaluation") or heldout_map.get(cid, {}),
                                          checkpoint, inherit_missing=False)
            evaluation = inherit_evidence(evaluation, exp, inherit_missing=False)
            acc = evaluation.get("heldout_accuracy")
            reason = None
            try:
                require_empirical(evaluation)
                if not cid or evaluation.get("checkpoint_id") != cid:
                    raise EvidenceUnavailable("Matching saved checkpoint identity required")
                if isinstance(acc, bool) or not isinstance(acc, (int, float)) or not math.isfinite(acc) or not 0 <= acc <= 1:
                    raise EvidenceUnavailable("Finite held-out accuracy in [0, 1] required")
                n = evaluation.get("heldout_n")
                if n is not None and (type(n) is not int or n <= 0):
                    raise EvidenceUnavailable("Held-out count must be a positive integer when supplied")
            except EvidenceUnavailable as exc:
                reason = str(exc)
            measured = reason is None
            grouped[condition].append({"experiment_id": cid, "parent_experiment_id": exp_id,
                "checkpoint_id": cid, "last10_avg": checkpoint.get("last10_avg", exp.get("last10_avg", 0)),
                "_heldout_measured": measured, "_heldout_acc": acc if measured else None,
                "_heldout_n": evaluation.get("heldout_n") if measured else None,
                "_evaluation_receipt": evaluation, "_exclusion_reason": reason})
    return grouped


def linear_fit(xs, ys):
    """Retained for reference -- not used for imputation; see group_checkpoints
    for the rationale behind using an anchored surrogate instead."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


# -----------------------------------------------------------------------------
# Sampling protocols
# -----------------------------------------------------------------------------

def sample_p1(rows):
    """Top-10 by training-last10 (descending)."""
    ordered = sorted(rows, key=lambda r: r["last10_avg"], reverse=True)
    return ordered[:10]


def sample_p2(rows, rng):
    """Uniform random 10 (without replacement)."""
    k = min(10, len(rows))
    return rng.sample(rows, k)


def sample_p3(rows, rng):
    """Stratified: 2 per decile from deciles 1,3,5,7,9 of training-last10.
    Nearest-decile fallback if a target decile is empty."""
    ordered = sorted(rows, key=lambda r: r["last10_avg"])
    n = len(ordered)
    if n == 0:
        return []
    # Partition into 10 deciles (round-robin tolerant to small n).
    deciles = [[] for _ in range(10)]
    for i, r in enumerate(ordered):
        d = min(9, int(10 * i / max(1, n)))
        deciles[d].append(r)
    target_idxs = [0, 2, 4, 6, 8]  # deciles 1,3,5,7,9 (0-indexed)
    chosen = []
    for di in target_idxs:
        pool = deciles[di][:]
        off = 1
        # nearest-decile fallback (walk outward)
        while not pool and off < 10:
            for alt in (di + off, di - off):
                if 0 <= alt <= 9 and deciles[alt]:
                    pool = deciles[alt][:]
                    break
            off += 1
        if not pool:
            continue
        take = min(2, len(pool))
        chosen.extend(rng.sample(pool, take))
    # de-duplicate (stratified draws may collide under fallback)
    seen = set()
    unique = []
    for r in chosen:
        key = r.get("experiment_id") or id(r)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED_DEFAULT, alpha=0.05):
    """Checkpoint selection summaries are descriptive, not independent-run CIs."""
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return (sum(vals) / len(vals) if vals else float("nan"), float("nan"), float("nan"))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def evaluate_protocol(rows, proto, rng):
    if proto == "p1":
        return sample_p1(rows)
    if proto == "p2":
        return sample_p2(rows, rng)
    if proto == "p3":
        return sample_p3(rows, rng)
    raise ValueError(f"unknown protocol {proto}")


def run(protocols, seed):
    master = load_master()
    heldout = load_heldout_map()
    grouped = group_checkpoints(master["experiments"], heldout)

    rng = random.Random(seed)
    rows_out = []

    # Per-protocol ranking: collect mean first, then re-iterate to assign rank.
    tmp = defaultdict(list)  # proto -> [(cond, mean, lo, hi, n, sparse, measured_ratio)]

    for cond_name, _pred in CONDITIONS:
        rows = grouped.get(cond_name, [])
        n_c = len(rows)
        sparse = n_c < 5
        for proto in protocols:
            sample = evaluate_protocol(rows, proto, rng)
            measured = [r for r in sample if r.get("_heldout_measured")]
            vals = [r["_heldout_acc"] for r in sample]
            mean, lo, hi = bootstrap_ci(vals, seed=seed + ord(proto[-1]))
            tmp[proto].append({
                "condition": cond_name,
                "mean_acc": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_checkpoints_total": n_c,
                "n_sampled": len(sample),
                "n_measured": len(measured),
                "sparse": sparse,
            })

    # Assign per-protocol rank by mean_acc descending (NaN last).
    for proto, entries in tmp.items():
        ranked = sorted(
            entries,
            key=lambda e: (math.isnan(e["mean_acc"]), -e["mean_acc"]),
        )
        for i, e in enumerate(ranked, start=1):
            e["rank"] = i if math.isfinite(e["mean_acc"]) else ""
        for e in entries:
            rows_out.append({
                "condition": e["condition"],
                "protocol": proto,
                "mean_acc": round(e["mean_acc"], 4) if not math.isnan(e["mean_acc"]) else "",
                "ci_low": round(e["ci_low"], 4) if not math.isnan(e["ci_low"]) else "",
                "ci_high": round(e["ci_high"], 4) if not math.isnan(e["ci_high"]) else "",
                "rank": e["rank"],
                "n_checkpoints_total": e["n_checkpoints_total"],
                "n_sampled": e["n_sampled"],
                "n_measured": e["n_measured"],
                "sparse": str(e["sparse"]).lower(),
                "interval_type": "none_descriptive_checkpoint_mean",
                "source": "measured_heldout_only" if e["n_measured"] else "no_eligible_heldout_measurements",
            })

    return rows_out


def write_tsv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition", "protocol", "mean_acc", "ci_low", "ci_high", "rank",
        "n_checkpoints_total", "n_sampled", "n_measured", "sparse", "interval_type", "source",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["p1", "p2", "p3", "all"], default="all")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--out", default=str(OUT_TSV))
    args = ap.parse_args(argv)

    protocols = ["p1", "p2", "p3"] if args.protocol == "all" else [args.protocol]

    if not MASTER.exists():
        print(f"ERROR: {MASTER} not found. Cannot run stratified heldout audit.",
              file=sys.stderr)
        return 2

    rows = run(protocols, args.seed)
    out_path = Path(args.out)
    write_tsv(rows, out_path)

    # Short textual summary to stdout for CI/log readability.
    print(f"wrote {len(rows)} rows to {out_path}")
    by_proto = defaultdict(list)
    for r in rows:
        if r["mean_acc"] != "":
            by_proto[r["protocol"]].append((r["condition"], r["mean_acc"], r["rank"]))
    for proto, entries in by_proto.items():
        entries.sort(key=lambda t: t[2])
        print(f"\n[{proto}] condition ordering:")
        for cond, acc, rk in entries:
            print(f"  #{rk:2d}  {cond:40s} mean_acc={acc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
