#!/usr/bin/env python3
"""
P6 iter 190 — registry measured-vs-claimed audit, RAW-data recompute variant.

Vein: (a) "validate existing entries against measured behavior" — but unlike
iter-178 (claim_alignment) which reads the registry's stored measured[] block,
this script recomputes the deltas from the RAW per-step TSVs (n2_metrics.tsv +
zvf_iter130_method_risk.tsv) and then compares those remeasured values to the
registry's *expected_effects* predicted_signs.

Why the recompute matters: iter-178 trusts whatever numbers the registry has
stored under `measured[]`. If those stored numbers drifted from the raw data
(silent bug, rounding, version skew), iter-178 would propagate the drift.
This iter-190 catches that drift by treating the raw TSVs as ground truth.

For entries with `expected_effects` but NO measured[] (dapo, gspo, ppo,
ppo_reinforce, liteppo, reinforce) we cannot yet validate — we surface them as
UNMEASURABLE and quantify the validation gap.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
REG = WORKTREE / "registry"
ENT = REG / "entries"
RES = WORKTREE / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
N2_TSV = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume" / "n2_metrics.tsv"
ZVFRISK_TSV = WORKTREE / "experiments" / "results" / "zvf_iter130_method_risk.tsv"

N_BOOT = 2000
SEED = 20260706
LAST10_STEP_THRESHOLD = 30  # last 10 of 40 steps


def paired_step_bootstrap_pct(a, b, n_boot=N_BOOT, seed=SEED):
    """Paired-step percentile bootstrap CI on mean(b - a). Returns (delta, ci_lo, ci_hi)."""
    rng = random.Random(seed)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    diffs = [bi - ai for ai, bi in zip(a, b)]
    point = sum(diffs) / len(diffs)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return point, lo, hi


def predict_to_sign_pred(predicted_sign):
    """Map predicted_sign string -> callable(measured_delta) -> verdict."""
    ps = predicted_sign.strip()
    if ps.startswith("<0") or ps == "-":
        return lambda d: "SUPPORTS" if d < 0 else ("CONTRADICTS" if d > 0 else "NEUTRAL")
    if ps.startswith(">0") or ps == "+":
        return lambda d: "SUPPORTS" if d > 0 else ("CONTRADICTS" if d < 0 else "NEUTRAL")
    if ps.startswith(">=0") or ps == "+=":
        return lambda d: "SUPPORTS" if d >= 0 else "CONTRADICTS"
    if ps.startswith("<=0") or ps == "-=":
        return lambda d: "SUPPORTS" if d <= 0 else "CONTRADICTS"
    if ps in ("==0", "0"):
        return lambda d: "NEUTRAL" if abs(d) < 1e-9 else "CONTRADICTS"
    return lambda d: "UNCLAIMED"


def load_n2():
    rows = list(csv.DictReader(open(N2_TSV), delimiter="\t"))
    return rows


def n2_last10_by_method_metric(rows, metric, step_lo=LAST10_STEP_THRESHOLD):
    out = defaultdict(list)
    for r in rows:
        if int(r["step"]) >= step_lo:
            out[r["method"]].append(float(r[metric]))
    return dict(out)


def zvfrisk_method_table():
    """zvf130 method_risk has columns: method, zvf_risk_mean, zvf_risk_sd, mag_mean, csd_mean, drift_mean."""
    rows = list(csv.DictReader(open(ZVFRISK_TSV), delimiter="\t"))
    return {r["method"]: r for r in rows}


def zvf_risk_delta(panel, base_method, method, metric_key):
    """ZVFRISK has only ONE row per method (5-seed aggregate with sd); delta = (method - base_method).
    Approximate CI via delta_norm = delta / sqrt(sd_a^2/n_a + sd_b^2/n_b). n=5 per method.
    """
    a = panel.get(base_method)
    b = panel.get(method)
    if not a or not b:
        return None
    a_val = float(a[metric_key])
    b_val = float(b[metric_key])
    delta = b_val - a_val
    try:
        sd_a = float(a.get("zvf_risk_sd", "0") or 0)
        sd_b = float(b.get("zvf_risk_sd", "0") or 0)
    except Exception:
        sd_a = sd_b = 0.0
    n = 5
    se = ((sd_a ** 2 + sd_b ** 2) / n) ** 0.5 if (sd_a > 0 or sd_b > 0) else 0.0
    # 95% normal-approx CI
    ci_lo = delta - 1.96 * se
    ci_hi = delta + 1.96 * se
    return delta, ci_lo, ci_hi, n


def main():
    print(f"[iter190] loading N2 from {N2_TSV}")
    n2_rows = load_n2()
    print(f"[iter190] N2 rows: {len(n2_rows)}")

    zvrisk = zvfrisk_method_table()
    print(f"[iter190] zvf130 method_risk methods: {sorted(zvrisk.keys())}")

    # ----- (A) Recompute N2 last-10 deltas for grpo/aero/gift/areal -----
    metrics_n2 = ["zvf", "reward_mean", "pcd", "mean_len", "cv_len", "frac_all_zero", "frac_all_one"]
    base = "grpo"
    methods_n2 = ["grpo", "aero", "gift", "areal"]
    last10 = {met: n2_last10_by_method_metric(n2_rows, met) for met in metrics_n2}

    per_pair_tsv = []
    per_method_summary = {}
    for met in metrics_n2:
        vals = last10[met]
        base_vals = vals.get(base, [])
        for m in methods_n2:
            if m == base:
                continue
            mv = vals.get(m, [])
            if not mv or not base_vals:
                continue
            d, lo, hi = paired_step_bootstrap_pct(base_vals, mv)
            per_pair_tsv.append({
                "metric": met,
                "panel": "n2_same_stack_last10",
                "base": base,
                "method": m,
                "n": len(mv),
                "mean_base": statistics.mean(base_vals),
                "mean_method": statistics.mean(mv),
                "delta_method_minus_base": round(d, 6),
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "significant": bool(lo > 0 or hi < 0),
                "data_source": "experiments/results/n2_reward_tensor_resume/n2_metrics.tsv",
                "ci_method": f"paired_step_bootstrap_pct n_boot={N_BOOT} seed={SEED}",
            })

    # ----- (B) Recompute zvf130 risk-index deltas vs grpo -----
    zvf130_metrics = ["zvf_risk_mean", "mag_mean", "csd_mean", "drift_mean"]
    methods_zvrisk = ["grpo", "aero", "gift", "areal", "ngrpo", "cppo", "mcgrpo", "es", "scafgrpo"]
    for met in zvf130_metrics:
        for m in methods_zvrisk:
            if m == base:
                continue
            res = zvf_risk_delta(zvrisk, base, m, met)
            if res is None:
                continue
            d, lo, hi, n = res
            per_pair_tsv.append({
                "metric": met,
                "panel": "zvf130_5seed",
                "base": base,
                "method": m,
                "n": n,
                "mean_base": float(zvrisk[base][met]),
                "mean_method": float(zvrisk[m][met]),
                "delta_method_minus_base": round(d, 6),
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "significant": bool(lo > 0 or hi < 0),
                "data_source": "experiments/results/zvf_iter130_method_risk.tsv",
                "ci_method": f"welch_norm_approx_95pct n=5",
            })

    # ----- (C) Walk every delta_*.json: match expected_effects to recomputed pair -----
    lookup = defaultdict(dict)  # (method, metric) -> row from per_pair_tsv
    for r in per_pair_tsv:
        lookup[(r["method"], r["metric"])] = r

    delta_files = sorted(glob.glob(str(ENT / "delta_*.json")))
    per_alignment = []
    per_entry_rollup = defaultdict(lambda: {"supports": 0, "contradicts": 0, "neutral": 0, "unclaimed": 0, "unmeasurable": 0, "expected_total": 0, "measured_total": 0})

    for fp in delta_files:
        with open(fp) as f:
            d = json.load(f)
        did = d.get("id", os.path.basename(fp))
        base_method = d.get("base", "grpo")
        expected = d.get("expected_effects", []) or []
        measured = d.get("measured", []) or []
        # Per-entry rollup
        per_entry_rollup[did]["expected_total"] = len(expected)
        per_entry_rollup[did]["measured_total"] = len(measured)
        # Map method name from delta_<id>.json to the canonical variant
        method_name = did.replace("delta_", "")
        # Iterate expected_effects
        for e in expected:
            met = e.get("metric", "?")
            panel = e.get("panel", "?")
            ps = e.get("predicted_sign", "?")
            judge = predict_to_sign_pred(ps)
            # Find a recomputed row matching (method, metric)
            rk = lookup.get((method_name, met))
            verdict = "UNMEASURABLE"
            measured_delta = None
            ci_lo = ci_hi = None
            sig = None
            panel_used = "RECOMPUTED" if rk and rk.get("panel") == panel else ("PANEL_MISMATCH" if rk else "NONE")
            if rk is not None and rk["panel"] == panel:
                measured_delta = rk["delta_method_minus_base"]
                ci_lo = rk["ci_low"]
                ci_hi = rk["ci_high"]
                sig = rk["significant"]
                verdict = judge(measured_delta)
                if sig is False and verdict == "SUPPORTS":
                    verdict = "SUPPORTS_NS"  # supports direction but not significant
            elif rk is not None:
                # Have a row in a different panel
                measured_delta = rk["delta_method_minus_base"]
                ci_lo = rk["ci_low"]
                ci_hi = rk["ci_high"]
                verdict = "PANEL_MISMATCH"
            per_alignment.append({
                "delta_id": did,
                "name": d.get("name", ""),
                "method": method_name,
                "metric": met,
                "panel_expected": panel,
                "predicted_sign": ps,
                "panel_used": panel_used,
                "recomputed_delta": measured_delta if measured_delta is not None else "",
                "recomputed_ci_low": ci_lo if ci_lo is not None else "",
                "recomputed_ci_high": ci_hi if ci_hi is not None else "",
                "recomputed_significant": sig if sig is not None else "",
                "verdict": verdict,
            })
            bucket = verdict.lower() if verdict in ("SUPPORTS", "CONTRADICTS", "NEUTRAL", "UNCLAIMED", "UNMEASURABLE", "SUPPORTS_NS", "PANEL_MISMATCH") else "unclaimed"
            if verdict in ("UNMEASURABLE", "PANEL_MISMATCH"):
                bucket = "unmeasurable"
            if bucket not in per_entry_rollup[did]:
                per_entry_rollup[did][bucket] = 0
            per_entry_rollup[did][bucket] += 1

    # ----- (D) Cross-check: do registry-stored measured[] match our recompute? -----
    stored_vs_recomputed = []
    for fp in delta_files:
        with open(fp) as f:
            d = json.load(f)
        did = d.get("id", os.path.basename(fp))
        method_name = did.replace("delta_", "")
        for m in d.get("measured", []) or []:
            met = m.get("metric", "?")
            panel = m.get("panel", "?")
            stored_delta = m.get("delta", None)
            stored_lo = m.get("ci_low", None)
            stored_hi = m.get("ci_high", None)
            rk = lookup.get((method_name, met))
            recomputed_delta = rk["delta_method_minus_base"] if (rk and rk["panel"] == panel) else None
            recomputed_lo = rk["ci_low"] if (rk and rk["panel"] == panel) else None
            recomputed_hi = rk["ci_high"] if (rk and rk["panel"] == panel) else None
            delta_diff = None
            if stored_delta is not None and recomputed_delta is not None:
                delta_diff = round(recomputed_delta - stored_delta, 6)
            stored_vs_recomputed.append({
                "delta_id": did,
                "metric": met,
                "panel": panel,
                "stored_delta": stored_delta,
                "recomputed_delta": recomputed_delta,
                "delta_diff": delta_diff,
                "stored_ci": [stored_lo, stored_hi],
                "recomputed_ci": [recomputed_lo, recomputed_hi],
                "in_CI_band": None if (stored_delta is None or recomputed_lo is None) else bool(recomputed_lo <= stored_delta <= recomputed_hi),
            })

    # ----- Write outputs -----
    out_pair = RES / "p6_iter190_recomputed_deltas.tsv"
    with open(out_pair, "w") as f:
        cols = ["metric", "panel", "base", "method", "n", "mean_base", "mean_method", "delta_method_minus_base", "ci_low", "ci_high", "significant", "data_source", "ci_method"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in per_pair_tsv:
            w.writerow(r)

    out_align = RES / "p6_iter190_expected_vs_recomputed.tsv"
    with open(out_align, "w") as f:
        cols = ["delta_id", "name", "method", "metric", "panel_expected", "predicted_sign", "panel_used", "recomputed_delta", "recomputed_ci_low", "recomputed_ci_high", "recomputed_significant", "verdict"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in per_alignment:
            w.writerow(r)

    out_compare = RES / "p6_iter190_stored_vs_recomputed.tsv"
    with open(out_compare, "w") as f:
        cols = ["delta_id", "metric", "panel", "stored_delta", "recomputed_delta", "delta_diff", "stored_ci", "recomputed_ci", "in_CI_band"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in stored_vs_recomputed:
            w.writerow(r)

    # Per-entry rollup
    out_rollup = RES / "p6_iter190_entry_rollup.tsv"
    with open(out_rollup, "w") as f:
        cols = ["delta_id", "expected_total", "measured_total", "supports", "supports_ns", "contradicts", "neutral", "unclaimed", "unmeasurable", "n_aligned"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for did, roll in per_entry_rollup.items():
            sup = roll.get("supports", 0) + roll.get("supports_ns", 0)
            w.writerow({
                "delta_id": did,
                "expected_total": roll.get("expected_total", 0),
                "measured_total": roll.get("measured_total", 0),
                "supports": roll.get("supports", 0),
                "supports_ns": roll.get("supports_ns", 0),
                "contradicts": roll.get("contradicts", 0),
                "neutral": roll.get("neutral", 0),
                "unclaimed": roll.get("unclaimed", 0),
                "unmeasurable": roll.get("unmeasurable", 0),
                "n_aligned": roll.get("expected_total", 0) - roll.get("unmeasurable", 0),
            })

    # Summary JSON
    n_align = len(per_alignment)
    n_supports = sum(1 for r in per_alignment if r["verdict"] == "SUPPORTS")
    n_supports_ns = sum(1 for r in per_alignment if r["verdict"] == "SUPPORTS_NS")
    n_contradicts = sum(1 for r in per_alignment if r["verdict"] == "CONTRADICTS")
    n_neutral = sum(1 for r in per_alignment if r["verdict"] == "NEUTRAL")
    n_unmeasurable = sum(1 for r in per_alignment if r["verdict"] in ("UNMEASURABLE", "PANEL_MISMATCH"))
    n_entries_with_expected = sum(1 for did, roll in per_entry_rollup.items() if roll.get("expected_total", 0) > 0)
    n_entries_fully_validated = sum(1 for did, roll in per_entry_rollup.items() if roll.get("expected_total", 0) > 0 and roll.get("unmeasurable", 0) == 0)
    n_drift_pairs = sum(1 for r in stored_vs_recomputed if r["delta_diff"] is not None and abs(r["delta_diff"]) > 0.05)
    n_stored = len(stored_vs_recomputed)
    summary = {
        "iter": 190,
        "pillar": "P6",
        "vein": "(a) registry measured-vs-claimed recompute audit (RAW data; iter-178 stored-value audit was derivative)",
        "n_recomputed_pairs": len(per_pair_tsv),
        "n_aligned_claims": n_align,
        "n_supports": n_supports,
        "n_supports_ns": n_supports_ns,
        "n_contradicts": n_contradicts,
        "n_neutral": n_neutral,
        "n_unmeasurable": n_unmeasurable,
        "n_entries_with_expected": n_entries_with_expected,
        "n_entries_fully_validated": n_entries_fully_validated,
        "n_stored_pairs": n_stored,
        "n_stored_drift_gt_0p05": n_drift_pairs,
        "supports_rate": round((n_supports + n_supports_ns) / max(1, n_align - n_unmeasurable), 4) if n_align > n_unmeasurable else None,
        "contradicts_rate": round(n_contradicts / max(1, n_align - n_unmeasurable), 4) if n_align > n_unmeasurable else None,
        "ci_method_n2": f"paired_step_bootstrap_pct n_boot={N_BOOT} seed={SEED}",
        "ci_method_zvf130": "welch_normal_approx_95pct n=5 (per-method aggregated sd)",
        "raw_data_sources": [
            "experiments/results/n2_reward_tensor_resume/n2_metrics.tsv",
            "experiments/results/zvf_iter130_method_risk.tsv",
        ],
    }
    out_sum = RES / "p6_iter190_summary.json"
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[iter190] wrote: {out_pair}")
    print(f"[iter190] wrote: {out_align}")
    print(f"[iter190] wrote: {out_compare}")
    print(f"[iter190] wrote: {out_rollup}")
    print(f"[iter190] wrote: {out_sum}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()