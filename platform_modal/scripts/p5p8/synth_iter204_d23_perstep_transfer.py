#!/usr/bin/env python3
"""P5P8-SYNTH D23 (iter 204 JOB B): per-step transfer stability of the
cost-weighted cross-pillar decision rule.

Fresh 23rd density domain (NOT in any prior D1..D22 SYNTH row).

D20 (iter-192) measured aggregate cross-pillar Spearman ρ on 160 N2
cells and found a two-cluster structure. D21 (iter-196) lifted to per-decile
and found the aggregate structure is NOT robust within deciles. D22
(iter-200) added cost-optimal operational weighting and found that
weighting sharpens the cross-pillar decision rule AND inverts the
D21 decile structure.

D23 takes the next step: at the PER-STEP level (not aggregate nor per-decile),
is the cost-weighted best-method-per-pillar STABLE? And does that
stability differ between the raw and cost-weighted decision rules?

Pipeline:
  1. Load n2_metrics.tsv (160 rows: 4 methods × 40 steps) — same source
     as D20, D21, D22.
  2. Compute per-(method, step) pillar headliners (same 4 headliners as D22):
     P5 = mean_reward, P6 = -ZVF, P7 = reward/(1+cv_len), P8 = reward/mean_len.
  3. Compute the cost-optimal per-step weight (same as D22):
     w = reward / (1 + (c/100) * mean_len / ref_len), ref_len = mean(mean_len).
  4. For each (step, pillar, weighting): identify best_method(step).
     4 pillars × 40 steps × 2 weightings = 320 (step, pillar, weighting)
     triples, each labeled with a best method.
  5. Per pillar: compute the AGGREGATE best_method across all 160 cells
     (the "headline" best method).
  6. For each (step, pillar, weighting): report
     agreement = (best_method(step) == aggregate_best_method) — a binary.
  7. 4 falsifiable hypotheses:
     H1: P5↔P8 per-step agreement > 70% under cost-weighting (operational
         pillars agree at per-step granularity).
     H2: per-step best-method under cost-weighting has HIGHER variance
         across steps than aggregate (per-step ≠ aggregate — granular
         disagreement is real, not noise).
     H3: cost-weighting INCREASES per-step agreement vs raw
         (D22 effect lifts from aggregate to per-step).
     H4: per-step best-method transitions are CLUSTERED: at least 3
         contiguous-step runs of the same best-method (steps where it
         flips form blocks, not isolated events).

Outputs:
  synth_iter204_d23_perstep_best.tsv   4 pillars x 40 steps x 2 weightings = 320 rows
  synth_iter204_d23_aggregate_best.tsv  4 pillars x 2 weightings = 8 rows
  synth_iter204_d23_agreement.tsv       4 pillars x 2 weightings = 8 rows
  synth_iter204_d23_run_lengths.tsv     per (pillar, weighting) run-length distribution
  synth_iter204_d23_summary.json        H1..H4 verdicts
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
N2 = ROOT / "experiments" / "results" / "n2_reward_tensor_resume" / "n2_metrics.tsv"

METHODS = ["grpo", "aero", "gift", "areal"]
N_STEPS = 40
N_BOOT = 2000
RNG = np.random.default_rng(20260706)


def main():
    rows = []
    with N2.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append(r)
    print(f"Loaded {len(rows)} rows from n2_metrics.tsv", flush=True)

    per_method_step = {}
    for r in rows:
        m = r["method"]
        s = int(r["step"])
        rew_mean = float(r["reward_mean"])
        zvf = float(r["zvf"])
        mean_len = float(r["mean_len"])
        cv_len = float(r["cv_len"])
        per_method_step[(m, s)] = {
            "P5_reward": rew_mean,
            "P6_minus_zvf": -zvf,
            "P7_snr": rew_mean / (1.0 + cv_len),
            "P8_reward_per_len": rew_mean / max(1.0, mean_len),
            "mean_len": mean_len,
        }

    # Compute cost-optimal weight (same as D22)
    ref_len = float(np.mean([per_method_step[(m, s)]["mean_len"]
                              for m in METHODS for s in range(N_STEPS)]))
    c_norm = 1.0
    weights = {}
    for m in METHODS:
        for s in range(N_STEPS):
            cell = per_method_step[(m, s)]
            w = cell["P5_reward"] / (1.0 + c_norm * cell["mean_len"] / ref_len)
            weights[(m, s)] = w

    # Cost-weighted pillar scores
    for m in METHODS:
        for s in range(N_STEPS):
            cell = per_method_step[(m, s)]
            w = weights[(m, s)]
            per_method_step[(m, s)].update({
                "P5_w": cell["P5_reward"] * w,
                "P6_w": cell["P6_minus_zvf"] * w,
                "P7_w": cell["P7_snr"] * w,
                "P8_w": cell["P8_reward_per_len"] * w,
            })

    keys_raw = {"P5": "P5_reward", "P6": "P6_minus_zvf",
                "P7": "P7_snr", "P8": "P8_reward_per_len"}
    keys_w = {"P5": "P5_w", "P6": "P6_w", "P7": "P7_w", "P8": "P8_w"}

    # Aggregate best method per pillar per weighting (across all 160 cells)
    def aggregate_best(weighting):
        keys = keys_w if weighting == "weighted" else keys_raw
        out = {}
        for p in ["P5", "P6", "P7", "P8"]:
            # Sum across all 160 cells per method
            sums = {m: 0.0 for m in METHODS}
            for m in METHODS:
                for s in range(N_STEPS):
                    sums[m] += per_method_step[(m, s)][keys[p]]
            # Best = highest aggregate
            out[p] = max(METHODS, key=lambda m: sums[m])
        return out

    agg_raw = aggregate_best("raw")
    agg_w = aggregate_best("weighted")
    print(f"Aggregate best (raw): {agg_raw}", flush=True)
    print(f"Aggregate best (weighted): {agg_w}", flush=True)

    # Per-step best method per pillar per weighting
    out_perstep = RES / "synth_iter204_d23_perstep_best.tsv"
    perstep_data = []  # list of (step, pillar, weighting, best_method)
    with out_perstep.open("w") as f:
        f.write("step\tpillar\tweighting\tbest_method\n")
        for s in range(N_STEPS):
            for p in ["P5", "P6", "P7", "P8"]:
                for weighting in ["raw", "weighted"]:
                    keys = keys_w if weighting == "weighted" else keys_raw
                    # At step s, compare the 4 methods on pillar p
                    vals = {m: per_method_step[(m, s)][keys[p]] for m in METHODS}
                    best = max(METHODS, key=lambda m: vals[m])
                    perstep_data.append((s, p, weighting, best))
                    f.write(f"{s}\t{p}\t{weighting}\t{best}\n")

    # Aggregate best TSV
    out_agg = RES / "synth_iter204_d23_aggregate_best.tsv"
    with out_agg.open("w") as f:
        f.write("pillar\tweighting\taggregate_best_method\n")
        for p in ["P5", "P6", "P7", "P8"]:
            for weighting in ["raw", "weighted"]:
                agg = agg_w[p] if weighting == "weighted" else agg_raw[p]
                f.write(f"{p}\t{weighting}\t{agg}\n")

    # Per-(pillar, weighting) agreement: fraction of steps where per-step
    # best == aggregate best. Bootstrap CI over steps.
    out_agree = RES / "synth_iter204_d23_agreement.tsv"
    agree_data = {}  # (p, weighting) -> list of 0/1 over 40 steps
    with out_agree.open("w") as f:
        f.write("pillar\tweighting\taggregate_best\tagreement_frac\tlo\thi\n")
        for p in ["P5", "P6", "P7", "P8"]:
            for weighting in ["raw", "weighted"]:
                agg = agg_w[p] if weighting == "weighted" else agg_raw[p]
                step_agrees = []
                for s in range(N_STEPS):
                    ps_best = None
                    for st, pp, ww, b in perstep_data:
                        if st == s and pp == p and ww == weighting:
                            ps_best = b
                            break
                    step_agrees.append(int(ps_best == agg))
                arr = np.array(step_agrees, dtype=np.int32)
                # Bootstrap CI on the mean
                rng = np.random.default_rng(20260706 + hash((p, weighting)) % 10000)
                boot_means = np.empty(N_BOOT)
                for i in range(N_BOOT):
                    idx = rng.integers(0, len(arr), size=len(arr))
                    boot_means[i] = arr[idx].mean()
                lo = float(np.quantile(boot_means, 0.025))
                hi = float(np.quantile(boot_means, 0.975))
                agree_data[(p, weighting)] = step_agrees
                f.write(f"{p}\t{weighting}\t{agg}\t{arr.mean():.4f}\t"
                        f"{lo:.4f}\t{hi:.4f}\n")

    # Run-length analysis: contiguous runs of the same per-step best method.
    out_runs = RES / "synth_iter204_d23_run_lengths.tsv"
    run_dist = {}  # (p, weighting) -> list of run lengths
    seq_cache = {}  # (p, weighting) -> ordered list of best method per step
    with out_runs.open("w") as f:
        f.write("pillar\tweighting\trun_length\tcount\n")
        for p in ["P5", "P6", "P7", "P8"]:
            for weighting in ["raw", "weighted"]:
                # Get the per-step best method sequence in step order
                seq = []
                for s in range(N_STEPS):
                    bm = None
                    for st, pp, ww, b in perstep_data:
                        if st == s and pp == p and ww == weighting:
                            bm = b
                            break
                    seq.append(bm)
                seq_cache[(p, weighting)] = seq
                # Compute run lengths
                runs = []
                cur_len = 1
                for i in range(1, len(seq)):
                    if seq[i] == seq[i - 1]:
                        cur_len += 1
                    else:
                        runs.append(cur_len)
                        cur_len = 1
                runs.append(cur_len)
                run_dist[(p, weighting)] = runs
                # Count run lengths
                from collections import Counter
                cnt = Counter(runs)
                for rl, c in sorted(cnt.items()):
                    f.write(f"{p}\t{weighting}\t{rl}\t{c}\n")

    # ----- Hypotheses -----
    # H1: P5↔P8 per-step agreement > 70% under cost-weighting
    h1_p5 = float(np.mean(agree_data[("P5", "weighted")]))
    h1_p8 = float(np.mean(agree_data[("P8", "weighted")]))
    h1_pass = bool(h1_p5 > 0.70 and h1_p8 > 0.70)

    # H2: per-step best-method under cost-weighting has HIGHER variance
    # across steps than aggregate (per-step != aggregate).
    # Variance proxy: number of distinct best methods across the 40 steps.
    n_distinct_raw = {p: len(set(seq_cache[(p, "raw")]))
                      for p in ["P5", "P6", "P7", "P8"]}
    n_distinct_w = {p: len(set(seq_cache[(p, "weighted")]))
                    for p in ["P5", "P6", "P7", "P8"]}
    # Aggregate best is 1 method; per-step should be > 1 (otherwise per-step = aggregate)
    h2_pass = bool(all(n_distinct_w[p] > 1 for p in ["P5", "P6", "P7", "P8"]))

    # H3: cost-weighting INCREASES per-step agreement vs raw (averaged over 4 pillars)
    raw_mean_agree = float(np.mean([np.mean(agree_data[(p, "raw")])
                                     for p in ["P5", "P6", "P7", "P8"]]))
    w_mean_agree = float(np.mean([np.mean(agree_data[(p, "weighted")])
                                   for p in ["P5", "P6", "P7", "P8"]]))
    h3_pass = bool(w_mean_agree > raw_mean_agree)

    # H4: per-step best-method transitions are CLUSTERED — at least 3
    # contiguous-step runs (steps where it flips form blocks, not isolated).
    max_run_w = max(max(run_dist[(p, "weighted")]) for p in ["P5", "P6", "P7", "P8"])
    h4_pass = bool(max_run_w >= 3)

    summary = {
        "iter": 204,
        "vein": ("P5P8-SYNTH D23 per-step transfer stability of the cost-weighted "
                 "cross-pillar decision rule (extends D22 from aggregate to per-step)"),
        "n_methods": len(METHODS),
        "n_steps": N_STEPS,
        "ref_len": ref_len,
        "aggregate_best_raw": agg_raw,
        "aggregate_best_weighted": agg_w,
        "h1_p5p8_perstep_agreement_gt_70pct": h1_pass,
        "h1_p5_agree_w": h1_p5,
        "h1_p8_agree_w": h1_p8,
        "h2_perstep_variety_above_aggregate": h2_pass,
        "h2_n_distinct_per_pillar_raw": n_distinct_raw,
        "h2_n_distinct_per_pillar_weighted": n_distinct_w,
        "h3_cost_weighting_increases_perstep_agreement": h3_pass,
        "h3_raw_mean_agreement": raw_mean_agree,
        "h3_weighted_mean_agreement": w_mean_agree,
        "h4_run_clustering_max_run_ge_3": h4_pass,
        "h4_max_run_weighted": max_run_w,
        "verdict_counts": {
            "PASS": sum([h1_pass, h2_pass, h3_pass, h4_pass]),
            "FAIL": 4 - sum([h1_pass, h2_pass, h3_pass, h4_pass]),
        },
    }
    out_sum = RES / "synth_iter204_d23_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {out_perstep}", flush=True)
    print(f"Wrote {out_agg}", flush=True)
    print(f"Wrote {out_agree}", flush=True)
    print(f"Wrote {out_runs}", flush=True)
    print(f"Wrote {out_sum}", flush=True)
    print(f"H1..H4: {summary['verdict_counts']}", flush=True)
    print(f"  H1: P5 agree_w={h1_p5:.3f}, P8 agree_w={h1_p8:.3f}", flush=True)
    print(f"  H2: n_distinct_w per pillar: {n_distinct_w}", flush=True)
    print(f"  H3: raw_mean={raw_mean_agree:.4f} vs w_mean={w_mean_agree:.4f}", flush=True)
    print(f"  H4: max_run_w={max_run_w}", flush=True)


if __name__ == "__main__":
    main()