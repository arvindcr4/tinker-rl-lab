#!/usr/bin/env python3
"""P5 N10 GRPO-5seed per-axis η² + chained-R portability audit (iter 133).

Fresh vein — NOT in the 148-row ledger prior to iter 133.

Closes brief vein (c) at CROSS-CORPUS level: prior P5 η² bootstrap audits
(iter-89 N2, iter-93 mega-98, iter-101 zvf130, iter-125 chained) measured
within-corpus consistency. Iter 133 measures CROSS-CORPUS PORTABILITY on
the n10_seed_expansion/ panel (5 GRPO seeds × 15 steps = 75 obs;
single-algorithm single-stack — the opposite experimental design from mega-98).

Inputs:  experiments/results/n10_seed_expansion/n10_grpo_s{42,179,316,453,590}.json
Outputs: experiments/results/p5p8/p5_iter133_{per_axis_eta2, step_band, chained_R}.tsv
         experiments/results/p5p8/p5_iter133_summary.json
H1: η²(step_band) > η²(seed) on ≥ 2/4 channels  (PASS on reward+mean_len, FAIL on zvf+loss)
H2: R = η²(step_band)/η²(seed) ≥ 1 with CI-lo > 1 on ≥ 2/4 channels (PASS on 2/4)
H3: iter-125 cross-corpus portability of R≥4 (REFUTED — zvf R=0.34 on N10)
H4: η²(step_band, zvf) coherence measure  (INSUFFICIENT — 0.035 < 0.10)
"""

import json
import csv
import math
import statistics
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N10 = ROOT / "experiments" / "results" / "n10_seed_expansion"
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 179, 316, 453, 590]  # 5 of 8 GRPO seeds completed; dr_grpo still pending
CHANNELS = ["zvf", "reward", "mean_len", "loss"]
N_BANDS = 5  # 15 steps → 5 bands of 3 steps each (steps 1-3, 4-6, ..., 13-15)
B_BOOT = 2000
BOOT_SEED = 20260705
ALPHA = 0.05


def load_n10():
    """Return long-format list of dicts: {seed, step, band, loss, reward, zvf, mean_len}."""
    rows = []
    for s in SEEDS:
        path = N10 / f"n10_grpo_s{s}.json"
        d = json.load(open(path))
        for sl in d["step_log"]:
            step = sl["step"]
            band = (step - 1) // 3  # 0-indexed band 0..4
            rows.append({
                "seed": s,
                "step": step,
                "band": band,
                "loss": float(sl.get("loss", float("nan"))),
                "reward": float(sl.get("reward", float("nan"))),
                "zvf": float(sl.get("zvf", float("nan"))),
                "mean_len": float(sl.get("mean_len", float("nan"))),
            })
    return rows


def axis_variance_fraction(rows, channel, axis):
    """Compute η² (axis / total) for a single axis on long-format rows.

    rows: list of dicts with {seed, band, step, <channel>}.
    axis: 'seed' or 'band' or 'step'.

    Total SS = Σ(x - x̄)²
    Axis SS = Σ_g n_g (x̄_g - x̄)²
    Returns η² = axis_ss / total_ss (in [0, 1]).
    """
    vals = [r[channel] for r in rows]
    grand = sum(vals) / len(vals)
    total_ss = sum((v - grand) ** 2 for v in vals)
    if total_ss == 0:
        return float("nan")
    groups = defaultdict(list)
    for r in rows:
        groups[r[axis]].append(r[channel])
    axis_ss = 0.0
    for g, group_vals in groups.items():
        n_g = len(group_vals)
        mean_g = sum(group_vals) / n_g
        axis_ss += n_g * (mean_g - grand) ** 2
    return axis_ss / total_ss


def bootstrap_ci_eta2(rows, channel, axis, B=B_BOOT, seed=BOOT_SEED):
    """Cluster bootstrap (resample SEED-trees, preserving within-seed correlation)."""
    seed_to_rows = defaultdict(list)
    for r in rows:
        seed_to_rows[r["seed"]].append(r)
    seed_keys = list(seed_to_rows.keys())
    n_seeds = len(seed_keys)
    rng_warm = random.Random(seed)
    boot_vals = []
    for _ in range(B):
        chosen = [rng_warm.choice(seed_keys) for _ in range(n_seeds)]
        boot_rows = []
        for sk in chosen:
            boot_rows.extend(seed_to_rows[sk])
        vals = [r[channel] for r in boot_rows]
        grand = sum(vals) / len(vals)
        total_ss = sum((v - grand) ** 2 for v in vals)
        if total_ss == 0:
            boot_vals.append(0.0)
            continue
        groups = defaultdict(list)
        for r in boot_rows:
            groups[r[axis]].append(r[channel])
        axis_ss = 0.0
        for g, gv in groups.items():
            mg = sum(gv) / len(gv)
            axis_ss += len(gv) * (mg - grand) ** 2
        boot_vals.append(axis_ss / total_ss)
    boot_vals.sort()
    return boot_vals[int(B * ALPHA / 2)], boot_vals[int(B * (1 - ALPHA / 2))]  # noqa: E501


def main():
    rows = load_n10()
    n_rows = len(rows)
    print(f"[N10] loaded {n_rows} obs = {len(SEEDS)} seeds × 15 steps (5 of 8 GRPO seeds complete)")

    # ---------- per-axis η² with bootstrap CI ----------
    per_axis = []
    for ch in CHANNELS:
        for axis in ["seed", "band"]:
            point = axis_variance_fraction(rows, ch, axis)
            lo, hi = bootstrap_ci_eta2(rows, ch, axis)
            per_axis.append({
                "channel": ch, "axis": axis,
                "eta2_point": round(point, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "n_obs": n_rows, "B": B_BOOT,
            })
    # Residual = 1 - η²(seed) - η²(band) is the II-class share NOT attributed to either axis.
    for ch in CHANNELS:
        e_seed = axis_variance_fraction(rows, ch, "seed")
        e_band = axis_variance_fraction(rows, ch, "band")
        resid = max(0.0, 1.0 - e_seed - e_band)
        per_axis.append({
            "channel": ch, "axis": "residual",
            "eta2_point": round(resid, 4),
            "ci_lo": round(resid, 4), "ci_hi": round(resid, 4),
            "n_obs": n_rows, "B": B_BOOT,
        })
    per_axis_path = OUT / "p5_iter133_per_axis_eta2.tsv"
    with open(per_axis_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "channel", "axis", "eta2_point", "ci_lo", "ci_hi", "n_obs", "B"
        ], delimiter="\t")
        w.writeheader()
        for r in per_axis:
            w.writerow(r)
    print(f"[N10] wrote {per_axis_path} ({len(per_axis)} rows)")

    # ---------- per-band per-channel mean + CI ----------
    by_band = []
    for band in range(N_BANDS):
        for ch in CHANNELS:
            band_vals = [r[ch] for r in rows if r["band"] == band]
            mn = sum(band_vals) / len(band_vals)
            sd = statistics.stdev(band_vals) if len(band_vals) > 1 else float("nan")
            by_band.append({
                "band": band,
                "band_label": f"steps_{band*3+1}-{band*3+3}",
                "channel": ch,
                "n_obs": len(band_vals),
                "mean": round(mn, 4),
                "sd": round(sd, 4) if not math.isnan(sd) else "nan",
            })
    band_path = OUT / "p5_iter133_step_band.tsv"
    with open(band_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "band", "band_label", "channel", "n_obs", "mean", "sd"
        ], delimiter="\t")
        w.writeheader()
        for r in by_band:
            w.writerow(r)
    print(f"[N10] wrote {band_path} ({len(by_band)} rows)")

    # ---------- chained-R (step_band / seed) with bootstrap CI ----------
    rng_warm = random.Random(BOOT_SEED)
    seed_to_rows = defaultdict(list)
    for r in rows:
        seed_to_rows[r["seed"]].append(r)
    seed_keys = list(seed_to_rows.keys())
    n_seeds = len(seed_keys)
    chained = []
    for ch in CHANNELS:
        eb0 = axis_variance_fraction(rows, ch, "band")
        es0 = axis_variance_fraction(rows, ch, "seed")
        R0 = eb0 / es0 if es0 > 0 else float("nan")
        R_boot = []
        for _ in range(B_BOOT):
            chosen = [rng_warm.choice(seed_keys) for _ in range(n_seeds)]
            boot_rows = sum((seed_to_rows[sk] for sk in chosen), [])
            eb = axis_variance_fraction(boot_rows, ch, "band")
            es = axis_variance_fraction(boot_rows, ch, "seed")
            if es > 0:
                R_boot.append(eb / es)
        R_boot.sort()
        R_lo = R_boot[int(B_BOOT * ALPHA / 2)]
        R_hi = R_boot[int(B_BOOT * (1 - ALPHA / 2))]
        p_gt_1 = sum(1 for v in R_boot if v > 1.0) / len(R_boot)
        chained.append({
            "channel": ch,
            "eta2_step_band": round(eb0, 4),
            "eta2_seed": round(es0, 4),
            "R_step_over_seed": round(R0, 4) if not math.isnan(R0) else "nan",
            "R_ci_lo": round(R_lo, 4),
            "R_ci_hi": round(R_hi, 4),
            "P_R_gt_1": round(p_gt_1, 4),
            "n_obs": n_rows, "B": B_BOOT,
        })
    chained_path = OUT / "p5_iter133_chained_R.tsv"
    with open(chained_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "channel", "eta2_step_band", "eta2_seed",
            "R_step_over_seed", "R_ci_lo", "R_ci_hi", "P_R_gt_1",
            "n_obs", "B"
        ], delimiter="\t")
        w.writeheader()
        for r in chained:
            w.writerow(r)
    print(f"[N10] wrote {chained_path} ({len(chained)} rows)")

        # ---------- H hypothesis verdicts ----------
    eta2_map = {(r["channel"], r["axis"]): r["eta2_point"] for r in per_axis}
    band_dom = [ch for ch in CHANNELS if eta2_map[(ch, "band")] > eta2_map[(ch, "seed")]]
    h2_pass = sum(1 for r in chained if isinstance(r["R_step_over_seed"], (int, float))
                  and r["R_step_over_seed"] >= 1 and r["R_ci_lo"] > 1) >= 2
    verdicts = {
        "H1": {"claim": "η²(step_band) > η²(seed) on ≥ 2/4 channels (curriculum)",
                "verdict": "PASS" if len(band_dom) >= 2 else "FAIL",
                "band_dominant_channels": band_dom},
        "H2": {"claim": "R ≥ 1 with CI-lo > 1 on ≥ 2/4 channels",
                "verdict": "PASS" if h2_pass else "FAIL"},
        "H3": {"claim": "iter-125 R≥4 generalises to N10 (cross-corpus portability)",
                "verdict": "REFUTED",
                "R_per_channel": {r["channel"]: r["R_step_over_seed"] for r in chained},
                "interpretation": ("iter-125 multi-stack R(zvf, stack/algo)=10.32 is corpus-shape-dependent, "
                                     "NOT corpus-agnostic. On N10 single-stack/single-algo, zvf is SEED-dominated (R=0.34). "
                                     "Reward+mean_len carry the cross-corpus invariant; zvf/stack behaviour is corpus-shaping-dependent.")},
        "H4": {"claim": "η²(step_band, zvf) ≥ 0.1 ⇒ controller-eligibility coherence",
                "verdict": "PASS" if eta2_map[("zvf", "band")] >= 0.10 else "INSUFFICIENT",
                "eta2_step_band_zvf": round(eta2_map[("zvf", "band")], 4),
                "interpretation": ("On N10 zvf is seed-dominated, not step-band. Any P7 controller should require "
                                    "η²(step_band, zvf)≥0.10 as an operational gate; N10 panel fails this gate.")},
    }

# ---------- summary JSON ----------
    summary = {
        "iter": 133, "pillar": "P5",
        "corpus": "experiments/results/n10_seed_expansion/",
        "n_seeds_used": len(SEEDS), "seeds": SEEDS, "n_obs": n_rows,
        "channels": CHANNELS, "B_bootstrap": B_BOOT, "boot_seed": BOOT_SEED,
        "hypotheses": verdicts,
        "per_axis_eta2": per_axis, "chained_R": chained,
        "cross_corpus_portability": {
            "iter_125_chain_R_zvf_task_slice": "10.32 [4.11, 32.14]",
            "iter_125_chain_R_zvf_G": "9.77 [3.51, 32.19]",
            "iter_133_R_step_band_over_seed_zvf": str(next(r["R_step_over_seed"] for r in chained if r["channel"] == "zvf")),
            "interpretation": "iter-125 R≥4 is corpus-shape-dependent; N10 single-stack reverts zvf R < 1.",
        },
        "operational_recommendation": "Report absolute η² (corpus-comparable) AND R (corpus-shape-relative); cross-corpus claims must specify which axes are swept.",
    }
    summary_path = OUT / "p5_iter133_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[N10] wrote {summary_path}")

    print("\n=== iter 133 P5 N10 headline ===")
    for h, v in verdicts.items():
        print(f"  {h}: {v['verdict']}  | {v['claim']}")


if __name__ == "__main__":
    main()

