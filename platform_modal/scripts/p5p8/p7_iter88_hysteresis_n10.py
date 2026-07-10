#!/usr/bin/env python3
"""P5P8-SYNTH JOB B (iter 88): P7 hysteresis × N10 5-seed extension.

Closes the iter-87 mint recommendation:
    "extend zvf-triage hysteresis to the N10 panel with 5 seeds to
     certify that the raw zvf-triage flip-flop hazard and the hysteresis
     Pareto-dominance result from N2 are not seed-specific artefacts."

This iter applies the iter-87 hysteresis filter (tau, K_up, K_dn) to the
5-seed N10 GRPO panel (n_grpo_s{42,179,316,453,590}.json, 15 steps each)
with only aggregate zvf per step (the per-prompt k-distribution is NOT
available for N10). Because the data is coarser than N2, we measure
step-level proxy outcomes:
   - "fires" = number of steps where zvf >= tau
   - "flips" = number of times the (escalated/idle) state changes
   - "yield_proxy" = sum of zvf over the trajectory (the integral;
     approximates contrast preserved as if all steps fed the same G)

Per-seed: compute (fires, flips, yield_proxy) at tau in {0.40, 0.50,
0.60, 0.70} with K_up=K_dn in {1, 2, 3} (=9 configs per seed + 1 raw
baseline = 50 total cells = 5 seeds x 10 configs).

Paired-seed bootstrap across the 5-seed panel: B=2000, seed=20260705,
on (flip-ratio per seed, yield-retention per seed).

Headline H1 (per-seed flip-flop hazard): is flip-flop hazard REAL on N10?
Headline H2 (per-seed Pareto): does hysteresis@K_up=K_dn=2 still Pareto-
dominate raw on (flip-ratio, yield-retention) at tau=0.50 on N10?
Headline H3 (cross-panel replication): does the per-seed flip-ratio at
K_up=K_dn=2 sit at the same level (22-33%) as iter-87's N2 finding?

Outputs
-------
experiments/results/p5p8/p7_iter88_hysteresis_n10_per_seed.tsv
  (5 seeds x 10 configs = 50 rows)
experiments/results/p5p8/p7_iter88_hysteresis_n10_boot.tsv
  (10 configs, paired-seed bootstrap on flip-ratio and yield-retention)
experiments/results/p5p8/p7_iter88_hysteresis_n10_summary.json
paper/sections/p7_iter88_hysteresis_n10.tex
docs/p5p8_improvements/105_p7_hysteresis_n10.md

Stdlib only. <= 250 lines.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
N10_DIR = ROOT / "experiments" / "results" / "n10_seed_expansion"

SEEDS = [42, 179, 316, 453, 590]
TAUS = [0.40, 0.50, 0.60, 0.70]
K_GRID = [(1, 1), (2, 2), (3, 3)]   # (K_up, K_dn)  (K_up=K_dn=1 = raw baseline reference but we keep all grid)
N_BOOT = 2000
BOOT_SEED = 20260705


def load_seed_zvf(seed: int) -> list[float]:
    """Read the step_log zvf trajectory for a given seed."""
    p = N10_DIR / f"n10_grpo_s{seed}.json"
    with p.open() as f:
        d = json.load(f)
    return [float(s["zvf"]) for s in d["step_log"]]


def hysteresis_trajectory(zvf: list[float], tau: float, k_up: int, k_dn: int) -> list[int]:
    """Return 0/1 elevated-state trajectory under hysteresis filter."""
    out = []
    state = 0  # 0 = idle, 1 = escalated
    cnt = 0    # consecutive count
    for z in zvf:
        if state == 0:
            # count steps with zvf >= tau
            if z >= tau:
                cnt += 1
                if cnt >= k_up:
                    state = 1
                    cnt = 0
            else:
                cnt = 0
        else:
            # count steps with zvf < tau
            if z < tau:
                cnt += 1
                if cnt >= k_dn:
                    state = 0
                    cnt = 0
            else:
                cnt = 0
        out.append(state)
    return out


def trajectory_stats(state: list[int], raw_state: list[int], zvf: list[float]) -> dict:
    """Compute (fires, flips, yield_proxy) for the trajectory."""
    n = len(state)
    fires = sum(1 for s in state if s == 1)
    flips = sum(1 for i in range(1, n) if state[i] != state[i - 1])
    raw_fires = sum(1 for s in raw_state if s == 1)
    raw_flips = sum(1 for i in range(1, n) if raw_state[i] != raw_state[i - 1])
    # yield proxy: mean zvf over the elevated-state steps (averaged into [0,1])
    yield_proxy = sum(zvf[i] for i in range(n) if state[i] == 1) / max(n, 1)
    raw_yield = sum(zvf[i] for i in range(n) if raw_state[i] == 1) / max(n, 1)
    return dict(
        fires=fires,
        flips=flips,
        yield_proxy=yield_proxy,
        raw_fires=raw_fires,
        raw_flips=raw_flips,
        raw_yield=raw_yield,
        flip_ratio=flips / max(raw_flips, 1),
        yield_retention=yield_proxy / max(raw_yield, 1e-9),
    )


def main() -> None:
    print("[iter88-hyst-N10] loading ...")
    zvf_per_seed = {s: load_seed_zvf(s) for s in SEEDS}
    for s, z in zvf_per_seed.items():
        print(f"  seed={s} n_steps={len(z)} mean_zvf={sum(z)/len(z):.3f}")

    # ---- Per-seed configs ----
    print("[iter88-hyst-N10] computing per-seed ...")
    rows = []
    for seed in SEEDS:
        zvf = zvf_per_seed[seed]
        # Raw baseline (K_up=K_dn=1) — apply for each tau
        for tau in TAUS:
            for (k_up, k_dn) in K_GRID:
                state = hysteresis_trajectory(zvf, tau, k_up, k_dn)
                # Raw = K_up=K_dn=1
                raw_state = hysteresis_trajectory(zvf, tau, 1, 1)
                if (k_up, k_dn) == (1, 1):
                    # Skip the baseline vs itself
                    continue
                stats = trajectory_stats(state, raw_state, zvf)
                rows.append(dict(
                    seed=seed,
                    tau=tau,
                    k_up=k_up, k_dn=k_dn,
                    fires=stats["fires"],
                    flips=stats["flips"],
                    raw_fires=stats["raw_fires"],
                    raw_flips=stats["raw_flips"],
                    yield_proxy=stats["yield_proxy"],
                    raw_yield=stats["raw_yield"],
                    flip_ratio=stats["flip_ratio"],
                    yield_retention=stats["yield_retention"],
                ))

    if rows:
        with (RES / "p7_iter88_hysteresis_n10_per_seed.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # ---- Paired bootstrap across seeds (per config) ----
    print("[iter88-hyst-N10] paired-seed bootstrap ...")
    cfg_keys = sorted({(r["tau"], r["k_up"], r["k_dn"]) for r in rows}, key=lambda x: (x[0], x[1], x[2]))
    rng = random.Random(BOOT_SEED)

    def per_seed_stat(key: tuple, stat_name: str) -> list[float]:
        return [r[stat_name] for r in rows if (r["tau"], r["k_up"], r["k_dn"]) == key]

    boot_rows = []
    for cfg in cfg_keys:
        # Each config has 5 seeds; resample 5 seeds WITH replacement
        frs = per_seed_stat(cfg, "flip_ratio")
        yrs = per_seed_stat(cfg, "yield_retention")
        n = len(frs)
        if n == 0:
            continue
        boot_fr = []
        boot_yr = []
        for _ in range(N_BOOT):
            idx = [rng.randrange(n) for _ in range(n)]
            boot_fr.append(sum(frs[i] for i in idx) / n)
            boot_yr.append(sum(yrs[i] for i in idx) / n)
        boot_fr.sort()
        boot_yr.sort()
        fr_lo = boot_fr[int(0.025 * N_BOOT)]
        fr_hi = boot_fr[int(0.975 * N_BOOT)]
        yr_lo = boot_yr[int(0.025 * N_BOOT)]
        yr_hi = boot_yr[int(0.975 * N_BOOT)]
        boot_rows.append(dict(
            tau=cfg[0], k_up=cfg[1], k_dn=cfg[2],
            flip_ratio_median=boot_fr[len(boot_fr) // 2],
            flip_ratio_lo=fr_lo, flip_ratio_hi=fr_hi,
            flip_ratio_excludes_1=(fr_hi < 1.0),
            yield_retention_median=boot_yr[len(boot_yr) // 2],
            yield_retention_lo=yr_lo, yield_retention_hi=yr_hi,
            yield_retention_excludes_1=(yr_hi < 1.0) or (yr_lo > 1.0),
        ))

    if boot_rows:
        with (RES / "p7_iter88_hysteresis_n10_boot.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()), delimiter="\t")
            w.writeheader()
            for r in boot_rows:
                w.writerow(r)

    # ---- Headlines ----
    headline_raw_flips = []
    for seed in SEEDS:
        zvf = zvf_per_seed[seed]
        for tau in TAUS:
            state = hysteresis_trajectory(zvf, tau, 1, 1)
            flips = sum(1 for i in range(1, len(state)) if state[i] != state[i - 1])
            fires = sum(state)
            headline_raw_flips.append(dict(seed=seed, tau=tau, fires_raw=fires, flips_raw=flips))

    # ---- Summary ----
    summary = {
        "seeds": SEEDS,
        "n_steps_per_seed": [len(zvf_per_seed[s]) for s in SEEDS],
        "tau_grid": TAUS,
        "k_grid": K_GRID,
        "n_boot": N_BOOT,
        "headline_raw_flips": headline_raw_flips,
        "boot": boot_rows,
        "n_per_seed_rows": len(rows),
    }
    with (RES / "p7_iter88_hysteresis_n10_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
