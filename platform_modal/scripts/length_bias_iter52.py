"""Iter 52: Correctness-Regime-Conditional Length-Bias Decomposition.

Novel angle vs iter28/32/36/40/44/48: instead of binning by fixed R thresholds
(which fails for bimodal data: arithmetic_easy sits in HIGH regime, gsm8k_cot
in LOW/MID), iter52 splits each trajectory at the per-task MEDIAN R. This
gives both tasks a meaningful above-median ("policy succeeding") vs
below-median ("policy failing") split with comparable sample sizes per regime.

Mechanism (Dr.GRPO signature): GRPO's per-prompt advantage is
(R_x - mean(R_x)) / std(R_x). Dr.GRPO drops the /std term, and combined
with the lack of /length normalization, lets length drift UP more than
GRPO when the policy is succeeding (above-median R) because the contrast
signal R_x - mean(R_x) loses sharpness (small std). When the policy is
failing (below-median R), group means and stds both disperse, so both
GRPO and Dr.GRPO get strong signal and behave similarly.

Testable prediction: in the above-median regime on gsm8k_cot (where length
budget is large), dL/dt is more positive for Dr.GRPO than for GRPO.

Cross-pillar with P2 (ZVF): the below-median regime typically overlaps the
high-ZVF regime (when groups collapse to all-zero, ZVF spikes). iter52
also decomposes the above-median regime slope by the ZVF of the surrounding
steps, to test whether the above-median length-drift signature is mediated
by ZVF collapse.

Outputs (5 TSVs):
  platform_hybrid/experiments/results/length_bias_iter52_regime_slopes.tsv
    per (task, algo, seed, regime): n_steps, dL/dt, R_mean, L_mean, ZVF_mean
  platform_hybrid/experiments/results/length_bias_iter52_grpo_vs_drgrpo.tsv
    paired bootstrap on dL/dt per (task, regime)
  platform_hybrid/experiments/results/length_bias_iter52_above_minus_below.tsv
    paired bootstrap on (above slope - below slope) per (task, algo)
  platform_hybrid/experiments/results/length_bias_iter52_zvf_coupling.tsv
    per (task, algo, regime): dL/dt further split by zvf_bin (low/mid/high)
  platform_hybrid/experiments/results/length_bias_iter52_summary.tsv
    one-line rollup per (task, regime) with mean/SD/CI/interpretation

Reads:
  platform_hybrid/experiments/results/drgrpo_vs_grpo.json      (arithmetic_easy: 5 seeds)
  platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json (gsm8k_cot: 3 seeds)
"""
from __future__ import annotations
import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
DRGRPO_JSON = RES / "drgrpo_vs_grpo.json"
GSM_JSON = RES / "drgrpo_gsm8k_cot_full.json"

RNG_SEED = 52
N_BOOT = 4000

# Regime labels
REGIMES = ("below", "above")  # below-median R, above-median R

# ZVF bins for the cross-pillar stratification
ZVF_BINS = [("zvf_low", 0.0, 0.25), ("zvf_mid", 0.25, 0.5), ("zvf_high", 0.5, 1.01)]


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """OLS slope & intercept on (xs, ys); returns (0.0, 0.0) if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx
    intercept = my - slope * mx
    return slope, intercept


def _median(xs: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    s = sorted(xs)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _zvf_bin(z: float) -> str:
    for name, lo, hi in ZVF_BINS:
        if lo <= z < hi:
            return name
    return "zvf_high"


def load_runs(path: Path, task: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d["runs"]:
        algo = r["algo"]
        seed = r["seed"]
        sl = r["step_log"]
        ts = [int(s["step"]) for s in sl]
        rs = [float(s["mean_reward"]) for s in sl]
        ls = [float(s["mean_comp_len"]) for s in sl]
        zs = [float(s.get("zvf", float("nan"))) for s in sl]
        out.append({
            "task": task, "algo": algo, "seed": seed,
            "t": ts, "R": rs, "L": ls, "zvf": zs,
        })
    return out


def per_run_regime_slopes(row: dict) -> list[dict]:
    t = row["t"]; R = row["R"]; L = row["L"]; Z = row["zvf"]
    med = _median(R)
    out = []
    for regime in REGIMES:
        if regime == "below":
            idx = [i for i, r in enumerate(R) if r < med]
        else:
            idx = [i for i, r in enumerate(R) if r >= med]
        if len(idx) < 2:
            out.append({
                "task": row["task"], "algo": row["algo"], "seed": row["seed"],
                "regime": regime, "median_R": round(med, 6),
                "n_steps": len(idx),
                "slope_dL_dt": "nan", "intercept": "nan",
                "R_mean": "nan", "L_mean": "nan", "ZVF_mean": "nan",
                "L_first": "nan", "L_last": "nan",
            })
            continue
        tt = [t[i] for i in idx]
        ll = [L[i] for i in idx]
        rr = [R[i] for i in idx]
        zz = [Z[i] for i in idx]
        slope, intercept = _ols(tt, ll)
        out.append({
            "task": row["task"], "algo": row["algo"], "seed": row["seed"],
            "regime": regime, "median_R": round(med, 6),
            "n_steps": len(idx),
            "slope_dL_dt": round(slope, 6),
            "intercept": round(intercept, 6),
            "R_mean": round(sum(rr) / len(rr), 6),
            "L_mean": round(sum(ll) / len(ll), 6),
            "ZVF_mean": round(sum(zz) / max(1, len(zz)), 6),
            "L_first": round(ll[0], 6),
            "L_last": round(ll[-1], 6),
        })
    return out


def paired_bootstrap(g: list[float], d: list[float], n_boot: int = N_BOOT,
                     rng: random.Random | None = None) -> dict:
    rng = rng or random.Random(RNG_SEED)
    diffs = [di - gi for gi, di in zip(g, d)]
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "sd_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "p_le0": 1.0, "n_pairs": 0}
    mean_diff = sum(diffs) / n
    var = sum((x - mean_diff) ** 2 for x in diffs) / max(1, n - 1)
    sd_diff = math.sqrt(var)
    idx = list(range(n))
    boots = []
    for _ in range(n_boot):
        s = [diffs[rng.choice(idx)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {
        "mean_diff": round(mean_diff, 6),
        "sd_diff": round(sd_diff, 6),
        "ci_lo": round(boots[int(0.025 * n_boot)], 6),
        "ci_hi": round(boots[int(0.975 * n_boot)], 6),
        "p_le0": round((sum(1 for b in boots if b <= 0) + 1) / (n_boot + 1), 4),
        "n_pairs": n,
    }


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                           delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    random.seed(RNG_SEED)
    rng = random.Random(RNG_SEED)

    rows: list[dict] = []
    rows.extend(load_runs(DRGRPO_JSON, "arithmetic_easy"))
    rows.extend(load_runs(GSM_JSON, "gsm8k_cot"))

    # ---- 1. Per-(task, algo, seed, regime) dL/dt
    slope_rows: list[dict] = []
    for r in rows:
        slope_rows.extend(per_run_regime_slopes(r))
    write_tsv(RES / "length_bias_iter52_regime_slopes.tsv", slope_rows,
              fieldnames=list(slope_rows[0].keys()))

    # ---- 2. Paired GRPO vs Dr.GRPO bootstrap on slope per (task, regime)
    paired_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for regime in REGIMES:
            sub = [r for r in slope_rows if r["task"] == task
                   and r["regime"] == regime
                   and isinstance(r["slope_dL_dt"], (int, float))]
            grpo_by_seed = {r["seed"]: r for r in sub if r["algo"] == "grpo"}
            drgrpo_by_seed = {r["seed"]: r for r in sub if r["algo"] in ("dr_grpo", "drgrpo")}
            common = sorted(set(grpo_by_seed) & set(drgrpo_by_seed))
            if not common:
                continue
            gv = [grpo_by_seed[s]["slope_dL_dt"] for s in common]
            dv = [drgrpo_by_seed[s]["slope_dL_dt"] for s in common]
            boot = paired_bootstrap(gv, dv, rng=rng)
            paired_rows.append({
                "task": task, "regime": regime, "n_pairs": boot["n_pairs"],
                "mean_grpo": round(sum(gv) / len(gv), 6),
                "mean_drgrpo": round(sum(dv) / len(dv), 6),
                "mean_diff": boot["mean_diff"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_le0": boot["p_le0"],
            })
    write_tsv(RES / "length_bias_iter52_grpo_vs_drgrpo.tsv", paired_rows,
              fieldnames=list(paired_rows[0].keys()))

    # ---- 3. above - below slope per (task, algo): does the drift gradient
    #         across regimes differ?
    hl_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for algo in ("grpo", "dr_grpo"):
            sub = [r for r in slope_rows if r["task"] == task
                   and r["algo"] == algo
                   and isinstance(r["slope_dL_dt"], (int, float))]
            by_seed = {}
            for r in sub:
                by_seed.setdefault(r["seed"], {})[r["regime"]] = r["slope_dL_dt"]
            for seed, d in by_seed.items():
                if "above" in d and "below" in d:
                    hl_rows.append({
                        "task": task, "algo": algo, "seed": seed,
                        "slope_below": d["below"],
                        "slope_above": d["above"],
                        "diff_above_minus_below": round(d["above"] - d["below"], 6),
                    })
    # paired bootstrap on diff_above_minus_below per (task, algo)
    hl_paired = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for algo in ("grpo", "dr_grpo"):
            vals = [r["diff_above_minus_below"] for r in hl_rows
                    if r["task"] == task and r["algo"] == algo]
            if not vals:
                continue
            mean_v = sum(vals) / len(vals)
            sd_v = (statistics.pstdev(vals) if len(vals) > 1 else 0.0)
            boots = []
            idx = list(range(len(vals)))
            for _ in range(N_BOOT):
                s = [vals[rng.choice(idx)] for _ in range(len(vals))]
                boots.append(sum(s) / len(s))
            boots.sort()
            hl_paired.append({
                "task": task, "algo": algo, "n_seeds": len(vals),
                "mean_above_minus_below": round(mean_v, 6),
                "sd": round(sd_v, 6),
                "ci_lo": round(boots[int(0.025 * N_BOOT)], 6),
                "ci_hi": round(boots[int(0.975 * N_BOOT)], 6),
            })
    if hl_paired:
        write_tsv(RES / "length_bias_iter52_above_minus_below.tsv", hl_paired,
                  fieldnames=list(hl_paired[0].keys()))
    else:
        # write empty
        write_tsv(RES / "length_bias_iter52_above_minus_below.tsv", [],
                  fieldnames=["task", "algo", "n_seeds", "mean_above_minus_below",
                              "sd", "ci_lo", "ci_hi"])

    # ---- 4. ZVF coupling: per-(task, algo, regime, zvf_bin) dL/dt
    zvf_rows = []
    for row in rows:
        t = row["t"]; R = row["R"]; L = row["L"]; Z = row["zvf"]
        med = _median(R)
        for regime in REGIMES:
            for zname, _, _ in ZVF_BINS:
                if regime == "below":
                    idx = [i for i in range(len(R))
                           if R[i] < med and _zvf_bin(Z[i]) == zname]
                else:
                    idx = [i for i in range(len(R))
                           if R[i] >= med and _zvf_bin(Z[i]) == zname]
                if len(idx) < 2:
                    continue
                tt = [t[i] for i in idx]
                ll = [L[i] for i in idx]
                slope, intercept = _ols(tt, ll)
                zvf_rows.append({
                    "task": row["task"], "algo": row["algo"],
                    "seed": row["seed"], "regime": regime, "zvf_bin": zname,
                    "n_steps": len(idx),
                    "slope_dL_dt": round(slope, 6),
                    "intercept": round(intercept, 6),
                    "R_mean": round(sum(R[i] for i in idx) / len(idx), 6),
                    "L_mean": round(sum(L[i] for i in idx) / len(idx), 6),
                })
    if zvf_rows:
        write_tsv(RES / "length_bias_iter52_zvf_coupling.tsv", zvf_rows,
                  fieldnames=list(zvf_rows[0].keys()))
    else:
        write_tsv(RES / "length_bias_iter52_zvf_coupling.tsv", [],
                  fieldnames=["task", "algo", "seed", "regime", "zvf_bin",
                              "n_steps", "slope_dL_dt", "intercept",
                              "R_mean", "L_mean"])

    # ---- 5. One-line rollup per (task, regime)
    summary_rows = []
    for pr in paired_rows:
        summary_rows.append({
            "task": pr["task"],
            "regime": pr["regime"],
            "n_pairs": pr["n_pairs"],
            "mean_grpo": pr["mean_grpo"],
            "mean_drgrpo": pr["mean_drgrpo"],
            "mean_diff_drgrpo_minus_grpo": pr["mean_diff"],
            "ci_lo": pr["ci_lo"],
            "ci_hi": pr["ci_hi"],
            "p_le0": pr["p_le0"],
            "interpretation": (
                "Dr.GRPO > GRPO" if pr["mean_diff"] > 0 and pr["ci_lo"] > 0
                else "GRPO > Dr.GRPO" if pr["mean_diff"] < 0 and pr["ci_hi"] < 0
                else "inconclusive"
            ),
        })
    write_tsv(RES / "length_bias_iter52_summary.tsv", summary_rows,
              fieldnames=list(summary_rows[0].keys()))

    print("=== iter52 regime-conditional summary ===")
    for r in summary_rows:
        print(r)
    print("\n=== iter52 above-below diff per (task, algo) ===")
    for r in hl_paired:
        print(r)


if __name__ == "__main__":
    main()