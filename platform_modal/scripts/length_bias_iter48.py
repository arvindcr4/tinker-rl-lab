"""Iter 48: Plateau-anchored length-bias slope decomposition.

Novel angle vs iter28/32/36/40/44 (all of which condition on R|L globally or on
ZVF bins without an explicit temporal anchor): iter48 anchors every per-run
trajectory at its OWN reward-plateau time t_plat, then compares the
length-drift rate dL/dt BEFORE vs AFTER the plateau.

Dr.GRPO signature prediction: in the rising phase [0, t_plat), both GRPO and
Dr.GRPO are length-compressing (the model is finding a concise solution). In
the plateau phase [t_plat, T], GRPO holds length flat (no signal), while
Dr.GRPO's flattened group baseline lets the policy drift in token count
without any reward pressure -> dL/dt_plateau should be MORE positive for
Dr.GRPO than for GRPO, especially on the GSM8k CoT task where length has
real headroom.

Mathematical setup:
  R_max = max_t R_t
  t_plat = first t with smoothed_R_t >= 0.9 * R_max   (smoothed = 3-step moving avg)
  rising  = L[0..t_plat)   with OLS slope = (dL/dt)_rise, signed as
            +ve = increasing length, -ve = decreasing length.
  plateau = L[t_plat..T)  with OLS slope = (dL/dt)_plat
  signature = (dL/dt)_plat - (dL/dt)_rise   (the marginal length tax incurred
             AFTER reward saturates)

We test whether signature_drgrpo > signature_grpo on both arithmetic_easy and
gsm8k_cot, by paired bootstrap across seeds.

Outputs (5 TSVs):
  platform_hybrid/experiments/results/length_bias_iter48_plateau.tsv        per (task, algo,
      seed): R_max, t_plat, R_at_plat, rising slope, plateau slope, signature
  platform_hybrid/experiments/results/length_bias_iter48_grpo_vs_drgrpo.tsv  paired bootstrap
      on (signature, rising slope, plateau slope)
  platform_hybrid/experiments/results/length_bias_iter48_summary.tsv          one-line rollup
      per task with mean/SD/CI for the paired comparison
  platform_hybrid/experiments/results/length_bias_iter48_rising_vs_plateau.tsv  within-algo
      paired: rising slope vs plateau slope per seed
  platform_hybrid/experiments/results/length_bias_iter48_zvf_anchored.tsv     per (task, algo,
      zvf_bin_around_plateau): the signature stratified by whether t_plat sits
      in a low/mid/high-ZVF step (cross-pillar with iter34 ZVF proxy)

Reads:
  platform_hybrid/experiments/results/drgrpo_vs_grpo.json   (arithmetic_easy: 5 seeds)
  platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json (gsm8k_cot: 3 seeds)
"""
from __future__ import annotations
import csv
import json
import math
import os
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
DRGRPO_JSON = RES / "drgrpo_vs_grpo.json"
GSM_JSON = RES / "drgrpo_gsm8k_cot_full.json"

RNG_SEED = 48
N_BOOT = 4000
PLATEAU_FRAC = 0.90        # smoothed_R_t >= 0.90 * R_max counts as plateau
SMOOTH_WIN = 3             # 3-step centered moving average
ZVF_BINS = ("low", "mid", "high")  # low<0.2, mid [0.2,0.5), high>=0.5


def _smooth(xs: list[float], win: int) -> list[float]:
    n = len(xs)
    if n == 0:
        return []
    out = []
    half = win // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(xs[lo:hi]) / max(1, hi - lo))
    return out


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """OLS slope & intercept on (xs, ys). Returns (0.0, 0.0) if degenerate."""
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


def _zvf_bin(zvf: float) -> str:
    if zvf < 0.2:
        return "low"
    if zvf < 0.5:
        return "mid"
    return "high"


def load_runs(path: Path, task: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d["runs"]:
        algo = r["algo"]
        seed = r["seed"]
        sl = r["step_log"]
        ts = [s["step"] for s in sl]
        rs = [float(s["mean_reward"]) for s in sl]
        ls = [float(s["mean_comp_len"]) for s in sl]
        zs = [float(s.get("zvf", float("nan"))) for s in sl]
        out.append({
            "task": task,
            "algo": algo,
            "seed": seed,
            "t": ts,
            "R": rs,
            "L": ls,
            "zvf": zs,
        })
    return out


def detect_plateau(rs: list[float]) -> tuple[int, float, float]:
    """Return (t_plat, R_at_plat, R_max). t_plat = first t with smoothed_R_t >=
    PLATEAU_FRAC * R_max. If never reached, t_plat = len(rs) - 1."""
    n = len(rs)
    if n == 0:
        return 0, 0.0, 0.0
    rmax = max(rs)
    if rmax <= 0:
        return n - 1, rs[-1], rmax
    sm = _smooth(rs, SMOOTH_WIN)
    threshold = PLATEAU_FRAC * rmax
    for i, v in enumerate(sm):
        if v >= threshold:
            return i, rs[i], rmax
    return n - 1, rs[-1], rmax


def per_run_signature(row: dict) -> dict:
    t = row["t"]; R = row["R"]; L = row["L"]; Z = row["zvf"]
    t_plat, r_at_plat, r_max = detect_plateau(R)
    # rising phase [0, t_plat]; plateau phase [t_plat, T)
    # always include at least one point on each side
    t1 = max(1, min(t_plat, len(t) - 1))
    t2 = max(t1 + 1, len(t))
    rising_t, rising_l = t[:t1], L[:t1]
    plat_t, plat_l = t[t1:t2], L[t1:t2]
    rise_slope, rise_int = _ols(rising_t, rising_l)
    plat_slope, plat_int = _ols(plat_t, plat_l)
    signature = plat_slope - rise_slope
    zvf_at_plat = Z[t_plat] if 0 <= t_plat < len(Z) else float("nan")
    return {
        "task": row["task"], "algo": row["algo"], "seed": row["seed"],
        "n_steps": len(t),
        "R_max": round(r_max, 6),
        "t_plat": t_plat,
        "R_at_plat": round(r_at_plat, 6),
        "zvf_at_plat": (round(zvf_at_plat, 6) if not math.isnan(zvf_at_plat) else "nan"),
        "rise_slope": round(rise_slope, 6),
        "rise_intercept": round(rise_int, 6),
        "plat_slope": round(plat_slope, 6),
        "plat_intercept": round(plat_int, 6),
        "signature": round(signature, 6),
        "zvf_bin_at_plat": _zvf_bin(zvf_at_plat) if not math.isnan(zvf_at_plat) else "nan",
    }


def paired_bootstrap(grpo_vals: list[float], drgrpo_vals: list[float],
                     n_boot: int = N_BOOT, rng: random.Random | None = None
                     ) -> dict:
    """Paired bootstrap on the per-seed difference drgrpo - grpo.

    Returns dict with mean_diff, sd_diff, ci_lo, ci_hi, p_le0, n_pairs.
    """
    rng = rng or random.Random(RNG_SEED)
    pairs = list(zip(grpo_vals, drgrpo_vals))
    diffs = [d - g for d, g in pairs]
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "sd_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "p_le0": 1.0, "n_pairs": 0}
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / max(1, n - 1)
    sd_diff = math.sqrt(var)
    boots = []
    idx = list(range(n))
    for _ in range(n_boot):
        sample = [diffs[rng.choice(idx)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    ci_lo = boots[int(0.025 * n_boot)]
    ci_hi = boots[int(0.975 * n_boot)]
    p_le0 = (sum(1 for b in boots if b <= 0) + 1) / (n_boot + 1)
    return {
        "mean_diff": round(mean_diff, 6),
        "sd_diff": round(sd_diff, 6),
        "ci_lo": round(ci_lo, 6),
        "ci_hi": round(ci_hi, 6),
        "p_le0": round(p_le0, 4),
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

    sig_rows = [per_run_signature(r) for r in rows]
    write_tsv(RES / "length_bias_iter48_plateau.tsv", sig_rows,
              fieldnames=list(sig_rows[0].keys()))

    # paired GRPO vs Dr.GRPO per task
    paired_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        sub = [r for r in sig_rows if r["task"] == task]
        grpo = [r for r in sub if r["algo"] == "grpo"]
        drgrpo = [r for r in sub if r["algo"] in ("dr_grpo", "drgrpo")]
        # align by seed
        grpo_by_seed = {r["seed"]: r for r in grpo}
        drgrpo_by_seed = {r["seed"]: r for r in drgrpo}
        common = sorted(set(grpo_by_seed) & set(drgrpo_by_seed))
        for metric in ("signature", "rise_slope", "plat_slope"):
            gv = [grpo_by_seed[s][metric] for s in common]
            dv = [drgrpo_by_seed[s][metric] for s in common]
            boot = paired_bootstrap(gv, dv, rng=rng)
            paired_rows.append({
                "task": task,
                "metric": metric,
                "n_pairs": boot["n_pairs"],
                "mean_grpo": round(sum(gv) / max(1, len(gv)), 6),
                "mean_drgrpo": round(sum(dv) / max(1, len(dv)), 6),
                "mean_diff": boot["mean_diff"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_le0": boot["p_le0"],
            })
    write_tsv(RES / "length_bias_iter48_grpo_vs_drgrpo.tsv", paired_rows,
              fieldnames=list(paired_rows[0].keys()))

    # within-algo: rising vs plateau slope per seed
    rp_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for algo in ("grpo", "dr_grpo"):
            sub = [r for r in sig_rows if r["task"] == task and r["algo"] == algo]
            for r in sub:
                rp_rows.append({
                    "task": task, "algo": algo, "seed": r["seed"],
                    "rise_slope": r["rise_slope"],
                    "plat_slope": r["plat_slope"],
                    "diff_plat_minus_rise": round(r["plat_slope"] - r["rise_slope"], 6),
                })
    write_tsv(RES / "length_bias_iter48_rising_vs_plateau.tsv", rp_rows,
              fieldnames=list(rp_rows[0].keys()))

    # zvf-anchored: bin the plateau step's ZVF and aggregate signature
    zvf_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for algo in ("grpo", "dr_grpo"):
            for zb in ZVF_BINS:
                sub = [r for r in sig_rows if r["task"] == task
                       and r["algo"] == algo and r["zvf_bin_at_plat"] == zb]
                if not sub:
                    continue
                sigs = [r["signature"] for r in sub]
                zvf_rows.append({
                    "task": task, "algo": algo, "zvf_bin_at_plat": zb,
                    "n_runs": len(sub),
                    "mean_signature": round(sum(sigs) / len(sigs), 6),
                    "sd_signature": round(statistics.pstdev(sigs) if len(sigs) > 1 else 0.0, 6),
                })
    write_tsv(RES / "length_bias_iter48_zvf_anchored.tsv", zvf_rows,
              fieldnames=list(zvf_rows[0].keys()))

    # one-line rollup
    summary_rows = []
    for pr in paired_rows:
        summary_rows.append({
            "task": pr["task"],
            "metric": pr["metric"],
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
    write_tsv(RES / "length_bias_iter48_summary.tsv", summary_rows,
              fieldnames=list(summary_rows[0].keys()))

    print("=== iter48 summary ===")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()