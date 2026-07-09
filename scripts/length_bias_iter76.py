"""length_bias_iter76.py — Iter 76 Pillar 4 (Length Bias / Dr.GRPO).

Reward-shock mean-reversion and phase-plane dissipativity. Iter68 showed
Dr.GRPO reverses ΔL sign 14 pp more on reward-up steps; iter72 measured
AR(1) persistence. This iter asks the CONTINUOUS, MECHANICAL analogue.

Three paired diagnostics on the (L, R) trajectory:
  1. Half-life τ of |L_t - L_baseline| after a reward shock |ΔR| > q75.
  2. Damping ratio ζ from a 2nd-order oscillator fit (ζ > 1 overdamped,
     ζ < 1 underdamped).
  3. Phase-plane loop area ∮ R·dL via Shoelace on (L, R).

Outputs:
  length_bias_iter76_{halflife,damping,looparea,summary}.tsv
  length_bias_iter76_meta.json
Headline: GSM8K CoT (3 seeds × 30 steps) and arithmetic-easy (5 × 40) NC.
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSM_PATH = os.path.join(ROOT, "experiments", "results", "drgrpo_gsm8k_cot_full.json")
ARITH_PATH = os.path.join(ROOT, "experiments", "results", "drgrpo_vs_grpo.json")
OUT_DIR = os.path.join(ROOT, "experiments", "results")

H_HALFLIFE = 10  # horizon for half-life fit
H_OSC = 12  # horizon for oscillator fit
Q_SHOCK = 0.75  # top-quartile threshold for |dR| shocks
B_BOOT = 2000
SEED_BOOT = 0xC0FFEE


def load_runs(path: str, task: str) -> List[dict]:
    """Return a list of (algo, seed, L, R) tuples extracted from the JSON."""
    d = json.load(open(path))
    out = []
    for r in d["runs"]:
        L = [s["mean_comp_len"] for s in r["step_log"]]
        R = [s["mean_reward"] for s in r["step_log"]]
        out.append({"task": task, "algo": r["algo"], "seed": r["seed"],
                    "L": L, "R": R})
    return out


def diffs(series: List[float]) -> List[float]:
    return [series[i] - series[i - 1] for i in range(1, len(series))]


def half_life_post_shock(L, dR, H=H_HALFLIFE):
    """Fit |L_{t+h} - L_baseline| = A exp(-ln2 h/τ) for each shock step."""
    if len(L) < H + 5:
        return {"tau_all": math.nan, "tau_up": math.nan, "tau_dn": math.nan,
                "n_shocks": 0, "n_up": 0, "n_dn": 0}
    abs_dr = [abs(x) for x in dR]
    sorted_abs = sorted(abs_dr)
    n = len(sorted_abs)
    q_idx = int(Q_SHOCK * n)
    threshold = sorted_abs[q_idx]
    taus_up, taus_dn, taus_all = [], [], []
    n_up = n_dn = 0
    for t in range(len(dR) - H):
        if abs_dr[t] < threshold:
            continue
        if t + H >= len(L):
            break
        window = L[t + 1: t + H + 1]
        baseline = sum(window) / len(window)
        devs = [abs(window[h] - baseline) for h in range(H)]
        A0 = max(devs[0], 1e-6)
        if A0 < 0.5:
            # shock too small to estimate — skip
            continue
        # fit τ via log-linear regression: log(devs[h]) = log(A) - (ln2/τ) * h
        # use all h where dev > 0.1 * A0 (well-defined region)
        xs, ys = [], []
        for h in range(H):
            if devs[h] > 0.1 * A0:
                xs.append(h)
                ys.append(math.log(devs[h]))
        if len(xs) < 3:
            continue
        n_pts = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n_pts * sxx - sx * sx
        if abs(denom) < 1e-9:
            continue
        slope = (n_pts * sxy - sx * sy) / denom  # negative for decay
        if slope >= 0:
            # not decaying — set τ = H (max)
            tau = float(H)
        else:
            tau = -math.log(2) / slope
            # clamp to [0.5, 4*H]
            tau = min(max(tau, 0.5), 4 * H)
        taus_all.append(tau)
        if dR[t] > 0:
            taus_up.append(tau)
            n_up += 1
        elif dR[t] < 0:
            taus_dn.append(tau)
            n_dn += 1

    def m(xs):
        return sum(xs) / len(xs) if xs else math.nan

    return {"tau_all": m(taus_all), "tau_up": m(taus_up),
            "tau_dn": m(taus_dn), "n_shocks": len(taus_all),
            "n_up": n_up, "n_dn": n_dn}


def damping_ratio(L, dR, H=H_OSC):
    """Fit 2nd-order AR to post-shock L; derive ζ from root discriminant."""
    if len(L) < H + 5:
        return {"zeta_all": math.nan, "zeta_up": math.nan,
                "zeta_dn": math.nan, "n_shocks": 0, "n_up": 0, "n_dn": 0}
    abs_dr = [abs(x) for x in dR]
    threshold = sorted(abs_dr)[int(Q_SHOCK * len(abs_dr))]
    zs_up, zs_dn, zs_all = [], [], []
    n_up = n_dn = 0
    for t in range(len(dR) - H):
        if abs_dr[t] < threshold or t + H >= len(L):
            continue
        centred = [L[t + h] - sum(L[t + 1: t + H + 1]) / H for h in range(H + 1)]
        if len(centred) < 5:
            continue
        A00 = A01 = A11 = b0 = b1 = 0.0
        for h in range(1, len(centred) - 1):
            y1, x, y2 = centred[h], centred[h - 1], centred[h + 1]
            A00 += y1 * y1
            A01 += y1 * x
            A11 += x * x
            b0 += y1 * y2
            b1 += x * y2
        det = A00 * A11 - A01 * A01
        if abs(det) < 1e-9:
            continue
        a1 = (A11 * b0 - A01 * b1) / det
        a2 = (A00 * b1 - A01 * b0) / det
        disc = a1 * a1 + 4 * a2
        if disc < -0.5:
            continue
        if disc >= 0:
            lam_dom = (a1 - math.sqrt(disc)) / 2  # more negative → dominant decay
            omega_n = math.sqrt(abs(a2)) if abs(a2) > 1e-6 else 1.0
            zeta = -lam_dom / (2 * omega_n) if omega_n > 1e-6 else 1.0
        else:
            omega_n2 = (a1 / 2) ** 2 - disc / 4
            omega_n = math.sqrt(omega_n2) if omega_n2 > 0 else 1.0
            zeta = -a1 / (2 * omega_n) if omega_n > 1e-6 else 0.5
        if not (-0.2 < zeta < 5.0):
            continue
        zs_all.append(zeta)
        if dR[t] > 0:
            zs_up.append(zeta); n_up += 1
        elif dR[t] < 0:
            zs_dn.append(zeta); n_dn += 1

    def m(xs):
        return sum(xs) / len(xs) if xs else math.nan

    return {"zeta_all": m(zs_all), "zeta_up": m(zs_up),
            "zeta_dn": m(zs_dn), "n_shocks": len(zs_all),
            "n_up": n_up, "n_dn": n_dn}


def loop_area(L: List[float], R: List[float]) -> float:
    """Signed phase-plane area ∮ R dL via the Shoelace formula on (L, R)
    pairs closed back to the starting point."""
    if len(L) < 3:
        return 0.0
    pts = list(zip(L, R))
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def paired_bootstrap(values_a: List[float], values_b: List[float],
                     statistic, B: int = B_BOOT, seed: int = SEED_BOOT
                     ) -> Tuple[float, float, float, float, float]:
    """Paired bootstrap on statistic(diff). Returns (mean_diff, ci_lo, ci_hi,
    p_two_sided, n_eff)."""
    rng = random.Random(seed)
    pairs = [(a, b) for a, b in zip(values_a, values_b)
             if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 2:
        return (math.nan, math.nan, math.nan, math.nan, len(pairs))
    obs = statistic([b - a for a, b in pairs])
    n = len(pairs)
    diffs_boot = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        sample = [pairs[i] for i in idx]
        diffs_boot.append(statistic([b - a for a, b in sample]))
    diffs_boot.sort()
    lo = diffs_boot[int(0.025 * B)]
    hi = diffs_boot[int(0.975 * B) - 1]
    if obs == 0:
        p = 1.0
    else:
        count = sum(1 for x in diffs_boot if (x > 0) == (obs > 0))
        p = min(1.0, 2 * count / B)
    return (obs, lo, hi, p, n)


def main() -> None:
    rng = random.Random(SEED_BOOT)
    gsm = load_runs(GSM_PATH, "gsm8k_cot")
    arith = load_runs(ARITH_PATH, "arith_easy")
    runs = gsm + arith

    by_key: Dict[Tuple[str, str, int], dict] = {}
    for r in runs:
        dR = diffs(r["R"])
        dL = diffs(r["L"])
        hl = half_life_post_shock(r["L"], dR)
        dm = damping_ratio(r["L"], dR)
        area = loop_area(r["L"], r["R"])
        by_key[(r["task"], r["algo"], r["seed"])] = {
            "task": r["task"], "algo": r["algo"], "seed": r["seed"],
            "tau_all": hl["tau_all"], "tau_up": hl["tau_up"],
            "tau_dn": hl["tau_dn"], "n_shocks_hl": hl["n_shocks"],
            "n_up_hl": hl["n_up"], "n_dn_hl": hl["n_dn"],
            "zeta_all": dm["zeta_all"], "zeta_up": dm["zeta_up"],
            "zeta_dn": dm["zeta_dn"], "n_shocks_dm": dm["n_shocks"],
            "n_up_dm": dm["n_up"], "n_dn_dm": dm["n_dn"],
            "loop_area": area,
        }

    # per-seed wide TSV
    wide_path = os.path.join(OUT_DIR, "length_bias_iter76_halflife.tsv")
    with open(wide_path, "w") as f:
        cols = ["task", "algo", "seed", "tau_all", "tau_up", "tau_dn",
                "n_shocks", "n_up", "n_dn"]
        f.write("\t".join(cols) + "\n")
        for k in sorted(by_key):
            v = by_key[k]
            row = [v["task"], v["algo"], str(v["seed"]),
                   f"{v['tau_all']:.4f}", f"{v['tau_up']:.4f}",
                   f"{v['tau_dn']:.4f}", str(v["n_shocks_hl"]),
                   str(v["n_up_hl"]), str(v["n_dn_hl"])]
            f.write("\t".join(row) + "\n")
    print(f"wrote {wide_path}", flush=True)

    damp_path = os.path.join(OUT_DIR, "length_bias_iter76_damping.tsv")
    with open(damp_path, "w") as f:
        cols = ["task", "algo", "seed", "zeta_all", "zeta_up", "zeta_dn",
                "n_shocks", "n_up", "n_dn"]
        f.write("\t".join(cols) + "\n")
        for k in sorted(by_key):
            v = by_key[k]
            row = [v["task"], v["algo"], str(v["seed"]),
                   f"{v['zeta_all']:.4f}", f"{v['zeta_up']:.4f}",
                   f"{v['zeta_dn']:.4f}", str(v["n_shocks_dm"]),
                   str(v["n_up_dm"]), str(v["n_dn_dm"])]
            f.write("\t".join(row) + "\n")
    print(f"wrote {damp_path}", flush=True)

    loop_path = os.path.join(OUT_DIR, "length_bias_iter76_looparea.tsv")
    with open(loop_path, "w") as f:
        cols = ["task", "algo", "seed", "loop_area"]
        f.write("\t".join(cols) + "\n")
        for k in sorted(by_key):
            v = by_key[k]
            row = [v["task"], v["algo"], str(v["seed"]),
                   f"{v['loop_area']:.4f}"]
            f.write("\t".join(row) + "\n")
    print(f"wrote {loop_path}", flush=True)

    # summary: paired bootstrap Dr.GRPO - GRPO per task and statistic
    summary_rows = []
    for task in ("gsm8k_cot", "arith_easy"):
        seeds_g = sorted({k[2] for k in by_key
                          if k[0] == task and k[1] == "grpo"})
        seeds_d = sorted({k[2] for k in by_key
                          if k[0] == task and k[1] == "dr_grpo"})
        common = [s for s in seeds_g if s in seeds_d]
        for stat_name, stat_key in [("tau_all", "tau_all"),
                                     ("tau_up", "tau_up"),
                                     ("tau_dn", "tau_dn"),
                                     ("zeta_all", "zeta_all"),
                                     ("zeta_up", "zeta_up"),
                                     ("zeta_dn", "zeta_dn"),
                                     ("loop_area", "loop_area")]:
            grpo_vals = [by_key[(task, "grpo", s)][stat_key]for s in common]
            dr_vals = [by_key[(task, "dr_grpo", s)][stat_key] for s in common]
            obs, lo, hi, p, n_eff = paired_bootstrap(
                grpo_vals, dr_vals, statistic=lambda xs: sum(xs) / len(xs))
            summary_rows.append({
                "task": task, "stat": stat_name,
                "grpo_mean": sum(grpo_vals) / len(grpo_vals) if grpo_vals else math.nan,
                "drgrpo_mean": sum(dr_vals) / len(dr_vals) if dr_vals else math.nan,
                "diff_dr_minus_gr": obs,
                "ci_lo": lo, "ci_hi": hi, "p_two_sided": p, "n_seeds": n_eff,
            })

    sum_path = os.path.join(OUT_DIR, "length_bias_iter76_summary.tsv")
    with open(sum_path, "w") as f:
        cols = ["task", "stat", "grpo_mean", "drgrpo_mean", "diff_dr_minus_gr",
                "ci_lo", "ci_hi", "p_two_sided", "n_seeds"]
        f.write("\t".join(cols) + "\n")
        for r in summary_rows:
            row = [r["task"], r["stat"], f"{r['grpo_mean']:.4f}",
                   f"{r['drgrpo_mean']:.4f}",
                   f"{r['diff_dr_minus_gr']:.4f}",
                   f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}",
                   f"{r['p_two_sided']:.4f}", str(int(r["n_seeds"]))]
            f.write("\t".join(row) + "\n")
    print(f"wrote {sum_path}", flush=True)

    meta = {
        "iter": 76,
        "task": "Pillar 4 (Length Bias / Dr.GRPO): Reward-shock mean-reversion and phase-plane dissipativity",
        "inputs": [os.path.basename(GSM_PATH), os.path.basename(ARITH_PATH)],
        "H_halflife": H_HALFLIFE,
        "H_osc": H_OSC,
        "Q_shock": Q_SHOCK,
        "B_boot": B_BOOT,
        "n_runs": len(by_key),
        "stats": ["half_life", "damping_ratio", "phase_plane_loop_area"],
    }
    meta_path = os.path.join(OUT_DIR, "length_bias_iter76_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()