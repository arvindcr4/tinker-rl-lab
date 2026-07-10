"""Iter 56 — Pillar 4 (Length Bias / Dr.GRPO): Reward-per-Token Efficiency Frontier.

Novel angle vs iter28/32/36/40/44/48/52: instead of regressing L on t in absolute
units, iter56 asks "how much REWARD does each TOKEN buy?". The single scalar

    rho(t) = R(t) / L(t)          (reward per token, units: 1/tokens)

measures the productive efficiency of the rollout. Dr.GRPO's signature should
manifest as rho decaying FASTER than GRPO on GSM8K CoT -- the policy spends
more tokens per response yet earns no proportional reward, so the marginal
token becomes unproductive.

Two complementary deliverables:

(A) Reward-per-token trajectory. Per (task, algo, seed), compute rho(t) for
    t=0..T-1. Slope of OLS d rho/dt summarises the efficiency drift.
    Paired bootstrap: Dr.GRPO's d rho/dt vs GRPO's d rho/dt.

(B) Cumulative token-tax. Define the "baseline" length L*(t) = L_0 * (R_t / R_0)
    i.e. the length the policy would need to keep reward-per-token constant
    at its starting ratio. Cumulative tax = sum_t max(0, L_t - L*(t)).
    Dr.GRPO should show LARGER cumulative tax than GRPO.

(C) Length phase portrait. At each step, plot (L_t, Delta L_t) where
    Delta L_t = L_{t+1} - L_t. Compute the centroid of the per-(task, algo)
    cloud and the cluster RMS radius. Dr.GRPO's centroid should sit at
    LARGER L and SMALLER |Delta L| (compressed late-stage trajectory).

(D) rho half-life. Time to rho(t) drop to rho_0 / 2 (or final rho if it never
    halves). Dr.GRPO should reach half-life earlier.

Outputs (5 TSVs):
  platform_hybrid/experiments/results/length_bias_iter56_rho_slopes.tsv
  platform_hybrid/experiments/results/length_bias_iter56_grpo_vs_drgrpo.tsv
  platform_hybrid/experiments/results/length_bias_iter56_cumulative_tax.tsv
  platform_hybrid/experiments/results/length_bias_iter56_phase_portrait.tsv
  platform_hybrid/experiments/results/length_bias_iter56_half_life.tsv
  platform_hybrid/experiments/results/length_bias_iter56_summary.tsv

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

RNG_SEED = 56
N_BOOT = 4000


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / sxx, my - (sxy / sxx) * mx


def load_runs(path: Path, task: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d["runs"]:
        sl = r["step_log"]
        ts = [int(s["step"]) for s in sl]
        rs = [float(s["mean_reward"]) for s in sl]
        ls = [float(s["mean_comp_len"]) for s in sl]
        zs = [float(s.get("zvf", float("nan"))) for s in sl]
        rho = [(rs[i] / ls[i]) if ls[i] > 0 else 0.0 for i in range(len(rs))]
        out.append({"task": task, "algo": r["algo"], "seed": r["seed"],
                    "t": ts, "R": rs, "L": ls, "zvf": zs, "rho": rho})
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
        "mean_diff": round(mean_diff, 8),
        "sd_diff": round(sd_diff, 8),
        "ci_lo": round(boots[int(0.025 * n_boot)], 8),
        "ci_hi": round(boots[int(0.975 * n_boot)], 8),
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

    rows = []
    rows.extend(load_runs(DRGRPO_JSON, "arithmetic_easy"))
    rows.extend(load_runs(GSM_JSON, "gsm8k_cot"))

    # ---- A. rho slope per (task, algo, seed)
    rho_rows = []
    for r in rows:
        t = r["t"]; rho = r["rho"]; R = r["R"]; L = r["L"]
        slope, intercept = _ols(t, rho)
        # Per-token efficiency loss in absolute terms over the run
        delta_rho = rho[-1] - rho[0]
        rho_rows.append({
            "task": r["task"], "algo": r["algo"], "seed": r["seed"],
            "rho_0": round(rho[0], 8),
            "rho_last": round(rho[-1], 8),
            "delta_rho": round(delta_rho, 8),
            "slope_drho_dt": round(slope, 8),
            "intercept": round(intercept, 8),
            "n_steps": len(rho),
        })
    write_tsv(RES / "length_bias_iter56_rho_slopes.tsv", rho_rows,
              fieldnames=list(rho_rows[0].keys()))

    # ---- A.2 Paired GRPO vs Dr.GRPO bootstrap on d rho/dt per task
    paired_rows = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for metric in ("slope_drho_dt", "delta_rho", "rho_last"):
            sub = rho_rows
            grpo = {r["seed"]: r for r in sub if r["task"] == task
                    and r["algo"] == "grpo"}
            drgrpo = {r["seed"]: r for r in sub if r["task"] == task
                      and r["algo"] in ("dr_grpo", "drgrpo")}
            common = sorted(set(grpo) & set(drgrpo))
            if not common:
                continue
            gv = [grpo[s][metric] for s in common]
            dv = [drgrpo[s][metric] for s in common]
            boot = paired_bootstrap(gv, dv, rng=rng)
            paired_rows.append({
                "task": task, "metric": metric, "n_pairs": boot["n_pairs"],
                "mean_grpo": round(sum(gv) / len(gv), 8),
                "mean_drgrpo": round(sum(dv) / len(dv), 8),
                "mean_diff": boot["mean_diff"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_le0": boot["p_le0"],
                "interpretation": (
                    "Dr.GRPO rho decays faster" if boot["mean_diff"] < 0
                    and boot["ci_hi"] < 0 and metric == "slope_drho_dt"
                    else "Dr.GRPO rho_last lower" if boot["mean_diff"] < 0
                    and boot["ci_hi"] < 0 and metric == "rho_last"
                    else "inconclusive"
                ),
            })
    write_tsv(RES / "length_bias_iter56_grpo_vs_drgrpo.tsv", paired_rows,
              fieldnames=list(paired_rows[0].keys()))

    # ---- B. Cumulative token tax per (task, algo, seed)
    tax_rows = []
    for r in rows:
        t = r["t"]; R = r["R"]; L = r["L"]
        R0 = R[0]
        if R0 <= 0:
            tax_rows.append({"task": r["task"], "algo": r["algo"], "seed": r["seed"],
                             "cumulative_tax": 0.0, "cumulative_tax_norm": 0.0,
                             "L_baseline_last": 0.0, "L_actual_last": round(L[-1], 4)})
            continue
        # baseline L*(t) = L_0 * (R_t / R_0) -- length needed to keep rho constant
        L_star = [L[0] * (R[i] / R0) for i in range(len(R))]
        tax = sum(max(0.0, L[i] - L_star[i]) for i in range(len(L)))
        # normalize by cumulative actual length so it is unit-free
        cum_L = sum(L)
        tax_norm = tax / cum_L if cum_L > 0 else 0.0
        tax_rows.append({
            "task": r["task"], "algo": r["algo"], "seed": r["seed"],
            "cumulative_tax": round(tax, 4),
            "cumulative_tax_norm": round(tax_norm, 6),
            "L_baseline_last": round(L_star[-1], 4),
            "L_actual_last": round(L[-1], 4),
            "R_first": round(R[0], 4),
            "R_last": round(R[-1], 4),
        })
    # paired bootstrap on cumulative_tax_norm per task
    tax_paired = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        grpo = {r["seed"]: r for r in tax_rows if r["task"] == task
                and r["algo"] == "grpo"}
        drgrpo = {r["seed"]: r for r in tax_rows if r["task"] == task
                  and r["algo"] in ("dr_grpo", "drgrpo")}
        common = sorted(set(grpo) & set(drgrpo))
        if not common:
            continue
        gv = [grpo[s]["cumulative_tax_norm"] for s in common]
        dv = [drgrpo[s]["cumulative_tax_norm"] for s in common]
        boot = paired_bootstrap(gv, dv, rng=rng)
        tax_paired.append({
            "task": task, "metric": "cumulative_tax_norm",
            "n_pairs": boot["n_pairs"],
            "mean_grpo": round(sum(gv) / len(gv), 6),
            "mean_drgrpo": round(sum(dv) / len(dv), 6),
            "mean_diff": boot["mean_diff"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "p_le0": boot["p_le0"],
        })
    write_tsv(RES / "length_bias_iter56_cumulative_tax.tsv",
              tax_rows + tax_paired,
              fieldnames=list((tax_rows + tax_paired)[0].keys())
              if (tax_rows + tax_paired) else
              ["task", "algo", "seed", "cumulative_tax",
               "cumulative_tax_norm", "L_baseline_last", "L_actual_last",
               "R_first", "R_last"])

    # ---- C. Length phase portrait centroid per (task, algo)
    portrait_rows = []
    for r in rows:
        L = r["L"]
        dL = [L[i + 1] - L[i] for i in range(len(L) - 1)]
        n = len(dL)
        if n == 0:
            continue
        # centroid = mean of (L_t, dL_t) for t in [0, n-1)
        Lc = sum(L[:n]) / n
        dLc = sum(dL) / n
        # RMS radius
        rms = math.sqrt(sum((L[i] - Lc) ** 2 + (dL[i] - dLc) ** 2
                            for i in range(n)) / n)
        portrait_rows.append({
            "task": r["task"], "algo": r["algo"], "seed": r["seed"],
            "n_pts": n,
            "centroid_L": round(Lc, 4),
            "centroid_dL": round(dLc, 6),
            "rms_radius": round(rms, 4),
            "L_first": round(L[0], 4),
            "L_last": round(L[-1], 4),
        })
    # paired GRPO vs Dr.GRPO on centroid_L and centroid_dL
    pp_paired = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for metric in ("centroid_L", "centroid_dL", "rms_radius"):
            grpo = {r["seed"]: r for r in portrait_rows if r["task"] == task
                    and r["algo"] == "grpo"}
            drgrpo = {r["seed"]: r for r in portrait_rows if r["task"] == task
                      and r["algo"] in ("dr_grpo", "drgrpo")}
            common = sorted(set(grpo) & set(drgrpo))
            if not common:
                continue
            gv = [grpo[s][metric] for s in common]
            dv = [drgrpo[s][metric] for s in common]
            boot = paired_bootstrap(gv, dv, rng=rng)
            pp_paired.append({
                "task": task, "metric": metric, "n_pairs": boot["n_pairs"],
                "mean_grpo": round(sum(gv) / len(gv), 6),
                "mean_drgrpo": round(sum(dv) / len(dv), 6),
                "mean_diff": boot["mean_diff"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_le0": boot["p_le0"],
            })
    write_tsv(RES / "length_bias_iter56_phase_portrait.tsv",
              portrait_rows + pp_paired,
              fieldnames=list((portrait_rows + pp_paired)[0].keys())
              if (portrait_rows + pp_paired) else
              ["task", "algo", "seed", "n_pts", "centroid_L",
               "centroid_dL", "rms_radius", "L_first", "L_last"])

    # ---- D. rho half-life: time to rho drop to rho_0 / 2
    hl_rows = []
    for r in rows:
        rho = r["rho"]; t = r["t"]
        rho0 = rho[0]
        if rho0 <= 0:
            hl_rows.append({"task": r["task"], "algo": r["algo"], "seed": r["seed"],
                            "rho_0": round(rho0, 6), "rho_half": "nan",
                            "t_half": "nan", "rho_final": round(rho[-1], 6),
                            "rho_final_over_rho0": "nan"})
            continue
        target = rho0 / 2.0
        # find first t where rho[t] <= target
        t_half = None
        for i in range(len(rho)):
            if rho[i] <= target:
                t_half = t[i]
                break
        if t_half is None:
            t_half = -1  # never halved
        rho_final_over_rho0 = rho[-1] / rho0 if rho0 > 0 else 0.0
        hl_rows.append({
            "task": r["task"], "algo": r["algo"], "seed": r["seed"],
            "rho_0": round(rho0, 6),
            "rho_half": round(target, 6),
            "t_half": t_half,
            "rho_final": round(rho[-1], 6),
            "rho_final_over_rho0": round(rho_final_over_rho0, 6),
            "halved": t_half != -1,
        })
    hl_paired = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for metric in ("rho_final_over_rho0",):
            grpo = {r["seed"]: r for r in hl_rows if r["task"] == task
                    and r["algo"] == "grpo"}
            drgrpo = {r["seed"]: r for r in hl_rows if r["task"] == task
                      and r["algo"] in ("dr_grpo", "drgrpo")}
            common = sorted(set(grpo) & set(drgrpo))
            if not common:
                continue
            gv = [grpo[s][metric] for s in common]
            dv = [drgrpo[s][metric] for s in common]
            boot = paired_bootstrap(gv, dv, rng=rng)
            hl_paired.append({
                "task": task, "metric": metric, "n_pairs": boot["n_pairs"],
                "mean_grpo": round(sum(gv) / len(gv), 6),
                "mean_drgrpo": round(sum(dv) / len(dv), 6),
                "mean_diff": boot["mean_diff"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_le0": boot["p_le0"],
            })
    write_tsv(RES / "length_bias_iter56_half_life.tsv",
              hl_rows + hl_paired,
              fieldnames=list((hl_rows + hl_paired)[0].keys())
              if (hl_rows + hl_paired) else
              ["task", "algo", "seed", "rho_0", "rho_half", "t_half",
               "rho_final", "rho_final_over_rho0", "halved"])

    # ---- Summary rollup
    summary_rows = []
    for p in paired_rows:
        summary_rows.append({
            "task": p["task"], "metric": p["metric"],
            "n_pairs": p["n_pairs"],
            "mean_grpo": p["mean_grpo"],
            "mean_drgrpo": p["mean_drgrpo"],
            "mean_diff": p["mean_diff"],
            "ci_lo": p["ci_lo"],
            "ci_hi": p["ci_hi"],
            "p_le0": p["p_le0"],
            "interpretation": p["interpretation"],
        })
    for p in tax_paired:
        summary_rows.append({
            "task": p["task"], "metric": p["metric"],
            "n_pairs": p["n_pairs"],
            "mean_grpo": p["mean_grpo"],
            "mean_drgrpo": p["mean_drgrpo"],
            "mean_diff": p["mean_diff"],
            "ci_lo": p["ci_lo"],
            "ci_hi": p["ci_hi"],
            "p_le0": p["p_le0"],
            "interpretation": (
                "Dr.GRPO pays more tax" if p["mean_diff"] > 0 and p["ci_lo"] > 0
                else "inconclusive"
            ),
        })
    for p in hl_paired:
        summary_rows.append({
            "task": p["task"], "metric": p["metric"],
            "n_pairs": p["n_pairs"],
            "mean_grpo": p["mean_grpo"],
            "mean_drgrpo": p["mean_drgrpo"],
            "mean_diff": p["mean_diff"],
            "ci_lo": p["ci_lo"],
            "ci_hi": p["ci_hi"],
            "p_le0": p["p_le0"],
            "interpretation": (
                "Dr.GRPO rho decays more" if p["mean_diff"] < 0 and p["ci_hi"] < 0
                else "inconclusive"
            ),
        })
    for p in pp_paired:
        summary_rows.append({
            "task": p["task"], "metric": p["metric"],
            "n_pairs": p["n_pairs"],
            "mean_grpo": p["mean_grpo"],
            "mean_drgrpo": p["mean_drgrpo"],
            "mean_diff": p["mean_diff"],
            "ci_lo": p["ci_lo"],
            "ci_hi": p["ci_hi"],
            "p_le0": p["p_le0"],
            "interpretation": (
                "Dr.GRPO centroid_L higher" if p["mean_diff"] > 0
                and p["ci_lo"] > 0 and p["metric"] == "centroid_L"
                else "Dr.GRPO centroid_dL smaller in magnitude"
                if p["mean_diff"] > 0 and p["ci_lo"] > 0
                and p["metric"] == "centroid_dL"
                else "inconclusive"
            ),
        })
    write_tsv(RES / "length_bias_iter56_summary.tsv", summary_rows,
              fieldnames=list(summary_rows[0].keys()))

    print("=== iter56 reward-per-token efficiency summary ===")
    for r in summary_rows:
        print(r)
    print("\n=== per-run rho_first / rho_last / slope ===")
    for r in rho_rows:
        print(r)


if __name__ == "__main__":
    main()