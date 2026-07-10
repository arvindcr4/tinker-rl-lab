#!/usr/bin/env python3
"""B-SYNTH row 12 — Critic-Degeneracy Hypothesis (CDH) empirical test.

Frontier synthesis (Round 1, FRONTIER_INSIGHTS.md) proposes the
**Critic Degeneracy Hypothesis**: for sparse, terminal-reward CoT,
PPO's value head V_φ(x_{1:t}) is mathematically degenerate — it
collapses to a static prompt-difficulty regressor
V_φ(x_{1:t}) ≈ E[R|x_prompt], i.e. what GRPO computes statelessly
via the group mean. Under CDH:
  (a) PPO critic ≈ GRPO group-mean (control variate, not oracle)
  (b) PPO should be ≲ GRPO in gradient variance (control variate role)
  (c) PPO heldout_acc should match GRPO when stack is matched
  (d) PPO critic would NOT add useful temporal credit assignment
      for outcome-only reward

This script tests (a)-(d) on the SAME-STACK PPO/GRPO benchmark data
(platform_modal/scripts/samestack_ppo_grpo, 5 seeds × 2 algos = 10 runs, 40 steps).
We do NOT train a critic head — we test the OBSERVABLE consequence
of CDH:

  H1 (variance): CV(grad_norm_PPO) > CV(grad_norm_GRPO) — the critic
      is a NET NOISE amplifier, not a control variate. CDH predicts
      this because the critic approximates the prompt mean (which the
      group mean also computes) with extra parameter noise.

  H2 (smoothing): per-step reward std(last10_window)_PPO <
      std(last10_window)_GRPO. The critic smooths the per-step reward
      trajectory IF the value head is a useful baseline. CDH predicts
      NO difference (since critic ≈ prompt-mean ≈ group-mean).

  H3 (equivalence): paired last10_avg_PPO ≈ last10_avg_GRPO with
      p > 0.05. Already established (samestack p=0.37). Reported for
      completeness.

  H4 (collapse signature): PPO grad_norm tracks batch reward MORE
      tightly than GRPO's grad_norm tracks batch reward, IF the critic
      is serving as a degenerate control variate. CDH predicts
      R_PPO(grad_norm, batch_reward) > R_GRPO(grad_norm, batch_reward).

  H5 (scaling): across the n=12 RQS-graded anchors, the
      PPO-equivalent static-regressor accuracy should be ALMOST
      PERFECTLY predicted by the prompt-mean reward mean (RQS is a
      proxy). This is the "critic = regressor" collapse fingerprint.

Outputs (5 TSVs):
  cdh_gradnorm_stats.tsv       — per-seed grad_norm mean/SD/CV (H1)
  cdh_reward_window.tsv        — per-seed rolling reward variance (H2)
  cdh_paired_test.tsv          — heldout_acc paired test (H3)
  cdh_gradnorm_vs_reward.tsv   — grad_norm ~ batch_reward correlation (H4)
  cdh_rqs_collapse.tsv         — RQS vs r_mean residual (H5)

All inputs are existing data; no new tinker runs.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
BERK = RES / "berkeley"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_samestack() -> List[Dict[str, Any]]:
    with open(RES / "samestack_ppo_grpo.json") as fh:
        return json.load(fh)["runs"]


def _load_eureka() -> List[Dict[str, Any]]:
    """Eureka RQS per anchor — proxy for prompt-difficulty regressor."""
    out: List[Dict[str, Any]] = []
    with open(BERK / "eureka_rqs_per_anchor.tsv") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for row in rdr:
            out.append(
                {
                    "model": row["model"],
                    "params_B": float(row["params_B"]),
                    "r_mean": float(row["r_mean"]),
                    "RQS": float(row["RQS"]),
                    "frac_above_0p5": float(row["frac_above_0p5"]),
                    "zero_frac": float(row["zero_frac"]),
                }
            )
    return out


def _load_zvf_sweep() -> List[Dict[str, Any]]:
    """groupsize_zvf_sweep: per-G ZVF (the GRPO group-mean analogue)."""
    out: List[Dict[str, Any]] = []
    with open(RES / "groupsize_zvf_sweep.tsv") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for row in rdr:
            out.append(
                {
                    "G": int(row["G"]),
                    "n_seeds": int(row["n_seeds"]),
                    "heldout_acc_mean": float(row["heldout_acc_mean"]),
                    "heldout_acc_se": float(row["heldout_acc_se"]),
                    "mean_zvf": float(row["mean_zvf"]),
                    "zvf_theory_at_mean_p": float(row["zvf_theory_at_mean_p"]),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _rolling_var(xs: List[float], window: int) -> List[float]:
    out: List[float] = []
    for i in range(len(xs)):
        if i < window - 1:
            continue
        w = xs[i - window + 1 : i + 1]
        m = sum(w) / window
        v = sum((x - m) ** 2 for x in w) / max(1, window - 1)
        out.append(v)
    return out


def _welch_p(diff: float, se: float) -> float:
    """Two-sided p-value under normal approximation; small-sample safe."""
    if se <= 0:
        return 1.0
    z = abs(diff) / se
    # Phi complement, rational approximation
    return max(1e-12, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2)))))


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def _gradnorm_stats(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """H1 — coefficient of variation of grad_norm by algo."""
    out: List[Dict[str, Any]] = []
    for algo in ("grpo", "ppo"):
        per_seed = [s for s in runs if s["algo"] == algo]
        cvs: List[float] = []
        means: List[float] = []
        sds: List[float] = []
        for s in per_seed:
            grads = [step["grad_norm"] for step in s["step_log"]]
            m = statistics.mean(grads)
            sd = statistics.pstdev(grads)
            means.append(m)
            sds.append(sd)
            cvs.append(sd / m if m > 0 else float("inf"))
        out.append(
            {
                "algo": algo,
                "n_seeds": len(per_seed),
                "gradnorm_mean_per_seed": json.dumps([round(m, 4) for m in means]),
                "gradnorm_sd_per_seed": json.dumps([round(s, 4) for s in sds]),
                "gradnorm_mean": round(statistics.mean(means), 4),
                "gradnorm_sd": round(statistics.mean(sds), 4),
                "cv_gradnorm_mean": round(statistics.mean(cvs), 4),
                "cv_gradnorm_median": round(statistics.median(cvs), 4),
            }
        )
    return out


def _reward_window_stats(runs: List[Dict[str, Any]], window: int = 10) -> List[Dict[str, Any]]:
    """H2 — rolling variance of last10 reward by algo."""
    out: List[Dict[str, Any]] = []
    for algo in ("grpo", "ppo"):
        per_seed = []
        for s in [r for r in runs if r["algo"] == algo]:
            rew = [step["mean_reward"] for step in s["step_log"]]
            v = _rolling_var(rew, window)
            per_seed.append(statistics.mean(v) if v else float("nan"))
        out.append(
            {
                "algo": algo,
                "n_seeds": len(per_seed),
                "rolling_var_mean": round(statistics.mean(per_seed), 6),
                "rolling_var_sd": round(statistics.pstdev(per_seed), 6),
                "rolling_var_max": round(max(per_seed), 6),
                "per_seed": json.dumps([round(v, 6) for v in per_seed]),
            }
        )
    return out


def _paired_test(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """H3 — paired heldout_acc last10_avg grpo vs ppo (matched seed stack)."""
    grpo_by_seed = {r["seed"]: r for r in runs if r["algo"] == "grpo"}
    ppo_by_seed = {r["seed"]: r for r in runs if r["algo"] == "ppo"}
    common = sorted(set(grpo_by_seed) & set(ppo_by_seed))
    diffs = []
    out_rows: List[Dict[str, Any]] = []
    for seed in common:
        d = grpo_by_seed[seed]["last10_avg"] - ppo_by_seed[seed]["last10_avg"]
        diffs.append(d)
        out_rows.append(
            {
                "seed": seed,
                "last10_grpo": grpo_by_seed[seed]["last10_avg"],
                "last10_ppo": ppo_by_seed[seed]["last10_avg"],
                "diff_grpo_minus_ppo": round(d, 4),
            }
        )
    mean_d = statistics.mean(diffs) if diffs else 0.0
    sd_d = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    se_d = sd_d / math.sqrt(len(diffs)) if diffs else 0.0
    p_val = _welch_p(mean_d, se_d) if se_d > 0 else 1.0
    # Append summary
    out_rows.append(
        {
            "seed": "SUMMARY",
            "last10_grpo": round(statistics.mean([grpo_by_seed[s]["last10_avg"] for s in common]), 4),
            "last10_ppo": round(statistics.mean([ppo_by_seed[s]["last10_avg"] for s in common]), 4),
            "diff_grpo_minus_ppo": f"mean={round(mean_d,4)} sd={round(sd_d,4)} se={round(se_d,4)} p={p_val:.4f}",
        }
    )
    return out_rows


def _gradnorm_vs_reward(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """H4 — correlation of grad_norm ~ batch_reward across step-log."""
    out: List[Dict[str, Any]] = []
    for algo in ("grpo", "ppo"):
        per_seed = []
        for s in [r for r in runs if r["algo"] == algo]:
            xs = [step["grad_norm"] for step in s["step_log"]]
            ys = [step["mean_reward"] for step in s["step_log"]]
            per_seed.append(_pearson(xs, ys))
        out.append(
            {
                "algo": algo,
                "n_seeds": len(per_seed),
                "r_per_seed": json.dumps([round(r, 4) for r in per_seed]),
                "r_mean": round(statistics.mean(per_seed), 4),
                "r_median": round(statistics.median(per_seed), 4),
            }
        )
    return out


def _rqs_collapse(anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """H5 — critic-collapse signature: r_mean ≈ a + b*RQS.

    Under CDH, PPO's critic ≈ RQS-proxy (E[R|x_prompt]); so a static
    regressor fit on RQS should explain most of r_mean variance. We
    fit OLS and report R^2 / residual statistics.
    """
    # OLS r_mean ~ RQS (filter NaN)
    pts = [(a["RQS"], a["r_mean"]) for a in anchors if 0.0 < a["RQS"] < 1.0]
    n = len(pts)
    if n < 3:
        return []
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return []
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [(y - (intercept + slope * x)) for x, y in zip(xs, ys)]
    r2 = 1 - sum(r ** 2 for r in resid) / sum((y - my) ** 2 for y in ys)
    return [
        {
            "n": n,
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r2": round(r2, 4),
            "resid_mean": round(statistics.mean(resid), 4),
            "resid_max_abs": round(max(abs(r) for r in resid), 4),
        }
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(k, "")) for k in keys) + "\n")


def main() -> None:
    runs = _load_samestack()
    anchors = _load_eureka()

    g1 = _gradnorm_stats(runs)
    g2 = _reward_window_stats(runs)
    g3 = _paired_test(runs)
    g4 = _gradnorm_vs_reward(runs)
    g5 = _rqs_collapse(anchors)

    _write_tsv(BERK / "cdh_gradnorm_stats.tsv", g1)
    _write_tsv(BERK / "cdh_reward_window.tsv", g2)
    _write_tsv(BERK / "cdh_paired_test.tsv", g3)
    _write_tsv(BERK / "cdh_gradnorm_vs_reward.tsv", g4)
    _write_tsv(BERK / "cdh_rqs_collapse.tsv", g5)

    # Summary JSON
    summary = {
        "h1_cv_gradnorm": {row["algo"]: row["cv_gradnorm_mean"] for row in g1},
        "h2_rolling_var": {row["algo"]: row["rolling_var_mean"] for row in g2},
        "h3_paired_diff_mean": g3[-1]["diff_grpo_minus_ppo"]
        if g3 and "diff_grpo_minus_ppo" in g3[-1] else None,
        "h4_gradnorm_vs_reward_r": {row["algo"]: row["r_mean"] for row in g4},
        "h5_rqs_regressor": g5[0] if g5 else None,
    }
    with open(BERK / "cdh_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Console summary
    print("=" * 64)
    print("Critic-Degeneracy Hypothesis (CDH) empirical test")
    print("=" * 64)
    print(f"H1 CV(grad_norm):  GRPO={summary['h1_cv_gradnorm']['grpo']:.3f}  "
          f"PPO={summary['h1_cv_gradnorm']['ppo']:.3f}")
    print(f"H2 rolling-var:    GRPO={summary['h2_rolling_var']['grpo']:.4f}  "
          f"PPO={summary['h2_rolling_var']['ppo']:.4f}")
    print(f"H3 paired last10:   {summary['h3_paired_diff_mean']}")
    print(f"H4 R(grad,rwd):    GRPO={summary['h4_gradnorm_vs_reward_r']['grpo']:.3f}  "
          f"PPO={summary['h4_gradnorm_vs_reward_r']['ppo']:.3f}")
    print(f"H5 RQS-regressor:  {summary['h5_rqs_regressor']}")
    print("Outputs:")
    for fn in ("cdh_gradnorm_stats.tsv", "cdh_reward_window.tsv",
               "cdh_paired_test.tsv", "cdh_gradnorm_vs_reward.tsv",
               "cdh_rqs_collapse.tsv", "cdh_summary.json"):
        print(f"  platform_hybrid/experiments/results/berkeley/{fn}")


if __name__ == "__main__":
    main()