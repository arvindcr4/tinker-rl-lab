#!/usr/bin/env python3
"""
Pillar 4 — Length Bias / Dr. GRPO signature analysis.

Measures:
  - Per-run Spearman(step, mean_comp_len)        : length trend across training
  - Per-run Spearman(step, mean_reward)           : reward trend across training
  - Per-run Spearman(mean_comp_len, mean_reward)  : within-run coupling
  - Last10 mean reward, last10 mean length, half-life growth
  - Dr.GRPO signature flag: (length trend up) AND (reward trend flat or down)

Inputs : experiments/results/drgrpo_vs_grpo.json         (Qwen2.5-0.5B / arithmetic,
                                                          40 steps, 5 GRPO + 5 Dr.GRPO)
         experiments/results/drgrpo_gsm8k_cot_full.json  (Qwen2.5-1.5B-Instruct / GSM8K-CoT,
                                                          30 steps, 3 GRPO + 3 Dr.GRPO)

Outputs: experiments/results/length_bias.tsv            (per-run metrics, header explained below)
         experiments/results/length_bias_summary.tsv    (per-algo summary)
         figures/length_vs_reward.pdf and .png          (2-panel: trajectories + scatter)
         figures/length_vs_reward.pdf copied to paper/figures/

Cite   : tong2025drgrpo (Dr.GRPO, arXiv:2503.20783 — Liu et al. 2025)
         mcgrpo2025    (median-centered advantage)
         wu2025grpo_dpo (arXiv:2510.00977 — equivalence of GRPO and DPO at the
                         contrastive level)
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIGS = ROOT / "figures"
PAPER_FIGS = ROOT / "paper" / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


# ----------------------------- core metrics --------------------------------

def spearman_with_nan(x: list[float], y: list[float]) -> tuple[float, float, int]:
    """Return Spearman rho, two-sided p-value, n. Returns NaN if constant."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan"), float("nan"), int(len(xa))
    rho, p = stats.spearmanr(xa, ya)
    return float(rho), float(p), int(len(xa))


def slope(x: list[float], y: list[float]) -> float:
    """OLS slope of y on x (simple linear regression, no intercept handling)."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2 or np.std(xa) == 0:
        return float("nan")
    return float(np.polyfit(xa, ya, 1)[0])


def half_diff(seq: list[float]) -> tuple[float, float]:
    """Return (mean of first half, mean of second half)."""
    if not seq:
        return float("nan"), float("nan")
    mid = len(seq) // 2
    return float(np.mean(seq[:mid])), float(np.mean(seq[mid:]))


def length_bias_flag(length_slope: float, reward_slope: float,
                     length_rho: float, reward_rho: float) -> int:
    """Dr.GRPO signature = length grows AND reward stays flat or drops.

    We use both slope signs (last-vs-first-half mean) and Spearman rho for robustness.
    """
    length_up = (length_slope > 0) or (length_rho > 0.0)
    reward_flat_or_down = (reward_slope <= 0) or (reward_rho <= 0.0)
    return int(length_up and reward_flat_or_down)


# ----------------------------- per-run analysis -----------------------------

def per_run_metrics(step_log: list[dict]) -> dict:
    steps = [s["step"] for s in step_log]
    rewards = [float(s["mean_reward"]) for s in step_log]
    lens = [float(s["mean_comp_len"]) for s in step_log]
    zvfs = [float(s.get("zvf", float("nan"))) for s in step_log]

    rho_lr, p_lr, _ = spearman_with_nan(steps, lens)          # length trend
    rho_rr, p_rr, _ = spearman_with_nan(steps, rewards)       # reward trend
    rho_pair, p_pair, _ = spearman_with_nan(lens, rewards)    # within-run coupling

    # half-life differences (last half minus first half)
    _, len_last = half_diff(lens)
    _, rew_last = half_diff(rewards)
    len_first = float(np.mean(lens[: len(lens) // 2]))
    rew_first = float(np.mean(rewards[: len(rewards) // 2]))

    dlen = len_last - len_first
    drew = rew_last - rew_first

    # slopes (per-step change)
    len_slope = slope(steps, lens)
    rew_slope = slope(steps, rewards)

    last10_rew = float(np.mean(rewards[-min(10, len(rewards)):]))
    last10_len = float(np.mean(lens[-min(10, len(lens)):]))
    first5_rew = float(np.mean(rewards[: min(5, len(rewards))]))
    first5_len = float(np.mean(lens[: min(5, len(lens))]))

    # Dr.GRPO signature (length up AND reward not growing)
    flag = length_bias_flag(len_slope, rew_slope, rho_lr, rho_rr)

    return {
        "n_steps": len(step_log),
        "first5_reward": round(first5_rew, 6),
        "last10_reward": round(last10_rew, 6),
        "reward_half_delta": round(drew, 6),
        "first5_len": round(first5_len, 4),
        "last10_len": round(last10_len, 4),
        "len_half_delta": round(dlen, 4),
        "spearman_step_len_rho": round(rho_lr, 4),
        "spearman_step_len_p": round(p_lr, 4),
        "spearman_step_reward_rho": round(rho_rr, 4),
        "spearman_step_reward_p": round(p_rr, 4),
        "spearman_len_reward_rho": round(rho_pair, 4),
        "spearman_len_reward_p": round(p_pair, 4),
        "len_slope_per_step": round(len_slope, 6) if not math.isnan(len_slope) else float("nan"),
        "rew_slope_per_step": round(rew_slope, 6) if not math.isnan(rew_slope) else float("nan"),
        "mean_zvf": round(float(np.nanmean(zvfs)), 6),
        "length_bias_flag": flag,
    }


# ----------------------------- main pipeline --------------------------------

def load_dataset(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    return d.get("runs", [])


def main() -> int:
    sources = [
        {
            "task": "arithmetic_easy_qwen2.5-0.5b",
            "path": RES / "drgrpo_vs_grpo.json",
            "step_count": 40,
            "n_seeds": 5,
        },
        {
            "task": "gsm8k_cot_hard_qwen2.5-1.5b",
            "path": RES / "drgrpo_gsm8k_cot_full.json",
            "step_count": 30,
            "n_seeds": 3,
        },
    ]

    rows: list[dict] = []
    for src in sources:
        for run in load_dataset(src["path"]):
            sl = run.get("step_log")
            if not sl:
                continue
            m = per_run_metrics(sl)
            m.update({
                "task": src["task"],
                "algo": run["algo"],
                "seed": run["seed"],
                "model": run.get("model", ""),
                "experiment": run.get("experiment", ""),
            })
            rows.append(m)

    # -------- per-run TSV -------------------------------------------------
    cols = [
        "task", "algo", "seed", "model",
        "n_steps",
        "first5_reward", "last10_reward", "reward_half_delta",
        "first5_len", "last10_len", "len_half_delta",
        "spearman_step_len_rho", "spearman_step_len_p",
        "spearman_step_reward_rho", "spearman_step_reward_p",
        "spearman_len_reward_rho", "spearman_len_reward_p",
        "len_slope_per_step", "rew_slope_per_step",
        "mean_zvf", "length_bias_flag",
    ]
    tsv = RES / "length_bias.tsv"
    with tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {tsv} ({len(rows)} rows)")

    # -------- per-algo summary TSV ---------------------------------------
    sum_rows: list[dict] = []
    for task in sorted({r["task"] for r in rows}):
        for algo in ("grpo", "dr_grpo"):
            sub = [r for r in rows if r["task"] == task and r["algo"] == algo]
            if not sub:
                continue
            rhos_l = np.array([r["spearman_step_len_rho"] for r in sub])
            rhos_r = np.array([r["spearman_step_reward_rho"] for r in sub])
            rhos_pair = np.array([r["spearman_len_reward_rho"] for r in sub])
            flags = np.array([r["length_bias_flag"] for r in sub])
            sum_rows.append({
                "task": task,
                "algo": algo,
                "n_seeds": len(sub),
                "mean_step_len_rho": round(float(np.mean(rhos_l)), 4),
                "sd_step_len_rho": round(float(np.std(rhos_l, ddof=1)) if len(rhos_l) > 1 else 0.0, 4),
                "mean_step_reward_rho": round(float(np.mean(rhos_r)), 4),
                "sd_step_reward_rho": round(float(np.std(rhos_r, ddof=1)) if len(rhos_r) > 1 else 0.0, 4),
                "mean_len_reward_rho": round(float(np.mean(rhos_pair)), 4),
                "sd_len_reward_rho": round(float(np.std(rhos_pair, ddof=1)) if len(rhos_pair) > 1 else 0.0, 4),
                "mean_last10_reward": round(float(np.mean([r["last10_reward"] for r in sub])), 4),
                "mean_last10_len": round(float(np.mean([r["last10_len"] for r in sub])), 4),
                "length_bias_flag_rate": round(float(np.mean(flags)), 4),
            })
    sum_cols = list(sum_rows[0].keys())
    sum_tsv = RES / "length_bias_summary.tsv"
    with sum_tsv.open("w") as f:
        f.write("\t".join(sum_cols) + "\n")
        for r in sum_rows:
            f.write("\t".join(str(r[c]) for c in sum_cols) + "\n")
    print(f"wrote {sum_tsv} ({len(sum_rows)} rows)")

    # -------- figure -----------------------------------------------------
    make_figure(rows)

    # -------- terminal summary ------------------------------------------
    print("\n=== length-bias summary (per-algo, per-task) ===")
    for r in sum_rows:
        print(f"  {r['task']:36s} {r['algo']:8s} "
              f"rho_len={r['mean_step_len_rho']:+.3f} "
              f"rho_reward={r['mean_step_reward_rho']:+.3f} "
              f"rho_pair={r['mean_len_reward_rho']:+.3f} "
              f"flag_rate={r['length_bias_flag_rate']:.2f}")

    return 0


# ----------------------------- figure ---------------------------------------

def make_figure(rows: list[dict]) -> None:
    """Two-panel figure:
       Left  : per-step (mean_reward, mean_comp_len) trajectories, GRPO vs Dr.GRPO
       Right : scatter of last10 reward vs last10 length, colored by algo.

    We only use GSM8K-CoT rows in the scatter (the harder task where the bias matters).
    """
    # group by task
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    # trajectories need step_log; reload
    def load_sl(task: str, algo: str, seed: int) -> list[dict] | None:
        if "arithmetic" in task:
            p = RES / "drgrpo_vs_grpo.json"
        elif "gsm8k" in task:
            p = RES / "drgrpo_gsm8k_cot_full.json"
        else:
            return None
        for run in json.loads(p.read_text())["runs"]:
            if run["algo"] == algo and run["seed"] == seed:
                return run.get("step_log")
        return None

    # pick the most-informative task: gsm8k_cot_hard_qwen2.5-1.5b
    hard_task = "gsm8k_cot_hard_qwen2.5-1.5b"
    easy_task = "arithmetic_easy_qwen2.5-0.5b"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    colors = {"grpo": "#1f77b4", "dr_grpo": "#d62728"}

    for ax, task in zip(axes, [hard_task, easy_task]):
        for r in by_task[task]:
            sl = load_sl(task, r["algo"], r["seed"])
            if not sl:
                continue
            steps = [s["step"] for s in sl]
            lens = [s["mean_comp_len"] for s in sl]
            rews = [s["mean_reward"] for s in sl]
            ax.plot(lens, rews, "-o", color=colors[r["algo"]], alpha=0.45,
                    markersize=3.5, linewidth=1.1,
                    label=r["algo"] if r is by_task[task][0] or
                    (r["algo"] == "grpo" and not any(rr["algo"] == "grpo" and rr is not r
                    for rr in by_task[task][:by_task[task].index(r)])) else None)
        # single legend handle per algo
        ax.plot([], [], "-o", color=colors["grpo"], label="GRPO")
        ax.plot([], [], "-o", color=colors["dr_grpo"], label="Dr. GRPO")
        ax.set_xlabel("mean completion length (tokens)")
        ax.set_ylabel("mean reward")
        title_short = "GSM8K-CoT, Qwen2.5-1.5B" if "gsm8k" in task else "Arithmetic, Qwen2.5-0.5B"
        ax.set_title(title_short)
        ax.grid(alpha=0.25)

    fig.suptitle("Length-vs-reward trajectories: Dr. GRPO should decouple them",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    pdf_path = FIGS / "length_vs_reward.pdf"
    png_path = FIGS / "length_vs_reward.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    shutil.copy(pdf_path, PAPER_FIGS / "length_vs_reward.pdf")
    print(f"wrote {pdf_path} and {png_path} (+ copy in paper/figures/)")


if __name__ == "__main__":
    sys.exit(main())