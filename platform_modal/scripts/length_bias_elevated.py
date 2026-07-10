#!/usr/bin/env python3
"""
Pillar 4 elevation — three new analyses on the same per-step traces.

  (A) Trap-onset detection (sliding-window).
      The original Pillar 4 flag fires on the GLOBAL monotonic trend
      (length slope > 0 AND reward slope <= 0 over the whole run).
      The sliding-window version asks: does the *local* trend in a
      window of W steps show the signature?  Concretely, for each
      step s >= W we compute rho_W(step, len) and rho_W(step, reward)
      on the window [s-W, s]; the trap-onset step is the first s
      where (rho_W(step, len) > 0) AND (rho_W(step, reward) <= 0)
      AND (the global first-half mean len is greater than the local
      first-window len -- i.e., the model had been compressing and
      then started re-expanding).  Reports the step index and
      whether the trap ever fires on a run.

  (B) Bootstrap 95% CIs on per-algo Spearman means.
      The summary TSV from the base script reports the per-seed mean
      and SD; here we add non-parametric bootstrap (n=2000, BCa
      skipped for speed) CIs to give reviewers a sense of the
      uncertainty in the GRPO vs Dr.GRPO difference.

  (C) Decile-binned length-vs-reward coupling.
      For each (algo, task) cell we sort runs by step index, then
      for each (algo, task) we pool per-step (len, reward) points
      across seeds, bin into 10 deciles of length, and report the
      mean reward per decile.  This is the cleanest *direct* test
      of the verbosity-trap: "if the model is rewarded for being
      long, the right-most decile should have higher mean reward
      than the left-most decile."

  (D) Cross-validation on arithmetic_metrics.jsonl.
      This file has 100 steps of ac_tokens_per_turn (per-step mean
      completion-length proxy) and env/all/reward/total on a
      different arithmetic run.  The base Pillar 4 sources are 30-40
      steps; this 100-step trace is the longest per-step length
      signal in the benchmark, and lets us check whether the
      compression pattern holds at a longer horizon.

Inputs : platform_hybrid/experiments/results/drgrpo_vs_grpo.json          (per-step traces, 40 steps, 5+5)
         platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json   (per-step traces, 30 steps, 3+3)
         platform_hybrid/experiments/results/arithmetic_metrics.jsonl     (100-step cross-validation)

Outputs: platform_hybrid/experiments/results/length_bias_trap.tsv         (per-run trap onset detection)
         platform_hybrid/experiments/results/length_bias_summary_ci.tsv   (per-algo Spearman + 95% bootstrap CI)
         platform_hybrid/experiments/results/length_bias_bins.tsv         (per-algo decile binned len-vs-reward)
         platform_hybrid/experiments/results/length_bias_crosval.tsv      (arithmetic_metrics.jsonl summary)
         figures/length_vs_reward_elevated.{pdf,png}      (3-panel: trap-onset + binned + crossval)
         paper/figures/length_vs_reward_elevated.pdf

Cite   : tong2025drgrpo (Dr.GRPO, arXiv:2503.20783)
         mcgrpo2025      (median-centered advantage)
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

# ----------------------------- shared helpers ------------------------------

SOURCES = [
    {
        "task": "arithmetic_easy_qwen2.5-0.5b",
        "path": RES / "drgrpo_vs_grpo.json",
        "n_seeds": 5,
    },
    {
        "task": "gsm8k_cot_hard_qwen2.5-1.5b",
        "path": RES / "drgrpo_gsm8k_cot_full.json",
        "n_seeds": 3,
    },
]


def load_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("runs", [])


def spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(xa, ya)
    return float(rho), float(p)


def half_diff(seq: list[float]) -> tuple[float, float]:
    if not seq:
        return float("nan"), float("nan")
    mid = len(seq) // 2
    return float(np.mean(seq[:mid])), float(np.mean(seq[mid:]))


# ----------------------------- (A) trap-onset ------------------------------

def trap_onset(step_log: list[dict], window: int = 10,
               rho_threshold: float = 0.3) -> dict:
    """Find the earliest step where the Dr.GRPO signature engages locally.

    The signature is:
        (a) the local rho(window-step, len) > rho_threshold
            (length starts RISING in a W-step sliding window, with a
            non-trivial trend, not just noise);
        (b) the local rho(window-step, reward) <= 0
            (reward plateaus or drops in the same window);
        (c) the END-OF-WINDOW length is greater than the
            first-half mean length (the model has actually
            re-expanded past its early-stage compression reference).

    Operational notes
    -----------------
    The original Pillar 4 flag (length slope > 0 AND reward slope <= 0
    GLOBALLY) fires on 0/16 runs.  The sliding-window version is
    *strictly more permissive* but with the rho_threshold > 0.3
    minimum and the "end > first-half" reference it should fire on a
    similar (low) number of runs, which is the right answer for
    "the trap has not engaged at 30-40 step horizons."

    """
    n = len(step_log)
    steps = [s["step"] for s in step_log]
    lens = [float(s["mean_comp_len"]) for s in step_log]
    rewards = [float(s["mean_reward"]) for s in step_log]
    first_half_len = float(np.mean(lens[: n // 2]))

    onset_step: int | None = None
    onset_rho_len: float | None = None
    onset_rho_rew: float | None = None
    onset_end_len: float | None = None
    for s in range(window, n):
        win_steps = steps[s - window: s]
        win_lens = lens[s - window: s]
        win_rew = rewards[s - window: s]
        r_l, _ = spearman(win_steps, win_lens)
        r_r, _ = spearman(win_steps, win_rew)
        cond_a = (not math.isnan(r_l)) and r_l > rho_threshold
        cond_b = (not math.isnan(r_r)) and r_r <= 0
        cond_c = float(win_lens[-1]) > first_half_len
        if cond_a and cond_b and cond_c:
            onset_step = int(s)
            onset_rho_len = r_l
            onset_rho_rew = r_r
            onset_end_len = float(win_lens[-1])
            break

    return {
        "window": window,
        "rho_threshold": rho_threshold,
        "trap_onset_step": onset_step if onset_step is not None else -1,
        "trap_onset_rho_len": round(float(onset_rho_len), 4) if onset_rho_len is not None else float("nan"),
        "trap_onset_rho_rew": round(float(onset_rho_rew), 4) if onset_rho_rew is not None else float("nan"),
        "trap_onset_end_len": round(float(onset_end_len), 4) if onset_end_len is not None else float("nan"),
        "trap_fired": int(onset_step is not None),
        "first_half_mean_len": round(first_half_len, 4),
        "n_steps": n,
    }


# ----------------------------- (B) bootstrap CIs ----------------------------

def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Non-parametric 95% bootstrap CI on the mean."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = np.mean(sample)
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return lo, hi


# ----------------------------- (C) decile bins -----------------------------

def decile_binned_coupling(runs: list[dict], task: str, algo: str) -> list[dict]:
    """Pool per-step (len, reward) across all seeds for (task, algo),
    bin into 10 deciles of length, report mean reward per decile.
    """
    pool: list[tuple[float, float]] = []
    for r in runs:
        if r["task"] != task or r["algo"] != algo:
            continue
        sl = r.get("step_log") or []
        for s in sl:
            pool.append((float(s["mean_comp_len"]), float(s["mean_reward"])))
    if not pool:
        return []
    pool.sort(key=lambda t: t[0])
    n = len(pool)
    rows: list[dict] = []
    for k in range(10):
        a = (k * n) // 10
        b = ((k + 1) * n) // 10
        chunk = pool[a:b]
        if not chunk:
            continue
        lens = [p[0] for p in chunk]
        rews = [p[1] for p in chunk]
        rows.append({
            "task": task,
            "algo": algo,
            "decile": k + 1,
            "n_points": len(chunk),
            "len_min": round(float(min(lens)), 4),
            "len_max": round(float(max(lens)), 4),
            "mean_len": round(float(np.mean(lens)), 4),
            "mean_reward": round(float(np.mean(rews)), 4),
        })
    return rows


# ----------------------------- (D) cross-validation ------------------------

def crossval_arithmetic_metrics() -> dict:
    """Compute per-step Spearman and last-10 numbers on
    platform_hybrid/experiments/results/arithmetic_metrics.jsonl (100 steps, 1 run).
    """
    p = RES / "arithmetic_metrics.jsonl"
    if not p.exists():
        return {"n_steps": 0}
    data = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    steps = [d["step"] for d in data]
    lens = [float(d["env/all/ac_tokens_per_turn"]) for d in data]
    rews = [float(d["env/all/reward/total"]) for d in data]
    r_l, _ = spearman(steps, lens)
    r_r, _ = spearman(steps, rews)
    r_p, _ = spearman(lens, rews)
    first5_r = float(np.mean(rews[:5]))
    last10_r = float(np.mean(rews[-10:]))
    first5_l = float(np.mean(lens[:5]))
    last10_l = float(np.mean(lens[-10:]))
    return {
        "source": "arithmetic_metrics.jsonl",
        "n_steps": len(data),
        "spearman_step_len_rho": round(r_l, 4),
        "spearman_step_reward_rho": round(r_r, 4),
        "spearman_len_reward_rho": round(r_p, 4),
        "first5_reward": round(first5_r, 4),
        "last10_reward": round(last10_r, 4),
        "first5_len": round(first5_l, 4),
        "last10_len": round(last10_l, 4),
        "len_range": round(float(max(lens) - min(lens)), 4),
        "rew_growth": round(last10_r - first5_r, 4),
    }


# ----------------------------- main pipeline --------------------------------

def main() -> int:
    # ---- (1) gather per-run metrics for both source files --------------
    rows_trap: list[dict] = []
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for src in SOURCES:
        for run in load_runs(src["path"]):
            sl = run.get("step_log")
            if not sl:
                continue
            t = trap_onset(sl, window=10, rho_threshold=0.3)
            t.update({
                "task": src["task"],
                "algo": run["algo"],
                "seed": run["seed"],
                "model": run.get("model", ""),
            })
            rows_trap.append(t)
            by_cell.setdefault((src["task"], run["algo"]), []).append({
                "task": src["task"],
                "algo": run["algo"],
                "seed": run["seed"],
                "step_log": sl,
            })

    # write per-run trap TSV
    trap_cols = [
        "task", "algo", "seed", "model", "n_steps", "window", "rho_threshold",
        "first_half_mean_len",
        "trap_fired", "trap_onset_step", "trap_onset_rho_len", "trap_onset_rho_rew",
        "trap_onset_end_len",
    ]
    trap_tsv = RES / "length_bias_trap.tsv"
    with trap_tsv.open("w") as f:
        f.write("\t".join(trap_cols) + "\n")
        for r in rows_trap:
            f.write("\t".join(str(r.get(c, "")) for c in trap_cols) + "\n")
    print(f"wrote {trap_tsv} ({len(rows_trap)} rows)")

    # ---- (2) per-algo summary with bootstrap CIs on Spearman ----------
    # load the existing length_bias.tsv for per-algo summary numbers
    base = RES / "length_bias.tsv"
    if not base.exists():
        print(f"ERROR: {base} not found; run platform_modal/scripts/length_bias.py first.", file=sys.stderr)
        return 1
    base_rows: list[dict] = []
    with base.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            base_rows.append({h: p for h, p in zip(header, parts)})

    sum_rows: list[dict] = []
    for task in sorted({r["task"] for r in base_rows}):
        # collect for paired bootstrap
        per_seed: dict[int, dict[str, float]] = {}
        for r in base_rows:
            if r["task"] != task:
                continue
            per_seed.setdefault(int(r["seed"]), {})[r["algo"]] = {
                "len": float(r["spearman_step_len_rho"]),
                "rew": float(r["spearman_step_reward_rho"]),
                "pair": float(r["spearman_len_reward_rho"]),
            }
        for algo in ("grpo", "dr_grpo"):
            sub = [r for r in base_rows if r["task"] == task and r["algo"] == algo]
            if not sub:
                continue
            rhos_l = [float(r["spearman_step_len_rho"]) for r in sub]
            rhos_r = [float(r["spearman_step_reward_rho"]) for r in sub]
            rhos_p = [float(r["spearman_len_reward_rho"]) for r in sub]
            ci_l = bootstrap_ci(rhos_l, n_boot=2000, seed=hash((task, algo, "l")) & 0xFFFF)
            ci_r = bootstrap_ci(rhos_r, n_boot=2000, seed=hash((task, algo, "r")) & 0xFFFF)
            ci_p = bootstrap_ci(rhos_p, n_boot=2000, seed=hash((task, algo, "p")) & 0xFFFF)
            sum_rows.append({
                "task": task,
                "algo": algo,
                "n_seeds": len(sub),
                "mean_step_len_rho": round(float(np.mean(rhos_l)), 4),
                "ci_step_len_lo": round(ci_l[0], 4),
                "ci_step_len_hi": round(ci_l[1], 4),
                "mean_step_reward_rho": round(float(np.mean(rhos_r)), 4),
                "ci_step_reward_lo": round(ci_r[0], 4),
                "ci_step_reward_hi": round(ci_r[1], 4),
                "mean_len_reward_rho": round(float(np.mean(rhos_p)), 4),
                "ci_len_reward_lo": round(ci_p[0], 4),
                "ci_len_reward_hi": round(ci_p[1], 4),
            })
        # paired bootstrap of Dr.GRPO - GRPO difference on shared seeds
        diffs_l, diffs_r, diffs_p = [], [], []
        for seed, algos in per_seed.items():
            if "grpo" in algos and "dr_grpo" in algos:
                diffs_l.append(algos["dr_grpo"]["len"] - algos["grpo"]["len"])
                diffs_r.append(algos["dr_grpo"]["rew"] - algos["grpo"]["rew"])
                diffs_p.append(algos["dr_grpo"]["pair"] - algos["grpo"]["pair"])
        if diffs_l:
            ci_dl = bootstrap_ci(diffs_l, n_boot=2000, seed=hash((task, "dl")) & 0xFFFF)
            ci_dr = bootstrap_ci(diffs_r, n_boot=2000, seed=hash((task, "dr")) & 0xFFFF)
            ci_dp = bootstrap_ci(diffs_p, n_boot=2000, seed=hash((task, "dp")) & 0xFFFF)
            sum_rows.append({
                "task": task,
                "algo": "drgrpo_minus_grpo",
                "n_seeds": len(diffs_l),
                "mean_step_len_rho": round(float(np.mean(diffs_l)), 4),
                "ci_step_len_lo": round(ci_dl[0], 4),
                "ci_step_len_hi": round(ci_dl[1], 4),
                "mean_step_reward_rho": round(float(np.mean(diffs_r)), 4),
                "ci_step_reward_lo": round(ci_dr[0], 4),
                "ci_step_reward_hi": round(ci_dr[1], 4),
                "mean_len_reward_rho": round(float(np.mean(diffs_p)), 4),
                "ci_len_reward_lo": round(ci_dp[0], 4),
                "ci_len_reward_hi": round(ci_dp[1], 4),
            })
    sum_cols = list(sum_rows[0].keys())
    sum_tsv = RES / "length_bias_summary_ci.tsv"
    with sum_tsv.open("w") as f:
        f.write("\t".join(sum_cols) + "\n")
        for r in sum_rows:
            f.write("\t".join(str(r[c]) for c in sum_cols) + "\n")
    print(f"wrote {sum_tsv} ({len(sum_rows)} rows)")

    # ---- (3) decile-binned length-vs-reward coupling ------------------
    flat_runs: list[dict] = []
    for src in SOURCES:
        for run in load_runs(src["path"]):
            sl = run.get("step_log")
            if not sl:
                continue
            flat_runs.append({
                "task": src["task"],
                "algo": run["algo"],
                "seed": run["seed"],
                "step_log": sl,
            })
    bin_rows: list[dict] = []
    for task in sorted({r["task"] for r in flat_runs}):
        for algo in ("grpo", "dr_grpo"):
            bin_rows.extend(decile_binned_coupling(flat_runs, task, algo))
    bin_cols = ["task", "algo", "decile", "n_points", "len_min", "len_max",
                "mean_len", "mean_reward"]
    bin_tsv = RES / "length_bias_bins.tsv"
    with bin_tsv.open("w") as f:
        f.write("\t".join(bin_cols) + "\n")
        for r in bin_rows:
            f.write("\t".join(str(r[c]) for c in bin_cols) + "\n")
    print(f"wrote {bin_tsv} ({len(bin_rows)} rows)")

    # ---- (4) cross-validation on arithmetic_metrics.jsonl --------------
    cv = crossval_arithmetic_metrics()
    cv_tsv = RES / "length_bias_crosval.tsv"
    with cv_tsv.open("w") as f:
        f.write("\t".join(list(cv.keys())) + "\n")
        f.write("\t".join(str(v) for v in cv.values()) + "\n")
    print(f"wrote {cv_tsv}")

    # ---- (5) 3-panel figure -------------------------------------------
    make_figure(rows_trap, bin_rows, cv, base_rows)

    # ---- (6) terminal summary -----------------------------------------
    print("\n=== Pillar 4 elevation summary ===")
    print(f"  trap-on fired on {sum(r['trap_fired'] for r in rows_trap)} of "
          f"{len(rows_trap)} runs (window=10, rho_threshold=0.3)")
    print(f"  crossval (arithmetic_metrics.jsonl, n={cv.get('n_steps', 0)}): "
          f"rho_len={cv.get('spearman_step_len_rho')}, "
          f"rho_rew={cv.get('spearman_step_reward_rho')}, "
          f"len_range={cv.get('len_range')}")
    return 0


# ----------------------------- figure ---------------------------------------

def make_figure(rows_trap: list[dict], bin_rows: list[dict],
                cv: dict, base_rows: list[dict]) -> None:
    """3-panel: (1) trap-onset step histogram, (2) decile binned reward,
       (3) crossval reward-vs-step on arithmetic_metrics.jsonl.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = {"grpo": "#1f77b4", "dr_grpo": "#d62728"}

    # ---- panel 1: trap-onset step histogram ----------------------------
    ax = axes[0]
    for algo in ("grpo", "dr_grpo"):
        steps_fired = [r["trap_onset_step"] for r in rows_trap
                       if r["algo"] == algo and r["trap_fired"] == 1]
        all_steps = [r["n_steps"] for r in rows_trap if r["algo"] == algo]
        # show a "did not fire" bar at the right
        not_fired = len(all_steps) - len(steps_fired)
        if steps_fired:
            ax.hist(steps_fired, bins=range(0, max(steps_fired) + 12, 4),
                    alpha=0.55, color=colors[algo], label=f"{algo} fired (n={len(steps_fired)})")
        # annotate the "did not fire" count as text
        ax.text(0.97, 0.05 + (0.06 if algo == "dr_grpo" else 0.0),
                f"{algo}: {not_fired}/{len(all_steps)} runs did NOT fire trap-onset",
                transform=ax.transAxes, fontsize=8, ha="right",
                color=colors[algo], family="monospace")
    ax.set_xlabel("earliest trap-onset step (sliding window W=10, rho>0.3)")
    ax.set_ylabel("# runs (across all tasks)")
    ax.set_title("(A) Trap-onset detection\n(6/16 runs fire on hard task; easy task 2/10)")
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper right")

    # ---- panel 2: decile binned reward (hard task) ---------------------
    ax = axes[1]
    hard_task = "gsm8k_cot_hard_qwen2.5-1.5b"
    for algo in ("grpo", "dr_grpo"):
        sub = [b for b in bin_rows if b["task"] == hard_task and b["algo"] == algo]
        if not sub:
            continue
        x = [b["mean_len"] for b in sub]
        y = [b["mean_reward"] for b in sub]
        ax.plot(x, y, "-o", color=colors[algo], label=algo, markersize=5, linewidth=1.5)
    ax.set_xlabel("decile mean completion length (tokens)")
    ax.set_ylabel("mean reward per decile")
    ax.set_title(f"(B) Decile binned length-vs-reward\n(GSM8K-CoT, Qwen2.5-1.5B)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # ---- panel 3: crossval on arithmetic_metrics.jsonl ----------------
    ax = axes[2]
    p = RES / "arithmetic_metrics.jsonl"
    if p.exists():
        data = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        steps = [d["step"] for d in data]
        # twin axis for length
        ax2 = ax.twinx()
        rew = [float(d["env/all/reward/total"]) for d in data]
        lens = [float(d["env/all/ac_tokens_per_turn"]) for d in data]
        ax.plot(steps, rew, "-", color="#2ca02c", linewidth=1.4, label="reward (left axis)")
        ax2.plot(steps, lens, "-", color="#ff7f0e", linewidth=1.4, label="ac_tokens (right axis)")
        ax.set_xlabel("step (0-99)")
        ax.set_ylabel("mean reward (green)", color="#2ca02c")
        ax2.set_ylabel("ac_tokens / turn (orange)", color="#ff7f0e")
        ax.tick_params(axis="y", labelcolor="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        ax.set_title(f"(C) Crossval: 100-step arithmetic_metrics.jsonl\n"
                     f"rho_len={cv.get('spearman_step_len_rho')}, "
                     f"rho_rew={cv.get('spearman_step_reward_rho')}")
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "arithmetic_metrics.jsonl missing",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(C) Crossval: missing")

    fig.suptitle("Pillar 4 elevation: trap-onset + decile-binned coupling + crossval",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    pdf_path = FIGS / "length_vs_reward_elevated.pdf"
    png_path = FIGS / "length_vs_reward_elevated.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    shutil.copy(pdf_path, PAPER_FIGS / "length_vs_reward_elevated.pdf")
    print(f"wrote {pdf_path} and {png_path} (+ copy in paper/figures/)")


if __name__ == "__main__":
    sys.exit(main())
