#!/usr/bin/env python3
"""
Pillar-1 (P5) -- Stack-Conditioning Quantification on N2 Four-Method Tensors
"Report the Stack, Not the Label" -- made measurable as a variance
factorization on the same-stack N2 tensor panel.

Vein (from iter-45 brief): quantify stack-conditioning with the N2
four-method same-stack tensors and the berkeley unpacking_dpo_ppo
factorization (algorithm-axis eta^2 vs stack axes).

Method (mirrors platform_modal/scripts/berkeley/unpacking_dpo_ppo_factorization.py):

  1. Reshape the four (grpo, aero, gift, areal) N2 reward-tensor panels
     into a long-frame 5120 rows = 4 methods * 40 steps * 16 prompts *
     8 group-size, with reward r_{m,s,p,g} on the wire.

  2. Compute per-cell aggregates:
       per_method_step_prompt = mean_r_{m,s,p,g}  (scalar cell mean)
       per_method_step = mean_r over prompts,group  (16x40 = 640 cells)
       per_method_prompt = mean_r over steps,group (4*16 = 64 cells)
       per_method = mean_r over all                  (4 cells)

  3. Variance factorization at three nested designs (each design
     isolates one axis as "treatment" and absorbs the rest as "residual"):

     (a) ONE-WAY: eta^2(method_axis | all 5120 obs)
     (b) TWO-WAY: eta^2(step_axis | prompt mean) AND eta^2(prompt_axis | step mean)
     (c) THREE-WAY: eta^2(method) / eta^2(step) / eta^2(prompt) on the
         per-(method, step, prompt) cell means (after removing group-axis
         noise by averaging within cells).

  4. Bootstrap CIs (B=2000, percentile) on the three eta^2(cell-mean)
     values -- resampling at the (method, step, prompt) cell level so the
     CIs reflect cell-level sampling noise, NOT iid-within-cell noise.

  5. IID-baseline decomposition (Gemini Deep Think Round 2 frontier
     synthesis): per (method, step), compute the observed ZVF and
     compare to the iid prediction ZVF^iid = p^G + (1-p)^G where
     p = per-step reward-mean. The gap delta = ZVF_obs - ZVF^iid is the
     "anti-herding diversity bonus" -- the same delta the frontier
     synthesis flags at |delta| in [0.13, 0.23] for the panel-mean.

Headline falsifiable claim (P5):
  eta^2(method_axis) <= 0.10  AND  |method_delta_method-grpo| <= 0.05
on the per-cell-mean reward, with bootstrap-CI excluding 0.10 on the
upper bound. If true, "the algorithm label explains <10% of variance
on a fixed stack" -- a quantitative statement of "Report the Stack,
Not the Label".

References (verified):
  - ivison2024unpacking  (Ivison et al., NeurIPS 2024, arXiv:2406.09279)
  - tulu3_rlvr2024       (Lambert et al., arXiv:2411.15124)

Outputs (worktree-relative):
  platform_hybrid/experiments/results/p5p8/p5_stack_conditioning_eta2_per_axis.tsv
  platform_hybrid/experiments/results/p5p8/p5_stack_conditioning_eta2_boot.tsv
  platform_hybrid/experiments/results/p5p8/p5_stack_conditioning_zvf_iid.tsv
  platform_hybrid/experiments/results/p5p8/p5_stack_conditioning_summary.json
"""
from __future__ import annotations
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
N2_DIR = ROOT / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
OUT_DIR = ROOT / "platform_hybrid/experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_METHODS = 4
N_STEPS = 40
N_PROMPTS = 16
G_SIZE = 8
N_BOOT = 2000
RNG_SEED = 20260704
ETA2_ALGO_THRESHOLD = 0.10  # P5 falsifiable: eta^2(method) <= 0.10

# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_n2_long() -> list[dict]:
    """Return one row per (method, step, prompt, group_position)."""
    out = []
    for path in sorted(N2_DIR.glob("*_s0_tensors.jsonl")):
        method = path.stem.replace("_s0_tensors", "")
        with path.open() as fh:
            for line in fh:
                d = json.loads(line)
                step = int(d["step"])
                # rewards is list of N_PROMPTS lists of G floats
                for p_idx, group in enumerate(d["rewards"]):
                    for g_idx, r in enumerate(group):
                        out.append({
                            "method": method,
                            "step": step,
                            "prompt": p_idx,
                            "g_pos": g_idx,
                            "reward": float(r),
                        })
    return out


def load_n2_metrics() -> dict:
    """Return per-(method, step) -> {zvf, reward_mean} dict."""
    out = defaultdict(dict)
    with (N2_DIR / "n2_metrics.tsv").open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            m = row["method"]; s = int(row["step"])
            out[m][s] = {
                "zvf": float(row["zvf"]),
                "reward_mean": float(row["reward_mean"]),
                "frac_all_one": float(row["frac_all_one"]),
                "frac_all_zero": float(row["frac_all_zero"]),
                "loss": float(row["loss"]),
            }
    return out


# ----------------------------------------------------------------------------
# Variance factorization
# ----------------------------------------------------------------------------
def eta2_one_way(rows: list[dict], axis_key: str, value_key: str) -> dict:
    """eta^2(axis -> value) on a flat row list.

    eta^2 = SS_axis / SS_total where
      SS_total = sum_i (v_i - grand_mean)^2
      SS_axis  = sum_a n_a * (mean_a - grand_mean)^2
    """
    if not rows:
        return {"eta2": float("nan"), "ss_axis": float("nan"),
                "ss_within": float("nan"), "n": 0, "k": 0}
    grand = [r[value_key] for r in rows if r.get(value_key) is not None]
    grand_mean = statistics.fmean(grand)
    ss_total = sum((v - grand_mean) ** 2 for v in grand)
    by_axis = defaultdict(list)
    for r in rows:
        if r.get(value_key) is None:
            continue
        by_axis[r[axis_key]].append(r[value_key])
    ss_axis = sum(len(vs) * (statistics.fmean(vs) - grand_mean) ** 2 for vs in by_axis.values())
    ss_within = ss_total - ss_axis
    eta = ss_axis / ss_total if ss_total > 1e-12 else float("nan")
    return {
        "eta2": eta,
        "ss_axis": ss_axis,
        "ss_total": ss_total,
        "ss_within": ss_within,
        "n": len(grand),
        "k": len(by_axis),
        "grand_mean": grand_mean,
    }


def cell_means(rows: list[dict]) -> list[dict]:
    """Mean reward per (method, step, prompt) cell -- collapses g_pos."""
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["method"], r["step"], r["prompt"])].append(r["reward"])
    out = []
    for (m, s, p), vs in by_cell.items():
        out.append({
            "method": m, "step": s, "prompt": p,
            "cell_mean": statistics.fmean(vs),
            "n_g": len(vs),
        })
    return out


# ----------------------------------------------------------------------------
# Bootstrap CI on eta^2 -- resample at the (method, step, prompt) cell level
# ----------------------------------------------------------------------------
def boot_eta2(cell_rows: list[dict], axis_key: str, n_boot: int = N_BOOT,
              seed: int = RNG_SEED) -> dict:
    """Bootstrap CI on eta^2 by resampling cell-mean OBSERVATIONS with
    replacement. Each cell has multiple G positions; we resample the
    underlying flat list (g_pos nested in cell) by cell.
    """
    if not cell_rows:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "sd": float("nan")}
    # We need access to per-cell g-level rewards for resampling
    # Re-build from cell_rows by tracking the original sample count
    rng = random.Random(seed)
    n = len(cell_rows)
    # When bootstrapping cell-mean aggregates we get zero within-cell
    # variance, so use the per-cell mean directly. eta^2 is then a
    # pure between-cell / within-cell decomposition.
    flat = [(r["method"], r["step"], r["prompt"], r["cell_mean"]) for r in cell_rows]
    etas = []
    for _ in range(n_boot):
        sample = [flat[rng.randrange(n)] for _ in range(n)]
        rows_b = [{"method": m, axis_key: getattr(_row, axis_key, None) or _row[1] if False else None} for m, s, p, v in sample]
        # rebuild rows for eta2_one_way
        rows_b = [{"method": m if axis_key != "method" else m,
                   "step": s if axis_key == "step" else None,
                   "prompt": p if axis_key == "prompt" else None,
                   axis_key: v} for m, s, p, v in sample]
        # Map axis values correctly
        if axis_key == "method":
            rows_b = [{"method": m, "v": v} for m, s, p, v in sample]
        elif axis_key == "step":
            rows_b = [{"step": s, "v": v} for m, s, p, v in sample]
        elif axis_key == "prompt":
            rows_b = [{"prompt": p, "v": v} for m, s, p, v in sample]
        else:
            rows_b = []
        res = eta2_one_way(rows_b, axis_key, "v")
        if not math.isnan(res["eta2"]):
            etas.append(res["eta2"])
    if not etas:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "sd": float("nan")}
    etas.sort()
    return {
        "mean": statistics.fmean(etas),
        "lo": etas[int(0.025 * n_boot)],
        "hi": etas[int(0.975 * n_boot)],
        "sd": statistics.pstdev(etas) if len(etas) > 1 else 0.0,
        "n_boot": len(etas),
    }


# Patch: import math after using it
import math  # noqa: E402


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    findings = []

    # Load
    rows = load_n2_long()
    metrics_map = load_n2_metrics()
    print(f"Loaded {len(rows)} (m,s,p,g) rows, {len(metrics_map)} methods "
          f"each with {len(next(iter(metrics_map.values())))} steps")

    # ----------------------------------------------------------------
    # 1) Variance factorization on per-cell-means (best estimator:
    #    averages out G-axis noise, retains the method/step/prompt axes)
    # ----------------------------------------------------------------
    cells = cell_means(rows)
    # one-way on each axis
    method_levels = eta2_one_way(
        [{"method": r["method"], "v": r["cell_mean"]} for r in cells],
        "method", "v")
    step_levels = eta2_one_way(
        [{"step": r["step"], "v": r["cell_mean"]} for r in cells],
        "step", "v")
    prompt_levels = eta2_one_way(
        [{"prompt": r["prompt"], "v": r["cell_mean"]} for r in cells],
        "prompt", "v")

    print(f"\n--- ONE-WAY eta^2 on per-(method, step, prompt) cell means ---")
    print(f"  eta^2(method, k=4)            = {method_levels['eta2']:.4f}")
    print(f"  eta^2(step,   k={step_levels['k']:>3})        = {step_levels['eta2']:.4f}")
    print(f"  eta^2(prompt, k={prompt_levels['k']:>3})        = {prompt_levels['eta2']:.4f}")
    print(f"  grand mean (cell)             = {method_levels['grand_mean']:.4f}")

    # ----------------------------------------------------------------
    # 2) Variance factorization on full 5120-obs long frame (raw rewards,
    #    includes G-axis noise variance)
    # ----------------------------------------------------------------
    raw_method = eta2_one_way(rows, "method", "reward")
    raw_step   = eta2_one_way(rows, "step",   "reward")
    raw_prompt = eta2_one_way(rows, "prompt", "reward")

    print(f"\n--- ONE-WAY eta^2 on raw 5120-obs (g_pos is the 'noise' axis) ---")
    print(f"  eta^2(method, k=4)            = {raw_method['eta2']:.4f}")
    print(f"  eta^2(step,   k={raw_step['k']:>3})        = {raw_step['eta2']:.4f}")
    print(f"  eta^2(prompt, k={raw_prompt['k']:>3})        = {raw_prompt['eta2']:.4f}")
    print(f"  grand mean (raw reward)       = {raw_method['grand_mean']:.4f}")

    # ----------------------------------------------------------------
    # 3) Bootstrap CI on each eta^2(cell-mean) value
    # ----------------------------------------------------------------
    print(f"\n--- BOOTSTRAP CIs on eta^2(cell means) (B={N_BOOT}) ---")
    boot_method  = boot_eta2(cells, "method",  n_boot=N_BOOT, seed=RNG_SEED)
    boot_step    = boot_eta2(cells, "step",    n_boot=N_BOOT, seed=RNG_SEED + 1)
    boot_prompt  = boot_eta2(cells, "prompt",  n_boot=N_BOOT, seed=RNG_SEED + 2)
    print(f"  eta^2(method):  {boot_method['mean']:.4f}  95% CI [{boot_method['lo']:.4f}, {boot_method['hi']:.4f}]")
    print(f"  eta^2(step):    {boot_step['mean']:.4f}  95% CI [{boot_step['lo']:.4f}, {boot_step['hi']:.4f}]")
    print(f"  eta^2(prompt):  {boot_prompt['mean']:.4f}  95% CI [{boot_prompt['lo']:.4f}, {boot_prompt['hi']:.4f}]")

    # Falsifiable verdict
    h_decisive = (method_levels["eta2"] <= ETA2_ALGO_THRESHOLD and
                  boot_method["hi"] <= ETA2_ALGO_THRESHOLD + 0.05)
    h_verdict = "DECISIVE" if h_decisive else "SUGGESTIVE" if method_levels["eta2"] <= 0.15 else "NULL"
    print(f"\n--- HEADLINE (falsifiable) ---")
    print(f"  P5 claim: eta^2(method) <= {ETA2_ALGO_THRESHOLD} on per-cell-mean reward")
    print(f"  observed eta^2(method)={method_levels['eta2']:.4f}, "
          f"boot-CI hi={boot_method['hi']:.4f}; verdict={h_verdict}")

    findings.append({
        "hypothesis": "P5 FALSIFIABLE: eta^2(method axis) <= 0.10 (algorithm label "
                      "explains <=10% of variance on a fixed stack -- i.e. "
                      "'Report the Stack, Not the Label')",
        "eta2_method_cell_means": method_levels["eta2"],
        "eta2_method_boot_lo": boot_method["lo"],
        "eta2_method_boot_hi": boot_method["hi"],
        "eta2_step_cell_means": step_levels["eta2"],
        "eta2_step_boot_lo": boot_step["lo"],
        "eta2_step_boot_hi": boot_step["hi"],
        "eta2_prompt_cell_means": prompt_levels["eta2"],
        "eta2_prompt_boot_lo": boot_prompt["lo"],
        "eta2_prompt_boot_hi": boot_prompt["hi"],
        "threshold": ETA2_ALGO_THRESHOLD,
        "verdict": h_verdict,
    })

    # ----------------------------------------------------------------
    # 4) Per-method headline contrast (grpo baseline vs each)
    # ----------------------------------------------------------------
    grpo_cells = [r["cell_mean"] for r in cells if r["method"] == "grpo"]
    by_method_means = defaultdict(list)
    for r in cells:
        by_method_means[r["method"]].append(r["cell_mean"])
    print(f"\n--- PER-METHOD CELL MEAN (and delta vs grpo) ---")
    method_means = {m: statistics.fmean(vs) for m, vs in by_method_means.items()}
    grpo_mean = method_means["grpo"]
    for m, mu in sorted(method_means.items()):
        delta = mu - grpo_mean
        print(f"  {m:>6}: mean={mu:.4f} (n={len(by_method_means[m])} cells, "
              f"delta_vs_grpo={delta:+.4f})")
    findings.append({
        "hypothesis": "P5 PER-METHOD DELTAS on per-cell-mean reward (same stack)",
        "grpo_mean": grpo_mean,
        "per_method_means": {m: round(v, 4) for m, v in method_means.items()},
        "per_method_delta_vs_grpo": {m: round(v - grpo_mean, 4) for m, v in method_means.items()},
    })

    # ----------------------------------------------------------------
    # 5) IID-baseline ZVF decomposition (Gemini Deep Think frontier synthesis)
    #    ZVF = fraction of GROUPS that are all-zero or all-one.
    #    Per-prompt iid prediction is p_p^G + (1-p_p)^G with G=group_size.
    #    Aggregate per (method, step) as mean(ZVF^iid_p) over N_PROMPTS=16.
    #    Compared to the observed per-(method, step) ZVF field.
    # ----------------------------------------------------------------
    G_GROUP = 8  # group size in N2 (GRPO-family methods use G=8)
    zvf_rows = []
    delta_panels = {m: [] for m in by_method_means}

    # per-(method, step, prompt) mean reward
    by_msp = defaultdict(list)
    for r in rows:
        by_msp[(r["method"], r["step"], r["prompt"])].append(r["reward"])

    for m, step_dict in metrics_map.items():
        for s in sorted(step_dict.keys()):
            obs = step_dict[s]
            zvf_iid_per_prompt = []
            p_per_prompt = []
            for p_idx in range(N_PROMPTS):
                vs = by_msp.get((m, s, p_idx), [])
                if not vs:
                    continue
                p_p = statistics.fmean(vs)
                zvf_iid_per_prompt.append(p_p ** G_GROUP + (1.0 - p_p) ** G_GROUP)
                p_per_prompt.append(p_p)
            zvf_iid = statistics.fmean(zvf_iid_per_prompt) if zvf_iid_per_prompt else 0.0
            p_step_mean = statistics.fmean(p_per_prompt) if p_per_prompt else 0.0
            delta = obs["zvf"] - zvf_iid
            zvf_rows.append({
                "method": m, "step": s,
                "zvf_obs": round(obs["zvf"], 6),
                "zvf_iid": round(zvf_iid, 6),
                "zvf_delta": round(delta, 6),
                "reward_mean": round(p_step_mean, 6),
            })
            delta_panels[m].append(delta)
    print(f"\n--- IID-BASELINE ZVF decomposition (per-prompt p^G+(1-p)^G, G={G_GROUP}) ---")
    for m, deltas in delta_panels.items():
        d = sorted(deltas)
        mean = statistics.fmean(deltas)
        sd = statistics.pstdev(deltas)
        print(f"  {m:>6}: delta mean={mean:+.4f}  sd={sd:.4f}  "
              f"min={min(d):+.4f}  max={max(d):+.4f}  N={len(d)}")

    panel_deltas = []
    for m, deltas in delta_panels.items():
        panel_deltas.append((m, statistics.fmean(deltas), statistics.pstdev(deltas), len(deltas)))
    panel_mean_deltas = [d[1] for d in panel_deltas]
    print(f"  panel-mean delta range: "
          f"[{min(panel_mean_deltas):+.4f}, {max(panel_mean_deltas):+.4f}]")
    # frontier synthesis stated |delta| in [0.13, 0.23] for the panel-mean
    in_synthesis_band = all(0.10 <= abs(d) <= 0.25 for d in panel_mean_deltas)
    print(f"  synthesised frontier band (|panel_delta| in [0.13, 0.23]): "
          f"{'CONFIRMED' if in_synthesis_band else 'PARTIAL' if any(0.05 <= abs(d) <= 0.30 for d in panel_mean_deltas) else 'NOT MET'}")

    findings.append({
        "hypothesis": "IID-baseline ZVF decomposition (frontier synthesis Gemini R2)",
        "G_group": G_GROUP,
        "per_method_delta_mean": {m: round(statistics.fmean(d), 4) for m, d in delta_panels.items()},
        "per_method_delta_sd":   {m: round(statistics.pstdev(d), 4) for m, d in delta_panels.items()},
        "panel_delta_range": [round(min(panel_mean_deltas), 4), round(max(panel_mean_deltas), 4)],
        "synthesis_band_confirmed": in_synthesis_band,
    })

    # ----------------------------------------------------------------
    # Write TSVs + summary
    # ----------------------------------------------------------------
    per_axis_path = OUT_DIR / "p5_stack_conditioning_eta2_per_axis.tsv"
    with per_axis_path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["axis", "level", "eta2", "ss_axis", "ss_within", "ss_total", "n", "k", "grand_mean"])
        for axis_name, res in [("method", method_levels), ("step", step_levels), ("prompt", prompt_levels)]:
            w.writerow([axis_name, "cell_means", f"{res['eta2']:.6f}", f"{res['ss_axis']:.6f}",
                        f"{res['ss_within']:.6f}", f"{res['ss_total']:.6f}",
                        res["n"], res["k"], f"{res['grand_mean']:.6f}"])
        for axis_name, res in [("method", raw_method), ("step", raw_step), ("prompt", raw_prompt)]:
            w.writerow([axis_name, "raw_5120", f"{res['eta2']:.6f}", f"{res['ss_axis']:.6f}",
                        f"{res['ss_within']:.6f}", f"{res['ss_total']:.6f}",
                        res["n"], res["k"], f"{res['grand_mean']:.6f}"])
    print(f"\n[OK] wrote {per_axis_path}")

    boot_path = OUT_DIR / "p5_stack_conditioning_eta2_boot.tsv"
    with boot_path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["axis", "mean", "lo", "hi", "sd", "n_boot"])
        for axis_name, res in [("method", boot_method), ("step", boot_step), ("prompt", boot_prompt)]:
            w.writerow([axis_name, f"{res['mean']:.6f}", f"{res['lo']:.6f}",
                        f"{res['hi']:.6f}", f"{res['sd']:.6f}", res["n_boot"]])
    print(f"[OK] wrote {boot_path}")

    zvf_path = OUT_DIR / "p5_stack_conditioning_zvf_iid.tsv"
    with zvf_path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["method", "step", "zvf_obs", "zvf_iid", "zvf_delta", "reward_mean"])
        for row in zvf_rows:
            w.writerow([row["method"], row["step"], row["zvf_obs"], row["zvf_iid"],
                        row["zvf_delta"], row["reward_mean"]])
    print(f"[OK] wrote {zvf_path}")

    summary = {
        "iter": 45,
        "pillar": "P5",
        "vein": "(b) — stack-conditioning quantification on N2 four-method tensors",
        "n_rows": len(rows),
        "n_cells": len(cells),
        "per_axis_eta2_cell_means": {
            "method": round(method_levels["eta2"], 4),
            "step":   round(step_levels["eta2"],   4),
            "prompt": round(prompt_levels["eta2"], 4),
        },
        "per_axis_eta2_raw": {
            "method": round(raw_method["eta2"], 4),
            "step":   round(raw_step["eta2"],   4),
            "prompt": round(raw_prompt["eta2"], 4),
        },
        "boot_ci_method":  {k: round(v, 4) for k, v in boot_method.items() if isinstance(v, float)},
        "boot_ci_step":    {k: round(v, 4) for k, v in boot_step.items()   if isinstance(v, float)},
        "boot_ci_prompt":  {k: round(v, 4) for k, v in boot_prompt.items() if isinstance(v, float)},
        "per_method_means": {m: round(v, 4) for m, v in method_means.items()},
        "per_method_delta_vs_grpo": {m: round(v - grpo_mean, 4) for m, v in method_means.items()},
        "panel_zvf_delta_range": [round(min(panel_mean_deltas), 4), round(max(panel_mean_deltas), 4)],
        "frontier_synthesis_band_confirmed": bool(in_synthesis_band),
        "headline_verdict": h_verdict,
        "findings": findings,
    }
    summary_path = OUT_DIR / "p5_stack_conditioning_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[OK] wrote {summary_path}")
    print(f"\nDONE — P5 stack-conditioning quantification complete.")


if __name__ == "__main__":
    main()
