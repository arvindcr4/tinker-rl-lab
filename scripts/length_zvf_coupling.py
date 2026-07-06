#!/usr/bin/env python3
"""
length_zvf_coupling.py — Pillar 4 cross-coupling with Pillar 2 ZVF.

Iter 12 novelty: per-step joint coupling of (length, ZVF, reward).
Hypothesis tested:
  H0 (Dr.GRPO/ZVF disjoint):  length bias and ZVF herding are independent
  H1 (length→ZVF causal):     compressing length reduces within-group contrast
                              (because rollouts converge to the same length
                              distribution → all-correct/all-wrong more likely)
  H2 (ZVF→length causal):     herding forces policy collapse to short, terse
                              answers (efficiency-only response)

For every run we compute:
  * Spearman(step, len), Spearman(step, zvf), Spearman(step, reward)
  * Joint Spearman(len, zvf), Spearman(len, reward), Spearman(zvf, reward)
  * Length-bin conditional ZVF (low/mid/high length segments)

Inputs : experiments/results/drgrpo_vs_grpo.json (Qwen2.5-0.5B arithmetic)
         experiments/results/drgrpo_gsm8k_cot_full.json (Qwen2.5-1.5B GSM8K-CoT)
Outputs: experiments/results/length_zvf_coupling.tsv  (per-run joint spearman)
         experiments/results/length_zvf_bincond.tsv   (per-run length-bin conditional ZVF)
         figures/length_zvf_coupling.pdf
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGS = ROOT / "figures"


def per_step_correlations(step_log):
    """Return joint Spearman correlations from a per-step log."""
    steps = np.array([e["step"] for e in step_log], dtype=float)
    rew = np.array([e["mean_reward"] for e in step_log], dtype=float)
    zvf = np.array([e["zvf"] for e in step_log], dtype=float)
    leng = np.array([e["mean_comp_len"] for e in step_log], dtype=float)

    def _rho(a, b):
        if len(a) < 3:
            return float("nan"), float("nan")
        r, p = spearmanr(a, b)
        return float(r), float(p)

    return {
        "n_steps": len(steps),
        "rho_step_len": _rho(steps, leng),
        "rho_step_zvf": _rho(steps, zvf),
        "rho_step_rew": _rho(steps, rew),
        "rho_len_zvf": _rho(leng, zvf),
        "rho_len_rew": _rho(leng, rew),
        "rho_zvf_rew": _rho(zvf, rew),
        "len_first5": float(leng[:5].mean()),
        "len_last5": float(leng[-5:].mean()),
        "zvf_first5": float(zvf[:5].mean()),
        "zvf_last5": float(zvf[-5:].mean()),
        "rew_first5": float(rew[:5].mean()),
        "rew_last5": float(rew[-5:].mean()),
    }


def length_bin_conditional_zvf(step_log, n_bins=3):
    """Bin steps by mean_comp_len (low/mid/high), report ZVF per bin.

    Tests H1: if compressing length raises ZVF (herding), then the
    low-length bin should have higher mean ZVF than the high-length bin.
    """
    leng = np.array([e["mean_comp_len"] for e in step_log], dtype=float)
    zvf = np.array([e["zvf"] for e in step_log], dtype=float)
    rew = np.array([e["mean_reward"] for e in step_log], dtype=float)

    if len(leng) < n_bins:
        return None

    # Quantile bins so each bin has roughly equal n_steps
    qs = np.quantile(leng, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    out = []
    for i in range(n_bins):
        mask = (leng >= qs[i]) & (leng < qs[i + 1])
        if mask.sum() == 0:
            continue
        out.append({
            "bin": f"L{i+1}",
            "len_min": float(leng[mask].min()),
            "len_max": float(leng[mask].max()),
            "n_steps": int(mask.sum()),
            "mean_len": float(leng[mask].mean()),
            "mean_zvf": float(zvf[mask].mean()),
            "mean_rew": float(rew[mask].mean()),
        })
    return out


def main():
    drgrpo = json.load(open(RESULTS / "drgrpo_vs_grpo.json"))
    gsm = json.load(open(RESULTS / "drgrpo_gsm8k_cot_full.json"))

    all_runs = []
    for r in drgrpo["runs"]:
        all_runs.append(("arithmetic_qwen2.5-0.5b", r["algo"], r["seed"], r))
    for r in gsm["runs"]:
        all_runs.append(("gsm8k_cot_qwen2.5-1.5b", r["algo"], r["seed"], r))

    # Per-run joint spearman
    joint_rows = []
    for task, algo, seed, r in all_runs:
        out = per_step_correlations(r["step_log"])
        joint_rows.append({
            "task": task,
            "algo": algo,
            "seed": seed,
            "model": r["model"],
            "n_steps": out["n_steps"],
            "rho_step_len": out["rho_step_len"][0],
            "rho_step_zvf": out["rho_step_zvf"][0],
            "rho_step_rew": out["rho_step_rew"][0],
            "rho_len_zvf": out["rho_len_zvf"][0],
            "rho_len_zvf_p": out["rho_len_zvf"][1],
            "rho_len_rew": out["rho_len_rew"][0],
            "rho_zvf_rew": out["rho_zvf_rew"][0],
            "len_first5": out["len_first5"],
            "len_last5": out["len_last5"],
            "zvf_first5": out["zvf_first5"],
            "zvf_last5": out["zvf_last5"],
            "rew_first5": out["rew_first5"],
            "rew_last5": out["rew_last5"],
        })

    # Length-bin conditional ZVF
    bin_rows = []
    for task, algo, seed, r in all_runs:
        bins = length_bin_conditional_zvf(r["step_log"], n_bins=3)
        if bins is None:
            continue
        for b in bins:
            bin_rows.append({
                "task": task,
                "algo": algo,
                "seed": seed,
                **b,
            })

    # Write outputs
    out_joint = RESULTS / "length_zvf_coupling.tsv"
    with open(out_joint, "w") as f:
        f.write("\t".join(joint_rows[0].keys()) + "\n")
        for row in joint_rows:
            f.write("\t".join(str(v) for v in row.values()) + "\n")
    print(f"wrote {out_joint} ({len(joint_rows)} rows)")

    out_bin = RESULTS / "length_zvf_bincond.tsv"
    with open(out_bin, "w") as f:
        f.write("\t".join(bin_rows[0].keys()) + "\n")
        for row in bin_rows:
            f.write("\t".join(str(v) for v in row.values()) + "\n")
    print(f"wrote {out_bin} ({len(bin_rows)} rows)")

    # Per-(task, algo) summary: mean rho_len_zvf and per-bin ZVF
    summary = {}
    for row in joint_rows:
        key = (row["task"], row["algo"])
        summary.setdefault(key, []).append(row)
    print("\nPer-(task, algo) mean rho(len, zvf):")
    for key, rows in summary.items():
        rhos = [r["rho_len_zvf"] for r in rows if not np.isnan(r["rho_len_zvf"])]
        if rhos:
            mean = float(np.mean(rhos))
            sd = float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0
            print(f"  {key}: n={len(rhos)} mean={mean:+.3f} sd={sd:.3f}")

    print("\nPer-(task, algo) bin-conditional ZVF (low/high length):")
    bin_summary = {}
    for row in bin_rows:
        key = (row["task"], row["algo"], row["bin"])
        bin_summary.setdefault(key, []).append(row["mean_zvf"])
    for key, vals in bin_summary.items():
        arr = np.array(vals)
        print(f"  {key}: mean ZVF={arr.mean():.3f} (sd={arr.std(ddof=1) if len(arr)>1 else 0:.3f}, n={len(arr)})")


if __name__ == "__main__":
    main()