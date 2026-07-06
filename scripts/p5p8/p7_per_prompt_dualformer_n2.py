#!/usr/bin/env python3
"""
Pillar-7 (P7) Per-Prompt Dualformer-Auto Reproduce + bootstrap CIs on
the iter-67 ddiv_triage headline.

Iter-71 vein (fresh, NOT in 80-row prior ledger): the iter-67 row-78
ddiv_triage counterfactual used per-STEP granularity (16 prompts
collapsed into 1 ZVF). This iter lifts to per-PROMPT granularity
(16 prompts x 40 steps x 4 methods = 2560 prompt-step decisions) and
pairs it with:

  (A) Berkeley row-01 Dualformer-Auto rule at per-prompt granularity
      (G'=2 for contrast prompts, G=8 for boundary prompts). Reproduces
      the 56.2% saving claim on the same N2 corpus and adds bootstrap CIs.
  (B) Per-method bootstrap CI on the iter-67 row-78 ddiv_triage@τ=0.05
      saved/fire headline at per-prompt granularity (n=16*40=640 prompt-
      step cells per method; B=2000).
  (C) Joint comparison: dualformer-Auto vs ddiv_triage@τ on
      cost_ratio and saved_per_fire across all 4 N2 methods.

Outputs:
  experiments/results/p5p8/p7_per_prompt_dualformer_summary.tsv
  experiments/results/p5p8/p7_per_prompt_ddiv_boot.tsv
  experiments/results/p5p8/p7_per_prompt_joint_comparison.tsv
  experiments/results/p5p8/p7_per_prompt_dualformer_summary.json

Stdlib only.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import statistics

WORK = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "experiments/results/n2_reward_tensor_resume"
OUT_DIR = WORK / "experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

G_BASE = 8
G_ESC = 16
G_DUALFORMER = 2
METHODS = ("grpo", "aero", "areal", "gift")
DDIV_TAU_GRID = (0.03, 0.05, 0.08, 0.10, 0.12)
N_BOOT = 2000
SEED = 20260705

# --- iid ZVF under extrapolation (binomial p^G + (1-p)^G) ---


def iid_zvf(p_hat: float, G: int) -> float:
    """Binomial probability of all-zero or all-one under iid assumption.

    ZVF = p^G + (1-p)^G, monotonically decreasing in G for p in (0,1).
    """
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1.0
    return p_hat ** G + (1.0 - p_hat) ** G


def load_n2() -> dict[str, list[dict]]:
    """Load N2 tensors; emit per-prompt records keyed by method.

    Each prompt-step record has:
      step, prompt_idx, K (successes in G_BASE), p_hat = K/G_BASE,
      zvf_actual (1 iff K=0 or K=G else 0), zvf_iid = p^G + (1-p)^G.
    """
    out: dict[str, list[dict]] = {}
    for m in METHODS:
        f = N2_DIR / f"{m}_s0_tensors.jsonl"
        rows: list[dict] = []
        with f.open() as fh:
            for line in fh:
                rec = json.loads(line)
                rewards = rec["rewards"]  # 16 x G=8
                for p_idx, group in enumerate(rewards):
                    K = int(round(sum(group)))
                    p_hat = K / G_BASE
                    zvf_actual = 1.0 if (K == 0 or K == G_BASE) else 0.0
                    zvf_iid_g8 = iid_zvf(p_hat, G_BASE)
                    zvf_iid_g16 = iid_zvf(p_hat, G_ESC)
                    delta = zvf_iid_g8 - zvf_iid_g16  # headroom for G=8 -> G=16
                    rows.append(
                        {
                            "method": m,
                            "step": rec["step"],
                            "prompt_idx": p_idx,
                            "K": K,
                            "p_hat": p_hat,
                            "zvf_actual_g8": zvf_actual,
                            "zvf_iid_g8": zvf_iid_g8,
                            "zvf_iid_g16": zvf_iid_g16,
                            "headroom_g8_to_g16": delta,
                            "boundary_prompt": zvf_actual == 1.0,
                            "contrast_prompt": zvf_actual == 0.0,
                        }
                    )
        out[m] = rows
    return out


# --- Berkeley row 01 Dualformer-Auto at per-prompt granularity ---


def dualformer_per_prompt(records: list[dict]) -> dict:
    """Apply Dualformer-Auto rule (G'=2 for contrast, G=8 for boundary).

    The rule: per-prompt, if K=0 or K=G (boundary), keep G=8 to preserve
    coverage of rare class; if K in (0, G), the contrast is already
    observable, so G=2 suffices (Berkeley row 01 56.2% saving claim).

    Saving is computed against the actual baseline of always G=8.
    """
    n_contrast = sum(1 for r in records if r["contrast_prompt"])
    n_boundary = sum(1 for r in records if r["boundary_prompt"])
    g_total_actual = len(records) * G_BASE
    g_total_dualformer = n_contrast * G_DUALFORMER + n_boundary * G_BASE
    saving = 1.0 - g_total_dualformer / g_total_actual
    return {
        "n_prompts": len(records),
        "n_contrast": n_contrast,
        "n_boundary": n_boundary,
        "g_total_actual": g_total_actual,
        "g_total_dualformer": g_total_dualformer,
        "saving": saving,
    }


# --- ddiv-triage trigger at per-prompt granularity (extension of iter-67) ---


def ddiv_per_prompt(records: list[dict], tau: float) -> dict:
    """Apply ddiv_triage@τ at per-prompt granularity.

    Fires when headroom_g8_to_g16 >= tau, i.e. when escalating from
    G=8 to G=16 would meaningfully reduce expected iid ZVF.

    This is the per-prompt refinement of the iter-67 row-78 step-level
    ddiv_triage. The per-prompt trigger is finer-grained (n=640 obs per
    method) and recovers a different mixture of prompts.
    """
    n_fires = 0
    n_saves = 0
    n_at_risk_g8 = 0  # zvf_iid_g8 in [0.10, 0.99] — i.e. degenerate enough to help
    n_recovered_g16 = 0  # zvf_iid_g16 in [0, 0.10) — saved by escalation
    n_wasted_g16 = 0  # fired but still saturated at G=16 (zvf_iid_g16 > 0.10)
    for r in records:
        if r["headroom_g8_to_g16"] >= tau:
            n_fires += 1
            if 0.10 <= r["zvf_iid_g8"] <= 0.99:
                n_at_risk_g8 += 1
            if r["zvf_iid_g16"] < 0.10:
                n_saves += 1
                n_recovered_g16 += 1
            else:
                n_wasted_g16 += 1
    cost_ratio = (n_fires * G_ESC + (len(records) - n_fires) * G_BASE) / (
        len(records) * G_BASE
    )
    saved_per_fire = n_saves / n_fires if n_fires > 0 else 0.0
    return {
        "tau": tau,
        "n_fires": n_fires,
        "n_saves": n_saves,
        "n_at_risk_g8": n_at_risk_g8,
        "n_recovered_g16": n_recovered_g16,
        "n_wasted_g16": n_wasted_g16,
        "cost_ratio": cost_ratio,
        "saved_per_fire": saved_per_fire,
    }


# --- Bootstrap CIs ---


def bootstrap_ci(values: list[float], B: int, seed: int,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of values."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(B):
        sample = [values[rng.randint(0, n - 1)]] * n
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    pt = sum(values) / n
    hi = means[int(B * (1 - alpha / 2))]
    return pt, lo, hi


def paired_bootstrap_diff_ci(a: list[float], b: list[float], B: int,
                             seed: int, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Paired bootstrap CI on (mean(a) - mean(b)).

    Resamples indices jointly so the diff distribution reflects the
    within-pair correlation structure.
    """
    rng = random.Random(seed)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    diffs = []
    for _ in range(B):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        ma = sum(a[i] for i in idx) / n
        mb = sum(b[i] for i in idx) / n
        diffs.append(ma - mb)
    diffs.sort()
    pt = sum(a) / n - sum(b) / n
    lo = diffs[int(B * alpha / 2)]
    hi = diffs[int(B * (1 - alpha / 2))]
    p_two = sum(1 for d in diffs if d > 0) / B
    p_two = 2 * min(p_two, 1 - p_two)
    return pt, lo, hi, p_two


def main() -> None:
    rng = random.Random(SEED)
    print("Loading N2 tensors...")
    n2 = load_n2()

    summary_rows: list[dict] = []
    ddiv_rows: list[dict] = []
    joint_rows: list[dict] = []

    # --- (A) Dualformer-Auto per-prompt by method ---
    print("Computing Dualformer-Auto per-prompt by method...")
    dualformer_by_method: dict[str, dict] = {}
    for m in METHODS:
        df = dualformer_per_prompt(n2[m])
        dualformer_by_method[m] = df
        summary_rows.append(
            {
                "section": "A_dualformer_per_prompt",
                "method": m,
                "tau": "",
                "n_prompts": df["n_prompts"],
                "n_contrast": df["n_contrast"],
                "n_boundary": df["n_boundary"],
                "n_fires": df["n_contrast"],  # G'=2 fires on contrast
                "n_saves": "",
                "cost_ratio": df["g_total_dualformer"] / df["g_total_actual"],
                "saved_per_fire": "",
                "metric": "saving",
                "value": df["saving"],
            }
        )

    # Aggregate Dualformer saving across methods (B=2000 paired-by-step)
    print("Bootstrapping Dualformer saving CI across methods (B=2000)...")
    per_step_savings: dict[str, list[float]] = {m: [] for m in METHODS}
    for m in METHODS:
        # Per-step saving (treat each step as one observation)
        step_dict: dict[int, list[dict]] = {}
        for r in n2[m]:
            step_dict.setdefault(r["step"], []).append(r)
        for step in sorted(step_dict.keys()):
            df_step = dualformer_per_prompt(step_dict[step])
            per_step_savings[m].append(df_step["saving"])

    # Paired-by-step cross-method CI on Dualformer saving
    print("Paired-bootstrap cross-method Dualformer saving CIs...")
    for m in METHODS:
        pt, lo, hi = bootstrap_ci(per_step_savings[m], N_BOOT, SEED)
        summary_rows.append(
            {
                "section": "A_dualformer_boot",
                "method": m,
                "tau": "",
                "n_prompts": "",
                "n_contrast": "",
                "n_boundary": "",
                "n_fires": "",
                "n_saves": "",
                "cost_ratio": "",
                "saved_per_fire": "",
                "metric": f"saving_ci_pt={pt:.4f}",
                "value": f"lo={lo:.4f} hi={hi:.4f}",
            }
        )

    # --- (B) ddiv-triage@τ per-prompt by method + bootstrap CIs ---
    print("Computing ddiv_triage per-prompt by method x tau...")
    ddiv_by_method_tau: dict[tuple[str, float], dict] = {}
    for m in METHODS:
        for tau in DDIV_TAU_GRID:
            d = ddiv_per_prompt(n2[m], tau)
            ddiv_by_method_tau[(m, tau)] = d
            ddiv_rows.append(
                {
                    "method": m,
                    "tau": tau,
                    "n_fires": d["n_fires"],
                    "n_saves": d["n_saves"],
                    "n_at_risk_g8": d["n_at_risk_g8"],
                    "n_recovered_g16": d["n_recovered_g16"],
                    "n_wasted_g16": d["n_wasted_g16"],
                    "cost_ratio": d["cost_ratio"],
                    "saved_per_fire": d["saved_per_fire"],
                }
            )

    # Bootstrap CI on saved/fire per (method, tau) at the per-prompt level
    # (resample prompts with replacement; reflect per-prompt noise)
    print("Bootstrapping saved/fire CI per (method, tau) at per-prompt level...")
    for m in METHODS:
        records = n2[m]
        n = len(records)
        for tau in DDIV_TAU_GRID:
            headroom_ok = [r["headroom_g8_to_g16"] >= tau for r in records]
            zvf_g16 = [r["zvf_iid_g16"] for r in records]
            zvf_g8 = [r["zvf_iid_g8"] for r in records]
            # Per-prompt "saved" outcome: fire AND recovered (zvf_iid_g16 < 0.10)
            fires_arr = [1 if h else 0 for h in headroom_ok]
            saves_arr = [
                1 if (h and z < 0.10) else 0 for h, z in zip(headroom_ok, zvf_g16)
            ]
            spf_samples = []
            for _ in range(N_BOOT):
                idx = [rng.randint(0, n - 1) for _ in range(n)]
                f_b = sum(fires_arr[i] for i in idx)
                s_b = sum(saves_arr[i] for i in idx)
                spf_samples.append(s_b / f_b if f_b > 0 else 0.0)
            spf_samples.sort()
            pt = sum(saves_arr) / sum(fires_arr) if sum(fires_arr) > 0 else 0.0
            lo = spf_samples[int(N_BOOT * 0.025)]
            hi = spf_samples[int(N_BOOT * 0.975)]
            summary_rows.append(
                {
                    "section": "B_ddiv_per_prompt_boot",
                    "method": m,
                    "tau": tau,
                    "n_prompts": n,
                    "n_contrast": "",
                    "n_boundary": "",
                    "n_fires": sum(fires_arr),
                    "n_saves": sum(saves_arr),
                    "cost_ratio": "",
                    "saved_per_fire": f"pt={pt:.4f}",
                    "metric": f"spf_ci_lo={lo:.4f}",
                    "value": f"hi={hi:.4f}",
                }
            )

    # --- (C) Joint comparison: Dualformer-Auto vs ddiv_triage@τ at matched cost ---
    print("Joint comparison: Dualformer-Auto vs ddiv_triage@τ at matched cost...")
    for m in METHODS:
        df = dualformer_by_method[m]
        df_cost = df["g_total_dualformer"] / df["g_total_actual"]
        for tau in DDIV_TAU_GRID:
            d = ddiv_by_method_tau[(m, tau)]
            # Matched cost: compare saved/fire at the cost_ratio closest to df_cost
            ddiv_cost = d["cost_ratio"]
            joint_rows.append(
                {
                    "method": m,
                    "tau": tau,
                    "df_cost": df_cost,
                    "ddiv_cost": ddiv_cost,
                    "cost_gap": ddiv_cost - df_cost,
                    "df_saving": df["saving"],
                    "ddiv_spf": d["saved_per_fire"],
                    "spf_gap": d["saved_per_fire"] - df["saving"],
                    "ddiv_n_fires": d["n_fires"],
                    "ddiv_n_saves": d["n_saves"],
                }
            )

    # Paired-bootstrap on saved/fire between Dualformer saving and ddiv_spf
    print("Paired-bootstrap diff: ddiv_spf - df_saving by method at tau=0.05...")
    for m in METHODS:
        step_records: dict[int, list[dict]] = {}
        for r in n2[m]:
            step_records.setdefault(r["step"], []).append(r)
        steps_sorted = sorted(step_records.keys())
        df_per_step: list[float] = []
        ddiv_per_step: list[float] = []
        for step in steps_sorted:
            d = dualformer_per_prompt(step_records[step])
            df_per_step.append(d["saving"])
            d = ddiv_per_prompt(step_records[step], 0.05)
            ddiv_per_step.append(d["saved_per_fire"])
        pt, lo, hi, p = paired_bootstrap_diff_ci(
            ddiv_per_step, df_per_step, N_BOOT, SEED + hash(m) % 10000
        )
        summary_rows.append(
            {
                "section": "C_ddiv_vs_dualformer_paired",
                "method": m,
                "tau": 0.05,
                "n_prompts": "",
                "n_contrast": "",
                "n_boundary": "",
                "n_fires": "",
                "n_saves": "",
                "cost_ratio": "",
                "saved_per_fire": f"pt={pt:.4f}",
                "metric": f"diff_lo={lo:.4f}",
                "value": f"hi={hi:.4f} p={p:.4f}",
            }
        )

    # --- Bootstrap CI on saves per 100 prompts (stable rate metric) ---
    print("Bootstrapping saves/100-prompts CI by method x tau...")
    for m in METHODS:
        records = n2[m]
        n = len(records)
        for tau in DDIV_TAU_GRID:
            headroom_ok = [r["headroom_g8_to_g16"] >= tau for r in records]
            zvf_g16 = [r["zvf_iid_g16"] for r in records]
            saves_arr = [
                1 if (h and z < 0.10) else 0 for h, z in zip(headroom_ok, zvf_g16)
            ]
            rates = []
            for _ in range(N_BOOT):
                idx = [rng.randint(0, n - 1) for _ in range(n)]
                rates.append(sum(saves_arr[i] for i in idx) / n * 100)
            rates.sort()
            pt = sum(saves_arr) / n * 100
            lo = rates[int(N_BOOT * 0.025)]
            hi = rates[int(N_BOOT * 0.975)]
            summary_rows.append(
                {
                    "section": "B_ddiv_save_rate_boot",
                    "method": m,
                    "tau": tau,
                    "n_prompts": n,
                    "n_contrast": "",
                    "n_boundary": "",
                    "n_fires": sum(headroom_ok),
                    "n_saves": sum(saves_arr),
                    "cost_ratio": "",
                    "saved_per_fire": f"rate_pt={pt:.4f}",
                    "metric": f"rate_lo={lo:.4f}",
                    "value": f"hi={hi:.4f}",
                }
            )

    # --- Write outputs ---
    print("Writing outputs...")
    with (OUT_DIR / "p7_per_prompt_dualformer_summary.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    with (OUT_DIR / "p7_per_prompt_ddiv_boot.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(ddiv_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for row in ddiv_rows:
            w.writerow(row)
    with (OUT_DIR / "p7_per_prompt_joint_comparison.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(joint_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for row in joint_rows:
            w.writerow(row)

    # JSON summary
    out_json = {
        "ts": "2026-07-05",
        "iteration": 71,
        "pillar": "P7",
        "vein": "per-prompt Dualformer-Auto reproduce + bootstrap CIs on iter-67 ddiv_triage headline",
        "data": "n2_reward_tensor_resume/{grpo,aero,areal,gift}_s0_tensors.jsonl",
        "G_base": G_BASE,
        "G_esc": G_ESC,
        "G_dualformer": G_DUALFORMER,
        "n_prompts_per_step": 16,
        "n_steps": 40,
        "n_methods": 4,
        "n_obs_per_method": 16 * 40,
        "n_boot": N_BOOT,
        "dualformer_per_method_saving_pt": {
            m: dualformer_by_method[m]["saving"] for m in METHODS
        },
        "dualformer_per_method_saving_ci": {
            m: bootstrap_ci(per_step_savings[m], N_BOOT, SEED) for m in METHODS
        },
        "berkeley_row01_claim": 0.562,
        "ddiv_tau_grid": list(DDIV_TAU_GRID),
        "ddiv_at_tau_0.05_by_method": {
            m: ddiv_by_method_tau[(m, 0.05)] for m in METHODS
        },
    }
    with (OUT_DIR / "p7_per_prompt_dualformer_summary.json").open("w") as f:
        json.dump(out_json, f, indent=2, default=str)

    # --- Print headline ---
    print("\n=== HEADLINE ===")
    print("Berkeley row 01 Dualformer-Auto per-prompt saving on N2 corpus:")
    for m in METHODS:
        df = dualformer_by_method[m]
        pt, lo, hi = bootstrap_ci(per_step_savings[m], N_BOOT, SEED)
        print(
            f"  {m:6s}: saving={df['saving']:.4f} CI=[{lo:.4f}, {hi:.4f}] "
            f"(n_contrast={df['n_contrast']}, n_boundary={df['n_boundary']})"
        )
    print("\nBerkeley row 01 claim: 56.2% saving")
    print()
    print("ddiv_triage@τ=0.05 per-prompt saved/fire by method (with bootstrap CI):")
    for m in METHODS:
        d = ddiv_by_method_tau[(m, 0.05)]
        step_records: dict[int, list[dict]] = {}
        for r in n2[m]:
            step_records.setdefault(r["step"], []).append(r)
        per_step_spf = [
            ddiv_per_prompt(step_records[s], 0.05)["saved_per_fire"]
            for s in sorted(step_records.keys())
        ]
        pt, lo, hi = bootstrap_ci(per_step_spf, N_BOOT, SEED)
        print(
            f"  {m:6s}: spf={d['saved_per_fire']:.4f} CI=[{lo:.4f}, {hi:.4f}] "
            f"cost_ratio={d['cost_ratio']:.4f} fires={d['n_fires']}"
        )


if __name__ == "__main__":
    main()