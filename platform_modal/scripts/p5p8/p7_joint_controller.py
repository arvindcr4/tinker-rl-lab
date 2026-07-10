#!/usr/bin/env python3
"""
Pillar-7 (P7) JOINT controller: Dualformer-Auto on contrast prompts AND
ddiv_triage on boundary prompts (iter-72 JOB B / SYNTH top item — drives
the iter-67 row-78 + iter-71 row-83 mint chain).

Iter-71 row 83 left open: "a joint controller applying Dualformer on
contrast prompts AND ddiv_triage on boundary prompts would combine
both savings". This script implements and measures that joint controller
on the same N2 same-stack corpus (40 steps × 16 prompts × 4 methods =
2560 prompt-step observations per method).

The rule (per prompt-step record r in step s with delta_div(s) at the
step level):
  If r is a contrast_prompt (zvf_actual == 0, K∈{1..G-1}):
      apply Dualformer -> G' = 2  (saves 6 rollouts vs G=8)
  Elif r is a boundary_prompt (zvf_actual == 1) AND delta_div(s) >= tau:
      apply ddiv_triage -> G_esc = 16  (recovers ZVF<0.10)
  Else:
      keep G_base = 8

Two distinct save types:
  - "rollout saves" from Dualformer on contrast: 6 rollouts saved per
    contrast prompt, total = 6 × n_contrast.
  - "zvf saves" from ddiv_triage on boundary: 1 ZVF saved per
    boundary prompt in a step where delta_div(s) >= tau.

Headline metric: **net_saves = rollout_saves + zvf_saves**.

Compared against:
  - ddiv_only (iter-67 row 78): per-step escalation, contrast→G=8.
    Saves only zvf_saves.
  - dualformer_only (Berkeley row 01 / iter-71): per-prompt G'=2 on
    contrast, G=8 on boundary. Saves only rollout_saves.

Outputs:
  experiments/results/p5p8/p7_joint_controller.tsv
  experiments/results/p5p8/p7_joint_controller_boot.tsv
  experiments/results/p5p8/p7_joint_controller_summary.json

Stdlib only. <=290 lines.
"""
from __future__ import annotations

import csv
import json
import pathlib
import random
from collections import defaultdict

WORK = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "experiments/results/n2_reward_tensor_resume"
OUT_DIR = WORK / "experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

G_BASE = 8
G_ESC = 16
G_DUALFORMER = 2
METHODS = ("grpo", "aero", "areal", "gift")
DDIV_TAU_GRID = (0.03, 0.04, 0.05, 0.06, 0.07)
N_BOOT = 2000
SEED = 20260705


def iid_zvf(p_hat: float, G: int) -> float:
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1.0
    return p_hat ** G + (1.0 - p_hat) ** G


def load_n2_with_step_ddiv() -> dict[str, list[dict]]:
    """Load N2 tensors and pre-compute step-level delta_div (iid - obs).

    Each prompt-step record carries:
      method, step, prompt_idx, K, p_hat, zvf_actual (boundary==1),
      contrast_prompt, boundary_prompt, zvf_iid_g8, zvf_iid_g16,
      step_zvf_iid (mean iid zvf_g8 over step prompts),
      step_zvf_obs (mean obs zvf over step prompts),
      step_delta_div = step_zvf_iid - step_zvf_obs.
    """
    out: dict[str, list[dict]] = {}
    for m in METHODS:
        f = N2_DIR / f"{m}_s0_tensors.jsonl"
        rows: list[dict] = []
        # First pass: collect per-step aggregates
        step_data: dict[int, dict] = defaultdict(
            lambda: {"prompts": [], "zvf_obs_total": 0}
        )
        all_recs: list[dict] = []
        with f.open() as fh:
            for line in fh:
                rec = json.loads(line)
                rewards = rec["rewards"]
                zvf_obs_total = sum(1.0 for g in rewards
                                    if (round(sum(g)) == 0 or round(sum(g)) == G_BASE))
                step_data[rec["step"]]["zvf_obs_total"] = zvf_obs_total
                step_data[rec["step"]]["prompts"] = rewards

        for step, sd in step_data.items():
            n_prompts = len(sd["prompts"])
            step_zvf_obs = sd["zvf_obs_total"] / n_prompts
            # iid ZVF over the empirical prompt p_hats (use the same p_hats)
            step_zvf_iid = 0.0
            for g in sd["prompts"]:
                K = round(sum(g))
                p_hat = K / G_BASE
                step_zvf_iid += iid_zvf(p_hat, G_BASE)
            step_zvf_iid /= n_prompts
            step_delta_div = step_zvf_iid - step_zvf_obs
            for p_idx, group in enumerate(sd["prompts"]):
                K = int(round(sum(group)))
                p_hat = K / G_BASE
                zvf_actual = 1.0 if (K == 0 or K == G_BASE) else 0.0
                zvf_iid_g8 = iid_zvf(p_hat, G_BASE)
                zvf_iid_g16 = iid_zvf(p_hat, G_ESC)
                rows.append({
                    "method": m,
                    "step": step,
                    "prompt_idx": p_idx,
                    "K": K,
                    "p_hat": p_hat,
                    "zvf_actual": zvf_actual,
                    "zvf_iid_g8": zvf_iid_g8,
                    "zvf_iid_g16": zvf_iid_g16,
                    "step_zvf_iid": step_zvf_iid,
                    "step_zvf_obs": step_zvf_obs,
                    "step_delta_div": step_delta_div,
                    "boundary_prompt": zvf_actual == 1.0,
                    "contrast_prompt": zvf_actual == 0.0,
                    "rollout_save_if_dualformer": 6.0 if zvf_actual == 0.0 else 0.0,
                })
        out[m] = rows
    return out


def joint_controller(records: list[dict], tau: float) -> dict:
    """Apply joint rule: Dualformer on contrast (non-fired), ddiv_triage on
    fired-step prompts (boundary OR contrast).

    Rule:
      if r["step_delta_div"] >= tau:
          # fired step: escalate ALL prompts in the step to G=16
          g_total += G_ESC
          if r["zvf_iid_g16"] < 0.10:
              n_zvf_saved += 1
      elif r["contrast_prompt"]:
          # Dualformer on contrast in non-fired step: G=2 saves 6 rollouts
          n_contrast += 1
          g_total += G_DUALFORMER
          rollout_saves += 6
      else:
          g_total += G_BASE

    Net saves = rollout_saves (from Dualformer) + zvf_saves (from ddiv_triage).
    """
    n = len(records)
    n_contrast = 0
    n_fired_steps = 0
    n_zvf_saved = 0
    g_total = 0
    rollout_saves = 0
    for r in records:
        if r["step_delta_div"] >= tau:
            n_fired_steps += 1
            g_total += G_ESC
            if r["zvf_iid_g16"] < 0.10:
                n_zvf_saved += 1
        elif r["contrast_prompt"]:
            n_contrast += 1
            g_total += G_DUALFORMER
            rollout_saves += int(r["rollout_save_if_dualformer"])
        else:
            # boundary prompt in non-fired step
            g_total += G_BASE
    cost_ratio = g_total / (n * G_BASE)
    return {
        "tau": tau,
        "n_contrast": n_contrast,
        "n_fired_steps": n_fired_steps,
        "n_zvf_saved": n_zvf_saved,
        "rollout_saves": rollout_saves,
        "cost_ratio": cost_ratio,
        "g_total": g_total,
    }


def dualformer_only(records: list[dict]) -> dict:
    n = len(records)
    n_contrast = sum(1 for r in records if r["contrast_prompt"])
    g_total = n_contrast * G_DUALFORMER + (n - n_contrast) * G_BASE
    rollout_saves = n_contrast * (G_BASE - G_DUALFORMER)
    return {
        "g_total": g_total,
        "cost_ratio": g_total / (n * G_BASE),
        "n_contrast": n_contrast,
        "rollout_saves": rollout_saves,
    }


def ddiv_only(records: list[dict], tau: float) -> dict:
    """Per-step escalation (no Dualformer). Saves count contrast prompts in
    fired steps where zvf_iid_g16 < 0.10 (the iter-67 row 78 save definition)."""
    n = len(records)
    n_fires = 0
    n_saves = 0
    g_total = 0
    for r in records:
        if r["step_delta_div"] >= tau:
            g_total += G_ESC
            n_fires += 1
            if r["zvf_iid_g16"] < 0.10:
                n_saves += 1
        else:
            g_total += G_BASE
    return {
        "tau": tau,
        "n_fires": n_fires,
        "n_saves": n_saves,
        "g_total": g_total,
        "cost_ratio": g_total / (n * G_BASE),
        "rollout_saves": 0,
    }


def bootstrap_ci_per_method_metric(records: list[dict], tau: float,
                                    B: int, seed: int) -> dict:
    """Paired bootstrap on the joint-controller's `net saves` per prompt-step."""
    rng = random.Random(seed)
    n = len(records)
    if n == 0:
        return {"pt": 0.0, "lo": 0.0, "hi": 0.0}
    net_save_indicators = []
    for r in records:
        if r["step_delta_div"] >= tau:
            net_save_indicators.append(
                1 if r["zvf_iid_g16"] < 0.10 else 0
            )
        elif r["contrast_prompt"]:
            net_save_indicators.append(6)
        else:
            net_save_indicators.append(0)
    sums = []
    for _ in range(B):
        s = 0
        for _ in range(n):
            s += net_save_indicators[rng.randint(0, n - 1)]
        sums.append(s)
    sums.sort()
    pt = sum(net_save_indicators)
    return {
        "pt": pt,
        "lo": sums[int(B * 0.025)],
        "hi": sums[int(B * 0.975)],
    }


def main():
    print("[p7_joint_controller] loading N2 tensors...")
    n2 = load_n2_with_step_ddiv()
    print(f"[p7_joint_controller] loaded {sum(len(v) for v in n2.values())} prompt-step obs")

    # 1) Joint controller at the 5-tau grid
    print("[p7_joint_controller] computing joint controller at tau grid...")
    joint_rows = []
    for m in METHODS:
        records = n2[m]
        for tau in DDIV_TAU_GRID:
            jc = joint_controller(records, tau)
            ci = bootstrap_ci_per_method_metric(records, tau, N_BOOT,
                                                  SEED + hash((m, tau)) % 10000)
            net_saves = jc["rollout_saves"] + jc["n_zvf_saved"]
            joint_rows.append({
                "method": m,
                "tau": tau,
                "n_contrast": jc["n_contrast"],
                "n_fired_steps": jc["n_fired_steps"],
                "n_zvf_saved": jc["n_zvf_saved"],
                "rollout_saves": jc["rollout_saves"],
                "net_saves": net_saves,
                "cost_ratio": jc["cost_ratio"],
                "g_total": jc["g_total"],
                "net_saves_lo": ci["lo"],
                "net_saves_hi": ci["hi"],
            })

    with (OUT_DIR / "p7_joint_controller.tsv").open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(joint_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for row in joint_rows:
            w.writerow(row)
    print(f"[p7_joint_controller] joint_controller.tsv: {len(joint_rows)} rows")

    # 2) Headline comparison at canonical tau=0.05
    print("[p7_joint_controller] computing comparison at canonical tau=0.05...")
    comparison_rows = []
    TAU = 0.05
    for m in METHODS:
        records = n2[m]
        jc = joint_controller(records, TAU)
        df = dualformer_only(records)
        dv = ddiv_only(records, TAU)
        net_joint = jc["rollout_saves"] + jc["n_zvf_saved"]
        comparison_rows.append({
            "method": m,
            "joint_rollout_saves": jc["rollout_saves"],
            "joint_zvf_saves": jc["n_zvf_saved"],
            "joint_net_saves": net_joint,
            "joint_cost_ratio": jc["cost_ratio"],
            "df_rollout_saves": df["rollout_saves"],
            "df_cost_ratio": df["cost_ratio"],
            "ddiv_zvf_saves": dv["n_saves"],
            "ddiv_rollout_saves": dv["rollout_saves"],
            "ddiv_cost_ratio": dv["cost_ratio"],
            "net_joint_minus_ddiv": net_joint - dv["n_saves"],
            "net_joint_minus_df": net_joint - df["rollout_saves"],
        })

    with (OUT_DIR / "p7_joint_controller_boot.tsv").open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comparison_rows[0].keys()),
                            delimiter="\t")
        w.writeheader()
        for row in comparison_rows:
            w.writerow(row)
    print(f"[p7_joint_controller] joint_controller_boot.tsv: {len(comparison_rows)} rows")

    # 3) Summary
    summary = {
        "ts": "2026-07-05",
        "iteration": 72,
        "pillar": "P7",
        "vein": "joint controller (Dualformer on contrast + ddiv_triage on boundary)",
        "G_base": G_BASE,
        "G_esc": G_ESC,
        "G_dualformer": G_DUALFORMER,
        "tau_grid": list(DDIV_TAU_GRID),
        "n_boot": N_BOOT,
        "n_obs_per_method": len(n2[METHODS[0]]),
        "n_methods": len(METHODS),
        "headlines": {},
    }
    for row in comparison_rows:
        m = row["method"]
        summary["headlines"][m] = {
            "joint_net_saves": row["joint_net_saves"],
            "joint_cost_ratio": row["joint_cost_ratio"],
            "joint_rollout_saves": row["joint_rollout_saves"],
            "joint_zvf_saves": row["joint_zvf_saves"],
            "df_only_rollout_saves": row["df_rollout_saves"],
            "df_only_cost_ratio": row["df_cost_ratio"],
            "ddiv_only_zvf_saves": row["ddiv_zvf_saves"],
            "ddiv_only_cost_ratio": row["ddiv_cost_ratio"],
            "net_joint_minus_ddiv": row["net_joint_minus_ddiv"],
            "net_joint_minus_df": row["net_joint_minus_df"],
        }
        summary["headlines"][m]["joint_cost_minus_df_cost"] = (
            row["joint_cost_ratio"] - row["df_cost_ratio"]
        )
        summary["headlines"][m]["joint_cost_minus_ddiv_cost"] = (
            row["joint_cost_ratio"] - row["ddiv_cost_ratio"]
        )
    total_joint_saves = sum(r["joint_net_saves"] for r in comparison_rows)
    total_ddiv_saves = sum(r["ddiv_zvf_saves"] for r in comparison_rows)
    total_df_saves = sum(r["df_rollout_saves"] for r in comparison_rows)
    summary["aggregate"] = {
        "total_joint_net_saves_all_methods": total_joint_saves,
        "total_ddiv_only_zvf_saves_all_methods": total_ddiv_saves,
        "total_df_only_rollout_saves_all_methods": total_df_saves,
        "joint_minus_ddiv_total": total_joint_saves - total_ddiv_saves,
        "joint_minus_df_total": total_joint_saves - total_df_saves,
        "joint_vs_best_single_total": total_joint_saves - max(total_ddiv_saves, total_df_saves),
        "per_method_best_tau": {},
    }
    for m in METHODS:
        sub = [r for r in joint_rows if r["method"] == m]
        if sub:
            best = max(sub, key=lambda r: r["net_saves"])
            summary["aggregate"]["per_method_best_tau"][m] = {
                "tau": best["tau"],
                "net_saves": best["net_saves"],
                "rollout_saves": best["rollout_saves"],
                "zvf_saves": best["n_zvf_saved"],
                "cost_ratio": best["cost_ratio"],
                "net_saves_lo": best["net_saves_lo"],
                "net_saves_hi": best["net_saves_hi"],
            }

    with (OUT_DIR / "p7_joint_controller_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("[p7_joint_controller] summary written")

    print("\n[p7_joint_controller] HEADLINE at tau=0.05:")
    for row in comparison_rows:
        print(f"  {row['method']:6s}: joint_net={row['joint_net_saves']:4d} "
              f"(rollout={row['joint_rollout_saves']:3d}, zvf={row['joint_zvf_saves']:3d}) "
              f"@ cost={row['joint_cost_ratio']:.3f} "
              f"| ddiv_zvf={row['ddiv_zvf_saves']:3d} @ cost={row['ddiv_cost_ratio']:.3f} "
              f"| df_rollout={row['df_rollout_saves']:3d} @ cost={row['df_cost_ratio']:.3f} "
              f"| joint-ddiv={row['net_joint_minus_ddiv']:+d} joint-df={row['net_joint_minus_df']:+d}")


if __name__ == "__main__":
    main()