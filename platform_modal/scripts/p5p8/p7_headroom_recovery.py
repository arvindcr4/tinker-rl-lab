#!/usr/bin/env python3
"""P7 Per-Prompt Headroom Ceiling + Controller Recovery Ratio on N2 + N10.

Vein (not in prior ledger): every P7 controller has been evaluated against
the fixed-G=8 baseline (cost ratio, headroom-bad rate, total rollouts) but
NOT against the THEORETICAL CEILING of per-prompt CONTRAST RECOVERY.

This script computes:
  (1) per-prompt headroom ceiling = ZVF(G_base=8) - ZVF(G_esc=16),
      the maximum iid-ZVF reduction the controller could achieve by
      escalating (since iid-ZVF = p^G + (1-p)^G is monotonically DECREASING
      in G for p in (0, 1), any G' > G_base improves contrast on mixed
      prompts and never helps on saturated ones).
  (2) per-controller actual ZVF change = ZVF(G_base) - ZVF(G_ctrl) where
      G_ctrl is the controller's chosen G_t for this (step, prompt) cell.
  (3) recovery_ratio = actual / ceiling, bounded in (-inf, 1]:
        = 0       (controller behaves like baseline, no change)
        = 1       (controller recovers ALL achievable headroom)
        < 0       (controller makes things WORSE than baseline -- over-de-escalation)
        = NaN/0   (saturated prompt: ceiling = 0, no recovery possible)

Two evidence bases:
  - N2 four-method tensors (40 steps x 16 prompts x 4 methods = 2,560 obs)
  - N10 5-seed GRPO panel (15 steps; per-step ZVF only -- ceiling UNKNOWN)

Headline questions (falsifiable):
  Q1. On N2, what is the per-prompt CONTRAST-RECOVERY headroom ceiling, and
      what fraction does each controller recover at step-level dispatch?
  Q2. Is the Hybrid's recovery strictly greater than Dualformer-Auto's on
      mixed prompts (since Hybrid may ESCALATE on boundary band)?
  Q3. Is step-level ZVF an aliased predictor of per-prompt headroom?
      (correlation between step_zvf and mean per-prompt headroom at step)

Outputs (under experiments/results/p5p8/):
  p7_headroom_recovery_n2_summary.tsv   -- one row per (method, controller)
  p7_headroom_recovery_n2_per_step.tsv  -- one row per (method, step)
  p7_headroom_recovery_n2_per_prompt.tsv -- one row per (method, step, prompt_index)
  p7_headroom_recovery_n10_summary.tsv  -- one row per (seed, controller)
  p7_headroom_recovery_n10_per_step.tsv -- one row per (seed, step)
  p7_headroom_recovery_summary.json

Stdlib only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N2_DIR = ROOT / "experiments" / "results" / "n2_reward_tensor_resume"
N10_DIR = ROOT / "experiments" / "results" / "n10_seed_expansion"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"

METHODS = ("grpo", "aero", "gift", "areal")
G_BASE = 8
G_ESC = 16
TARGET_ZVF = 0.99

N10_SEEDS = (42, 179, 316, 453, 590)
N10_STEPS = 15

TAU = 0.7
TAU_DELTA = 0.2
G_DES = 4

BOOT_SEED = 20260704
N_BOOT = 2000


def zvf_iid(p: float, g: int) -> float:
    """I.I.D. binomial ZVF: P(all same) = p^g + (1-p)^g."""
    if p <= 0.0 or p >= 1.0:
        return 1.0
    return p ** g + (1.0 - p) ** g


def is_saturated(k: int, g_base: int = G_BASE) -> bool:
    return k == 0 or k == g_base


def load_n2_rewards():
    out = {}
    for m in METHODS:
        path = N2_DIR / f"{m}_s0_tensors.jsonl"
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                out[(m, r["step"])] = r["rewards"]
    return out


def load_n10_step_log():
    out = {}
    for s in N10_SEEDS:
        path = N10_DIR / f"n10_grpo_s{s}.json"
        with open(path) as f:
            d = json.load(f)
        out[s] = d["step_log"]
    return out


def per_step_zvf(rewards_16: list) -> float:
    z = []
    for p in rewards_16:
        k = sum(1 for r in p if r >= 0.5)
        z.append(zvf_iid(k / G_BASE, G_BASE))
    return sum(z) / len(z) if z else 1.0


def per_step_controllers(z_step: float):
    """Return (C1, C2, C3) G_t for the step given z_step.

    C1 zvf-triage:   escalate to G_ESC if z >= TAU, else G_BASE
    C2 dualformer:   de-escalate to G_DES if z >= TAU, else G_BASE
    C3 hybrid:       escalate on [TAU, TAU+TAU_DELTA), de-escalate on z>=TAU+TAU_DELTA, else G_BASE
    """
    c1 = G_ESC if z_step >= TAU else G_BASE
    c2 = G_DES if z_step >= TAU else G_BASE
    if z_step >= TAU and z_step < TAU + TAU_DELTA:
        c3 = G_ESC
    elif z_step >= TAU + TAU_DELTA:
        c3 = G_DES
    else:
        c3 = G_BASE
    return c1, c2, c3


def analyze_n2(n2_rewards):
    """Per (method, step, prompt_index) row with:
       - k, p_hat, is_saturated
       - z_base, ceiling (z_base - z_esc), z_esc = ZVF(p_hat, G_ESC)
       - z_c1, z_c2, z_c3 (ZVF under each controller's G_t)
       - c1_change, c2_change, c3_change (z_base - z_ctrl)
       - c1_recovery, c2_recovery, c3_recovery (change / ceiling if ceiling>0)
    """
    rows = []
    for m in METHODS:
        for s in range(40):
            r16 = n2_rewards.get((m, s))
            if r16 is None:
                continue
            z_step = per_step_zvf(r16)
            c1_g, c2_g, c3_g = per_step_controllers(z_step)
            for pi, p_rewards in enumerate(r16):
                k = sum(1 for r in p_rewards if r >= 0.5)
                p_hat = k / G_BASE
                sat = is_saturated(k)
                # z_base = ZVF(p_hat, G_BASE)
                z_base = zvf_iid(p_hat, G_BASE)
                # ceiling = z_base - z_esc; for saturated, z_esc = z_base = 1.0 -> ceiling = 0
                z_esc = zvf_iid(p_hat, G_ESC)
                ceiling = z_base - z_esc
                # Controller outcomes
                z_c1 = zvf_iid(p_hat, c1_g)
                z_c2 = zvf_iid(p_hat, c2_g)
                z_c3 = zvf_iid(p_hat, c3_g)
                c1_change = z_base - z_c1
                c2_change = z_base - z_c2
                c3_change = z_base - z_c3
                c1_rec = c1_change / ceiling if ceiling > 1e-9 else None
                c2_rec = c2_change / ceiling if ceiling > 1e-9 else None
                c3_rec = c3_change / ceiling if ceiling > 1e-9 else None
                rows.append({
                    "method": m, "step": s, "prompt_index": pi, "k": k,
                    "p_hat": p_hat, "is_saturated": int(sat),
                    "z_step": round(z_step, 4),
                    "z_base": round(z_base, 4),
                    "ceiling": round(ceiling, 4),
                    "c1_g": c1_g, "c2_g": c2_g, "c3_g": c3_g,
                    "c1_zvf_after": round(z_c1, 4),
                    "c2_zvf_after": round(z_c2, 4),
                    "c3_zvf_after": round(z_c3, 4),
                    "c1_change": round(c1_change, 4),
                    "c2_change": round(c2_change, 4),
                    "c3_change": round(c3_change, 4),
                    "c1_recovery": round(c1_rec, 4) if c1_rec is not None else "",
                    "c2_recovery": round(c2_rec, 4) if c2_rec is not None else "",
                    "c3_recovery": round(c3_rec, 4) if c3_rec is not None else "",
                })
    return rows


def aggregate_n2(rows):
    summary = []
    per_step = []
    for m in METHODS:
        mrows = [r for r in rows if r["method"] == m]
        # Mixed-only rows: where ceiling > 0
        mixed_rows = [r for r in mrows if r["ceiling"] > 1e-9]
        # Pooled recovery over all mixed prompts (mean change / mean ceiling is wrong;
        # use sum(change) / sum(ceiling) for the population-level ratio).
        sum_ceiling = sum(r["ceiling"] for r in mixed_rows)
        sum_c1 = sum(r["c1_change"] for r in mixed_rows)
        sum_c2 = sum(r["c2_change"] for r in mixed_rows)
        sum_c3 = sum(r["c3_change"] for r in mixed_rows)
        summary.append({
            "method": m, "scope": "mixed_only",
            "controller": "C1_zvf_triage@0.70",
            "n_prompts": len(mixed_rows),
            "sum_ceiling": round(sum_ceiling, 4),
            "sum_change": round(sum_c1, 4),
            "recovery_ratio": round(sum_c1 / sum_ceiling, 4) if sum_ceiling > 0 else 0.0,
        })
        summary.append({
            "method": m, "scope": "mixed_only",
            "controller": "C2_dualformer@0.70",
            "n_prompts": len(mixed_rows),
            "sum_ceiling": round(sum_ceiling, 4),
            "sum_change": round(sum_c2, 4),
            "recovery_ratio": round(sum_c2 / sum_ceiling, 4) if sum_ceiling > 0 else 0.0,
        })
        summary.append({
            "method": m, "scope": "mixed_only",
            "controller": "C3_hybrid@0.70+0.20",
            "n_prompts": len(mixed_rows),
            "sum_ceiling": round(sum_ceiling, 4),
            "sum_change": round(sum_c3, 4),
            "recovery_ratio": round(sum_c3 / sum_ceiling, 4) if sum_ceiling > 0 else 0.0,
        })
        # Per-step recovery (mean over mixed prompts at the step)
        for s in range(40):
            srows = [r for r in mrows if r["step"] == s]
            if not srows:
                continue
            z_step = srows[0]["z_step"]
            c1_g = srows[0]["c1_g"]; c2_g = srows[0]["c2_g"]; c3_g = srows[0]["c3_g"]
            mixed_s = [r for r in srows if r["ceiling"] > 1e-9]
            n_mixed = len(mixed_s)
            ceil_s = sum(r["ceiling"] for r in mixed_s)
            c1_s = sum(r["c1_change"] for r in mixed_s)
            c2_s = sum(r["c2_change"] for r in mixed_s)
            c3_s = sum(r["c3_change"] for r in mixed_s)
            per_step.append({
                "method": m, "step": s, "z_step": round(z_step, 4),
                "n_mixed": n_mixed,
                "ceiling_sum": round(ceil_s, 4),
                "c1_recovery": round(c1_s / ceil_s, 4) if ceil_s > 0 else "",
                "c2_recovery": round(c2_s / ceil_s, 4) if ceil_s > 0 else "",
                "c3_recovery": round(c3_s / ceil_s, 4) if ceil_s > 0 else "",
                "c1_g": c1_g, "c2_g": c2_g, "c3_g": c3_g,
            })
    return summary, per_step


def bootstrap_recovery_ci(per_prompt_rows, controller_col, n_boot=N_BOOT, seed=BOOT_SEED):
    """Block-bootstrap (method, step) of population-level recovery ratio."""
    rng = random.Random(seed)
    cells = {}
    for r in per_prompt_rows:
        cells.setdefault((r["method"], r["step"]), []).append(r)
    cell_keys = list(cells.keys())
    boot_ratios = []
    for _ in range(n_boot):
        pool_ceiling = 0.0
        pool_change = 0.0
        for _ in range(len(cell_keys)):
            ck = rng.choice(cell_keys)
            for r in cells[ck]:
                if r["ceiling"] > 1e-9:
                    pool_ceiling += r["ceiling"]
                    pool_change += r[controller_col]
        ratio = pool_change / pool_ceiling if pool_ceiling > 0 else 0.0
        boot_ratios.append(ratio)
    boot_ratios.sort()
    n = len(boot_ratios)
    lo = boot_ratios[int(0.025 * n)]
    hi = boot_ratios[int(0.975 * n)]
    return lo, hi


def aliasing_test(per_step):
    """Pearson correlation between step-level ZVF and mean per-prompt headroom at step."""
    out = []
    for m in METHODS:
        ms = [r for r in per_step if r["method"] == m]
        zs = [r["z_step"] for r in ms]
        cs = [r["ceiling_sum"] / max(r["n_mixed"], 1) for r in ms]
        if len(zs) > 2:
            zbar = sum(zs) / len(zs)
            cbar = sum(cs) / len(cs)
            num = sum((zi - zbar) * (ci - cbar) for zi, ci in zip(zs, cs))
            dz = math.sqrt(sum((zi - zbar) ** 2 for zi in zs))
            dc = math.sqrt(sum((ci - cbar) ** 2 for ci in cs))
            rho = num / (dz * dc) if dz * dc > 0 else 0.0
        else:
            rho = 0.0
        out.append({"method": m, "pearson_rho": round(rho, 4),
                    "n_steps": len(ms)})
    return out


def aggregate_n10(n10_step_log):
    """N10 has only step-level ZVF; ceiling UNKNOWN at prompt level.
    We report per-step controller outputs and total compute change vs baseline.
    """
    summary = []
    per_step = []
    for s, slogs in n10_step_log.items():
        for entry in slogs:
            step = entry["step"]
            z = entry["zvf"]
            c1_g, c2_g, c3_g = per_step_controllers(z)
            per_step.append({
                "seed": s, "step": step, "zvf_step": round(z, 4),
                "c1_g": c1_g, "c2_g": c2_g, "c3_g": c3_g,
            })
    # Per-seed totals
    for s in N10_SEEDS:
        srows = [r for r in per_step if r["seed"] == s]
        total_G_base = sum(G_BASE for _ in srows)
        total_c1 = sum(r["c1_g"] for r in srows)
        total_c2 = sum(r["c2_g"] for r in srows)
        total_c3 = sum(r["c3_g"] for r in srows)
        summary.append({
            "seed": s, "controller": "C0_baseline",
            "total_G": total_G_base, "savings": 0.0,
        })
        summary.append({
            "seed": s, "controller": "C1_zvf_triage@0.70",
            "total_G": total_c1,
            "savings": round((total_G_base - total_c1) / total_G_base, 4),
        })
        summary.append({
            "seed": s, "controller": "C2_dualformer@0.70",
            "total_G": total_c2,
            "savings": round((total_G_base - total_c2) / total_G_base, 4),
        })
        summary.append({
            "seed": s, "controller": "C3_hybrid@0.70+0.20",
            "total_G": total_c3,
            "savings": round((total_G_base - total_c3) / total_G_base, 4),
        })
    return summary, per_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("== Loading N2 reward tensors ==")
    n2 = load_n2_rewards()
    print(f"  loaded {len(n2)} (method, step) cells")

    print("== Loading N10 step_log ==")
    n10 = load_n10_step_log()
    print(f"  loaded {len(n10)} seeds x {N10_STEPS} steps")

    print("== Computing per-prompt headroom + per-controller change (N2) ==")
    rows = analyze_n2(n2)
    print(f"  {len(rows)} prompt-step rows")
    n2_summary, n2_per_step = aggregate_n2(rows)

    print("== Bootstrap CIs on N2 recovery ratios (mixed-prompt scope) ==")
    for m in METHODS:
        mrows = [r for r in rows if r["method"] == m]
        for col, label in [("c1_change", "C1_zvf_triage@0.70"),
                           ("c2_change", "C2_dualformer@0.70"),
                           ("c3_change", "C3_hybrid@0.70+0.20")]:
            lo, hi = bootstrap_recovery_ci(mrows, col)
            for entry in n2_summary:
                if entry["method"] == m and entry["controller"] == label:
                    entry["recovery_lo"] = round(lo, 4)
                    entry["recovery_hi"] = round(hi, 4)

    aliasing = aliasing_test(n2_per_step)

    n10_summary, n10_per_step = aggregate_n10(n10)

    print("\n== N2 mixed-prompt recovery ratios (n=2,560 - saturated = ~720 mixed) ==")
    for e in n2_summary:
        print(f"  {e['method']:>5s} {e['controller']:>22s}: "
              f"recovery={e['recovery_ratio']:.4f} "
              f"[{e.get('recovery_lo', 0):.4f}, {e.get('recovery_hi', 0):.4f}] "
              f"n={e['n_prompts']} ceil_sum={e['sum_ceiling']:.3f} "
              f"change_sum={e['sum_change']:.3f}")

    print("\n== Step-ZVF vs per-prompt headroom aliasing (Pearson rho) ==")
    for e in aliasing:
        print(f"  {e['method']:>5s}: rho = {e['pearson_rho']:+.4f} (n_steps={e['n_steps']})")

    print("\n== N10 compute totals (per seed, per controller) ==")
    for e in n10_summary:
        print(f"  seed={e['seed']} {e['controller']:>22s}: total_G={e['total_G']} savings={e['savings']:+.4f}")

    if not args.write:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "p7_headroom_recovery_n2_summary.tsv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scope", "controller", "n_prompts",
                                          "sum_ceiling", "sum_change", "recovery_ratio",
                                          "recovery_lo", "recovery_hi"],
                           delimiter="\t")
        w.writeheader()
        for e in n2_summary:
            w.writerow(e)

    per_step_path = OUT_DIR / "p7_headroom_recovery_n2_per_step.tsv"
    with open(per_step_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(n2_per_step[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for e in n2_per_step:
            w.writerow(e)

    per_prompt_path = OUT_DIR / "p7_headroom_recovery_n2_per_prompt.tsv"
    with open(per_prompt_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n10_summary_path = OUT_DIR / "p7_headroom_recovery_n10_summary.tsv"
    with open(n10_summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(n10_summary[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for e in n10_summary:
            w.writerow(e)

    n10_per_step_path = OUT_DIR / "p7_headroom_recovery_n10_per_step.tsv"
    with open(n10_per_step_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(n10_per_step[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for e in n10_per_step:
            w.writerow(e)

    summary_json = OUT_DIR / "p7_headroom_recovery_summary.json"
    summary_json.write_text(json.dumps({
        "config": {
            "G_BASE": G_BASE, "G_ESC": G_ESC, "G_DES": G_DES,
            "TAU": TAU, "TAU_DELTA": TAU_DELTA,
            "TARGET_ZVF": TARGET_ZVF,
            "n_methods": 4, "n_steps_n2": 40, "n_prompts_per_step": 16,
            "n_seeds_n10": 5, "n_steps_n10": N10_STEPS,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "headroom_definition": "ZVF(G_base) - ZVF(G_esc); capped per-prompt at ZVF(G_base) for mixed prompts",
            "recovery_definition": "sum(ZVF(G_base) - ZVF(G_ctrl)) / sum(ZVF(G_base) - ZVF(G_esc)) over mixed prompts",
        },
        "n2_summary": n2_summary,
        "aliasing": aliasing,
        "n10_summary": n10_summary,
    }, indent=2))

    print(f"\nWrote:")
    for p in [summary_path, per_step_path, per_prompt_path,
              n10_summary_path, n10_per_step_path, summary_json]:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()