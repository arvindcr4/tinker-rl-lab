#!/usr/bin/env python3
"""P7 iter-83: Iso-Yield Dynamic Grouping (Iso-G) controller prototype.

Frontier synthesis (FRONTIER_INSIGHTS.md Round 2): "Iso-Yield Dynamic
Grouping (Iso-G). Mechanism: Abandon [static G]". For each (method,
step, prompt) on the N2 four-method 2560-decision corpus, choose G' to
achieve Y(p_hat, G') >= tau_y at minimum rollout cost; Y = 1 - ZVF_iid.

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_iter83_iso_g_per_prompt.tsv
  platform_hybrid/experiments/results/p5p8/p7_iter83_iso_g_per_method.tsv
  platform_hybrid/experiments/results/p5p8/p7_iter83_iso_g_summary.json
"""
import json
from collections import defaultdict
from math import lgamma, exp, log
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
OUT = WORK / "platform_hybrid/experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
G_ESC = 16
G_DES = 4
G_CHOICES = [2, 4, 6, 8, 10, 12, 16, 24, 32]


def yield_iid(p: float, G: int) -> float:
    return 1.0 - (p ** G + (1.0 - p) ** G)


def midrange_prob(k: int, G: int) -> float:
    """Pr(0.05 <= p <= 0.95 | Beta(k+1, G-k+1)) via Simpson's rule."""
    a, b = k + 1, G - k + 1
    log_norm = lgamma(a + b) - lgamma(a) - lgamma(b)
    def lp(p): return log_norm + (a - 1) * log(p + 1e-300) + (b - 1) * log(1 - p + 1e-300)
    n = 200  # Simpson's rule panels
    h = 0.90 / n
    s = exp(lp(0.05)) + exp(lp(0.95))
    for i in range(1, n):
        x = 0.05 + i * h
        w = 4 if i % 2 == 1 else 2
        s += w * exp(lp(x))
    return s * h / 3.0


def iso_g(k: int, tau_y: float) -> int:
    """Smallest G' in G_CHOICES with yield_iid(k/G_BASE, G') >= tau_y."""
    p = k / G_BASE
    for gp in G_CHOICES:
        if yield_iid(p, gp) >= tau_y:
            return gp
    return G_BASE


def load_n2():
    rows = []
    for m in METHODS:
        with open(N2_DIR / f"{m}_s0_tensors.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                for pi, rewards in enumerate(rec["rewards"]):
                    rows.append({
                        "method": m, "step": rec["step"],
                        "prompt_idx": pi, "k": int(round(sum(rewards))),
                    })
    return rows


def decisions_for(rows, step_zvf_cache, ctrl, tau):
    gps = []
    for r in rows:
        k, p = r["k"], r["k"] / G_BASE
        z = step_zvf_cache[(r["method"], r["step"])]
        if ctrl == "C0_fixed": gp = G_BASE
        elif ctrl == "C1_zvf_triage": gp = G_ESC if z >= tau else G_BASE
        elif ctrl == "C2_dualformer":
            gp = 2 if p >= 0.95 else 4 if p >= 0.85 else 8 if p >= 0.70 else 16
        elif ctrl == "C3_hybrid": gp = G_DES if z >= tau + 0.20 else G_ESC if z >= tau else G_BASE
        elif ctrl == "C4_bayesian": gp = G_ESC if midrange_prob(k, G_BASE) >= 0.60 else G_BASE
        elif ctrl.startswith("C5_iso_g"):
            gp = iso_g(k, float(ctrl.split("_")[-1]))
        else: raise ValueError(ctrl)
        gps.append(gp)
    return gps


def main():
    rows = load_n2()
    print(f"[load] N={len(rows)} prompt-step decisions "
          f"({len(METHODS)} methods × 40 steps × 16 prompts)")

    # Pre-compute per-(method, step) zvf = fraction of boundary prompts
    step_zvf_cache = {}
    by_ms = defaultdict(list)
    for r in rows:
        by_ms[(r["method"], r["step"])].append(r)
    for key, obs in by_ms.items():
        step_zvf_cache[key] = sum(1 for x in obs if x["k"] in (0, G_BASE)) / len(obs)

    controllers = [
        ("C0_fixed", 0.70), ("C1_zvf_triage", 0.70), ("C2_dualformer", 0.70),
        ("C3_hybrid", 0.70), ("C4_bayesian", 0.60),
        ("C5_iso_g_0.50", 0.50), ("C5_iso_g_0.70", 0.70),
        ("C5_iso_g_0.90", 0.90), ("C5_iso_g_0.95", 0.95),
    ]
    baseline_rollouts = len(rows) * G_BASE

    decisions = {c: decisions_for(rows, step_zvf_cache, c, t)
                 for c, t in controllers}

    # Per-prompt rows
    pp = []
    for i, r in enumerate(rows):
        p = r["k"] / G_BASE
        y_base = yield_iid(p, G_BASE)
        for ctrl, _ in controllers:
            gp = decisions[ctrl][i]
            pp.append({
                "method": r["method"], "step": r["step"],
                "prompt_idx": r["prompt_idx"], "k": r["k"],
                "p_hat": round(p, 4), "controller": ctrl,
                "G_prime": gp,
                "yield_base": round(y_base, 4),
                "yield_ctrl": round(yield_iid(p, gp), 4),
                "delta_yield": round(yield_iid(p, gp) - y_base, 4),
            })

    pp_path = OUT / "p7_iter83_iso_g_per_prompt.tsv"
    cols = ["method", "step", "prompt_idx", "k", "p_hat", "controller",
            "G_prime", "yield_base", "yield_ctrl", "delta_yield"]
    with open(pp_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in pp:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"[write] {pp_path} ({len(pp)} rows)")

    # Per-method summary
    per_method = []
    for m in METHODS:
        mrows = [r for r in pp if r["method"] == m]
        baseline_r = 640 * G_BASE
        for ctrl, _ in controllers:
            crows = [r for r in mrows if r["controller"] == ctrl]
            tg = sum(r["G_prime"] for r in crows)
            tdy = sum(r["delta_yield"] for r in crows)
            extra = tg - baseline_r
            per_method.append({
                "method": m, "controller": ctrl,
                "total_rollouts": tg, "baseline_rollouts": baseline_r,
                "cost_ratio": round(tg / baseline_r, 4),
                "n_fires": sum(1 for r in crows if r["G_prime"] != G_BASE),
                "n_escalated": sum(1 for r in crows if r["G_prime"] > G_BASE),
                "n_deescalated": sum(1 for r in crows if r["G_prime"] < G_BASE),
                "total_delta_yield": round(tdy, 4),
                "mean_delta_yield_per_prompt": round(tdy / len(crows), 4),
                "yield_per_1000_extra_rollouts": (
                    round(tdy * 1000.0 / extra, 2) if extra > 0 else "inf"
                ),
            })

    pm_path = OUT / "p7_iter83_iso_g_per_method.tsv"
    pm_cols = ["method", "controller", "total_rollouts", "baseline_rollouts",
               "cost_ratio", "n_fires", "n_escalated", "n_deescalated",
               "total_delta_yield", "mean_delta_yield_per_prompt",
               "yield_per_1000_extra_rollouts"]
    with open(pm_path, "w") as f:
        f.write("\t".join(pm_cols) + "\n")
        for r in per_method:
            f.write("\t".join(str(r[c]) for c in pm_cols) + "\n")
    print(f"[write] {pm_path} ({len(per_method)} rows)")

    # Headlines
    summary = {
        "iter": 83, "pillar": "P7", "vein": "iso_g_controller",
        "n_prompt_step_decisions": len(rows), "n_methods": len(METHODS),
        "n_steps": 40, "n_prompts_per_step": 16,
        "G_BASE": G_BASE, "G_ESC": G_ESC, "G_DES": G_DES,
        "G_CHOICES": G_CHOICES,
        "baseline_rollouts": baseline_rollouts,
        "controllers": [c for c, _ in controllers],
        "headline": {},
    }
    # Best cost-ratio Iso-G variant with non-negative yield
    iso_pos = [r for r in per_method
               if r["controller"].startswith("C5_iso_g") and r["total_delta_yield"] > 0]
    if iso_pos:
        best = min(iso_pos, key=lambda r: r["cost_ratio"])
        summary["headline"]["best_cost_ratio_under_iso_g"] = {
            "controller": best["controller"],
            "cost_ratio": best["cost_ratio"],
            "total_delta_yield": best["total_delta_yield"],
            "n_fires": best["n_fires"],
        }
    # Best yield-per-1k-extra across all controllers
    finite = [r for r in per_method
              if isinstance(r["yield_per_1000_extra_rollouts"], (int, float))]
    if finite:
        best = max(finite, key=lambda r: r["yield_per_1000_extra_rollouts"])
        summary["headline"]["best_yield_per_1k_extra"] = {
            "controller": best["controller"], "method": best["method"],
            "yield_per_1k_extra": best["yield_per_1000_extra_rollouts"],
            "cost_ratio": best["cost_ratio"],
        }

    json_path = OUT / "p7_iter83_iso_g_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {json_path}")

    print("\n=== Iso-G headline ===")
    for k, v in summary["headline"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()