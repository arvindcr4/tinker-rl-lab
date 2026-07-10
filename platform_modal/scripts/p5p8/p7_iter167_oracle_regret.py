#!/usr/bin/env python3
"""
Iter 167 — P7 Oracle-Regret Counterfactual

For each (method, step, prompt) observation on the N2 four-method tensor
corpus, compute the *oracle-optimal* adaptive group-size G* then measure the
regret of each empirical controller (C0 fixed, C1 zvf-triage, C2 Dualformer,
C3 Hybrid, C5 Iso-G) against oracle.

Oracle criterion (cost-effective, per frontier synthesis Round 2):
    maximize  DeltaY(p_hat, G') / max(1, G' - G_base)
where
    DeltaY(p_hat, G') = ZVF_iid(p_hat, G_base) - ZVF_iid(p_hat, G')
    ZVF_iid(p, G)     = p^G + (1-p)^G
and G' is restricted to {2, 4, 6, 8, 10, 12, 16, 24, 32}. If p_hat is exactly
0 or 1, no G' restores contrast (DeltaY stays 0) so the oracle stays at
G_base=8.

This is the *operational* form of "how much yield would an oracle controller
capture, and how much of that can each empirical controller actually
recover?" — exactly the "from diagnostic to controller" demonstration the
P7 paper promises.

Outputs
-------
- platform_hybrid/experiments/results/p5p8/p7_iter167_oracle_per_obs.tsv      (2560 obs: oracle G*, oracle ΔY, G from each controller, regret)
- platform_hybrid/experiments/results/p5p8/p7_iter167_oracle_per_step.tsv     (4×40: per-step oracle yield vs each controller yield)
- platform_hybrid/experiments/results/p5p8/p7_iter167_oracle_regret_by_method.tsv  (4 methods × 5 controllers: cumulative regret & % oracle captured)
- platform_hybrid/experiments/results/p5p8/p7_iter167_oracle_regret_summary.json
- platform_hybrid/experiments/results/p5p8/p7_iter167_oracle_regret_bootstrap.tsv  (bootstrap CIs on %oracle captured per controller, B=2000)
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import statistics
from typing import Dict, List, Tuple

WORKTREE = "/home/claude/tinker-rl-lab-minimax"
DATA_DIR = os.path.join(WORKTREE, "platform_hybrid/experiments/results/n2_reward_tensor_resume")
OUT_DIR = os.path.join(WORKTREE, "platform_hybrid/experiments/results/p5p8")
os.makedirs(OUT_DIR, exist_ok=True)

G_BASE = 8
G_MENU = [2, 4, 6, 8, 10, 12, 16, 24, 32]

METHODS = ["aero", "areal", "gift", "grpo"]


# ---------- helpers (pure math, vectorised trivially) ----------

def zvf_iid(p: float, G: int) -> float:
    """P(K=0) + P(K=G) under Bin(G, p)."""
    if p <= 0.0:
        return 1.0 if G >= 1 else 0.0  # K=0 always
    if p >= 1.0:
        return 1.0
    return p**G + (1.0 - p) ** G


def yield_iid(p: float, G: int) -> float:
    return 1.0 - zvf_iid(p, G)


def per_prompt_delta_y(p_hat: float, G_prime: int) -> float:
    """Contrast restored by moving from G_base to G'."""
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 0.0
    return yield_iid(p_hat, G_prime) - yield_iid(p_hat, G_BASE)


def per_prompt_yield_per_extra(p_hat: float, G_prime: int) -> float:
    """Yield restored per additional rollout (cost-effective metric)."""
    dy = per_prompt_delta_y(p_hat, G_prime)
    extra = max(1, G_prime - G_BASE)
    return dy / extra


def oracle_g_star(p_hat: float) -> Tuple[int, float, float, float]:
    """Pick G' that maximises ΔY / max(1, G' - 8). Return (G*, dY, dY_per_extra, was_oracle_active)."""
    best_g, best_score, best_dy = G_BASE, -math.inf, 0.0
    for g in G_MENU:
        if g == G_BASE:
            continue
        dY = per_prompt_delta_y(p_hat, g)
        if dY <= 1e-12:
            continue
        score = per_prompt_yield_per_extra(p_hat, g)
        if score > best_score:
            best_g, best_score, best_dy = g, score, dY
    # If best_score still -inf or zero, oracle stays at base.
    if best_score <= 1e-12:
        return G_BASE, 0.0, 0.0, False
    return best_g, best_dy, best_score, True


# ---------- empirical controller rules ----------

def ctrl_zvf_triage(step_zvf: float, tau: float = 0.70) -> int:
    """C1: escalate to G'=16 when step-zvf ≥ τ, else G'=8."""
    return 16 if step_zvf >= tau else G_BASE


def ctrl_dualformer_auto(p_hat: float) -> int:
    """C2: Dualformer per-prompt G from Berkeley row 01.
       Bands on p_hat ∈ (0,1):  p̂≤0.25 -> G=16 ;  0.25<p̂≤0.5 -> G=8 ;
       0.5<p̂≤0.75 -> G=4 ;  p̂>0.75 -> G=2. Stays G=8 if p̂ == 0 or 1.
    """
    if p_hat <= 0.0 or p_hat >= 1.0:
        return G_BASE
    if p_hat <= 0.25:
        return 16
    if p_hat <= 0.5:
        return G_BASE
    if p_hat <= 0.75:
        return 4
    return 2


def ctrl_hybrid(step_zvf: float, tau_low: float = 0.70, tau_high: float = 0.90) -> int:
    """C3: escalate to G=16 in band [τ_low, τ_high); else G=8 (saturation band ≥ τ_high stays at base)."""
    if tau_low <= step_zvf < tau_high:
        return 16
    return G_BASE


def ctrl_iso_g(p_hat: float, tau_y: float = 0.90) -> int:
    """C5: Iso-G, smallest G' achieving Y >= τ_y."""
    if p_hat <= 0.0 or p_hat >= 1.0:
        return G_BASE
    for g in sorted(set(G_MENU)):
        if yield_iid(p_hat, g) >= tau_y:
            return g
    return G_BASE


CONTROLLERS = {
    "C0_fixed_G8":     lambda p_hat, step_zvf: G_BASE,
    "C1_zvf_triage":   lambda p_hat, step_zvf: ctrl_zvf_triage(step_zvf),
    "C2_dualformer":   lambda p_hat, step_zvf: ctrl_dualformer_auto(p_hat),
    "C3_hybrid":       lambda p_hat, step_zvf: ctrl_hybrid(step_zvf),
    "C5_isog_y090":    lambda p_hat, step_zvf: ctrl_iso_g(p_hat, 0.90),
}


# ---------- load corpus ----------

def load_corpus() -> List[dict]:
    """Return flat list of {method, seed, step, prompt_idx, k, p_hat, rewards[8], zvf}."""
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_s0_tensors.jsonl"))):
        method = os.path.basename(path).split("_")[0]
        for ln in open(path):
            d = json.loads(ln)
            step = d["step"]
            step_zvf = d["zvf"]
            for pi, prompt_rewards in enumerate(d["rewards"]):
                k = int(sum(prompt_rewards))
                p_hat = k / G_BASE
                rows.append({
                    "method": method,
                    "seed": d.get("seed", 0),
                    "step": step,
                    "prompt_idx": pi,
                    "k": k,
                    "p_hat": p_hat,
                    "rewards": prompt_rewards,
                    "step_zvf": step_zvf,
                })
    return rows


# ---------- main computations ----------

def compute_obs_table(corpus: List[dict]) -> List[dict]:
    obs = []
    for r in corpus:
        p_hat = r["p_hat"]
        zvf_step = r["step_zvf"]
        # Oracle
        g_star, dY_star, score_star, oracle_active = oracle_g_star(p_hat)
        # Controllers
        ctrl_rows = {}
        for cname, fn in CONTROLLERS.items():
            g_chosen = fn(p_hat, zvf_step)
            dY_c = per_prompt_delta_y(p_hat, g_chosen)
            ctrl_rows[cname] = (g_chosen, dY_c)
        # regret vs oracle (only dY terms)
        obs.append({
            "method": r["method"],
            "step": r["step"],
            "prompt_idx": r["prompt_idx"],
            "k": r["k"],
            "p_hat": p_hat,
            "step_zvf": zvf_step,
            "oracle_g": g_star,
            "oracle_dY": dY_star,
            "oracle_active": bool(oracle_active),
            "oracle_score": score_star,
            **{f"{c}_G": v[0] for c, v in ctrl_rows.items()},
            **{f"{c}_dY": v[1] for c, v in ctrl_rows.items()},
        })
    return obs


def per_step_aggregate(obs: List[dict]) -> List[dict]:
    """For each (method, step): mean oracle_dY, mean ctrl_dY across prompts."""
    by = {}
    for o in obs:
        key = (o["method"], o["step"])
        by.setdefault(key, {"oracle_dY": [], "step_zvf": o["step_zvf"]})
        for c in CONTROLLERS:
            by[key].setdefault(f"{c}_dY", []).append(o[f"{c}_dY"])
        by[key]["oracle_dY"].append(o["oracle_dY"])
    out = []
    for (method, step), v in sorted(by.items()):
        row = {"method": method, "step": step, "step_zvf": v["step_zvf"]}
        row["oracle_dY_mean"] = statistics.fmean(v["oracle_dY"])
        for c in CONTROLLERS:
            row[f"{c}_dY_mean"] = statistics.fmean(v[f"{c}_dY"])
        out.append(row)
    return out


def per_method_aggregate(obs: List[dict]) -> List[dict]:
    """For each (method, controller): cumulative absolute ΔY, cumulative dY/cost, % oracle captured,
       and the *cost-effective* alternative: oracle picks G' to maximise dY/extras; controller
       picks dY/extras directly. We report both axes."""
    out = []
    for method in METHODS:
        m_obs = [o for o in obs if o["method"] == method]
        oracle_abs = sum(o["oracle_dY"] for o in m_obs)
        oracle_extras_total = sum(max(0, o["oracle_g"] - G_BASE) for o in m_obs)
        oracle_costeff = oracle_abs / max(1, oracle_extras_total) * 1000.0  # ΔY per 1000 extra rollouts
        for c in CONTROLLERS:
            ctrl_abs = sum(o[f"{c}_dY"] for o in m_obs)
            ctrl_extras = sum(max(0, o[f"{c}_G"] - G_BASE) for o in m_obs)
            ctrl_costeff = ctrl_abs / max(1, ctrl_extras) * 1000.0 if ctrl_extras > 0 else 0.0
            captured = (ctrl_abs / oracle_abs) * 100.0 if oracle_abs > 0 else 0.0
            regret = oracle_abs - ctrl_abs
            # Cost-effective ratio of controller cosine to oracle cosine
            costeff_ratio = (ctrl_costeff / oracle_costeff) if oracle_costeff > 0 else 0.0
            out.append({
                "method": method,
                "controller": c,
                "oracle_total_abs_dY": oracle_abs,
                "controller_total_abs_dY": ctrl_abs,
                "pct_oracle_abs_captured": captured,
                "regret_abs_dY": regret,
                "oracle_costeff_dY_per_1k": oracle_costeff,
                "controller_costeff_dY_per_1k": ctrl_costeff,
                "costeff_ratio_to_oracle": costeff_ratio,
                "ctrl_total_extras": ctrl_extras,
                "oracle_total_extras": oracle_extras_total,
                "n_obs": len(m_obs),
                "n_oracle_active": sum(1 for o in m_obs if o["oracle_active"]),
            })
    return out


# ---------- bootstrap CIs on the two dual axes ----------

def bootstrap_axes(obs: List[dict], B: int = 2000, seed: int = 20260705) -> List[dict]:
    """Per (method, controller):
       (a) pct_oracle_abs_captured (% point): ctrl_abs / oracle_abs.
       (b) costeff_ratio: (ctrl_abs/ctrl_extras) / (oracle_abs/oracle_extras).
       Bootstrap 95% percentile CIs (B resamples, step-zvf already attached to each obs)."""
    rng = random.Random(seed)
    out = []
    for method in METHODS:
        m_obs = [o for o in obs if o["method"] == method]
        for c in CONTROLLERS:
            abs_boots, ce_boots = [], []
            for _ in range(B):
                rs = [m_obs[rng.randrange(len(m_obs))] for _ in range(len(m_obs))]
                o_abs = sum(o["oracle_dY"] for o in rs)
                c_abs = sum(o[f"{c}_dY"] for o in rs)
                abs_boots.append((c_abs / o_abs) * 100.0 if o_abs > 0 else 0.0)
                o_ex = sum(max(0, o["oracle_g"] - G_BASE) for o in rs)
                c_ex = sum(max(0, o[f"{c}_G"] - G_BASE) for o in rs)
                if o_ex > 0 and c_ex > 0:
                    ce_boots.append((c_abs / c_ex) / (o_abs / o_ex))
                else:
                    ce_boots.append(0.0)
            abs_boots.sort(); ce_boots.sort()
            point_abs = (sum(o[f"{c}_dY"] for o in m_obs) / sum(o["oracle_dY"] for o in m_obs)) * 100.0
            # point for costeff ratio
            c_abs_pt = sum(o[f"{c}_dY"] for o in m_obs)
            c_ex_pt = sum(max(0, o[f"{c}_G"] - G_BASE) for o in m_obs)
            o_abs_pt = sum(o["oracle_dY"] for o in m_obs)
            o_ex_pt = sum(max(0, o["oracle_g"] - G_BASE) for o in m_obs)
            point_ce = ((c_abs_pt / c_ex_pt) / (o_abs_pt / o_ex_pt)) if (c_ex_pt > 0 and o_ex_pt > 0) else 0.0
            out.append({
                "method": method,
                "controller": c,
                "pct_abs_point": point_abs,
                "pct_abs_ci95_lo": abs_boots[int(0.025 * B)],
                "pct_abs_ci95_hi": abs_boots[int(0.975 * B) - 1],
                "costeff_ratio_point": point_ce,
                "costeff_ratio_ci95_lo": ce_boots[int(0.025 * B)],
                "costeff_ratio_ci95_hi": ce_boots[int(0.975 * B) - 1],
                "B": B,
                "seed": seed,
            })
    return out


# ---------- IO helpers ----------

def write_tsv(path: str, rows: List[dict]) -> None:
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # cast numpy-ish values
            r2 = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in r.items()}
            w.writerow(r2)


def main() -> None:
    corpus = load_corpus()
    print(f"[iter167] loaded {len(corpus)} obs")
    obs = compute_obs_table(corpus)
    print(f"[iter167] computed {len(obs)} obs with oracle + controllers")

    write_tsv(os.path.join(OUT_DIR, "p7_iter167_oracle_per_obs.tsv"), obs)
    print(f"[iter167] wrote per-obs TSV ({len(obs)} rows)")

    per_step = per_step_aggregate(obs)
    write_tsv(os.path.join(OUT_DIR, "p7_iter167_oracle_per_step.tsv"), per_step)
    print(f"[iter167] wrote per-step TSV ({len(per_step)} rows)")

    per_method = per_method_aggregate(obs)
    write_tsv(os.path.join(OUT_DIR, "p7_iter167_oracle_regret_by_method.tsv"), per_method)
    print(f"[iter167] wrote per-method TSV ({len(per_method)} rows)")

    boot = bootstrap_axes(obs)
    write_tsv(os.path.join(OUT_DIR, "p7_iter167_oracle_regret_bootstrap.tsv"), boot)
    print(f"[iter167] wrote bootstrap TSV ({len(boot)} rows)")

    summary = {
        "n_obs": len(obs),
        "methods": METHODS,
        "g_base": G_BASE,
        "g_menu": G_MENU,
        "controllers": list(CONTROLLERS.keys()),
        "oracle_active_rate_pct": {
            method: 100.0 * sum(1 for o in obs if o["method"] == method and o["oracle_active"]) / sum(1 for o in obs if o["method"] == method)
            for method in METHODS
        },
        "per_method": per_method,
        "bootstrap": boot,
    }
    with open(os.path.join(OUT_DIR, "p7_iter167_oracle_regret_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter167] wrote summary JSON")

    # Quick console summary
    print("\n=== Pct oracle contrast ABS-captured (point) ===")
    print(f"{'method':<8} {'C0_fixed':>10} {'C1_zvf':>10} {'C2_dual':>10} {'C3_hyb':>10} {'C5_isoG':>10}")
    for method in METHODS:
        ms = [r for r in per_method if r["method"] == method]
        row = [f"{method:<8}"]
        for c in CONTROLLERS:
            v = next(r["pct_oracle_abs_captured"] for r in ms if r["controller"] == c)
            row.append(f"{v:>9.1f}%")
        print(" ".join(row))
    print("\n=== Cost-effective ratio (ctrl / oracle) > 1.0 means controller > oracle per extra rollout ===")
    print(f"{'method':<8} {'C0':>8} {'C1':>8} {'C2':>8} {'C3':>8} {'C5':>8}")
    for method in METHODS:
        ms = [r for r in per_method if r["method"] == method]
        row = [f"{method:<8}"]
        for c in CONTROLLERS:
            v = next(r["costeff_ratio_to_oracle"] for r in ms if r["controller"] == c)
            row.append(f"{v:>7.2f}x")
        print(" ".join(row))
    print("\n=== Controller absolute ΔY (cost-effective extra contrast across 640 prompts) ===")
    print(f"{'method':<8} {'C0':>9} {'C1':>9} {'C2':>9} {'C3':>9} {'C5':>9} {'oracle':>9}")
    for method in METHODS:
        ms = [r for r in per_method if r["method"] == method]
        row = [f"{method:<8}"]
        for c in CONTROLLERS:
            v = next(r["controller_total_abs_dY"] for r in ms if r["controller"] == c)
            row.append(f"{v:>8.3f}")
        oracle_v = ms[0]["oracle_total_abs_dY"]
        row.append(f"{oracle_v:>8.3f}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
