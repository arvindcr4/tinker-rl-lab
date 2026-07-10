#!/usr/bin/env python3
"""
Iter 175 - P7 Calibrated-Hybrid (C6) Controller (v2 - intersection logic).

The C6 from v1 degenerated into C2 because the AlphaProof-style "gamma* = 0
smoothing" was applied as a scalar blend instead of a hard *anchor*. This v2
treats gamma* = 0 as the literal "no-act" prior and only yields escalation
when Dualformer's per-prompt wishful target AND the empirical step-zvf
evidence concur (i.e., an intersection of two independent signals).

Algorithmically:
    zvf_signal = step_zvf (raw, 0..1)
    bimodal    = 4 * p_hat * (1 - p_hat)  (peaks at p_hat=0.5; >0 only on
                                         non-degenerate prompts)
    g_dual     = Berkeley row 01 Dualformer auto-G target
    fire_due_to_dualformer = (g_dual != G_base) and (bimodal > 0.05)
    fire_due_to_zvf        = zvf_signal >= zvf_tau
    # Calibrated-hybrid (C6) - intersection-and-fallback:
    if   fire_due_to_dualformer AND fire_due_to_zvf:
        return g_dual                                # both signals agree
    elif fire_due_to_zvf:
        return 16                                    # zvf alone -> safe G=16
    else:
        return G_base                                # gamma*=0 anchor

Default tunables (alpha=0.5, zvf_tau=0.70, gamma_tau=0.20) are surface-only
parameters; the v2 algorithm only depends on zvf_tau. The sweep covers
zvf_tau in {0.55, 0.65, 0.70, 0.75, 0.85} to expose the Pareto frontier
across the operative Operating Point the controller chooses.

Compared against the 5 existing controllers from iter-167 (C0..C5) on:
  (a) contrast-restored total |delta Y| per (method, controller)
  (b) compute cost in extra rollouts
  (c) Pareto-optimality vs iter-167 oracle (per-prompt marginal-cost oracle)

Outputs (platform_hybrid/experiments/results/p5p8/p7_iter175_*):
  per_obs.tsv        (2560 obs: G_C6, dY_C6, fire flags)
  per_summary.tsv    (4 methods x 6 controllers: contrast + cost + extras)
  pareto.tsv         (per-method Pareto with optimal flag)
  bootstrap_ci.tsv   (B=2000 percentile-CI on %oracle-captured)
  sweep.tsv          (5 zvf_tau x 4 methods: C6 Pareto wins + cost)
  summary.json       (verdicts + cross-paper coupling)
"""
from __future__ import annotations
import csv, glob, json, math, os, random
from typing import Dict, List

WORKTREE = "/home/claude/tinker-rl-lab-minimax"
DATA_DIR = os.path.join(WORKTREE, "platform_hybrid/experiments/results/n2_reward_tensor_resume")
OUT_DIR = os.path.join(WORKTREE, "platform_hybrid/experiments/results/p5p8")
os.makedirs(OUT_DIR, exist_ok=True)
G_BASE = 8
G_MENU = [2, 4, 6, 8, 10, 12, 16, 24, 32]
METHODS = ["aero", "areal", "gift", "grpo"]


# ---------- pure-math helpers (identical to iter-167 for direct comparability) ----------

def zvf_iid(p, G):
    if p <= 0.0:
        return 1.0 if G >= 1 else 0.0
    if p >= 1.0:
        return 1.0
    return p ** G + (1.0 - p) ** G

def yield_iid(p, G):
    return 1.0 - zvf_iid(p, G)

def per_prompt_delta_y(p_hat, g_prime):
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 0.0
    return yield_iid(p_hat, g_prime) - yield_iid(p_hat, G_BASE)

def oracle_g_star(p_hat):
    """Marginal cost-effective oracle."""
    if p_hat <= 0.0 or p_hat >= 1.0:
        return G_BASE, 0.0
    best_g, best_score, best_dy = G_BASE, -math.inf, 0.0
    for g in G_MENU:
        if g == G_BASE:
            continue
        dY = per_prompt_delta_y(p_hat, g)
        if dY <= 1e-12:
            continue
        score = dY / max(1, g - G_BASE)
        if score > best_score:
            best_g, best_score, best_dy = g, score, dY
    if best_score <= 1e-12:
        return G_BASE, 0.0
    return best_g, best_dy


# ---------- Berkeley row 01 Dualformer auto-G ----------

def dualformer_g_target(p_hat):
    """Berkeley row 01 auto-G band:
        (0, 0.30]  -> 2
        (0.30, 0.55] -> 8 (base)
        (0.55, 0.70] -> 12
        (0.70, 0.85] -> 16
        (0.85, 1.0) -> 24
    Degenerate p in {0, 1}: G_base.
    """
    if p_hat <= 0.0 or p_hat >= 1.0:
        return G_BASE
    if p_hat <= 0.30:
        return 2
    if p_hat <= 0.55:
        return G_BASE
    if p_hat <= 0.70:
        return 12
    if p_hat <= 0.85:
        return 16
    return 24


# ---------- C6 Calibrated-Hybrid: intersection-and-fallback ----------

def c6_decision(p_hat, step_zvf, zvf_tau=0.70):
    """Return (G_C6, fire_dual_bool, fire_zvf_bool). The decision tree:

    fire_due_to_dualformer = (g_dual != G_base) and (bimodal > 0.05)
    fire_due_to_zvf        = step_zvf >= zvf_tau

    if   fire_due_to_dualformer AND fire_due_to_zvf: return g_dual
    elif fire_due_to_zvf:                            return 16
    else:                                           return G_base   # gamma*=0 anchor
    """
    if p_hat <= 0.0 or p_hat >= 1.0:
        return G_BASE, False, False, G_BASE
    g_dual = dualformer_g_target(p_hat)
    bimodal = 4.0 * p_hat * (1.0 - p_hat)
    fire_due_to_dualformer = (g_dual != G_BASE) and (bimodal > 0.05)
    fire_due_to_zvf = step_zvf >= zvf_tau
    if fire_due_to_dualformer and fire_due_to_zvf:
        return g_dual, fire_due_to_dualformer, fire_due_to_zvf, g_dual
    elif fire_due_to_zvf:
        return 16, fire_due_to_dualformer, fire_due_to_zvf, 16
    else:
        return G_BASE, fire_due_to_dualformer, fire_due_to_zvf, G_BASE


# ---------- existing empirical controllers (verbatim from iter-167) ----------

def ctrl_zvf_triage(step_zvf, tau=0.70):
    return 16 if step_zvf >= tau else G_BASE

def ctrl_hybrid(step_zvf, tau_low=0.70, tau_high=0.90):
    if tau_low <= step_zvf < tau_high:
        return 16
    return G_BASE

def ctrl_iso_g(p_hat, tau_y=0.90):
    if p_hat <= 0.0 or p_hat >= 1.0:
        return G_BASE
    for g in sorted(set(G_MENU)):
        if yield_iid(p_hat, g) >= tau_y:
            return g
    return G_BASE

EMPIRICAL = {
    "C0_fixed_G8":   lambda p_hat, step_zvf: G_BASE,
    "C1_zvf_triage": lambda p_hat, step_zvf: ctrl_zvf_triage(step_zvf),
    "C2_dualformer": lambda p_hat, step_zvf: dualformer_g_target(p_hat),
    "C3_hybrid":     lambda p_hat, step_zvf: ctrl_hybrid(step_zvf),
    "C5_isog_y090":  lambda p_hat, step_zvf: ctrl_iso_g(p_hat, 0.90),
}


# ---------- corpus ----------

def load_corpus():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_s0_tensors.jsonl"))):
        method = os.path.basename(path).split("_")[0]
        if method not in METHODS:
            continue
        with open(path) as fh:
            for ln in fh:
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
                        "step_zvf": step_zvf,
                    })
    return rows


# ---------- per-observation evaluation ----------

def obs_table(corpus, zvf_tau=0.70):
    out = []
    for r in corpus:
        p_hat, zvf_step = r["p_hat"], r["step_zvf"]
        g_star, dY_star = oracle_g_star(p_hat)
        g_c6, fire_d, fire_z, g_dual = c6_decision(p_hat, zvf_step, zvf_tau=zvf_tau)
        dY_c6 = per_prompt_delta_y(p_hat, g_c6)
        dY_dual_target = per_prompt_delta_y(p_hat, g_dual)
        ctrl_g, ctrl_dy = {}, {}
        for cname, fn in EMPIRICAL.items():
            g_c = fn(p_hat, zvf_step)
            ctrl_g[cname] = g_c
            ctrl_dy[cname] = per_prompt_delta_y(p_hat, g_c)
        out.append({
            "method": r["method"], "step": r["step"], "prompt_idx": r["prompt_idx"],
            "k": r["k"], "p_hat": p_hat, "step_zvf": zvf_step,
            "oracle_g": g_star, "oracle_dy": dY_star,
            "g_dualformer_target": g_dual, "dy_dualformer_target": dY_dual_target,
            "fire_due_to_dualformer": int(fire_d),
            "fire_due_to_zvf": int(fire_z),
            "g_c6": g_c6, "dy_c6": dY_c6,
            "g_c0": ctrl_g["C0_fixed_G8"], "dy_c0": ctrl_dy["C0_fixed_G8"],
            "g_c1": ctrl_g["C1_zvf_triage"], "dy_c1": ctrl_dy["C1_zvf_triage"],
            "g_c2": ctrl_g["C2_dualformer"], "dy_c2": ctrl_dy["C2_dualformer"],
            "g_c3": ctrl_g["C3_hybrid"], "dy_c3": ctrl_dy["C3_hybrid"],
            "g_c5": ctrl_g["C5_isog_y090"], "dy_c5": ctrl_dy["C5_isog_y090"],
        })
    return out


# ---------- aggregation ----------

def per_method_summary(obs):
    rows = []
    by_method = {}
    for o in obs:
        by_method.setdefault(o["method"], []).append(o)
    cn_order = ["C0_fixed_G8", "C1_zvf_triage", "C2_dualformer",
                "C3_hybrid", "C5_isog_y090", "C6_calibrated"]
    cn_dy_key = {"C6_calibrated": "dy_c6",
                 "C0_fixed_G8": "dy_c0", "C1_zvf_triage": "dy_c1",
                 "C2_dualformer": "dy_c2", "C3_hybrid": "dy_c3",
                 "C5_isog_y090": "dy_c5"}
    cn_g_key = {"C6_calibrated": "g_c6",
                "C0_fixed_G8": "g_c0", "C1_zvf_triage": "g_c1",
                "C2_dualformer": "g_c2", "C3_hybrid": "g_c3",
                "C5_isog_y090": "g_c5"}
    for m, lst in by_method.items():
        oracle_total = sum(o["oracle_dy"] for o in lst)
        for cn in cn_order:
            dy_key, g_key = cn_dy_key[cn], cn_g_key[cn]
            ctrl_dy = sum(o[dy_key] for o in lst)
            extras = sum(o[g_key] - G_BASE for o in lst)
            pct = (ctrl_dy / oracle_total * 100.0) if oracle_total > 1e-12 else 0.0
            rows.append({
                "method": m,
                "controller": cn,
                "oracle_total_abs_dy": oracle_total,
                "controller_total_dy": ctrl_dy,
                "ctrl_total_extras": extras,
                "pct_oracle_captured": pct,
                "n_obs": len(lst),
            })
    return rows


def pareto_per_method(per_method_rows):
    out = []
    by_method = {}
    for r in per_method_rows:
        by_method.setdefault(r["method"], []).append(r)
    pareto_by_method = {}
    for m, lst in by_method.items():
        pareto_set = []
        for ri in lst:
            dominated = False
            for rj in lst:
                if ri is rj:
                    continue
                if (rj["ctrl_total_extras"] <= ri["ctrl_total_extras"]
                        and rj["controller_total_dy"] >= ri["controller_total_dy"]
                        and (rj["ctrl_total_extras"] < ri["ctrl_total_extras"]
                              or rj["controller_total_dy"] > ri["controller_total_dy"])):
                    dominated = True
                    break
            out.append({**ri, "pareto_optimal": int(not dominated)})
            if not dominated:
                pareto_set.append(ri["controller"])
        pareto_by_method[m] = pareto_set
    return out, pareto_by_method


def bootstrap_pct_ci(obs, cn, B=2000, seed=20260705):
    rng = random.Random(seed)
    cn_dy = {"C6_calibrated": "dy_c6",
             "C0_fixed_G8": "dy_c0", "C1_zvf_triage": "dy_c1",
             "C2_dualformer": "dy_c2", "C3_hybrid": "dy_c3",
             "C5_isog_y090": "dy_c5"}
    by_method = {}
    for o in obs:
        by_method.setdefault(o["method"], []).append(o)
    rows = []
    for m, lst in by_method.items():
        oracle_total = sum(o["oracle_dy"] for o in lst)
        n = len(lst)
        if oracle_total < 1e-12:
            rows.append({"method": m, "controller": cn,
                         "pct_point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "B": B})
            continue
        dy_arr = [o[cn_dy[cn]] for o in lst]
        oracle_arr = [o["oracle_dy"] for o in lst]
        boots = []
        for _ in range(B):
            idxs = [rng.randrange(n) for _ in range(n)]
            ctrl_dy = sum(dy_arr[i] for i in idxs)
            oracle_s = sum(oracle_arr[i] for i in idxs)
            pct = (ctrl_dy / oracle_s * 100.0) if oracle_s > 1e-12 else 0.0
            boots.append(pct)
        boots.sort()
        pt = sum(dy_arr) / oracle_total * 100.0
        rows.append({"method": m, "controller": cn,
                     "pct_point": pt,
                     "ci_lo": boots[int(0.025 * B)],
                     "ci_hi": boots[int(0.975 * B)],
                     "B": B})
    return rows


# ---------- main ----------

def main(B=2000, seed=20260705):
    print(f"[iter175 v2] intersection C6 with zvf_tau=0.70 default, B={B}")
    corpus = load_corpus()
    print(f"[iter175 v2] corpus size = {len(corpus)} obs")
    obs = obs_table(corpus, zvf_tau=0.70)
    pobs_path = os.path.join(OUT_DIR, "p7_iter175_per_obs.tsv")
    with open(pobs_path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(obs[0].keys()), delimiter="\t")
        w.writeheader()
        for o in obs:
            w.writerow(o)
    print(f"[iter175 v2] wrote {pobs_path} ({len(obs)} rows)")
    per_m = per_method_summary(obs)
    pms = os.path.join(OUT_DIR, "p7_iter175_per_summary.tsv")
    with open(pms, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_m[0].keys()), delimiter="\t")
        w.writeheader()
        for r in per_m:
            w.writerow(r)
    print(f"[iter175 v2] wrote {pms}")
    pareto_rows, pareto_by_method = pareto_per_method(per_m)
    pareto_path = os.path.join(OUT_DIR, "p7_iter175_pareto.tsv")
    with open(pareto_path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pareto_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in pareto_rows:
            w.writerow(r)
    print(f"[iter175 v2] wrote {pareto_path} (Pareto by method: {pareto_by_method})")
    cn_to_test = ["C2_dualformer", "C3_hybrid", "C5_isog_y090", "C6_calibrated"]
    boot_rows = []
    for cn in cn_to_test:
        for r in bootstrap_pct_ci(obs, cn, B=B, seed=seed):
            boot_rows.append(r)
    bc_path = os.path.join(OUT_DIR, "p7_iter175_bootstrap_ci.tsv")
    with open(bc_path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(boot_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in boot_rows:
            w.writerow(r)
    print(f"[iter175 v2] wrote {bc_path} ({len(boot_rows)} rows)")
    # Sweep over zvf_tau in {0.55, 0.65, 0.70, 0.75, 0.85}
    sweeps = []
    for zt in [0.55, 0.65, 0.70, 0.75, 0.85]:
        sw_obs = obs_table(corpus, zvf_tau=zt)
        sw_summary = per_method_summary(sw_obs)
        sw_pareto, sw_pareto_by_method = pareto_per_method(sw_summary)
        c6_wins = sum(1 for r in sw_pareto if r["controller"] == "C6_calibrated" and r["pareto_optimal"] == 1)
        for m in METHODS:
            r6 = next(r for r in sw_summary if r["controller"] == "C6_calibrated" and r["method"] == m)
            sweeps.append({
                "zvf_tau": zt, "method": m,
                "c6_pct_oracle_captured": r6["pct_oracle_captured"],
                "c6_total_extras": r6["ctrl_total_extras"],
                "c6_total_dy": r6["controller_total_dy"],
                "c6_pareto_wins_total": c6_wins,
                "pareto_set": ";".join(sorted(set(sw_pareto_by_method.get(m, [])))),
            })
    sweep_path = os.path.join(OUT_DIR, "p7_iter175_sweep.tsv")
    with open(sweep_path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweeps[0].keys()), delimiter="\t")
        w.writeheader()
        for r in sweeps:
            w.writerow(r)
    print(f"[iter175 v2] wrote {sweep_path} ({len(sweeps)} rows)")
    summary = {
        "n_obs": len(obs),
        "zvf_tau_default": 0.70,
        "per_method_c6_pct": {r["method"]: r["pct_oracle_captured"]
                                for r in per_m if r["controller"] == "C6_calibrated"},
        "per_method_c6_extras": {r["method"]: r["ctrl_total_extras"]
                                   for r in per_m if r["controller"] == "C6_calibrated"},
        "per_method_c2_pct": {r["method"]: r["pct_oracle_captured"]
                                for r in per_m if r["controller"] == "C2_dualformer"},
        "per_method_c1_pct": {r["method"]: r["pct_oracle_captured"]
                                for r in per_m if r["controller"] == "C1_zvf_triage"},
        "per_method_c5_pct": {r["method"]: r["pct_oracle_captured"]
                                for r in per_m if r["controller"] == "C5_isog_y090"},
        "pareto_by_method": pareto_by_method,
        "n_c6_pareto_wins_default": sum(1 for r in pareto_rows if r["controller"] == "C6_calibrated" and r["pareto_optimal"] == 1),
        "sweep_best_zt": max({s["zvf_tau"] for s in sweeps}, key=lambda z: sum(s["c6_pct_oracle_captured"] for s in sweeps if s["zvf_tau"] == z)),
        "sweep_summary_per_zt": {z: [s for s in sweeps if s["zvf_tau"] == z] for z in [0.55, 0.65, 0.70, 0.75, 0.85]},
    }
    summary_path = os.path.join(OUT_DIR, "p7_iter175_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"[iter175 v2] wrote {summary_path}")
    return summary


if __name__ == "__main__":
    main()
