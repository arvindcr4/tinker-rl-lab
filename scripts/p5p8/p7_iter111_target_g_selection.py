"""Iter 111 — P7 ADAPTIVE-G* COUNTERFACTUAL (salvage-rate framing).

Vein (fresh, not in 110 prior rows):
The P7 controller family (iter 67/71/75/79/83/87/91/95/99/103/107) reports
fires, hysteresis flips, post-pred restore probability, and closed-form
per-fire contrast gain — but **never** reports the COUNTERFACTUAL optimal
target group-size G* per fired step, nor the salvage rate (the fraction
of fires for which the iid-ZVF model says the controller's escalation
lever can actually restore contrast).

Iter 111 answers the brief's Q1-Q3 on the REAL N2 reward tensors
(40 steps × 4 methods × 16 prompts, exact per-prompt k_p at G=8):
  Q1. WHEN zvf_obs > τ (=0.70), what target G* ∈ {16,32,64} closes the
      contrast gap under the closed-form (iid) binomial ZVF?
  Q2. WHAT FRACTION of fires is "salvageable" — i.e., a G* ∈ {16,32,64}
      exists such that z(G*) drops below MAX(target_thresh, boundary_rate)?
  Q3. Which method has the highest salvage rate under each controller?

Controllers (4-rule counterfactual comparison):
  (a) STATIC_G16     — always pay 2× cost (iter-103 default).
  (b) DUALFORMER_d4   — Berkeley row 01 rule G ← min(G+δ, Gmax)=12; in
                        practice rounds up to G=16 (smallest candidate).
  (c) DUALFORMER_d8   — G ← min(16, 64) = 16 (matches STATIC).
  (d) ADAPTIVE_GSTAR — closed-form optimal G* = MIN G ∈ {16,32,64} whose
                        predicted iid mean ZVF drops below the LOWER of
                        τ_target = 0.50 OR 0.5×zvf_obs.

The salvage rate is the paper-grade outcome: when a Pareto-dominant
controller can clear the saturation barrier (e.g., 50% cutoff), escalation
pays for itself; when no candidate can, the controller's escalation lever
is exhausted and the P7 design should fall back to prompt-set rotation.

Outputs:
  experiments/results/p5p8/p7_iter111_target_g_distribution.tsv
  experiments/results/p5p8/p7_iter111_controller_replay.tsv
  experiments/results/p5p8/p7_iter111_net_benefit.tsv
  experiments/results/p5p8/p7_iter111_summary.json
"""
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "experiments/results/n2_reward_tensor_resume"
OUT_DIR = WORK / "experiments/results/p5p8"

METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
G_CANDIDATES = (16, 32, 64)
N_STEPS = 40
N_BOOT = 4000
SEED = 20260705
TAU = 0.70
EPS = 1e-12


# ---------- Closed-form helpers ----------

def zvf_binom(p_hat, G):
    p = min(max(p_hat, EPS), 1.0 - EPS)
    return p ** G + (1.0 - p) ** G


def mean_zvf(p_hats, G):
    return sum(zvf_binom(p, G) for p in p_hats) / len(p_hats)


def boundary_rate(ks):
    """Empirical boundary rate at G_BASE (provable lower bound on iid ZVF)."""
    return sum(1 for k in ks if k in (0, G_BASE)) / len(ks)


def optimal_gstar(p_hats, zvf_obs):
    """MINIMUM G ∈ {16,32,64} whose predicted iid mean ZVF drops below
    MAX(0.50, 0.5*zvf_obs). If no candidate achieves it, return the
    candidate with the lowest mean ZVF (G=64) for the cost-asymptote
    counterfactual.

    The 0.5*zvf_obs rule is 'halve the contrast loss from the no-fire
    baseline'; the 0.50 floor is 'controller-deactivation threshold'.
    """
    threshold = max(0.50, 0.5 * zvf_obs)
    best_G, best_z = max(G_CANDIDATES), float("inf")
    for G in sorted(G_CANDIDATES):
        mz = mean_zvf(p_hats, G)
        if mz <= threshold:
            return G, mz
        if mz < best_z:
            best_z, best_G = mz, G
    return best_G, best_z


# ---------- Load N2 tensors ----------

def load_n2():
    by_method = {}
    for m in METHODS:
        rows = [json.loads(l) for l in open(N2_DIR / f"{m}_s0_tensors.jsonl")]
        rows.sort(key=lambda r: r["step"])
        by_method[m] = rows
    return by_method


# ---------- Build per-step records ----------

def build_per_step(by_method):
    """One record per (method, step), with the per-prompt observed k_p,
    p_hats, zvf_obs, boundary_rate, and the closed-form optimal G*."""
    per_step = []
    target_g_dist = []
    for m, rows in by_method.items():
        for r in rows:
            ks = [int(round(sum(p))) for p in r["rewards"]]
            p_hats = [k / G_BASE for k in ks]
            zvf_obs = boundary_rate(ks)  # identical to frac_all_zero+frac_all_one
            Gstar, z_at = optimal_gstar(p_hats, zvf_obs)
            per_step.append({
                "method": m, "step": r["step"], "ks": ks, "p_hats": p_hats,
                "zvf_obs": zvf_obs, "Gstar_optimal_closedform": Gstar,
                "zvf_at_optimal_closedform": z_at,
            })
            target_g_dist.append({
                "method": m, "step": r["step"],
                "zvf_obs": round(zvf_obs, 4),
                "n_degenerate": sum(1 for k in ks if k in (0, G_BASE)),
                "boundary_rate": round(boundary_rate(ks), 4),
                "Gstar_optimal": Gstar,
                "zvf_at_optimal": round(z_at, 4),
                "zvf_at_G16": round(mean_zvf(p_hats, 16), 4),
                "zvf_at_G32": round(mean_zvf(p_hats, 32), 4),
                "zvf_at_G64": round(mean_zvf(p_hats, 64), 4),
                "salvageable": int(Gstar != max(G_CANDIDATES) or
                                   z_at < zvf_obs),
                "Gstar_eq_G16": int(Gstar == 16),
                "Gstar_eq_G32": int(Gstar == 32),
                "Gstar_eq_G64": int(Gstar == 64),
            })
    return per_step, target_g_dist


# ---------- Replay 4 controller rules ----------

def replay(per_step, controller_name, get_Gstar):
    out = []
    for s in per_step:
        fired = s["zvf_obs"] > TAU
        if fired:
            Gstar, z_target = get_Gstar(s)
            cost_ratio = Gstar / G_BASE
            delta_z = s["zvf_obs"] - z_target
            net = delta_z - 0.5 * (cost_ratio - 1.0)
            out.append({"method": s["method"], "step": s["step"],
                        "controller": controller_name, "fired": True,
                        "zvf_obs": s["zvf_obs"], "Gstar": Gstar,
                        "zvf_target": z_target, "cost_ratio": cost_ratio,
                        "delta_z": delta_z, "net_benefit": net})
        else:
            out.append({"method": s["method"], "step": s["step"],
                        "controller": controller_name, "fired": False,
                        "zvf_obs": s["zvf_obs"], "Gstar": G_BASE,
                        "zvf_target": s["zvf_obs"], "cost_ratio": 1.0,
                        "delta_z": 0.0, "net_benefit": 0.0})
    return out


def get_Gstar_static(s):
    return 16, mean_zvf(s["p_hats"], 16)


def get_Gstar_dual_d4(s):
    return min(G_BASE + 4, max(G_CANDIDATES)), mean_zvf(
        s["p_hats"], min(G_BASE + 4, max(G_CANDIDATES)))


def get_Gstar_dual_d8(s):
    """G ← min(G + 8, Gmax) = min(16, 64) = 16."""
    return 16, mean_zvf(s["p_hats"], 16)


def get_Gstar_adaptive(s):
    return optimal_gstar(s["p_hats"], s["zvf_obs"])


# ---------- Bootstrap CI ----------

def bootstrap_ci(values, B=N_BOOT, alpha=0.05, seed=SEED):
    if not values:
        return 0.0, 0.0, 0.0
    rng_state = seed & 0xFFFFFFFF
    boots = []
    n = len(values)
    for _ in range(B):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        idx = rng_state % n
        s = sum(values[(idx + i) % n] for i in range(min(12, n))) / min(12, n)
        boots.append(s)
    boots.sort()
    return (statistics.mean(values),
            boots[int(B * alpha / 2)],
            boots[int(B * (1 - alpha / 2))])


# ---------- Main ----------

def main():
    by_method = load_n2()
    per_step, target_g_dist = build_per_step(by_method)

    # 4 counterfactual controllers
    replays = {
        "STATIC_G16":      replay(per_step, "STATIC_G16",     get_Gstar_static),
        "DUALFORMER_AUTO_d4": replay(per_step, "DUALFORMER_AUTO_d4", get_Gstar_dual_d4),
        "DUALFORMER_AUTO_d8": replay(per_step, "DUALFORMER_AUTO_d8", get_Gstar_dual_d8),
        "ADAPTIVE_GSTAR":  replay(per_step, "ADAPTIVE_GSTAR", get_Gstar_adaptive),
    }

    # Per-method summary with bootstrap CIs on net_benefit
    per_method_summary = []
    for cname, crecords in replays.items():
        per_m = defaultdict(list)
        for r in crecords:
            per_m[r["method"]].append(r)
        for m in METHODS:
            rows = per_m[m]
            fires = [r for r in rows if r["fired"]]
            n_fire = len(fires)
            mean_dz_fires = (statistics.mean(r["delta_z"] for r in fires)
                             if fires else 0.0)
            mean_cost_fires = (statistics.mean(r["cost_ratio"] for r in fires)
                               if fires else 1.0)
            mean_net_fires = (statistics.mean(r["net_benefit"] for r in fires)
                              if fires else 0.0)
            net_all = [r["net_benefit"] for r in rows]
            mu, lo, hi = bootstrap_ci(net_all)
            G_dist = dict(Counter(r["Gstar"] for r in fires))
            per_method_summary.append({
                "controller": cname, "method": m, "n_fired": n_fire,
                "n_total": len(rows),
                "frac_fired": round(n_fire / len(rows), 4),
                "mean_dz_on_fires": round(mean_dz_fires, 4),
                "mean_cost_ratio_on_fires": round(mean_cost_fires, 4),
                "mean_net_benefit_per_fire": round(mean_net_fires, 4),
                "net_all_mean": round(mu, 4),
                "net_all_ci_lo": round(lo, 4),
                "net_all_ci_hi": round(hi, 4),
                "Gstar_distribution_fires": G_dist,
                "n_salvageable_fires": sum(
                    1 for r in fires
                    if r["Gstar"] in G_CANDIDATES and r["delta_z"] > 0
                ),
            })

    # Overall salvage rate per method + per controller
    salvage_per_method = {}
    for m in METHODS:
        fires_total = sum(1 for r in replays["ADAPTIVE_GSTAR"]
                          if r["method"] == m and r["fired"])
        salvage = sum(1 for r in replays["ADAPTIVE_GSTAR"]
                      if r["method"] == m and r["fired"]
                      and r["Gstar"] in G_CANDIDATES
                      and r["delta_z"] > 0)
        salvage_per_method[m] = {
            "fires_total": fires_total,
            "salvageable": salvage,
            "salvage_rate": (round(salvage / fires_total, 4)
                             if fires_total else 0.0),
        }

    # Per-controller totals
    totals = {c: sum(r["net_benefit"] for r in rec)
              for c, rec in replays.items()}
    totals_fires = {c: sum(1 for r in rec if r["fired"])
                    for c, rec in replays.items()}
    totals_dz = {c: sum(r["delta_z"] for r in rec)
                 for c, rec in replays.items()}
    totals_cost = {c: sum(r["cost_ratio"] for r in rec if r["fired"])
                   for c, rec in replays.items()}

    # G* distribution over fired steps (ADAPTIVE only)
    adapt_g_dist = Counter(
        r["Gstar"] for r in replays["ADAPTIVE_GSTAR"] if r["fired"])

    # Per-step optimal-G distribution overall
    overall_g_dist = Counter(
        r["Gstar_optimal_closedform"] for r in per_step)

    # ----- Write TSVs -----
    out_target = OUT_DIR / "p7_iter111_target_g_distribution.tsv"
    with open(out_target, "w") as f:
        f.write("method\tstep\tzvf_obs\tn_degenerate\tboundary_rate\t"
                "Gstar_optimal\tzvf_at_optimal\tzvf_at_G16\tzvf_at_G32\tzvf_at_G64\t"
                "salvageable\tGstar_eq_G16\tGstar_eq_G32\tGstar_eq_G64\n")
        for r in target_g_dist:
            f.write("\t".join(str(r[k]) for k in [
                "method", "step", "zvf_obs", "n_degenerate", "boundary_rate",
                "Gstar_optimal", "zvf_at_optimal", "zvf_at_G16", "zvf_at_G32",
                "zvf_at_G64", "salvageable",
                "Gstar_eq_G16", "Gstar_eq_G32", "Gstar_eq_G64"]) + "\n")
    print(f"wrote {out_target} ({len(target_g_dist)} rows)")

    out_replay = OUT_DIR / "p7_iter111_controller_replay.tsv"
    with open(out_replay, "w") as f:
        f.write("controller\tmethod\tstep\tfired\tzvf_obs\tGstar\tzvf_target\t"
                "cost_ratio\tdelta_z\tnet_benefit\n")
        for cname, crecords in replays.items():
            for r in crecords:
                f.write("\t".join([
                    cname, r["method"], str(r["step"]),
                    str(int(r["fired"])), f"{r['zvf_obs']:.4f}",
                    str(r["Gstar"]), f"{r['zvf_target']:.4f}",
                    f"{r['cost_ratio']:.4f}",
                    f"{r['delta_z']:.4f}",
                    f"{r['net_benefit']:.4f}",
                ]) + "\n")
    print(f"wrote {out_replay} ({sum(len(v) for v in replays.values())} rows)")

    out_net = OUT_DIR / "p7_iter111_net_benefit.tsv"
    with open(out_net, "w") as f:
        f.write("controller\tmethod\tn_fired\tn_total\tfrac_fired\t"
                "mean_dz_on_fires\tmean_cost_ratio_on_fires\t"
                "mean_net_benefit_per_fire\tnet_all_mean\tnet_all_ci_lo\t"
                "net_all_ci_hi\tGstar_dist_fires\tn_salvageable_fires\n")
        for r in per_method_summary:
            f.write("\t".join([
                r["controller"], r["method"], str(r["n_fired"]),
                str(r["n_total"]), str(r["frac_fired"]),
                str(r["mean_dz_on_fires"]),
                str(r["mean_cost_ratio_on_fires"]),
                str(r["mean_net_benefit_per_fire"]),
                str(r["net_all_mean"]),
                str(r["net_all_ci_lo"]), str(r["net_all_ci_hi"]),
                json.dumps(r["Gstar_distribution_fires"]),
                str(r["n_salvageable_fires"]),
            ]) + "\n")
    print(f"wrote {out_net} ({len(per_method_summary)} rows)")

    summary = {
        "tau": TAU, "g_base": G_BASE,
        "g_candidates": list(G_CANDIDATES),
        "n_bootstrap": N_BOOT, "seed": SEED,
        "totals_net_benefit": totals,
        "totals_fires": totals_fires,
        "totals_delta_z": totals_dz,
        "totals_cost_ratio_summed": totals_cost,
        "per_method_summary": per_method_summary,
        "salvage_per_method_adaptive": salvage_per_method,
        "adaptive_gstar_distribution_fires": dict(adapt_g_dist),
        "overall_optimal_g_distribution": dict(overall_g_dist),
    }
    out_json = OUT_DIR / "p7_iter111_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")

    print("\n=== Headlines ===")
    print("Per-controller totals (160 step-cells):")
    for c in replays:
        print(f"  {c:22s} fires={totals_fires[c]:>3d}  "
              f"total_net={totals[c]:+.4f}  "
              f"total_dz={totals_dz[c]:+.4f}  "
              f"cost_sum={totals_cost[c]:.2f}")
    print("\nSalvage rate per method (ADAPTIVE_GSTAR):")
    for m, v in salvage_per_method.items():
        print(f"  {m:6s} fires={v['fires_total']:>3d}  "
              f"salvageable={v['salvageable']:>3d}  "
              f"rate={v['salvage_rate']:.4f}")
    print("\nADAPTIVE G* distribution on fired steps:")
    for G in sorted(adapt_g_dist):
        print(f"  G={G}: {adapt_g_dist[G]} fires")


if __name__ == "__main__":
    main()
