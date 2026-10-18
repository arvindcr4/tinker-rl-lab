"""
P7 τ seed-robustness (DECISION-STABILITY) on N10 panel — iter 63.

Vein: brief vein (c) — seed-robustness of trigger threshold, sharpened
beyond the existing sec:p7-controller-seedrobust (which reports only
fires/seed mean ± CI). This script adds:
  (1) cross-seed DECISION AGREEMENT (Jaccard over fire sets, plus
      per-step full agreement);
  (2) SAVINGS analysis (rollouts vs always-G=8) — the iter-51 controller
      is escalation-only on N10 (no zvf≥0.95), so we add a
      TWO-THRESHOLD extension (tau_esc, tau_des) parameterised
      symmetrically and identify the operating window where savings > 0
      and cross-seed agreement is high;
  (3) SATURATED-FIRE counter (steps with zvf==1.0 that get escalated
      — these are the iter-59 "operationally inert" branch).

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_tau_seed_stability.tsv
  platform_hybrid/experiments/results/p5p8/p7_tau_two_threshold_sweep.tsv
  platform_hybrid/experiments/results/p5p8/p7_tau_seed_stability_summary.json
"""
import json, math, itertools, random, statistics
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # platform_modal/scripts, for _paths
from _paths import REPO_ROOT
WORKTREE = REPO_ROOT
N10_DIR = WORKTREE / "platform_hybrid/experiments/results/n10_seed_expansion"
OUT = WORKTREE / "platform_hybrid/experiments/results/p5p8/corrected-review20-v2"

SEEDS = [42, 179, 316, 453, 590]
G_BASELINE = 8
G_ESC = 16
G_DES = 4

def load_seed(seed):
    p = N10_DIR / f"n10_grpo_s{seed}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    zvf_seq = [s["zvf"] for s in d.get("step_log", [])]
    return {
        "seed": seed,
        "zvf": zvf_seq,
        "heldout_acc": d.get("heldout_acc"),
        "mean_zvf": d.get("mean_zvf"),
        "last10_zvf": sum(zvf_seq[-10:]) / max(1, min(10, len(zvf_seq))),
    }

def apply_controller(zvf_seq, tau_esc, tau_des=0.95, g_esc=G_ESC, g_des=G_DES, g_base=G_BASELINE):
    """Two-threshold controller: tau_esc=escalate above; tau_des=de-escalate above."""
    out = []
    for z in zvf_seq:
        if z >= tau_des:
            out.append(g_des)
        elif z >= tau_esc:
            out.append(g_esc)
        else:
            out.append(g_base)
    return out

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    u = sa | sb
    if not u: return 1.0
    return len(sa & sb) / len(u)

def bootstrap_ci(values, B=10000, seed=59001, alpha=0.05):
    if not values: return (0.0, 0.0, 0.0)
    n = len(values)
    rng = random.Random(seed)
    means = []
    for _ in range(B):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (sum(values)/n, means[int(B*alpha/2)], means[int(B*(1-alpha/2))])

def pearson(x, y):
    n = len(x)
    if n < 2: return 0.0
    mx, my = sum(x)/n, sum(y)/n
    sx = math.sqrt(sum((xi-mx)**2 for xi in x))
    sy = math.sqrt(sum((yi-my)**2 for yi in y))
    if sx == 0 or sy == 0: return 0.0
    return sum((xi-mx)*(yi-my) for xi,yi in zip(x,y)) / (sx*sy)


def kendall_tau(a, b):
    """Kendall tau-b, excluding joint ties from both denominator factors.

    Return None when either trajectory has no comparable pairs. Such a
    trajectory has undefined rank correlation, not zero correlation.
    """
    if len(a) != len(b):
        raise ValueError("Kendall trajectories must have equal lengths")
    if not all(math.isfinite(v) for v in itertools.chain(a, b)):
        raise ValueError("Kendall trajectories must be finite")
    concord = discord = ties_a = ties_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da * db > 0:
                concord += 1
            else:
                discord += 1
    comparable = concord + discord
    denom = math.sqrt((comparable + ties_a) * (comparable + ties_b))
    return (concord - discord) / denom if denom else None


def rollout_accounting(g_choices, g_base=G_BASELINE):
    """Savings against the same number of constant-G baseline decisions."""
    actual = sum(g_choices)
    baseline = g_base * len(g_choices)
    if baseline <= 0:
        raise ValueError("Rollout accounting requires a positive baseline total")
    return {
        "baseline_rollouts": baseline,
        "actual_rollouts": actual,
        "saved_rollouts": baseline - actual,
        "net_savings_pct": (baseline - actual) / baseline * 100.0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = [s for s in (load_seed(s) for s in SEEDS) if s is not None]
    n_seeds = len(panel)
    n_steps = min(len(s["zvf"]) for s in panel)
    for s in panel:
        s["zvf"] = s["zvf"][:n_steps]
    print(f"[p7] loaded {n_seeds} seeds × {n_steps} steps")
    print(f"[p7] zvf ranges per seed: " + ", ".join(
        f"s{s['seed']}:[{min(s['zvf']):.2f},{max(s['zvf']):.2f}]" for s in panel
    ))
    # zvf_max across all seeds
    zvf_max_global = max(max(s["zvf"]) for s in panel)
    print(f"[p7] global zvf max across seeds: {zvf_max_global}")
    print(f"[p7] saturated steps (zvf=1.0): "
          f"{sum(z == 1.0 for s in panel for z in s['zvf'])}")

    # ---------- Analysis 1: single-τ sweep, τ_esc only (matches paper sec) ----------
    TAUS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    rows_single = []
    for tau in TAUS:
        per_seed_fires, per_seed_savings, jaccards, step_full_agree = [], [], [], []
        per_seed_wrong, per_seed_sat = [], []
        g_choice_per_seed = []
        for s in panel:
            g_choices = apply_controller(s["zvf"], tau_esc=tau, tau_des=1.01)  # disable de-esc
            fires = [i for i, g in enumerate(g_choices) if g != G_BASELINE]
            per_seed_fires.append(len(fires))
            sav = rollout_accounting(g_choices)["net_savings_pct"]
            per_seed_savings.append(sav)
            per_seed_wrong.append(sum(1 for i in range(n_steps) if s["zvf"][i] > 0.99 and g_choices[i] == G_ESC))
            per_seed_sat.append(sum(1 for i in range(n_steps) if s["zvf"][i] == 1.0 and g_choices[i] == G_ESC))
            g_choice_per_seed.append(g_choices)
        for a, b in itertools.combinations(range(n_seeds), 2):
            jaccards.append(jaccard(
                [i for i, g in enumerate(g_choice_per_seed[a]) if g != G_BASELINE],
                [i for i, g in enumerate(g_choice_per_seed[b]) if g != G_BASELINE],
            ))
        for i in range(n_steps):
            choices = [g_choice_per_seed[k][i] for k in range(n_seeds)]
            step_full_agree.append(1 if len(set(choices)) == 1 else 0)
        fire_boot = bootstrap_ci(per_seed_fires, B=10000)
        sav_boot = bootstrap_ci(per_seed_savings, B=10000)
        cv = statistics.pstdev(per_seed_fires) / statistics.mean(per_seed_fires) if statistics.mean(per_seed_fires) > 0 else 0.0
        rows_single.append({
            "tau_esc": tau,
            "tau_des": 1.01,
            "n_seeds": n_seeds,
            "fires_per_seed_mean": round(statistics.mean(per_seed_fires), 3),
            "fires_per_seed_std": round(statistics.pstdev(per_seed_fires), 3),
            "fires_per_seed_cv": round(cv, 3),
            "fires_per_seed_ci_lo": round(fire_boot[1], 3),
            "fires_per_seed_ci_hi": round(fire_boot[2], 3),
            "savings_pct_mean": round(statistics.mean(per_seed_savings), 3),
            "savings_pct_ci_lo": round(sav_boot[1], 3),
            "savings_pct_ci_hi": round(sav_boot[2], 3),
            "fire_set_jaccard_mean": round(statistics.mean(jaccards), 3),
            "fire_set_jaccard_min": round(min(jaccards), 3),
            "per_step_full_agreement": round(statistics.mean(step_full_agree), 3),
            "wrong_fires_total": int(sum(per_seed_wrong)),
            "saturated_fires_total": int(sum(per_seed_sat)),
        })

    # Save single-τ sweep
    tsv_path = OUT / "p7_tau_seed_stability.tsv"
    cols = list(rows_single[0].keys())
    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows_single:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\n[saved] {tsv_path}")

    # ---------- Analysis 2: two-threshold sweep (τ_esc, τ_des) grid ----------
    print(f"\n[p7] two-threshold grid sweep (tau_esc, tau_des):")
    grid = []
    for tau_esc in [0.50, 0.60, 0.70, 0.80, 0.85]:
        for tau_des in [0.80, 0.85, 0.90]:
            row = {"tau_esc": tau_esc, "tau_des": tau_des, "de_esc_active": False}
            des_fires_total = 0
            esc_fires_total = 0
            savings_pct = []
            all_choices = []
            for s in panel:
                g_choices = apply_controller(s["zvf"], tau_esc=tau_esc, tau_des=tau_des)
                all_choices.extend(g_choices)
                des_fires_total += sum(1 for g in g_choices if g == G_DES)
                esc_fires_total += sum(1 for g in g_choices if g == G_ESC)
                sav = rollout_accounting(g_choices)["net_savings_pct"]
                savings_pct.append(sav)
            row["de_esc_fires_total"] = des_fires_total
            row["esc_fires_total"] = esc_fires_total
            row["de_esc_active"] = des_fires_total > 0
            row["savings_pct_mean"] = round(statistics.mean(savings_pct), 3)
            row["savings_pct_ci_lo"] = round(bootstrap_ci(savings_pct, B=10000)[1], 3)
            row["savings_pct_ci_hi"] = round(bootstrap_ci(savings_pct, B=10000)[2], 3)
            row.update(rollout_accounting(all_choices))
            row["net_savings_pct"] = round(row["net_savings_pct"], 3)
            grid.append(row)
            print(f"  (esc={tau_esc}, des={tau_des}) esc_fires={esc_fires_total} "
                  f"des_fires={des_fires_total} savings={row['savings_pct_mean']:+.2f}%")
    # save
    grid_path = OUT / "p7_tau_two_threshold_sweep.tsv"
    cols2 = list(grid[0].keys())
    with open(grid_path, "w") as f:
        f.write("\t".join(cols2) + "\n")
        for r in grid:
            f.write("\t".join(str(r[c]) for c in cols2) + "\n")
    print(f"[saved] {grid_path}")

    # ---------- Analysis 3: cross-seed Kendall-τ rank correlation of ZVF trajectory ----------
    # Are the per-step zvf trajectories themselves in the same order across seeds?
    # pairwise kendall between zvf trajectories (already truncated to n_steps)
    kendall_pairwise = []
    for a, b in itertools.combinations(range(n_seeds), 2):
        kendall_pairwise.append(kendall_tau(panel[a]["zvf"], panel[b]["zvf"]))
    defined_tau = [tau for tau in kendall_pairwise if tau is not None]
    tau_summary = {
        "variant": "tau-b",
        "mean": round(statistics.mean(defined_tau), 3) if defined_tau else None,
        "min": round(min(defined_tau), 3) if defined_tau else None,
        "max": round(max(defined_tau), 3) if defined_tau else None,
        "n_defined_pairs": len(defined_tau),
        "n_undefined_pairs": len(kendall_pairwise) - len(defined_tau),
    }
    print(f"\n[p7] cross-seed Kendall tau-b of per-step zvf trajectories: {tau_summary}")

    # ---------- Analysis 4: per-seed heldout vs ZVF trajectories ----------
    heldout = [s["heldout_acc"] for s in panel]
    mean_zvf = [s["mean_zvf"] for s in panel]
    last10_zvf = [s["last10_zvf"] for s in panel]
    r1 = pearson(heldout, mean_zvf)
    r2 = pearson(heldout, last10_zvf)
    # bootstrap CIs
    rng = random.Random(59001)
    r1b, r2b = [], []
    for _ in range(10000):
        idx = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        r1b.append(pearson([heldout[i] for i in idx], [mean_zvf[i] for i in idx]))
        r2b.append(pearson([heldout[i] for i in idx], [last10_zvf[i] for i in idx]))
    r1b.sort(); r2b.sort()
    r1_ci = [round(r1b[250], 3), round(r1b[9749], 3)]
    r2_ci = [round(r2b[250], 3), round(r2b[9749], 3)]
    print(f"[p7] pearson(heldout, mean_zvf) = {r1:.3f} CI95={r1_ci}")
    print(f"[p7] pearson(heldout, last10_zvf) = {r2:.3f} CI95={r2_ci}")

    # ---------- Summary JSON ----------
    summary = {
        "n_seeds": n_seeds,
        "n_steps_per_seed": n_steps,
        "seeds": [s["seed"] for s in panel],
        "global_zvf_max": zvf_max_global,
        "interpretation": (
            "The single-threshold sweep disables de-escalation. The two-threshold "
            "grid reports active de-escalation from observed action counts. "
            "Savings are baseline minus actual rollouts; negative values mean extra cost."
        ),
        "single_tau_sweep": [
            {
                "tau_esc": r["tau_esc"],
                "fires_per_seed": r["fires_per_seed_mean"],
                "fires_ci": [r["fires_per_seed_ci_lo"], r["fires_per_seed_ci_hi"]],
                "savings_pct": r["savings_pct_mean"],
                "savings_ci": [r["savings_pct_ci_lo"], r["savings_pct_ci_hi"]],
                "jaccard_fire_sets": r["fire_set_jaccard_mean"],
                "per_step_full_agreement": r["per_step_full_agreement"],
            } for r in rows_single
        ],
        "two_threshold_grid": grid,
        "cross_seed_kendall_tau_zvf_trajectory": tau_summary,
        "panel": {
            "heldout_acc": [round(v, 4) for v in heldout],
            "mean_zvf": [round(v, 4) for v in mean_zvf],
            "last10_zvf": [round(v, 4) for v in last10_zvf],
            "pearson_heldout_vs_mean_zvf": round(r1, 3),
            "pearson_heldout_vs_mean_zvf_ci95": r1_ci,
            "pearson_heldout_vs_last10_zvf": round(r2, 3),
            "pearson_heldout_vs_last10_zvf_ci95": r2_ci,
        },
    }
    summary_path = OUT / "p7_tau_seed_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[saved] {summary_path}")

    # ---------- Print headline numbers ----------
    print()
    print("=" * 70)
    headline = next(r for r in rows_single if r["tau_esc"] == 0.70)
    print("HEADLINE: at τ_esc=0.70 (the iter-51 default), per-seed firing")
    print(f"         agreement is Jaccard={headline['fire_set_jaccard_mean']:.3f}, "
          f"full step agreement={headline['per_step_full_agreement']:.3f}")
    print(f"         {n_seeds} seeds; fires/seed={headline['fires_per_seed_mean']:.3f}, "
          f"CI=[{headline['fires_per_seed_ci_lo']:.3f}, {headline['fires_per_seed_ci_hi']:.3f}]")
    print("=" * 70)

if __name__ == "__main__":
    main()
