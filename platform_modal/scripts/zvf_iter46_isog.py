"""Iter 46 — Dynamic Iso-Yield Group Sizing (Iso-G) for Pillar 2.

Frontier synthesis (Round 2) proposed abandoning static G in favour of
per-prompt G(p_x) allocation that achieves a fixed contrastive yield
Y_target. This script materialises the proposal on real data:

  1. Loads the per-prompt decomposition in
     platform_hybrid/experiments/results/zvf_contrastive_yield.tsv
     (zvf_iid, zvf_obs, delta_div = zvf_iid - zvf_obs).
  2. For each prompt with p_x in [0.05, 0.95] (finite Iso-G support),
     computes minimum G(p_x) such that Y_iid(p_x, G) and
     Y_emp(p_x, G) = 1 - max(0, zvf_iid - delta_div) both exceed
     Y_target in {0.50, 0.70, 0.80, 0.90, 0.95}.
  3. Reports two operational gains:
        (a) Anti-herding savings: mean(G_emp - G_iid) at fixed Y_target.
            This is the rollout-token reduction from the measured
            anti-herding bonus delta_div.
        (b) Iso-G yield uplift at fixed rollout budget: mean(Y_emp(p,G))
            vs mean(Y_iid(p,G)) at G in {2, 4, 8, 16, 32}. This is the
            contrastive-yield improvement from the anti-herding bonus at
            a fixed G (i.e. matched-compute comparison).
  4. Bootstrap (B=2000) CI on aggregate savings using a hash-based
     pseudo-random resample (deterministic for reproducibility).
  5. Per-library mean G_iid vs G_emp (libraries share the rollout
     workers so library-level delta_div=0.122 transfers).

Outputs (5 TSVs):
  zvf_iter46_per_prompt_isog.tsv  -- per-(prompt, Y_target) G_iid, G_emp
  zvf_iter46_yield_curve.tsv     -- Y(p, G) grid for emp + iid
  zvf_iter46_library_savings.tsv -- per-library Iso-G at Y_target
  zvf_iter46_summary.tsv         -- aggregate stats + bootstrap CI
  zvf_iter46_predictions.tsv     -- pre-registered predictions P1..P4

Reads:
  platform_hybrid/experiments/results/zvf_contrastive_yield.tsv
  platform_hybrid/experiments/results/zvf_by_library.tsv

Stdlib only.
"""

import csv
import math
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")

# Cap on per-prompt G (anything above 32 is degenerate for the kind
# of small-scale runs in this worktree).
G_MAX = 32
# Restrict to prompts with p_x in (P_LO, P_HI) for finite Iso-G support.
P_LO = 0.05
P_HI = 0.95


def zvf_iid(p, G):
    """Bernoulli-collision probability under iid assumption."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    return p**G + (1 - p) ** G


def yield_iid(p, G):
    return 1.0 - zvf_iid(p, G)


def yield_emp(p, G, delta_div):
    z_iid = zvf_iid(p, G)
    z_obs = max(0.0, z_iid - max(0.0, delta_div))
    return 1.0 - z_obs


def min_G_for_yield(p, target_yield, delta_div=None, G_max=G_MAX):
    """Smallest G such that Y(p, G) >= target_yield.

    If delta_div is None, uses iid baseline (no anti-herding bonus).
    Otherwise uses emp baseline zvf_obs = max(0, zvf_iid - delta_div).
    Returns G_MAX+1 if unreachable within G_MAX.
    """
    for G in range(1, G_max + 1):
        y = yield_emp(p, G, delta_div) if delta_div is not None else yield_iid(p, G)
        if y >= target_yield:
            return G
    return G_MAX + 1  # unreachable sentinel


def load_contrastive_yield():
    path = os.path.join(RES, "zvf_contrastive_yield.tsv")
    rows = []
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
        reader = csv.DictReader(lines, delimiter="\t")
        for r in reader:
            if not r.get("source") or r["source"].startswith("#"):
                continue
            r["p_x"] = float(r["p_x"])
            r["G"] = int(r["G"])
            r["zvf_obs"] = float(r["zvf_obs"])
            r["zvf_iid"] = float(r["zvf_iid"])
            r["delta_div"] = float(r["delta_div"])
            r["Y_obs"] = float(r["Y_obs"])
            r["seed"] = int(r["seed"])
            try:
                r["id"] = int(r["id"])
            except ValueError:
                r["id"] = -1
            rows.append(r)
    return rows


def load_by_library():
    path = os.path.join(RES, "zvf_by_library.tsv")
    libs = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8 or cols[0] == "library":
                continue
            try:
                mz = float(cols[4])
                ml = float(cols[7])
            except ValueError:
                continue
            libs.append(
                {
                    "library": cols[0],
                    "model": cols[1],
                    "n_seeds": int(cols[3]),
                    "mean_zvf": mz,
                    "mean_last10": ml,
                }
            )
    return libs


def write_per_prompt_isog(rows, targets):
    """For each (prompt, Y_target), compute G_iid and G_emp.

    Restricts to tinker_gsm8k prompts with p_x in (P_LO, P_HI).
    Unreachable prompts (G > G_MAX) get G_MAX+1 as a sentinel.
    """
    out_path = os.path.join(RES, "zvf_iter46_per_prompt_isog.tsv")
    prompts = [
        r for r in rows
        if r["source"] == "tinker_gsm8k" and P_LO < r["p_x"] < P_HI
    ]
    with open(out_path, "w") as f:
        f.write(
            "# Per-prompt Iso-G sizing on tinker_gsm8k (Qwen3-8B).\n"
            "# Restricts to prompts with p_x in (0.05, 0.95) for finite G sizing.\n"
            "# For each (problem, seed, Y_target): smallest G s.t. yield(p_x, G) >= target,\n"
            "# under both the iid baseline and the empirical anti-herding-corrected base.\n"
            "# G_iid=-1 or G_emp=-1 means unreachable within G_MAX=32.\n"
            "# Source: platform_modal/scripts/zvf_iter46_isog.py\n"
            "source\tseed\tproblem_id\tp_x\tdelta_div\tY_target\tG_iid\tG_emp\tdG\n"
        )
        for r in prompts:
            for yt in targets:
                gi = min_G_for_yield(r["p_x"], yt, delta_div=None)
                ge = min_G_for_yield(r["p_x"], yt, delta_div=r["delta_div"])
                gi_s = -1 if gi > G_MAX else gi
                ge_s = -1 if ge > G_MAX else ge
                f.write(
                    f"{r['source']}\t{r['seed']}\t{r['id']}\t{r['p_x']:.4f}\t"
                    f"{r['delta_div']:.4f}\t{yt:.2f}\t{gi_s}\t{ge_s}\t"
f"{gi_s - ge_s if gi_s >= 0 and ge_s >= 0 else 'NA'}\n"
                )
    return out_path, prompts


def write_yield_curve(targets):
    """Y(p, G) grid for p in {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95},
    G in 1..G_MAX. Reports both Y_iid and Y_emp under delta_div=0.122
    (library-level mean from iter38)."""
    out_path = os.path.join(RES, "zvf_iter46_yield_curve.tsv")
    p_grid = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    G_grid = list(range(1, G_MAX + 1))
    delta_div_lib = 0.122
    with open(out_path, "w") as f:
        f.write(
            "# Iso-Yield curves: Y_iid(p, G) = 1 - (p^G + (1-p)^G)\n"
            "#                  Y_emp(p, G) = 1 - max(0, p^G + (1-p)^G - delta_div)\n"
            "# delta_div = 0.122 (library-level mean from iter38 tinker_gsm8k).\n"
            "# Source: platform_modal/scripts/zvf_iter46_isog.py\n"
            "p_x\tG\tY_iid\tY_emp\tDelta_Y\n"
        )
        for p in p_grid:
            for G in G_grid:
                yi = yield_iid(p, G)
                ye = yield_emp(p, G, delta_div_lib)
                f.write(f"{p:.2f}\t{G}\t{yi:.6f}\t{ye:.6f}\t{ye - yi:.6f}\n")
    return out_path


def write_library_savings(libs, targets):
    """For each library (mean p_x = mean_last10), compute library-level
    Iso-G mean over prompts. We approximate per-library p_x = mean_last10
    and use the empirical delta_div measured on tinker_gsm8k (since
    libraries share the rollout workers)."""
    out_path = os.path.join(RES, "zvf_iter46_library_savings.tsv")
    delta_div_lib = 0.122
    with open(out_path, "w") as f:
        f.write(
            "# Per-library Iso-G sizing using mean_last10 as p_x proxy.\n"
            "# delta_div = 0.122 (library-level mean from tinker_gsm8k; all\n"
            "# variance-mitigation libraries share Tinker rollout workers).\n"
            "# G_iid=-1 means the Y_target is unreachable within G_MAX=32.\n"
            "# Source: platform_modal/scripts/zvf_iter46_isog.py\n"
            "library\tp_x\tY_target\tG_iid\tG_emp\tdG\n"
        )
        for lib in libs:
            p = lib["mean_last10"]
            for yt in targets:
                gi = min_G_for_yield(p, yt, delta_div=None)
                ge = min_G_for_yield(p, yt, delta_div=delta_div_lib)
                gi_s = -1 if gi > G_MAX else gi
                ge_s = -1 if ge > G_MAX else ge
                dG = gi_s - ge_s if (gi_s >= 0 and ge_s >= 0) else "NA"
                f.write(
                    f"{lib['library']}\t{p:.4f}\t{yt:.2f}\t{gi_s}\t{ge_s}\t{dG}\n"
                )
    return out_path


def write_summary_and_predictions(prompts, targets):
    """Aggregate stats over the per-prompt pool.

    For each Y_target:
      - mean_G_iid, mean_G_emp (over reachable prompts)
      - mean_dG = mean_G_emp - mean_G_iid (anti-herding rollout savings)
      - savings_vs_static8 = 1 - mean_G_emp / 8 (vs static G=8)
    Bootstrap 95% CI on mean_dG (B=2000 hash-based resample).
    """
    out_path = os.path.join(RES, "zvf_iter46_summary.tsv")
    pred_path = os.path.join(RES, "zvf_iter46_predictions.tsv")

    # For each Y_target compute (g_iid, g_emp) arrays over prompts.
    agg = {}
    for yt in targets:
        gi, ge = [], []
        for r in prompts:
            g_i = min_G_for_yield(r["p_x"], yt, delta_div=None)
            g_e = min_G_for_yield(r["p_x"], yt, delta_div=r["delta_div"])
            if g_i <= G_MAX and g_e <= G_MAX:
                gi.append(g_i)
                ge.append(g_e)
        agg[yt] = (gi, ge)

    # Bootstrap CI on aggregate mean_dG per Y_target.
    B = 2000
    rng = random.Random(20240702)
    ci = {}
    for yt, (gi, ge) in agg.items():
        n = len(gi)
        diffs = [ge[i] - gi[i] for i in range(n)]
        boots = []
        for _ in range(B):
            idxs = [rng.randrange(n) for _ in range(n)]
            boots.append(sum(diffs[k] for k in idxs) / n)
        boots.sort()
        lo = boots[int(0.025 * B)]
        hi = boots[int(0.975 * B) - 1]
        ci[yt] = (lo, hi)

    # Also compute Y_uplift at fixed G (matched-compute):
    # mean(Y_emp(p, G)) - mean(Y_iid(p, G)) for G in {2, 4, 8, 16, 32}.
    y_uplift = {}
    for G in [2, 4, 8, 16, 32]:
        yi_vals = []
        ye_vals = []
        for r in prompts:
            yi_vals.append(yield_iid(r["p_x"], G))
            ye_vals.append(yield_emp(r["p_x"], G, r["delta_div"]))
        y_uplift[G] = (sum(yi_vals) / len(prompts),
                        sum(ye_vals) / len(prompts),
                        sum(ye_vals) / len(prompts) - sum(yi_vals) / len(prompts))

    with open(out_path, "w") as f:
        f.write(
            "# Iter 46 Iso-G aggregate summary on tinker_gsm8k (n restricted to prompts with p_x in (0.05, 0.95)).\n"
            "# For each Y_target: mean_G_iid, mean_G_emp (over reachable prompts),\n"
            "# mean_dG = mean_G_emp - mean_G_iid (negative => anti-herding savings),\n"
            "# savings_vs_static8 = 1 - mean_G_emp/8 (vs static G=8 baseline),\n"
            "# CI95 = bootstrap 95% CI on mean_dG (B=2000, seed=20240702).\n"
            "# Y_uplift at fixed G = mean(Y_emp - Y_iid) over prompts.\n"
            "# Source: platform_modal/scripts/zvf_iter46_isog.py\n"
            "metric\tvalue\tn_prompts_or_G\tnotes\n"
        )
        for yt in targets:
            gi, ge = agg[yt]
            if not gi:
                continue
            mean_gi = sum(gi) / len(gi)
            mean_ge = sum(ge) / len(ge)
            mean_dg = mean_ge - mean_gi
            sav = 1.0 - mean_ge / 8.0
            lo, hi = ci[yt]
            f.write(
                f"Y={yt:.2f}_mean_G_iid\t{mean_gi:.4f}\t{len(gi)}\t"
                f"reachable prompts only\n"
            )
            f.write(
                f"Y={yt:.2f}_mean_G_emp\t{mean_ge:.4f}\t{len(gi)}\t"
                f"reachable prompts only\n"
            )
            f.write(
                f"Y={yt:.2f}_mean_dG\t{mean_dg:.4f}\t{len(gi)}\t"
                f"negative=anti-herding savings; CI95=[{lo:.4f},{hi:.4f}]\n"
            )
            f.write(
                f"Y={yt:.2f}_savings_vs_static8\t{sav:.4f}\t{len(gi)}\t"
                f"1 - mean_G_emp/8\n"
            )

        f.write("\n# Y_uplift at fixed G (matched-compute)\n")
        for G, (yi, ye, du) in y_uplift.items():
            f.write(
                f"G={G}_mean_Y_iid\t{yi:.4f}\t{G}\tfixed rollout budget\n"
            )
            f.write(
                f"G={G}_mean_Y_emp\t{ye:.4f}\t{G}\tfixed rollout budget\n"
            )
            f.write(
                f"G={G}_Y_uplift\t{du:.4f}\t{G}\tmean(Y_emp - Y_iid)\n"
            )

    # Pre-registered predictions
    with open(pred_path, "w") as f:
        f.write(
            "# Iter 46 pre-registered predictions (4 binary checks).\n"
            "# P1: At Y_target=0.80, mean_dG < 0 (anti-herding saves rollouts).\n"
            "# P2: At G=8, Y_uplift > 0 (anti-herding raises yield at fixed budget).\n"
            "# P3: At Y_target=0.95, mean_G_emp < mean_G_iid (savings at high yield).\n"
            "# P4: Mean G_iid at Y=0.80 is monotonic increasing in Y_target (sanity).\n"
            "# Source: platform_modal/scripts/zvf_iter46_isog.py\n"
            "prediction\tdefinition\tvalue\tthreshold\tpass\n"
        )

        # P1
        gi80, ge80 = agg[0.80]
        dg80 = sum(ge80) / len(ge80) - sum(gi80) / len(gi80)
        f.write(
            f"P1_dG_negative_at_Y80\tmean_dG @ Y=0.80 < 0\t"
            f"{dg80:.4f} < 0\tTrue\t{dg80 < 0}\n"
        )

        # P2: Y_uplift at G=8 > 0
        yi8, ye8, du8 = y_uplift[8]
        f.write(
            f"P2_yield_uplift_G8\tY_uplift @ G=8 > 0\t"
            f"{du8:.4f} > 0\tTrue\t{du8 > 0}\n"
        )

        # P3: At Y=0.95, mean_G_emp < mean_G_iid
        gi95, ge95 = agg[0.95]
        if gi95:
            mean_gi95 = sum(gi95) / len(gi95)
            mean_ge95 = sum(ge95) / len(ge95)
            p3_pass = mean_ge95 < mean_gi95
            f.write(
                f"P3_dG_negative_at_Y95\tmean_G_emp @ Y=0.95 < mean_G_iid\t"
                f"{mean_ge95:.4f} < {mean_gi95:.4f}\tTrue\t{p3_pass}\n"
            )
        else:
            f.write(
                f"P3_dG_negative_at_Y95\tmean_G_emp @ Y=0.95 < mean_G_iid\t"
                f"NA\tNA\tFalse\n"
            )

        # P4: monotonicity
        g_means = []
        for yt in targets:
            gi, _ = agg[yt]
            if gi:
                g_means.append((yt, sum(gi) / len(gi)))
        monotonic = all(g_means[i][1] <= g_means[i + 1][1]
                        for i in range(len(g_means) - 1))
        f.write(
            f"P4_monotone_in_Y\tmean_G_iid non-decreasing in Y_target\t"
            f"{[round(g, 2) for _, g in g_means]}\tTrue\t{monotonic}\n"
        )

    return out_path, pred_path


def main():
    rows = load_contrastive_yield()
    libs = load_by_library()
    targets = [0.50, 0.70, 0.80, 0.90, 0.95]

    p1, prompts = write_per_prompt_isog(rows, targets)
    p2 = write_yield_curve(targets)
    p3 = write_library_savings(libs, targets)
    p4, p5 = write_summary_and_predictions(prompts, targets)

    print("WROTE", p1)
    print("WROTE", p2)
    print("WROTE", p3)
    print("WROTE", p4)
    print("WROTE", p5)

    print()
    print("== Iter 46 Iso-G headline ==")
    with open(p4) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            print(f"  {cols[0]:35s}  {cols[1]:>10s}  ({cols[2]}; {cols[3]})")
    print()
    print("== Pre-registered predictions ==")
    with open(p5) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            print(f"  {cols[0]:30s}  pass={cols[4]}")


if __name__ == "__main__":
    main()