"""
Iter 151 / row 19 — B-SP25 L8 AlphaProof-MCTS-ZVF prototype.

Lecture: SP25 L8 — Thomas Hubert (DeepMind) — AlphaProof / AlphaZero.

Verified citations (WebFetch on 2026-07-04):
  - AlphaProof (DeepMind) — IMO 2024 silver-medal system combining a pretrained
    LM with AlphaZero-style RL on Lean statements. Methodology published in
    Nature, s41586-025-09833-y (Nov 12, 2025); blog announcement July 25, 2024.
  - AlphaZero — "Mastering Chess and Shogi by Self-Play with a General
    Reinforcement Learning Algorithm", Silver, Hubert, Schrittwieser, et al.,
    arXiv:1712.01815 (submitted Dec 5, 2017).

Mapping: A3 (post-training science) + A5 (inference-time reasoning).
AlphaProof's central mechanism is AlphaZero-style MCTS over Lean-formalized
proof states, where the value baseline V(s_t) is learned. In GRPO/RLVR terms,
the analogue is a tree-discounted baseline β_tree(t; γ, h), since the group-
mean baseline β_group = μ_g is the depth-0, undiscounted instantiation.

Five pre-registered hypotheses on iter127 Pillar-2 group-size data + iter130
variance-mitigation 9-method suite.

Inputs (read-only):
  platform_hybrid/experiments/results/group_size_advantage_variance.tsv   (16x40 = 640 rows)
  platform_hybrid/experiments/results/variance_mitigation.tsv             (9 methods x 5 seeds x 122 steps = 5490 rows)
Outputs (written):
  platform_hybrid/experiments/results/berkeley/alphaproof_<...>.tsv + .json

Author: Berkeley-curriculum mining iter 151.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
IT127_TSV = ROOT / "experiments" / "results" / "group_size_advantage_variance.tsv"
IT130_TSV = ROOT / "experiments" / "results" / "variance_mitigation.tsv"
OUTDIR = ROOT / "experiments" / "results" / "berkeley"

GRPS = [2, 4, 8, 16]  # iter127 fixed G values
WINDOWS = [1, 2, 5, 10, 20]  # tree-window baseline depths
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]  # tree-baseline discount factors


def read_tsv(path: Path):
    """Minimal TSV reader."""
    out = []
    with path.open() as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(head):
                continue
            out.append(
                {
                    h: (float(v) if h not in ("method", "G") and v else v)
                    for h, v in zip(head, parts, strict=False)
                }
            )
    return out


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in header) + "\n")


def sign_test(rows, col_a, col_b):
    """Wilcoxon-style sign test for paired (a, b) samples.

    Returns +1 / -1 / 0 win counts and binom p of sign a>b."""
    pos = neg = zero = 0
    for r in rows:
        a = r[col_a]
        b = r[col_b]
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("sign test requires finite pairs")
        if a > b:
            pos += 1
        elif a < b:
            neg += 1
        else:
            zero += 1
    n_eff = pos + neg
    if n_eff == 0:
        return {"pos": pos, "neg": neg, "zero": zero, "n_eff": 0, "binom_p": 1.0}
    p2 = min(1.0, 2 * sum(math.comb(n_eff, k) for k in range(min(pos, neg) + 1)) / 2**n_eff)
    return {
        "pos": pos,
        "neg": neg,
        "zero": zero,
        "n_eff": n_eff,
        "binom_p": p2,
        "alternative": "two-sided",
    }


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    sp = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if sp == 0:
        return 0.0
    return (ma - mb) / sp


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    rx = rank(xs)
    ry = rank(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank_avg = (i + 1 + j) / 2
        for k in range(i, j):
            rk[order[k]] = rank_avg
        i = j
    return rk


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def tree_baseline(traj, t, gamma=0.5, h=5, mode="forward"):
    """AlphaProof-style discounted tree baseline.

    trajectory is a list of (mean_reward, step). Returns V_tree(s_t; γ, h).
    mode='forward': V(t) = Σ_{i=0..h-1} γ^i * r(t+i)  (look-ahead)
    mode='backward': V(t) = Σ_{i=0..h-1} γ^i * r(t-i) (look-back)
    """
    if not traj:
        return 0.0
    res = 0.0
    for i in range(h):
        if mode == "forward":
            idx = t + i
        else:
            idx = t - i
        if 0 <= idx < len(traj):
            res += (gamma**i) * traj[idx]
    return res


def main():
    it127 = read_tsv(IT127_TSV)
    it130 = read_tsv(IT130_TSV)
    if not it127 or not it130:
        raise ValueError("INSUFFICIENT_DATA: both independent-run datasets are required")
    print(f"loaded: iter127 n={len(it127)}, iter130 n={len(it130)}")

    # ---- Build per-cell iter127 trajectories ----
    by_g_seed = defaultdict(list)
    for r in it127:
        key = (int(r["G"]), int(r["seed"]))
        by_g_seed[key].append(r)
    for k in by_g_seed:
        by_g_seed[k].sort(key=lambda r: r["step"])

    # ---- H1: Tree-window ZVF baseline reduction (iter127 + iter130) ----
    # For each (G, seed) trajectory, compute advantage_variance under a
    # window-mean baseline of size w. tree_baseline_advantage_var_proxy =
    # std(|r_step - V_tree|) over the trajectory.
    h1_rows = []
    for (g, sd), traj in by_g_seed.items():
        traj_sorted = traj
        rew = [t["mean_reward"] for t in traj_sorted]
        adv_orig = [t["advantage_variance"] for t in traj_sorted]
        # Advantage variance is reported separately; it is not a zero-variance fraction.
        zvf_naive = sum(adv_orig) / len(adv_orig)
        for w in WINDOWS:
            # Look-back window mean of mean_reward
            tree_rew = []
            for idx in range(len(rew)):
                lo = max(0, idx - w + 1)
                window = rew[lo : idx + 1]
                v = sum(window) / len(window)
                tree_rew.append(abs(rew[idx] - v))
            # tree-advantage proxy: mean of |r - V_tree| within window
            tree_adv_proxy = sum(tree_rew) / len(tree_rew)
            h1_rows.append(
                {
                    "G": g,
                    "seed": sd,
                    "w": w,
                    "mean_advantage_variance": zvf_naive,
                    "mean_reward_deviation": tree_adv_proxy,
                }
            )
    write_tsv(
        OUTDIR / "alphaproof_tree_window.tsv",
        ["G", "seed", "w", "mean_advantage_variance", "mean_reward_deviation"],
        h1_rows,
    )
    print(f"H1 wrote alphaproof_tree_window.tsv n={len(h1_rows)}")

    by_w = defaultdict(list)
    for r in h1_rows:
        by_w[r["w"]].append(r["mean_reward_deviation"])
    h1_summary = {
        w: {
            "n_runs": len(values),
            "mean_reward_deviation": statistics.fmean(values),
            "sd_across_runs": statistics.stdev(values) if len(values) > 1 else None,
        }
        for w, values in sorted(by_w.items())
    }
    h1_verdict = "EXPLORATORY_PROXY_ONLY"
    print(f"H1: reward deviations by window = {h1_summary} -> {h1_verdict}")

    # ---- H2: Compute equivalence (Tree-ZVF(G=2, w=2) ≳ ZVF(G=4, w=1)) ----
    # Compare zvf_naive at G=4 vs tree_adv_proxy at G=2, w=2 paired by seed
    by_g_w = defaultdict(list)
    for r in h1_rows:
        by_g_w[(r["G"], r["w"])].append(r)
    g2_w2 = by_g_w[(2, 2)]
    g4_w1 = by_g_w[(4, 1)]
    seed_to_g2w2 = {r["seed"]: r["mean_reward_deviation"] for r in g2_w2}
    seed_to_g4w1 = {r["seed"]: r["mean_reward_deviation"] for r in g4_w1}
    paired = []
    for sd in seed_to_g2w2:
        if sd in seed_to_g4w1:
            paired.append(
                {
                    "seed": sd,
                    "g2_w2": seed_to_g2w2[sd],
                    "g4_w1": seed_to_g4w1[sd],
                    "delta": seed_to_g2w2[sd] - seed_to_g4w1[sd],
                }
            )
    write_tsv(
        OUTDIR / "alphaproof_compute_equivalence.tsv", ["seed", "g2_w2", "g4_w1", "delta"], paired
    )
    h2_st = sign_test(paired, "g2_w2", "g4_w1")
    # prediction: g2_w2 <= g4_w1 (tree-baseline partially substitutes)
    h2_d = cohens_d([p["g2_w2"] for p in paired], [p["g4_w1"] for p in paired])
    h2_neg = sum(1 for p in paired if p["delta"] <= 0)
    h2_verdict = "INSUFFICIENT_DATA" if len(paired) < 2 else "EXPLORATORY_PROXY_ONLY"
    print(f"H2: g2_w2 vs g4_w1, neg={h2_neg}/{len(paired)} d={h2_d:.3f} -> {h2_verdict}")

    # ---- H3: Calibrated γ < 1 reduces ZVF magnitude channel (iter127) ----
    # Sweep γ at fixed window h=5 over forward-looking tree baseline.
    h3_rows = []
    for (g, sd), traj in by_g_seed.items():
        rew = [t["mean_reward"] for t in traj]
        adv_orig = [t["advantage_variance"] for t in traj]
        zvf_naive = sum(adv_orig) / len(adv_orig)
        for gamma in GAMMAS:
            # discounted cumulative return from each step (h=5 forward)
            tree_rew = []
            for t in range(len(rew)):
                v = tree_baseline(rew, t, gamma=gamma, h=5, mode="forward")
                tree_rew.append(abs(rew[t] - v))
            mag = sum(tree_rew) / len(tree_rew)
            h3_rows.append(
                {
                    "G": g,
                    "seed": sd,
                    "gamma": gamma,
                    "mag": mag,
                    "mean_advantage_variance": zvf_naive,
                    "delta_mag": mag - 0.0,
                }
            )
    write_tsv(
        OUTDIR / "alphaproof_gamma_sweep.tsv",
        ["G", "seed", "gamma", "mag", "mean_advantage_variance", "delta_mag"],
        h3_rows,
    )
    by_g = defaultdict(list)
    for r in h3_rows:
        by_g[r["gamma"]].append(r["delta_mag"])
    h3_summary = {g: sum(v) / len(v) for g, v in by_g.items()}
    # find argmin
    gamma_opt = min(h3_summary, key=lambda g: h3_summary[g])
    h3_neg_at_opt = sum(1 for r in h3_rows if r["gamma"] == gamma_opt and r["delta_mag"] < 0)
    h3_total_at_opt = sum(1 for r in h3_rows if r["gamma"] == gamma_opt)
    # CDH-consistent prediction: gamma_opt < 1.0
    h3_verdict = "EXPLORATORY_PROXY_ONLY"
    print(
        f"H3: mean Δmag by γ = {h3_summary}, γ*={gamma_opt}, "
        f"neg_at_opt={h3_neg_at_opt}/{h3_total_at_opt} -> {h3_verdict}"
    )

    # ---- H4: Tree-baseline ZVF sign-stable against GRPO across 9 methods (iter130) ----
    # For each method, build mean reward trajectory and compute correlation
    # between naive ZVF trajectory and tree-baseline ZVF trajectory.
    by_method_seed = defaultdict(list)
    for r in it130:
        by_method_seed[(r["method"], int(r["seed"]))].append(r)
    for k in by_method_seed:
        by_method_seed[k].sort(key=lambda r: r["step"])
    h4_rows = []
    for (method, sd), traj in by_method_seed.items():
        rew = [t["reward_mean"] for t in traj]
        zvf_naive = [t["zvf"] for t in traj]
        for w in [2, 5]:
            tree_rew = []
            for t in range(len(rew)):
                lo = max(0, t - w + 1)
                window = rew[lo : t + 1]
                v = sum(window) / len(window)
                tree_rew.append(abs(rew[t] - v))
            tree_zvf_proxy = sum(tree_rew) / len(tree_rew)
            corr = pearson(zvf_naive, tree_rew)
            h4_rows.append(
                {
                    "method": method,
                    "seed": sd,
                    "w": w,
                    "zvf_naive_mean": sum(zvf_naive) / len(zvf_naive),
                    "tree_zvf_proxy": tree_zvf_proxy,
                    "pearson_zvf_to_treeadv": corr,
                }
            )
    write_tsv(
        OUTDIR / "alphaproof_method_sign.tsv",
        ["method", "seed", "w", "zvf_naive_mean", "tree_zvf_proxy", "pearson_zvf_to_treeadv"],
        h4_rows,
    )
    # pearson > 0 means sign-stable
    pos_corr = sum(1 for r in h4_rows if r["pearson_zvf_to_treeadv"] > 0)
    h4_strict = sum(1 for r in h4_rows if r["pearson_zvf_to_treeadv"] > 0.5)
    n_total = len(h4_rows)
    h4_verdict = "INSUFFICIENT_DATA" if n_total < 2 else "EXPLORATORY_PROXY_ONLY"
    print(
        f"H4: positive Pearson on {pos_corr}/{n_total} (strict>{0.5}: {h4_strict}/{n_total}) -> {h4_verdict}"
    )

    # ---- H5: Δ(naive vs tree) tracks heldout_acc at final step (iter130) ----
    final_step = {}
    for r in it130:
        k = (r["method"], int(r["seed"]))
        if k not in final_step or r["step"] > final_step[k]["step"]:
            final_step[k] = r
    paired_acc = []
    for r in h4_rows:
        k = (r["method"], r["seed"])
        if k in final_step:
            final = final_step[k]
            paired_acc.append(
                {
                    "method": r["method"],
                    "seed": r["seed"],
                    "w": r["w"],
                    "tree_zvf_proxy": r["tree_zvf_proxy"],
                    "zvf_naive_mean": r["zvf_naive_mean"],
                    "heldout_acc_final": final["heldout_acc"],
                }
            )
    # Stratify by method and window: never duplicate seeds by pooling windows.
    correlations = []
    for method in sorted({p["method"] for p in paired_acc}):
        for window in [2, 5]:
            cells = [p for p in paired_acc if p["method"] == method and p["w"] == window]
            if len(cells) >= 3:
                correlations.append(
                    {
                        "method": method,
                        "w": window,
                        "n_seeds": len(cells),
                        "rho_proxy": spearman(
                            [p["heldout_acc_final"] for p in cells],
                            [p["tree_zvf_proxy"] for p in cells],
                        ),
                        "rho_zvf": spearman(
                            [p["heldout_acc_final"] for p in cells],
                            [p["zvf_naive_mean"] for p in cells],
                        ),
                    }
                )
    # Averages are descriptive only, not pooled-sample inferential estimates.
    sp = statistics.fmean(c["rho_proxy"] for c in correlations) if correlations else 0.0
    sp_naive = statistics.fmean(c["rho_zvf"] for c in correlations) if correlations else 0.0
    write_tsv(
        OUTDIR / "alphaproof_final_acc_corr.tsv",
        ["method", "seed", "w", "tree_zvf_proxy", "zvf_naive_mean", "heldout_acc_final"],
        paired_acc,
    )
    # prediction: both are positive
    h5_verdict = "EXPLORATORY_PROXY_ONLY" if correlations else "INSUFFICIENT_DATA"
    print(
        f"H5: spearman ρ(tree, heldout)={sp:.3f}, ρ(naive, heldout)={sp_naive:.3f} -> {h5_verdict}"
    )

    # Aggregated trajectory proxies cannot establish changes in within-group ZVF.
    # Do not promote directional heuristics or pooled dependent correlations to evidence.
    h1_verdict = h3_verdict = h4_verdict = "EXPLORATORY_PROXY_ONLY"
    h2_verdict = "EXPLORATORY_PROXY_ONLY" if len(paired) >= 2 else "INSUFFICIENT_DATA"
    h5_verdict = "EXPLORATORY_PROXY_ONLY" if correlations else "INSUFFICIENT_DATA"
    # ---- Evidence summary ----
    summary = {
        "limitations": "Reward deviation and advantage variance are separate estimands. No ZVF reduction, compute equivalence, stability or inferential claim is established; repeated windows/methods are dependent.",
        "ts": "2026-07-04",
        "iteration": 19,
        "pillar": "B-SP25",
        "lecture": "SP25 L8 — Thomas Hubert / DeepMind (AlphaProof + AlphaZero)",
        "papers": {
            "alphaproof": {
                "title": "AlphaProof: a formal-mathematics AI for IMO competition problems",
                "venue": "Nature (s41586-025-09833-y, methodology Nov 12, 2025) and DeepMind blog (Jul 25, 2024)",
                "components": [
                    "pretrained LM (Gemini)",
                    "AlphaZero MCTS over Lean states",
                    "value-net + policy prior",
                ],
                "year": 2024,
            },
            "alphazero": {
                "title": "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",
                "authors": "Silver, Hubert, Schrittwieser, Antonoglou, Lai, Guez, Lanctot, Sifre, Kumaran, Graepel, Lillicrap, Simonyan, Hassabis",
                "arxiv": "1712.01815",
                "venue": "arXiv preprint (cs.AI)",
                "year": 2017,
            },
        },
        "framework": "AlphaZero MCTS over Lean states -> tree-baseline V(s_t; γ, h). Operationalised in GRPO/RLVR as discounted/look-back baseline β_tree(t; γ, h); depth-0 h=1, γ=1 reduces to GRPO group-mean. Five pre-registered hypotheses on iter127 + iter130 data.",
        "data_sources": {
            "iter127_group_size_advantage_variance": str(IT127_TSV.relative_to(ROOT)),
            "iter130_variance_mitigation": str(IT130_TSV.relative_to(ROOT)),
        },
        "hypotheses": [
            {
                "id": "H1",
                "claim": "Descriptive tree-window mean reward deviation; no ZVF inference",
                "summary": h1_summary,
                "verdict": h1_verdict,
            },
            {
                "id": "H2",
                "claim": "Descriptive reward-deviation contrast across G/window; not compute equivalence",
                "n_paired": len(paired),
                "neg_delta": h2_neg,
                "cohens_d": h2_d,
                "sign_test": h2_st,
                "verdict": h2_verdict,
            },
            {
                "id": "H3",
                "claim": "Descriptive discounted-baseline deviation sweep; no ZVF inference",
                "mean_delta_by_gamma": h3_summary,
                "gamma_optimal": gamma_opt,
                "neg_at_opt": f"{h3_neg_at_opt}/{h3_total_at_opt}",
                "verdict": h3_verdict,
            },
            {
                "id": "H4",
                "claim": "Within-run association of reward-deviation proxy with observed ZVF; not stability",
                "n_total": n_total,
                "n_pos_corr": pos_corr,
                "n_strict": h4_strict,
                "verdict": h4_verdict,
            },
            {
                "id": "H5",
                "claim": "Method/window-specific seed correlations with final accuracy; exploratory",
                "method_window_seed_correlations": correlations,
                "mean_descriptive_rho_proxy": sp,
                "mean_descriptive_rho_zvf": sp_naive,
                "verdict": h5_verdict,
            },
        ],
    }
    with (OUTDIR / "alphaproof_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote alphaproof_summary.json")
    print(f"\nFinal summary:")
    for h in summary["hypotheses"]:
        print(f"  {h['id']}: {h['verdict']} -- {h['claim']}")


if __name__ == "__main__":
    main()
