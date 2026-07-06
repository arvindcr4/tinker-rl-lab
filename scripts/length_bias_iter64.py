"""
Iter 64 — Pillar 4 (Length Bias / Dr.GRPO): Conditional Length Response to
Reward Direction (CLRRD).

Per-step (ΔR_t, ΔL_t) trajectory; condition E[ΔL | sign(ΔR), ZVF tier].

Headline: on GSM8K CoT, GRPO's length growth is reward-aligned
(positive E[ΔL | ΔR>0] − E[ΔL | ΔR<0]); Dr.GRPO's is uniformly shrunk.
"""
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
SRC = ROOT / "experiments" / "results"

DATA_FILES = [
    SRC / "drgrpo_gsm8k_cot_full.json",   # GSM8K CoT, G=16, 3 seeds each
    SRC / "drgrpo_vs_grpo.json",          # arithmetic-easy, G=8, 5 seeds each
]

# ZVF tier cuts (per-step zvf value)
ZVF_HIGH = 0.70   # groups mostly homogeneous
ZVF_LOW = 0.30    # groups mostly heterogeneous

SMOOTH_K = 3      # centred moving-average window for R and L trajectories
N_BOOT = 2000
SEED = 20260702


def write_tsv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def smooth(xs, k):
    n = len(xs)
    out = [xs[i] for i in range(n)]
    if k <= 1:
        return out
    half = k // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = sum(xs[lo:hi]) / (hi - lo)
    return out


def step_records(run):
    """Return per-step (R_smooth, L_smooth, zvf) list aligned to step index."""
    log = run["step_log"]
    R = [s["mean_reward"] for s in log]
    L = [s["mean_comp_len"] for s in log]
    Z = [s["zvf"] for s in log]
    Rs = smooth(R, SMOOTH_K)
    Ls = smooth(L, SMOOTH_K)
    return [{"R": Rs[i], "L": Ls[i], "Z": Z[i]} for i in range(len(log))]


def deltas(recs):
    """Compute (ΔR, ΔL) per consecutive pair, drop the first sample."""
    out = []
    for i in range(1, len(recs)):
        out.append({
            "dR": recs[i]["R"] - recs[i - 1]["R"],
            "dL": recs[i]["L"] - recs[i - 1]["L"],
            "Z": recs[i]["Z"],
        })
    return out


def cell_mean(dlist, key_dR_sign=None, zvf_tier=None):
    """Average ΔL over steps matching the conditions."""
    vals = []
    for d in dlist:
        if key_dR_sign is None:
            pass
        elif key_dR_sign == "pos" and d["dR"] <= 0:
            continue
        elif key_dR_sign == "neg" and d["dR"] >= 0:
            continue
        elif key_dR_sign == "zero" and abs(d["dR"]) > 1e-9:
            continue
        if zvf_tier == "high" and d["Z"] < ZVF_HIGH:
            continue
        if zvf_tier == "mid" and not (ZVF_LOW <= d["Z"] < ZVF_HIGH):
            continue
        if zvf_tier == "low" and d["Z"] >= ZVF_LOW:
            continue
        vals.append(d["dL"])
    if not vals:
        return None
    return sum(vals) / len(vals)


def _safe(v):
    return 0.0 if v is None else v


def paired_bootstrap(per_seed_grpo, per_seed_drgrpo, fn, n_boot=N_BOOT, seed=SEED):
    """Generic paired bootstrap across seeds. fn(list_of_dlist) -> scalar or None."""
    import random
    rng = random.Random(seed)
    seeds = list(range(len(per_seed_grpo)))
    diffs = []
    g_means = [_safe(fn(s)) for s in per_seed_grpo]
    d_means = [_safe(fn(s)) for s in per_seed_drgrpo]
    for _ in range(n_boot):
        idx = [rng.randrange(len(seeds)) for _ in seeds]
        g_b = sum(_safe(fn(per_seed_grpo[i])) for i in idx) / len(idx)
        d_b = sum(_safe(fn(per_seed_drgrpo[i])) for i in idx) / len(idx)
        diffs.append(d_b - g_b)
    diffs.sort()
    n = len(diffs)
    lo = diffs[int(0.025 * n)]
    hi = diffs[int(0.975 * n) - 1]
    mean_diff = sum(d - g for g, d in zip(g_means, d_means)) / len(g_means)
    p_le0 = sum(1 for x in diffs if x <= 0) / n
    return {
        "n_pairs": len(g_means),
        "mean_grpo": sum(g_means) / len(g_means) if g_means else None,
        "mean_drgrpo": sum(d_means) / len(d_means) if d_means else None,
        "mean_diff": mean_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_le0": p_le0,
    }


def align_pairs(grpo_runs, drgrpo_runs):
    """Match seeds; return parallel lists of per-seed dlist."""
    by_seed_g = {r["seed"]: r for r in grpo_runs}
    by_seed_d = {r["seed"]: r for r in drgrpo_runs}
    common = sorted(set(by_seed_g) & set(by_seed_d))
    g, d = [], []
    for s in common:
        g.append(deltas(step_records(by_seed_g[s])))
        d.append(deltas(step_records(by_seed_d[s])))
    return common, g, d


def main():
    # ---------- Load & bucket per (experiment, algo) ----------
    by_exp = {}
    for fp in DATA_FILES:
        data = json.load(open(fp))
        for r in data["runs"]:
            exp = r["experiment"]
            algo = r["algo"]
            by_exp.setdefault(exp, {"grpo": [], "dr_grpo": []})
            by_exp[exp].setdefault(algo, []).append(r)

    # Per-run CLRRD table (each row = one seed).
    per_run_rows = []
    paired_rows = []

    for exp, runs_by_algo in by_exp.items():
        grpo = runs_by_algo.get("grpo", [])
        drgrpo = runs_by_algo.get("dr_grpo", [])
        if not grpo or not drgrpo:
            continue
        common, g_lists, d_lists = align_pairs(grpo, drgrpo)

        # ---- per-run rows
        for seed, glist, dlist in zip(common, g_lists, d_lists):
            for algo_name, lst, full_run in (("grpo", glist, None),
                                            ("dr_grpo", dlist, None)):
                pass  # filled below

        for seed, glist, dlist in zip(common, g_lists, d_lists):
            for algo_name, lst in (("grpo", glist), ("dr_grpo", dlist)):
                # cells
                for cond_label, sign in (("all", None),
                                          ("pos", "pos"),
                                          ("neg", "neg"),
                                          ("zero", "zero")):
                    for tier_label in (None, "high", "mid", "low"):
                        key = f"{cond_label}_{tier_label or 'all'}"
                        m = cell_mean(lst, key_dR_sign=sign, zvf_tier=tier_label)
                        per_run_rows.append({
                            "experiment": exp,
                            "algo": algo_name,
                            "seed": seed,
                            "cell": key,
                            "mean_dL": round(m, 6) if m is not None else "",
                            "n_steps": sum(
                                1 for x in lst
                                if (sign is None
                                    or (sign == "pos" and x["dR"] > 0)
                                    or (sign == "neg" and x["dR"] < 0)
                                    or (sign == "zero" and abs(x["dR"]) <= 1e-9))
                                and (tier_label is None
                                     or (tier_label == "high" and x["Z"] >= ZVF_HIGH)
                                     or (tier_label == "mid" and ZVF_LOW <= x["Z"] < ZVF_HIGH)
                                     or (tier_label == "low" and x["Z"] < ZVF_LOW))
                            ),
                        })

        # ---- paired GRPO vs Dr.GRPO bootstrap on key cells
        for cond_label, sign in (("all", None), ("pos", "pos"),
                                  ("neg", "neg")):
            for tier_label in (None, "high", "mid", "low"):
                key = f"{cond_label}_{tier_label or 'all'}"
                b = paired_bootstrap(
                    g_lists, d_lists,
                    lambda lst, s=sign, t=tier_label: cell_mean(lst, s, t),
                )
                paired_rows.append({
                    "experiment": exp,
                    "cell": key,
                    "n_pairs": b["n_pairs"],
                    "mean_grpo": round(b["mean_grpo"], 6) if b["mean_grpo"] is not None else "",
                    "mean_drgrpo": round(b["mean_drgrpo"], 6) if b["mean_drgrpo"] is not None else "",
                    "mean_diff": round(b["mean_diff"], 6),
                    "ci_lo": round(b["ci_lo"], 6),
                    "ci_hi": round(b["ci_hi"], 6),
                    "p_le0": round(b["p_le0"], 4),
                })

    # ---- The headline statistics:
    # alignment = E[dL | dR>0] - E[dL | dR<0]
    # positive => reward-aligned length growth
    # Both GRPO/Dr.GRPO typically have NEGATIVE alignment on a converging task
    # (they compress on reward-up steps). The "Dr.GRPO loses reward-responsiveness"
    # headline is therefore sign-blind: it's |alignment| that shrinks.
    summary_rows = []
    for exp, runs_by_algo in by_exp.items():
        grpo = runs_by_algo.get("grpo", [])
        drgrpo = runs_by_algo.get("dr_grpo", [])
        if not grpo or not drgrpo:
            continue
        common, g_lists, d_lists = align_pairs(grpo, drgrpo)

        def alignment(lst):
            pos = cell_mean(lst, "pos")
            neg = cell_mean(lst, "neg")
            if pos is None or neg is None:
                return None
            return pos - neg

        def abs_alignment(lst):
            a = alignment(lst)
            return abs(a) if a is not None else None

        def align_low(lst):
            pos = cell_mean(lst, "pos", "low")
            neg = cell_mean(lst, "neg", "low")
            if pos is None or neg is None:
                return None
            return pos - neg

        def align_high(lst):
            pos = cell_mean(lst, "pos", "high")
            neg = cell_mean(lst, "neg", "high")
            if pos is None or neg is None:
                return None
            return pos - neg

        def raw_drift(lst):
            v = cell_mean(lst)
            return v

        def compression_on_pos(lst):
            """How much L shrinks on reward-up steps (positive value = compression)."""
            m = cell_mean(lst, "pos")
            return -m if m is not None else None

        def compression_on_neg(lst):
            m = cell_mean(lst, "neg")
            return -m if m is not None else None

        b_align = paired_bootstrap(g_lists, d_lists, alignment)
        b_abs_align = paired_bootstrap(g_lists, d_lists, abs_alignment)
        b_low = paired_bootstrap(g_lists, d_lists, align_low)
        b_high = paired_bootstrap(g_lists, d_lists, align_high)
        b_drift = paired_bootstrap(g_lists, d_lists, raw_drift)
        b_comp_pos = paired_bootstrap(g_lists, d_lists, compression_on_pos)
        b_comp_neg = paired_bootstrap(g_lists, d_lists, compression_on_neg)

        for label, bres, interp_fn in (
            ("alignment_all", b_align,
             lambda x: "GRPO more reward-responsive than Dr.GRPO"
                       if x["mean_diff"] < 0 else "inconclusive"),
            ("|alignment|_all", b_abs_align,
             lambda x: "Dr.GRPO has weaker reward-direction coupling"
                       if x["mean_diff"] < 0 else "inconclusive"),
            ("alignment_low_zvf", b_low,
             lambda x: "Dr.GRPO loses alignment in low-ZVF (heterogeneous groups)"
                       if x["mean_diff"] < 0 else "inconclusive"),
            ("alignment_high_zvf", b_high,
             lambda x: "Dr.GRPO loses alignment in high-ZVF (homogeneous groups)"
                       if x["mean_diff"] < 0 else "inconclusive"),
            ("raw_drift_all", b_drift,
             lambda x: "Dr.GRPO has higher baseline length drift (less compression)"
                       if x["mean_diff"] > 0 else "inconclusive"),
            ("compression_on_pos_dR", b_comp_pos,
             lambda x: "GRPO compresses more on reward-up steps"
                       if x["mean_diff"] > 0 else "inconclusive"),
            ("compression_on_neg_dR", b_comp_neg,
             lambda x: "Dr.GRPO compresses less on reward-down steps"
                       if x["mean_diff"] < 0 else "inconclusive"),
        ):
            summary_rows.append({
                "experiment": exp,
                "metric": label,
                "n_pairs": bres["n_pairs"],
                "mean_grpo": round(bres["mean_grpo"], 6) if bres["mean_grpo"] is not None else "",
                "mean_drgrpo": round(bres["mean_drgrpo"], 6) if bres["mean_drgrpo"] is not None else "",
                "mean_diff": round(bres["mean_diff"], 6),
                "ci_lo": round(bres["ci_lo"], 6),
                "ci_hi": round(bres["ci_hi"], 6),
                "p_le0": round(bres["p_le0"], 4),
                "interpretation": interp_fn(bres),
            })

    write_tsv(RES / "length_bias_iter64_per_run.tsv", per_run_rows,
              fieldnames=["experiment", "algo", "seed", "cell", "mean_dL", "n_steps"])
    write_tsv(RES / "length_bias_iter64_paired.tsv", paired_rows,
              fieldnames=["experiment", "cell", "n_pairs", "mean_grpo",
                          "mean_drgrpo", "mean_diff", "ci_lo", "ci_hi", "p_le0"])
    write_tsv(RES / "length_bias_iter64_summary.tsv", summary_rows,
              fieldnames=["experiment", "metric", "n_pairs", "mean_grpo",
                          "mean_drgrpo", "mean_diff", "ci_lo", "ci_hi",
                          "p_le0", "interpretation"])

    print("=== iter64 CLRRD summary ===")
    for r in summary_rows:
        print(r)
    print("\n=== paired (first 12 rows) ===")
    for r in paired_rows[:12]:
        print(r)


if __name__ == "__main__":
    main()