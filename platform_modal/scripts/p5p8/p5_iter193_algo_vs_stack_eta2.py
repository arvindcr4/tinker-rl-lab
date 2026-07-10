#!/usr/bin/env python3
"""P5 iter-193 — Algorithm-axis vs Stack-axis variance decomposition.

"Report the Stack, Not the Label": quantify how much outcome variance the
algorithm LABEL explains (4 GRPO-family methods, same stack, N2 corpus) vs how
much the STACK axes explain (G / temperature / task_slice / model / seed, mega
corpus). Shared outcome channels measured on both corpora: zvf, reward, length.

Reuses the Berkeley Ivison-et-al. factorization recipe
(platform_modal/scripts/berkeley/unpacking_dpo_ppo_factorization.py: axis_variance_fraction),
adds unbiased omega^2 and stratified bootstrap CIs on every eta^2.

Fresh vs iter-189 (which only computed stack-axis eta^2 on mega): iter-193 adds
the ALGORITHM axis from the N2 four-method same-stack corpus and reports the
label-to-stack variance ratio side-by-side.

stdlib only. Outputs -> platform_hybrid/experiments/results/p5p8/p5_iter193_*.tsv|json
"""
from __future__ import annotations
import csv, json, math, os, random
from collections import defaultdict
from statistics import fmean

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "p5p8")
os.makedirs(OUT, exist_ok=True)
N2 = os.path.join(RES, "n2_reward_tensor_resume", "n2_metrics.tsv")
MEGA = os.path.join(RES, "mega_20260704", "cells.tsv")
B = 2000
SEED = 20260706
CI = 0.95


# ----------------- variance decomposition -----------------
def eta2_omega2(groups):
    """groups: dict[label -> list[float]]. Returns (eta2, omega2, k, N)."""
    grand = [v for vs in groups.values() for v in vs]
    N = len(grand)
    k = len(groups)
    if N < 2 or k < 1:
        return float("nan"), float("nan"), k, N
    gm = fmean(grand)
    ss_total = sum((x - gm) ** 2 for x in grand)
    ss_axis = sum(len(vs) * (fmean(vs) - gm) ** 2 for vs in groups.values())
    ss_within = ss_total - ss_axis
    eta2 = ss_axis / ss_total if ss_total > 1e-12 else float("nan")
    # omega^2 (unbiased): (SS_axis - (k-1) MS_within) / (SS_total + MS_within)
    df_within = N - k
    if df_within <= 0 or ss_total <= 1e-12:
        omega2 = float("nan")
    else:
        ms_within = ss_within / df_within
        omega2 = (ss_axis - (k - 1) * ms_within) / (ss_total + ms_within)
    return eta2, omega2, k, N


def boot_ci_eta2(groups, rng):
    """Stratified bootstrap: resample replicates within each level."""
    labels = list(groups.keys())
    ests = []
    for _ in range(B):
        bs = {}
        for lab in labels:
            vs = groups[lab]
            n = len(vs)
            if n == 0:
                continue
            bs[lab] = [vs[rng.randrange(n)] for _ in range(n)]
        e, _, _, _ = eta2_omega2(bs)
        if e == e:  # not nan
            ests.append(e)
    if not ests:
        return float("nan"), float("nan")
    ests.sort()
    lo = ests[int((1 - CI) / 2 * len(ests))]
    hi = ests[int((1 + CI) / 2 * len(ests)) - 1]
    return lo, hi


# ----------------- load N2 (algorithm axis) -----------------
def load_n2():
    rows = []
    with open(N2) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    # metric aliases -> canonical channel name
    chan = {"zvf": "zvf", "reward": "reward_mean", "len": "mean_len"}
    out = {}
    for canon, col in chan.items():
        g = defaultdict(list)
        for r in rows:
            try:
                v = float(r[col])
            except (ValueError, KeyError):
                continue
            if v != v:
                continue
            g[r["method"]].append(v)
        out[canon] = dict(g)
    n_methods = len({r["method"] for r in rows})
    n_steps = len({(r["method"], r["step"]) for r in rows}) // max(n_methods, 1)
    return out, n_methods, n_steps


# ----------------- load mega (stack axes) -----------------
def load_mega():
    rows = []
    with open(MEGA) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    chan = {"zvf": "zvf", "reward": "mean_reward", "len": "mean_completion_len"}
    factors = ["model_family", "task_slice", "G", "temperature", "seed"]
    # channel -> factor -> {level -> [vals]}
    out = {}
    for canon, col in chan.items():
        out[canon] = {}
        for fac in factors:
            g = defaultdict(list)
            for r in rows:
                try:
                    v = float(r[col])
                except (ValueError, KeyError):
                    continue
                if v != v:
                    continue
                g[r[fac]].append(v)
            out[canon][fac] = dict(g)
    return out, len(rows), factors


# ----------------- main -----------------
def main():
    rng = random.Random(SEED)
    n2, n_methods, n_steps = load_n2()
    mega, n_cells, factors = load_mega()
    channels = ["zvf", "reward", "len"]

    rows = []  # per (corpus, axis, channel)
    # Algorithm axis (N2, same stack)
    for ch in channels:
        g = n2[ch]
        e, w, k, N = eta2_omega2(g)
        lo, hi = boot_ci_eta2(g, rng)
        rows.append(dict(corpus="n2_samestack", axis="algorithm", channel=ch,
                         eta2=e, omega2=w, ci_lo=lo, ci_hi=hi, k_levels=k, n_obs=N))
    # Stack axes (mega)
    for ch in channels:
        for fac in factors:
            g = mega[ch][fac]
            e, w, k, N = eta2_omega2(g)
            lo, hi = boot_ci_eta2(g, rng)
            rows.append(dict(corpus="mega", axis=fac, channel=ch,
                             eta2=e, omega2=w, ci_lo=lo, ci_hi=hi, k_levels=k, n_obs=N))

    # write per-axis TSV
    with open(os.path.join(OUT, "p5_iter193_axis_eta2.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus", "axis", "channel", "eta2",
                           "omega2", "ci_lo", "ci_hi", "k_levels", "n_obs"],
                           delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ratio table: per channel, max stack eta2 vs algorithm eta2
    ratio_rows = []
    for ch in channels:
        alg = next(r for r in rows if r["axis"] == "algorithm" and r["channel"] == ch)
        stack = [r for r in rows if r["corpus"] == "mega" and r["channel"] == ch]
        best = max(stack, key=lambda r: r["eta2"] if r["eta2"] == r["eta2"] else -1)
        ratio = (best["eta2"] / alg["eta2"]) if alg["eta2"] > 1e-9 else float("inf")
        ratio_rows.append(dict(channel=ch, algo_eta2=alg["eta2"],
                               algo_ci_lo=alg["ci_lo"], algo_ci_hi=alg["ci_hi"],
                               top_stack_axis=best["axis"], top_stack_eta2=best["eta2"],
                               top_stack_ci_lo=best["ci_lo"], top_stack_ci_hi=best["ci_hi"],
                               stack_to_label_ratio=ratio,
                               ci_disjoint=bool(alg["ci_hi"] < best["ci_lo"])))
    with open(os.path.join(OUT, "p5_iter193_ratio.tsv"), "w", newline="") as f:
        cols = ["channel", "algo_eta2", "algo_ci_lo", "algo_ci_hi", "top_stack_axis",
                "top_stack_eta2", "top_stack_ci_lo", "top_stack_ci_hi",
                "stack_to_label_ratio", "ci_disjoint"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in ratio_rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ---- hypotheses ----
    verdicts = {}
    # H1: algorithm-axis eta2 < 0.10 on every channel (label explains <10%)
    verdicts["H1_algo_eta2_lt_0.10_all"] = all(
        r["eta2"] < 0.10 for r in rows if r["axis"] == "algorithm")
    # H2: for zvf & reward, top stack axis eta2 CI strictly above algorithm CI (disjoint)
    verdicts["H2_stack_ci_above_algo_zvf_reward"] = all(
        rr["ci_disjoint"] for rr in ratio_rows if rr["channel"] in ("zvf", "reward"))
    # H3: G is the single dominant stack axis for zvf (highest eta2 among stack)
    zvf_stack = [r for r in rows if r["corpus"] == "mega" and r["channel"] == "zvf"]
    verdicts["H3_G_top_for_zvf"] = (max(zvf_stack, key=lambda r: r["eta2"])["axis"] == "G")
    # H4: seed axis eta2 < 0.10 on every channel (seed noise is small vs stack)
    verdicts["H4_seed_eta2_lt_0.10_all"] = all(
        r["eta2"] < 0.10 for r in rows if r["axis"] == "seed")
    # H5: stack-to-label ratio > 3x on zvf and reward
    verdicts["H5_ratio_gt_3x_zvf_reward"] = all(
        rr["stack_to_label_ratio"] > 3.0 for rr in ratio_rows
        if rr["channel"] in ("zvf", "reward"))

    summary = dict(
        n2_methods=n_methods, n2_steps_per_method=n_steps, mega_cells=n_cells,
        stack_factors=factors, channels=channels,
        algorithm_axis_eta2={r["channel"]: r["eta2"] for r in rows if r["axis"] == "algorithm"},
        ratio_table=ratio_rows, verdicts=verdicts,
        n_pass=sum(verdicts.values()), n_total=len(verdicts),
        params=dict(B=B, seed=SEED, ci=CI))
    with open(os.path.join(OUT, "p5_iter193_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"N2: {n_methods} methods x ~{n_steps} steps | mega: {n_cells} cells")
    print("\n=== eta^2 by axis ===")
    for r in rows:
        print(f"{r['corpus']:13s} {r['axis']:12s} {r['channel']:7s} "
              f"eta2={r['eta2']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
              f"omega2={r['omega2']:.4f} k={r['k_levels']} N={r['n_obs']}")
    print("\n=== stack-to-label ratio ===")
    for rr in ratio_rows:
        print(f"{rr['channel']:7s} algo={rr['algo_eta2']:.4f} "
              f"top={rr['top_stack_axis']}({rr['top_stack_eta2']:.4f}) "
              f"ratio={rr['stack_to_label_ratio']:.1f}x disjoint={rr['ci_disjoint']}")
    print("\n=== verdicts ===")
    for h, v in verdicts.items():
        print(f"  {'PASS' if v else 'FAIL'}  {h}")
    print(f"\n{summary['n_pass']}/{summary['n_total']} PASS")


if __name__ == "__main__":
    main()
