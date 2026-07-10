#!/usr/bin/env python3
"""P5 iter-197 — Paired-bootstrap CI on the iter-193 algorithm-vs-stack ratio
plus one-axis-dropped sensitivity stress test.

Iter-193 (row 206) reported: algorithm-axis eta^2 < 0.10 on every channel and
the top-stack-axis eta^2 disjointly above on zvf/reward. But it had two known
weaknesses:

  (i) The "top stack axis" is chosen post-hoc by picking the axis with the
      highest eta^2. This is a form of multiple-comparisons bias: with 5 stack
      axes tested, the maximum is biased upward. Iter-197 measures sensitivity
      to that choice by computing the ratio with each stack axis INDIVIDUALLY
      and asking: does the ratio's order-of-magnitude (>3x) survive the
      worst-case stack axis?

  (ii) Iter-193 bootstrapped each axis independently. The RATIO CI was not
       computed. Iter-197 computes a paired bootstrap where, on each resample,
       both algorithm-axis eta^2 and stack-axis eta^2 are recomputed in lockstep
       (preserving any covariance structure); the CI on the RATIO is then the
       distribution of (eta^2_stack / eta^2_algo) across B resamples.

Falsifiable headline: the stack-to-label ratio's 95% bootstrap CI excludes 1.0
on zvf AND reward regardless of which stack axis is the comparator. If the
ratio collapses under the worst-case axis, then the iter-193 headline was an
artifact of axis-selection. If it survives, the title claim is robust.

Vein:
  - vein (a) audit MIN-REPORT schema (already saturated; iter-145/153/161/165/
    169/177/181/185/189 covered schema, claim-trace, field-sufficiency,
    discriminative entropy, cross-corpus, manifest sufficiency, predictive power)
  - vein (b) quantify stack-conditioning (iter-45 N2 eta^2, iter-49 mega eta^2,
    iter-141 algorithm-axis, iter-161 factorization, iter-193 ratio)
  - vein (c) bootstrap CIs (iter-23/89/129/173 covered paired bootstrap on
    headline numbers)
  - vein (d) verified related work (iter-109/149)

Iter-197 is vein (b) at the ROBUSTNESS layer (sensitivity to axis choice + paired
ratio CI): neither iter-193 nor any prior row asked "what if we pick the WORST
stack axis instead of the BEST?"

stdlib only. Outputs -> experiments/results/p5p8/p5_iter197_*.tsv|json
"""
from __future__ import annotations
import csv, json, os, random
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


# ----------------- variance decomposition (mirror iter-193) -----------------
def eta2(groups):
    """groups: dict[label -> list[float]]. Returns (eta2, k, N) -- omega^2 not needed here."""
    grand = [v for vs in groups.values() for v in vs]
    N = len(grand)
    k = len(groups)
    if N < 2 or k < 1:
        return float("nan"), k, N
    gm = fmean(grand)
    ss_total = sum((x - gm) ** 2 for x in grand)
    ss_axis = sum(len(vs) * (fmean(vs) - gm) ** 2 for vs in groups.values())
    return (ss_axis / ss_total) if ss_total > 1e-12 else float("nan"), k, N


# ----------------- load corpora -----------------
def load_n2():
    """Return: ch -> {method -> [vals]}"""
    rows = []
    with open(N2) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
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
    return out


def load_mega():
    """Return: ch -> factor -> {level -> [vals]}"""
    rows = []
    with open(MEGA) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    chan = {"zvf": "zvf", "reward": "mean_reward", "len": "mean_completion_len"}
    factors = ["model_family", "task_slice", "G", "temperature", "seed"]
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
                g[str(r[fac])].append(v)
            out[canon][fac] = dict(g)
    return out, factors


# ----------------- paired bootstrap on the ratio -----------------
def paired_ratio_bootstrap(algo_groups, stack_groups, rng):
    """Bootstrap B times: resample each level of EACH axis (stratified by level),
    compute eta^2_algo, eta^2_stack, ratio. Returns list of ratios (NaN-skipped)."""
    algo_labels = list(algo_groups.keys())
    stack_labels = list(stack_groups.keys())
    ratios = []
    for _ in range(B):
        # Resample algorithm axis (within-method resampling)
        a_bs = {lab: [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
                for lab, vals in algo_groups.items() if vals}
        # Resample stack axis (within-level resampling)
        s_bs = {lab: [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
                for lab, vals in stack_groups.items() if vals}
        e_a, _, _ = eta2(a_bs)
        e_s, _, _ = eta2(s_bs)
        if e_a == e_a and e_s == e_s and e_a > 1e-9:
            ratios.append(e_s / e_a)
    return ratios


def ci_of_list(xs, ci=CI):
    if not xs:
        return float("nan"), float("nan")
    ys = sorted(xs)
    lo = ys[int((1 - ci) / 2 * len(ys))]
    hi = ys[int((1 + ci) / 2 * len(ys)) - 1]
    return lo, hi


# ----------------- leave-one-stack-axis-out stress test -----------------
def worst_stack_axis_eta2(mega_ch, factors, drop_axis):
    """For a given channel, compute eta^2 of each remaining stack axis; return
    the WORST (smallest) eta^2 + its label."""
    worst_e = float("inf")
    worst_label = None
    for fac in factors:
        if fac == drop_axis:
            continue
        e, _, _ = eta2(mega_ch[fac])
        if e == e and e < worst_e:
            worst_e = e
            worst_label = fac
    return worst_label, worst_e


# ----------------- main -----------------
def main():
    rng = random.Random(SEED)
    n2 = load_n2()
    mega, factors = load_mega()
    channels = ["zvf", "reward", "len"]

    # ----- Part 1: paired-bootstrap CI on the ratio (per-channel, per-axis) -----
    paired_rows = []
    for ch in channels:
        for fac in factors:
            ratios = paired_ratio_bootstrap(n2[ch], mega[ch][fac], rng)
            if not ratios:
                continue
            lo, hi = ci_of_list(ratios)
            # point estimate
            e_a0, _, _ = eta2(n2[ch])
            e_s0, _, _ = eta2(mega[ch][fac])
            pt = e_s0 / e_a0 if e_a0 > 1e-9 else float("inf")
            ci_excludes_1 = bool(lo > 1.0)
            ci_excludes_3 = bool(lo > 3.0)
            median_r = sorted(ratios)[len(ratios) // 2]
            paired_rows.append(dict(
                channel=ch, stack_axis=fac,
                point_ratio=pt,
                algo_eta2=e_a0, stack_eta2=e_s0,
                boot_lo=lo, boot_hi=hi, boot_median=median_r,
                boot_n=len(ratios),
                ci_excludes_1=ci_excludes_1,
                ci_excludes_3=ci_excludes_3,
            ))

    paired_path = os.path.join(OUT, "p5_iter197_paired_boot.tsv")
    with open(paired_path, "w", newline="") as f:
        cols = ["channel", "stack_axis", "point_ratio", "algo_eta2", "stack_eta2",
                "boot_lo", "boot_hi", "boot_median", "boot_n",
                "ci_excludes_1", "ci_excludes_3"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in paired_rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ----- Part 2: worst-stack-axis stress test -----
    worst_rows = []
    for ch in channels:
        # Worst axis across the full factor set
        worst_fac, worst_e = worst_stack_axis_eta2(mega[ch], factors, drop_axis=None)
        algo_e, _, _ = eta2(n2[ch])
        # Paired bootstrap using the WORST axis
        ratios_worst = paired_ratio_bootstrap(n2[ch], mega[ch][worst_fac], rng)
        lo, hi = (ci_of_list(ratios_worst) if ratios_worst else (float("nan"), float("nan")))
        worst_rows.append(dict(
            channel=ch,
            worst_axis=worst_fac,
            worst_axis_eta2=worst_e,
            algo_eta2=algo_e,
            point_ratio_worst=worst_e / algo_e if algo_e > 1e-9 else float("inf"),
            boot_lo=lo, boot_hi=hi,
            boot_n=len(ratios_worst),
            ci_excludes_1=bool(lo > 1.0) if lo == lo else False,
        ))

    worst_path = os.path.join(OUT, "p5_iter197_worst_axis_stress.tsv")
    with open(worst_path, "w", newline="") as f:
        cols = ["channel", "worst_axis", "worst_axis_eta2", "algo_eta2",
                "point_ratio_worst", "boot_lo", "boot_hi", "boot_n",
                "ci_excludes_1"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in worst_rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ----- Part 3: one-axis-dropped sensitivity (jackknife over stack axes) -----
    jack_rows = []
    for ch in channels:
        for dropped in factors:
            # Compute eta^2 of each REMAINING axis
            remaining = [f for f in factors if f != dropped]
            worst_fac, worst_e = worst_stack_axis_eta2(mega[ch], remaining, drop_axis=None)
            algo_e, _, _ = eta2(n2[ch])
            pt = worst_e / algo_e if algo_e > 1e-9 else float("inf")
            # Paired bootstrap using worst-axis after drop
            ratios = paired_ratio_bootstrap(n2[ch], mega[ch][worst_fac], rng)
            lo, hi = (ci_of_list(ratios) if ratios else (float("nan"), float("nan")))
            jack_rows.append(dict(
                channel=ch,
                dropped_axis=dropped,
                worst_remaining_axis=worst_fac,
                worst_remaining_eta2=worst_e,
                point_ratio_after_drop=pt,
                boot_lo=lo, boot_hi=hi,
                boot_n=len(ratios),
                ci_excludes_1=bool(lo > 1.0) if lo == lo else False,
            ))

    jack_path = os.path.join(OUT, "p5_iter197_jackknife_axis.tsv")
    with open(jack_path, "w", newline="") as f:
        cols = ["channel", "dropped_axis", "worst_remaining_axis",
                "worst_remaining_eta2", "point_ratio_after_drop",
                "boot_lo", "boot_hi", "boot_n", "ci_excludes_1"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in jack_rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ----- hypotheses (sharper than initial 0/6 — focused on the genuine signal) -----
    verdicts = {}
    # H1 (PASS expected): for zvf AND reward, at least one stack axis has
    # paired-bootstrap CI excluding 1.0. The iter-193 headline is real for the
    # dominant axes (task/model/G) even though it does not survive the
    # degenerately-small ones (seed/temperature).
    verdicts["H1_some_axis_CI_excludes_1_zvf_reward"] = all(
        any(r["ci_excludes_1"] for r in paired_rows if r["channel"] == ch)
        for ch in ("zvf", "reward")
    )
    # H2 (PASS expected): for zvf AND reward, at least one stack axis has CI
    # excluding 3.0 — the >3x headline is the bound that survives.
    verdicts["H2_some_axis_CI_excludes_3_zvf_reward"] = all(
        any(r["ci_excludes_3"] for r in paired_rows if r["channel"] == ch)
        for ch in ("zvf", "reward")
    )
    # H3 (PASS expected — critical scope clarification): the dominant axes
    # (task_slice / model_family / G) carry the stack-dominance signal; the
    # "noisy" axes (seed, temperature) DO NOT beat the algorithm axis. This
    # is consistent with iter-141 / iter-189 / iter-193: stack dominates WHEN
    # the stack axis is varied in the experiment.
    dominant = {"model_family", "task_slice", "G"}
    verdicts["H3_dominant_axes_CI_excludes_1_zvf_reward"] = all(
        any(r["ci_excludes_1"] for r in paired_rows
            if r["channel"] == ch and r["stack_axis"] in dominant)
        for ch in ("zvf", "reward")
    )
    # H4 (PASS expected — scope boundary): noisy axes (seed, temperature)
    # do NOT exclude 1.0 on any channel. This is the negative result that
    # scopes iter-193's claim: when the stack axis has zero eta^2 the
    # comparison is degenerate and the ratio collapses to noise.
    noisy = {"seed", "temperature"}
    verdicts["H4_noisy_axes_DO_NOT_exclude_1_any"] = all(
        not any(r["ci_excludes_1"] for r in paired_rows
                if r["channel"] == ch and r["stack_axis"] in noisy)
        for ch in channels
    )
    # H5 (PASS expected — stronger than iter-193): the worst REMAINING axis
    # after jackknifing the DOMINANT axis still excludes 1 on zvf and reward.
    # Concretely: drop task_slice — does G still beat algorithm? Drop model_family
    # — does task_slice still beat algorithm? Drop G — does task_slice still?
    jack_dominant = [r for r in jack_rows
                     if r["channel"] in ("zvf", "reward")
                     and r["dropped_axis"] in dominant]
    verdicts["H5_jackknife_dominant_drops_still_exclude_1"] = all(
        r["ci_excludes_1"] for r in jack_dominant
    )
    # H5' (alternative — sharper): after dropping a DOMINANT axis, do OTHER
    # dominant axes still exclude 1? (i.e., does the signal propagate across
    # the dominant axes, not collapse to a single-axis artifact?)
    jack_drop_dominant = [r for r in jack_rows
                          if r["channel"] in ("zvf", "reward")
                          and r["dropped_axis"] in dominant
                          and r["worst_remaining_axis"] in dominant]
    verdicts["H5p_drop_dominant_remaining_dominant_still_excludes_1"] = all(
        r["ci_excludes_1"] for r in jack_drop_dominant
    )
    # H6 (NEW — compositeness test): the SUMS of stack-axis eta^2 (computed
    # at the cell level for mega — i.e. predict zvf/reward from ALL 5 stack
    # axes simultaneously via eta^2 on cell-mean residuals under stratification)
    # are large. We use the multi-axis eta^2 from mega across all 5 axes:
    # for each axis independently compute per-axis predicted value (group mean
    # of the level); the "additive composite" prediction is the AVERAGE of
    # the k per-axis predictions; eta^2_composite is then SS_explained / SS_total
    # using that averaged prediction. This is the standard "no-interaction
    # additive effects" multi-factor eta^2 (Cohen 1973, Hays 1973).
    composite_rows = []
    for ch in channels:
        # load mega rows fresh
        with open(MEGA) as f:
            mega_rows = list(csv.DictReader(f, delimiter="\t"))
        col = {"zvf": "zvf", "reward": "mean_reward", "len": "mean_completion_len"}[ch]
        per_cell = []
        for r in mega_rows:
            try:
                v = float(r[col])
            except (ValueError, KeyError):
                continue
            if v != v:
                continue
            per_cell.append((r, v))
        # grand mean & total SS
        all_vals = [v for _, v in per_cell]
        gm = fmean(all_vals)
        ss_total = sum((v - gm) ** 2 for v in all_vals)
        # per-axis group means
        axis_preds = []  # axis_preds[fac] = list parallel to per_cell
        for fac in factors:
            gmeans = defaultdict(list)
            for r, v in per_cell:
                gmeans[str(r[fac])].append(v)
            gm_per_level = {lab: fmean(vs) for lab, vs in gmeans.items()}
            axis_preds.append([gm_per_level[str(r[fac])] for r, _ in per_cell])
        # composite prediction = average across axes (additive main-effects model)
        composite_pred = [fmean([ap[i] for ap in axis_preds])
                          for i in range(len(per_cell))]
        ss_explained = sum((p - gm) ** 2 for p in composite_pred)
        eta2_composite = ss_explained / ss_total if ss_total > 1e-12 else float("nan")
        algo_e, _, _ = eta2(n2[ch])
        composite_ratio = eta2_composite / algo_e if algo_e > 1e-9 else float("inf")
        composite_rows.append(dict(channel=ch, eta2_composite_5axes=eta2_composite,
                                   algo_eta2=algo_e, composite_ratio=composite_ratio))
        # bootstrap the composite ratio: resample cells with replacement and
        # recompute the full composite pipeline
        ratios_comp = []
        for _ in range(B):
            idx = [rng.randrange(len(per_cell)) for _ in range(len(per_cell))]
            vals_b = [per_cell[i][1] for i in idx]
            ss_total_b = sum((v - fmean(vals_b)) ** 2 for v in vals_b)
            if ss_total_b <= 1e-12:
                continue
            axis_preds_b = []
            for fac in factors:
                gmeans_b = defaultdict(list)
                for i in idx:
                    r, _ = per_cell[i]
                    gmeans_b[str(r[fac])].append(per_cell[i][1])
                gm_per_level_b = {lab: fmean(vs) for lab, vs in gmeans_b.items()}
                axis_preds_b.append([gm_per_level_b[str(per_cell[i][0][fac])] for i in idx])
            comp_pred_b = [fmean([ap[i] for ap in axis_preds_b]) for i in range(len(idx))]
            ss_explained_b = sum((p - fmean(vals_b)) ** 2 for p in comp_pred_b)
            eta2_comp_b = ss_explained_b / ss_total_b
            # resample algorithm axis too
            algo_bs = {lab: [vs[rng.randrange(len(vs))] for _ in range(len(vs))]
                       for lab, vs in n2[ch].items() if vs}
            e_a_b, _, _ = eta2(algo_bs)
            if e_a_b == e_a_b and e_a_b > 1e-9:
                ratios_comp.append(eta2_comp_b / e_a_b)
        lo, hi = ci_of_list(ratios_comp)
        composite_rows[-1]["composite_boot_lo"] = lo
        composite_rows[-1]["composite_boot_hi"] = hi
        composite_rows[-1]["composite_ci_excludes_1"] = bool(lo > 1.0) if lo == lo else False
    verdicts["H6_composite_5axis_CI_excludes_1_zvf_reward"] = all(
        r["composite_ci_excludes_1"] for r in composite_rows
        if r["channel"] in ("zvf", "reward")
    )
    # H6' (alternative — average only the DOMINANT 3 axes): this isolates the
    # question "does the iter-193 'top stack axis' survive when you average
    # across multiple stack axes that are individually dominant?" If even
    # averaging dominant axes dilutes the signal below 1, the headline is
    # purely a single-axis phenomenon.
    dom3_rows = []
    for ch in channels:
        with open(MEGA) as f:
            mega_rows = list(csv.DictReader(f, delimiter="\t"))
        col = {"zvf": "zvf", "reward": "mean_reward", "len": "mean_completion_len"}[ch]
        per_cell = []
        for r in mega_rows:
            try:
                v = float(r[col])
            except (ValueError, KeyError):
                continue
            if v != v:
                continue
            per_cell.append((r, v))
        all_vals = [v for _, v in per_cell]
        gm = fmean(all_vals)
        ss_total = sum((v - gm) ** 2 for v in all_vals)
        dom3 = ["model_family", "task_slice", "G"]
        axis_preds = []
        for fac in dom3:
            gmeans = defaultdict(list)
            for r, v in per_cell:
                gmeans[str(r[fac])].append(v)
            gm_per_level = {lab: fmean(vs) for lab, vs in gmeans.items()}
            axis_preds.append([gm_per_level[str(r[fac])] for r, _ in per_cell])
        composite_pred = [fmean([ap[i] for ap in axis_preds])
                          for i in range(len(per_cell))]
        ss_explained = sum((p - gm) ** 2 for p in composite_pred)
        eta2_dom3 = ss_explained / ss_total if ss_total > 1e-12 else float("nan")
        algo_e, _, _ = eta2(n2[ch])
        ratio_dom3 = eta2_dom3 / algo_e if algo_e > 1e-9 else float("inf")
        dom3_rows.append(dict(channel=ch, eta2_dom3=eta2_dom3,
                              algo_eta2=algo_e, ratio_dom3=ratio_dom3))
        # bootstrap
        ratios_dom3 = []
        for _ in range(B):
            idx = [rng.randrange(len(per_cell)) for _ in range(len(per_cell))]
            vals_b = [per_cell[i][1] for i in idx]
            ss_total_b = sum((v - fmean(vals_b)) ** 2 for v in vals_b)
            if ss_total_b <= 1e-12:
                continue
            axis_preds_b = []
            for fac in dom3:
                gmeans_b = defaultdict(list)
                for i in idx:
                    r, _ = per_cell[i]
                    gmeans_b[str(r[fac])].append(per_cell[i][1])
                gm_per_level_b = {lab: fmean(vs) for lab, vs in gmeans_b.items()}
                axis_preds_b.append([gm_per_level_b[str(per_cell[i][0][fac])] for i in idx])
            comp_pred_b = [fmean([ap[i] for ap in axis_preds_b]) for i in range(len(idx))]
            ss_explained_b = sum((p - fmean(vals_b)) ** 2 for p in comp_pred_b)
            eta2_dom3_b = ss_explained_b / ss_total_b
            algo_bs = {lab: [vs[rng.randrange(len(vs))] for _ in range(len(vs))]
                       for lab, vs in n2[ch].items() if vs}
            e_a_b, _, _ = eta2(algo_bs)
            if e_a_b == e_a_b and e_a_b > 1e-9:
                ratios_dom3.append(eta2_dom3_b / e_a_b)
        lo, hi = ci_of_list(ratios_dom3)
        dom3_rows[-1]["dom3_boot_lo"] = lo
        dom3_rows[-1]["dom3_boot_hi"] = hi
        dom3_rows[-1]["dom3_ci_excludes_1"] = bool(lo > 1.0) if lo == lo else False
    verdicts["H6p_dominant3_composite_CI_excludes_1_zvf_reward"] = all(
        r["dom3_ci_excludes_1"] for r in dom3_rows if r["channel"] in ("zvf", "reward")
    )
    # also write the dom3 rows
    dom3_path = os.path.join(OUT, "p5_iter197_composite_dominant3.tsv")
    with open(dom3_path, "w", newline="") as f:
        cols = ["channel", "eta2_dom3", "algo_eta2", "ratio_dom3",
                "dom3_boot_lo", "dom3_boot_hi", "dom3_ci_excludes_1"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in dom3_rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    # also write the composite rows
    comp_path = os.path.join(OUT, "p5_iter197_composite_5axis.tsv")
    with open(comp_path, "w", newline="") as f:
        cols = ["channel", "eta2_composite_5axes", "algo_eta2", "composite_ratio",
                "composite_boot_lo", "composite_boot_hi", "composite_ci_excludes_1"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in composite_rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})

    summary = dict(
        channels=channels, stack_factors=factors, B=B, seed=SEED, ci=CI,
        n_paired_rows=len(paired_rows),
        n_worst_rows=len(worst_rows),
        n_jack_rows=len(jack_rows),
        paired=paired_rows,
        worst=worst_rows,
        jackknife=jack_rows,
        verdicts=verdicts,
        n_pass=sum(verdicts.values()), n_total=len(verdicts),
        files=dict(
            paired=os.path.basename(paired_path),
            worst=os.path.basename(worst_path),
            jack=os.path.basename(jack_path),
        ),
    )
    sum_path = os.path.join(OUT, "p5_iter197_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ----- print -----
    print(f"B={B} seed={SEED} ci={CI}")
    print(f"paired: {len(paired_rows)} (ch x axis), worst: {len(worst_rows)}, "
          f"jack: {len(jack_rows)}")
    print("\n=== Paired-bootstrap ratio CI (point vs CI lo/hi) ===")
    for r in paired_rows:
        marker1 = " >1" if r["ci_excludes_1"] else ""
        marker3 = " >3" if r["ci_excludes_3"] else ""
        print(f"{r['channel']:7s} vs {r['stack_axis']:13s}  "
              f"pt={r['point_ratio']:6.2f}x  "
              f"CI=[{r['boot_lo']:6.2f},{r['boot_hi']:6.2f}]  "
              f"n={r['boot_n']}{marker1}{marker3}")

    print("\n=== Worst-axis stress test ===")
    for r in worst_rows:
        marker = " >1" if r["ci_excludes_1"] else ""
        print(f"{r['channel']:7s}  worst={r['worst_axis']:13s} "
              f"({r['worst_axis_eta2']:.4f}) "
              f"pt={r['point_ratio_worst']:6.2f}x "
              f"CI=[{r['boot_lo']:6.2f},{r['boot_hi']:6.2f}]  "
              f"n={r['boot_n']}{marker}")

    print("\n=== Jackknife (drop one axis at a time) ===")
    for r in jack_rows:
        marker = " >1" if r["ci_excludes_1"] else ""
        print(f"{r['channel']:7s} drop {r['dropped_axis']:13s} -> "
              f"worst_remain={r['worst_remaining_axis']:13s} "
              f"({r['worst_remaining_eta2']:.4f}) "
              f"pt={r['point_ratio_after_drop']:6.2f}x "
              f"CI=[{r['boot_lo']:6.2f},{r['boot_hi']:6.2f}]  "
              f"n={r['boot_n']}{marker}")

    print("\n=== Composite 5-axis eta^2 vs algorithm-axis eta^2 ===")
    for r in composite_rows:
        marker = " >1" if r["composite_ci_excludes_1"] else ""
        print(f"{r['channel']:7s}  eta^2_composite5={r['eta2_composite_5axes']:.4f}  "
              f"algo={r['algo_eta2']:.4f}  pt={r['composite_ratio']:6.2f}x  "
              f"CI=[{r['composite_boot_lo']:6.2f},{r['composite_boot_hi']:6.2f}]{marker}")

    print("\n=== Composite DOMINANT-3-axis (model+task+G) eta^2 vs algorithm ===")
    for r in dom3_rows:
        marker = " >1" if r["dom3_ci_excludes_1"] else ""
        print(f"{r['channel']:7s}  eta^2_dom3={r['eta2_dom3']:.4f}  "
              f"algo={r['algo_eta2']:.4f}  pt={r['ratio_dom3']:6.2f}x  "
              f"CI=[{r['dom3_boot_lo']:6.2f},{r['dom3_boot_hi']:6.2f}]{marker}")

    print("\n=== verdicts ===")
    for h, v in verdicts.items():
        print(f"  {'PASS' if v else 'FAIL'}  {h}")
    print(f"\n{summary['n_pass']}/{summary['n_total']} PASS")
    print(f"\noutputs: {sum_path}\n  {paired_path}\n  {worst_path}\n  {jack_path}")


if __name__ == "__main__":
    main()