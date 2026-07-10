#!/usr/bin/env python3
"""P7 iter-75: EXACT finite-pool (hypergeometric) contrast-preservation vs the
i.i.d. binomial model used by the iter-47 per-prompt G* allocator.

Motivation (frontier synthesis, Round 2): observed ZVF under-predicts the i.i.d.
baseline p^G+(1-p)^G because sampling *without replacement* from a finite rollout
pool anti-herds (delta_div>0). The iter-47 per-prompt optimal-G analysis
(p7_per_prompt_optimal_g) scored candidate G' with the i.i.d. binomial model
CP_binom(G'|p)=1-(p^G'+(1-p)^G'). But the honest counterfactual for "what if we
had only rolled out G' of the 8 samples" is to SUBSAMPLE G' of the *actual* 8
rewards -> the exact contrast-preservation probability is hypergeometric:

    CP_exact(G' | k, N) = 1 - [C(k,G') + C(N-k,G')] / C(N,G')

with C(k,G')=0 for k<G'. This is model-free (pure enumeration over the observed
pool). We show CP_exact >= CP_binom everywhere (finite-pool never collides more
than i.i.d.), quantify the per-prompt anti-herding bonus delta_ex = CP_exact-CP_binom,
and rebuild the per-prompt Iso-G allocator with the EXACT rule. The binomial
allocator leaves savings on the table: prompts it labels "starved at G'=2" are in
fact contrastive under exact accounting.

Data: platform_hybrid/experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl
      40 steps x 16 prompts x 8 rollouts (binary) per method.
Outputs: platform_hybrid/experiments/results/p5p8/p7_exact_finite_pool_{per_prompt,summary}.tsv,
         p7_exact_finite_pool_summary.json
Stdlib only. ~1-2 min on 4 cores (single-thread fine).
"""
import json, math, os, random
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NDIR = os.path.join(ROOT, "experiments", "results", "n2_reward_tensor_resume")
ODIR = os.path.join(ROOT, "experiments", "results", "p5p8")
METHODS = ["grpo", "aero", "gift", "areal"]
NBOOT = 2000
SEED = 20260705
TAU_C = [0.3, 0.5, 0.7]        # target within-pool contrast probability
GCAND = [2, 3, 4, 5, 6, 7]     # candidate reduced group sizes (< N=8)


def cp_exact(k, N, g):
    """Exact P(subsample of size g from a fixed pool of k ones / N-k zeros is
    contrastive), i.e. 1 - P(all-same). Hypergeometric, no model."""
    if g > N:
        return None
    denom = math.comb(N, g)
    same = (math.comb(k, g) if k >= g else 0) + (math.comb(N - k, g) if (N - k) >= g else 0)
    return 1.0 - same / denom


def cp_binom(k, N, g):
    """i.i.d. binomial model used by iter-47: 1 - (p^g + (1-p)^g), p=k/N."""
    p = k / N
    return 1.0 - (p ** g + (1.0 - p) ** g)


def brute_cp(rewards, g):
    """Ground-truth CP by enumerating all C(N,g) subsets of the actual rollouts."""
    N = len(rewards)
    tot = 0
    contr = 0
    for sub in combinations(rewards, g):
        s = sum(sub)
        tot += 1
        if 0 < s < g:
            contr += 1
    return contr / tot


def load(method):
    rows = []
    with open(os.path.join(NDIR, f"{method}_s0_tensors.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def alloc_gstar(k, N, tau, use_exact):
    """Smallest candidate G' (ascending) whose CP >= tau; else keep N (G_base).
    Returns (gstar, cp_at_gstar_exact, economized_bool)."""
    fn = cp_exact if use_exact else cp_binom
    for g in GCAND:
        if fn(k, N, g) >= tau:
            return g, cp_exact(k, N, g), True
    return N, cp_exact(k, N, N) if N in GCAND else 1.0, False


def main():
    random.seed(SEED)
    os.makedirs(ODIR, exist_ok=True)

    # ---- 0. validate exact formula against brute force on a few real prompts
    val_max_err = 0.0
    sample = load("grpo")[0]
    for row_rewards in sample["rewards"][:8]:
        rw = [int(x) for x in row_rewards]
        N = len(rw)
        k = sum(rw)
        for g in GCAND:
            bf = brute_cp(rw, g)
            ex = cp_exact(k, N, g)
            val_max_err = max(val_max_err, abs(bf - ex))
    assert val_max_err < 1e-9, f"exact formula mismatch vs brute force: {val_max_err}"

    # ---- 1. per-prompt table + per-method aggregation
    per_prompt_rows = []
    # per (method, tau): step -> dict of aggregates for bootstrap
    step_agg = {}  # (method,tau) -> list over steps of (roll_ex, roll_bin, presv_ex, presv_bin, N_prompts)
    delta_ex_all = []  # per-prompt CP_exact-CP_binom at the exact-chosen G' vs same g

    for method in METHODS:
        rows = load(method)
        for tau in TAU_C:
            step_agg[(method, tau)] = []
        for r in rows:
            step = r["step"]
            N = r["group_size"]
            per_step_tau = {tau: dict(roll_ex=0, roll_bin=0, presv_ex=0.0,
                                      presv_bin_actual=0.0, np=0, phantom=0) for tau in TAU_C}
            for pi, rw in enumerate(r["rewards"]):
                rw = [int(x) for x in rw]
                k = sum(rw)
                degenerate = (k == 0 or k == N)  # no G can create contrast
                for tau in TAU_C:
                    g_ex, cpx_ex, econ_ex = alloc_gstar(k, N, tau, use_exact=True)
                    g_bin, cpx_bin, econ_bin = alloc_gstar(k, N, tau, use_exact=False)
                    d = per_step_tau[tau]
                    d["np"] += 1
                    d["roll_ex"] += g_ex
                    d["roll_bin"] += g_bin
                    # preserved contrast = exact CP at the chosen G' (ground truth)
                    d["presv_ex"] += cp_exact(k, N, g_ex)
                    d["presv_bin_actual"] += cp_exact(k, N, g_bin)
                    # phantom = binom kept G=8 (thought starved) but exact would economize
                    if (not econ_bin) and econ_ex and (not degenerate):
                        d["phantom"] += 1
                    if tau == 0.5 and not degenerate:
                        # per-prompt delta at a common g=2 for the anti-herding table
                        delta_ex_all.append(cp_exact(k, N, 2) - cp_binom(k, N, 2))
                    per_prompt_rows.append((method, step, pi, k, tau, g_ex, g_bin,
                                            round(cp_exact(k, N, g_ex), 6),
                                            round(cp_exact(k, N, g_bin), 6),
                                            int(degenerate)))
            for tau in TAU_C:
                step_agg[(method, tau)].append(per_step_tau[tau])

    # ---- 2. per-method x tau summary with bootstrap CIs on savings gap
    summary_rows = []
    method_stats = {}
    for method in METHODS:
        method_stats[method] = {}
        for tau in TAU_C:
            steps = step_agg[(method, tau)]
            tot_ex = sum(s["roll_ex"] for s in steps)
            tot_bin = sum(s["roll_bin"] for s in steps)
            tot_g8 = sum(s["np"] * 8 for s in steps)
            presv_ex = sum(s["presv_ex"] for s in steps)
            presv_bin = sum(s["presv_bin_actual"] for s in steps)
            phantom = sum(s["phantom"] for s in steps)
            npr = sum(s["np"] for s in steps)
            # bootstrap over steps on (extra savings of exact vs binom) = roll_bin - roll_ex
            gaps = []
            for _ in range(NBOOT):
                samp = [steps[random.randrange(len(steps))] for _ in range(len(steps))]
                gb = sum(s["roll_bin"] - s["roll_ex"] for s in samp)
                gaps.append(gb)
            gaps.sort()
            lo = gaps[int(0.025 * NBOOT)]
            hi = gaps[int(0.975 * NBOOT)]
            extra_saves = tot_bin - tot_ex
            row = dict(method=method, tau=tau, n_prompts=npr,
                       rollouts_exact=tot_ex, rollouts_binom=tot_bin, rollouts_g8=tot_g8,
                       cost_ratio_exact=round(tot_ex / tot_g8, 4),
                       cost_ratio_binom=round(tot_bin / tot_g8, 4),
                       extra_saves_exact_vs_binom=extra_saves,
                       extra_saves_ci_lo=lo, extra_saves_ci_hi=hi,
                       preserved_contrast_exact=round(presv_ex, 3),
                       preserved_contrast_binom=round(presv_bin, 3),
                       phantom_starved_prompts=phantom)
            summary_rows.append(row)
            method_stats[method][str(tau)] = row

    # pooled delta_ex (anti-herding bonus at g=2)
    n_d = len(delta_ex_all)
    mean_d = sum(delta_ex_all) / n_d
    # bootstrap CI on mean delta
    md = []
    for _ in range(NBOOT):
        s = sum(delta_ex_all[random.randrange(n_d)] for _ in range(n_d)) / n_d
        md.append(s)
    md.sort()
    d_lo, d_hi = md[int(0.025 * NBOOT)], md[int(0.975 * NBOOT)]

    # ---- 3. write outputs
    with open(os.path.join(ODIR, "p7_exact_finite_pool_per_prompt.tsv"), "w") as f:
        f.write("method\tstep\tprompt\tk\ttau\tg_exact\tg_binom\tcp_at_g_exact\tcp_at_g_binom\tdegenerate\n")
        for r in per_prompt_rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    cols = ["method", "tau", "n_prompts", "rollouts_exact", "rollouts_binom", "rollouts_g8",
            "cost_ratio_exact", "cost_ratio_binom", "extra_saves_exact_vs_binom",
            "extra_saves_ci_lo", "extra_saves_ci_hi", "preserved_contrast_exact",
            "preserved_contrast_binom", "phantom_starved_prompts"]
    with open(os.path.join(ODIR, "p7_exact_finite_pool_summary.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in summary_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # pooled headline at tau=0.5
    pooled = {}
    for tau in TAU_C:
        te = sum(r["rollouts_exact"] for r in summary_rows if r["tau"] == tau)
        tb = sum(r["rollouts_binom"] for r in summary_rows if r["tau"] == tau)
        tg8 = sum(r["rollouts_g8"] for r in summary_rows if r["tau"] == tau)
        ph = sum(r["phantom_starved_prompts"] for r in summary_rows if r["tau"] == tau)
        pooled[str(tau)] = dict(rollouts_exact=te, rollouts_binom=tb, rollouts_g8=tg8,
                                cost_ratio_exact=round(te / tg8, 4),
                                cost_ratio_binom=round(tb / tg8, 4),
                                extra_saves_exact_vs_binom=tb - te,
                                pct_extra_saving=round(100 * (tb - te) / tg8, 3),
                                phantom_starved_prompts=ph)

    summary = dict(
        n_methods=len(METHODS), n_steps=40, n_prompts_per_step=16, N_pool=8,
        candidate_g=GCAND, tau_c=TAU_C, nboot=NBOOT, seed=SEED,
        exact_formula_max_err_vs_bruteforce=val_max_err,
        anti_herding_bonus_g2=dict(
            n_nondegenerate=n_d, mean_delta_exact_minus_binom=round(mean_d, 6),
            ci95=[round(d_lo, 6), round(d_hi, 6)],
            interpretation="CP_exact - CP_binom at G'=2, pooled over non-degenerate prompts; >0 confirms finite-pool anti-herding"),
        pooled=pooled, method_stats=method_stats)
    with open(os.path.join(ODIR, "p7_exact_finite_pool_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- console
    print(f"[validate] exact vs brute-force max err = {val_max_err:.2e} (must be ~0)")
    print(f"[anti-herding] mean(CP_exact-CP_binom) at G'=2 = {mean_d:+.4f} "
          f"[{d_lo:+.4f},{d_hi:+.4f}] over {n_d} non-degenerate prompts")
    for tau in TAU_C:
        p = pooled[str(tau)]
        print(f"[tau={tau}] cost_ratio exact={p['cost_ratio_exact']} binom={p['cost_ratio_binom']} "
              f"| extra saving {p['pct_extra_saving']}% ({p['extra_saves_exact_vs_binom']} rollouts) "
              f"| phantom-starved prompts recovered={p['phantom_starved_prompts']}")


if __name__ == "__main__":
    main()
