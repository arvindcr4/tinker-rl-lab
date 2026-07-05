"""Iter 187 — P7 Mutual-Information-Indexed Adaptive-G controller.

Fresh vein: per-prompt posterior entropy gain dH from a fire, computed
exactly via Beta-Binomial conjugate posterior (stdlib only). For each
(4 × 40 × 16 = 2560) prompt-step cell:
  pre = Beta(k+1, n-k+1)
  dH  = H(pre) - E_j ~ BetaBinomial[ H(Beta(k+j+1, 8-k+G_esc-j+1)) ]

Outputs: per_prompt.tsv (2560), per_tier.tsv, per_step.tsv (160),
per_method.tsv (4), summary.json.
"""
import json, math, os, statistics
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "experiments/results/n2_reward_tensor_resume"
OUT = WORK / "experiments/results/p5p8"
METHODS = ["grpo", "aero", "gift", "areal"]
GB, GE, NS, NP, TAU = 8, 8, 40, 16, 0.70
SEED, NBOOT = 20260705, 4000


def lbeta(a, b): return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def digamma(x):
    if x <= 0: return float("-inf")
    r = 0
    while x < 8:
        r -= 1 / x; x += 1
    inv = 1 / x; inv2 = inv * inv
    r += math.log(x) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120
        - inv2 * (1 / 252 - inv2 / 240)))
    return r


def hbit(a, b):  # H(Beta(a,b)) in bits
    psi_a, psi_b, psi_ab = digamma(a), digamma(b), digamma(a + b)
    return (lbeta(a, b) - (a - 1) * psi_a - (b - 1) * psi_b
            + (a + b - 2) * psi_ab) / math.log(2)


def dh(k, n, ge):
    Hpre = hbit(k + 1, n - k + 1)
    Hacc = wacc = 0
    lbp = lbeta(k + 1, n - k + 1)
    for j in range(ge + 1):
        log_c = math.lgamma(ge + 1) - math.lgamma(j + 1) - math.lgamma(ge - j + 1)
        lbj = lbeta(k + j + 1, n - k + ge - j + 1)
        lw = log_c + lbj - lbp
        if lw > -500:
            w = math.exp(lw); Hacc += w * hbit(k + j + 1, n - k + ge - j + 1); wacc += w
    return Hpre - Hacc / wacc


def tier(k):
    return "boundary" if k in (0, GB) else ("edge" if k in (1, GB - 1) else "mid")


def fire_bit(plist, low):
    n_b = sum(1 for p in plist if p[2] == "boundary")
    zvf = n_b / len(plist)
    return ((zvf <= 1 - TAU) if low else (zvf > 1 - TAU), zvf)


def main():
    by = {}
    for m in METHODS:
        rows = sorted([json.loads(l) for l in open(N2_DIR / f"{m}_s0_tensors.jsonl")],
                       key=lambda r: r["step"])
        by[m] = rows
    print("Loaded N2 tensors.")

    # Per-(method, step) prompt list: [(pidx, k, tier, dH), ...]
    out = {}
    for m in METHODS:
        for r in by[m]:
            pl = [(i, int(round(sum(p))), tier(int(round(sum(p)))),
                    dh(int(round(sum(p))), GB, GE))
                   for i, p in enumerate(r["rewards"])]
            out[(m, r["step"])] = pl
    print(f"Computed dH for {sum(len(v) for v in out.values())} prompt-obs.")

    pp_rows, ps_rows, meth_rows = [], [], {}
    tier_acc = {"boundary": [], "edge": [], "mid": []}
    for (m, st), pl in out.items():
        fhi, zvf = fire_bit(pl, low=False)
        flo, _ = fire_bit(pl, low=True)
        for (i, k, t, dhv) in pl:
            pp_rows.append({"m": m, "s": st, "i": i, "k": k, "ph": k / GB, "t": t,
                            "hpre": hbit(k + 1, GB - k + 1), "hpost": hbit(k + 1, GB - k + 1) + dhv,
                            "dh": dhv, "fc1": int(fhi), "fanti": int(flo), "z": zvf})
            tier_acc[t].append(dhv)
        fired_mean = statistics.mean([p[3] for p in pl]) if fhi and pl else float("nan")
        max_dh = max(p[3] for p in pl)
        regret = (max_dh - fired_mean) if fhi and pl else 0
        ps_rows.append({"m": m, "s": st, "z": zvf, "fire": int(fhi),
                        "nb": sum(1 for p in pl if p[2] == "boundary"),
                        "ne": sum(1 for p in pl if p[2] != "boundary"),
                        "mdh_fire": fired_mean, "mx_dh": max_dh, "regret": regret})
    for m in METHODS:
        mr = [r for r in pp_rows if r["m"] == m]
        fhi = [r for r in mr if r["fc1"] == 1]
        flo = [r for r in mr if r["fanti"] == 1]
        nfhi = sum(1 for s in range(NS) if any(r["fc1"] == 1 and r["s"] == s for r in mr))
        nflo = sum(1 for s in range(NS) if any(r["fanti"] == 1 and r["s"] == s for r in mr))
        meth_rows[m] = {"n": len(mr), "nsh": nfhi, "nsa": nflo,
                          "frh": nfhi / NS, "fra": nflo / NS,
                          "mdh_all": statistics.mean(r["dh"] for r in mr),
                          "mdh_c1": statistics.mean(r["dh"] for r in fhi) if fhi else 0,
                          "mdh_a": statistics.mean(r["dh"] for r in flo) if flo else 0}

    # Bootstrap CIs
    import random
    rnd = random.Random(SEED)

    def bci(vals, nb=NBOOT):
        n = len(vals)
        if not n: return (0, 0, 0)
        means = sorted(statistics.mean(rnd.choices(vals, k=n)) for _ in range(nb))
        return (statistics.mean(vals), means[int(0.025 * nb)],
                means[max(int(0.975 * nb) - 1, 0)])

    tsum = {}
    for t in ("boundary", "edge", "mid"):
        v = tier_acc[t]
        if not v: continue
        m, lo, hi = bci(v)
        tsum[t] = {"n": len(v), "mean": m, "lo": lo, "hi": hi,
                    "med": statistics.median(v),
                    "f_gt03": sum(1 for x in v if x > 0.3) / len(v)}

    pmboot = {m: dict(n=len(vals := [r["dh"] for r in pp_rows
                                          if r["m"] == m and r["fc1"] == 1]),
                       **{k: v for k, v in zip(("mean", "lo", "hi"), bci(vals))})
               for m in METHODS if any(r["m"] == m and r["fc1"] == 1 for r in pp_rows)}

    # Regression slope dH ~ zvf_step
    pairs = [(p["z"], p["mdh_fire"]) for p in ps_rows
             if not math.isnan(p["mdh_fire"])]
    n = len(pairs)
    sxm = sum(z for z, _ in pairs) / n
    sym = sum(m for _, m in pairs) / n
    slope = (sum((z - sxm) * (m - sym) for z, m in pairs)
              / sum((z - sxm) ** 2 for z, _ in pairs)) if n else 0
    bsl = []
    for _ in range(NBOOT):
        s = [pairs[rnd.randrange(n)] for _ in range(n)]
        sxb = sum(z for z, _ in s) / n
        syb = sum(m for _, m in s) / n
        num = sum((z - sxb) * (m - syb) for z, m in s)
        den = sum((z - sxb) ** 2 for z, _ in s)
        bsl.append(num / den if den > 0 else 0)
    bsl.sort()
    slope_lo, slope_hi = bsl[int(0.025 * NBOOT)], bsl[max(int(0.975 * NBOOT) - 1, 0)]

    regrets = [p["regret"] for p in ps_rows if p["fire"] == 1]
    mr_, mrlo, mrhi = bci(regrets)
    total_reg = sum(regrets) * NP

    print("\n--- HEADLINES ---")
    for t in ("boundary", "edge", "mid"):
        d = tsum[t]
        print(f"  {t:9s} n={d['n']:4d}  mean dH = {d['mean']:.4f} "
              f"[{d['lo']:.4f},{d['hi']:.4f}]  frac>0.3: {d['f_gt03']:.3f}")
    for m, d in pmboot.items():
        print(f"  {m:5s} n_fires={d['n']:4d}  mean dH on fires = "
              f"{d['mean']:.4f} [{d['lo']:.4f},{d['hi']:.4f}]")
    print(f"\nRegression dH ~ zvf_step slope = {slope:.4f} bits/unit "
          f"95% CI [{slope_lo:.4f},{slope_hi:.4f}] — "
          f"{'NEGATIVE' if slope < 0 else 'POSITIVE'} (C1 fires on "
          f"{'LOW-dH' if slope < 0 else 'HIGH-dH'} steps)")
    print(f"\nPer-fired-step regret: mean = {mr_:.4f} bits  "
          f"CI [{mrlo:.4f}, {mrhi:.4f}]  total = {total_reg:.2f} bits")
    print("\nPer-method:")
    for m in METHODS:
        d = meth_rows[m]
        print(f"  {m:5s} mean_dH_C1 = {d['mdh_c1']:.4f}  "
              f"steps_C1 = {d['nsh']}/40  fire-rate_C1 = {d['frh']:.3f}")

    os.makedirs(OUT, exist_ok=True)
    with open(OUT / "p7_iter187_infogain_per_prompt.tsv", "w") as f:
        f.write("method\tstep\tprompt_idx\tk\tp_hat\ttier\t"
                "H_pre_bits\tE_post_bits\tdH_param_bits\tfire_step_c1\t"
                "fire_step_anti\tzvf_step\n")
        for r in pp_rows:
            f.write(f"{r['m']}\t{r['s']}\t{r['i']}\t{r['k']}\t{r['ph']:.4f}\t"
                     f"{r['t']}\t{r['hpre']:.6f}\t{r['hpost']:.6f}\t"
                     f"{r['dh']:.6f}\t{r['fc1']}\t{r['fanti']}\t{r['z']:.4f}\n")
    with open(OUT / "p7_iter187_infogain_per_tier.tsv", "w") as f:
        f.write("tier\tn_obs\tmean_dH\tmedian_dH\tci95_lo\tci95_hi\t"
                "frac_dH_above_0p3\n")
        for t in ("boundary", "edge", "mid"):
            if t not in tsum: continue
            d = tsum[t]
            f.write(f"{t}\t{d['n']}\t{d['mean']:.6f}\t{d['med']:.6f}\t"
                     f"{d['lo']:.6f}\t{d['hi']:.6f}\t{d['f_gt03']:.4f}\n")
    with open(OUT / "p7_iter187_infogain_per_step.tsv", "w") as f:
        f.write("method\tstep\tzvf_step\tfire\tn_boundary\tn_mid_edge\t"
                "mean_dH_on_fired_prompts\tmax_dH_over_prompts\tdH_regret\n")
        for r in ps_rows:
            f.write(f"{r['m']}\t{r['s']}\t{r['z']:.4f}\t{r['fire']}\t{r['nb']}\t"
                     f"{r['ne']}\t{r['mdh_fire']:.6f}\t{r['mx_dh']:.6f}\t{r['regret']:.6f}\n")
    with open(OUT / "p7_iter187_infogain_per_method.tsv", "w") as f:
        f.write("method\tn_prompts\tn_steps_fired_c1\tn_steps_fired_anti\t"
                "fire_step_rate_c1\tfire_step_rate_anti\t"
                "n_fire_prompts_c1\tn_fire_prompts_anti\tmean_dH_overall\t"
                "mean_dH_on_fires_c1\tmean_dH_on_fires_anti\n")
        for m in METHODS:
            d = meth_rows[m]
            f.write(f"{m}\t{d['n']}\t{d['nsh']}\t{d['nsa']}\t"
                     f"{d['frh']:.4f}\t{d['fra']:.4f}\t"
                     f"{sum(1 for r in pp_rows if r['m']==m and r['fc1']==1)}\t"
                     f"{sum(1 for r in pp_rows if r['m']==m and r['fanti']==1)}\t"
                     f"{d['mdh_all']:.6f}\t{d['mdh_c1']:.6f}\t{d['mdh_a']:.6f}\n")
    summary = {
        "headline": {
            "sign": "NEGATIVE" if slope < 0 else "POSITIVE",
            "slope": slope, "ci95": [slope_lo, slope_hi],
            "interpret": ("C1 fires on LOW-dH steps (counter to info-positive)"
                          if slope < 0 else "C1 fires on HIGH-dH steps"),
        },
        "tier_summary": tsum, "method_summary": meth_rows,
        "per_method_bootstraps": pmboot,
        "regret": {"mean": mr_, "ci95": [mrlo, mrhi], "total_bits": total_reg,
                    "n_fired_steps": len(regrets)},
        "settings": {"G_BASE": GB, "G_ESC": GE, "TAU_FIRE": TAU,
                     "N_BOOT": NBOOT, "SEED": SEED},
    }
    with open(OUT / "p7_iter187_infogain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote 5 artifacts to {OUT}/p7_iter187_*.{{tsv,json}}")


if __name__ == "__main__":
    main()
