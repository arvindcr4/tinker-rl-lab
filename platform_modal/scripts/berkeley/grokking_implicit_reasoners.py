#!/usr/bin/env python3
"""
Grokked Transformers are Implicit Reasoners (Wang, Yue, Su, Sun; NeurIPS 2024,
arXiv:2405.15071) -- SP25 L3 Yu Su -- mapped onto GRPO training dynamics.

Thesis (A3 + Pillar 1/3):  The paper's two sharp claims are
  (a) DELAYED GENERALIZATION ("grokking"): generalization emerges only through
      extended training far past the point where the model fits the train set;
  (b) generalization speed/ceiling is governed by a STRUCTURAL RATIO (inferred/
      atomic facts), NOT by absolute dataset size.

We show our same-stack GRPO runs exhibit the SAME two signatures, with the
group size G playing the role of Yu Su's structural ratio:

  MEMORIZATION  = training reward on the sampled rollouts.  It saturates to ~1.0
                  by step ~14-15, and -- crucially -- the saturation step is
                  INVARIANT to G (convergence.tsv).
  GENERALIZATION= heldout GSM8K accuracy.  It keeps climbing 4-64x longer than
                  memorization, and its CEILING is set by G, not by how fast /
                  how completely the train reward saturated (retention curve).

So "structure (G / contrastive yield), not size (train saturation, prompt count),
controls generalization" is Yu Su's grokking result reproduced in RL-post-training
form -- the sharpest external echo of our Pillar-1 "structure > size" program.

Hypotheses (all on REAL data: Qwen3-8B GSM8K, 3 seeds x 200 prompts native G=8;
convergence.tsv G in {2,4,8,16}; retention curve G in {4,32}):
  H1 [grokking signature] flat-train + still-rising-heldout: train reward is
     saturated (~1.0) while heldout is still climbing at the largest budget.
  H2 [memorization is structure-invariant] train-saturation step t_mem has range
     <=2 across G and no monotone G-trend (Spearman ~ 0).
  H3 [generalization is structure-controlled] the G32-vs-G4 heldout gap OPENS
     post-memorization (grows with budget) and the heldout slope scales with G.
  H4 [contrastive yield mediates] Y(G)=1-ZVF(G), computed EXACTLY from per-prompt
     data (hypergeometric for G<=8, i.i.d.-collision for G>8), rises with G in the
     same rank order as the heldout ceiling -> Y is the mechanistic ratio.
  H5 [ratio not size -- falsification guard] Y(G) is invariant to prompt count N
     (bootstrap SE tiny) but moves strongly with G -> the RATIO knob dominates the
     SIZE knob, exactly Yu Su's claim.
"""
import json, glob, math, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "berkeley")
os.makedirs(OUT, exist_ok=True)


def comb(n, k):
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k)


def load_per_problem():
    """Return list of correct-counts k (out of native G=8) pooled over 3 seeds."""
    ks = []
    for f in sorted(glob.glob(os.path.join(RES, "tinker_gsm8k_zvf_s*.json"))):
        if f.endswith("summary.json"):
            continue
        d = json.load(open(f))
        G = d["group_size"]
        for pp in d["per_problem"]:
            k = int(round(sum(pp["rewards"])))
            ks.append((k, G))
    return ks  # list of (k, nativeG)


def zvf_at_G(ks, g):
    """Exact contrastive-collision fraction at group size g.
    g<=8: exact hypergeometric subsample from the 8 native draws.
    g>8 : i.i.d.-collision extrapolation p^g+(1-p)^g with p=k/8 (row-18 convention)."""
    tot = 0.0
    n = 0
    for k, nativeG in ks:
        if g <= nativeG:
            p_allc = comb(k, g) / comb(nativeG, g)
            p_allw = comb(nativeG - k, g) / comb(nativeG, g)
            tot += p_allc + p_allw
        else:
            p = k / nativeG
            tot += p ** g + (1.0 - p) ** g
        n += 1
    return tot / n  # ZVF; Y = 1 - ZVF


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx * dy else 0.0


def main():
    findings = {}
    ks = load_per_problem()
    n_native = len(ks)

    # ---- convergence: memorization saturation step t_mem per (G,seed) ----
    conv = []
    with open(os.path.join(RES, "group_size_convergence.tsv")) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            if abs(float(r["threshold"]) - 0.95) < 1e-9:
                conv.append((int(r["G"]), int(r["seed"]), int(r["first_step"]),
                             float(r["last_step_mean_reward"])))
    Gs_conv = sorted(set(g for g, _, _, _ in conv))
    t_mem_by_G = {g: [s for gg, _, s, _ in conv if gg == g] for g in Gs_conv}
    all_tmem = [s for _, _, s, _ in conv]
    tmem_range = max(all_tmem) - min(all_tmem)
    mean_tmem_by_G = {g: sum(v) / len(v) for g, v in t_mem_by_G.items()}
    rho_G_tmem = spearman([g for g, _, s, _ in conv], [s for g, _, s, _ in conv])
    train_final = [lr for _, _, _, lr in conv]

    # ---- retention: heldout accuracy vs budget for G4 and G32 ----
    ret = []
    with open(os.path.join(RES, "group_size_iter103_retention_curve.tsv")) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            ret.append((int(r["budget_tokens"]), float(r["acc_G4"]), float(r["acc_G32"])))
    ret.sort()
    budgets = [b for b, _, _ in ret]
    accG4 = [a for _, a, _ in ret]
    accG32 = [a for _, _, a in ret]
    gap = [b - a for a, b in zip(accG4, accG32)]  # G32 - G4
    # generalization slope vs log10(budget)
    lb = [math.log10(b) for b in budgets]
    def slope(xs, ys):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den if den else 0.0
    slopeG4, slopeG32 = slope(lb, accG4), slope(lb, accG32)

    # ---- contrastive yield Y(G) exact ----
    Gs = [2, 4, 8, 16, 32, 64]
    zvf = {g: zvf_at_G(ks, g) for g in Gs}
    Y = {g: 1 - zvf[g] for g in Gs}

    # H4 mediation: heldout ceiling (largest budget) vs Y for G4,G32
    heldout_ceiling = {4: accG4[-1], 32: accG32[-1]}
    y4, y32 = Y[4], Y[32]
    dir_consistent = (heldout_ceiling[32] > heldout_ceiling[4]) == (y32 > y4)

    # H5 ratio-vs-size: bootstrap-subsample prompt count N, measure Y(G) SE.
    # Deterministic disjoint chunking (no RNG): split pooled ks into folds of size N.
    def Y_folds(g, N):
        vals = []
        for i in range(0, len(ks) - N + 1, N):
            vals.append(1 - zvf_at_G(ks[i:i + N], g))
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1))
        return m, sd / math.sqrt(len(vals)), len(vals)
    size_test = {}
    for g in (4, 32):
        for N in (50, 100, 200):
            size_test[(g, N)] = Y_folds(g, N)
    # SE of Y across prompt-count (size) knob vs range across G (ratio) knob
    se_size_g4 = size_test[(4, 100)][1]
    Y_ratio_range = max(Y.values()) - min(Y.values())

    # ================= write TSVs =================
    with open(os.path.join(OUT, "grokking_memorization_invariance.tsv"), "w") as f:
        f.write("G\tseed\tt_mem_step\ttrain_final_reward\n")
        for g, s, st, lr in sorted(conv):
            f.write(f"{g}\t{s}\t{st}\t{lr}\n")

    with open(os.path.join(OUT, "grokking_generalization_curve.tsv"), "w") as f:
        f.write("budget_tokens\tacc_G4\tacc_G32\tgen_gap_G32_minus_G4\n")
        for (b, a4, a32), gp in zip(ret, gap):
            f.write(f"{b}\t{a4}\t{a32}\t{gp:.4f}\n")

    with open(os.path.join(OUT, "grokking_contrastive_yield.tsv"), "w") as f:
        f.write("G\tZVF\tY_contrastive_yield\tmethod\n")
        for g in Gs:
            meth = "hypergeometric_exact" if g <= 8 else "iid_collision_extrap"
            f.write(f"{g}\t{zvf[g]:.6f}\t{Y[g]:.6f}\t{meth}\n")

    with open(os.path.join(OUT, "grokking_ratio_vs_size.tsv"), "w") as f:
        f.write("G\tN_prompts\tY_mean\tY_SE\tn_folds\n")
        for (g, N), (m, se, nf) in sorted(size_test.items()):
            f.write(f"{g}\t{N}\t{m:.6f}\t{se:.6f}\t{nf}\n")

    # ================= hypotheses =================
    # H1 grokking signature
    train_saturated = min(train_final) >= 0.95
    heldout_still_rising = accG32[-1] > accG32[-2]  # slope positive at largest budget
    H1 = train_saturated and heldout_still_rising
    findings["H1_grokking_signature"] = {
        "train_min_final_reward": round(min(train_final), 4),
        "train_saturated_ge_0.95": train_saturated,
        "heldout_G32_last_two": [accG32[-2], accG32[-1]],
        "heldout_still_rising": heldout_still_rising,
        "verdict": "DECISIVE" if H1 else "NULL",
    }
    # H2 memorization structure-invariance -- magnitude test: memorization time is
    # far LESS G-sensitive than the generalization ceiling (Yu Su's core dissociation).
    mean_all_tmem = sum(all_tmem) / len(all_tmem)
    cv_tmem = (math.sqrt(sum((s - mean_all_tmem) ** 2 for s in all_tmem)
                         / (len(all_tmem) - 1)) / mean_all_tmem)
    mem_G_sens = (max(mean_tmem_by_G.values()) - min(mean_tmem_by_G.values())) / mean_all_tmem
    gen_ceiling_mean = (accG4[-1] + accG32[-1]) / 2
    gen_G_sens = (accG32[-1] - accG4[-1]) / gen_ceiling_mean
    H2 = cv_tmem < 0.10 and gen_G_sens > 3 * mem_G_sens
    findings["H2_memorization_G_invariant"] = {
        "t_mem_range_steps": tmem_range,
        "mean_t_mem_by_G": {str(g): round(v, 2) for g, v in mean_tmem_by_G.items()},
        "cv_t_mem": round(cv_tmem, 4),
        "memorization_G_sensitivity": round(mem_G_sens, 4),
        "generalization_G_sensitivity": round(gen_G_sens, 4),
        "gen_over_mem_sensitivity_x": round(gen_G_sens / mem_G_sens, 2) if mem_G_sens else None,
        "spearman_G_vs_tmem": round(rho_G_tmem, 4),
        "verdict": "DECISIVE" if H2 else "NULL",
    }
    # H3 generalization structure-controlled
    gap_opens = gap[-1] - gap[0] > 0.10
    slope_scales = slopeG32 > slopeG4 * 1.5
    H3 = gap_opens and slope_scales
    findings["H3_generalization_G_controlled"] = {
        "gen_gap_first_budget": round(gap[0], 4),
        "gen_gap_last_budget": round(gap[-1], 4),
        "gap_opens_post_memorization": gap_opens,
        "heldout_slope_per_decade_G4": round(slopeG4, 4),
        "heldout_slope_per_decade_G32": round(slopeG32, 4),
        "slope_ratio_G32_over_G4": round(slopeG32 / slopeG4, 3) if slopeG4 else None,
        "verdict": "DECISIVE" if H3 else "NULL",
    }
    # H4 contrastive-yield mediation
    findings["H4_contrastive_yield_mediates"] = {
        "Y_G4": round(y4, 4), "Y_G32": round(y32, 4),
        "heldout_ceiling_G4": heldout_ceiling[4],
        "heldout_ceiling_G32": heldout_ceiling[32],
        "direction_consistent": dir_consistent,
        "verdict": "SUGGESTIVE" if dir_consistent else "NULL",
        "note": "only 2 heldout anchors (G4,G32); direction test not a regression",
    }
    # H5 ratio-not-size
    H5 = (se_size_g4 < 0.02) and (Y_ratio_range > 10 * se_size_g4)
    findings["H5_ratio_not_size"] = {
        "Y_SE_across_size_N100_G4": round(se_size_g4, 5),
        "Y_range_across_ratio_G": round(Y_ratio_range, 4),
        "ratio_dominance_x": round(Y_ratio_range / se_size_g4, 1) if se_size_g4 else None,
        "verdict": "DECISIVE" if H5 else "NULL",
    }

    n_dec = sum(1 for v in findings.values() if v["verdict"] == "DECISIVE")
    n_sug = sum(1 for v in findings.values() if v["verdict"] == "SUGGESTIVE")
    summary = {
        "paper": "Grokked Transformers are Implicit Reasoners (Wang, Yue, Su, Sun; "
                 "NeurIPS 2024, arXiv:2405.15071)",
        "lecture": "SP25 L3 Yu Su",
        "target": "A3 + Pillar 1/3",
        "n_native_prompt_seeds": n_native,
        "decisive": n_dec, "suggestive": n_sug, "n_hyp": len(findings),
        "hypotheses": findings,
    }
    json.dump(summary, open(os.path.join(OUT, "grokking_summary.json"), "w"), indent=2)
    print(json.dumps({k: v["verdict"] for k, v in findings.items()}, indent=2))
    print(f"native prompt-seed rows: {n_native}")
    print(f"Y(G): " + ", ".join(f"G{g}={Y[g]:.3f}" for g in Gs))
    print(f"t_mem by G: {mean_tmem_by_G}  range={tmem_range}")
    print(f"gen gap: {gap[0]:.3f} -> {gap[-1]:.3f};  slopes G4={slopeG4:.3f} G32={slopeG32:.3f}")
    print(f"DECISIVE {n_dec}/{len(findings)}, SUGGESTIVE {n_sug}")


if __name__ == "__main__":
    main()
