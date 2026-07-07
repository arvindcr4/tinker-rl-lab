#!/usr/bin/env python3
r"""
Tree Search for Language Model Agents (Koh, McAleer, Fried, Salakhutdinov 2024,
arXiv:2407.01476) -- SP25 L6 Ruslan Salakhutdinov -- mapped onto GRPO's ZVF.

Thesis (target A5 inference-time x Pillar-2 ZVF):
  Koh et al. replace parallel best-of-N sampling with a best-first, value-guided
  SEQUENTIAL tree search, and show it beats best-of-N at matched compute -- most
  on HARD tasks. Their gain comes entirely from breaking the COVERAGE wall:
  finding at least one success on a prompt where i.i.d. sampling keeps failing.

  GRPO's zero-variance fraction (ZVF) has TWO walls, and they behave oppositely:
     ZVF(g) = P[all wrong]  +  P[all correct]
              \___________/    \____________/
              COVERAGE wall     SATURATION wall
              (1-p)^g           p^g
  Tree search (Koh et al.) attacks ONLY the coverage wall -- it can rescue a
  hard, all-wrong group by conditioning later expansions on earlier failures.
  It is powerless against the saturation wall: no search over the SAME policy can
  make an all-CORRECT group contrastive, so it cannot restore GRPO's gradient
  signal in the easy/saturated regime.  This is a clean, paper-facing distinction
  between inference-time search (needs ONE solution) and RL training signal
  (needs a CONTRASTIVE pair).

We compute both walls EXACTLY from real per-prompt data (600 Qwen3-8B GSM8K
groups, native G=8, 3 seeds x 200 prompts): all-wrong = C(8-k,g)/C(8,g),
all-correct = C(k,g)/C(8,g).  g<=8 exact hypergeometric; g>8 i.i.d. p^g+(1-p)^g
(row-18 convention).

Hypotheses (all on REAL data):
  H1 [two walls decompose & cross over] ZVF(g)=allwrong(g)+allcorrect(g) exactly;
     the coverage wall dominates at small g and the saturation wall at large g.
     Report the crossover g* where allcorrect first exceeds allwrong.
  H2 [tree-search addressable share shrinks] the FRACTION of ZVF that is coverage
     (all-wrong / ZVF) -- the part tree search could rescue -- falls monotonically
     with g, while the saturation share rises: tree search's ROI declines exactly
     where GRPO's does, but for the OPPOSITE prompts.
  H3 [hardness targeting] all-wrong mass concentrates on hard (low-p) prompts,
     matching Koh et al.'s "search helps hard tasks": bottom-p-tertile holds the
     large majority of coverage-wall mass.  DECISIVE if share >= 0.80.
  H4 [sequential compute advantage] an oracle best-first search needs ~1/p
     expansions to cover a solvable prompt; compare mean oracle-sequential
     expansions to the parallel g needed for matched coverage.  Report the
     compute ratio at the g where parallel coverage first reaches 0.90.
  H5 [saturation wall is search-invariant -- falsification guard] the all-correct
     mass p^g is NOT reducible by any search over the same policy; as g grows it
     becomes the DOMINANT component of ZVF, so tree search cannot rescue GRPO's
     large-g signal collapse.  DECISIVE if saturation share > 0.5 at g=32.
"""
import json, glob, math, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "berkeley")
os.makedirs(OUT, exist_ok=True)
NATIVE = 8


def comb(n, k):
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


def load_per_problem():
    """Return list of correct-counts k (out of native G=8) pooled over 3 seeds."""
    ks = []
    for f in sorted(glob.glob(os.path.join(RES, "tinker_gsm8k_zvf_s*.json"))):
        if f.endswith("summary.json"):
            continue
        d = json.load(open(f))
        for pp in d["per_problem"]:
            k = int(round(sum(pp["rewards"])))
            ks.append(k)
    return ks


def walls_at_g(ks, g):
    """Return (mean_allwrong, mean_allcorrect) at subsample group size g."""
    aw = ac = 0.0
    for k in ks:
        if g <= NATIVE:
            aw += comb(NATIVE - k, g) / comb(NATIVE, g)
            ac += comb(k, g) / comb(NATIVE, g)
        else:  # i.i.d. extrapolation past native draws
            p = k / NATIVE
            aw += (1.0 - p) ** g
            ac += p ** g
    n = len(ks)
    return aw / n, ac / n


def walls_per_prompt(k, g):
    if g <= NATIVE:
        return comb(NATIVE - k, g) / comb(NATIVE, g), comb(k, g) / comb(NATIVE, g)
    p = k / NATIVE
    return (1.0 - p) ** g, p ** g


def main():
    ks = load_per_problem()
    n = len(ks)
    assert n > 0, "no per-problem data loaded"
    gs = [2, 3, 4, 5, 6, 7, 8, 16, 32]

    # ---- H1/H2/H5: two-wall decomposition per g ----
    rows = []
    crossover = None
    for g in gs:
        aw, ac = walls_at_g(ks, g)
        zvf = aw + ac
        cov = 1.0 - aw
        yield_ = 1.0 - zvf
        cov_share = aw / zvf if zvf > 0 else float("nan")   # tree-search addressable
        sat_share = ac / zvf if zvf > 0 else float("nan")   # search-invariant
        rows.append((g, aw, ac, zvf, cov, yield_, cov_share, sat_share))
        if crossover is None and ac > aw:
            crossover = g
    with open(os.path.join(OUT, "tree_search_two_walls.tsv"), "w") as fh:
        fh.write("g\tall_wrong\tall_correct\tzvf\tcoverage\tyield\t"
                 "coverage_share\tsaturation_share\n")
        for r in rows:
            fh.write("\t".join(f"{x:.6f}" if isinstance(x, float) else str(x)
                               for x in r) + "\n")

    # H2 (corrected): the saturation wall DOMINATES at every g -- tree search is
    # orthogonal to most dead signal. (Original monotone-decrease conjecture is
    # FALSIFIED: coverage share is small & U-shaped, ~0.15-0.20 throughout.)
    cov_shares = [r[6] for r in rows]
    sat_shares = [r[7] for r in rows]
    h2_sat_dominates_all_g = all(s > 0.5 for s in sat_shares)
    h2_cov_share_max = max(cov_shares)
    h2_cov_share_monotone = all(cov_shares[i] >= cov_shares[i + 1] - 1e-9
                                for i in range(len(cov_shares) - 1))  # expected False

    # ---- H3: hardness targeting -- where does all-wrong mass live? ----
    # p-tertiles by per-prompt p_hat=k/8; use a representative small g (g=4).
    g_h3 = 4
    strata = {"hard(p<=1/3)": [], "mid(1/3<p<=2/3)": [], "easy(p>2/3)": []}
    for k in ks:
        p = k / NATIVE
        aw, ac = walls_per_prompt(k, g_h3)
        if p <= 1.0 / 3:
            strata["hard(p<=1/3)"].append((aw, ac))
        elif p <= 2.0 / 3:
            strata["mid(1/3<p<=2/3)"].append((aw, ac))
        else:
            strata["easy(p>2/3)"].append((aw, ac))
    tot_aw = sum(aw for s in strata.values() for aw, _ in s)
    tot_ac = sum(ac for s in strata.values() for _, ac in s)
    h3_rows = []
    for name, vals in strata.items():
        saw = sum(a for a, _ in vals)
        sac = sum(c for _, c in vals)
        h3_rows.append((name, len(vals),
                        saw / tot_aw if tot_aw else 0.0,
                        sac / tot_ac if tot_ac else 0.0))
    hard_aw_share = h3_rows[0][2]
    with open(os.path.join(OUT, "tree_search_hardness_targeting.tsv"), "w") as fh:
        fh.write("stratum\tn_prompts\tall_wrong_mass_share\tall_correct_mass_share\n")
        for r in h3_rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\n")

    # ---- H4: oracle sequential vs parallel compute at matched coverage ----
    # Oracle best-first over the same policy covers a solvable prompt (p>0) in
    # expected 1/p expansions (geometric); unsolvable (p=0) is uncoverable by any
    # method. Parallel needs g draws for coverage 1-(1-p)^g. Compare at the g
    # where mean parallel coverage first reaches 0.90.
    g_star_cov = None
    for g, aw, *_ in rows:
        if (1.0 - aw) >= 0.90:
            g_star_cov = g
            break
    solvable = [k for k in ks if k > 0]
    oracle_seq = sum(NATIVE / k for k in solvable) / len(solvable)  # E[1/p], p=k/8
    parallel_cost = float(g_star_cov) if g_star_cov else float("nan")
    compute_ratio = parallel_cost / oracle_seq if oracle_seq else float("nan")
    frac_uncoverable = (len(ks) - len(solvable)) / len(ks)
    with open(os.path.join(OUT, "tree_search_compute_ratio.tsv"), "w") as fh:
        fh.write("metric\tvalue\n")
        fh.write(f"n_prompts\t{n}\n")
        fh.write(f"frac_uncoverable_p0\t{frac_uncoverable:.6f}\n")
        fh.write(f"g_star_parallel_cov0.90\t{parallel_cost:.6f}\n")
        fh.write(f"oracle_seq_expansions_E[1/p]\t{oracle_seq:.6f}\n")
        fh.write(f"parallel_over_sequential_ratio\t{compute_ratio:.6f}\n")

    # ---- H5: saturation-wall invariance summary ----
    sat32 = [r[7] for r in rows if r[0] == 32][0]
    cov32 = [r[6] for r in rows if r[0] == 32][0]

    # ---- verdicts ----
    r8 = [r for r in rows if r[0] == 8][0]
    verdicts = {
        "H1_two_walls_crossover_g": crossover,
        "H1_exact_decomp_holds": True,  # ZVF==aw+ac by construction (exact)
        "H2_saturation_dominates_every_g": h2_sat_dominates_all_g,
        "H2_coverage_share_max_across_g": round(h2_cov_share_max, 4),
        "H2_original_monotone_conjecture_FALSIFIED": not h2_cov_share_monotone,
        "H3_hard_tertile_allwrong_share": round(hard_aw_share, 4),
        "H3_decisive_ge_0.80": hard_aw_share >= 0.80,
        "H4_g_star_parallel_cov0.90": g_star_cov,
        "H4_oracle_seq_expansions": round(oracle_seq, 3),
        "H4_parallel_over_seq_ratio": round(compute_ratio, 3),
        "H4_frac_uncoverable": round(frac_uncoverable, 4),
        "H5_saturation_share_at_g32": round(sat32, 4),
        "H5_decisive_saturation_dominant_g32": sat32 > 0.5,
        "H5_coverage_share_at_g32": round(cov32, 4),
        "sanity_zvf_at_g8": round(r8[3], 4),
        "n_prompts": n,
        "citation": "Koh, McAleer, Fried, Salakhutdinov 2024, arXiv:2407.01476, "
                    "Tree Search for Language Model Agents (SP25 L6)",
    }
    with open(os.path.join(OUT, "tree_search_summary.json"), "w") as fh:
        json.dump(verdicts, fh, indent=2)

    # console
    print(f"n_prompts={n}")
    print("g  all_wrong all_correct  zvf   cov_share sat_share")
    for r in rows:
        print(f"{r[0]:>2} {r[1]:.4f}   {r[2]:.4f}    {r[3]:.4f} {r[6]:.4f}   {r[7]:.4f}")
    print(f"\ncrossover g* (all_correct>all_wrong): {crossover}")
    print(f"H2 saturation dominates every g: {h2_sat_dominates_all_g} "
          f"(coverage share max={h2_cov_share_max:.4f}, "
          f"monotone conjecture falsified={not h2_cov_share_monotone})")
    print(f"H3 hard-tertile all-wrong share: {hard_aw_share:.4f} "
          f"(decisive>=0.80: {hard_aw_share>=0.80})")
    print(f"H4 parallel g*={g_star_cov} vs oracle-seq {oracle_seq:.2f} "
          f"-> ratio {compute_ratio:.2f}; uncoverable(p=0)={frac_uncoverable:.4f}")
    print(f"H5 saturation share @g32: {sat32:.4f} "
          f"(decisive>0.5: {sat32>0.5}); coverage share @g32: {cov32:.4f}")


if __name__ == "__main__":
    main()
