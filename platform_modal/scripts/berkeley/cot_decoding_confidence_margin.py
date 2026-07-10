#!/usr/bin/env python3
"""B-F24 row 20 — F24 L1 Denny Zhou, "Chain-of-Thought Reasoning Without
Prompting" (Wang & Zhou, arXiv:2402.10200, NeurIPS 2024).

CoT-decoding's central object is an *answer-confidence margin*
  Delta = mean over answer tokens of ( p(top-1) - p(top-2) ),
which reliably selects correct reasoning paths -- better than sequence
probability. We port Delta to the GRPO group level as the decisiveness
margin
  M_t = | 2 * mean_reward_t - 1 |   in [0,1]
(0 = maximally uncertain group, half right; 1 = unanimous group).

We test five hypotheses on the same-stack group-size sweep
(platform_hybrid/experiments/results/groupsize_zvf_sweep.json: 4 group sizes x 3 seeds,
40 steps each, per-step zvf / mean_reward / entropy / advantage_variance
/ grad_norm):

  H1 margin validity      : within-run rho(M_t, zvf_t) strongly positive
                            (M is a real confidence axis; ZVF = P(M=1)).
  H2 training -> confidence: M_t slope>0 and entropy_t slope<0 per run
                            (training surfaces confident answering).
  H3 confidence -> correct : run-level Spearman(mean_entropy, heldout_acc)<0
                            (CoT-decoding: concentration predicts correctness).
  H4 RL/CoT-decoding tension: M at peak-learning step (max advantage_variance)
                            < terminal M, and within-run rho(M,adv_var)<0.
                            The learning frontier is the LOW-confidence region
                            CoT-decoding would discard.
  H5 group-size link      : mean terminal M decreases monotonically with G
                            (larger G sits more on the frontier -> Pillar 3).
"""
import json
import math
import os
from statistics import mean, pstdev

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "platform_hybrid/experiments/results/groupsize_zvf_sweep.json")
OUT = os.path.join(ROOT, "platform_hybrid/experiments/results/berkeley")
os.makedirs(OUT, exist_ok=True)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


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
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def ols_slope(xs, ys):
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def margin(mr):
    return abs(2.0 * mr - 1.0)


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")


def main():
    data = json.load(open(SRC))
    runs = data["runs"]

    per_run = []  # aggregated per-run record
    h1_rows, h2_rows, h4_rows = [], [], []
    for r in runs:
        G, seed = r["group_size"], r["seed"]
        log = r["step_log"]
        steps = [s["step"] for s in log]
        M = [margin(s["mean_reward"]) for s in log]
        zvf = [s["zvf"] for s in log]
        ent = [s["entropy"] for s in log]
        av = [s["advantage_variance"] for s in log]

        rho_M_zvf = pearson(M, zvf)
        slope_M = ols_slope(steps, M)
        slope_ent = ols_slope(steps, ent)
        rho_M_av = pearson(M, av)
        # peak-learning step = max advantage_variance
        peak_i = max(range(len(av)), key=lambda i: av[i])
        M_peak = M[peak_i]
        M_term = mean(M[-5:])  # terminal (last-5) confidence
        ent_mean = mean(ent)

        h1_rows.append((G, seed, round(rho_M_zvf, 4)))
        h2_rows.append((G, seed, round(slope_M, 5), round(slope_ent, 5),
                        int(slope_M > 0 and slope_ent < 0)))
        h4_rows.append((G, seed, round(M_peak, 4), round(M_term, 4),
                        int(M_peak < M_term), round(rho_M_av, 4)))
        per_run.append(dict(G=G, seed=seed, heldout=r["heldout_acc"],
                            ent_mean=ent_mean, M_term=M_term, rho_M_zvf=rho_M_zvf,
                            rho_M_av=rho_M_av, M_peak=M_peak,
                            slope_M=slope_M, slope_ent=slope_ent))

    # ---- H1 ----
    med_rho1 = sorted(x[2] for x in h1_rows)[len(h1_rows) // 2]
    h1_decisive = med_rho1 > 0.7
    write_tsv(os.path.join(OUT, "cot_decoding_h1_margin_validity.tsv"),
              ["group_size", "seed", "rho_M_zvf"], h1_rows)

    # ---- H2 ----
    n_both = sum(x[4] for x in h2_rows)
    h2_decisive = n_both >= 10
    write_tsv(os.path.join(OUT, "cot_decoding_h2_train_confidence.tsv"),
              ["group_size", "seed", "slope_M", "slope_entropy", "both_signs"], h2_rows)

    # ---- H3 ----
    ent_means = [p["ent_mean"] for p in per_run]
    held = [p["heldout"] for p in per_run]
    rho3 = spearman(ent_means, held)
    h3_decisive = (rho3 < -0.3)  # concentration (low entropy) -> higher acc
    write_tsv(os.path.join(OUT, "cot_decoding_h3_confidence_accuracy.tsv"),
              ["group_size", "seed", "mean_entropy", "heldout_acc"],
              [(p["G"], p["seed"], round(p["ent_mean"], 4), round(p["heldout"], 4)) for p in per_run])

    # ---- H4 ----
    n_peak_lt_term = sum(x[4] for x in h4_rows)
    med_rho_av = sorted(x[5] for x in h4_rows)[len(h4_rows) // 2]
    h4_decisive = (n_peak_lt_term >= 10 and med_rho_av < 0)
    write_tsv(os.path.join(OUT, "cot_decoding_h4_frontier_tension.tsv"),
              ["group_size", "seed", "M_peaklearn", "M_terminal", "peak_lt_term", "rho_M_advvar"], h4_rows)

    # ---- H5 ----
    by_G = {}
    for p in per_run:
        by_G.setdefault(p["G"], []).append(p["M_term"])
    Gs = sorted(by_G)
    meanM_by_G = [(G, round(mean(by_G[G]), 4)) for G in Gs]
    vals = [v for _, v in meanM_by_G]
    monotone_dec = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    rho5 = spearman([p["G"] for p in per_run], [p["M_term"] for p in per_run])
    h5_decisive = monotone_dec and rho5 < 0
    write_tsv(os.path.join(OUT, "cot_decoding_h5_group_size_margin.tsv"),
              ["group_size", "mean_terminal_margin"], [(G, v) for G, v in meanM_by_G])

    decisive = sum([h1_decisive, h2_decisive, h3_decisive, h4_decisive, h5_decisive])
    summary = {
        "row": 20,
        "source": "F24 L1 Denny Zhou -- Wang & Zhou, CoT Reasoning Without Prompting (arXiv:2402.10200, NeurIPS 2024)",
        "target": "A5 + Pillar 2/3",
        "data": "platform_hybrid/experiments/results/groupsize_zvf_sweep.json (4 G x 3 seeds, 40 steps)",
        "margin_def": "M_t = |2*mean_reward_t - 1| (group-level analog of CoT-decoding Delta)",
        "H1_margin_validity": {"median_rho_M_zvf": round(med_rho1, 4), "decisive": h1_decisive},
        "H2_train_confidence": {"n_runs_both_signs": n_both, "of": len(h2_rows), "decisive": h2_decisive},
        "H3_confidence_accuracy": {"spearman_entropy_heldout": round(rho3, 4), "decisive": h3_decisive},
        "H4_frontier_tension": {"n_peak_lt_terminal": n_peak_lt_term, "of": len(h4_rows),
"median_rho_M_advvar": round(med_rho_av, 4), "decisive": h4_decisive},
        "H5_group_size_margin": {"mean_terminal_margin_by_G": dict(meanM_by_G),
                                  "monotone_decreasing": monotone_dec,
                                  "spearman_G_margin": round(rho5, 4), "decisive": h5_decisive},
        "n_decisive": decisive, "n_hypotheses": 5,
        "verdict": "DECISIVE" if decisive >= 4 else ("SUGGESTIVE" if decisive >= 2 else "NULL"),
    }
    json.dump(summary, open(os.path.join(OUT, "cot_decoding_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
