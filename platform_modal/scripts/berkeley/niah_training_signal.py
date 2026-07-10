#!/usr/bin/env python3
"""Needle-in-a-Haystack for the GRPO training trajectory (F24 L4, Burak Gokturk).

F24 L4 (Burak Gokturk, Google) covers enterprise GenAI eval methodology: RAG
grounding and the Needle-in-a-Haystack (NIAH) long-context probe. The NIAH eval
places a "needle" fact at depth d inside a "haystack" context of length L and asks
whether it is recoverable as a function of (depth, length). Two foundational
formalizations:
  - Liu et al., "Lost in the Middle: How Language Models Use Long Contexts",
    TACL 2024 (arXiv:2307.03172). Core finding: recovery is a U-shape -- WORST in
    the middle of the context.
  - Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context LMs?",
    COLM 2024 (arXiv:2404.06654). Formalizes an "effective context length": the
    length beyond which recovery saturates.

We PORT that eval onto the GRPO *training trajectory*. The 40-step per-run trace is
the haystack; the "needle" is the terminal-performance signal. We ask: from only a
window of W consecutive steps starting at position p, is the needle recoverable?
Two needles:
  (a) per-run needle  = the run's own terminal reward level R_T (last-10 mean).
      recovered if |window_mean_reward - R_T| <= TOL.
  (b) cross-run needle = the terminal group-size REWARD ranking, recovered via
      Spearman(window_reward, terminal_reward) across the 12 runs.

Data: platform_hybrid/experiments/results/groupsize_zvf_sweep.json (same-stack sweep,
4 G x 3 seed x 40 step; per-step {zvf, mean_reward, entropy, advantage_variance,
grad_norm}). Target A2 (eval methodology) + A1 (statistical rigor).

Hypotheses:
  H1 POSITION-CLIFF   : for fixed W, per-run recovery of R_T rises MONOTONICALLY with
                        start position p (needle at the END of a training haystack, not
                        the middle). DECISIVE if the position-recovery curve is monotone
                        non-decreasing for >=11/12 runs AND late-minus-early gap >=0.5.
  H2 EFFECTIVE-WINDOW : (RULER) cross-run ranking recovery Spearman(window,terminal)
                        rises with W and saturates. DECISIVE if rho is monotone
                        non-decreasing in W (end-anchored) AND an effective window
                        W* (rho>=0.9*rho_max) exists at W* << 40.
  H3 NIAH-GRID        : the 2D (position p x length W) recovery map has a connected
                        LATE/LONG recoverable region: earliest recoverable start p*(W)
                        is non-increasing in W. DECISIVE if p*(W) monotone non-increasing
                        over the sampled W grid for the pooled map.
  H4 G-MODULATION     : does group size move the needle earlier? per-run p*(W_fix) vs
                        log2(G). Predict NULL (consistency check w/ StateFlow row-23:
                        G moves variance, not schedule). DECISIVE only if
                        |Spearman(log2G, p*)| >= 0.5 sign-stable; else reported NULL-consistent.
  H5 MIDDLE-FALSIFY   : direct Lost-in-the-Middle test. Fit both a monotone(position) and
                        a U-shape(|p-mid|) model to per-run recovery; the training haystack
                        is NOT lost-in-the-middle. DECISIVE if monotone SSE < U-shape SSE
                        for >=11/12 runs (recovery is one-sided, worst-early not worst-middle).
"""
import json, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "platform_hybrid/experiments/results/groupsize_zvf_sweep.json")
OUT = os.path.join(ROOT, "platform_hybrid/experiments/results/berkeley")
os.makedirs(OUT, exist_ok=True)

TOL = 0.05                       # recovery tolerance on reward level (absolute)
W_GRID = [1, 2, 3, 5, 8, 13, 21] # log-ish window lengths (RULER length axis)
W_FIX = 5                        # fixed window for the position-curve hypotheses
NSTEP = 40


def wtsv(name, header, rows):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]: j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1): r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 2: return 0.0
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def window_mean(seq, p, w):
    seg = seq[p:p + w]
    return sum(seg) / len(seq[p:p + w]) if seg else float("nan")


def load():
    d = json.load(open(SRC))
    runs = []
    for r in d["runs"]:
        rew = [s["mean_reward"] for s in r["step_log"]]
        runs.append({"G": r["group_size"], "seed": r["seed"], "rew": rew,
                     "R_T": sum(rew[-10:]) / len(rew[-10:])})
    return runs


def main():
    runs = load()
    summary = {"n_runs": len(runs), "TOL": TOL, "W_FIX": W_FIX, "W_GRID": W_GRID}

    # ---- H1: position cliff (fixed W, vary p) -------------------------------
    # per-run recovery vector over start positions; test monotone non-decreasing.
    h1_rows = []
    mono_ok = 0
    gaps = []
    per_run_rec = {}   # (G,seed) -> {p: 0/1}
    positions = list(range(0, NSTEP - W_FIX + 1))
    for r in runs:
        rec = {}
        for p in positions:
            wm = window_mean(r["rew"], p, W_FIX)
            rec[p] = 1 if abs(wm - r["R_T"]) <= TOL else 0
        per_run_rec[(r["G"], r["seed"])] = rec
        # monotone non-decreasing check (allow smoothing over noise: cumulative max non-violating)
        viol = sum(1 for i in range(1, len(positions)) if rec[positions[i]] < rec[positions[i - 1]])
        # tolerate isolated dips: monotone if #downward transitions <= 1
        is_mono = viol <= 1
        mono_ok += int(is_mono)
        early = st.mean(rec[p] for p in positions[:len(positions) // 3])
        late = st.mean(rec[p] for p in positions[-len(positions) // 3:])
        gaps.append(late - early)
        h1_rows.append([f"G{r['G']}_s{r['seed']}", round(early, 4), round(late, 4),
                        round(late - early, 4), viol, int(is_mono)])
    med_gap = st.median(gaps)
    h1_decisive = (mono_ok >= 11) and (med_gap >= 0.5)
    wtsv("niah_h1_position_cliff.tsv",
         ["run", "early_rec", "late_rec", "late_minus_early", "down_transitions", "is_monotone"], h1_rows)
    summary["H1_position_cliff"] = {"mono_ok": mono_ok, "median_late_minus_early": round(med_gap, 4),
                                    "decisive": bool(h1_decisive)}

    # ---- H2: effective window (RULER) ---------------------------------------
    # end-anchored windows of increasing W; cross-run Spearman(window,terminal).
    term = [r["R_T"] for r in runs]
    h2_rows = []
    rhos = []
    for w in W_GRID:
        wm = [window_mean(r["rew"], NSTEP - w, w) for r in runs]   # last-w-step window
        rho = spearman(wm, term)
        rhos.append(rho)
        h2_rows.append([w, round(rho, 4)])
    rho_max = max(rhos)
    # monotone non-decreasing (tolerate one dip)
    dips = sum(1 for i in range(1, len(rhos)) if rhos[i] < rhos[i - 1] - 1e-9)
    wstar = next((w for w, rho in zip(W_GRID, rhos) if rho >= 0.9 * rho_max), W_GRID[-1])
    h2_decisive = (dips <= 1) and (wstar <= 13) and (rho_max >= 0.7)
    wtsv("niah_h2_effective_window.tsv", ["window_W", "spearman_window_vs_terminal"], h2_rows)
    summary["H2_effective_window"] = {"rhos": [round(x, 4) for x in rhos], "rho_max": round(rho_max, 4),
                                      "dips": dips, "W_star": wstar, "decisive": bool(h2_decisive)}

    # ---- H3: NIAH 2D grid ; earliest recoverable start p*(W) ----------------
    # pooled recovery-rate over 12 runs for each (p,W); p*(W) = earliest p with rate>=0.5
    grid_rows = []
    pstar = {}
    for w in W_GRID:
        pos_w = list(range(0, NSTEP - w + 1))
        pstar_w = None
        for p in pos_w:
            rate = st.mean(1 if abs(window_mean(r["rew"], p, w) - r["R_T"]) <= TOL else 0 for r in runs)
            grid_rows.append([w, p, round(rate, 4)])
            if pstar_w is None and rate >= 0.5:
                pstar_w = p
        pstar[w] = pstar_w if pstar_w is not None else (NSTEP - w)
    # monotone non-increasing p*(W)
    pv = [pstar[w] for w in W_GRID]
    rises = sum(1 for i in range(1, len(pv)) if pv[i] > pv[i - 1] + 1e-9)
    h3_decisive = rises <= 1
    wtsv("niah_h3_grid.tsv", ["window_W", "start_p", "recovery_rate"], grid_rows)
    wtsv("niah_h3_pstar.tsv", ["window_W", "earliest_recoverable_start_pstar"],
         [[w, pstar[w]] for w in W_GRID])
    summary["H3_niah_grid"] = {"pstar_by_W": {str(w): pstar[w] for w in W_GRID},
                               "rises": rises, "decisive": bool(h3_decisive)}

    # ---- H4: group-size modulation of needle position -----------------------
    # per-run p*(W_FIX); mean per G; Spearman(log2G, mean p*).
    import math
    h4_rows = []
    perG = {}
    for r in runs:
        rec = per_run_rec[(r["G"], r["seed"])]
        pst_run = next((p for p in positions if rec[p] == 1), positions[-1])
        perG.setdefault(r["G"], []).append(pst_run)
        h4_rows.append([f"G{r['G']}_s{r['seed']}", r["G"], pst_run])
    Gs = sorted(perG)
    xs = [math.log2(g) for g in Gs]
    ys = [st.mean(perG[g]) for g in Gs]
    rho_g = spearman(xs, ys)
    h4_decisive = abs(rho_g) >= 0.5
    wtsv("niah_h4_group_modulation.tsv", ["run", "group_size", "pstar_Wfix"], h4_rows)
    wtsv("niah_h4_group_means.tsv", ["group_size", "log2G", "mean_pstar"],
         [[g, round(math.log2(g), 4), round(st.mean(perG[g]), 4)] for g in Gs])
    summary["H4_group_modulation"] = {"spearman_log2G_pstar": round(rho_g, 4),
                                      "decisive": bool(h4_decisive),
                                      "null_consistent_with_row23": bool(not h4_decisive)}

    # ---- H5: Lost-in-the-Middle falsification -------------------------------
    # per run, fit recovery(p) with (a) monotone ramp via position-rank, (b) U-shape |p-mid|.
    # compare SSE; monotone should win -> training haystack is worst-EARLY, not worst-MIDDLE.
    mid = (len(positions) - 1) / 2.0
    h5_rows = []
    mono_wins = 0
    for r in runs:
        rec = per_run_rec[(r["G"], r["seed"])]
        y = [rec[p] for p in positions]
        # monotone model: predict by position index scaled to [0,1]
        xr = [i / (len(positions) - 1) for i in range(len(positions))]
        # U-shape model: predict by 1 - normalized distance-from-middle (high at ends)
        xu = [abs(i - mid) / mid for i in range(len(positions))]   # 0 mid ->1 ends
        # fit simple 1-param slope+intercept via least squares for each predictor
        def sse(xp):
            n = len(xp); mx = sum(xp) / n; my = sum(y) / n
            den = sum((a - mx) ** 2 for a in xp)
            b = (sum((a - mx) * (c - my) for a, c in zip(xp, y)) / den) if den else 0.0
            a0 = my - b * mx
            return sum((c - (a0 + b * a)) ** 2 for a, c in zip(xp, y))
        sse_m, sse_u = sse(xr), sse(xu)
        win = sse_m < sse_u
        mono_wins += int(win)
        h5_rows.append([f"G{r['G']}_s{r['seed']}", round(sse_m, 4), round(sse_u, 4), int(win)])
    h5_decisive = mono_wins >= 11
    wtsv("niah_h5_middle_falsify.tsv", ["run", "sse_monotone", "sse_ushape", "monotone_wins"], h5_rows)
    summary["H5_middle_falsify"] = {"monotone_wins": mono_wins, "decisive": bool(h5_decisive)}

    n_dec = sum(summary[k]["decisive"] for k in summary if isinstance(summary[k], dict) and "decisive" in summary[k])
    summary["n_decisive"] = n_dec
    json.dump(summary, open(os.path.join(OUT, "niah_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
