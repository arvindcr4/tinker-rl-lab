#!/usr/bin/env python3
"""Regenerate the six Pillar-4 (length-bias) figures directly from the released
source data. Outputs .pdf + .png into paper/figures/. All numbers are computed
from experiments/results/* so the figures match the section tables/captions.
"""
import json, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

RES = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

GRPO_C, DR_C = "#1f4e79", "#c0392b"

def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)

def load_tsv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f, delimiter="\t"))

def steps(run, key):
    return np.array([s[key] for s in run["step_log"]], float)

def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", stem)

gsm = load_json("drgrpo_gsm8k_cot_full.json")["runs"]
arith = load_json("drgrpo_vs_grpo.json")["runs"]

def algo_norm(a):
    return "dr_grpo" if a in ("dr_grpo", "drgrpo") else "grpo"

# ---------------------------------------------------------------- Figure 1
# length_vs_reward: per-step (len, reward) trajectories, two panels.
def fig_length_vs_reward():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, runs, title in [(axL, gsm, "GSM8K-CoT (Qwen2.5-1.5B-Instruct)"),
                            (axR, arith, "Arithmetic (Qwen2.5-0.5B)")]:
        for r in runs:
            c = GRPO_C if algo_norm(r["algo"]) == "grpo" else DR_C
            L, R = steps(r, "mean_comp_len"), steps(r, "mean_reward")
            ax.plot(L, R, "-", color=c, alpha=0.35, lw=1)
            ax.scatter(L[0], R[0], color=c, marker="o", s=28, zorder=3)
            ax.scatter(L[-1], R[-1], color=c, marker="*", s=90, zorder=3)
        ax.set_xlabel("mean completion length (tokens)")
        ax.set_ylabel("mean reward")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    axL.legend(handles=[Line2D([], [], color=GRPO_C, label="GRPO"),
                        Line2D([], [], color=DR_C, label="Dr.GRPO"),
                        Line2D([], [], color="gray", marker="o", ls="", label="step 0"),
                        Line2D([], [], color="gray", marker="*", ls="", label="final")],
               fontsize=8, loc="best")
    fig.suptitle("Per-step (length, reward) trajectories move toward shorter, "
                 "higher-reward—opposite the verbosity trap", fontsize=10)
    save(fig, "length_vs_reward")

# ---------------------------------------------------------------- Figure 2
# length_vs_reward_elevated: (A) trap-onset counts, (B) decile E[R|L] on GSM8K,
# (C) 100-step crossval reward vs length.
def sliding_onset(runs, w=10):
    fires = 0
    for r in runs:
        L, R = steps(r, "mean_comp_len"), steps(r, "mean_reward")
        first_half = L[:len(L)//2].mean()
        fired = False
        for s in range(w, len(L)):
            seg = slice(s-w, s)
            rl = spearmanr(np.arange(w), L[seg]).correlation
            rr = spearmanr(np.arange(w), R[seg]).correlation
            if rl is not None and rl > 0.3 and (rr is None or rr <= 0) and L[s-1] > first_half:
                fired = True
                break
        fires += fired
    return fires

def fig_elevated():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    # (A) trap-onset counts
    a_fire, g_fire = sliding_onset(arith), sliding_onset(gsm)
    axes[0].bar(["Arithmetic\n(n=10)", "GSM8K-CoT\n(n=6)"], [a_fire, g_fire],
                color=["#7f8c8d", "#7f8c8d"])
    for i, (n, v) in enumerate([(10, a_fire), (6, g_fire)]):
        axes[0].text(i, v+0.1, f"{v}/{n}", ha="center", fontsize=9)
    axes[0].set_ylabel("runs with local trap-onset")
    axes[0].set_title("(A) Sliding-window trap-onset\n(W=10, none sustained)", fontsize=9)
    axes[0].set_ylim(0, 7)
    # (B) decile E[R|L] on GSM8K, pooled across seeds
    pts = []
    for r in gsm:
        for L, R in zip(steps(r, "mean_comp_len"), steps(r, "mean_reward")):
            pts.append((L, R))
    pts = np.array(sorted(pts))
    dec = np.array_split(pts, 10)
    xd = [d[:, 0].mean() for d in dec]
    yd = [d[:, 1].mean() for d in dec]
    axes[1].plot(xd, yd, "o-", color="#2c3e50")
    axes[1].set_xlabel("completion length (decile mean)")
    axes[1].set_ylabel("mean reward")
    axes[1].set_title("(B) Decile E[R|L], GSM8K-CoT\nlonger → lower reward", fontsize=9)
    axes[1].grid(alpha=0.25)
    # (C) 100-step crossval
    rew, ln = [], []
    with open(os.path.join(RES, "arithmetic_metrics.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            r = d.get("env/all/reward/total", d.get("reward"))
            l = d.get("env/all/ac_tokens_per_turn",
                      d.get("ac_tokens_per_turn", d.get("len")))
            if r is not None and l is not None:
                rew.append(r); ln.append(l)
    x = np.arange(len(rew))
    ax = axes[2]
    ax.plot(x, rew, color="#27ae60", label="reward")
    ax.set_ylabel("reward", color="#27ae60")
    ax.set_xlabel("training step")
    ax2 = ax.twinx()
    ax2.plot(x, ln, color="#8e44ad", label="length")
    ax2.set_ylabel("mean length (tokens)", color="#8e44ad")
    ax.set_title("(C) 100-step crossval\nreward grows, length at cap", fontsize=9)
    save(fig, "length_vs_reward_elevated")

# ---------------------------------------------------------------- Figure 3
# length_zvf_coupling: (A) rho(len,zvf) per cell w/ CI, (B) tertile ZVF, (C) co-evolution
def fig_zvf_coupling():
    rows = load_tsv("length_zvf_coupling.tsv")
    # aggregate mean rho_len_zvf per (task, algo)
    cells = {}
    for r in rows:
        k = (r["task"], algo_norm(r["algo"]))
        cells.setdefault(k, []).append(float(r["rho_len_zvf"]))
    order = [("arithmetic_qwen2.5-0.5b", "grpo"), ("arithmetic_qwen2.5-0.5b", "dr_grpo"),
             ("gsm8k_cot_qwen2.5-1.5b", "grpo"), ("gsm8k_cot_qwen2.5-1.5b", "dr_grpo")]
    labels = ["Arith\nGRPO", "Arith\nDr.GRPO", "GSM8K\nGRPO", "GSM8K\nDr.GRPO"]
    means = [np.mean(cells[k]) for k in order]
    # bootstrap CI across seeds
    def boot(vals, B=5000):
        vals = np.array(vals); rng = np.random.default_rng(0)
        bs = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(B)]
        return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    cis = [boot(cells[k]) for k in order]
    err = [[m-lo for m, (lo, hi) in zip(means, cis)],
           [hi-m for m, (lo, hi) in zip(means, cis)]]
    cols = [GRPO_C, DR_C, GRPO_C, DR_C]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    axes[0].bar(labels, means, yerr=err, color=cols, capsize=4)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel(r"$\rho(\mathrm{len},\mathrm{zvf})$")
    axes[0].set_title("(A) Length–ZVF coupling: sign flip by task", fontsize=9)
    # (B) tertile ZVF
    bc = load_tsv("length_zvf_bincond.tsv")
    tert = {}
    for r in bc:
        task = "arith" if r["task"].startswith("arithmetic") else "gsm8k"
        tert.setdefault((task, r["bin"]), []).append(float(r["mean_zvf"]))
    bins = ["L1", "L2", "L3"]
    a_vals = [np.mean(tert[("arith", b)]) for b in bins]
    g_vals = [np.mean(tert[("gsm8k", b)]) for b in bins]
    x = np.arange(3); wd = 0.35
    axes[1].bar(x-wd/2, a_vals, wd, label="Arithmetic", color="#e67e22")
    axes[1].bar(x+wd/2, g_vals, wd, label="GSM8K-CoT", color="#2980b9")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["low", "mid", "high"])
    axes[1].set_xlabel("length tertile")
    axes[1].set_ylabel("mean ZVF")
    axes[1].set_title("(B) Length-conditioned ZVF tertiles", fontsize=9)
    axes[1].legend(fontsize=8)
    # (C) co-evolution arithmetic GRPO seed 42
    r42 = next(r for r in arith if algo_norm(r["algo"]) == "grpo" and r["seed"] == 42)
    st = np.array([s["step"] for s in r42["step_log"]])
    L = steps(r42, "mean_comp_len"); Z = steps(r42, "zvf")
    Ln = (L - L.min()) / (L.max() - L.min())
    ax = axes[2]
    ax.plot(st, Ln, color=GRPO_C, label="mean length (norm)")
    ax.plot(st, Z, color=DR_C, label="ZVF")
    ax.set_xlabel("training step"); ax.set_ylabel("normalized value")
    ax.set_title("(C) Co-evolution, arith GRPO seed 42\nZVF up as length compresses", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    save(fig, "length_zvf_coupling")

# ---------------------------------------------------------------- Figure 4
# length_bias_reward_shape: scatter (L,R) + binned E[R|L] per algo, two panels.
def fig_reward_shape():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, runs, title in [(axL, arith, "Arithmetic (Qwen2.5-0.5B)"),
                            (axR, gsm, "GSM8K-CoT (Qwen2.5-1.5B-Instruct)")]:
        for algo, c in [("grpo", GRPO_C), ("dr_grpo", DR_C)]:
            pts = []
            for r in runs:
                if algo_norm(r["algo"]) != algo:
                    continue
                for L, R in zip(steps(r, "mean_comp_len"), steps(r, "mean_reward")):
                    pts.append((L, R))
            pts = np.array(sorted(pts))
            ax.scatter(pts[:, 0], pts[:, 1], s=8, color=c, alpha=0.18)
            q = np.array_split(pts, 4)
            xb = [b[:, 0].mean() for b in q]; yb = [b[:, 1].mean() for b in q]
            ax.plot(xb, yb, "o-", color=c, lw=2,
                    label=("GRPO" if algo == "grpo" else "Dr.GRPO"))
        ax.set_xlabel("mean completion length (tokens)")
        ax.set_ylabel("mean reward")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle(r"Reward-shape decomposition $E[R\mid L]$ (quartile-binned)", fontsize=10)
    save(fig, "length_bias_reward_shape")

# ---------------------------------------------------------------- Figure 5
# length_bias_mechanism: (A) scatter L vs R all runs, (B) OLS dL/dR bars w/ CI,
# (C) ZVF mediation proportions.
def fig_mechanism():
    per = load_tsv("length_bias_mechanism_per_run.tsv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    # (A) scatter
    for runs, mk in [(gsm, "o"), (arith, "s")]:
        for r in runs:
            c = GRPO_C if algo_norm(r["algo"]) == "grpo" else DR_C
            L, R = steps(r, "mean_comp_len"), steps(r, "mean_reward")
            axes[0].scatter(L, R, s=10, marker=mk, color=c, alpha=0.4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("mean length (tokens, log)")
    axes[0].set_ylabel("mean reward")
    axes[0].set_title("(A) L vs R, all 16 runs\n(circles=GSM8K, squares=arith)", fontsize=9)
    axes[0].grid(alpha=0.25)
    # (B) OLS dL/dR per task/algo with SD-based CI across seeds
    groups = [("arithmetic_easy_qwen2.5-0.5b", "grpo"),
              ("arithmetic_easy_qwen2.5-0.5b", "dr_grpo"),
              ("gsm8k_cot_hard_qwen2.5-1.5b", "grpo"),
              ("gsm8k_cot_hard_qwen2.5-1.5b", "dr_grpo")]
    labels = ["Arith\nGRPO", "Arith\nDr.GRPO", "GSM8K\nGRPO", "GSM8K\nDr.GRPO"]
    means, errs = [], []
    for task, algo in groups:
        vals = [float(r["ols_dL_dR"]) for r in per
                if r["task"] == task and algo_norm(r["algo"]) == algo]
        means.append(np.mean(vals))
        errs.append(np.std(vals) / max(len(vals), 1) ** 0.5 * 1.96)
    cols = [GRPO_C, DR_C, GRPO_C, DR_C]
    axes[1].bar(labels, means, yerr=errs, color=cols, capsize=4)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel(r"OLS $\widehat{dL/dR}$")
    axes[1].set_title("(B) Length–reward slope\n(arith |slope| smaller under Dr.GRPO)", fontsize=9)
    # (C) ZVF mediation proportion (indirect via ZVF) computed from pooled step data
    def mediation(runs, algo):
        Ls, Rs, Zs = [], [], []
        for r in runs:
            if algo_norm(r["algo"]) != algo:
                continue
            Ls += list(steps(r, "mean_comp_len"))
            Rs += list(steps(r, "mean_reward"))
            Zs += list(steps(r, "zvf"))
        L, R, Z = map(lambda a: (np.array(a) - np.mean(a)) / (np.std(a) + 1e-9), (Ls, Rs, Zs))
        total = np.polyfit(L, R, 1)[0]                       # R ~ L
        A = np.vstack([L, Z]).T
        beta = np.linalg.lstsq(A, R, rcond=None)[0]          # R ~ L + Z
        direct = beta[0]
        indirect = total - direct
        prop = 0.0 if abs(total) < 1e-9 else indirect / total
        return abs(direct), abs(indirect), abs(total), prop
    x = np.arange(4); wd = 0.6
    props = []
    for task_runs, algo in [(arith, "grpo"), (arith, "dr_grpo"), (gsm, "grpo"), (gsm, "dr_grpo")]:
        props.append(mediation(task_runs, algo)[3])
    axes[2].bar(labels, props, color=cols, width=wd)
    axes[2].set_ylabel("proportion of L→R mediated by ZVF")
    axes[2].set_title("(C) ZVF mediation\n(amplified on arith, bypassed on GSM8K)", fontsize=9)
    save(fig, "length_bias_mechanism")

# ---------------------------------------------------------------- Figure 6
# length_bias_iter24: (A) windowed rho drift early/late, (B) sign-flip frac late,
# (C) length-vs-step slope, (D) first-diff coupling.
def windowed_rho(run, w=10):
    L, R = steps(run, "mean_comp_len"), steps(run, "mean_reward")
    out = []
    for s in range(w, len(L) + 1):
        out.append(spearmanr(np.arange(w), R[s-w:s]).correlation if False else
                   spearmanr(L[s-w:s], R[s-w:s]).correlation)
    return np.array([o for o in out if o is not None])

def fig_iter24():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7))
    groups = [("Arith", "grpo", arith), ("Arith", "dr_grpo", arith),
              ("GSM8K", "grpo", gsm), ("GSM8K", "dr_grpo", gsm)]
    labels = ["Arith\nGRPO", "Arith\nDr.GRPO", "GSM8K\nGRPO", "GSM8K\nDr.GRPO"]
    cols = [GRPO_C, DR_C, GRPO_C, DR_C]
    early, late, signfrac, slopes, fdcoup = [], [], [], [], []
    for _, algo, runs in groups:
        e_all, l_all, sf_all, sl_all, fd_all = [], [], [], [], []
        for r in runs:
            if algo_norm(r["algo"]) != algo:
                continue
            wr = windowed_rho(r)
            if len(wr) >= 3:
                third = len(wr) // 3
                e_all.append(wr[:third].mean()); l_all.append(wr[-third:].mean())
                sf_all.append((wr[-third:] >= 0).mean())
            L = steps(r, "mean_comp_len")
            sl_all.append(np.polyfit(np.arange(len(L)), L, 1)[0])
            dL = np.diff(steps(r, "mean_comp_len")); dR = np.diff(steps(r, "mean_reward"))
            fd_all.append(spearmanr(dL, dR).correlation)
        early.append(np.mean(e_all)); late.append(np.mean(l_all))
        signfrac.append(np.mean(sf_all)); slopes.append(np.mean(sl_all))
        fdcoup.append(np.mean(fd_all))
    x = np.arange(4); wd = 0.35
    axes[0].bar(x-wd/2, early, wd, label="early", color="#95a5a6")
    axes[0].bar(x+wd/2, late, wd, label="late", color="#2c3e50")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel(r"$\rho_w(\mathrm{len},R)$")
    axes[0].set_title("(A) Windowed ρ drift", fontsize=9); axes[0].legend(fontsize=7)
    axes[1].bar(labels, signfrac, color=cols)
    axes[1].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("frac late windows ρ≥0")
    axes[1].set_title("(B) Sign-flip test", fontsize=9)
    axes[2].bar(labels, slopes, color=cols)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_xticklabels(labels, fontsize=7)
    axes[2].set_ylabel("length-vs-step slope")
    axes[2].set_title("(C) Length slope (all <0)", fontsize=9)
    axes[3].bar(labels, fdcoup, color=cols)
    axes[3].axhline(0, color="k", lw=0.8)
    axes[3].set_xticklabels(labels, fontsize=7)
    axes[3].set_ylabel(r"$\rho(\Delta L,\Delta R)$")
    axes[3].set_title("(D) First-diff coupling", fontsize=9)
    save(fig, "length_bias_iter24")

if __name__ == "__main__":
    fig_length_vs_reward()
    fig_elevated()
    fig_zvf_coupling()
    fig_reward_shape()
    fig_mechanism()
    fig_iter24()
    print("done")
