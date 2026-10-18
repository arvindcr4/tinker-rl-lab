#!/usr/bin/env python3
"""Independent recompute of every number retained in paper_P4_length_bias (Deep Review pass 2026-10-18).

Inputs (raw Modal run outputs, per-seed step logs):
  platform_hybrid/experiments/results/drgrpo_vs_grpo.json        (modal_drgrpo_vs_grpo.py)
  platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json (modal_drgrpo_gsm8k_cot.py)
Optional cross-check: W&B inventory dump (modal-open-stack mirror of the Modal volume files).
Seeds are the unit of analysis throughout; step-level points are never treated as replicates.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "platform_hybrid" / "experiments" / "results"
OUT = RES / "review_2026-10-18" / "P4_length_bias"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    "arithmetic": (RES / "drgrpo_vs_grpo.json", 10),  # (file, MAX_NEW cap in driver)
    "gsm8k_cot": (RES / "drgrpo_gsm8k_cot_full.json", 200),
}


def spearman(x, y):
    return float(stats.spearmanr(x, y).statistic)


def exact_perm_p(a, b):
    """Two-sided exact label-permutation p for difference in means (seed = unit)."""
    a, b = np.asarray(a), np.asarray(b)
    pool = np.concatenate([a, b])
    obs = abs(b.mean() - a.mean())
    n = len(a)
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        m = np.zeros(len(pool), bool)
        m[list(idx)] = True
        d = abs(pool[~m].mean() - pool[m].mean())
        hits += d >= obs - 1e-12
        tot += 1
    return hits / tot, tot


def mde_d(n1, n2, alpha=0.05, power=0.8):
    """Smallest Cohen's d detectable by a two-sided pooled two-sample t test."""
    df = n1 + n2 - 2
    tc = stats.t.ppf(1 - alpha / 2, df)
    lo, hi = 0.0, 20.0
    for _ in range(200):
        d = (lo + hi) / 2
        nc = d * np.sqrt(n1 * n2 / (n1 + n2))
        pw = 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
        lo, hi = (d, hi) if pw < power else (lo, d)
    return hi


rows, summary, comparisons = [], [], []
for task, (path, cap) in SOURCES.items():
    runs = json.load(open(path))["runs"]
    per_algo = {}
    for r in runs:
        sl = r["step_log"]
        step = [s["step"] for s in sl]
        L = [s["mean_comp_len"] for s in sl]
        R = [s["mean_reward"] for s in sl]
        row = dict(
            task=task,
            algo=r["algo"],
            seed=r["seed"],
            model=r["model"],
            n_steps=len(sl),
            cap=cap,
            rho_step_len=spearman(step, L),
            rho_step_reward=spearman(step, R),
            rho_len_reward=spearman(L, R),
            last10_reward=float(np.mean(R[-10:])),
            len_step0=L[0],
            len_first5=float(np.mean(L[:5])),
            len_last5=float(np.mean(L[-5:])),
            len_min=min(L),
            len_max=max(L),
            max_len_over_cap=max(L) / cap,
        )
        rows.append(row)
        per_algo.setdefault(r["algo"], []).append(row)
    for algo, rs in per_algo.items():
        s = dict(task=task, algo=algo, n=len(rs))
        for k in [
            "rho_step_len",
            "rho_step_reward",
            "rho_len_reward",
            "last10_reward",
            "len_step0",
            "len_first5",
            "len_last5",
            "max_len_over_cap",
        ]:
            v = np.array([x[k] for x in rs])
            s[k + "_mean"] = float(v.mean())
            s[k + "_sd"] = float(v.std(ddof=1))
        s["n_runs_rho_step_len_neg"] = int(sum(x["rho_step_len"] < 0 for x in rs))
        summary.append(s)
    g = {a: per_algo[a] for a in ("grpo", "dr_grpo")}
    for k in ["rho_step_len", "rho_len_reward", "len_last5"]:
        a = [x[k] for x in g["grpo"]]
        b = [x[k] for x in g["dr_grpo"]]
        w = stats.ttest_ind(b, a, equal_var=False)
        n1, n2 = len(a), len(b)
        v1, v2 = np.var(a, ddof=1) / n1, np.var(b, ddof=1) / n2
        df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
        pp, nperm = exact_perm_p(a, b)
        comparisons.append(
            dict(
                task=task,
                metric=k,
                n_per_arm=n1,
                diff_dr_minus_grpo=float(np.mean(b) - np.mean(a)),
                welch_t=float(w.statistic),
                welch_df=float(df),
                welch_p=float(w.pvalue),
                exact_perm_p=pp,
                n_perms=nperm,
                mde_d_80pct=mde_d(n1, n2),
            )
        )


# Held-out accuracy, seed = unit. Arithmetic: post-training only (driver has no pre-RL eval).
# GSM8K-CoT: same 200 held-out items evaluated greedily before and after training in each run.
heldout = []


def t_ci(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return float(v.mean()), float(se), float(v.mean() - h), float(v.mean() + h)


ar = {
    a: [
        r["heldout_acc"]
        for r in json.load(open(SOURCES["arithmetic"][0]))["runs"]
        if r["algo"] == a
    ]
    for a in ("grpo", "dr_grpo")
}
gs_runs = json.load(open(SOURCES["gsm8k_cot"][0]))["runs"]
gs = {
    a: [r["heldout_post_acc"] - r["heldout_pre_acc"] for r in gs_runs if r["algo"] == a]
    for a in ("grpo", "dr_grpo")
}
for a in ("grpo", "dr_grpo"):
    m, se, lo, hi = t_ci(ar[a])
    heldout.append(
        dict(
            task="arithmetic",
            quantity=f"{a}_post_acc",
            n=len(ar[a]),
            mean=m,
            se=se,
            ci_lo=lo,
            ci_hi=hi,
            df=len(ar[a]) - 1,
        )
    )
    m, se, lo, hi = t_ci(gs[a])
    heldout.append(
        dict(
            task="gsm8k_cot",
            quantity=f"{a}_gain_post_minus_pre",
            n=len(gs[a]),
            mean=m,
            se=se,
            ci_lo=lo,
            ci_hi=hi,
            df=len(gs[a]) - 1,
        )
    )
for task, d in (("arithmetic", ar), ("gsm8k_cot", gs)):
    a, b = d["grpo"], d["dr_grpo"]
    w = stats.ttest_ind(b, a, equal_var=False)
    v1, v2 = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    df = (v1 + v2) ** 2 / (v1**2 / (len(a) - 1) + v2**2 / (len(b) - 1))
    pp, nperm = exact_perm_p(a, b)
    heldout.append(
        dict(
            task=task,
            quantity="dr_minus_grpo",
            n=len(a),
            mean=float(np.mean(b) - np.mean(a)),
            se=float(np.sqrt(v1 + v2)),
            ci_lo=float(w.statistic),
            ci_hi=float(w.pvalue),
            df=float(df),
        )
    )
    heldout[-1]["note"] = f"ci_lo=Welch t, ci_hi=Welch p; exact perm p={pp:.3f} over {nperm}"
for h in heldout:
    h.setdefault("note", "t-based 95% CI over seeds")


def dump(name, recs):
    keys = list(recs[0].keys())
    with open(OUT / name, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in recs:
            f.write(
                "\t".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys)
                + "\n"
            )


dump("per_run.tsv", rows)
dump("summary.tsv", summary)
dump("seed_level_comparisons.tsv", comparisons)
dump("heldout_seed_level.tsv", heldout)
for h in heldout:
    print(h)

# Cross-check against the shipped length_bias_summary.tsv (paper Table 1 source).
shipped = {}
for line in list(open(RES / "length_bias_summary.tsv"))[1:]:
    f = line.rstrip("\n").split("\t")
    shipped[("arithmetic" if f[0].startswith("arith") else "gsm8k_cot", f[1])] = f
mism = []
for s in summary:
    f = shipped[(s["task"], s["algo"])]
    for col, key in [
        (3, "rho_step_len_mean"),
        (4, "rho_step_len_sd"),
        (5, "rho_step_reward_mean"),
        (6, "rho_step_reward_sd"),
        (7, "rho_len_reward_mean"),
        (8, "rho_len_reward_sd"),
        (9, "last10_reward_mean"),
    ]:
        if abs(float(f[col]) - s[key]) > 6e-4:
            mism.append((s["task"], s["algo"], key, f[col], s[key]))

# Optional W&B mirror cross-check (read-only inventory dump).
wb = Path(sys.argv[1]) if len(sys.argv) > 1 else None
wb_checked = wb_mism = 0
if wb and wb.exists():
    idx = {}
    for r in json.load(open(wb)):
        if r.get("project") == "modal-open-stack" and "drgrpo" in r.get("name", ""):
            idx[r["name"]] = r["summary"]
    for task, (path, _) in SOURCES.items():
        pref = (
            "tinkerrl-results--drgrpo-"
            if task == "arithmetic"
            else "tinkerrl-results--drgrpo_gsm8k-"
        )
        for r in json.load(open(path))["runs"]:
            w = idx.get(f"{pref}{r['algo']}_s{r['seed']}")
            if w is None:
                continue
            wb_checked += 1
            if (
                abs(w["last10_avg"] - r["last10_avg"]) > 1e-9
                or abs(w["elapsed_seconds"] - r["elapsed_seconds"]) > 1e-6
            ):
                wb_mism += 1

with open(OUT / "README.txt", "w") as f:
    f.write(f"table1_mismatches_vs_length_bias_summary.tsv={mism}\n")
    f.write(f"wandb_mirror_runs_checked={wb_checked} mismatches={wb_mism}\n")
print("Table-1 mismatches:", mism)
print("W&B mirror checked:", wb_checked, "mismatches:", wb_mism)
for s in summary:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
for c in comparisons:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
