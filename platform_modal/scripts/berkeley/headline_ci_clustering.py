#!/usr/bin/env python3
"""
Berkeley F25 L8 (Sida Wang, Meta) -- "Predictable Noise in LLMs / Adding Error
Bars to Evals" operationalised as a SEED-CLUSTERING / DESIGN-EFFECT audit of
TinkerRL-Bench headline numbers across all 4 pillars.

Row 07 (Miller arXiv:2411.00640) put simple bootstrap CIs on 7 headline numbers
treating the available observations as i.i.d.  Sida Wang's distinct lecture point
is that eval noise is *structured*: training-step metrics within one seed are
strongly autocorrelated, so pooling S seeds x M steps and treating the S*M rows
as independent inflates the effective sample size and yields a FALSELY NARROW CI.
The honest unit of replication is the seed (the cluster), not the step.

For every headline metric with real (seed x step) data we compute:
  - point estimate (grand mean of per-seed means)
  - naive pooled 95% CI      (bootstrap over all S*M rows as if i.i.d.)
  - seed-clustered 95% CI     (cluster bootstrap: resample seeds, use seed means)
  - ICC  = between-seed var / total var  (one-way random-effects ANOVA)
  - DEFF = 1 + (m_bar - 1) * ICC          (Kish design effect)
  - n_eff = n_pooled / DEFF               (honest effective n)
  - CI-width inflation = width_naive / width_cluster
  - verdict: HONEST (DEFF < 2) vs INFLATED (naive CI understates uncertainty)

This directly closes the open thread flagged in row-07 H6 ("n=52 pooled across 3
experiments, not pure seeds -> Miller would call for cluster sensitivity").

Outputs (experiments/results/berkeley/):
  headline_ci_clustering.tsv        one row per audited headline
  headline_ci_clustering_icc.tsv    variance-component breakdown
  headline_ci_clustering_summary.json

All CIs computed from REAL repo data. Deterministic (numpy default_rng(0)).
"""
import csv, json, os, math
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "berkeley")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
B = 10000  # bootstrap resamples


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def icc_oneway(groups):
    """One-way random-effects ICC(1) + Kish DEFF from a list of per-seed arrays."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    k = len(groups)
    ni = [len(g) for g in groups]
    N = sum(ni)
    grand = np.concatenate(groups).mean()
    # mean squares
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    # m0 (average cluster size, harmonic-style correction for unbalanced)
    m0 = (N - sum(n ** 2 for n in ni) / N) / (k - 1)
    var_b = max(0.0, (msb - msw) / m0)
    icc = var_b / (var_b + msw) if (var_b + msw) > 0 else 0.0
    m_bar = N / k
    deff = 1.0 + (m_bar - 1.0) * icc
    return dict(icc=icc, deff=deff, m_bar=m_bar, N=N, k=k,
                var_between=var_b, var_within=msw, n_eff=N / deff if deff > 0 else N)


def boot_ci_pooled(pooled):
    """Naive: resample all N pooled observations i.i.d."""
    pooled = np.asarray(pooled, float)
    idx = RNG.integers(0, len(pooled), size=(B, len(pooled)))
    means = pooled[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_ci_cluster(groups):
    """Cluster bootstrap: resample SEEDS with replacement, mean of seed-means."""
    seed_means = np.array([np.mean(g) for g in groups], float)
    k = len(seed_means)
    idx = RNG.integers(0, k, size=(B, k))
    means = seed_means[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_cluster_ci(seed_a, seed_b):
    """Paired-by-seed delta (a-b): cluster bootstrap over seeds of the per-seed diff."""
    d = np.array(seed_a, float) - np.array(seed_b, float)
    k = len(d)
    idx = RNG.integers(0, k, size=(B, k))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(deff):
    if deff < 1.5:
        return "HONEST"
    if deff < 3.0:
        return "MILD_INFLATION"
    return "INFLATED"


# ----------------------------------------------------------------------------
# data loaders -> {seed: [obs...]} for a chosen metric/window
# ----------------------------------------------------------------------------
def load_samestack():
    d = json.load(open(os.path.join(RES, "samestack_ppo_grpo.json")))
    out = defaultdict(dict)  # algo -> seed -> {last10:[...], all:[...], heldout:float}
    for run in d["runs"]:
        algo, seed = run["algo"], str(run["seed"])
        rewards = [s["mean_reward"] for s in run["step_log"]]
        out[algo][seed] = dict(all=rewards, last10=rewards[-10:],
                               heldout=run["heldout_acc"], last10_avg=run["last10_avg"])
    return out


def load_group_size(metric):
    rows = list(csv.DictReader(open(os.path.join(RES, "group_size_advantage_variance.tsv")),
                               delimiter="\t"))
    by = defaultdict(lambda: defaultdict(list))  # G -> seed -> [metric over steps]
    for r in rows:
        by[r["G"]][r["seed"]].append(float(r[metric]))
    return by


def load_bfcl():
    rows = list(csv.DictReader(open(os.path.join(RES, "bfclv4_tool_use.tsv")),
                               delimiter="\t"))
    dense = defaultdict(list); sparse = defaultdict(list)
    for r in rows:
        dense[r["seed"]].append(float(r["reward_dense"]))
        sparse[r["seed"]].append(float(r["reward_sparse"]))
    return dense, sparse


# ----------------------------------------------------------------------------
# audit rows
# ----------------------------------------------------------------------------
audit = []   # master table
icc_rows = []


def add_mean_headline(mid, pillar, claim, groups_dict, window_key=None):
    """groups_dict: seed -> list(obs). Build naive vs clustered CI + ICC."""
    groups = list(groups_dict.values())
    pooled = [x for g in groups for x in g]
    point = float(np.mean([np.mean(g) for g in groups]))  # grand mean of seed means
    lo_n, hi_n = boot_ci_pooled(pooled)
    lo_c, hi_c = boot_ci_cluster(groups)
    ic = icc_oneway(groups)
    w_n = hi_n - lo_n; w_c = hi_c - lo_c
    infl = w_c / w_n if w_n > 0 else float("nan")
    audit.append(dict(
        metric_id=mid, pillar=pillar, headline=claim, point=round(point, 5),
        n_seeds=len(groups), n_pooled=len(pooled),
        ci_naive=f"[{lo_n:.4f}, {hi_n:.4f}]", w_naive=round(w_n, 5),
        ci_cluster=f"[{lo_c:.4f}, {hi_c:.4f}]", w_cluster=round(w_c, 5),
        icc=round(ic["icc"], 4), deff=round(ic["deff"], 3),
        n_eff=round(ic["n_eff"], 2), width_inflation=round(infl, 3),
        verdict=verdict(ic["deff"])))
    icc_rows.append(dict(metric_id=mid, pillar=pillar,
                         var_between=round(ic["var_between"], 6),
                         var_within=round(ic["var_within"], 6),
                         icc=round(ic["icc"], 4), m_bar=round(ic["m_bar"], 1),
                         deff=round(ic["deff"], 3), n_eff=round(ic["n_eff"], 2)))


# ---- Pillar 1: same-stack PPO vs GRPO (last-10 mean_reward per seed) ----
ss = load_samestack()
grpo10 = {s: v["last10"] for s, v in ss["grpo"].items()}
ppo10 = {s: v["last10"] for s, v in ss["ppo"].items()}
add_mean_headline("P1_grpo_last10", "P1",
                  "GRPO same-stack last-10 mean_reward = 0.979 (paper: 0.9789+/-0.0067)",
                  grpo10)
add_mean_headline("P1_ppo_last10", "P1",
                  "PPO same-stack last-10 mean_reward = 0.918 (paper: 0.9181+/-0.0497)",
                  ppo10)

# paired GRPO-PPO delta (the p=0.75 headline) -- correct unit is the seed
seeds = sorted(set(grpo10) & set(ppo10))
a = [np.mean(grpo10[s]) for s in seeds]
b = [np.mean(ppo10[s]) for s in seeds]
pd_pt, pd_lo, pd_hi = paired_cluster_ci(a, b)
# naive pooled paired: pool all step-level diffs as if iid (5x10=50)
pooled_diff = [x - y for s in seeds
               for x, y in zip(grpo10[s], ppo10[s])]
lo_pn, hi_pn = boot_ci_pooled(pooled_diff)
audit.append(dict(
    metric_id="P1_grpo_minus_ppo_paired", pillar="P1",
    headline="GRPO-PPO paired last10 delta = +0.061 pt (equivalence claim, p=0.75 in paper on heldout)",
    point=round(pd_pt, 5), n_seeds=len(seeds), n_pooled=len(pooled_diff),
    ci_naive=f"[{lo_pn:.4f}, {hi_pn:.4f}]", w_naive=round(hi_pn - lo_pn, 5),
    ci_cluster=f"[{pd_lo:.4f}, {pd_hi:.4f}]", w_cluster=round(pd_hi - pd_lo, 5),
    icc="paired", deff=round((hi_pn - lo_pn) and (pd_hi - pd_lo) / (hi_pn - lo_pn), 3),
    n_eff=len(seeds), width_inflation=round((pd_hi - pd_lo) / (hi_pn - lo_pn), 3)
    if (hi_pn - lo_pn) > 0 else float("nan"),
    verdict="CLUSTER_REQUIRED (delta CI must straddle 0 for equivalence)"))

# ---- Pillar 3: group-size headline metrics (per-seed step traces) ----
gz_reward = load_group_size("mean_reward")
gz_zvf = load_group_size("zvf")
add_mean_headline("P3_reward_G2", "P3",
                  "Group-size G=2 mean_reward (low-G endpoint of +24% swing claim)",
                  gz_reward["2"])
add_mean_headline("P3_reward_G16", "P3",
                  "Group-size G=16 mean_reward (high-G endpoint of +24% swing claim)",
                  gz_reward["16"])
add_mean_headline("P3_zvf_G2", "P3",
                  "ZVF at G=2 = 0.845 (frontier r2 ZVF-decay endpoint)",
                  gz_zvf["2"])
add_mean_headline("P3_zvf_G16", "P3",
                  "ZVF at G=16 = 0.631 (frontier r2 ZVF-decay endpoint)",
                  gz_zvf["16"])

# ---- Pillar 4: bfclv4 dense vs sparse tool-use reward ----
dense, sparse = load_bfcl()
add_mean_headline("P4_bfcl_dense", "P4",
                  "bfclv4 dense-shaped tool-use reward = 0.186 (ReAct row-13 H1)", dense)
add_mean_headline("P4_bfcl_sparse", "P4",
                  "bfclv4 sparse tool-use reward = 0.113 (ReAct row-13 H1)", sparse)

# ----------------------------------------------------------------------------
# write outputs
# ----------------------------------------------------------------------------
cols = ["metric_id", "pillar", "headline", "point", "n_seeds", "n_pooled",
        "ci_naive", "w_naive", "ci_cluster", "w_cluster", "icc", "deff",
        "n_eff", "width_inflation", "verdict"]
with open(os.path.join(OUT, "headline_ci_clustering.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
    w.writeheader()
    for r in audit:
        w.writerow(r)

icc_cols = ["metric_id", "pillar", "var_between", "var_within", "icc", "m_bar",
            "deff", "n_eff"]
with open(os.path.join(OUT, "headline_ci_clustering_icc.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=icc_cols, delimiter="\t")
    w.writeheader()
    for r in icc_rows:
        w.writerow(r)

# summary + headline verdicts
mean_rows = [r for r in audit if isinstance(r["deff"], (int, float))]
inflated = [r for r in mean_rows if r["deff"] >= 1.5]
max_infl = max(mean_rows, key=lambda r: r["deff"])
summary = dict(
    source="Berkeley F25 L8 Sida Wang -- Predictable Noise / Adding Error Bars to Evals",
    method="seed-clustering design-effect audit (ICC + Kish DEFF + cluster bootstrap)",
    n_headlines_audited=len(audit),
    n_pooled_CI_inflated=len(inflated),
    max_deff=dict(metric=max_infl["metric_id"], deff=max_infl["deff"],
                  icc=max_infl["icc"], width_inflation=max_infl["width_inflation"],
                  ci_naive=max_infl["ci_naive"], ci_cluster=max_infl["ci_cluster"]),
    mean_deff=round(float(np.mean([r["deff"] for r in mean_rows])), 3),
    mean_icc=round(float(np.mean([r["icc"] for r in mean_rows
                                  if isinstance(r["icc"], (int, float))])), 4),
    paired_delta_equivalence=dict(
        metric="P1_grpo_minus_ppo_paired", point=round(pd_pt, 5),
        cluster_ci=[round(pd_lo, 5), round(pd_hi, 5)],
        straddles_zero=bool(pd_lo <= 0 <= pd_hi),
        naive_pooled_ci=[round(lo_pn, 5), round(hi_pn, 5)],
        naive_straddles_zero=bool(lo_pn <= 0 <= hi_pn)),
    key_finding=(
        "Training-step metrics carry high within-seed autocorrelation: pooling "
        "S seeds x M steps as i.i.d. understates headline-CI width by up to "
        f"{max_infl['width_inflation']}x (DEFF={max_infl['deff']}, "
        f"n_eff={max_infl['n_eff']} vs n_pooled={max_infl['n_pooled']}). "
        "Every TinkerRL-Bench headline must report SEED-CLUSTERED error bars; "
        "the papers that already aggregate per-seed then take seed-SE are correct, "
        "and this audit quantifies the trap for any pooled-CI reporting (closes "
        "row-07 H6 cluster-sensitivity thread)."),
)
json.dump(summary, open(os.path.join(OUT, "headline_ci_clustering_summary.json"), "w"),
          indent=2)

# console
print("=== HEADLINE CI CLUSTERING AUDIT ===")
for r in audit:
    print(f"{r['metric_id']:28s} {r['pillar']:3s} pt={r['point']:<9} "
          f"naive={r['ci_naive']:20s} cluster={r['ci_cluster']:20s} "
          f"DEFF={r['deff']} infl={r['width_inflation']}x -> {r['verdict']}")
print("\n--- ICC / variance components ---")
for r in icc_rows:
    print(f"{r['metric_id']:28s} ICC={r['icc']:<7} DEFF={r['deff']:<6} "
          f"n_eff={r['n_eff']:<6} (var_b={r['var_between']}, var_w={r['var_within']})")
print("\nSUMMARY:", json.dumps(summary["max_deff"], indent=0))
print("paired equivalence:", summary["paired_delta_equivalence"])
print(f"\n{len(inflated)}/{len(mean_rows)} pooled-CI headlines are >=1.5x DEFF-inflated")
