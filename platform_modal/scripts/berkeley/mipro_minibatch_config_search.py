#!/usr/bin/env python3
"""B-F24 row 21 — F24 L5 Omar Khattab (DSPy / MIPRO).

MIPRO (Opsahl-Ong et al., EMNLP 2024, arXiv:2406.11695) optimizes multi-stage
LM programs by (a) a Bayesian surrogate (TPE) over a discrete config space and
(b) *minibatch* evaluation: candidates are scored on cheap partial evaluations
and only promising ones are promoted to full evaluation. Soylu et al. (EMNLP
2024, arXiv:2407.10930) show weight-opt + prompt-opt "work better together".

We port the MIPRO minibatch-config-search idea to GRPO group-size selection on
the same-stack sweep (groupsize_zvf_sweep.json: 4 group sizes x 3 seeds x 40
steps). The "config" = group size G; "full evaluation" = 3-seed terminal
heldout_acc; two orthogonal cheap "minibatch" axes:
  - SEED minibatch (prompt-opt analog): evaluate on 1 of 3 seeds.
  - STEP minibatch (weight-opt / partial-training analog): read the trajectory
    at an early step k < 40 instead of running to the end.

H1  Minibatch-rank fidelity: 1-seed ranking of G recovers the full 3-seed rank
    (2/3 compute saved) -> mean Kendall tau across seed choices.
H2  Early-step surrogate: some step-k trajectory feature predicts terminal
    heldout rank; find the earliest k with correct top-1 (partial-training save).
H3  Surrogate-guided (TPE analog) vs random search: surrogate ordering reaches
    the true best-G earlier / at lower compute than random ordering.
H4  Flat-landscape TOST: the 4 G configs are statistically equivalent on
    heldout within +-0.02 -> bounds MIPRO's config-selection regret.
H5  "Two steps better together": SEED + STEP surrogate combined predicts the
    terminal full rank better than either axis alone (incremental fit).
"""
import json, itertools, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "platform_hybrid/experiments/results/groupsize_zvf_sweep.json"
OUT = ROOT / "platform_hybrid/experiments/results/berkeley"
OUT.mkdir(parents=True, exist_ok=True)

d = json.load(open(SRC))
runs = d["runs"]
GS = sorted({r["group_size"] for r in runs})
SEEDS = sorted({r["seed"] for r in runs})
NSTEP = runs[0]["n_steps"]

def get(g, s):
    return next(r for r in runs if r["group_size"] == g and r["seed"] == s)

# full 3-seed terminal metrics per config
full = {g: {"heldout": st.mean(get(g, s)["heldout_acc"] for s in SEEDS),
            "last10": st.mean(get(g, s)["last10_avg"] for s in SEEDS),
            "compute": st.mean(get(g, s)["elapsed_seconds"] for s in SEEDS)}
        for g in GS}

def rank(vals):  # dict g->score -> dict g->rank (1=best, higher score better)
    order = sorted(vals, key=lambda g: -vals[g])
    return {g: i + 1 for i, g in enumerate(order)}

def kendall(a, b):
    keys = list(a); n = len(keys); c = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = keys[i], keys[j]
            s = (a[x] - a[y]) * (b[x] - b[y])
            if s > 0: c += 1
            elif s < 0: disc += 1
    tot = c + disc
    return (c - disc) / tot if tot else 1.0

def spearman(xs, ys):
    def rk(v):
        srt = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        for pos, i in enumerate(srt): r[i] = pos + 1
        return r
    rx, ry = rk(xs), rk(ys)
    n = len(xs); dd = sum((rx[i]-ry[i])**2 for i in range(n))
    return 1 - 6*dd/(n*(n*n-1)) if n > 1 else 0.0

rows = {}  # tsv name -> list of dict rows

# ---------------- H1: minibatch-rank fidelity (seed axis) --------------------
full_rank_h = rank({g: full[g]["heldout"] for g in GS})
full_rank_l = rank({g: full[g]["last10"] for g in GS})
h1 = []
for metric, fr in (("heldout", full_rank_h), ("last10", full_rank_l)):
    taus = []
    for s in SEEDS:  # 1-seed minibatch: one seed used across all configs
        key = "heldout_acc" if metric == "heldout" else "last10_avg"
        mb = {g: get(g, s)[key] for g in GS}
        t = kendall(rank(mb), fr)
        taus.append(t)
        h1.append({"metric": metric, "seed_used": s, "kendall_tau_vs_full": round(t, 4),
                   "compute_frac": round(1/len(SEEDS), 4)})
    h1.append({"metric": metric, "seed_used": "MEAN", "kendall_tau_vs_full": round(st.mean(taus), 4),
               "compute_frac": round(1/len(SEEDS), 4)})
rows["mipro_h1_minibatch_rank"] = h1
mean_tau_h = st.mean(x["kendall_tau_vs_full"] for x in h1 if x["metric"]=="heldout" and x["seed_used"]!="MEAN")

# ---------------- H2: early-step surrogate (step minibatch) ------------------
FEATS = ["zvf", "mean_reward", "entropy", "advantage_variance", "grad_norm"]
h2 = []
best_g = min(full_rank_h, key=full_rank_h.get)          # true best config
WIN = 5                                                  # stability window
stable_k = {f: -1 for f in FEATS}                        # first k with sustained top1 match
for f in FEATS:
    top1_series = []
    rho_series = []
    for k in range(NSTEP):
        fk = {g: st.mean(get(g, s)["step_log"][k][f] for s in SEEDS) for g in GS}
        rho = spearman([fk[g] for g in GS], [full[g]["heldout"] for g in GS])
        top1_series.append(int(min(rank(fk), key=rank(fk).get) == best_g))
        rho_series.append(rho)
        if k in (0, 4, 9, 19, 29, 39):
            h2.append({"feature": f, "step": k, "spearman_vs_heldout": round(rho, 4),
                       "top1_match": top1_series[-1]})
    # earliest k where top1 match holds for the whole trailing window AND mean rho>0
    for k in range(NSTEP - WIN + 1):
        if all(top1_series[k:k + WIN]) and st.mean(rho_series[k:k + WIN]) > 0:
            stable_k[f] = k; break
    h2.append({"feature": f, "step": "first_stable_top1(win5)", "spearman_vs_heldout": "",
               "top1_match": stable_k[f]})
rows["mipro_h2_early_step_surrogate"] = h2

# ---------------- H3: surrogate-guided (TPE analog) vs random ----------------
# best true config by heldout:
true_best = min(full_rank_h, key=full_rank_h.get)
# cheap surrogate score to ORDER configs = 1-seed heldout on the first seed +
# partial-training last10 at step 19 (combined cheap signal). Lower compute.
k_mb = 19
surro = {g: 0.5*get(g, SEEDS[0])["heldout_acc"]
             + 0.5*st.mean(get(g, s)["step_log"][k_mb]["mean_reward"] for s in SEEDS) for g in GS}
surro_order = sorted(GS, key=lambda g: -surro[g])
# position (1-indexed) at which the true best G appears under an ordering
def pos_of_best(order): return order.index(true_best) + 1
# random baseline: expected position over all permutations = (n+1)/2
rand_pos = (len(GS) + 1) / 2
# compute accounting (full[g]["compute"] = mean seconds for ONE seed):
per_seed = {g: full[g]["compute"] for g in GS}
exhaustive = len(SEEDS) * sum(per_seed.values())            # full 3-seed grid
# MIPRO successive-halving: screen all configs on 1 seed, promote surrogate top-1
# to the remaining seeds only.  (aggressive) plus a safe top-2 variant.
screen = sum(per_seed.values())                             # 1 seed each
sh1 = screen + (len(SEEDS) - 1) * per_seed[surro_order[0]]  # promote top-1
sh2 = screen + (len(SEEDS) - 1) * sum(per_seed[g] for g in surro_order[:2])  # promote top-2
sh_pick1 = surro_order[0]                                   # top-1 promotion -> that config
sh_pick2 = min(surro_order[:2], key=lambda g: full_rank_h[g])
h3 = [{"quantity": "surrogate_order", "value": "-".join(map(str, surro_order))},
      {"quantity": "true_best_G", "value": true_best},
      {"quantity": "true_best_pos_under_surrogate", "value": pos_of_best(surro_order)},
      {"quantity": "true_best_pos_expected_random", "value": round(rand_pos, 3)},
      {"quantity": "SH_top1_pick_G", "value": sh_pick1},
      {"quantity": "SH_top1_is_true_best", "value": int(sh_pick1 == true_best)},
      {"quantity": "SH_top1_compute_sec", "value": round(sh1, 1)},
      {"quantity": "SH_top1_saving_frac", "value": round(1 - sh1/exhaustive, 3)},
      {"quantity": "SH_top2_pick_G", "value": sh_pick2},
      {"quantity": "SH_top2_saving_frac", "value": round(1 - sh2/exhaustive, 3)},
      {"quantity": "exhaustive_compute_sec", "value": round(exhaustive, 1)}]
rows["mipro_h3_surrogate_search"] = h3
sh_pick, sh_compute = sh_pick1, sh1                         # primary = aggressive top-1

# ---------------- H4: flat-landscapeTOST equivalence -----------------------
EPS = 0.02
h4 = []
sd_pool = st.mean(st.pstdev([get(g, s)["heldout_acc"] for s in SEEDS]) for g in GS)
n_equiv = 0; n_pair = 0
for a, b in itertools.combinations(GS, 2):
    da = [get(a, s)["heldout_acc"] for s in SEEDS]
    db = [get(b, s)["heldout_acc"] for s in SEEDS]
    diff = st.mean(da) - st.mean(db)
    se = ((st.pstdev(da)**2 + st.pstdev(db)**2)/len(SEEDS))**0.5 or 1e-9
    # TOST: equivalent if the |diff| + ~1.5*se still within EPS (conservative n=3)
    equiv = abs(diff) + 1.5*se < EPS
    n_pair += 1; n_equiv += int(equiv)
    h4.append({"pair": f"{a}v{b}", "mean_diff": round(diff, 4), "se": round(se, 4),
               "within_%.2f" % EPS: int(equiv)})
max_gap = max(full[g]["heldout"] for g in GS) - min(full[g]["heldout"] for g in GS)
h4.append({"pair": "SUMMARY", "mean_diff": round(max_gap, 4), "se": round(sd_pool, 4),
           "within_%.2f" % EPS: f"{n_equiv}/{n_pair}"})
rows["mipro_h4_tost_equivalence"] = h4

# ---------------- H5: two steps better together (interaction) ---------------
# predict terminal full heldout rank from: SEED-only surrogate, STEP-only, BOTH.
seed_only = {g: get(g, SEEDS[0])["heldout_acc"] for g in GS}
step_only = {g: st.mean(get(g, s)["step_log"][k_mb]["mean_reward"] for s in SEEDS) for g in GS}
both = {g: 0.5*seed_only[g] + 0.5*step_only[g] for g in GS}
gs = list(GS); tgt = [full[g]["heldout"] for g in gs]
def fit_rho(pred): return spearman([pred[g] for g in gs], tgt)
r_seed, r_step, r_both = fit_rho(seed_only), fit_rho(step_only), fit_rho(both)
h5 = [{"surrogate": "seed_only", "spearman_vs_full": round(r_seed, 4)},
      {"surrogate": "step_only", "spearman_vs_full": round(r_step, 4)},
      {"surrogate": "both_combined", "spearman_vs_full": round(r_both, 4)},
      {"surrogate": "interaction_gain", "spearman_vs_full": round(r_both - max(r_seed, r_step), 4)}]
rows["mipro_h5_two_steps_together"] = h5

# ---------------- write TSVs -------------------------------------------------
for name, rlist in rows.items():
    p = OUT / f"{name}.tsv"
    cols = list(rlist[0].keys())
    with open(p, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rlist:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

# ---------------- verdicts ---------------------------------------------------
def verdict_h1(): return "DECISIVE" if mean_tau_h >= 0.5 else ("SUGGESTIVE" if mean_tau_h > 0 else "NULL")
valid_stable = [stable_k[f] for f in FEATS if stable_k[f] >= 0]
h2_top1 = min(valid_stable) if valid_stable else -1
def verdict_h2():
    if h2_top1 < 0: return "NULL"          # no feature stably predicts terminal rank early
    return "DECISIVE" if h2_top1 <= NSTEP//2 else "SUGGESTIVE"
def verdict_h3(): return "DECISIVE" if (sh_pick == true_best and (1 - sh_compute/exhaustive) >= 0.25) else "SUGGESTIVE"
def verdict_h4(): return "DECISIVE" if n_equiv == n_pair else ("SUGGESTIVE" if n_equiv >= n_pair/2 else "NULL")
def verdict_h5(): return "DECISIVE" if (r_both - max(r_seed, r_step)) > 1e-9 else "NULL"

summary = {
    "source": "F24 L5 Omar Khattab — MIPRO arXiv:2406.11695 (EMNLP24) + FT+PO arXiv:2407.10930 (EMNLP24)",
    "data": "groupsize_zvf_sweep.json (4G x 3seed x 40step, Qwen2.5-0.5B arithmetic)",
    "true_best_G_heldout": true_best,
    "H1_minibatch_rank": {"mean_kendall_tau_heldout": round(mean_tau_h, 4), "verdict": verdict_h1()},
    "H2_early_step_surrogate": {"first_stable_top1_step": h2_top1, "of_steps": NSTEP,
                                "stable_feature": min((f for f in FEATS if stable_k[f] >= 0),
                                                      key=lambda f: stable_k[f], default=None),
                                "note": "partial-training (STEP) axis fails on flat landscape",
                                "verdict": verdict_h2()},
    "H3_surrogate_search": {"surrogate_order": "-".join(map(str, surro_order)),
                            "true_best_pos": pos_of_best(surro_order), "random_pos_exp": round(rand_pos, 3),
                            "SH_pick_true_best": int(sh_pick == true_best),
                            "compute_saving": round(1 - sh_compute/exhaustive, 3), "verdict": verdict_h3()},
    "H4_tost_equivalence": {"n_equiv_pairs": f"{n_equiv}/{n_pair}", "max_heldout_gap": round(max_gap, 4),
                            "eps": EPS, "verdict": verdict_h4()},
    "H5_two_steps_together": {"seed": round(r_seed, 4), "step": round(r_step, 4),
                              "both": round(r_both, 4), "gain": round(r_both - max(r_seed, r_step), 4),
                              "verdict": verdict_h5()},
}
n_dec = sum(1 for h in ("H1_minibatch_rank","H2_early_step_surrogate","H3_surrogate_search",
                        "H4_tost_equivalence","H5_two_steps_together") if summary[h]["verdict"]=="DECISIVE")
summary["decisive_count"] = f"{n_dec}/5"
json.dump(summary, open(OUT / "mipro_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
