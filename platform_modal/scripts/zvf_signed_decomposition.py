#!/usr/bin/env python3
"""Iter 58 - Pillar 2 (ZVF): Signed decomposition into starvation (ZVF-) vs
saturation (ZVF+).

Raw ZVF = fraction of groups with zero within-group reward variance. It sums two
structurally opposite collisions:
  * ZVF-  : all-wrong groups (K=0)  -> gradient starvation, pathological.
  * ZVF+  : all-correct groups (K=G) -> task mastered, benign saturation.
Raw ZVF is therefore non-monotone with training health (iter54: 0 perfect
separators, wide CI). This script computes the SIGNED decomposition, validates it
exactly on GSM8K per-group traces, applies an i.i.d. odds-split where only
(zvf, mean_reward, G) are logged, and shows ZVF- restores the health ordering
that raw ZVF inverts. It also asks whether AERO cuts the pathological half.

Frontier synthesis (Round 2): ZVF is a *censored contrast probability*, not
difficulty; the clean formalization splits the collision mass by the sign of the
saturated reward. This script operationalizes that split.

Inputs (all under platform_hybrid/experiments/results/):
  tinker_gsm8k_zvf_s*.json          exact per-group rewards  (converged anchor)
  variance_mitigation.tsv           per-step zvf+reward, G=8  (plateau; AERO...)
  groupsize_zvf_sweep.json          per-step zvf+reward, G=2..16 (converged)
  drgrpo_gsm8k_cot_full.json        per-step zvf+reward (struggling low-acc)
  tool_code_reward_diagnostics.tsv  zvf=1.0 at reward 0 (collapse)
Outputs:
  platform_hybrid/experiments/results/zvf_signed_summary.tsv
  platform_hybrid/experiments/results/zvf_signed_failure_corr.tsv
  platform_hybrid/experiments/results/zvf_signed_aero.tsv
"""
import csv, glob, json, math, os, statistics as st
from collections import defaultdict

RES = "platform_hybrid/experiments/results"


def odds_split(zvf, p, G):
    """Partition observed collision mass zvf into all-wrong vs all-correct using
    the i.i.d. odds of a zero-variance group being all-wrong: (1-p)^G vs p^G.
    Exact in the homogeneous-difficulty limit; a lower bound on ZVF- at high p
    (difficulty heterogeneity concentrates all-wrong mass -- validated below)."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    w, c = (1 - p) ** G, p ** G
    tot = w + c
    if tot <= 0:
        return zvf, 0.0
    return zvf * w / tot, zvf * c / tot


def classify(peak, last10):
    if peak > 0.7 and last10 < 0.35:
        return "collapse"
    if peak < 0.5:
        return "plateau"
    if last10 < 0.85 * peak:
        return "drift"
    return "converged"


rows = []  # summary rows

# ---- 1. GSM8K exact signed split (validation anchor) -----------------------
gsm_exact = []
for f in sorted(glob.glob(f"{RES}/tinker_gsm8k_zvf_s*.json")):
    if "summary" in f:
        continue
    d = json.load(open(f))
    G = d["group_size"]
    nw = nc = n = 0
    for pp in d["per_problem"]:
        r = pp["rewards"]
        k = sum(1 for x in r if x > 0.5)
        g = len(r)
        n += 1
        if k == 0:
            nw += 1
        elif k == g:
            nc += 1
    p = d["overall_accuracy"]
    zvf = (nw + nc) / n
    zneg, zpos = nw / n, nc / n
    # odds-split at aggregate p, to quantify heterogeneity underestimate
    onw, onc = odds_split(zvf, p, G)
    gsm_exact.append((zneg, onw))  # (exact ZVF-, odds ZVF- at agg p)
    rows.append(dict(source="gsm8k", label=os.path.basename(f).split("_")[-1].replace(".json", ""),
                     G=G, acc=p, raw_zvf=zvf, zvf_neg=zneg, zvf_pos=zpos,
                     method="exact", outcome="converged"))

# heterogeneity underestimate: exact ZVF- vs odds ZVF- (agg p) for gsm8k anchor
exact_neg = st.mean([e for e, o in gsm_exact])
odds_neg = st.mean([o for e, o in gsm_exact])

# ---- 2. variance_mitigation: per-step odds-split, G=8 ----------------------
G_VM = 8
vm = defaultdict(list)
for r in csv.DictReader(open(f"{RES}/variance_mitigation.tsv"), delimiter="\t"):
    vm[(r["method"], r["seed"])].append((int(r["step"]), float(r["zvf"]), float(r["reward_mean"])))
vm_meth = defaultdict(list)
for (m, s), seq in vm.items():
    seq.sort()
    zn = zp = zt = 0.0
    for _, z, p in seq:
        a, b = odds_split(z, p, G_VM)
        zn += a; zp += b; zt += z
    n = len(seq)
    last10 = st.mean([p for _, _, p in seq[-10:]])
    peak = max(p for _, _, p in seq)
    vm_meth[m].append((zt / n, zn / n, zp / n, last10, peak))
for m, lst in vm_meth.items():
    rz = st.mean([x[0] for x in lst]); zn = st.mean([x[1] for x in lst])
    zp = st.mean([x[2] for x in lst]); l10 = st.mean([x[3] for x in lst])
    pk = st.mean([x[4] for x in lst])
    rows.append(dict(source="variance_mitigation", label=m, G=G_VM, acc=l10,
                     raw_zvf=rz, zvf_neg=zn, zvf_pos=zp, method="odds",
                     outcome=classify(pk, l10)))

# ---- 3. groupsize sweep: per-step odds-split, G=2..16 ----------------------
sw = json.load(open(f"{RES}/groupsize_zvf_sweep.json"))
byG = defaultdict(list)
for run in sw["runs"]:
    G = run["group_size"]
    zn = zp = zt = 0.0
    sl = run["step_log"]
    for s in sl:
        a, b = odds_split(s["zvf"], s["mean_reward"], G)
        zn += a; zp += b; zt += s["zvf"]
    n = len(sl)
    byG[G].append((zt / n, zn / n, zp / n, run["last10_avg"],
                   max(s["mean_reward"] for s in sl)))
for G, lst in sorted(byG.items()):
    rz = st.mean([x[0] for x in lst]); zn = st.mean([x[1] for x in lst])
    zp = st.mean([x[2] for x in lst]); l10 = st.mean([x[3] for x in lst])
    pk = st.mean([x[4] for x in lst])
    rows.append(dict(source="groupsize_sweep", label=f"G{G}", G=G, acc=l10,
                     raw_zvf=rz, zvf_neg=zn, zvf_pos=zp, method="odds",
                     outcome=classify(pk, l10)))

# ---- 4. drgrpo gsm8k cot: per-step odds-split (struggling, low acc) --------
try:
    dg = json.load(open(f"{RES}/drgrpo_gsm8k_cot_full.json"))
    byalgo = defaultdict(list)
    for run in dg["runs"]:
        G = 8
        sl = run["step_log"]
        zn = zp = zt = 0.0
        for s in sl:
            a, b = odds_split(s["zvf"], s["mean_reward"], G)
            zn += a; zp += b; zt += s["zvf"]
        n = len(sl)
        byalgo[run["algo"]].append((zt / n, zn / n, zp / n, run["last10_avg"],
                                    max(s["mean_reward"] for s in sl)))
    for a, lst in byalgo.items():
        rz = st.mean([x[0] for x in lst]); zn = st.mean([x[1] for x in lst])
        zp = st.mean([x[2] for x in lst]); l10 = st.mean([x[3] for x in lst])
        pk = st.mean([x[4] for x in lst])
        rows.append(dict(source="drgrpo_cot", label=a, G=8, acc=l10,
                         raw_zvf=rz, zvf_neg=zn, zvf_pos=zp, method="odds",
                         outcome=classify(pk, l10)))
except FileNotFoundError:
    pass

# ---- 5. tool_use collapse: zvf=1.0 at reward 0 -> pure ZVF- -----------------
seen = set()
for r in csv.DictReader(open(f"{RES}/tool_code_reward_diagnostics.tsv"), delimiter="\t"):
    key = r["model"]
    if key in seen:
        continue
    seen.add(key)
    z = float(r["zvf"]); p = float(r["reward_mean"])
    zn, zp = odds_split(z, p, 8)  # p=0 -> all mass to ZVF-
    rows.append(dict(source="tool_use", label=r["model"], G=8, acc=float(r["last10_avg"]),
                     raw_zvf=z, zvf_neg=zn, zvf_pos=zp, method="odds",
                     outcome="collapse"))

# ---- write summary ---------------------------------------------------------
cols = ["source", "label", "G", "acc", "raw_zvf", "zvf_neg", "zvf_pos", "method", "outcome"]
with open(f"{RES}/zvf_signed_summary.tsv", "w") as fh:
    fh.write(f"# Signed ZVF decomposition (iter58). ZVF- all-wrong starvation, "
             f"ZVF+ all-correct saturation. exact=per-group (gsm8k); "
             f"odds=i.i.d. sign-split of logged (zvf,reward,G).\n")
    fh.write(f"# GSM8K anchor: exact ZVF-={exact_neg:.4f} vs odds-split ZVF- at "
             f"aggregate acc={odds_neg:.4f} (odds-split lower-bounds ZVF- for "
             f"high-acc runs; difficulty heterogeneity concentrates all-wrong mass).\n")
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

# ---- separation test: healthy(converged) vs unhealthy(plateau/collapse/drift)
def auc(pos, neg):
    """Mann-Whitney AUC: P(score_unhealthy > score_healthy). 1.0 => perfect."""
    if not pos or not neg:
        return float("nan")
    w = 0
    for a in pos:
        for b in neg:
            w += 1.0 if a > b else 0.5 if a == b else 0.0
    return w / (len(pos) * len(neg))

healthy = [r for r in rows if r["outcome"] == "converged"]
unhealthy = [r for r in rows if r["outcome"] in ("plateau", "collapse", "drift")]
tests = []
for feat in ("raw_zvf", "zvf_neg", "zvf_pos"):
    hp = [r[feat] for r in unhealthy]
    hn = [r[feat] for r in healthy]
    a = auc(hp, hn)
    # perfect separation? min(unhealthy) > max(healthy) for ZVF-, opposite for ZVF+
    sep_hi = min(hp) > max(hn) if hp and hn else False   # unhealthy strictly higher
    sep_lo = max(hp) < min(hn) if hp and hn else False   # unhealthy strictly lower
    tests.append((feat, a, st.mean(hp), st.mean(hn), sep_hi, sep_lo))
with open(f"{RES}/zvf_signed_failure_corr.tsv", "w") as fh:
    fh.write("# ZVF feature separability: unhealthy {plateau,collapse,drift} vs "
             "healthy {converged}. AUC=P(unhealthy>healthy); 1.0 or 0.0 => perfect.\n")
    fh.write(f"# n_healthy={len(healthy)} n_unhealthy={len(unhealthy)}\n")
    fh.write("feature\tauc\tmean_unhealthy\tmean_healthy\tperfect_sep_high\tperfect_sep_low\n")
    for feat, a, mu, mh, shi, slo in tests:
        fh.write(f"{feat}\t{a:.4f}\t{mu:.4f}\t{mh:.4f}\t{shi}\t{slo}\n")

# ---- AERO test: which half does AERO cut vs GRPO? --------------------------
def get(src, lab):
    for r in rows:
        if r["source"] == src and r["label"] == lab:
            return r
    return None
g = get("variance_mitigation", "grpo"); a = get("variance_mitigation", "aero")
with open(f"{RES}/zvf_signed_aero.tsv", "w") as fh:
    fh.write("# Does AERO cut the pathological ZVF- (starvation) or the benign "
             "ZVF+ (saturation)? Signed decomposition of matched-stack runs.\n")
    fh.write("method\traw_zvf\tzvf_neg\tzvf_pos\tacc\n")
    for m in ("grpo", "aero", "cppo", "ngrpo", "scafgrpo", "mcgrpo", "gift", "areal", "es"):
        r = get("variance_mitigation", m)
        if r:
            fh.write(f"{m}\t{r['raw_zvf']:.4f}\t{r['zvf_neg']:.4f}\t{r['zvf_pos']:.4f}\t{r['acc']:.4f}\n")
    if g and a:
        d_raw = a["raw_zvf"] - g["raw_zvf"]
        d_neg = a["zvf_neg"] - g["zvf_neg"]
        d_pos = a["zvf_pos"] - g["zvf_pos"]
        frac_neg = d_neg / d_raw if d_raw != 0 else float("nan")
        fh.write(f"# AERO-GRPO: dRawZVF={d_raw:.4f} dZVF-={d_neg:.4f} dZVF+={d_pos:.4f} "
                 f"frac_of_reduction_from_ZVF-={frac_neg:.3f}\n")

print(f"rows={len(rows)} gsm8k exact_ZVF-={exact_neg:.4f} odds_ZVF-(aggp)={odds_neg:.4f}")
for feat, a2, mu, mh, shi, slo in tests:
    print(f"  {feat:8s} AUC={a2:.3f} unhealthy={mu:.3f} healthy={mh:.3f} sep_hi={shi} sep_lo={slo}")
if g and a:
    print(f"  AERO cuts raw {a['raw_zvf']-g['raw_zvf']:+.3f}; ZVF- {a['zvf_neg']-g['zvf_neg']:+.3f}; ZVF+ {a['zvf_pos']-g['zvf_pos']:+.3f}")
