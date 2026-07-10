#!/usr/bin/env python3
"""A1 (frontier backlog): PCD/LARQ vs ZVF head-to-head.

Tests the frontier cross-examination claim (Pillar 2) that ZVF's weak outcome
correlation (Spearman rho=0.2687 on mean_zvf vs last10_avg; zvf_failure_correlation.tsv)
is because ZVF measures only the *existence* of contrast, not its *magnitude* or *sign*.
Proposed replacements:
  PCD  = mean within-group reward variance  = E_x[p_x(1-p_x)]   (magnitude of contrast)
  LARQ = E_x[ p_hat + beta*4*p_hat(1-p_hat) ]                   (sign-resolving quality)

This script runs the parts that need NO new training, on data already in the repo:
  1. Mastery-incapacity aliasing         (raw GSM8K per-group reward tensors)
  2. ZVF U-shape vs PCD parabola in p_x  (same tensors)
  3. Micro-jitter falsification          (same tensors; the frontier's sharpest sub-test)
  4. Cross-run predictive check          (zvf_summary.tsv): reproduce rho(ZVF,outcome)
     and compare against LARQ's sign-resolving first term (mean_reward), the piece of
     LARQ that is reconstructable from the existing per-run summary.

Full PCD-vs-ZVF predictive horse race (frontier target rho>=0.45) needs per-group reward
tensors logged for ALL anchors; only the GSM8K runs carry them, so #4 tests the
sign-resolution claim with the available proxy and flags the data gap. Stdlib only.
"""
import json, glob, math, os, random

RES = "platform_hybrid/experiments/results"
random.seed(0)

# ---------- helpers (no numpy/scipy dependency) ----------
def pvar(xs):
    """population variance"""
    n = len(xs); m = sum(xs)/n
    return sum((x-m)**2 for x in xs)/n

def rankdata(xs):
    """average ranks, ties shared"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0]*len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j+1 < len(xs) and xs[order[j+1]] == xs[order[i]]:
            j += 1
        avg = (i+j)/2.0 + 1.0
        for k in range(i, j+1):
            ranks[order[k]] = avg
        i = j+1
    return ranks

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sx = sum((x-mx)**2 for x in xs); sy = sum((y-my)**2 for y in ys)
    if sx == 0 or sy == 0: return float('nan')
    cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    return cov/math.sqrt(sx*sy)

def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))

# ---------- load raw per-group reward tensors (GSM8K, 3 seeds, G=8) ----------
groups = []   # each: list of 0/1 rewards for one prompt-group
for f in sorted(glob.glob(f"{RES}/tinker_gsm8k_zvf_s*.json")):
    d = json.load(open(f))
    if "per_problem" not in d: continue
    for pp in d["per_problem"]:
        groups.append([float(r) for r in pp["rewards"]])
G = len(groups[0])
print(f"# Loaded {len(groups)} prompt-groups (G={G}) from GSM8K reward tensors\n")

def zvf_ind(g):  return 1.0 if pvar(g) == 0.0 else 0.0
def pcd(g):      return pvar(g)                       # = p_hat(1-p_hat)
def phat(g):     return sum(g)/len(g)

# ---------- 1. mastery-incapacity aliasing ----------
mastered  = [g for g in groups if phat(g) == 1.0]
incapable = [g for g in groups if phat(g) == 0.0]
frontier  = [g for g in groups if 0.0 < phat(g) < 1.0]
print("## 1. Mastery-incapacity aliasing (ZVF cannot tell them apart)")
print(f"   mastered (p=1):   n={len(mastered):4d}  mean ZVF-ind={sum(map(zvf_ind,mastered))/max(1,len(mastered)):.3f}  "
      f"mean PCD={sum(map(pcd,mastered))/max(1,len(mastered)):.4f}  LARQ term1 (p_hat)=1.000")
print(f"   incapable (p=0):  n={len(incapable):4d}  mean ZVF-ind={sum(map(zvf_ind,incapable))/max(1,len(incapable)):.3f}  "
      f"mean PCD={sum(map(pcd,incapable))/max(1,len(incapable)):.4f}  LARQ term1 (p_hat)=0.000")
print(f"   frontier (0<p<1): n={len(frontier):4d}  mean ZVF-ind={sum(map(zvf_ind,frontier))/max(1,len(frontier)):.3f}  "
      f"mean PCD={sum(map(pcd,frontier))/max(1,len(frontier)):.4f}")
print("   -> ZVF-ind=1 for BOTH mastered and incapable (aliased); LARQ term1 separates them 1.0 vs 0.0.\n")

# ---------- 2. ZVF U-shape vs PCD parabola across p_x ----------
print("## 2. Shape across success rate p_x = k/G  (ZVF U-shaped; PCD single-peaked at 0.5)")
print("   k/G   n     mean_ZVF_ind   mean_PCD")
rows2 = []
for k in range(G+1):
    bucket = [g for g in groups if abs(phat(g)-k/G) < 1e-9]
    if not bucket: continue
    z = sum(map(zvf_ind,bucket))/len(bucket); p = sum(map(pcd,bucket))/len(bucket)
    rows2.append((k/G, len(bucket), z, p))
    print(f"   {k}/{G}  {len(bucket):4d}   {z:.3f}          {p:.4f}")
print()

# ---------- 3. micro-jitter falsification ----------
EPS = 1e-4
def jitter(g): return [r + random.uniform(0, EPS) for r in g]
zvf_before = sum(map(zvf_ind, groups))/len(groups)
pcd_before = sum(map(pcd, groups))/len(groups)
jg = [jitter(g) for g in groups]
zvf_after = sum(map(zvf_ind, jg))/len(jg)
pcd_after = sum(map(pcd, jg))/len(jg)
print("## 3. Micro-jitter falsification (add eps~U(0,1e-4), e.g. a length penalty)")
print(f"   batch ZVF:  {zvf_before:.4f}  ->  {zvf_after:.4f}   (collapses: falsely reports a 'healthy' all-mixed batch)")
print(f"   batch PCD:  {pcd_before:.6f}  ->  {pcd_after:.6f}   (invariant: delta={abs(pcd_after-pcd_before):.2e})")
print(f"   -> a real diagnostic must be jitter-invariant; ZVF is not, PCD is.\n")

# ---------- 4. cross-run predictive check on zvf_summary.tsv ----------
print("## 4. Cross-run predictive check (zvf_summary.tsv): reproduce rho(ZVF,outcome),")
print("##    compare LARQ's sign-resolving first term (mean_reward). Full PCD needs per-group")
print("##    tensors for every anchor (only GSM8K has them) -> flagged as a logging gap.")
rows = []
hdr = None
for line in open(f"{RES}/zvf_summary.tsv"):
    if line.startswith("#"): continue
    parts = line.rstrip("\n").split("\t")
    if hdr is None: hdr = parts; continue
    rows.append(dict(zip(hdr, parts)))
def col(r, k):
    try: return float(r[k])
    except: return float('nan')
# use rows with a valid outcome (last10_avg) and diagnostics
use = [r for r in rows if r.get("last10_avg","") not in ("","nan") and r.get("mean_zvf","") not in ("","nan")]
zvf  = [col(r,"mean_zvf")    for r in use]
mr   = [col(r,"mean_reward") for r in use]
out  = [col(r,"last10_avg")  for r in use]
# collapse label (binary)
iscol = [1.0 if r.get("failure_label","")=="collapse" else 0.0 for r in use]
print(f"   n_runs={len(use)}")
print(f"   Spearman(mean_zvf, last10_avg)      = {spearman(zvf,out):+.4f}   [baseline ~0.27]")
print(f"   Spearman(mean_reward, last10_avg)   = {spearman(mr,out):+.4f}   [LARQ term1: resolves mastery sign]")
print(f"   |improvement from sign resolution|  = {abs(spearman(mr,out))-abs(spearman(zvf,out)):+.4f}")
print(f"   Spearman(mean_zvf, is_collapse)     = {spearman(zvf,iscol):+.4f}")
print(f"   Spearman(mean_reward, is_collapse)  = {spearman(mr,iscol):+.4f}")

# ---------- write machine-readable outputs ----------
os.makedirs(RES, exist_ok=True)
with open(f"{RES}/pcd_vs_zvf_shape.tsv","w") as f:
    f.write("p_x\tn\tmean_zvf_ind\tmean_pcd\n")
    for a,b,c,d in rows2: f.write(f"{a:.4f}\t{b}\t{c:.4f}\t{d:.4f}\n")
with open(f"{RES}/pcd_vs_zvf_summary.tsv","w") as f:
    f.write("metric\tvalue\n")
    f.write(f"n_groups\t{len(groups)}\n")
    f.write(f"zvf_batch_before_jitter\t{zvf_before:.4f}\n")
    f.write(f"zvf_batch_after_jitter\t{zvf_after:.4f}\n")
    f.write(f"pcd_batch_before_jitter\t{pcd_before:.6f}\n")
    f.write(f"pcd_batch_after_jitter\t{pcd_after:.6f}\n")
    f.write(f"n_runs_crossrun\t{len(use)}\n")
    f.write(f"spearman_zvf_outcome\t{spearman(zvf,out):.4f}\n")
    f.write(f"spearman_meanreward_outcome\t{spearman(mr,out):.4f}\n")
    f.write(f"spearman_zvf_collapse\t{spearman(zvf,iscol):.4f}\n")
    f.write(f"spearman_meanreward_collapse\t{spearman(mr,iscol):.4f}\n")
print(f"\n# wrote {RES}/pcd_vs_zvf_shape.tsv and {RES}/pcd_vs_zvf_summary.tsv")
