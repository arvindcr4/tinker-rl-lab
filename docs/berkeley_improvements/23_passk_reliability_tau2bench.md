# 23 — pass^k Reliability Audit (F25 L10 Clay Bavor / τ²-Bench)

**Target:** A1 (statistical rigor of headline numbers) + A2 (eval-protocol
hardening) · **Status:** validated · **Pillar tag:** B-F25
**Evidence:** `scripts/berkeley/passk_reliability_audit.py`,
`experiments/results/berkeley/passk_*.tsv`,
`experiments/results/berkeley/passk_reliability_summary.json`

## Course idea → verified citations
Clay Bavor's F25 lecture (Sierra) is built on the **τ-bench / τ²-bench** thesis:
a deployed agent is judged by **reliability**, not average accuracy. The metric is
**pass^k** = P(a task succeeds on *all k* i.i.d. trials), averaged over tasks.

- **τ-bench** — "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World
  Domains", Yao, Shinn, Razavi, Narasimhan, 2024, **arXiv:2406.12045**. *Introduced
  pass^k.* (verified via search 2026-07-04)
- **τ²-Bench** — "τ²-Bench: Evaluating Conversational Agents in a Dual-Control
  Environment", Barres, Narasimhan et al. (Sierra), 2025, **arXiv:2506.07982**.
  *Re-centers evaluation on reliability for dual-control agents.* (verified;
  OpenReview LGmO9VvuP5)

## Gap this closes
Every headline in the 4 papers is **pass^1** (mean accuracy). No number reports
reliability under repeated trials — exactly the quantity τ²-Bench argues matters
for deployment. Prior rows 10 (`sweagent_passk_aci`) used **pass@k** (best-of-k
*union*, which *rises* with k); pass^k is its mirror image (*falls* with k) and had
never been computed here.

## Method (real data)
Per-task success probabilities `p_x` from **`zvf_iter46_per_prompt_isog.tsv`**
(Qwen3-8B on tinker_gsm8k, **505 distinct (seed,problem) tasks**). Identities:

```
pass^k          = E_D[p^k]            reliability   (DECREASES in k)
pass@k (best-of)= 1 − E_D[(1−p)^k]    any-of-k      (INCREASES in k)
homogeneous     = μ^k                 naive, ignores per-task dispersion
```
By Jensen (p^k convex) **E[p^k] ≥ μ^k**: task dispersion *inflates* true pass^k
above the naive prediction. The excess is a functional of the same per-task
variance σ²_p that drives the Pillar-2 ZVF / group-size collapse.

## Results — 5/5 hypotheses DECISIVE
Real distribution: n=505, μ=**0.674**, σ_p=**0.192**, Var=0.037.

- **H1 — reliability collapse.** Mean accuracy 0.674 falls to **pass^5 = 0.235**:
  reporting only pass^1 **overstates 5-trial reliability by 43.9 pp**. The naive
  μ^5=0.139 *under*-states the true pass^5 by +0.096 (Jensen). Both errors are
  large and same-signed structural — one direction can't be fixed by the other.
  → `passk_reliability_curve.tsv`
- **H2 — variance is a first-order term.** pass^k = μ^k + C(k,2)μ^{k−2}σ² +
  C(k,3)μ^{k−3}m₃ + …. At k=3 the 3rd-order form is **exact** (err 1e-16, p³ is
  degree-3) — a closed validation of the expansion. At k=6 the naive μ^k errs
  0.098; adding **only** the σ² term removes **83%** of that error. Per-task
  variance is not a nuisance — it is the dominant correction. → `passk_moment_expansion.tsv`
- **H3 — the scissor.** At k=5, best-of pass@k=0.967 while reliability pass^5=0.235
  (**gap 0.73**). Dispersion is double-edged with **opposite signs**: it *raises*
  pass^k (+0.096; always-pass tasks dominate the all-k set) but *lowers* pass@k
  (−0.029; always-fail tasks are never rescued). A paper that shows only the rising
  pass@k curve **hides a brittle agent** the falling pass^k exposes. → `passk_scissor.tsv`
- **H4 — mean accuracy cannot rank reliability.** Two task sets with **identical**
  mean 0.674 but variance 0.037 vs 0.114 give **pass^5 = 0.235 vs 0.520**. Mean
  accuracy is provably insufficient to rank deployment reliability; you must report
  pass^k. → `passk_equal_mean_diff_var.tsv`
- **H5 — full-benchmark reconstruction (ZVF bridge).** Adding reliable (p≈1) mass
  w₁=0.60 to match the reported G=8 mean reward 0.869, the full-benchmark
  **pass^5 = 0.694** vs naive μ^5=0.497 — a Jensen reliability excess of **+0.197**
  even at high accuracy. The correction is governed by σ²_p, the **same dispersion
  that drives Pillar-2 ZVF collapse** — one statistic, two consequences.
  → `passk_full_reconstruction.tsv`

## Go / no-go
**GO — paper-facing (A1/A2).** Add a **pass^k reliability column** alongside every
pass^1 headline in the eval-protocol section, and a one-line note that the reliability
gap is a σ²_p functional (cross-referencing the ZVF analysis). Cheap: pass^k is a
pure post-hoc functional of already-collected per-prompt rewards — no new training.
Caveat: `p_x` here is quantized at 1/8 (G=8) and truncated to (0.05,0.95); H5's
reconstruction is a stated-assumption sensitivity, not a measurement. Recommend
recomputing pass^k directly from raw per-prompt reward traces before the camera-ready.

## Cross-pillar link
Bridges **A1 error-bars** (rows 03/20/21), **Pillar-2 ZVF dispersion** (rows
09/15/17/19/22), and **pass@k** (row 10) into a single reliability statement:
mean accuracy, ZVF collapse, and pass^k are three readouts of one per-task
variance σ²_p. (frontier synthesis: Pillar-2 "ZVF is observed signal availability,
not difficulty" — pass^k is the deployment-facing readout of that same availability.)
