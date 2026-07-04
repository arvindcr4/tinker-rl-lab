# 21 — Predictable-Noise Power Audit: MDE + TOST equivalence for the same-stack GRPO≈PPO claim

**Source lecture.** Berkeley F25 L8 — Sida Wang (Meta), *"Predictable Noise in
LLMs / Adding Error Bars to Evals."*
**Citations (verified 2026-07-04 via arxiv.org/abs):**
- Evan Miller, *Adding Error Bars to Evals: A Statistical Approach to Language
  Model Evaluations*, **arXiv:2411.00640** (2024, stat.AP). ✓
- Sida Wang, *Measuring all the noises of LLM Evals*, **arXiv:2512.21326**
  (submitted 2025-12-24, rev. 2026-03-29). ✓ — introduces the **all-pairs paired
  method to improve statistical power** in comparative LLM evals (directly the
  power/pairing angle used in H5).

**Target.** A1 (statistical rigor) — Pillar 1 (same-stack PPO vs GRPO), with a
cross-pillar echo to CDH (row 12).

---

## The gap this fills (why it is NOT a repeat of rows 03/07/20)

Rows 03/07/20 operationalised the *descriptive* half of Sida Wang's lecture —
**how wide are the honest error bars** (i.i.d. bootstrap → paired bootstrap →
seed-clustering DEFF). None touched the *forecasting* half that Miller's paper is
actually built around and that Wang 2512.21326 sharpens: **given the predictable
seed-level noise, what effect can this study resolve, and can a "no significant
difference" be upgraded to a *positive equivalence* claim?**

This matters because the flagship Pillar-1 result is a **null** (heldout paired
Δ = −0.002, p = 0.37; last10 p = 0.75). A null is a reviewer liability: *"your
GRPO≈PPO claim is just an underpowered study with n=5 seeds."* The only rigorous
rebuttal is (a) a **minimum-detectable-effect (MDE)** showing the study *could*
have caught a meaningful gap, and (b) a **TOST equivalence test** that turns the
null into a bounded positive statement. That is exactly what this audit builds.

## Method (all on real `experiments/results/samestack_ppo_grpo.json`, 5 seeds × 2 algos)

- **Retrospective power / MDE** via the noncentral-t (paired), df = n−1, α = .05,
  with a normal-approximation fallback for the extreme-noncentrality regime where
  `scipy.stats.nct` underflows.
- **TOST equivalence bound** = the tightest margin δ at which both one-sided tests
  reject = the larger endpoint of the 90% CI (`|Δ| ± t_{.95,df}·SE`). If that
  bound < a margin, we may *positively assert* |GRPO−PPO| < margin.
- **Pooling power-inflation**: honest seed-clustered n=5 vs the row-20 illusion of
  pooling S×M=50 last-10 step rewards as i.i.d.
- **Pairing gain** (Wang all-pairs): paired SE vs the two-sample unpaired SE.

## Results — 5/5 DECISIVE

| # | Hypothesis | Result | Verdict |
|---|---|---|---|
| H1 | Heldout study is well-powered (MDE₈₀ < 1pt and < 5pt lit gap) | **MDE₈₀ = 0.0075** (0.75pp); power to catch a 5pt cross-stack gap = **1.00** | **DECISIVE** |
| H2 | Null → positive equivalence (bound < 1pt) | **equivalence bound = 0.0063** → *"\|GRPO−PPO heldout\| < 0.63pp"* at 95% | **DECISIVE** |
| H3 | Pooling fabricates power | honest MDE₈₀ / pooled-illusion MDE₈₀ = **3.83×** (ICC=0.72, DEFF=7.45, √DEFF=2.73) | **DECISIVE** |
| H4 | Metric choice is an error-bar decision | last10 equivalence bound **0.163** = **26×** the heldout bound; drops **3.8×** (0.163→0.043) when the single PPO seed-456 collapse is removed | **DECISIVE** |
| H5 | Pairing is what powers the claim (Wang all-pairs) | heldout paired SE **2.32×** smaller than unpaired → paired MDE 0.0075 vs unpaired 0.0174 | **DECISIVE** |

### The paper-critical sentence
The heldout equivalence claim is **not** an underpowered null: the design resolves
a 0.75pp gap at 80% power and has power ≈ 1.0 against the 5pp gaps reported
cross-stack, so TOST licenses the positive statement **|GRPO−PPO| < 0.63pp (95%)**.
The **last10** metric, by contrast, can only assert **|Δ| < 16.3pp** — a near-
vacuous bound driven **entirely** by a single stability event (PPO seed-456 last-10
avg collapses to 0.72 while its heldout is 0.995). So the *choice of headline
metric is itself an error-bar decision*: **heldout is the honest equivalence
headline; last10 must carry the caveat that its width is one seed's late-training
instability, not an expected-return difference.**

### Cross-pillar bridge (CDH, row 12)
The step-level ICC on PPO last-10 rewards is **0.72 (DEFF 7.45)** — an independent
re-measurement of the same PPO-instability signature that row 12 saw as 156×
grad-norm and row 20 saw as ICC(PPO)/ICC(GRPO)=5.5×. Here it manifests as **the
seed-456 collapse that single-handedly destroys last10's statistical power** — the
Critic-Degeneracy mechanism viewed through the *power* lens.

## Go / No-Go
**GO — validated.** Converts the Pillar-1 null into a defensible bounded
equivalence and pre-empts the "underpowered null" objection. Paper-facing edit
(Pillar-1 eval section): report the heldout MDE₈₀ (0.75pp), the TOST bound
(|Δ|<0.63pp), and headline heldout over last10 with the seed-456 caveat. Cite
Miller 2411.00640 + Wang 2512.21326 and the 2.3× pairing gain as the justification
for the paired design.

## Artifacts
- `scripts/berkeley/predictable_noise_power_audit.py`
- `experiments/results/berkeley/pnpa_power_mde.tsv`, `pnpa_tost_equivalence.tsv`,
  `pnpa_pooling_power_inflation.tsv`, `pnpa_outlier_robustness.tsv`,
  `pnpa_paired_vs_unpaired.tsv`, `pnpa_sample_size_forecast.tsv`, `pnpa_summary.json`
