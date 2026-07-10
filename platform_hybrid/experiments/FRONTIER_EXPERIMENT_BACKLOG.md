# Frontier Cross-Examination → New Experiment Backlog

Distilled from the 30 Gemini Deep Think + 22 ChatGPT Pro cross-examination rounds
(see `frontier_calls/digests/frontier_P{1..4}.md` and `paper/sections/frontier_synthesis_*.tex`).
Ranked by **feasibility** (Tier A = no new training) then impact. Each frontier prediction is
falsifiable and, per the sections, framed as "to be tested, not a result."

---

## TIER A — runnable NOW on existing data / checkpoints (no new training)

### A1. PCD & LARQ vs ZVF head-to-head  (Pillar 2)  ★ easiest, high value  — ✅ EXECUTED 2026-07-03
- **Script:** `scripts/pcd_vs_zvf.py` (stdlib only). **Outputs:** `experiments/results/pcd_vs_zvf_{shape,summary}.tsv`.
- **Results on real repo data (600 GSM8K prompt-groups, G=8; 80 anchor rows):**
  - *Aliasing confirmed:* 76 mastered (p=1) and 19 incapable (p=0) groups both score ZVF-ind=1.000,
    PCD=0 — ZVF cannot tell mastery from incapacity; LARQ's first term separates them 1.0 vs 0.0.
  - *Shape confirmed:* ZVF-ind is 1.0 only at p=0 and p=1 and 0 elsewhere; PCD is a clean parabola,
    0 at the ends → peak 0.25 at p=0.5. (Frontier's "sharp parabola vs flat plateau.")
  - *Micro-jitter falsification (sharpest):* eps~U(0,1e-4) makes batch **ZVF collapse 0.158→0.000**
    (falsely "all-healthy") while **PCD is invariant** (0.153802→0.153802, Δ=3e-7).
  - *Cross-run (suggestive, confounded):* Spearman(mean_reward, outcome)=+0.95 vs
    Spearman(mean_zvf, outcome)=+0.56 — the sign/level ZVF discards carries far more outcome signal.
    CAVEAT: contemporaneous mean_reward↔outcome is partly mechanical; a clean leading-indicator test
    needs early-window PCD/LARQ vs late outcome, and full PCD needs per-group tensors logged for all
    anchors (only GSM8K carries them). (Also: paper's 0.27 is on 23 *pooled* rows; 0.56 here is on 80
    per-seed rows — different aggregation unit, not a contradiction.)
- **Verdict:** structural claims (#1-#3) fully validated on real data; predictive claim (#4) directionally
  supported but needs per-anchor tensor logging + an early-window split to hit the rho≥0.45 bar cleanly.


- **Hypothesis:** ZVF's weak outcome correlation (ρ=0.27) is because it measures *existence*
  not *magnitude* of contrast. PCD = (G−1)/G · E[p_x(1−p_x)] and LARQ should predict better.
- **Method:** on the existing 80 anchors' rollout reward logs, compute PCD, LARQ_β, ZVF per
  run; Spearman-correlate each with final held-out outcome. Target: ρ(PCD)≳0.45 vs ρ(ZVF)=0.27.
- **Killer sub-test (micro-jitter falsification):** add ε∼U(0,1e−4) to rewards → ZVF flatlines
  to 0 (falsely "healthy") while PCD is invariant. One-line demonstration ZVF is fragile.
- **Infra:** pure re-analysis; extend `scripts/zvf_*` / reuse anchor reward TSVs. No GPU.

### A2. Contrastive-yield re-plot of the scaling null  (Pillar 1)  ★ attacks the headline null
- **Hypothesis:** the flat cross-scale slope (`tab:scaling-cross`) is a wrong-abscissa artifact;
  gain lives on the contrastive-yield axis C_eff = T·G·E[Y_G(p_x)]·KL, Y_G(p)=1−p^G−(1−p)^G.
- **Method:** recompute per-anchor cumulative C_eff from logged p_x, G, KL, steps; re-fit
  R_max (or ΔGRPO) vs C_eff instead of log10(N). Prediction: "failing" FLOP-fit anchors recover
  log-linear geometry; report G*=argmax_G Y_G(p)/G per anchor.
- **Infra:** re-analysis of existing traces (needs per-step p_x + KL, likely already logged);
  extend `scripts/fit_saturation_model.py` / `scaling_law_*`. No GPU.

### A3. Length-adversarial truncation test  (Pillar 4)  ★ frontier-flagged "highest-value follow-up"
- **Hypothesis:** if GRPO's held-out accuracy relies on length-padding ("stumbling into reward"),
  it craters non-linearly under harsh caps; a length-invariant Dr.GRPO degrades gracefully.
- **Method:** take the EXISTING converged GRPO & Dr.GRPO checkpoints; eval held-out at a sweep of
  generation caps T_max ≪ E[|y_GRPO|] (e.g. 64/128/256/512). Plot accuracy vs cap; compare curvature.
- **Infra:** inference only on existing checkpoints via `experiments/length_bias*` eval path.
  "One truncation sweep away from executed" — a few GPU-hours, no training.

### A4. CLMP length-mediation on existing rollouts  (Pillar 4)
- **Hypothesis:** McNemar-style marginal tests are confounded by length-mediated exploit.
- **Method:** treat length L as mediator between algo A∈{GRPO,Dr.GRPO} and success S; compute
  Pearl NDE/NIE (length-stratified, non-destructive — NOT iso-brevity truncation) and
  GER = NDE/TE on existing held-out data. Sanity check: current (length-compressed, ρ(L,R)<0)
  regime should give small NIE for both → consistent with observed equivalence. Validates the
  estimator before A-tier→C-tier regime work.
- **Infra:** re-analysis of per-item {length, success, algo}; new `scripts/clmp_mediation.py`. No GPU.

### A5. BEI on the matched-stack PPO/GRPO run  (Pillar 3 / Pillar 1)
- **Hypothesis:** the p=0.75 PPO≈GRPO null holds exactly when the induced *updates* agree.
  BEI = cos(g_PPO,g_GRPO)·min(‖g‖ ratio) ≥ 0.97 ⇒ held-out agreement within 1pp.
- **Method:** on the same-stack batches, log both update vectors and compute BEI; correlate with
  the observed null. Also compute BEI_G across the G-sweep — prediction: climbs sublinearly and
  **saturates before** G=16 (so G=16 is *not* yet PPO-equivalent given the measured 52%-of-√G SNR).
- **Infra:** needs gradient-vector logging on a handful of replayed batches (borderline A/B); no
  new training, small GPU. Extend `experiments/base_instruct_paired.py` grad hooks.

---

## TIER B — modest new runs (existing infra, new configs)

### B1. The 2×2 super-group decisive test: variance vs ZVF  (Pillar 3)  ★ resolves *why* G matters
- **Design (tokens/steps/KL fixed):** Arm A natural small G · Arm B variance-only (same unlocked
  prompts, all K(G−K) pairs) · Arm C ZVF-only (more unlocked prompts, single random win-loss pair
  or covariance-matched noise) · Arm D natural large G. Target swing Δ_{32-2}≈0.020.
- **Preregistered rule:** ZVF-dominant if Arm C recovers ≥75% of the gain and Arm B ≤25%;
  variance-dominant if reversed. (Frontier prior: ZVF-dominant — the G4→G32 gain concentrates in
  high-p bins, +0.253 at p≥0.75 vs +0.010 at p<0.5.)
- **Complement (single-batch, no training):** residual-isomorphism — subtract synthetic
  margin-scaled DPO gradient from empirical GRPO gradient; cos(residual, V_KL)≈1 confirms the
  contrast-projection theorem exactly; <0.99 proves the group baseline carries orthogonal structure.
- **Infra:** 4 GRPO arms via `group_size_*` infra; the residual check is Tier-A.

### B2. Difficulty-stratified G-sweep re-analysis  (Pillar 3)
- Re-bin the existing G-sweep by per-prompt p_x; test whether the G-retention gain is
  concentrated in high-p bins (ZVF-escape signature) vs uniform (√G variance signature).
  Likely partly doable on logged data → could promote to Tier A if p_x is per-prompt logged.

### B3. Iso-Yield dynamic grouping  (Pillar 2/3)
- Route mastered/frontier prompts to small G and spend freed budget on stubborn tails
  (marginal yield peaks near p_x≈1/G, not ½). Test C_eff-optimal adaptive grouping vs static G
  at matched compute. Modified sampler over `group_size_token_normalized.py`.

---

## TIER C — new training regime (biggest lift, biggest novelty)

### C1. Length-confounded sparse-RLVR regime  (Pillar 4)  ★ the regime that reveals Dr.GRPO
- **Conditions:** CV(L)≥0.5, |corr(L,R)|≥0.25, E[L|A<0]−E[L|A>0]≥50–100 tok; hard math/symbolic/
  code-debug/proof tasks, binary exact-match reward, no length penalty, caps 512–1024, temp 0.8–1.0,
  G∈{8,16}, p_x∈[0.1,0.4]. **Train length-spurious / test length-anti-spurious split.**
- **Decision rule:** Dr.GRPO advantage only if Δ_held-out ≥ +2–5pp AND length drift ≥30% smaller,
  at matched rollout budget and matched train reward.
- Pairs naturally with A3 (truncation) + A4 (CLMP, predicts GER<0.15 GRPO vs >0.85 Dr.GRPO here).

### C2. Preregistered curve-collapse scaling law  (Pillar 1)  — earns the word "law"
- Sweep N×T; fit H(N,T)=H_∞(1−exp[−(N^γ T/τ0)^β]); freeze on 60–70% of cells; law is earned only
  if it predicts blind large-N/large-T cells within ≤1.5–2.0pp (≤10% rel. err in H) with seed-stable
  exponents (CV(β),CV(γ)<0.2). Expensive (many cross-scale runs); the honest current claim is taxonomic.

### C3. Memorization-vs-generalization ladder  (Pillar 4)
- Four disjoint held-out strata S0 (contamination sentinel) / S1 (new vars) / S2 (held-out template) /
  S3 (external); per-item exposure score M (BM25 + n-gram + embedding + template match); bin by
  M-decile. Flagship estimand = adjusted low-similarity S2/S3 effect (logistic w/ difficulty+seed
  random effects + length covariate). Hard negative controls: length-matched replay; answer-preserving
  perturbation consistency C=Pr(f(x)=f(T(x))). Mostly eval engineering + existing checkpoints.

---

## Recommended first move
Run **A1 + A2 + A3** — all three attack the paper's three headline hedges (ZVF ρ=0.27, the flat
scaling null, the Dr.GRPO equivalence) using only data/checkpoints already in hand, and A3 is the
frontier models' own single-highest-value pick. If A1/A2 land, B1 becomes the flagship mechanistic
result; C1 is the natural training-regime follow-up that A3+A4 are designed to feed.
