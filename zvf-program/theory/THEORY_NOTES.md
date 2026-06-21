# ZVF Program — Pillar 2 (Theory): Gap Ledger & Verification Notes

**Status:** DRAFT / SKELETON. `zvf_theory.tex` compiles (8 pages, pdflatex x2,
exit 0). Every theorem is a **proof sketch**, not a proof. Nothing here should
be cited as "proven" until the gaps below are discharged **by the author**.

This file is the authoritative ledger. It mirrors the inline
`% TODO(proof-gap:)` comments and `\GAP{...}` markers in the `.tex`.

---

## 0. Honesty summary (read first)

| Result | Claim | Status | One-line risk |
|---|---|---|---|
| **T1** | ZVF is an asymptotically normal estimator with closed-form 95% CI | **Sketch.** The *CI itself is rigorous* (it's the textbook binomial-proportion CLT). The *"U-statistic" framing is not yet justified* and is arguably trivial as written. | The genuine U-statistic content (and the only place "asymptotic normality of a *U-statistic*" is non-trivial) needs a **degree-2** statistic whose kernel can be **degenerate** at the boundary, changing the √m rate. |
| **T2** | High ZVF ⇒ provably need ≥ N(ZVF) rollouts before improvement | **Sketch.** The geometric tail bound is correct, **but** it bounds rollouts-to-*nonzero-gradient*, not rollouts-to-*improvement*, and uses population ZVF where only an estimate exists. | "Improvement" ≠ "one informative group." Need either a weaker claim or a separate improvement lemma. |
| **T3** | Closed-form optimal group size G* maximizing signal-per-rollout | **Sketch.** FOC is correctly derived *given the objective*, but **the objective S(p,G) is an unjustified modelling choice** and uniqueness of G* is unproven. | Different reasonable definitions of "learning signal" give different G*. |

**The single biggest threat to all three:** Assumption 3 (across-group i.i.d.).
Curriculum sampling, replay buffers, and within-epoch correlation all break it,
and all three proofs use it.

**No empirical numbers are fabricated.** The only constants in the paper are
algebraic (1.96, δ, ε) and the illustrative example inherited verbatim from the
Pillar-1 appendix (p=0.5, K=8 ⇒ ZVF ≈ 0.008).

---

## 1. Lemmas / steps that still need a RIGOROUS proof

Ordered by how load-bearing they are.

### T1 (Estimator)
1. **[CRITICAL] Choose the canonical object.** As written, ZVF = mean of i.i.d.
   Bernoulli(θ_K) indicators ⇒ Theorem 1 is the **binomial-proportion CLT**, and
   "U-statistic" adds nothing (an order-1 U-statistic *is* a sample mean). Decide:
   - (a) Honestly call it a sample-mean estimator and drop "U-statistic," **or**
   - (b) Make it a genuine **degree-2 U-statistic** by estimating a pairwise
     functional (e.g. the across-group variance of the group-variance, or a
     covariance between ZVF and another diagnostic). Only then is the
     U-statistic CLT (Hoeffding) non-trivial.
2. **[CRITICAL, if (b)] Kernel non-degeneracy / Hájek projection.** Prove the
   first-order projection of the degree-2 kernel is non-zero on the interior
   θ_K ∈ (0,1); **characterize the degeneracy at θ_K ∈ {0,1}** where the rate
   changes from √m to m (different limiting law — weighted χ²). This is the real
   math content and is currently only flagged.
3. **Two-stage variance check (Lemma 1 / Thm 1).** Z_g is generated in two
   stages: draw p_g ~ φ, then K Bernoulli(p_g) trials. Confirm
   Var(Z_g) = θ_K(1−θ_K) **exactly** (true because Z_g is itself binary), and —
   crucially — confirm the **reporting pipeline averages Z_g (the indicator),
   not h_K (the conditional probability)**. If a pipeline ever plugs in an
   estimated p_g and averages h_K(p̂_g), the variance and the CI change. Verify
   against the actual `scripts/zvf_compute_cross_framework.py` rule
   `(var(axis=-1, ddof=1) ≤ ε).mean()` — that averages the indicator, so the
   Wald CI is the right one, but **state this explicitly**.
4. **Boundary CI.** Justify the Wilson-score recommendation near ZVF ∈ {0,1};
   give the exact Wilson formula (currently only named).
5. **Finite-m unbiasedness vs. the ε tolerance.** Unbiasedness uses
   E[Z_g] = h_K(p_g). With ε > 0 and binary rewards this is exact, but **note
   ε only matters for non-binary rewards** (out of scope here) — state that the
   binary assumption makes ε immaterial to T1.

### T2 (Lower bound)
6. **[CRITICAL] "Improvement" vs. "nonzero gradient."** The proof delivers
   rollouts-until-first-informative-group = rollouts-until-nonzero-gradient.
   A nonzero gradient is **necessary but not sufficient** for a policy
   *improvement* (monotone increase in expected reward). Options:
   - Weaken the theorem statement to "nonzero update" (clean, defensible), **or**
   - Add an **improvement lemma**: under bounded reward variance and a step-size
     condition, one informative group yields expected improvement ≥ c·(signal)
     (e.g. via an NPG / policy-gradient ascent guarantee). This is a real,
     separate theorem.
7. **[CRITICAL] Estimation-aware bound.** Eq. (T2) uses the *population* ZVF.
   Replace with the **one-sided lower confidence bound** from T1 to keep the
   "you provably need ≥ N" guarantee valid; restate at the matching confidence
   level (Corollary 2). Currently uses the two-sided 95% half-width — fix to a
   one-sided 95% (z = 1.645, not 1.96) and propagate.
8. **Mixture vs. fixed-p geometric (Lemma 2).** The "Geometric(q̄)" model uses
   the mixture informativeness rate q̄ = E_φ[q(p)]. The per-batch rate is random;
   justify replacing E[1/q̂] by 1/q̄. **Jensen gives E[1/q̂] ≥ 1/q̄**, so the
   expected-cost equation is an *under*-estimate — fine for a *lower* bound but
   the direction must be stated, and Eq. (exp-rollouts) reframed as a bound.

### T3 (Optimality)
9. **[CRITICAL] Justify the objective S(p,G).** Currently
   S(p,G) ∝ (1−h_G(p))·p(1−p) by analogy. Derive it from the **GRPO gradient
   magnitude** or the **Fisher information** of a group. Candidate objectives —
   (i) P(informative)·effect-size, (ii) E[Â²] (expected squared group-relative
   advantage), (iii) per-group Fisher information — **do not all yield the same
   G***. The entire theorem's correctness hinges on this.
10. **Differentiation under the integral (DCT) at the boundary.** ∂h_G/∂G
    involves p^G ln p, which → −∞·0 as p → 0/1. Prove integrability of p^G ln p
    against φ; the interior-support assumption (Ass. 4) is necessary — verify it
    suffices.
11. **Existence/uniqueness of G\*.** The FOC may have multiple roots for skewed
    φ (J(G) possibly multimodal). Either prove **quasi-concavity of J in G**
    (⇒ unique interior max) or fall back to the discrete argmax over
    {2,3,…} (always exists; safe). State which.
12. **Discrete↔continuous rounding.** Add the standard argument
    G\* ∈ {⌊G_c⌋, ⌈G_c⌉} where G_c is the continuous optimizer.
13. **Beta-prior closed form (Corollary 3).** Write out the three required
    moments E_φ[p^{G+1}(1−p)], E_φ[p(1−p)], E_φ[(1−p)^{G+1}p] explicitly as
    Beta-function ratios B(α+·,β+·)/B(α,β); verify the bookkeeping (currently
    asserted, not shown).
14. **Monotonicity claim.** "G\* decreasing in extremity max(p,1−p)" is asserted
    from the h_G(p) ~ max(p,1−p)^G heuristic; prove it (e.g. comparative statics
    / implicit function theorem on the FOC).

### Controller (Section 6)
15. **Closed-loop stability.** Setting G_{t+1} ← G\*(φ̂_t) where φ̂_t is fit from
    data sampled at G_t creates a feedback loop. Convergence/oscillation of this
    recursion is **unanalyzed**. At minimum flag (done); ideally give a
    contraction condition on the map φ̂ ↦ G\* ↦ data ↦ φ̂.
16. **Difficulty-prior estimation.** The controller fits φ̂_t (Beta MoM/MLE)
    from recent r̄_g. The CI in T1 assumes φ fixed; under online φ̂ the coverage
    guarantee is only approximate. Quantify.

---

## 2. Assumptions each result depends on

| Assumption | Statement | Used by | Fragility |
|---|---|---|---|
| **A1 Binary verifiable rewards** | r ∈ {0,1} from a verifier | T1, T2, T3 | Low — this is the declared scope. Breaks for dense/PRM rewards (already out of scope in Pillar 1). |
| **A2 Within-group i.i.d.** | K rollouts i.i.d. Bernoulli(p_g) given x_g, π | T1, T2, T3 | Medium — sampling temperature/dedup/nucleus truncation can correlate rollouts; minibatch decoding usually OK. |
| **A3 Across-group i.i.d. prompts** | p_1,…,p_m i.i.d. from a fixed population | **T1, T2, T3 (all)** | **HIGH — the load-bearing risk.** Violated by curriculum, replay, importance sampling, within-epoch ordering. CLT and geometric tail both assume it. |
| **A4 Non-degenerate support** | φ puts mass on (0,1), θ_K ∈ (0,1) | T1 (boundary), T3 (DCT) | Medium — fails in SFT-saturated regime (θ_K → 1), exactly where Pillar 1 already says ZVF loses power. |

Additional implicit assumptions to surface in the final version:
- **Fixed K within a batch** for T1's CI (K enters θ_K). When the controller
  varies G across prompts, the estimand is θ over the *realized* G distribution
  — re-derive or stratify by G.
- **m known / non-random.** Batch size m is treated as fixed; if groups are
  dropped (e.g. length filtering) m is random — minor, but note it.
- **Verifier is deterministic and noiseless.** A noisy verifier injects extra
  Bernoulli noise that changes p_g's interpretation.

---

## 3. Known risks where the theory could BREAK

1. **U-statistic kernel degeneracy (T1).** If the canonical object is the
   degree-2 U-statistic, the Hájek projection **vanishes at θ_K ∈ {0,1}**. Near
   collapse (ZVF → 1) — *precisely the regime the controller cares about* — the
   limiting distribution is **not** Gaussian (it's a weighted χ²/second-order),
   the rate is m not √m, and the Wald/Wilson CI is wrong. This is the most
   technically dangerous gap: the estimator is least trustworthy exactly where
   it's most decision-relevant.
2. **Non-i.i.d. groups (A3) ⇒ wrong variance.** Positive across-group
   correlation inflates the true Var(ZVF) beyond θ_K(1−θ_K)/m, so the T1 CI is
   **too narrow** (anti-conservative) and the T2 geometric tail ZVF^{n_g} is
   wrong. Mitigation to investigate: block/cluster-robust or batched-bootstrap
   variance; effective sample size m_eff < m.
3. **Improvement ≠ informativeness (T2).** Overclaiming "improvement" when the
   math only gives "nonzero gradient" is a correctness bug, not just a wording
   nit. Reviewers will catch it.
4. **Objective mis-specification (T3).** If S(p,G) doesn't match the actual
   GRPO update's signal, G\* optimizes the wrong thing and the controller
   set-point is miscalibrated. Needs grounding in the gradient, not analogy.
5. **Feedback instability (controller).** The G\* ← φ̂ ← data(G\*) loop may
   oscillate; an unstable controller is worse than a fixed G.
6. **ε / tolerance interaction.** Immaterial under A1 (binary), but if rewards
   are ever near-binary-with-noise, ε starts mattering and Z_g is no longer a
   clean indicator of p_g ∈ {0,1}.

---

## 4. References to locate and cite properly

All currently placeholder `\cite{TODO-...}` keys; **bib entries intentionally
not invented.** Locate and replace:

| Key | What it is | Why needed | Where used |
|---|---|---|---|
| `TODO-shao-grpo` | Shao et al., GRPO / DeepSeekMath (canonical GRPO paper) | Defines group-relative, critic-free RL the whole paper sits on | Intro |
| `TODO-zhou-demystify` | Zhou et al., "Demystifying GRPO" | Their per-prompt analysis of advantages vs (p,K); θ_K = E_φ[h_K(p)] is the population quantity they integrate against; T1 CI = their missing uncertainty quantification | Remark (T1) |
| `TODO-razin` | Razin et al., "mixed-group probability" | 1 − h_K(p) = mixed-group probability; T1 turns their per-prompt quantity into a calibrated estimand. **Confirm exact definition & page/eq.** | Prelim, Remark (T1) |
| `TODO-zvf-pillar1` | ZVF Program Pillar 1 (companion, the mechanistic-diagnostic paper) | Source of the ZVF definition, non-tautology result, scope, cross-framework pipeline | Throughout |

**Verification tasks for citations (do not skip):**
- Confirm `h_K(p) = p^K + (1−p)^K` matches Razin's *exact* "mixed-group" /
  "all-same" definition (complement vs. direct) — get page & equation number.
- Confirm θ_K corresponds to Zhou et al.'s "effective batch fraction" /
  "non-zero-advantage fraction" notion — get page & equation number.
- Get the canonical GRPO citation (DeepSeekMath, Shao et al. 2024) details.
- Pillar-1 is an internal cross-reference — wire it to the actual paper/appendix
  label once the program's bibliography is unified.

---

## 5. Suggested order of attack (for the author)

1. **Decide T1's canonical object** (sample-mean honesty vs. degree-2
   U-statistic). This unblocks everything and determines whether the
   degeneracy analysis is even needed. (Gaps 1–2)
2. **Fix T2's claim wording** to "nonzero update," or commit to proving the
   improvement lemma. Then make it estimation-aware with the one-sided bound.
   (Gaps 6–7)
3. **Ground T3's objective** in the GRPO gradient / Fisher information before
   touching uniqueness or the Beta closed form. (Gap 9)
4. **Stress-test A3** (across-group i.i.d.) empirically on real logs: measure
   the effective sample size / across-group autocorrelation to see how badly the
   CI is anti-conservative in practice.
5. Discharge the mechanical gaps (Wilson formula, DCT integrability, Beta
   moments, rounding) last — they're real but low-risk.
