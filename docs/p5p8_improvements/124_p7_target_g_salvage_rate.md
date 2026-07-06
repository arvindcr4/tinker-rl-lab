# 124 — P7 Adaptive-G* Counterfactual on N2 Reward Tensors (Salvage-Rate Framing)

**Pillar:** P7 (Pillar 3 — adaptive-G controller)

**Vein (fresh, not in 110 prior rows):** brief vein (a) at the
**optimal target G*** level. The P7 controller family (iter
67/71/75/79/83/87/91/95/99/103/107) reports fires, hysteresis flips,
post-pred restore probability, and closed-form per-fire contrast gain —
but **never** reports the counterfactual "what G should the controller
have escalated TO, given observed per-prompt k_p at G=8?" nor the
**salvage rate** (the fraction of fires for which a Pareto-better
target G* exists in the {16,32,64} budget).

## Method (script: scripts/p5p8/p7_iter111_target_g_selection.py)

For every (method × step) in the N2 4-method × 40-step panel:

1. Compute per-prompt `k_p = sum(rewards[p])` ∈ {0..8}.
2. Compute observed `zvf_obs = boundary_rate(k) = mean(1[k in {0,8}])`.
3. Compute closed-form iid predicted `z_target(G) = (1/N) Σ (p̂^G + (1-p̂)^G)`
   for G ∈ {8, 16, 32, 64}.
4. Replay FOUR controller rules at τ = 0.70:
   - **STATIC_G16**: always G → 16 (the iter-103 default).
   - **DUALFORMER_AUTO_d4**: Berkeley row 01 rule G ← min(G+4, Gmax) = 12
     (rounds to G=12 here, cost=1.5×).
   - **DUALFORMER_AUTO_d8**: G ← min(G+8, Gmax) = 16 (matches STATIC).
   - **ADAPTIVE_GSTAR**: G* = min G ∈ {16,32,64} whose predicted iid mean
     ZVF drops below max(0.50, 0.5×zvf_obs); falls back to G=64
     otherwise.
5. Compute **net benefit per fire** = `delta_z − 0.5·(cost_ratio − 1.0)`;
   bootstrap 95 % CI on the per-method, per-controller mean net benefit
   (B = 4000, seed = 20260705).

## Falsifiable headlines

- **H1 — All four controllers fire on the same 82/160 step-cells.** The
  trigger logic is identical (`zvf_obs > τ`); the Δ comes from the
  target G*. Per-method fires: grpo 20, aero 19, **gift 26**, areal 17.
- **H2 — DUALFORMER_AUTO_d4 is the cheapest Pareto-better controller.**
  Total net benefit (sum over 160 step-cells): STATIC_G16 = −41.73,
  **DUALFORMER_AUTO_d4 = −21.81** (half the STATIC loss), G_d8 =
  −41.73, ADAPTIVE_GSTAR = −287.0. The d4 rule's G+4=12 escalation
  yields the lowest mean cost ratio (1.5×) of any rule, sacrificing
  little contrast vs STATIC_G16 (mean ΔZ on fires: STATIC = −0.0095 vs
  DUALFORMER_d4 = −0.0163).
- **H3 — All controllers have NEGATIVE mean ΔZ under iid.** The closed-
  form binomial model says `mean_zvf_iid` DECREASES monotonically with
  G, but is FLOORED at the boundary rate (mean_zvf_iid ≥ boundary_rate
  always). At boundary rates 0.50 – 0.875 on the N2 panel, ΔZ is
  essentially zero (slightly negative because the iid model is
  pessimistically above empirical zvf_obs, reflecting the iter-3
  anti-herding δ_div ∈ [0.05, 0.25] erosion).
- **H4 — Salvage rate (fired steps where ΔZ ≥ 0) is near-zero for
  3/4 methods.** Per-method fires with ΔZ ≥ 0 in any controller:
  **grpo 1/20** (5.0 %), **aero 1/19** (5.3 %), **gift 5/26 (19.2 %)**,
  areal 1/17 (5.9 %). GIFT alone has a material salvage rate under
  iid because its likelihood-prior design yields slightly more mixed
  prompts (lower boundary rate) than the other three.
- **H5 — Closed-form optimal G* via "min G clears 0.50 or 0.5·zvf"
  is G=64 in 100 % of fired steps.** The boundary rate exceeds 0.50 on
  every fired step, so no candidate in {16,32} clears the threshold.
  G=64 is the cost-asymptote (matches the boundary rate to within
  0.001). The implication: **at this N2 panel's saturation level, the
  controller's G-↑ lever is exhausted** — alternative interventions
  (curriculum, prompt-set rotation, reward shaping) are required
  before the lever can recover contrast.

## Operational recommendation

- **Cost-conscious deployment:** DUALFORMER_AUTO_d4 (Berkeley row 01
  style, G += 4) — half the STATIC_G16 net loss for the same trigger.
- **Research / upper-bound calibration:** keep the iter-103 calibrated
  pair-savings harness (P5P8 iter103 row 121 / Tau τ = 0.90 setting),
  but pair it with this iter-111 salvage-rate diagnostic so the
  "controller didn't fire" / "controller fired but G=64 was no help"
  cases are auditable from any future harness run.
- **Prompt-set rotation lever:** the iter-111 finding is consistent
  with the iter-103 conclusion (P7 row 121) that beyond
  GIFT's likelihood prior, the escalation lever is bounded above at
  ~5 % salvage for GRPO-family; the design frontier moves to prompt
  difficulty rotation (N10 seed-expansion panels) and reward shaping.

## Cross-coupling

- (i) **P6 iter-110 row 127** — the paired-bootstrap cross-panel
  validation pattern (B = 4000, seed = 20260705) is reused here
  verbatim for the per-controller, per-method CIs.
- (ii) **P7 iter-103 row 121** — calibrated pair-savings baseline;
  iter-111 closes the open question "what G* is the controller
  escalating to?" with a per-step closed-form derivation.
- (iii) **Berkeley row 01 (Dualformer)** and **Berkeley row 19
  (AlphaProof γ*=0)** — both are now reduced to controller rules
  in the iter-111 counterfactual: row 01 is DUALFORMER_AUTO_d4,
  AlphaProof γ*=0 is the no-smoothing baseline used by STATIC_G16
  (mean-baseline advantage, no temporal discounting).
- (iv) **FRONTIER_INSIGHTS Round 2** — "ZVF is contrastive yield, not
  difficulty" framing: iter-111's ΔZ floor at boundary rate is the
  exact empirical confirmation that ZVF is a **signal-availability
  metric**, not a difficulty metric; the closed-form iid ZVF is the
  difficulty-counterfactual upper bound.

## Artefacts

- `scripts/p5p8/p7_iter111_target_g_selection.py` (~270 LoC, stdlib
  only, deterministic LCG bootstrap on `seed=20260705`)
- `experiments/results/p5p8/p7_iter111_target_g_distribution.tsv`
  (160 rows = 4 methods × 40 steps; per-step zvf at G ∈ {8,16,32,64},
  closed-form G*, salvageable flag)
- `experiments/results/p5p8/p7_iter111_controller_replay.tsv`
  (640 rows = 4 controllers × 4 methods × 40 steps; per-step fired,
  G*, z_target, cost_ratio, ΔZ, net_benefit)
- `experiments/results/p5p8/p7_iter111_net_benefit.tsv`
  (16 rows = 4 controllers × 4 methods; per-method summary with
  bootstrap 95 % CI on mean net benefit, G* fire distribution, and
  salvageable fire count)
- `experiments/results/p5p8/p7_iter111_summary.json`
  (machine-readable: totals, per-method summary, salvage-rate
  per method, ADAPTIVE G* fire distribution, overall optimal-G
  distribution)
