# 8-Paper Originality Map — from Deep Research (2026-07-06)

Source: Gemini Deep Research (complete) + ChatGPT Deep Research (pending, will cross-check).
Raw: `deepresearch_gemini_raw.md`. Brief: `deep_research_brief.md`.

> ⚠️ **Verify every citation before use.** Deep Research can hallucinate/misattribute papers and dates
> (several are dated 2026). Treat the *angles* as the deliverable; confirm each cited paper on arXiv/OpenReview
> before relying on it. Papers to verify: Nimmaturi et al. (scaling), Ghosh et al. "Predictable GRPO",
> λ-GRPO, MC-GRPO, G2RPO-A, AERO, NGRPO, EvilGenie.

## The genuinely-original angle per pillar (Gemini)

| # | Pillar | Original angle to own | Avoid (already done) |
|---|--------|----------------------|----------------------|
| **P1** | Scaling/dynamics | **Layer-Specific Saturation Law** — per-layer reward saturation + asynchronous layer-freezing schedule to cut FLOPs without changing the macro curve | another whole-model sigmoid/exponential saturation fit |
| **P2** | ZVF / zero-advantage | **Cross-Prompt Latent Contrastive Advantage** — when a group collapses to zero variance, compute advantage against a cross-prompt latent baseline (not within-group) | scalar virtual-reward / temperature tricks on a single prompt's group (AERO/NGRPO territory) |
| **P3** | Group size G | **Token-Complexity-Bounded Asynchronous Group Sizing** — define G as a continuous function of a token/compute budget, not an integer count of trajectories | integer-G sweeps / uniform-cost rollout accounting (AERO, MC-GRPO) |
| **P4** | Length bias | **Semantic-Density Time-Decay Token Normalization** — weight tokens by semantic entropy / state-transition surprise, not uniformly | static length penalty or uniform learnable token weighting (Dr.GRPO, λ-GRPO) |
| **P5** | Reporting standard | **GRPO Post-Training Datasheet (GPTD) + Cryptographic Rollout Provenance** — prove the model didn't self-generate/encounter the eval benchmark during millions of rollouts (contamination provenance) | a static checklist that restates Model Cards / Datasheets / repro checklists |
| **P6** | Registry | **Dynamic State-Space Ontology (temporal-graph meta-registry)** — version the *evolving* run state (verifier-logic updates, reward drift) over time, not just launch config | a static JSON schema that duplicates W&B/MLflow run metadata |
| **P7** | Controller | **PID-Controlled Temperature & Clipping** — classical PID control using live advantage-variance / ACR / KL-drift as sensor inputs, with convergence guarantees | heuristic adaptive-G / adaptive rollout volume (AERO, G2RPO-A) |
| **P8** | Fraud/anomaly | **Latent-Space Trajectory Auditing for in-loop fraud interception** — map hidden states during rollout to catch the transition to a "deception/exploitation manifold"; + an **infra-telemetry manipulation / backend-spoof labeled benchmark** | post-hoc output judging / agent-level code reward-hacking (EvilGenie); credit-card fraud (current P8 — drop it) |

## Originality ranking (Gemini)

- **Largest open opportunity (systems/infra white space):** **P5 reporting standard, P6 registry, P8 infra-telemetry fraud** — no labeled dataset or machine-readable spec exists for infrastructure-level RL-run integrity or config cataloging.
- **Middle:** **P1 scaling** (layer-wise angle is open), **P7 controller** (formal control-theory framing is open).
- **Most crowded / hardest to be novel:** **P2 zero-variance** and **P4 length bias** — actively worked (AERO/NGRPO, Dr.GRPO/λ-GRPO); need a fundamentally different axis to avoid derivative work.

## Why this matters (ties to the adversarial reviews)
The reviews called P5/P6/P8 "vaporware **as executed**." The landscape says P5/P6/P8 are the **best originality bets** — *if* executed as the novel versions above (provenance protocol / temporal-graph registry / infra-telemetry benchmark) rather than the thin bookkeeping / wrong-domain versions. So the fix for the three weakest pillars is a **pivot to the open angle**, and P2/P4 (crowded) are where you should *narrow claims* hardest.

---

## Citation verification (web search, 2026-07-06) — angles hold up

**Confirmed REAL prior art (trust the "avoid" guidance):**
- MC-GRPO (median-centered, small-rollout) — arXiv 2601.22582 + GitHub — relevant to P3.
- λ-GRPO (learnable token weighting for length) — HF papers 06/2026 — relevant to P4.
- G2RPO-A (adaptive guidance) — HF papers — relevant to P7.
- GVPO (Group Variance Policy Optimization) — OpenReview, cited 29 — relevant to P2.
- EvilGenie (arXiv 2511.21654) + TRACE (arXiv 2601.20103) — reward-hack *code/output* benchmarks — relevant to P8.

**NEW P2 blocker (not in Gemini's report):** **AVSPO — "Advantage Collapse in GRPO", ICML 2026 poster** — directly reduces advantage collapse 58–63%. → **P2 is the MOST crowded pillar**; hardest to be original. Either narrow the claim sharply or change axis.

**White space CONFIRMED:**
- **P5** — no GRPO-specific reporting standard/datasheet found → open.
- **P8** — existing benchmarks (EvilGenie, TRACE, RHB) are code/output-level reward hacking, **NOT** infra-level telemetry manipulation / backend spoofing → the infra-telemetry + in-loop latent-trajectory-auditing angle is open, but must explicitly differentiate from EvilGenie & TRACE.
- **P6** — no dynamic/temporal RL-run meta-registry found → open.

**Revised originality ranking (verified):** P5 ≈ P6 ≈ P8 (widest open) > P1 ≈ P7 (open axis exists) > P3 (MC-GRPO/AERO crowd it) > **P2 (most crowded — AVSPO/GVPO/AERO/NGRPO)**.
