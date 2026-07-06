# P1–P7 critique synthesis — Deep Think + GPT‑5.5 Pro (2026-07-06)

Raw: `p1p7_critique_deepthink_raw.md`, `p1p7_critique_gpt55pro_raw.md`. Both = adversarial ICLR/NeurIPS reviewer + M.Tech examiner.

| Pillar | Deep Think | GPT‑5.5 Pro | New prior-art threat (verify!) | Consensus |
|---|---|---|---|---|
| **P1 scaling** | needs-pivot → freeze by forward-KL | needs-pivot (#2) → **causal predictive** layer-freeze, ≥25–40% FLOP saving at matched KL | **SALF** (Semantic-Aware Layer-Freezing) | **Pivot to causal/predictive; strong #2** |
| **P2 advantage** | needs-pivot → k-NN difficulty isocline | **too-crowded-avoid** → needs a theorem | **BV-Blend** ("almost directly in your lane"), AVSPO | **Weak — my real experiment already showed cross-prompt baseline is a dead-end** |
| **P3 group size** | too-crowded-avoid → variance-gated halting | needs-pivot → token-budgeted allocation + **staleness bound** | **VIP**, Pilot-Commit, DARS | **Pivot: drop "continuous G" → token-cost + staleness** |
| **P4 length** | **STRONG (#1)** → free KL-surprise mask | **too-crowded-avoid (#6)** → "entropy-weighted GRPO with a nicer name" | **EAPO** (token-level credit in RLVR) | **⚠️ SPLIT** — only survives if semantic density is **verifier-causal**, not entropy |
| **P5 provenance** | needs-pivot → active LSH regularizer (#2) | **STRONG (#1)** → reference protocol + verifier, precise threat model | **Atlas** (attestable ML lineage) | **✅ CONSENSUS TOP PICK (all 4 AI passes)** |
| **P6 registry** | too-crowded-avoid → Meta-RL rollback | needs-pivot → **fold into P5** as run-forensics | W&B artifact lineage | **Merge into P5** |
| **P7 controller** | needs-pivot → Control-Lyapunov/CBF proof | **too-crowded-avoid** | **AGPO**, TAMPO, ABC-GRPO | **Avoid — AGPO owns the control surface** |

## The signal across ALL passes (2 Deep Researches + 2 critiques + 1 real experiment)
1. **P5 is the single most-endorsed opening** — top/strong in every pass. Build it as a **reference provenance protocol + verifier** (Merkle commitments over prompts/rollouts/reward/verifier-code/RNG/checkpoints/eval-queries) with a **precise, tiered threat model** — NOT a datasheet, NOT "PROVE absolute non-contamination." Threat to beat: Atlas. **P6 folds in as the run-forensics layer.**
2. **P1 is the best *technical* paper** — but only as **causal predictive layer-freezing** (predict freezable layers from first 5–10% of training; show FLOP savings at matched KL/reward). Threat: SALF.
3. **P3 is salvageable** as token-budgeted rollout allocation + staleness bounds (drop "continuous G"). Threat: VIP.
4. **P2 is weak** — my real experiment (72% easy-collapse, isocline recovers ≈0) + both critics + BV-Blend all say the cross-prompt-baseline idea is a dead end. Redirect to difficulty-curriculum or shelve.
5. **P4 is a coin-flip** — Deep Think loves it, GPT kills it (EAPO). Only pursue with a **verifier-causal** span-weighting (counterfactual masking), not entropy weighting.
6. **P7: drop** — AGPO/TAMPO/ABC-GRPO already occupy adaptive temperature+clipping.

## Recommended focus (2–3 papers)
**P5 (provenance protocol+verifier, absorbing P6)** = the flagship original contribution · **P1 (causal layer-freeze)** = the technical paper · then **P3 (token-budget+staleness)** as the third. Verify every named threat (SALF, BV-Blend, VIP, EAPO, AGPO, Atlas) on arXiv before committing.
