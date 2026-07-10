# Gameplan — Applying the Agentic RL Survey to My Work

Source: **"The Landscape of Agentic Reinforcement Learning for LLMs: A Survey"**
(arXiv:2509.02547v4, Jan 24 2026, published TMLR 01/2026)
Local copy: `~/Downloads/agentic-rl-llms-survey-2509.02547v4.pdf`
OpenReview: https://openreview.net/forum?id=RY19y2RI1O

Highest-value sections for my work: **Table 2** (GRPO variants), **§6.2** (training efficiency), **§6.4** (mechanistic debate).

---

## 1. ZVF Program (`tinker-rl-lab/zvf-program/`) — biggest wins

### 1.1 Position against the now-canonized related-work set (Table 2)
The direct competitor set for zero-variance filtering:

| Method | Mechanism | Relation to ZVF |
|---|---|---|
| **GRESO** | Pre-rollout filtering | Closest neighbor — must differentiate explicitly |
| **DAPO** | Decoupled clip + dynamic sampling (drops zero-gradient groups) | Overlapping motivation |
| **EDGE-GRPO** | Entropy-driven advantage + guided error correction to mitigate **"advantage collapse"** | Adopt "advantage collapse" as terminology — survey-blessed vocabulary |
| **DARS** | Reallocates rollouts from medium to hardest problems | The reallocation counterpart to filtering |
| TreePo | Self-guided rollout to cut compute | Secondary |
| Skywork R1V2 | Selective sample buffer | Secondary |
| Dr.GRPO | Eliminates bias in GRPO objective | Cite in theory pillar |
| Posterior-GRPO | Rewards only successful processes | Secondary |

**Action:** audit theory/position drafts — if GRESO and DAPO aren't explicitly differentiated, fix before submission. First thing a TMLR-caliber reviewer will hit.

### 1.2 Upgrade the thesis: from "compute savings" to "capability regime" (§6.4)
Survey's empirical synthesis: RL creates *genuinely new* capability (vs. amplifying existing ones) when:
1. rewards are high-fidelity / verifiable (executable, formally checkable);
2. tasks have compositional multi-step structure;
3. the base model is in the **intermediate regime** — neither near-random nor near-ceiling.

Zero-variance groups are exactly the near-random (all-fail) and near-ceiling (all-pass) cells.
→ **Reframe ZVF: it concentrates gradient signal in the regime where RL installs new capability rather than reweighting old ones.** Belongs in the position paper's introduction.

### 1.3 Fix evaluation protocol before the real 403-cell sweep
- Report **pass@k (k≈32) alongside pass@1** per sweep cell. ~2/3 of RL papers only report pass@1; pass@k frontiers distinguish amplification from new capability (§6.4). Cheap to add, big positioning payoff.
- **Multiple seeds** on at least a diagonal subset — Vattikonda et al.: RL is extremely hyperparameter-sensitive at scale.
- Consider **reference-policy reset** (ProRL) as a sweep axis for long runs.

### 1.4 Natural follow-up pillar: filtering → reallocation
DARS + Self-Evolving Curriculum (§3.4: problem selection as non-stationary bandit) suggest:
treat the rollout budget freed by dropping zero-variance groups as a **bandit allocation problem** over remaining cells.
Clean incremental paper on top of existing infrastructure.

---

## 2. Fast-apply / OAPL thread

- **§6.2 recipe for Osmosis-Apply-style small model:** large teacher generates SFT data → on-policy RL refinement. Reported to beat either alone at ~half the compute of pure SFT (Vattikonda et al.).
- **CHORD**: SFT as dynamically weighted auxiliary loss inside on-policy RL — the more integrated variant.
- Use **diff-match / execution rewards** (verifiable reward = the condition where RL reliably delivers per §6.4).
- **Gap = opportunity:** the survey treats off-policy methods as background (one line in §2) and never engages them for LLM agents. OAPL (arXiv:2602.19362, 3× sample efficiency vs GRPO) targets territory this 500-paper survey leaves uncharted → good novelty position.

---

## 3. Fraud LLM vs XGBoost (`~/fraud-llm-vs-xgboost`)

- **"Hallucination tax" finding (§6.1):** outcome-only RL degrades abstention; mixing solvable/unsolvable training examples restores it.
  → Give the LoRA model an explicit **abstain/escalate class**, train with unanswerable cases.
- **Guru-dataset finding (§6.2):** RL gains track pretraining exposure; under-exposed domains (tabular/simulation) need dedicated training.
  → Ready-made explanation for why Qwen-0.5B trails XGBoost on tabular fraud.

---

## 4. Maintainer-orchestrator loop

§6.5 deployment patterns validate the supervisor-worker design. Steal:
- Trigger **HITL review on model uncertainty signals**, not fixed rules.
- Treat the hierarchy as a **credit-assignment structure**, not just an org chart.
- Sandboxing + process-based rewards penalizing unsafe intermediate steps (§6.1) for autonomous repo work.

---

## Next step (optional)
Diff ZVF theory/position drafts against the claims and citations above to find what's missing — where this survey converts most directly into revisions.
