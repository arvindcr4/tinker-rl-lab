# AntiVibe Senior Audit: P7 — ZVF Controller

> **Target:** `platform_hybrid/paper/paper_P7_zvf_controller.tex`  
> **Ship unit:** Park; optional absorb 0/1867 + ZVF/PCD separation only  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Retrospective controller + adaptive-G proposal without cost-matched fixed-G arm.
- Audited controller code contains zero PCD occurrences despite contribution (ii).

### Key Decisions
- Retire near-term controller experiment claim.
- Do not import 92.3% figure (P12 shows by-construction base rate).

### Flags (vibe / integrity smells)
- E3 described as GSM8K-style but is two-digit addition.
- U-shape table drops the sole non-monotone model.
- 12 trailing TikZ figures after bibliography → multiply-defined labels.

### Edge Cases & Failure Modes
- Adaptive spends 186 rollouts vs baselines at 120 — not cost-matched.

### Testability / Offline checks
- String-search controller for PCD; count TikZ post-bibliography.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
