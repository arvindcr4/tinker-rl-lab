# AntiVibe Senior Audit: P6 — GRPO Registry

> **Target:** `platform_hybrid/paper/paper_P6_registry.tex`  
> **Ship unit:** Merge with P5 after integrity disclosure  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Machine-readable registry + stackdiff + prediction audits.
- Integrity flag: iter-194 amendments flipped AERO/AREAL predicted signs after observing negatives.

### Key Decisions
- Disclose amendment prominently or delete prediction audit that became supports_rate=1.0.
- Reconcile abstract registry counts to shipped artifact (not 3 vs 11 vs 48).

### Flags (vibe / integrity smells)
- Post-hoc sign flip is the most serious integrity smell in the portfolio.
- 17× exhibit shared with P5 — assign one home only.

### Edge Cases & Failure Modes
- Schema validation 44/48 is resource-tier, not empirical ranking.

### Testability / Offline checks
- Registry schema validate; show amendment timeline vs observed deltas.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
