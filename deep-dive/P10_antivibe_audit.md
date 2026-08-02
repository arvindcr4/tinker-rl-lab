# AntiVibe Senior Audit: P10 — ZVF Theory

> **Target:** `zvf-program/theory/paper_P10_zvf_theory.tex`  
> **Ship unit:** Theorem core T1–T3 only; strip empirics/placeholders  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Theory note with asymptotic CI for Bernoulli ZVF + wasted-compute bounds.
- Standalone PDF still ships placeholder figures and non-recomputing empirics.

### Key Decisions
- Algebraic T1–T3 are reusable into P2 note methods layer.
- Empirical confirmation tables without sources are X-tier.

### Flags (vibe / integrity smells)
- Theory assumes i.i.d. Bernoulli groups that P2 falsifies as sampling model — state scope.
- Duplicate TikZ dependency maps.

### Edge Cases & Failure Modes
- E-T1 coverage claims need their JSON artifacts present.

### Testability / Offline checks
- Compile with only theorem sections; no figure file required.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
