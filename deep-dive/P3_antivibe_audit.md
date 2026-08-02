# AntiVibe Senior Audit: P3 — Group Size

> **Target:** `platform_hybrid/paper/paper_P3_group_size.tex`  
> **Ship unit:** None as standalone — retire; plateau narrative lives elsewhere if kept  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Paper marketed SNR / G-selection mechanism on arithmetic with heavy FALLBACK_ROWS constants.
- Portfolio: headline SNR is re-expression of 1−ZVF under two-valued advantage_variance.

### Key Decisions
- Do not merge fabricated grid into P2.
- Any kept content is n=3, 0.5B, synthetic arithmetic, no separable G effect.

### Flags (vibe / integrity smells)
- FALLBACK_ROWS ±0.03 band labeled as if measured bootstrap CI.
- G=32 training comparison is sampling-only / non-comparable.

### Edge Cases & Failure Modes
- TOST / T_crit / 1024M extrapolation derive from constants, not data.

### Testability / Offline checks
- Grep FALLBACK_ROWS; refuse any table sourced from it in submissions.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
