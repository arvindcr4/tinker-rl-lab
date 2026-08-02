# AntiVibe Senior Audit: P9 — DNB Benchmark

> **Target:** `platform_hybrid/paper/neurips_2026_variants/paper_P9_dnb.tex`  
> **Ship unit:** Rebuild only after single-ledger reconciliation  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Benchmark/instrumentation claim with advertised make reproduce-main.
- Makefile lacks reproduce-main; frontier table disagrees with named JSON sources.

### Key Decisions
- Rebuild every table from one named ledger; drop short traces.
- Exclude status=failed from ZVF pools.

### Flags (vibe / integrity smells)
- Compute card 60–180× off and wrong hardware class.
- Tier A defined four incompatible ways.

### Edge Cases & Failure Modes
- Last-10 over 3–5 points sold as last-10.

### Testability / Offline checks
- make -n reproduce-main must exist before any ARTIFACT_CANDIDATE claim.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
