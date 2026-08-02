# AntiVibe Senior Audit: P4 — Length Bias

> **Target:** `platform_hybrid/paper/paper_P4_length_bias.tex`  
> **Ship unit:** Optional ≤6 pp measurement-validity note only  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Short-horizon GRPO vs Dr.GRPO under tight token caps.
- Portfolio: no length-headroom regime; 'uncapped' panels still at cap.

### Key Decisions
- Null GRPO–Dr.GRPO is not informative under non-identifiability.
- McNemar must be per-seed; pooling 3×200 into 600 outcomes is pseudo-replication.

### Flags (vibe / integrity smells)
- Abstract model name mismatch (Qwen3-8B vs Qwen2.5-1.5B runs) is a standing claim bug.
- Contentless TikZ appendix figures inflate page count without evidence.

### Edge Cases & Failure Modes
- DECISIVE results on ~4.7-token completions are toy-regime.

### Testability / Offline checks
- Per-seed McNemar; report cap occupancy; fix model attribution or drop claim.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
