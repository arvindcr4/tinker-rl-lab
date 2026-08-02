# AntiVibe Senior Audit: P8 — Workshop Artifact

> **Target:** `platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex`  
> **Ship unit:** Do not ship as P9 docs; migrate only recomputing panels into P11  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Workshop writeup upgrades manifest statuses in prose.
- P11 now absorbs matched-budget E-R2b G=2×160 vs G=16×20 (2560 rollouts, 2 seeds).

### Key Decisions
- Regenerate any artifact docs from run_manifest.tex, not main text.
- Drop interrupted 100%/100% frontier row (partial=true).

### Flags (vibe / integrity smells)
- GSM8K label on synthetic two-digit addition task.
- Pooled r=−0.769 flips to +0.35 at run level; verifier returned True unconditionally historically.

### Edge Cases & Failure Modes
- Two-seed matched-budget is trajectory observation, not optimum.

### Testability / Offline checks
- Read er2b_g{2,16}_s{123,456}.json last10 reward/ZVF; assert 2560 rollouts.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
