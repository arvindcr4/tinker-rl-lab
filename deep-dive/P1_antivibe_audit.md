# AntiVibe Senior Audit: P1 — GRPO Scaling Laws

> **Target:** `platform_hybrid/paper/paper_P1_scaling.tex (+ sections/scaling_*.tex)`  
> **Ship unit:** platform_hybrid/paper/paper_P1_identifiability_note.tex (2 pp short unit)  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Long draft is a multi-iter elevation archive (~47 pp) over managed Tinker GSM8K anchors (0.6B–1T).
- Scientific object is cross-scale identifiability of reward/saturation laws, not a powered multi-seed RCT.
- Short note is the publication unit; long draft is thesis/archive.

### Key Decisions
- Negative claim over positive law: flat slopes + λ at bound + constant AICc winners is the honest story.
- Withdraw MoE-vs-dense (+0.338, p=0.023): Nemotron-3-Super is LatentMoE; mislabelled dense collapse anchor.
- Strike external 'pre-registered' language for internal iteration-ledger predictions.

### Flags (vibe / integrity smells)
- Long PDF still looks like a scaling-law paper by page count — submit the 2 pp note only.
- Most anchors are single-seed descriptive; any significance language is a regression risk.
- Leave-Nemotron sensitivity had no trusted TSV; do not resurrect it.

### Edge Cases & Failure Modes
- HF/W&B identity conflicts: only HF-arbitrated Nemotron path is authoritative (0.55 zero-reward exact).
- 671B vs 685B-class DeepSeek naming must stay reconciled.

### Testability / Offline checks
- Offline: recompute Nemotron zero-frac from frontier_gsm8k_nemotron-120b.json reward_trace.
- Offline: permutation tables in scaling_law_iter109b_permtest.tsv.
- No GPU required for the short-note claims.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
