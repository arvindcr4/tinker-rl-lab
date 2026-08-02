# AntiVibe Senior Audit: P11 — Single-Stack Reproducibility Audit (spine)

> **Target:** `zvf-program/audit/paper_P11_reproducibility_audit.tex`  
> **Ship unit:** Canonical spine (12 pp) + tmlr_package_p11/  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Preregistered single-stack re-implementation of DAPO/GSPO/Dr.GRPO/AERO vs GRPO.
- Fail-closed 40 arm–seed units; exact noncentral paired-t MDE + BH; all INCONCLUSIVE.
- Secondary cost: DAPO ZVF 0.693→0.000 at 3.61× rollouts.

### Key Decisions
- Drop survival/RETAINS framing when published_delta=null for all arms.
- DISAPPEARS for DAPO is superseded (MDE80=0.01012 > 0.01 margin).
- Absorb E-R2b matched-budget as bounded secondary panel only.

### Flags (vibe / integrity smells)
- Historical DISAPPEARS text must never reappear as a result label.
- Pilot is effective n=1; table is held-out score not Δ vs GRPO.
- Companion cites to unpublished minreport/registry drafts — abstract now uses public prior art.

### Edge Cases & Failure Modes
- Replay can move held-out by 0.004 > DAPO point Δ 0.001.
- Stack lock ≠ sample-budget lock (DAPO 3.61× rollouts).

### Testability / Offline checks
- python3 zvf-program/audit/aggregate_audit.py + unittest; verdicts all INCONCLUSIVE.
- Overlap check vs NeurIPS 36320: drafts/P11_NEURIPS_OVERLAP_CHECK.md.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
