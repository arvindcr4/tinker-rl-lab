# AntiVibe Senior Audit: P2 — Zero-Variance Fraction

> **Target:** `platform_hybrid/paper/paper_P2_zvf.tex (demoted 46 pp)`  
> **Ship unit:** platform_hybrid/paper/paper_P2_zvf_falsification_note.tex (3 pp)  
> **Date:** 2026-08-02  
> **Contract:** `.claude/skills/antivibe/agents/auditor.md` (senior flags, not tutorial)

### Architecture Summary
- Original long draft mixed real group tensors with variance_mitigation.tsv simulation rows.
- Surviving science: Bernoulli ZVF model sign-fails across regimes (GSM8K vs synthetic arithmetic).
- Short note freezes only recomputable δ = ZVF_iid − ZVF_obs panels.

### Key Decisions
- Delete simulation-as-measured (negative rewards/accuracies in variance_mitigation.tsv).
- Demote pass@G − p^G = 1−ZVF to a lemma, not a 1.11e-16 'discovery'.
- Do not pool ZVF–outcome correlations across heterogeneous cells.

### Flags (vibe / integrity smells)
- Any residual abstract language about AUROC/collapse prediction is a claim leak.
- Seed order in summary JSON was previously swapped; keep zvf_per_seed aligned to seeds[].

### Edge Cases & Failure Modes
- GSM8K δ mean +0.1224 CI excludes 0; arithmetic δ mean −0.0703 CI excludes 0 — opposite signs.
- Importing P7 92.3% base-rate figure would re-inject a by-construction artifact.

### Testability / Offline checks
- Recompute from tinker_gsm8k_zvf_s{42,123,456}.json and groupsize_zvf_sweep.json only.
- Bootstrap with fixed RNG seed for CI bit-stability.

    ---
    *Replaces the 2026-08-02 template clone that only swapped the paper title.
    Grounded in `drafts/PORTFOLIO_DECISION.md` + 12-paper verification wave.*
