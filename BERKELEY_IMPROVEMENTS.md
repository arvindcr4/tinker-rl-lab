# Berkeley RDI Agents Curriculum → TinkerRL-Bench Improvement Ledger

_Ranked ledger of every improvement idea ever logged, scraped from the three
Berkeley RDI agents courses (F24, SP25, F25). Status: proposed → prototyped →
validated → rejected._

| id | source lecture | target | idea (1 line) | status | evidence path |
| --- | --- | --- | --- | --- | --- |
| 05 | F24 L10 — Percy Liang (Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models) | A2 + A1 | Cybench-style 4-tier (Easy/Medium/Hard/Expert) capability-graded decomposition of the 5 Pillar-1 anchors sharpens the iter125/129 2-tier gap: H2 (|tier-frontier ρ| > |global ρ|) DECISIVE (1.000 vs 0.065); H3 (bimodality is the L2/L3 cut) DECISIVE (4-tier L1−L4 gap 0.674 > 2-tier 0.609); RQS covariate non-monotone across tiers (L2=0.76 > L1=0.60 > L3=0.35 > L4=0.00) | **validated** | `experiments/results/berkeley/cybench_{tier_assignment,tier_scaling,tier_shift}.tsv` + `cybench_summary.json` ; `docs/berkeley_improvements/05_cybench_capability_tiers.md` ; `scripts/berkeley/cybench_capability_tiers.py` |
| 01 | F24 L8 — Yuandong Tian (Dualformer) | A5 + A3 | Reframe GRPO G as Dualformer fast/slow/auto mode allocation; auto-mode saves 56% compute vs always-G=16 on iter127 n=20 | **prototyped** | `experiments/results/berkeley/dualformer_*.tsv` ; `docs/berkeley_improvements/01_dualformer_fast_slow_auto_grpo.md` ; `scripts/berkeley/dualformer_fast_slow_auto.py` |
| 02 | SP25 L2 — Jason Weston (DPO, Iterative RPO, CoVe) | A3 | Show GRPO single-pair loss = DPO small-β online limit; Iterative RPO ≡ GRPO+replay; iter123 SNR slope (+0.366/dec) contains theory (+0.500); G*_IRPO = G*_GRPO at every T | **prototyped** | `experiments/results/berkeley/dpo_iterative_rpo_{grpo_equivalence,snr_scaling,optimal_g,loss_equivalence}.tsv` + `dpo_iterative_rpo_summary.json` ; `docs/berkeley_improvements/02_dpo_iterative_rpo_vs_grpo.md` ; `scripts/berkeley/dpo_iterative_rpo_vs_grpo.py` |
| 03 | _rejected_ | _open_ | _LLM-as-Optimizer difficulty-aware prompt pool — Newton-style iterative prompt mutation; defer: requires prompt-mutation pipeline not yet in stack; lower ROI than row 08 (Pillar-1 covariate already covered)_ | rejected | _—_ |
| 04 | _proposed_ | _open_ | _next idea: SP25 L12 Dawn Song (DataSentinel/AgentPoison/Progent) → orchestrator guardrail audit + test cases_ | proposed | _—_ |
| 06 | _proposed_ | _open_ | _next idea: F24 L9 Jim Fan (Eureka) → LLM-designed reward functions as Pillar 1 scaling-law exogenous variable_ | **PROMOTED → row 08 (validated)** | _see row 08_ |
| 07 | F25 L8 — Sida Wang (Adding Error Bars to Evals / Measuring all the noises of LLM Evals) | A1 | Audit 7 Pillar-1/2/3/4 headline numbers under Miller's recipe (pair/non-pair bootstrap, equiv-region TOST); 4 DECISIVE (H1, H2, H6, H7), 3 NULL (H3, H4, H5) | **prototyped** | `experiments/results/berkeley/adding_error_bars_audit.tsv` + `adding_error_bars_summary.json` ; `docs/berkeley_improvements/03_adding_error_bars_to_evals.md` ; `scripts/berkeley/adding_error_bars_to_evals.py` |
| 08 | F24 L9 — Jim Fan (Eureka — Human-Level Reward Design via Coding LLMs) | A3 | Reward-Design Quality Score (RQS = geometric mean of {variance, frac_above_0.5, peak−trough, 1−2·zero_frac}) as Pillar-1 exogenous covariate on 12 anchors + iter127 n=20 cells; cap-alone ΔAICc +1.14 vs cap+RQS (borderline NULL), 12-anchor cap-residual ρ(RQS)=+0.225 SUGGESTIVE, iter127 cross-pillar ρ(richness, residual)=−0.569 **p=0.029** DECISIVE | **validated** | `experiments/results/berkeley/eureka_{rqs_per_anchor,aic_compare,aic_anchors,residualization,cross_pillar}.tsv` + `eureka_summary.json` ; `docs/berkeley_improvements/04_eureka_reward_design_quality.md` ; `scripts/berkeley/eureka_reward_design_quality.py` |

## Conventions

- **id**: two-digit zero-padded, monotonically increasing as new entries land.
- **source lecture**: `<semester> L<n> — <speaker> (<key paper>)`. The key paper
  must be a real arXiv paper; we verify the citation (title/authors/year/venue)
  via arXiv MCP / WebFetch before adding the row.
- **target**: A1 (statistical rigor), A2 (eval methodology), A3 (post-training
  science), A4 (tool-use / agentic RL), A5 (inference-time reasoning), B1
  (orchestrator), B2 (safety/security). See `BERKELEY_IMPROVEMENT_BRIEF.md`.
- **idea**: one-line description of the concrete improvement (not a topic).
- **status**: `proposed` (idea, not yet prototyped) → `prototyped` (code +
  initial TSV) → `validated` (TSV shows a real effect, written up in
  `docs/berkeley_improvements/`) → `rejected` (prototyped, no useful effect).
- **evidence path**: path to the script, the TSV outputs, and the proposal doc.

## Update protocol

1. Pick a new id (next integer).
2. Add a row at the top.
3. Update the matching `docs/berkeley_improvements/<id>_<slug>.md`.
4. Append one line to `./AUTORESEARCH_FINDINGS.jsonl` with pillar `B-F24` /
   `B-SP25` / `B-F25` / `B-SYNTH`.
5. Commit when the row reaches `prototyped` or beyond.
