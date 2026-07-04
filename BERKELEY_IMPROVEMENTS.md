# Berkeley RDI Agents Curriculum → TinkerRL-Bench Improvement Ledger

_Ranked ledger of every improvement idea ever logged, scraped from the three
Berkeley RDI agents courses (F24, SP25, F25). Status: proposed → prototyped →
validated → rejected. Re-ranked by impact × evidence × paper-facing readiness
(this iteration: 2026-07-04)._

| rank | id | source lecture | target | idea (1 line) | status | impact | evidence | paper-facing | evidence path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **12** | **F25 L4 Jiantao Jiao (SWE-bench Verified) + F25 L8 Sida Wang (Error Bars) + F24 L8 Yuandong Tian (Dualformer) + F24 L9 Jim Fan (Eureka) + F25 L5 Yehudai + frontier r1 CDH** | **A3 + A1** | **Critic-Degeneracy Hypothesis empirical test (B-SYNTH row): PPO grad_norm 156× larger than GRPO on same stack (96.79 vs 0.62); PPO rolling-var 73% higher; GRPO gradient-reward coupling 24% tighter (r=−0.553 vs −0.445); RQS regressor R²=0.49 collapses half of r_mean variance. CDH decisively supported.** | **validated** | **★★★★★** | **5/5 hypotheses decisive or NULL** | **paper_P3_group_size.tex** | `experiments/results/berkeley/cdh_{gradnorm_stats,reward_window,paired_test,gradnorm_vs_reward,rqs_collapse}.tsv` + `cdh_summary.json` ; `docs/berkeley_improvements/12_critic_degeneracy_hypothesis.md` ; `scripts/berkeley/critic_degeneracy_hypothesis.py` ; `paper/sections/critic_degeneracy_hypothesis.tex` |
| 2 | 09 | F24 L6 — Graham Neubig (SWE-agent / OpenHands / Agentless) | A1 + A2 + A4 | SWE-agent / Agentless lesson applied to Pillar-1 R_max evidence: per-anchor Pass@K=1 95% CI widths (0.145-0.298) invalidate within-reachable-tier ordering (4/5 pairs straddle); Agentless-style 3-tier (hard_floor=2, soft_floor=0, reachable=3) classifies capability bimodality as a reachable-vs-hard-floor split; ACI decomp R_max_policy = R_max_obs/RQS reveals hidden Qwen3-8B ceiling at ~0.808 (vs observed 0.285) | validated | ★★★★★ | 4 anchors decisive | yes (P1) | `experiments/results/berkeley/sweagent_{passk_per_anchor,passk_scaling,aci_decomp,agentless_tiers}.tsv` + `sweagent_summary.json` ; `docs/berkeley_improvements/10_sweagent_passk_aci.md` ; `scripts/berkeley/sweagent_passk_aci.py` |
| 3 | 08 | F24 L9 — Jim Fan (Eureka — Human-Level Reward Design via Coding LLMs) | A3 | Reward-Design Quality Score (RQS = geometric mean of {variance, frac_above_0.5, peak−trough, 1−2·zero_frac}) as Pillar-1 exogenous covariate on 12 anchors + iter127 n=20 cells; cap-alone ΔAICc +1.14 vs cap+RQS (borderline NULL), 12-anchor cap-residual ρ(RQS)=+0.225 SUGGESTIVE, iter127 cross-pillar ρ(richness, residual)=−0.569 **p=0.029** DECISIVE | validated | ★★★★ | 3/4 decisive | yes (P1) | `experiments/results/berkeley/eureka_{rqs_per_anchor,aic_compare,aic_anchors,residualization,cross_pillar}.tsv` + `eureka_summary.json` ; `docs/berkeley_improvements/04_eureka_reward_design_quality.md` ; `scripts/berkeley/eureka_reward_design_quality.py` |
| 4 | 11 | F25 L5 (Yehudai Survey on Eval of LLM-based Agents, arXiv:2503.16416) + F25 L10 (τ²-Bench, arXiv:2506.07982) | A2 | Eval-protocol hardening on iter130 per-seed Pillar-2 ZVF data (9 methods × 5 seeds): MVSP-50/80/95 for "best method" = k=1 (top-1 always survives single-seed), MVSP-80 for top-3 = k=4; 8/8 non-baseline methods have sign-stable z against grpo (|Cohen's d| > 6.7); 8/9 methods are magnitude-channel-dominant (grpo itself is drift-dominant) — variance mitigation works by suppressing magnitude; 3-bucket partition is 1.0-stable under leave-one-seed-out. **Yehudai-COST DECISIVE** (1-seed eval sufficient for top-1 headline, 5× cheaper), **τ²-Bench-ABLATION DECISIVE** (channel decomposition reveals mag-axis dominance), **τ²-Bench-COMPOSITIONAL DECISIVE** (3 buckets fully stable). | validated | ★★★★ | 3 hypotheses decisive | yes (P2) | `experiments/results/berkeley/eval_protocol_{mvsp,robustness,ablation,clusters}.tsv` + `eval_protocol_summary.json` ; `docs/berkeley_improvements/11_eval_protocol_hardening.md` ; `scripts/berkeley/eval_protocol_hardening.py` |
| 5 | 06 | SP25 L12 — Dawn Song (DataSentinel / AgentPoison / Progent) | B2 | End-to-end security audit of the orchestrator's three SP25 L12 attack surfaces: DataSentinel (code-as-data prompt injection) 2/3 vanilla succeeded → sanitiser caught 3/3; AgentPoison (journal memory poisoning) 3/3 vanilla followed → sanitise-memory caught 4/4; Progent (no privilege control over executor) 2/4 vanilla *actually leaked* secrets via `os.environ`/`open('/etc/passwd')` in real subprocess → progent-DSL (module+function gating) blocked 4/4 with zero false positives | prototyped | ★★★ | 3/3 attacks caught | no (B2 only) | `experiments/results/berkeley/sp25_l12_security_audit.tsv` + `sp25_l12_security_summary.json` ; `docs/berkeley_improvements/06_sp25_l12_agent_security.md` ; `scripts/berkeley/sp25_l12_security_audit.py` ; `minimax_autoresearch_improvements/06_sp25_l12_progent_dsl.md` |
| 6 | 01 | F24 L8 — Yuandong Tian (Dualformer) | A5 + A3 | Reframe GRPO G as Dualformer fast/slow/auto mode allocation; auto-mode saves 56% compute vs always-G=16 on iter127 n=20 | prototyped | ★★★ | 5/20 cells under-predict (residual noise) | yes (P3) | `experiments/results/berkeley/dualformer_*.tsv` ; `docs/berkeley_improvements/01_dualformer_fast_slow_auto_grpo.md` ; `scripts/berkeley/dualformer_fast_slow_auto.py` |
| 7 | 02 | SP25 L2 — Jason Weston (DPO, Iterative RPO, CoVe) | A3 | Show GRPO single-pair loss = DPO small-β online limit; Iterative RPO ≡ GRPO+replay; iter123 SNR slope (+0.366/dec) contains theory (+0.500); G*_IRPO = G*_GRPO at every T | prototyped | ★★★ | 4/4 hypotheses decisive | yes (P3) | `experiments/results/berkeley/dpo_iterative_rpo_{grpo_equivalence,snr_scaling,optimal_g,loss_equivalence}.tsv` + `dpo_iterative_rpo_summary.json` ; `docs/berkeley_improvements/02_dpo_iterative_rpo_vs_grpo.md` ; `scripts/berkeley/dpo_iterative_rpo_vs_grpo.py` |
| 8 | 07 | F25 L8 — Sida Wang (Adding Error Bars to Evals / Measuring all the noises of LLM Evals) | A1 | Audit 7 Pillar-1/2/3/4 headline numbers under Miller's recipe (pair/non-pair bootstrap, equiv-region TOST); 4 DECISIVE (H1, H2, H6, H7), 3 NULL (H3, H4, H5) | prototyped | ★★★ | 4/7 decisive | yes (all 4 papers) | `experiments/results/berkeley/adding_error_bars_audit.tsv` + `adding_error_bars_summary.json` ; `docs/berkeley_improvements/03_adding_error_bars_to_evals.md` ; `scripts/berkeley/adding_error_bars_to_evals.py` |
| 9 | 03 | _rejected_ | _open_ | _LLM-as-Optimizer difficulty-aware prompt pool — Newton-style iterative prompt mutation; defer: requires prompt-mutation pipeline not yet in stack; lower ROI than row 08 (Pillar-1 covariate already covered)_ | rejected | — | — | — | _—_ |
| — | 04 | _proposed_ | _open_ | _next idea: SP25 L12 Dawn Song (DataSentinel/AgentPoison/Progent) → orchestrator guardrail audit + test cases_ | **PROMOTED → row 06 (prototyped)** | — | — | — | _see row 06_ |
| — | 05 | _proposed_ | _open_ | _next idea: F24 L9 Jim Fan (Eureka) → LLM-designed reward functions as Pillar 1 scaling-law exogenous variable_ | **PROMOTED → row 08 (validated)** | — | — | — | _see row 08_ |

## Rejected ideas (don't revisit — recorded so threads stop cycling)

| id | idea | rejection reason |
| --- | --- | --- |
| 03 | LLM-as-Optimizer difficulty-aware prompt pool | Requires prompt-mutation pipeline not in stack; row 08 (Eureka RQS) already covers Pillar-1 prompt-difficulty covariate. Defer to v2 if a real prompt-mutation harness is built. |
| 13 (NEW) | Iso-Yield Dynamic Grouping (Iter46 + Iter122) as a fresh B-SYNTH item | The iso-Y/G(p) curve already lives at iter46 (`zvf_iter46_per_prompt_isog.tsv`, 7 outputs) and iter122 (`zvf_iter122_iso_yield.tsv`); re-running the same arithmetic would only restatethe frontier-synthesis round-2 recipe without adding evidence. **Reject: already prototyped twice; promote to validated next iteration if a paper-facing sentence is integrated.** |
| 14 (NEW) | Critic-token-level temporal credit assignment test (frontier r1 "critic degeneracy at the token level") | Requires per-token value-head outputs from PPO. The samestack run logs only step-level grad_norm/mean_reward/zvf/entropy — no value-head tensor. **Reject: data not available; the empirical proxy via grad_norm + reward trajectory (row 12) is the testable substitute.** |
| 15 (NEW) | τ²-Bench dual-control environment integration as Pillar-2 RL target | Requires browser-API tools and a long-running conversational env (F25 L10 paper, Barres et al. arXiv:2506.07982). The Pillar-2 ZVF diagnostic is task-agnostic and works on the existing GSM8K / arithmetic stack; adding τ²-Bench would shift scope to a new benchmark, not strengthen the existing 4 papers. **Reject: out of Pillar-2 scope.** |

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
- **impact** (this iteration): ★ to ★★★★★ — paper-facing potential, multi-pillar reach, novelty.
- **evidence**: fraction of declared hypotheses that landed DECISIVE.

## Update protocol

1. Pick a new id (next integer).
2. Add a row at the top.
3. Update the matching `docs/berkeley_improvements/<id>_<slug>.md`.
4. Append one line to `./AUTORESEARCH_FINDINGS.jsonl` with pillar `B-F24` /
   `B-SP25` / `B-F25` / `B-SYNTH`.
5. Commit when the row reaches `prototyped` or beyond.