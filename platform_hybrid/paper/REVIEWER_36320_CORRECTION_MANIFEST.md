# Reviewer #36320 correction manifest

This is a correction manifest for local manuscript descendants. `paper.tex` is
a current descendant matching the reviewed topic, **not** the authenticated,
byte-identical reviewed PDF. Local edits do not alter the reviewed record.

## Live roster (P1--P12)

| ID | Root | Scoped correction |
|---|---|---|
| P1 | `platform_hybrid/paper/paper_P1_scaling.tex` | No pooled checkpoints, scaling replications, or capacity ceiling. |
| P2 | `platform_hybrid/paper/paper_P2_zvf.tex` | ZVF/GU is descriptive; centered contrast is not total gradient. |
| P3 | `platform_hybrid/paper/paper_P3_group_size.tex` | Budget-specific measurements; no universal group-size recommendation. |
| P4 | `platform_hybrid/paper/paper_P4_length_bias.tex` | Bounded null only. |
| P5 | `platform_hybrid/paper/paper_P5_minreport.tex` | Require provenance, estimands, and missing-cell reporting. |
| P6 | `platform_hybrid/paper/paper_P6_registry.tex` | Registry conflicts are quarantines, not reconciliations. |
| P7 | `platform_hybrid/paper/paper_P7_zvf_controller.tex` | Controller remains prospective, not a demonstrated benefit. |
| P8 | `platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex` | Case-specific artifact evidence, no ranking. |
| P9 | `platform_hybrid/paper/neurips_2026_variants/paper_P9_dnb.tex` | Tiered cells and quarantined provenance. |
| P10 | `zvf-program/theory/paper_P10_zvf_theory.tex` | Theory is scoped to centered reward contrast. |
| P11 | `zvf-program/audit/paper_P11_reproducibility_audit.tex` | Later single-stack audit is future-paper evidence. |
| P12 | `platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex` | Routing proposal needs prospective evaluation. |

## Historical audit-only roots (six)

`R01_acm/acm_main.tex`, `R02_main_zvf/main_zvf.tex`,
`R06_min_report/min_report_rl.tex`, `R07_grpo_registry/grpo_registry.tex`,
`U01_main_compendium/main.tex`, and `P08_fraud/paper_P8_fraud.tex` are
absorbed history, not live manuscripts. Each carries an August 2026
claim-consistency audit note; they are not modernized or compiled by this task.

## Reviewed-record corrections

- The submitted runner is group-standardized and is not canonical GRPO. Equal
  within-group rewards zero its centered reward-contrast term only; KL,
  auxiliary losses, clipping/ratios, and completion masking can affect total
  gradients and transfer.
- Submitted synthetic tool-use cells: reward 0, ZVF 1, GU 0, held-out **not
  evaluated**. GSM8K ZVF without a named per-run source is only heterogeneous
  descriptive context. Later per-run files are future-paper evidence.
- The selected ten-checkpoint aggregate and its capacity-ceiling reading are
  withdrawn. Individual historical rows are selection analysis, never pooled.
- The Qwen PPO row is a provenance conflict: 22.5% and 35.0% came from different
  runs and remain quarantined. The later audit also retains unresolved
  exact-ID/model conflicts. No PPO-versus-GRPO direction is claimed.
- High-reward/high-ZVF is only a binary-reward theoretical solved-task regime
  unless a named pre-submission cell supports it. Unsigned ZVF cannot select a
  retire/distill versus retry/resample/warm-start action.
- External deployment benefit and use-inspired claims are withdrawn. A future
  use is only a prospectively evaluated operator decision policy or
  algorithm-adoption audit.

## Mechanical verification

`scripts/reviewer_36320_corpus_check.py` enumerates the canonical 12 live
roots, the six audit-only roots, and `paper.tex`, then verifies these scope and
provenance markers without compiling archives.
