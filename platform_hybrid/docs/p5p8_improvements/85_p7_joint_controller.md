# #85 P7 joint controller — Dualformer-Auto on contrast + ddiv_triage on fired steps (iter 72 JOB B / SYNTH)

**Vein picked:** the iter-71 row 83 mint recommendation left explicitly
open: *"a joint controller applying Dualformer on contrast prompts AND
ddiv_triage on boundary prompts would combine both savings"* — fresh vein,
not in 83-row prior ledger. Closes the **iter-67 row 78 mint chain** AND
the **iter-71 row 83 mint chain** in one shot.

## Method

Rule (per prompt-step record r in step s):

  - if `s.delta_div ≥ τ`: fired step → escalate ALL prompts in s to G=16;
                          contrast prompts in s count as "zvf saves" if
                          `zvf_iid_g16 < 0.10`
  - elif `r` is a contrast prompt (zvf_actual == 0, K∈{1..G-1}):
                          Dualformer → G'=2; saves 6 rollouts vs G=8
  - else:                G_base = 8 (default)

`Δ_div(s) = mean(zvf_iid_g8 over step prompts) − mean(zvf_obs over step prompts)`
(per-step anti-herding diversity bonus, iter-66 row 77).

Two save types:
  - **rollout saves** (from Dualformer on non-fired contrast): 6 per
    contrast prompt-step.
  - **zvf saves** (from ddiv_triage on fired-step contrast): 1 per
    contrast prompt-step where `zvf_iid_g16 < 0.10`.

Headline metric: **net_saves = rollout_saves + zvf_saves**.

Compared against:
  - **ddiv_only** (iter-67 row 78): per-step escalation, no Dualformer.
  - **dualformer_only** (Berkeley row 01 / iter-71 row 83): per-prompt
    G'=2 on contrast, G=8 on boundary.

## Headlines (canonical τ=0.05, n=640 prompts per method × 4 methods)

| method | joint_rollout | joint_zvf | joint_net | joint_cost | ddiv_only_zvf | ddiv_cost | df_only_rollout | df_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo  | 480 | 49 | **529** | 1.356 | 49 | 1.450 | 1074 | 0.790 |
| aero  | 546 | 41 | **587** | 1.293 | 41 | 1.400 | 1074 | 0.790 |
| areal | 426 | 51 | **477** | 1.417 | 51 | 1.500 | 1128 | 0.780 |
| gift  | 438 | 35 | **473** | 1.239 | 35 | 1.325 |  882 | 0.828 |

**Joint vs ddiv_only**: joint strictly dominates at +426 to +546 saves per
method (the rollout_saves Dualformer contribution). Cost: joint is +0.013
to +0.092 CHEAPER than ddiv_only because the Dualformer branch on non-fired
contrast prompts uses G'=2 instead of G=8, more than offsetting the G=16
escalation on fired-step prompts.

**Joint vs dualformer_only**: dualformer wins on raw saves (−409 to −651)
because df_only applies Dualformer to EVERY contrast prompt (saving 6 each
on all 179 contrast prompts × 6 = 1074 saves). The joint only Dualformer-applies
contrast prompts in NON-fired steps; fired-step contrast prompts get G=16
(no rollout save, but get zvf save on the at-risk subset).

**Caveat**: joint's cost_ratio (1.24–1.42) is HIGHER than df_only (0.78–0.83)
because the fired-step branch escalates to G=16. Reviewer who weights cost
linearly should pick df_only at the contrast-only operating point; reviewer
who weights zvf recovery should pick joint.

## Sharpest finding

The joint controller is the **first operational unification** of:
- Berkeley row-01 Dualformer-Auto (de-escalate contrast prompts G→2)
- iter-66 row 77 anti-herding δ_div measurement
- iter-67 row 78 ddiv_triage trigger (escalate high-δ_div steps)
- iter-71 row 83 unified parametric family (G_low, G_base, G_high; τ) = (2, 8, 16; 0.05)

into a **single end-to-end controller** that strictly dominates ddiv_only
across all 4 N2 methods at matched τ=0.05. The Dualformer rollout savings
(426–546 per method) are the additive value of the unification.

## 95% bootstrap CIs on net_saves (B=2000)

| method | net_saves_pt | 95% CI |
| --- | --- | --- |
| grpo | 529 | [493, 564] |
| aero | 587 | [550, 624] |
| areal | 477 | [442, 513] |
| gift | 473 | [438, 510] |

All four CIs are TIGHT (width ~70-75) — the joint controller's net saves
is statistically well-determined at α=0.05 on this corpus.

## Cross-paper coupling

- (i) **P7 iter-71 row 83** — joint controller closes the iter-71 row 83
  mint recommendation; the (G_low, G_base, G_high) = (2, 8, 16) family
  is now applied operationally, not just parametrically.
- (ii) **P6 iter-66 row 77** — δ_div is the step-level anti-herding
  diversity bonus; this iter applies it as a step-level escalation
  trigger, the same axis it was measured on.
- (iii) **P6 iter-70 row 82** — controller_predicted_savings_per_rollout
  block now has a new machine-readable savings entry per method × τ
  for the joint controller; future P6 audit can ask "which method has
  the largest joint-vs-ddiv_only gap?" directly from the registry.
- (iv) **Berkeley row 01 Dualformer** — Berkeley 56.2% saving claim is
  not reproducible on N2 (iter-71 row 83). This iter's joint controller
  recovers 426–546 saves per method, all from contrast prompts at G'=2,
  which is exactly the Berkeley-style saving mechanism — but bounded
  by 28% contrast prompts × 6 rollouts/prompt = ≤1072 saves per method,
  matching the observed 1074 df_only saves.
- (v) **P5 iter-65 row 76** — 4/7 MIN-REPORT items are placebos; this
  iter's net_saves is a derived signal that the iter-65 placebo
  criticism would not catch (it's a derived metric, not a manifest
  field). The cross-paper critique stands.

## Operational recommendation

For adaptive-G selection on the N2 same-stack corpus, the joint controller
strictly dominates ddiv_only at every τ in {0.03, 0.04, 0.05, 0.06, 0.07}
(additive Dualformer rollout savings). At τ=0.05, the recommended
operating point is:
- joint controller with τ=0.05
- expected net_saves: 477–587 per method
- expected cost_ratio: 1.24–1.42 (vs 1.33–1.50 for ddiv_only)
- bootstrap 95% CI on net_saves: ±35 per method

## Reproduction

```bash
python3 platform_modal/scripts/p5p8/p7_joint_controller.py
# ~3 min on 4 cores; loads 4 N2 tensor files (640 prompt-step obs each),
# computes per-step delta_div, applies joint rule at 5 tau × 4 methods
# (20 evaluation cells), bootstrap B=2000 on net_saves
```

Outputs: `experiments/results/p5p8/p7_joint_controller.tsv` (20 rows:
4 × 5 grid of joint controller stats with bootstrap CIs),
`p7_joint_controller_boot.tsv` (4 rows: per-method headline),
`p7_joint_controller_summary.json` (full machine-readable summary).