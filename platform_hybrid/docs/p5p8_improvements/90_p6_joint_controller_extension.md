# #90 P5P8 SYNTH / P6 JOB B: joint-controller cost-ratio registry extension (iter 76)

**Vein picked:** fresh, closes the iter-72 row 85 mint recommendation:
*"cost-adjusted joint controller incorporating iter-72 cost_ratio into
iter-70 row 82 controller_predicted_savings_per_rollout block —
extends the registry-readable joint controller savings to include
cost_ratio per method × τ"*.

The iter-70 row 82 block lifted the iter-67 row 78 `δ_div`-triage
paired-counterfactual into a per-method × per-τ registry entry. The
iter-72 row 85 joint controller extended the family to a 2-branch
{dualformer rollout + δ_div zvf} controller and reported
`net_saves` and `cost_ratio` for 4 methods × 5 τ values. That
information was **NOT** yet in the registry — only on disk. This iter
lifts the iter-72 row 85 outputs into the registry, with a schema
extension for joint-controller-specific fields.

## Method

### Inputs
- `experiments/results/p5p8/p7_joint_controller.tsv` (20 rows: 4 methods × 5 τ)
- `experiments/results/p5p8/p7_joint_controller_boot.tsv` (4 bootstrap rows)
- `registry/entries/tinker_{grpo,aero,areal,gift}_qwen3.5-4b_gsm8k.json` (4 stack entries)
- `registry/entries/delta_{aero,areal,gift}.json` (3 delta entries)
- `registry/schema.json` (was 34/34 PASS, kept)

### Schema extension (additive, optional, nullable)

Two additive-optional sub-blocks are added — neither breaks the
existing fields:

1. `controller_predicted_savings_per_rollout.joint_controller` (in
   `variant_delta_record`): contains the per-method joint predictions;
   nullable.
2. `outcomes.joint_controller_predictions` (in `stack_record`): the
   mirror block, with `G`, `n_steps`, and the per-method prediction
   list.

Prediction items get 6 new nullable integer fields:
`net_saves`, `n_contrast_prompts`, `n_fired_steps`, `n_zvf_saved`,
`n_rollout_saves`, `g_total`; and `source_iter` (nullable string).
All `additionalProperties` constraints are preserved by extending
the items schema explicitly.

### Validation gate
- **Schema validation**: 34/34 entries PASS after the patch (`ok=34, fail=0`).
- **Sharpest registry-readable fact**: at τ=0.05, the
  AERO joint controller has the **highest** `savings_per_rollout_pt`
  in the entire registry — `net_saves=587 / g_total=6622 × 1000 ≈ 88.6`
  per 1000 rollouts — at `cost_ratio_pt = 1.293`. This is now
  machine-readable from the registry without re-running iter-72.

## Headline findings (lifted from iter-72 row 85)

| Method | τ=0.03 | τ=0.04 | τ=0.05 | τ=0.06 | τ=0.07 |
| --- | --- | --- | --- | --- | --- |
| grpo: net_saves / cost_ratio | 223/1.77 | 343/1.62 | **529/1.36** | 713/1.14 | 731/1.11 |
| aero: net_saves / cost_ratio | 325/1.65 | 409/1.53 | **587/1.29** | 752/1.13 | 770/1.10 |
| areal: net_saves / cost_ratio | 187/1.95 | 308/1.71 | **477/1.42** | 638/1.20 | 656/1.18 |
| gift: net_saves / cost_ratio | 191/1.92 | 296/1.74 | **473/1.41** | 643/1.20 | 657/1.18 |

At τ=0.05 (the iter-72 row 85 canonical operating point):

- **GRPO** joint controller: 529 net saves (rollout=480 + zvf=49) at cost_ratio=1.356
- **AERO** joint controller: 587 net saves (rollout=546 + zvf=41) at cost_ratio=1.293
- **AREAL** joint controller: 477 net saves (rollout=426 + zvf=51) at cost_ratio=1.422
- **GIFT** joint controller: 473 net saves (rollout=438 + zvf=35) at cost_ratio=1.412

**Sharpest observation**: AERO has the **highest net_saves per
cost-ratio unit** at τ=0.05 — 587/1.293 = 454 net saves per unit
cost, exceeding GRPO (529/1.356 = 390), AREAL (477/1.422 = 335), and
GIFT (473/1.412 = 335). This was already in iter-72 row 85 but is
now machine-readable in the registry, with provenance linked to
`p7_joint_controller.py`.

## Cross-paper coupling

1. **Iter-70 row 82 → iter-72 row 85 → iter-76 (this iter)**: the
   three-block family now forms a complete controller → measurement
   → audit pipeline. The registry's `controller_predicted_savings_per_rollout`
   block carries (a) iter-67 row 78 `δ_div-triage` (single-controller,
   iter-70), (b) iter-72 row 85 joint-controller predictions
   (`net_saves` + `cost_ratio`, this iter), with provenance to
   `p7_joint_controller.py`.
2. **P7 ↔ P8 (JOB A)**: the iter-89 row 89 cost-per-decision story is
   the per-deployment analog of the joint-controller's
   per-(method, τ) cost_ratio. Selective-LLM (w=0.1) at $0.000102
   per decision is the deployment-level cost floor; the joint
   controller's cost_ratio at τ=0.05 averages 1.39 — the
   registry-readable cost.
3. **P5 ↔ P6**: the v2.0 stack-axis extension (iter-73 row 86) added
   `model_family` and `task_slice` to the manifest. This iter's
   `joint_controller_predictions` block surfaces a per-method × per-τ
   prediction that can be tagged with the `model_family` axis.

## Reproducibility

- `scripts/p5p8/p6_controller_joint_extension.py` (~210 LoC, stdlib only)
- `experiments/results/p5p8/p6_joint_controller_extension.tsv` (20 rows)
- `experiments/results/p5p8/p6_joint_controller_extension.json`
- Patched `registry/schema.json` (+2 nullable sub-blocks, +6 nullable
  item fields, `additionalProperties: false` preserved)
- Patched 4 `registry/entries/tinker_*_qwen3.5-4b_gsm8k.json` (added
  `outcomes.joint_controller_predictions`)
- Patched 3 `registry/entries/delta_{aero,areal,gift}.json` (added
  `controller_predicted_savings_per_rollout.joint_controller`)

Schema validation `34/34 PASS` before and after; net registry
readable entries: 7 patched (4 stack + 3 delta), 27 unchanged.
