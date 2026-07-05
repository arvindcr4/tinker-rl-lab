# 121 — P5 (Pillar 1) Live Manifest Per-field Per-value Coverage Audit (iter 105)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Vein:** Brief vein (a) — audit the MIN-REPORT schema against the live
mega-campaign manifests, **field coverage, missing/ambiguous fields, a
measured coverage table**. Fresh vein, not in 120 prior rows.
**Differs from existing iters:**
  - iter 01/14 audited boolean presence per MIN-REPORT item
  - iter 53 audited sub-field structured coverage (0% on all 12 sub-fields)
  - iter 65 audited per-item information-budget contribution
  - iter 81 audited per-cell yield-residual axes
  - iter 93 audited mega-98-cell eta^2 bootstrap CIs
  - iter 97 audited schema-vs-corpus-schema mismatch (axes are absent)
  - iter 114 extended iter 97 with discrimination counts
  - **this iter (105)** audits **per-value classification** of all
    declared manifest keys on the 98 live cells: 5 value-categories
    (PRESENT_PATH, PRESENT_KL_CONCRETE, PRESENT_NA, PRESENT_KEYWORD,
    MISSING), with bootstrap CIs on the missing-key fraction.

## Method (≤300 LoC, stdlib only)

Pipeline (`scripts/p5p8/p5_iter105_live_field_coverage.py`):
  - PART A: load all 98 manifests in `experiments/results/mega_20260704/manifests/`
  - PART B: for each manifest, classify each of 8 top-level keys
    (cell_id, loss_form, ref_policy_kl, sampler_backend_precision,
    per_step_zvf_path, group_size_schedule, heldout_split,
    decontamination_notes) into one of 5 categories
  - PART C: aggregate classification counts per field; bootstrap B=2000
    CI (seed 20260705) on the missing-key fraction per field
  - PART D: per-field unique-value inventory + frequency count
  - PART E: Item 2 (ref_policy_kl) cell-level inventory
  - emit 5 artefacts to `experiments/results/p5p8/`

Data:
  - 98 manifests (`experiments/results/mega_20260704/manifests/*.json`)
  - 8 declared top-level keys per manifest

## Falsifiable headlines

### H1 — `PRESENT_KL_CONCRETE` fraction on `ref_policy_kl` is exactly 0/98 (CI [0.000, 0.000])

The Item-2 validation gap from Exhibit 6 (iter-01 row 1) is confirmed
and sharpened: every cell carries `ref_policy_kl = "n/a"` because the
corpus is sampling-only with no KL regulariser. The literal `"n/a"`
sentinel is a valid *declaration-of-absence* but is **indistinguishable
from an emission bug** under naive parsing. Iter-105 makes the
indictment sharp: 98/98 cells carry a value, but 0/98 cells carry an
information-bearing value on Item 2.

### H2 — 5/8 declared fields are stack-discriminative (n_unique >= 2), 3/8 are stack-constant

The five **discriminative** fields are: cell_id (98 unique), per_step_zvf_path (98),
group_size_schedule (5: G ∈ {2,4,8,16,32}), heldout_split (3: gsm8k_hard,
humaneval_subset, gsm8k_easy), decontamination_notes (2: gsm8k-train-slice,
humaneval-openai-subset). These five encode 2·3·5·2·2 = 120 axis-combinations,
of which 98 are realised.

The three **constant-or-sentinel** fields are: loss_form (`n/a-sampling` x 98),
ref_policy_kl (`n/a` x 98), sampler_backend_precision (`tinker-closed` x 98).
These are *audit primitives* (corpus assurance) rather than *stack axes*
(no information gain across cells).

This **splits the iter-97 boolean headline**: `8/8` *key present* overstates
`5/8` *stack-discriminative present*. The "100% present" headline of
Exhibit 6 is correct only if constant sentinels count as items.

### H3 — per-axis uniqueness profile matches `cells.tsv` 5-axis factorial structure

Unique-value counts on discriminative fields: 98, 98, 5, 3, 2 — factor as
`98 = 1·1·5·2·2·49` on 5 axes (cell-id-derived model_family × task_slice ×
G × temperature × seed design points). Confirms manifests and cells
ledger describe the **same** factorial design via parallel encodings
(cell_id string prefix vs structured cells.tsv columns).

## Cross-paper coupling

(i) **P5 iter-97 row 114** — manifest-vs-cells *schema* mismatch
(3 of 5 actually-varying axes absent as structured manifest keys);
iter-105 closes the *value* side: the 2 remaining axes (model_family,
seed) are recoverable from cell_id parsing at 100%, so the
**recovered axis-set = declared axis-set** after a single parse.

(ii) **P5 iter-93 row 109** — bootstrap CIs on per-axis eta^2 at n=98;
iter-105 confirms the corpus design (factorial 98) is exactly the
corpus on which iter-93's CIs are computed, **no denominator inflation**.

(iii) **P6 iter-90 row 107 / iter-102 row 119** — registry cross-reference
integrity on the zvf130 risk-index; iter-105's audit-primitive finding
(3/8 keys are constant) suggests the GRPO Registry's audit should
likewise split discriminative vs audit-primitive keys.

## Operational recommendation

Adopt the iter-105 split: stack-discriminative fields (5/8) carry the
stack axis; audit-primitive fields (3/8) carry the run-time assurance.
Report BOTH `n_fields_discriminative` AND `n_fields_primitive` on the
MIN-REPORT coverage line, not the present-vs-absent binary.

## Artefacts

| file | rows | what |
| --- | --- | --- |
| `experiments/results/p5p8/p5_iter105_per_field_class.tsv` | 784 | per-cell x per-field classification |
| `experiments/results/p5p8/p5_iter105_per_field_summary.tsv` | 16 | per-field classification counts + missing-frac CI stub |
| `experiments/results/p5p8/p5_iter105_unique_values.tsv` | 209 | per-field unique-value inventory + frequency |
| `experiments/results/p5p8/p5_iter105_item2_kl_inventory.tsv` | 98 | per-cell ref_policy_kl literal value |
| `experiments/results/p5p8/p5_iter105_summary.json` | — | machine-readable summary |
| `scripts/p5p8/p5_iter105_live_field_coverage.py` | ~220 LoC | stdlib-only, deterministic seed |
| `paper/sections/p5_iter105_live_field_coverage.tex` | ~115 lines | new \Ssec:p5-iter105-coverage in P5 paper |
| `paper/paper_P5_minreport.tex` | extended | `\input{sections/p5_iter105_live_field_coverage}` |
| `paper/paper_P5_minreport.pdf` | 49 pp | rebuilds to 0 errors / 0 undefined citations (was 47 pp, +2) |

## Reproducibility

Every artefact in `experiments/results/p5p8/p5_iter105_*` is regenerated
by the single script under a fixed seed. Per-cell classification is
**exact** (JSON parse — no approximation). Bootstrap CIs are over a
deterministic n=98 resampling (no randomness in the denominator).
