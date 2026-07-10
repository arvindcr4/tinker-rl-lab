# P6 iter-102 — Registry cross-reference integrity + zvf130 stack coverage gap-fill + sig-robustness

**Pillar:** P6 (Pillar 2 — GRPO-Registry)
**Target class:** T1 (statistical rigor) + T3 (cross-paper coupling / internal consistency)
**Vein:** brief veins (a) validate entries against measured behavior, (b) coverage
audit, (c) schema/CI validation, (d) add missing entries — hybridised into one
CI-style integrity guard for the zvf_iter130 risk-index panel.

## Problem

The zvf_iter130 risk-index (`experiments/results/zvf_iter130_method_risk.tsv`) is the
ground truth for the **9 real 5-seed GRPO-family methods**
(`grpo, ngrpo, aero, cppo, mcgrpo, areal, gift, es, scafgrpo`; the `scaling_law_*`
and `tool_use_*` rows are n_seeds=1 placeholders, excluded). The registry stored each
method's risk delta in **two independent representations** with no cross-check:

- **(A)** `zvf130_<m>.json` stack entry → `outcomes.{zvf_risk_mean, delta_vs_grpo_*}`
- **(B)** `delta_<m>.json` variant entry → `measured[]` block (panel `zvf130_5seed`)

Nothing verified (A) against the TSV, (B) against the TSV, or (A)==(B). Coverage was
also incomplete: only **5/9** methods had a stack entry — and the missing four included
`grpo` itself, the reference point of every `delta_vs_grpo`.

## What was built

1. `scripts/p5p8/p6_registry_crossref_integrity.py` — CI-style guard. For every real
   method it (i) checks the stored `zvf_risk_mean` and `delta_vs_grpo_mean` against the
   TSV, (ii) recomputes the delta-vs-grpo with a conservative **Welch two-sample t**,
   (iii) cross-checks (A)==(B), (iv) reports stack/delta coverage, and (v) emits a
   sig-robustness table. Exits nonzero only on a hard point-estimate mismatch → drop-in CI.
2. `scripts/p5p8/p6_fill_zvf130_stack_gap.py` — creates the 4 missing stack entries
   (`zvf130_{grpo,aero,areal,gift}.json`, all provenance-tagged to the TSV) and stamps a
   `sig_robust_bootstrap_and_welch` field (+ Welch CI) into every zvf130-derived block
   (5 prior stacks, 8 delta entries), so the optimistic bootstrap `significant` flag
   never stands alone.
3. `registry/schema.json` — `measured_delta` extended with `welch_ci_low/high`,
   `welch_sig`, `sig_robust_bootstrap_and_welch`, `sig_robust_note`. **39/39 entries
   validate** under `jsonschema` (draft 2020-12).

## Results (all measured this iteration)

- **Integrity: 40 PASS / 0 hard FAIL.** Every stored point estimate matches the TSV
  ground truth to ≤5e-4. The dual representation (A)==(B) is consistent for all 5
  overlapping methods.
- **Coverage: stack 5/9 → 9/9**, delta 8/8. The base `grpo` is now catalogued as a stack.
- **Sig-robustness (the sharp finding):** the risk-reduction-vs-grpo significance is
  **CI-method-dependent** for the two smallest/highest-variance deltas. Under the iter90
  paired Gaussian-residual bootstrap all 8 methods are "significant", but under a
  conservative Welch two-sample t only **6/8 survive** (`areal, cppo, es, gift, mcgrpo,
  scafgrpo`). **`aero` and `ngrpo` flip to NS** (aero Welch CI [-0.311, +0.015];
  ngrpo [-0.273, +0.011] — both straddle 0). These are exactly the closest-to-grpo
  methods, where a n=5 unpaired test has no power.

| method | Δ vs grpo | Welch 95% CI | Welch sig | boot sig | **sig_robust** |
|---|---|---|---|---|---|
| scafgrpo | -0.352 | [-0.496, -0.209] | ✓ | ✓ | **yes** |
| es | -0.273 | [-0.417, -0.128] | ✓ | ✓ | **yes** |
| gift | -0.263 | [-0.407, -0.119] | ✓ | ✓ | **yes** |
| areal | -0.246 | [-0.388, -0.104] | ✓ | ✓ | **yes** |
| mcgrpo | -0.174 | [-0.317, -0.031] | ✓ | ✓ | **yes** |
| cppo | -0.151 | [-0.295, -0.007] | ✓ | ✓ | **yes** |
| aero | -0.148 | [-0.311, +0.015] | ✗ | ✓ | **no (flip)** |
| ngrpo | -0.131 | [-0.273, +0.011] | ✗ | ✓ | **no (flip)** |

## Interpretation

The registry's job is to make label-underdetermined claims machine-checkable. This
iteration turns "the entries look right" into "every number is regression-guarded against
its ground truth, and every significance flag carries a robustness verdict." The
practical upshot: a reviewer who prefers a conservative two-sample test should read
`aero` and `ngrpo`'s risk reductions as **directional but not established** at n=5 —
the registry now says so in-band (`sig_robust_bootstrap_and_welch=false`).

## Outputs
- `experiments/results/p5p8/p6_crossref_integrity.tsv` (per-check)
- `experiments/results/p5p8/p6_crossref_summary.json`
- `experiments/results/p5p8/p6_sig_robustness.tsv` / `.json`
- 4 new + 13 stamped registry entries; schema patch.

## Reproduce
```
python3 scripts/p5p8/p6_fill_zvf130_stack_gap.py   # idempotent
python3 scripts/p5p8/p6_registry_crossref_integrity.py   # exit 0 = clean
```
