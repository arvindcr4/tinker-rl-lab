# P6 #04 — Registry entries validated against N2 same-stack measured deltas

**Class:** T2 (fresh-data evidence) + T3 (cross-paper coupling).
**Status:** validated.
**Paper:** `paper/paper_P6_registry.tex` (§ p6_population, new § p6_measured_evidence).
**Build:** `paper/build/paper_P6_registry.pdf` rebuilds with 0 errors after the patch.

## Question
The seed entries in `registry/entries/*.json` claim certain label-level outcomes
(e.g., the managed-runtime "DAPO" arm at mean ZVF 0.578 versus the open-trainer
"DAPO" arm at 0.00). Do these match what the four-method N2 same-stack run
(GRPO / AERO / GIFT / AREAL, G=8, seed=0, 40 steps) actually logged when the
*deltas* themselves are isolated from the stack?

## What we did
`scripts/p5p8/registry_validate.py` (≤300 LoC, stdlib + jsonschema) runs three
jobs end to end:

1. **Schema validation + MIN-REPORT leaf coverage table.**
   Every entry in `registry/entries/*.json` is parsed against
   `registry/schema.json` (draft 2020-12). Result: **15/15 entries PASS**
   (12 `stack`, 3 `variant_delta`). Per-item coverage is reported as
   `non-null leaves / total leaves`. Outputs:
   - `experiments/results/p5p8/registry_schema_check.tsv`
2. **Measured variant deltas from N2 reward tensors** (last-10 steps, paired
   bootstrap CI, deterministic LCG with `seed=0`, `n_boot=2000`). The four
   methods share a stack (Tinker-managed sampler, G=8, seed=0) so the deltas
   isolate the variant label, not the runtime. Outputs:
   - `experiments/results/p5p8/registry_measured_deltas.tsv`
   - `experiments/results/p5p8/registry_measured_deltas.json`
3. **Per-prompt pooled reward-mean deltas** (each prompt's mean over all
   steps it's seen, paired bootstrap). This controls for prompt-set drift
   and answers the cleaner counterfactual question.

Outputs reproducible: `python3 scripts/p5p8/registry_validate.py --write`.

## Key findings (real, paired bootstrap, n=2000)

### Same-stack deltas (GRPO baseline vs each variant, last 10 steps, seed 0)

| metric        | variant |  Δ (paired) | 95% CI             | significant? |
|---------------|---------|-------------|-------------------|--------------|
| reward_mean   | aero    | +0.0141     | [+0.0000, +0.0391]| no           |
| reward_mean   | gift    | −0.0164     | [−0.0625, +0.0469]| no           |
| reward_mean   | areal   | +0.0195     | [−0.0078, +0.0547]| no           |
| zvf           | aero    | +0.0250     | [−0.0625, +0.1250]| no           |
| zvf           | gift    | −0.1250     | [−0.2500, +0.0000]| no (boundary)|
| zvf           | areal   | +0.0563     | [−0.1250, +0.1875]| no           |
| loss          | gift    | +16722.41   | [+14911.78, +19371.83]| **yes**   |
| mean_len      | aero    | −31.08      | [−41.12, −23.43]  | **yes**      |
| mean_len      | areal   | −30.13      | [−41.10, −16.25]  | **yes**      |
| cv_len        | aero    | +0.034      | [+0.014, +0.069]  | **yes**      |
| cv_len        | areal   | +0.038      | [+0.019, +0.062]  | **yes**      |

**Per-prompt pooled (n=16 prompts):** every delta is non-significant
(GRPO−aero Δ=+0.007; GRPO−gift Δ=−0.011; GRPO−areal Δ=+0.006).

### Interpretation for the paper

- **Reward / ZVF deltas are all within paired bootstrap noise at the same
  stack.** This is the "no algorithm effect once stack is fixed" finding
  predicted by the Pillar-1 framing and earlier `framework_comparison.json`.
  The N2 four-method run is the cleanest demonstration of it yet: same
  sampler, same $G{=}8$, same seed, same prompt pool, same number of steps.
- **Length and length-CV deltas are real.** AERO and AREAL produce
  ~30-token shorter mean completions on the last 10 steps with significantly
  higher length-CV. This is *not* an algorithm advantage — it is a different
  prompt–completion distribution under the same reward, consistent with the
  sequence-level importance ratio (GSPO family) collapsing long
  completions.
- **GIFT's loss has a +16,722 absolute shift (baseline ≈ −17,000)**. This
  indicates a different objective additive constant — the gamma-style
  likelihood baseline the GIFT family is named for. It is *not* a reward
  improvement; reward_mean is in fact slightly worse for GIFT.

### Coverage audit (all 12 stack entries; 15 total records)

| family | n | badge range | dominant gap |
|--------|---|-------------|--------------|
| A: framework dumps (Qwen3-8B) | 4 | 60–74 | `loss_form` (0–1/6 leaves) |
| B: open Colab trainer (E3) | 4 | 96 | none — every field reported |
| C: managed-runtime (Qwen3.5-4B) | 4 | 57–62 | `loss_form` (0–2/6) + `decontamination` (0/2) |

The two `decontamination` gaps on every Family-A/C entry are the canonical
"Item 7 honesty marker" — see iter 1's `minreport_field_coverage.md`. The
new N2 row exposes *one* additional gap: no entry records the `n_boot`
or `paired-bootstrap` choice used to derive its `outcomes.mean_last10_*`
fields; we recommend adding `outcomes.ci_method` to the schema in a
follow-up.

## What this changes in the paper

- `paper/sections/p6_population.tex` — the population table now reports a
  `Schema ✓` column derived from the validation run, plus a row-level
  footnote on `Family C`'s coverage gap (`loss_form` 0–2/6).
- `paper/sections/p6_measured_evidence.tex` (NEW) — a one-table section
  that ties the registry's *claimed* deltas (per `outcomes`) to the
  N2 measured deltas. This is the section that makes the registry
  auditable against measured data, not just against manifests.

Both are bounded: ≤60 lines of new LaTeX, no new BibTeX entries,
0 schema edits, 0 entry edits, no claim about effect sizes from the
directional Colab arms. The PDF rebuilds cleanly.

## Reproducibility

```bash
python3 scripts/p5p8/registry_validate.py --write
# expected: "15/15 pass" on schema; reward_mean deltas all non-significant
# under paired bootstrap; length-CV deltas for AERO/AREAL significant
```

Inputs read: `registry/schema.json`, `registry/entries/*.json`,
`experiments/results/n2_reward_tensor_resume/n2_metrics.tsv`,
`experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl`.

## What we did NOT do (deliberate, scope-protective)

- We did not change the registry schema or any entry. Adding
  `outcomes.ci_method` would be the natural next iteration.
- We did not run a heldout / train split on the N2 tensors; the run is a
  single seed (`s0`), 40 steps, 16 prompts. Larger-N replication is the
  responsibility of the N10 seed-expansion panel (`n10_seed_expansion/`).
- We did not invoke real Tinker compute. The N2 tensors already exist
  and were not re-run.