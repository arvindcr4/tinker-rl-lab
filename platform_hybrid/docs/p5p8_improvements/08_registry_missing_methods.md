# P6 #12 — Add missing-method entries: bridging registry ↔ N2 + zvf_iter130 evidence

**Class:** T3 (cross-paper coupling) + T2 (fresh-data evidence).
**Status:** validated (31/31 entries pass `jsonschema`).
**Paper:** `paper/paper_P6_registry.tex` (§ p6_measured_evidence + § p6_population).
**Build:** `paper/build/paper_P6_registry.pdf` rebuilds clean after the § 4 patch.

## Question
Iter 1's coverage audit and iter 2's N2 validation both stopped at the
existing 12 stack + 3 variant-delta seed records. The worktree actually
has measured evidence for **9 distinct GRPO-family methods** (the
zvf_iter130 risk index enumerates 9 by name), but only **4 of those 9**
(grpo, dapo, drgrpo, gspo) had a corresponding registry entry. The
remaining 8 methods either had N2 same-stack per-step tensor evidence
(aero / gift / areal) or zvf-iter130 risk-index evidence
(ngrpo / cppo / mcgrpo / es / scafgrpo). The registry cannot falsify its
own thesis unless it catalogs the methods whose measured behavior is on
file. This iteration closes that gap.

## What we did
`scripts/p5p8/add_missing_entries.py` (≤300 LoC, stdlib + jsonschema)
emits and validates 16 records end-to-end:

| Records | n | Purpose |
|---|---|---|
| New stack entries (N2 same-stack) | 3 (aero, gift, areal) | Isolated variant label on the Tinker-managed Qwen3.5-4B / GSM8K / G=8 / 40-step stack |
| New stack entries (zvf-iter130 batch) | 5 (ngrpo, cppo, mcgrpo, es, scafgrpo) | Per-method risk-index row, n=5 seeds |
| New variant-delta records | 8 (one per method above) | Per-component list derived from a TO_VERIFY source paper; citation marked `UNVERIFIED_<method>` + `TBD_<method>` |

All 31 entries (`12 original + 16 new + 3 original variant-deltas`) parse
under `registry/schema.json` (draft 2020-12) with **0 failures**.

Outputs reproducible: `python3 scripts/p5p8/add_missing_entries.py --write`.

## What the new entries look like

Two provenance shapes (one per source).

### N2 same-stack entries (aero / gift / areal)

```jsonc
// tinker_aero_qwen3.5-4b_gsm8k.json
{
  "label_claimed": "aero",
  "framework": {"name": "tinker", "openness": "managed"},
  "model": "Qwen/Qwen3.5-4B",
  "task": "gsm8k",
  "seeds": [0],
  "provenance": {"source_artifacts": [
      "N2 same-stack four-method run (managed sampler, G=8, 1 seed(s), 40 steps); "
      "isolated variant label, see experiments/results/n2_reward_tensor_resume/aero_s0_tensors.jsonl"
  ]},
  "variant_deltas_applied": [{"delta_id": "delta_aero", "component": "advantage_guided_evolution",
                              "status": "implemented",
                              "note": "isolated via N2 same-stack run; managed-runtime admits only "
                                      "the label flip — the variant-internal machinery is closed"}],
  "outcomes": {"mean_last10_train_reward": 0.7628, "mean_zvf": 0.6422, ...}
}
```

### zvf-iter130 batch entries (ngrpo / cppo / mcgrpo / es / scafgrpo)

```jsonc
// zvf130_ngrpo.json
{
  "label_claimed": "ngrpo",
  "framework": {"name": "worktree-zvf130-batch", "openness": "open"},
  "model": "Qwen/Qwen3-8B-base (shared zvf-iter130 batch harness)",
  "task": "gsm8k (canonical 16-prompt fixed eval, zvf_iter130 spec)",
  "seeds": [0, 1, 2, 3, 4],
  "provenance": {"source_artifacts": [
      "zvf_iter130_method_risk.tsv row method=ngrpo (zvf_risk_mean=0.4467, n_seeds=5); "
      "per-method risk index, single-batch harness"
  ]},
  "variant_deltas_applied": [{"delta_id": "delta_ngrpo", "component": "per_prompt_normalization",
                              "status": "unknown",
                              "note": "single-batch risk-index harness; per-component isolation not run"}]
}
```

## Coverage delta

31 records now in `registry/entries/`:

| Family | Before | After | Δ |
|---|---:|---:|---:|
| Stack entries | 12 | 20 | +8 |
| Variant-delta records | 3 | 11 | +8 |
| **Total** | **15** | **31** | **+16** |
| Methods covered (label set) | 4 (grpo/dapo/drgrpo/gspo + colab grpo-adaptiveg) | **9** (the full zvf_iter130 risk-index set) | +5 |
| Frameworks covered | 5 (trl/verl/openrlhf/tinker/colab-open) | 6 (+ worktree-zvf130-batch) | +1 |

## Per-entry MIN-REPORT badge (the new 8 entries)

| entry_id | leaves / total | badge |
|---|---:|---:|
| tinker_aero_qwen3.5-4b_gsm8k | 15 / 23 | 65.2 |
| tinker_gift_qwen3.5-4b_gsm8k | 15 / 23 | 65.2 |
| tinker_areal_qwen3.5-4b_gsm8k | 15 / 23 | 65.2 |
| zvf130_ngrpo | 9 / 23 | 39.1 |
| zvf130_cppo | 9 / 23 | 39.1 |
| zvf130_mcgrpo | 9 / 23 | 39.1 |
| zvf130_es | 9 / 23 | 39.1 |
| zvf130_scafgrpo | 9 / 23 | 39.1 |

Pattern: the **N2 same-stack entries** (Family D in `p6_population.tex`)
populate **the same 15/23 leaves** the existing Family C entries do — the
managed-runtime loss-form + decontamination gaps are the same, but every
item the manager exposes (sampler / telemetry / group_size) is fully
reported. The **zvf-iter130 entries** (Family E) populate only 9/23, all
in the same columns (sampler_backend, telemetry, heldout_split) — these
methods were never driven through the full MIN-REPORT spec because the
zvf-iter130 batch harness is a single-batch risk-index snapshot, not a
training run.

## What this changes in the paper

- `paper/sections/p6_population.tex` — population table now lists **20 stack
  entries** across 5 framework families (A: 4 framework-dump, B: 4 colab-open,
  C: 4 tinker-managed, D: 3 N2 same-stack, E: 5 zvf-iter130-batch). Schema ✓
  column stays **31/31 = 100%** after the patch.
- `paper/sections/p6_measured_evidence.tex` — extended "measured variant
  delta" table now references the new D-family entries directly; the AE/AL
  length-CV findings from iter 2 are still within paired-bootstrap noise on
  the GRPO baseline at `seed=0`, `G=8`, `bf16`, `T=0.7`.
- `paper/paper_P6_registry.tex` — single § 4 patch (no new section file),
  rebuilds to 18 pages with 0 errors and 0 undefined refs.

Bounded: ≤30 lines added to the registry-paper sections combined, no
schema edits, no claim about effect sizes from the directional zvf-iter130
arms, and no paper-facing text treats the placeholder citations as real —
they are marked `UNVERIFIED_<method>` in the variant-delta `bibkey`.

## Citation honesty

Each new variant-delta record carries an explicit `UNVERIFIED_<method>`
bibkey plus `TBD_<method>` arxiv placeholder until the source paper is
fetched through `mcp__firecrawl__firecrawl_research_search_papers` and a
real BibTeX entry is added. The integrity audit (`jsonschema`) accepts
them because every required field is a non-null string. Once the BibTeX
addition lands, the placeholder names self-flag for a `grep
UNVERIFIED_ registry/entries/` cleanup. This is the policy the worktree
already follows for `delta_dapo / delta_drgrpo / delta_gspo`.

## What we did NOT do (deliberate, scope-protective)

- We did not run Tinker compute. All 16 new entries are derived from
  artifacts that already exist (`n2_metrics.tsv`,
  `zvf_iter130_method_risk.tsv`, `<method>_s0_tensors.jsonl`).
- We did not add the proposed `outcomes.ci_method` from iter 2's ledger
  row #10; that is its own deliverable for a future iteration.
- We did not issue `git push` / `gh PR`; the entries land as a
  local worktree commit candidate only.
- We did not fetch the variant-delta source papers — that step is gated
  on the next arxiv-MCP round; `UNVERIFIED_<method>` markers make the
  unresolved work machine-searchable.

## Reproducibility

```bash
python3 scripts/p5p8/add_missing_entries.py --write
# expected: "wrote 16 entries under registry/entries"
#           "validation: 16/16 PASS"

python3 -c "import json,glob,jsonschema; s=json.load(open('registry/schema.json')); \
  [jsonschema.validate(json.load(open(p)), s) for p in sorted(glob.glob('registry/entries/*.json'))]"
# expected: no output (= all 31 entries parsed)
```

Inputs read: `registry/schema.json`, `experiments/results/n2_reward_tensor_resume/n2_metrics.tsv`,
`experiments/results/zvf_iter130_method_risk.tsv`. Outputs written:
`registry/entries/{tinker_aero_qwen3.5-4b_gsm8k,tinker_gift_qwen3.5-4b_gsm8k,tinker_areal_qwen3.5-4b_gsm8k,zvf130_{ngrpo,cppo,mcgrpo,es,scafgrpo}}.json`,
`registry/entries/delta_{aero,gift,areal,ngrpo,cppo,mcgrpo,es,scafgrpo}.json`,
`experiments/results/p5p8/missing_entry_audit.tsv`,
`experiments/results/p5p8/missing_entry_validation.tsv`.
