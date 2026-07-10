# paper/sections/ — INDEX

**Purpose:** Per-section and appendix `.tex` inputs pulled into `../main.tex` via `\input{sections/...}`. Files ending `_anon` are the anonymized twins for `../main_anon.tex`. Most content lives here, not inline in main.tex.

**Key files (body):**
- `abstract.tex`, `intro.tex`, `related_work_v2.tex`, `conclusion.tex` — core body (each has `_anon` twin)
- `checklist.tex` (+`_anon`) — NeurIPS reproducibility checklist

**Key files (appendices / reviewer addenda):**
- `appendix_zvf_formalization.tex`, `zvf_counterfactual_appendix.tex`, `zvf_pipeline_spec.tex` — ZVF (Zero-Variance-Fraction) formalization + pipeline (answers W1)
- `group_size_reconcile.tex` — reconciles G-size contradiction (W2)
- `framework_configs_appendix.tex` — cross-framework hyperparameter configs
- `stat_rigor_updates.tex` (+`_anon`), `statistical_rigor_addendum.tex` — statistical-rigor / survival analysis
- `tool_use_code_expanded.tex` — expanded tool-use & code-gen results
- `heldout_stratified.tex`, `base_vs_instruct_paired.tex` — held-out + base-vs-instruct evaluation
- `frontier_scope_clarification.tex`, `extended_related_work.tex`, `variance_mitigation_comparison.tex` — scope + extended RW + variance methods
- `figures_regeneration_note.tex` — note on deterministic figure regeneration

**Find it fast:**
- to edit a section → grep `\input{sections/<name>}` in `../main.tex` for placement/order
- appendix load order is main.tex lines ~2258–2270
- for the anon build, edit the matching `*_anon.tex` too
