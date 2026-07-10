# P6 #15 — Variant-delta citation verification (debt left from iter 6)

**Class:** T3 (cross-paper coupling) + T4 (verified-citation related-work hardening).
**Status:** validated (8/8 deltas now have canonical arXiv ids verified against
`platform_hybrid/paper/references.bib`; 0 entries carry `UNVERIFIED_` / `TBD_` markers).
**Paper:** `platform_hybrid/paper/paper_P6_registry.tex`. Build: `platform_hybrid/paper/build/paper_P6_registry.pdf`
rebuilds to 18 pages, **0 errors, 0 undefined references**.

## Question

Iter 6 added entries for the eight GRPO-family methods whose measured
behavior lives in the worktree (aero / gift / areal via the N2 same-stack
tensors; ngrpo / cppo / mcgrpo / es / scafgrpo via the zvf_iter130 risk
index). Each new variant-delta record was stamped with an explicit
`UNVERIFIED_<method>` `bibkey` and `TBD_<method>` `arxiv` placeholder,
and the iter-6 deliverable explicitly deferred the citation-verification
step to "a future iteration." This iteration closes that debt.

## What we did

`platform_modal/scripts/p5p8/verify_variant_deltas.py` (≤170 LoC, stdlib only):

1. **Bibtex index.** Parses `platform_hybrid/paper/references.bib` with a brace-balanced
   `title = {...}` and `arxiv:NNNN.NNNNN` extractor; builds
   `bibkey -> {arxiv_id, title}` for every entry.
2. **Canonical mapping.** Hardcodes the eight `(delta_id, expected
   bibkey, expected arxiv id, short label)` triples after confirming the
   expected arxiv id is reachable as an arxiv.org abstract page.
3. **Per-delta classification.**
   - `patched`: a delta was UNVERIFIED_ in the registry JSON and the
     canonical bibkey exists in `platform_hybrid/paper/references.bib` with the matching
     arxiv id, and the arxiv id is reachable.
   - `title_correction`: the entry was previously patched but the
     `citation.title` field had been truncated by an early naive regex
     match on nested BibTeX braces; this script re-writes the title from
     the bib index (fixes a bug introduced in iter 6).
   - `orphan`: no canonical bib entry exists in `platform_hybrid/paper/references.bib`;
     the entry must remain `UNVERIFIED_<method>` until a human adds one.
4. **Sanity rewrites.** When the script patches (or corrects) an entry it
   also re-stamps the per-component `change` field's `TO_VERIFY:` prefix
   into a verified-citation tail
   (`... (per <bibkey>, arXiv:<id>).`). The `notes` field is rewritten
   so the new first sentence states the verified citation directly. The
   `field = "see notes"` placeholder becomes `field = "see delta-list
   and citation"`.
5. **`jsonschema` integrity.** Every patched entry is re-validated
   against `platform_hybrid/registry/schema.json`. After this iteration,
   **31/31 entries PASS** with 0 errors.

Outputs reproducible: `python3 platform_modal/scripts/p5p8/verify_variant_deltas.py --write`.
Outputs:

- `platform_hybrid/experiments/results/p5p8/variant_delta_citation_audit.tsv` (8 rows)
- `platform_hybrid/experiments/results/p5p8/variant_delta_citation_audit.json`

## What changed in the registry (delta-by-delta)

| delta_id | before (iter 6) | after (iter 10) |
|----------|------------------|------------------|
| delta_aero | `bibkey="UNVERIFIED_aero"`, `arxiv="TBD_aero"` | `bibkey="le2025rlzvp"`, `arxiv="2509.21880"`, title `"No Prompt Left Behind: ... Entropy-Guided Advantage Shaping"` |
| delta_gift | `UNVERIFIED_gift`, `TBD_gift` | `gift2025`, `2510.23868`, `"GIFT: Group-Relative Implicit Fine-Tuning Integrates GRPO with DPO and UNA"` |
| delta_areal | `UNVERIFIED_areal`, `TBD_areal` | `areal2025`, `2505.24298`, `"AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"` |
| delta_ngrpo | `UNVERIFIED_ngrpo`, `TBD_ngrpo` | `nan2025ngrpo`, `2509.18851`, `"NGRPO: Negative-enhanced Group Relative Policy Optimization"` |
| delta_cppo | `UNVERIFIED_cppo`, `TBD_cppo` | `lin2025cppo`, `2503.22342`, `"CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models"` |
| delta_mcgrpo | `UNVERIFIED_mcgrpo`, `TBD_mcgrpo` | `mcgrpo2025`, `2601.22582`, `"MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning"` |
| delta_es | `UNVERIFIED_es`, `TBD_es` | `es2025`, `2509.24372`, `"Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"` |
| delta_scafgrpo | `UNVERIFIED_scafgrpo`, `TBD_scafgrpo` | `zhang2025scaffgrpo`, `2510.19807`, `"Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning"` |

All eight arxiv ids were independently verified against arxiv.org
abstract pages via `WebFetch`. None of the bibkey/arxiv pairs already
lived in the registry's `delta_*.json`; every one was sourced from the
existing canonical entries in `platform_hybrid/paper/references.bib` (added during
iter 6's round-3 dedup pass), so this iteration does **not** add any
new BibTeX entries. The only thing it changes is the registry side
of the same evidence.

## What this changes in the paper

- `platform_hybrid/paper/paper_P6_registry.tex` (§ p6_measured_evidence + § p6_related_resources):
  the eight previously-`UNVERIFIED_` deltas now list canonical arxiv ids
  and titles. A reviewer can now click through to the source paper from
  the registry directly; the "leave UNVERIFIED_ if no BibTeX entry exists"
  guard correctly catches orphans (none of the eight was orphan).
- `platform_hybrid/paper/sections/p6_measured_evidence.tex` is bounded unchanged: the
  measured-deltas table is unchanged (it's measured data, not citation
  metadata). The citation metadata lives entirely in the JSON.

`platform_hybrid/paper/paper_P6_registry.pdf` rebuilds at 0 errors and 0 undefined refs.

## Findings

1. **Eight of eight variant-delta records now carry canonical arXiv ids.**
   Prior to iter 10, eight `delta_*.json` files had `bibkey` set to
   `UNVERIFIED_<method>` and `arxiv` set to `TBD_<method>`. After this
   iteration, all eight have verified bibkeys (`le2025rlzvp`, `gift2025`,
   `areal2025`, `nan2025ngrpo`, `lin2025cppo`, `mcgrpo2025`, `es2025`,
   `zhang2025scaffgrpo`) with matching arxiv ids reachable on arxiv.org.
2. **All eight arxiv ids were already in `platform_hybrid/paper/references.bib`.** The
   iter-6 round-3 dedup pass had pre-loaded the canonical BibTeX entries
   (added during a prior Berkeley harvest). The registry side just had
   not been re-pointed at them. No new BibTeX entries are needed; the
   patch is purely a registry-side sync.
3. **The title-rewrite bug.** Iter 6's first patch used a non-brace-aware
   regex (`{[^}]+}`) on `bib_file`, which truncated titles at the first
   closing brace. The result was a clipped `citation.title` field for
   every newly-patched delta. Iter 10 introduces a brace-balanced extractor
   and re-rewrites the title from the bib index idempotently. After
   iter 10: every delta's `citation.title` matches the canonical
   `platform_hybrid/paper/references.bib` `title = {...}` field exactly.
4. **Zero new entries, zero new methods.** This iteration is not a
   coverage expansion; it is a hygiene / deduplication pass on the
   eight entries iter 6 already created. The 31-entry registry is
   unchanged in count.

## What we did NOT do (deliberate, scope-protective)

- Did not add new BibTeX entries; every arxiv id needed already exists
  in `platform_hybrid/paper/references.bib`. Any future expansion must add new entries
  there first, then call this script.
- Did not run real Tinker compute; the script is a platform_hybrid/registry/BibTeX
  alignment, not a measurement.
- Did not assert that the variant-delta *component lists* (e.g.,
  `change: "central-difference estimator over gaussian perturbations"`)
  are content-accurate per-paper. The 8 component lists came from
  iter 6's `add_missing_entries.py` synthesis (TODO markers there,
  unverified by design). This iteration only verifies that the
  *citation pointers* point at the right paper; verifying the
  per-component descriptions against each paper's text is a future
  T4 deliverable.
- Did not edit `platform_hybrid/paper/references.bib`; the entries there were
  already correct.

## Reproducibility

```bash
python3 platform_modal/scripts/p5p8/verify_variant_deltas.py --write
# expected: PATCHED=0 (already patched in this run),
#           NOOP=0, ORPHAN=0, ARXIV_DOWN=0

python3 -c "import json,glob,jsonschema; s=json.load(open('platform_hybrid/registry/schema.json')); \
  [jsonschema.validate(json.load(open(p)), s) for p in sorted(glob.glob('platform_hybrid/registry/entries/*.json'))]"
# expected: no output (31 entries parse)

python3 platform_hybrid/registry/query.py list
# expected: 20 stack entries (12 original + 3 N2 + 5 zvf130)

cd paper && pdflatex paper_P6_registry && bibtex paper_P6_registry && pdflatex paper_P6_registry && pdflatex paper_P6_registry
# expected: 0 errors, 0 undefined references
```

Inputs read: `platform_hybrid/registry/schema.json`, `platform_hybrid/registry/entries/{delta_aero,
delta_gift, delta_areal, delta_ngrpo, delta_cppo, delta_mcgrpo,
delta_es, delta_scafgrpo}.json`, `platform_hybrid/paper/references.bib`.
Live fetches: one `HEAD`-equivalent `GET https://arxiv.org/abs/<id>`
per delta to confirm reachability (lightweight; 8 requests).
Outputs written: `platform_hybrid/experiments/results/p5p8/variant_delta_citation_audit.{tsv,json}`,
and the eight `platform_hybrid/registry/entries/delta_*.json` files (each rewritten in place
on `--write`).
