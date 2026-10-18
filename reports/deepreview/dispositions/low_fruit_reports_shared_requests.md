# Reports-cluster shared-source requests (18 October 2026)

No bibliography, shared methods section or TikZ source was edited. These are
requests, not completed fixes. Preserve current numeric results and verify
the central source before applying. Local conference dependencies were copied
read-only from the original checkout because these ignored output directories
were absent in the worktree; only the local body/root sources were edited.

## Verified exact replacements

- Findings **725, 875**, `refs.bib` in the ACL, RAIT and PES IEEE source
  directories, entry `pineau2020improving`:
  old `note      = {JMLR vol.\ 22 paper 164, DOI 10.5555/3546258.3546422}`
  → new `note      = {DOI 10.5555/3546258.3546422}`.
  Volume and article number already have dedicated fields. The surname is
  already `d'Alch{\'e}-Buc` in current source; do not change it from PDF extraction.
- Finding **874**, PES IEEE `refs.bib`, `rafailov2023dpo`:
  old `note      = {arXiv:2305.18290; NeurIPS 2023 (proceedings vol. 36)}`
  → new `note      = {arXiv:2305.18290}`.
  The proceedings and volume already have dedicated fields.
- Finding **979**, current thesis `references.bib`:
  old `title     = {Tulu 3: Pushing Frontiers in Open Language Model Post-Training},`
  → new `title     = {{TÜLU} 3: Pushing Frontiers in Open Language Model Post-Training},`.
  Matches the manuscript's spelling without changing citation identity.
- Finding **980**, current thesis `references.bib`:
  old `note      = {arXiv:2210.10760 (2022)}`
  → new `note      = {arXiv:2210.10760}`;
  old `note      = {arXiv:2106.09685 (2021)}`
  → new `note      = {arXiv:2106.09685}`;
  old `note      = {arXiv:2305.20050 (2023)}`
  → new `note      = {arXiv:2305.20050}`.
  Remove only redundant preprint years, not publication years.
- Finding **982**, current thesis `references.bib`:
  old `note      = {arXiv:2507.18014; identifies three consistent training phases: slow start, rapid improvement, plateau}`
  → new `note      = {arXiv:2507.18014}`.
  Removes an inline reading note, not a scientific analysis.

## Requests not safe to turn into automatic replacements

Findings **726, 727, 825–827, 873, 876** require central citation-style,
software-version or bibliographic decisions; no guessed author lists, new
citations, access dates, or publication metadata have been supplied here.
Findings **978, 981** concern BibTeX title-case protection and must be checked
against the central bibliography/style; current `Part I` is already capitalized
in the .bib input, so literal `part i`→`Part I` is not a valid source patch.

## Locked thesis Chapter 6: do not apply in this pass

The entire `ch06_results_core.md` remains byte-identical to branch HEAD.
Requested old→new text for a future explicitly authorized unlocked revision:

- **997**: `in that paragraph` → `in the source document`.
- **998**: `and that GRPO and Dr. GRPO produce` → `and GRPO and Dr. GRPO produce`.
- **999**: `and mixes outcome definitions` → `and mix outcome definitions`.
- **1000**: `β_OLS` → `$\beta_{\mathrm{OLS}}$`;
  `R̄_late` → `$\bar R_{\mathrm{late}}$`.
- **1001**: `R̄_max` → `$\bar R_{\mathrm{max}}$`.
- **1002**: `Δ_1T` → `$\Delta_{1\mathrm{T}}$` (typesetting only;
  a missing mathematical definition still requires source verification).
- **1003**: `σ(r_tail)/|μ(r_tail)|` →
  `$\sigma(r_{\mathrm{tail}})/|\mu(r_{\mathrm{tail}})|$`.

These locked-block requests are not marked fixed. Historical PDF encodings may
differ from current Unicode source; match the exact current substring first.
