# October 4 publication repair validation

The corrected thesis candidate and its public-review package are complete as local review artifacts. The research project remains incomplete in the broader sense: this work does not supply missing original training provenance, decontamination, human approval, external permissions or hosted CI capacity. No new publication files or private evidence were pushed.

## Delivered candidate

- [306-page PDF](reports/public_revision_2026-10-04/thesis/Thesis_Report_ArvindCR_C1_Coverage_Derived_2026-10-04_PUBLIC.pdf): 1,679,101 bytes; SHA-256 `4840e78d8d17d23396f4699de1fcfc8c1d78b5649136f611f93ee63e1a9ed016`.
- [Exact publication inventory](reports/public_revision_2026-10-04/PUBLICATION_MANIFEST.json), [source availability](reports/public_revision_2026-10-04/thesis/public_source_availability.json), and [quality receipt](reports/public_revision_2026-10-04/thesis/public_quality_receipt.json).
- Deterministic thesis-only ZIP: 82 files; SHA-256 `8c7afaf54334eef19b9356ec4f02ace7d756e95a5441d6b2ee328d2851f2190c`. Created and verified locally; regenerate using [package instructions](PUBLIC_REVIEW_PACKAGE.md).
- Numerical audit: 915 passing value checks and 22 context-only values; no missing or unbound values in the recovered local verification. Methods distinguish recomputation, stored summaries, source transcriptions and arithmetic. The public derivative explicitly declares incomplete reproducibility: 31 original-source patterns remain withheld, while 135 patterns refer to tracked repository sources (some outside the thesis ZIP).

Recovered 1,022 exact original local files, totaling 22,769,102 bytes, only into ignored locations. A private SHA-256 recovery inventory and the complete local verification report are retained outside the public tree. No private raw exports, source patches or provider histories were staged. No denied collaborator run was accessed.

## Corrections and review

The revised manuscript removes unsupported causal budget/harness claims, bounds training-suite exclusion separately from content decontamination, explains checkpoint/exposure confounding, limits E4 mechanism claims to surviving summaries, corrects Jensen's inequality and the Family D record count, and bounds novelty to the actual literature survey. Front matter and generated figure captions carry those limits. Historical scientific results and the C1 case ledger are unchanged.

All 24 figures compiled and the final PDF compiled with Pandoc/Tectonic. Rendered physical pages 1–156 were reviewed by the authoring assistant; an independent assistant reviewed 157–306. Dense and changed pages were enlarged. A final caption adjustment changed only pages 15 and 74, both re-reviewed; independent verification found the remaining reviewed pages pixel-identical. This is nonhuman assistant review, not academic approval, source-authenticity certification, or independent experimental regrading.

All-page text bounds, replacement-character, selected privacy-pattern and embedded-file checks reported zero findings. Gitleaks reported two generic-key matches in the JSON field named `token`; inspection confirmed they are ordinary model-name/score prose (Llama and Qwen descriptions), not credentials. The scan is not a guarantee of complete privacy. No scanner configuration was weakened.

## Validation

- 1,139 unique tests passed, one non-CPU-device test skipped, no remaining failures. Results combine the full CPU run, two targeted frozen-release reruns and two native-LaTeX tests; they are not a second complete test run.
- CPU coverage: 83.98%, exceeding the unchanged 77% gate.
- Final affected provenance/package suite: 89 passed.
- Ruff, formatting, mypy (20 files), repository policy, docs, lock/export freshness and all nine audits passed.
- Exact October 4 checker: 81 files and 78 thesis-source bindings verified; ZIP build and verification passed.
- Frozen October 3 checks remain valid. Its historical stale PDF quality receipt remains an explicit historical limitation. Root README is bound by that frozen manifest and was preserved; new candidate navigation is in PUBLIC_RESEARCH_CHECKS.md and PUBLIC_REVIEW_PACKAGE.md.

The latest known hosted CI run for the preceding pushed code commit was blocked before execution by account billing, rather than by a test failure. No billing changes or additional hosted job triggers were attempted here. No GPU, model-training, paid-compute or external collaborator jobs ran.

## Remaining work

1. Review and authorize publication of this new candidate; no upload or institutional submission is implied. Human academic approval and signatures remain separate.
2. Reconstruct original consumed training rows and dirty execution source, then perform content-level decontamination. Identity receipts do not establish those facts.
3. Obtain legitimately authorized original trajectories and unavailable external scopes if stronger causal or original-contract completeness claims are desired. Current text preserves those limits.
4. Resolve the account-level CI billing block through the account owner, then run hosted CI for the exact reviewed commit.
5. Optional tooling follow-up: `utils/stats.py` still accepts inactive `--format`/`--rliable` options, and `verify_results.py` applies experiment-level expectations to individual seeds. Appendix B accurately records these limitations; this publication repair did not redesign those interfaces.

The E1–E14 campaign and C1 diagnostics do not establish a training gain. Stored-summary agreement and source transcriptions are narrower evidence than recomputation, and local verification is narrower than a publicly replayable experiment archive.
