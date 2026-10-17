# October 4 publication repair validation

The corrected thesis candidate and its public-review package are complete as local review artifacts. The research project remains incomplete in the broader sense: this work does not supply missing original training provenance, decontamination, human approval, external permissions or hosted CI capacity. No new publication files or private evidence were pushed.

## Delivered candidate

- [306-page PDF](reports/public_revision_2026-10-04/thesis/Thesis_Report_ArvindCR_C1_Coverage_Derived_2026-10-04_PUBLIC.pdf): 1,679,697 bytes; SHA-256 `d6add94956973dc319de9d4e6f30a2263fd28d3a4939ced05156775ef6f8213d`.
- [Exact publication inventory](reports/public_revision_2026-10-04/PUBLICATION_MANIFEST.json), [source availability](reports/public_revision_2026-10-04/thesis/public_source_availability.json), and [quality receipt](reports/public_revision_2026-10-04/thesis/public_quality_receipt.json).
- Deterministic thesis-only ZIP: 82 publication files plus one package index (83 ZIP members); SHA-256 `71f3b214a7a8782537fa9b328b9e39dd0ad68b9db0171c619f51d3c1d8e46957`. Created and verified locally; regenerate using [package instructions](PUBLIC_REVIEW_PACKAGE.md).
- Numerical audit: 915 passing value checks and 22 context-only values; no missing or unbound values in the recovered local verification. Methods distinguish recomputation, stored summaries, source transcriptions and arithmetic. The public derivative explicitly declares incomplete reproducibility: 31 original-source patterns remain withheld, while 135 patterns refer to tracked repository sources (some outside the thesis ZIP).

Recovered 1,022 exact original local files, totaling 22,769,102 bytes, only into ignored locations. A private SHA-256 recovery inventory and the complete local verification report are retained outside the public tree. No private raw exports, source patches or provider histories were staged. No denied collaborator run was accessed.

## Corrections and review

The revised manuscript removes unsupported causal budget/harness claims, bounds training-suite exclusion separately from content decontamination, explains checkpoint/exposure confounding, limits E4 mechanism claims to surviving summaries, corrects Jensen's inequality and the Family D record count, and bounds novelty to the actual literature survey. Front matter and generated figure captions carry those limits. Historical scientific results and the C1 case ledger are unchanged.

All 24 figures compiled and the final PDF compiled with Pandoc/Tectonic. Rendered physical pages 1–156 were reviewed by the authoring assistant; an independent assistant reviewed 157–306. Dense and changed pages were enlarged. The final CLI-documentation rebuild changed only physical pages 12, 226–238 and 306. Page 12 was re-reviewed by the authoring assistant; the independent assistant re-reviewed 226–238 and 306 and verified the remaining assigned page images pixel-identical. This is nonhuman assistant review, not academic approval, source-authenticity certification, or independent experimental regrading.

All-page text bounds, replacement-character, selected privacy-pattern and embedded-file checks reported zero findings. Gitleaks reported two generic-key matches in the JSON field named `token`; inspection confirmed they are ordinary model-name/score prose (Llama and Qwen descriptions), not credentials. The scan is not a guarantee of complete privacy. No scanner configuration was weakened.

## Validation

- One complete final invocation on settled code `ce3bbaa4f67b503df79b3a81593df22f17359d0f`: **1,164 passed, one non-CPU-device skip, zero failures, zero deselections**, including both native-LaTeX tests and the installed property-test dependencies. This is the direct final-run result, not a sum of partial runs. Runtime was 99.15 seconds.
- Coverage: **84.68%**, exceeding the unchanged 77% gate. Two expected warnings arose from the module-entrypoint test and an intentionally undefined constant-input correlation test.
- Independent assistant code review: 44 focused tests passed after resolving the four reproduced incomplete/conflicting-input failure cases. Final complete-suite results above include those regressions.
- Ruff, formatting (123 files), mypy (20 files), repository policy, docs, lock/export freshness and all nine audits passed.
- Exact October 4 checker: 81 files and 78 thesis-source bindings verified; ZIP build and verification passed.
- Frozen October 3 checks remain valid. Its historical stale PDF quality receipt remains an explicit historical limitation. Root README is bound by that frozen manifest and was preserved; new candidate navigation is in PUBLIC_RESEARCH_CHECKS.md and PUBLIC_REVIEW_PACKAGE.md.

The latest known hosted CI run for the preceding pushed code commit was blocked before execution by account billing, rather than by a test failure. No billing changes or additional hosted job triggers were attempted here. No GPU, model-training, paid-compute or external collaborator jobs ran.

## Completed CLI repairs

`utils/stats.py --format csv|latex|both` now writes the requested numeric summary using the same bootstrap computation as the terminal report. `--rliable` fails explicitly because the CLI input has no normalized task matrix; it no longer silently claims to act on that option. Missing/nonfinite metrics and empty inputs fail rather than becoming zero scores.

`utils/verify_results.py` now retains seed identity from logs, JSON and filenames. Expectations can name a seed, provide a per-seed mapping, or explicitly declare seed independence. Only the established headline seed42 is scoped in the shipped defaults; other original numbers remain unchanged and unscoped until evidence supports their applicability. Unknown/conflicting seeds, boolean/missing metrics, incomplete seed blocks and malformed matching files cannot silently pass. Strict mode checks every declared seed reference. REPRODUCE.md and thesis Appendix B describe the implemented behavior.

Library delivery remains blocked by the supported helper reporting that prepared uploads are unavailable in this runtime. No Library identity or successful upload was confirmed; no alternative write route was attempted.

## Remaining work

1. Review and authorize publication of this new candidate; no upload or institutional submission is implied. Human academic approval and signatures remain separate.
2. Reconstruct original consumed training rows and dirty execution source, then perform content-level decontamination. Identity receipts do not establish those facts.
3. Obtain legitimately authorized original trajectories and unavailable external scopes if stronger causal or original-contract completeness claims are desired. Current text preserves those limits.
4. Resolve the account-level CI billing block through the account owner, then run hosted CI for the exact reviewed commit.
5. Independent per-seed reference data are still needed for seeds whose applicable historical expectations are not established. The verifier now reports these cases as UNVERIFIED rather than borrowing another seed’s result. Supplying missing evidence remains a research task, not a CLI defect.

The E1–E14 campaign and C1 diagnostics do not establish a training gain. Stored-summary agreement and source transcriptions are narrower evidence than recomputation, and local verification is narrower than a publicly replayable experiment archive.
