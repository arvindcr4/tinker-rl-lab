# Review 21 residual code corrections

Scope: deterministic implementation repairs after the first review-21 corrections. No training, external jobs, empirical reruns, historical-result edits, or publication were performed.

- R6/R13/R17: iter112 exports actual cohort counts and explicit composed-XGB-proxy provenance. Iter116 reads only corrected inputs, rejects legacy outcome records, uses actual N, charges screening on all rows plus the additional routed cost, and preserves unavailable cost-per-catch and comparisons when nothing is caught. Its summaries no longer invent dataset sizes or infer universal dominance from the absence of a gradient win.
- R6/R13: iter152's price sweep now charges additive screening. It does not have second-stage predictions, so hybrid cost-per-catch is unavailable rather than using the baseline's catches. The helper accepts an explicitly supplied composed catch count, but the main experiment does not fabricate one. Seed-CV cells hold the feature set fixed.
- R14: iter88 retains infeasible fixed-recall costs as null, excluding them from arithmetic. Single-sensor cost bootstrap reports an unavailable interval if any replicate has undefined cost; such replicates no longer become zero differences.
- R17: PR-AUC point estimates and bootstrap contrasts reuse the same retained row IDs, which are exported. Iter160 uses retained-negative population arithmetic and exports actual N/prevalence. Iter200 returns each original row exactly once when the requested pool equals the release pool. Lower-than-release synthetic expansions still use explicitly documented replacement sampling.
- R21: cohort ranking now measures held-out within-cohort permutation Brier-loss increases on a shared trained model. Rank correlations align by feature and handle ties; constant rankings are unavailable. Iter140's firing-preservation check uses event-set IoU and exports anchor retention/support, rather than majority-negative overall agreement. Iter192's directional high-V_std comparison uses actual V_std strata, not V_mean or absolute gaps.
- R24/R25: summary groups use the table's actual pair IDs; first-flip summaries return sigma values; best-single selection uses feature counts and 24-full reports 24 features; threshold conclusions are generated from current records; single-sensor and subgroup bootstrap seeds use SHA-256 rather than salted Python hashes.

## Validation

`/tmp/tinker-review-venv/bin/python -m pytest -q tests/test_pro_review_19_regressions.py tests/test_pro_review_20_regressions.py tests/test_pro_review_21_regressions.py tests/test_pro_review_21_residuals.py`

40 tests pass, including 11 new residual tests. These use synthetic fixtures and checked-in helper/analysis blocks, with no model fitting. The iter116 downstream test runs the full entry point against temporary synthetic tables, including a zero-catch scenario and rejection of legacy records. Compile and whitespace checks pass; undefined-name lint passes for edited Python files. The new test file passes the repository lint configuration. Existing unrelated unused-variable/import and redundant-f-string lint findings in analysis scripts remain outside this repair.

## Evidence limits

These repairs do not validate historical outputs. Corrected analyses still need an authorized rerun and independent evaluation before scientific claims can change. XGB aggregate-feature proxies and hypothetical price tiers are not actual LLM measurements. Permutation attribution, IoU thresholds and V_std contrasts remain exploratory; no new empirical support, multiple-testing correction, or external-validity claim is established. Broader fit caching/shared-core consolidation and full live-LLM evidence remain unimplemented design/evidence work.
