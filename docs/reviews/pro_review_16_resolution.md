# Review 16 resolution (partial source)

Source: `review_16_modal_p5p8_1.md`, visible findings 1–7. The supplied export ends at heading “2.”; missing sections and final priorities are **not supplied and not inferred**.

Protocol: review16-v2. No saved research results were regenerated or edited. Controller files use v2 names; other affected analysis outputs use the review16_v2 directory. Changes invalidate comparisons to old computed summaries until separately reviewed recomputation is authorized.

| Finding | Resolution | Synthetic regression |
|---|---|---|
| 1: double-charged controller costs | Applied G is charged once for each actual step and actual prompt count. Final trigger has no charge, baseline derives from observed rows | All eight three-step trigger patterns, unequal prompt counts |
| 2: invalid saved-prompt/oracle claims | Recovery is explicitly UNMEASURED/null, not inferred from degenerate group means. “Oracle” renamed interior heuristic, with no upper-bound claim. Replay is observed-state schedule cost only, not causal training outcomes | Degenerate/mixed groups return missing recovery; no oracle API |
| 3: wrong ZVF SE | H1 means remain descriptive; no CI is emitted without independent ZVF observations. All adjacent group means checked descriptively | Huge accuracy SE cannot affect ZVF output; interior reversal prevents monotonic description |
| 4: additive eta and wrong omega | Removed aggregate marginal eta ratio and headline; retain separate nonadditive associations. Standard one-way omega uses residual degrees of freedom, preserving negative estimates, undefined saturated design returns NaN | Hand-computed 14/22 omega; saturated case; aggregate key absent |
| 5: ignored B axis | Exhaustive five-profile mapping requires A, B and C for honest/informative; both registry-fail cases explicit | All eight truth-table combinations |
| 6: AUC ties/undefined classes | Average-rank equivalent Mann–Whitney implementation, finite binary inputs validated; undefined single class raises | Equal/perfect/reversed scores and permutation; invalid inputs |
| 7: declaration credit and model substrings | Unmeasured declarations earn zero with DECLARED_UNVERIFIED status; exact IDs or two explicit cell-slug aliases only | Unmeasured recognized declaration, substring mismatch and punctuation collision |

Validation: `/tmp/tinker-review-venv/bin/python -m pytest -q tests/reviews/test_review16.py` — 7 passed. CPU-only synthetic data; no training, paid jobs, or evidence fabrication. H1 inference and recovery inference deliberately remain unavailable pending genuine supporting data.
