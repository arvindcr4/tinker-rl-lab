# Review 24 resolution — review24-v2 (partial)

The review ends during design item 8; its missing end and any later findings are explicitly skipped.

Implemented:
- Dry-run is passed through to the training boundary and returns synthetic output before any training initialization, even with torch installed. Missing torch no longer silently substitutes measurements. Dry-run/training CSVs are separated and evidence kind recorded. Removed import-time emissions tracker and single-step-irrelevant gamma/lambda sweeps.
- Iter99 requires explicit original group labels and shuffles their existing multiset, preserving group sizes; caller derives labels beside pooled curves.
- Iter75 summary-interval draws refit the identical constrained estimator, with 95% interval-derived SEs. Failed fits cannot narrow the output silently. Inference is explicitly sensitivity analysis; calibrated null rejection is disabled pending an actual null model/sampling protocol.
- Iter91/67 share sorted earliest log-budget crossing logic, including exact and flat hits, nonmonotone curves and duplicate-budget rejection.
- Iter87 intersects observed group budgets and rejects fewer than two matches.
- Iter71 fails on missing ZVF/reference evidence rather than assuming perfect utility.
- Iter104/108 conditional CCF keeps original indices, conditions both endpoints, records pair counts, and returns NaN for sparse/constant bins. Iter104 exports counts. Seed derivation uses stable master/task/quantile/statistic identities.
- Iter100 FEVD horizon labels now select cumulative terms 0 through h-1.
- Changed literal analysis artifact filenames use v2, preserving historical outputs.

Validation: `python -m pytest -q tests/reviews/test_review24.py` (11 passed). Synthetic CPU tests only, including a poisoned training initialization that dry-run cannot reach.

Remaining: iter95 joint-ceiling resampling/failed-fit accounting and model-dependence language; iter79 bound/extrapolation labeling; full iter71 identifiability analysis; downstream NaN/exclusion reporting and iter108 exported per-lag counts; an analytical FEVD oracle test. No historical scientific claim is recertified by these targeted repairs.
