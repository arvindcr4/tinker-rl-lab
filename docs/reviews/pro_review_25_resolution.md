# Review 25 resolution — review25-v2

The supplied review is partial: it ends during item 7. Missing continuation and any later findings are explicitly skipped, not reconstructed.

Implemented:
1. Figure136 indexes task/algorithm/seed/hypothesis, rejects duplicates, pairs only intersecting seeds, requires two matches, and removes hardcoded 8/8/significance title. New v2 figure filenames.
2. Iter56 cumulative-tax, portrait, and half-life run/comparison outputs have separate single-grain schemas, staged before replacement. Sparse R0 rows retain the run schema. Corrected outputs use v2 filenames.
3. Iter68 AUC now integrates signed GRPO-minus-Dr.GRPO differences on its unit-spaced observation grid. Reversing order reverses sign. Corrected outputs use v2 filenames; real irregular step-grid integration remains a future improvement.
4. Iter60 only identifies a quadratic maximum for negative curvature; singular/convex fits remain unavailable, not zero. Paired statistics exclude nonfinite pairs and require two valid pairs. Corrected outputs use v2 filenames. Exclusion counts should be added in a future schema revision.
5. Not reproduced in current source: iter72 contains no first-divergence statistic or fds bootstrap; it implements AR/cross-lag analysis. No speculative edit was made to a different estimator.
6. Figure40 now plots paired differences and their actual supplied interval endpoints. Removed dead fabricated individual error-bar arrays. Execution moved behind main guard; output uses v2 filename.
7. Missing/nonfinite/out-of-domain ZVF is unavailable and excluded from named ZVF bins, rather than high ZVF.

Validation: `python -m pytest -q tests/reviews/test_review25.py` (12 passed), synthetic fixtures only. No original experiment data or generated artifacts modified. These repairs do not recertify historical conclusions; regeneration requires a separately reviewed evidence protocol.
