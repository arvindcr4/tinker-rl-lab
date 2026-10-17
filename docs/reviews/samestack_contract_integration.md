# Same-stack integration boundaries

The canonical `samestack_v2` entrypoints implement the versioned v3 protocol and manifest-bound analysis from the same-stack review. They require full source-bound completion inventories and never upgrade historical records merely because seed numbers match. The Colab CLI stops a session only when it explicitly created it with `--new`; failures preserve it for recovery.

The explicitly named `legacy_*` modules retain the separate PR10/11 historical publication/recovery contracts. `legacy_v2_analysis.py` and `legacy_stats_lib.py` use the candidate publication-plan schema, including conservative null parametric inference at zero variance. The archive publication gate invokes that legacy analysis explicitly. Its gold-bound independent regrading and hardware-policy verifier remain in `legacy_verify_and_merge.py`. The legacy driver supports recovery and rejects new training execution; new runs use the canonical driver.

These are distinct contracts, not interchangeable analyses. In particular, canonical exact grading/degenerate-limit statistics and legacy last-box/zero-variance rules may differ. Neither legacy receipts nor legacy publication outputs substitute for canonical completion validation, final-analysis eligibility, or verified equivalence. Tests target each named contract explicitly rather than changing expectations to conceal their differences.

No stored experimental results, generated thesis artifacts, generated demo artifacts, `platform_calibration.py`, or `Platform.jsx` are rewritten by this integration. The pre-existing platform display's inference limitations remain outside this repair.
