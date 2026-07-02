# utils/ — INDEX

**Purpose:** Shared helper library imported across the experiment scripts — seeding and
statistics/verification (rliable-based aggregate metrics, bootstrap CIs, significance tests).

**Key files:**
- `seed.py` — `set_global_seed()` / `get_seed_from_args()`: deterministic seeding across random/numpy/torch + CLI parsing.
- `stats.py` — statistical analysis tooling: rliable aggregate metrics (IQM), bootstrap confidence intervals, learning-curve plotting with CIs, results tables.
- `verify_results.py` — load multi-seed result JSONs, run Welch t-test / Mann-Whitney U, and verify measured numbers against expected values (`verify()`, `main()` → exit code).
- `__init__.py` — package marker.

**Find it fast:**
- reproducible seeding in a script → `from utils.seed import set_global_seed`
- compute IQM / bootstrap CI / significance → `utils/stats.py`
- check a result JSON matches claimed numbers → `utils/verify_results.py`
