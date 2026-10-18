# Review 18 resolution (partial source)

Source: `review_18_modal_p5p8_3.md`. Five visible P0 findings are addressed. The export cuts off just after the P1 `paired_bootstrap()` heading and “Confirmed bug:”; its missing explanation, remaining findings, and five-first list are **not supplied and not inferred**.

Protocol review18-v2 writes analysis outputs into `p5p8/review18_v2`. No stored results, original registry entries, or generated analyses were edited during this work. Tests use synthetic data in temporary directories only.

| Visible finding | Resolution | Regression |
|---|---|---|
| Dry-run writes entries; stale post-validation | `plan_entry_patches` deep-copies and transforms in memory. Validate complete proposed entries/schema before persistence. Only --apply writes entries/schema; then reload both from disk for post-audit. Existing schema backup is preserved | Byte-identical dry-run registry/schema; proposed post-validation passes; apply performs a second load and validates persisted canonical status |
| Unconditional claim amendment/SUPPORTS | Removed registry mutation from the robustness script entirely. Exact required checks and nonempty evidence gate a proposed sign for human review; original entries and computed booleans are stored in a uniquely named append-only proposal. No retrospective SUPPORTS claim or hardcoded robustness outcome | Each failing check, missing evidence, and all-pass case; mutation assignment absent. All-pass still requires independent validation |
| Positional step/seed pairing | Shared strict keyed joins reject duplicates, missing identities, empty pairs, nonfinite values. N2 uses the declared 0–39 protocol and complete last-K/per-seed windows. Iter190 accepts identity mappings rather than anonymous arrays | Reordered steps/seeds preserve delta; missing and duplicate steps/seeds fail; anonymous pairing rejected |
| Wrong per-metric SD | Each metric selects its own SD and sample-count field (`<stem>_sd`, `<stem>_n`, or explicitly supplied n_seeds). Missing uncertainty returns a descriptive estimate with null interval/significance; no assumed n=5 | Deliberately divergent risk versus magnitude SD; missing n/SD; negative SD; full synthetic output remains DESCRIPTIVE_ONLY and registry unchanged |
| Skipped schema validation counted green | `validate_entry` returns None/SKIPPED when dependency unavailable. Strict CLI exits nonzero before auditing or mutation. Actual post-validation failures return nonzero | Missing dependency is never True and CLI raises SystemExit |

Validation: `/tmp/tinker-review-venv/bin/python -m pytest -q tests/reviews/test_review16.py tests/reviews/test_review18.py` — 15 passed. Changed Python modules compile. No training, paid jobs, or empirical result regeneration.

Limitations: step-wise bootstrap remains a within-run descriptive uncertainty method, not independent-training-seed inference. This patch does not invent the truncated P1 finding or fix unseen statistical claims. Iter194's repeated RNG seeds are explicitly documented as bootstrap randomness, not independent runs; a passing robustness proposal alone cannot validate a retrospective prediction.
