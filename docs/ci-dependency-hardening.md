# CI dependency binding and lint migration

R15.21 changes bind existing action major releases to official upstream commits.
Read-only verification on 2026-10-17 used:

- https://api.github.com/repos/actions/checkout/git/ref/tags/v7 → `3d3c42e5aac5ba805825da76410c181273ba90b1`
- https://api.github.com/repos/actions/setup-python/git/ref/tags/v6 → `ece7cb06caefa5fff74198d8649806c4678c61a1`
- https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_checksums.txt → linux_x64 SHA-256 `551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`

The scanner archive is downloaded to runner scratch space, verified against the
tracked digest, and only then extracted and executed. Shell regressions use
local fixture archives to check success, digest mismatch, download failure and
invalid archive failure. They do not execute a downloaded scanner or validate
GitHub-hosted runtime behavior. Existing read-only workflow permissions remain.

F821/F811 are no longer exempted in exploratory scripts. The lint target also
checks name safety across platform trees, autoresearch, analysis, utils, tests,
and tools, beyond the narrower style/formatting scope. These source repairs do
not regenerate historical results or establish scientific validity.

## Incomplete Python 3.11 zip migration

The supported Ruff target is now py311. B905 becomes newly applicable at this
target, and remains temporarily ignored pending call-by-call review. Blindly
setting strict=False would hide mismatched scientific pairings; strict=True
would break intentional adjacent-window/truncating uses. Neither blanket fix
has been applied. Other bug-finding checks remain enabled.

Snapshot inventory at integration base `378de199` plus this patch: **98 calls**
in the curated `RUFF_PATHS` style scope (Ruff 0.17.1). Line numbers may drift:

- `platform_hybrid/paper/figures/generate_figures.py`: 2 calls, lines 163, 227
- `platform_local/trl_integrations/scientific.py`: 2 calls, lines 46, 58
- `platform_tinker/tinkerrl/critic.py`: 3 calls, lines 109, 228, 259
- `platform_tinker/tinkerrl/grpo.py`: 17 calls, lines 873, 934, 1000, 1085, 1095, 1306, 2933, 3041, 3042, 3060, 3065, 3079, 3081, 3085, 3090, 3142, 3884
- `tests/reviews/test_hybrid_framework_review.py`: 1 calls, lines 128
- `tests/reviews/test_review09_resume.py`: 1 calls, lines 37
- `tests/reviews/test_review23_output_closure.py`: 1 calls, lines 23
- `tests/reviews/test_review27_diagnostic_followup.py`: 1 calls, lines 351
- `tests/test_analysis_validation.py`: 1 calls, lines 90
- `tests/test_cpu_causal_lm_smoke.py`: 5 calls, lines 61, 125, 133, 135, 142
- `tests/test_cpu_critic_resume_smoke.py`: 1 calls, lines 31
- `tests/test_cpu_optimizer_smoke.py`: 3 calls, lines 78, 242, 248
- `tests/test_critic.py`: 5 calls, lines 26, 28, 39, 50, 52
- `tests/test_dataset_revision_fidelity.py`: 1 calls, lines 22
- `tests/test_experiments_next_quality.py`: 1 calls, lines 109
- `tests/test_grpo_loss.py`: 2 calls, lines 65, 241
- `tests/test_loss_numerical_stability.py`: 3 calls, lines 18, 22, 31
- `tests/test_pcd_vs_zvf.py`: 1 calls, lines 39
- `tests/test_portfolio_comparability_audit.py`: 1 calls, lines 34
- `tests/test_portfolio_eval_integrity.py`: 1 calls, lines 165
- `tests/test_pro_review_01_analysis.py`: 1 calls, lines 595
- `tests/test_pro_review_02_modal_analysis.py`: 1 calls, lines 605
- `tests/test_pro_review_29.py`: 1 calls, lines 39
- `tests/test_pro_review_29_mains.py`: 1 calls, lines 41
- `tests/test_properties_advantages.py`: 11 calls, lines 89, 103, 122, 132, 142, 158, 182, 195, 197, 200, 226
- `tests/test_properties_stats.py`: 2 calls, lines 160, 161
- `tests/test_public_analysis.py`: 1 calls, lines 603
- `tests/test_registry_provenance_review.py`: 2 calls, lines 83, 91
- `tests/test_samestack_protocol.py`: 1 calls, lines 59
- `tests/test_tinker_sdk_loss_integration.py`: 3 calls, lines 104, 106, 135
- `tools/audit_portfolio_comparability.py`: 1 calls, lines 268
- `tools/audit_thesis_numbers.py`: 6 calls, lines 183, 239, 446, 2568, 2574, 2797
- `tools/check_public_results.py`: 3 calls, lines 381, 394, 409
- `tools/check_thesis_evidence.py`: 9 calls, lines 110, 111, 121, 156, 211, 459, 465, 481, 490
- `utils/stats.py`: 2 calls, lines 317, 528
