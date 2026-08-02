# Deep Dive: Tooling, Outputs & Utilities

## Overview
The supporting toolchain splits cleanly into three tiers: **(1) maintained infrastructure** (`utils/`, `tools/`) that experiments, audits, and CI depend on; **(2) programmatic deck generators** (`outputs/build_*_deck.py`) that produce `.pptx`/`.pdf` artifacts by reading the repo directly at build time; and **(3) one-shot root scripts** (`fix_links.py`, `refactor.py`, `split_chunks.py`, etc.) that were point-fixes for past migrations and are essentially frozen. The unifying pattern is *code-as-artifact*: decks, audits, and verifications all derive their numbers from the checkout, never from hand-typed constants, so a reviewer rebuilding the project regenerates the same evidence.

## Key Components
- `outputs/build_program_review_deck.py:page_count` — opens each P1–P12 paper PDF via `pypdf` so slide labels reflect actual manuscript length.
- `outputs/build_progress_update_deck.py:e1_rows` — regex-parses `zvf-program/audit/COLAB_EXECUTION_STATUS.md` and aggregates per-arm means at deck-build time.
- `utils/stats.py:compute_bootstrap_ci` — 10k-resample CI plus Welch t-test, Mann-Whitney U, IQM, and optional `rliable` aggregate metrics (Agarwal 2021).
- `utils/tinker_grpo.py:run_grpo_training` — canonical GRPO loop against the Tinker service client: per-group advantage normalization, custom loss closure, checkpoint/save-reload cadence.
- `utils/verify_results.py:verify` — artifact-reviewer harness that matches result `.json`/`.log` files against `expected_results.json` within ±5 / ±10 pt tolerances.
- `utils/audit_utils.py:AuditSuiteResult` — frozen dataclass envelope (`AuditIssue`/`AuditResult`/`AuditSuiteResult`) every repo audit emits, with `METRIC name=N` rendering for CI.
- `utils/seed.py:set_global_seed` — sets Python/NumPy/Torch/CUDA + `CUBLAS_WORKSPACE_CONFIG`, returns env metadata for the run ledger.
- `tools/check_stale_verdicts.py` — content gate that fails any unmarked quote of the superseded DAPO `DISAPPEARS` verdict after the 2026-08-02 reanalysis.
- `tools/check_repo_policy.py` — required-files, secret-like, >100 MiB, and merge-marker gate; composes with `check_stale_verdicts`.
- `tools/check_wheel.py` — asserts the built wheel ships every supported module plus the `tinkerrl` console entry point.
- `upload_tinker_to_wandb.py:parse_log` — converts plain Tinker training logs into historical W&B runs (with a `wandb.log` monkey-patch for VRAM metrics).

## Concepts & Decisions

### Code-as-deck (python-pptx from repo state)
- **What**: Each `build_*_deck.py` is a single ~400–800-line script that hand-rolls slides via `python-pptx`, sharing one dark-navy palette and 16:9 geometry (`W, H = 13.333, 7.5`).
- **Why**: Forces every visible number to be re-derived at build time from PDFs, git history (`subprocess.check_output(["git", "rev-list", ...])`), or audit TSVs — no slide can drift from the evidence.
- **Trade-offs**: No template layer; layout is imperative (`add_shape`/`add_text` with inch coordinates). Cheap to write, costly to restyle. The four decks duplicate the palette and primitives rather than factoring out a shared helper module.
- **Alternatives**: Marp/Slidev (loses programmatic access to PDFs/git), a shared `outputs/_deck_lib.py` (would pay itself back only if a fifth deck lands).

### Maintained utils vs. throwaway root scripts
- **What**: `utils/` is imported by training scripts, the wheel, and verifiers; `tools/` is wired into CI/pre-commit. The root `.py` files are un-factored, single-purpose text mutators.
- **Why used here**: Clear lifecycle signal — anything in `utils/`/`tools/` is part of the supported surface (`check_wheel.py` even enumerates `utils/seed.py`, `utils/stats.py` as required wheel members); anything at the root is historical.
- **Trade-offs**: The root scripts (`refactor.py` doing ~530 lines of `str.replace` over `train_grpo_unsloth.py`, `update_audit.py` regex-rewriting a LaTeX section) are write-once/read-never and would be cleaner as Codemod `libcst` transforms. They survive because the migration is done and the diff is frozen.
- **Alternatives**: Promote surviving root scripts into `scripts/migrations/` with a README; delete the rest.

### Statistical rigor as a utility
- **Why**: `stats.py` cites Colas (2019), Agarwal (2021), Patterson (2024) and ships IQM + rliable handler with a `try/except ImportError` fallback to bootstrap-only. `verify_results.py` pins tolerance per metric (`--last10-tolerance 0.05`, `--peak-tolerance 0.10`) and documents their justification in `ARTIFACT.md §6`.
- **Trade-offs**: Default expected-values are hardcoded from `main.tex` Table 2; drifts if the table is edited without regenerating `expected_results.json`.

### Audit framework as structured data
- **What**: Every audit returns `AuditSuiteResult(audits=(...))`; issues are `AuditIssue(code, message, location)` frozen dataclasses with `slots=True`.
- **Why**: Renders deterministic `METRIC name=N` lines that CI can grep, and `_coerce_issue` accepts str/tuple/dataclass so individual audits can stay terse.
- **Alternatives**: Plain asserts with print — loses the suite-level pass/fail rollup and the lazy-cached `AuditContext` file reader.

## Related Code
- Decks consume: `platform_hybrid/paper/paper_P*.pdf`, `platform_hybrid/paper/figures/v2/`, `zvf-program/audit/COLAB_EXECUTION_STATUS.md`, `outputs/wandb_*_2026-07-12.png`, and git history.
- `verify_results.py` consumes `platform_hybrid/paper/expected_results.json` and `results/**/*.json|*.log`.
- `upload_tinker_to_wandb.py` consumes `experiments/tinker-runs/logs/*.log`; produces W&B project `tinker-rl-scaling`.
- `tools/check_stale_verdicts.py` enforces `zvf-program/audit/STATISTICAL_REANALYSIS.md`.
- `utils/tinker_grpo.py` consumes Tinker SDK + a caller-supplied `reward_fn` and dataset.

## Start Here
1. `outputs/build_progress_update_deck.py` — smallest deck, clearest example of repo-grounded slide generation.
2. `utils/stats.py` — the statistical spine of every claim in the portfolio.
3. `utils/audit_utils.py` + `tools/check_stale_verdicts.py` — pair these to see how the audit dataclass contract becomes a CI gate.

---
*Generated by AntiVibe (full-repo pass) · 2026-08-02*
