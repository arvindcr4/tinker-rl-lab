UV ?= uv
RUFF ?= $(UV) run --no-sync ruff
PYTHON ?= $(UV) run --no-sync python
AUDIT_PATHS := utils/audit_utils.py platform_local/run_all_audits.py platform_local/submission_claim_audit.py platform_local/paper_sync_audit.py platform_local/anonymization_repro_audit.py platform_local/claim_strength_audit.py platform_local/submission_package_audit.py platform_local/submission_workflow_audit.py platform_local/export_guard_audit.py platform_local/reviewer_caveat_audit.py platform_local/scientific_audit.py
FIGURE_PATHS := platform_hybrid/paper/figure_module.py platform_hybrid/paper/figures/gen_figures.py platform_hybrid/paper/figures/generate_figures.py platform_hybrid/paper/figures/wave6_sensitivity.py platform_hybrid/paper/neurips_2026_variants/figures
GRPO_PATHS := platform_tinker/tinkerrl platform_tinker/grpo_100_math.py platform_tinker/grpo_100_xlam.py platform_tinker/grpo_exp_a_baseline.py platform_tinker/grpo_gsm8k_base.py platform_tinker/grpo_tooluse_tinker.py
SUBMISSION_PATHS := platform_modal/scripts/build_university_submission.py
RUFF_PATHS := $(SUBMISSION_PATHS) platform_local/unified platform_local/trl_integrations $(GRPO_PATHS) platform_hybrid/registry/provenance/minreport.py $(AUDIT_PATHS) $(FIGURE_PATHS) utils tests tools

.PHONY: bootstrap check lint lint-ruff format format-check typecheck test coverage package lock-check secrets secrets-history docs-check submission submission-check public-check

bootstrap:
	$(UV) sync --locked --extra dev
	$(UV) run --no-sync pre-commit install

# `coverage` runs the full test suite with the .coveragerc fail_under gate, so it
# replaces a plain `test` run here.
check: lint format-check typecheck coverage package lock-check secrets docs-check public-check

# Split so pre-commit can reuse the exact same linted file list as CI
# (`make lint` = `make lint-ruff` + the repository policy gate).
lint: lint-ruff
	$(PYTHON) tools/check_repo_policy.py

lint-ruff:
	$(RUFF) check $(RUFF_PATHS)

format:
	$(RUFF) format $(RUFF_PATHS)

format-check:
	$(RUFF) format --check $(RUFF_PATHS)

typecheck:
	$(UV) run --no-sync mypy

test:
	$(UV) run --no-sync pytest tests/

coverage:
	$(UV) run --no-sync pytest tests/ -q --cov --cov-config=.coveragerc --cov-report=term

package:
	$(UV) lock --check
	$(UV) build --wheel
	$(PYTHON) tools/check_wheel.py dist/*.whl

# requirements-lock.txt must be the hashed export of uv.lock (Docker installs it).
# Regenerate with: $(LOCK_EXPORT) -o requirements-lock.txt
LOCK_EXPORT = $(UV) export --frozen --no-dev --extra all --extra dev --no-emit-project --format requirements-txt -q
lock-check:
	$(UV) lock --check
	@$(LOCK_EXPORT) --no-header | diff -q - $$(f=$$(mktemp); grep -v '^#' requirements-lock.txt > $$f; echo $$f) > /dev/null \
		|| { echo "requirements-lock.txt is stale; run: $(LOCK_EXPORT) -o requirements-lock.txt"; exit 1; }

# Secret scan (gitleaks >= 8.30, `brew install gitleaks`; config .gitleaks.toml).
# `secrets` scans uncommitted changes (fast). `secrets-history` scans every commit
# and fails until the two leaked Tinker keys are rotated and fingerprinted in
# .gitleaksignore (see SECURITY notes).
GITLEAKS ?= gitleaks
secrets:
	$(GITLEAKS) git --pre-commit --config .gitleaks.toml --redact --no-banner .

secrets-history:
	$(GITLEAKS) git --config .gitleaks.toml --redact --no-banner --log-opts=--all .

submission:
	$(PYTHON) platform_modal/scripts/build_university_submission.py

submission-check:
	$(PYTHON) tools/check_thesis_evidence.py
	$(PYTHON) submission/demo/run_demo.py --self-test

# Standard library only: use `make public-check PYTHON=python3` without uv or a GPU.
# The 2026-10-03 manifest is read-only; this target never refreshes expected hashes.
public-check:
	$(PYTHON) -B tools/check_public_release.py
	$(PYTHON) -B tools/check_public_results.py
	$(PYTHON) -B -m unittest discover -s tests -p 'test_public*.py' -v

docs-check:
	@test -f PUBLIC_RESEARCH_CHECKS.md
	@test -f BUILD_PUBLIC_RESEARCH.md
	@test -f LATEX_AUDIT.md
	@test -f SUBMISSION.md
	@test -f README.md
	@test -f REPRODUCE.md
	@test -f CONTRIBUTING.md
	@test -f SECURITY.md
	@test -f ARTIFACT.md
	@test -f LICENSE
