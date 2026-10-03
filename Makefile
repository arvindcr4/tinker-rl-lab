UV ?= uv
RUFF ?= $(UV) run --no-sync ruff
PYTHON ?= $(UV) run --no-sync python
AUDIT_PATHS := utils/audit_utils.py platform_local/run_all_audits.py platform_local/submission_claim_audit.py platform_local/paper_sync_audit.py platform_local/anonymization_repro_audit.py platform_local/claim_strength_audit.py platform_local/submission_package_audit.py platform_local/submission_workflow_audit.py platform_local/export_guard_audit.py platform_local/reviewer_caveat_audit.py platform_local/scientific_audit.py
FIGURE_PATHS := platform_hybrid/paper/figure_module.py platform_hybrid/paper/figures/gen_figures.py platform_hybrid/paper/figures/generate_figures.py platform_hybrid/paper/figures/wave6_sensitivity.py platform_hybrid/paper/neurips_2026_variants/figures
GRPO_PATHS := platform_tinker/tinkerrl platform_tinker/grpo_100_math.py platform_tinker/grpo_100_xlam.py platform_tinker/grpo_exp_a_baseline.py platform_tinker/grpo_gsm8k_base.py platform_tinker/grpo_tooluse_tinker.py
SUBMISSION_PATHS := platform_modal/scripts/build_university_submission.py
RUFF_PATHS := $(SUBMISSION_PATHS) platform_local/unified platform_local/trl_integrations $(GRPO_PATHS) platform_hybrid/registry/provenance/minreport.py $(AUDIT_PATHS) $(FIGURE_PATHS) utils tests tools

.PHONY: bootstrap check lint lint-ruff format format-check test package docs-check submission submission-check public-check

bootstrap:
	$(UV) sync --locked --extra dev
	$(UV) run --no-sync pre-commit install

check: lint format-check test package docs-check public-check

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

test:
	$(UV) run --no-sync pytest tests/

package:
	$(UV) lock --check
	$(UV) build --wheel
	$(PYTHON) tools/check_wheel.py dist/*.whl

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
	@test -f SUBMISSION.md
	@test -f README.md
	@test -f REPRODUCE.md
	@test -f CONTRIBUTING.md
	@test -f SECURITY.md
	@test -f ARTIFACT.md
	@test -f LICENSE
