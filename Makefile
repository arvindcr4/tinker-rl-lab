UV ?= uv
RUFF ?= $(UV) run --no-sync ruff
PYTHON ?= $(UV) run --no-sync python
AUDIT_PATHS := utils/audit_utils.py platform_local/run_all_audits.py platform_local/submission_claim_audit.py platform_local/paper_sync_audit.py platform_local/anonymization_repro_audit.py platform_local/claim_strength_audit.py platform_local/submission_package_audit.py platform_local/submission_workflow_audit.py platform_local/export_guard_audit.py
FIGURE_PATHS := platform_hybrid/paper/figure_module.py platform_hybrid/paper/figures/gen_figures.py platform_hybrid/paper/figures/generate_figures.py platform_hybrid/paper/figures/wave6_sensitivity.py platform_hybrid/paper/neurips_2026_variants/figures
GRPO_PATHS := platform_tinker/tinkerrl platform_tinker/grpo_100_math.py platform_tinker/grpo_100_xlam.py platform_tinker/grpo_exp_a_baseline.py platform_tinker/grpo_gsm8k_base.py platform_tinker/grpo_tooluse_tinker.py
RUFF_PATHS := platform_local/unified platform_local/trl_integrations $(GRPO_PATHS) platform_hybrid/registry/provenance/minreport.py $(AUDIT_PATHS) $(FIGURE_PATHS) utils tests tools

.PHONY: bootstrap check lint format format-check test package docs-check

bootstrap:
	$(UV) sync --locked --extra dev
	$(UV) run --no-sync pre-commit install

check: lint format-check test package docs-check

lint:
	$(RUFF) check $(RUFF_PATHS)
	$(PYTHON) tools/check_repo_policy.py

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

docs-check:
	@test -f README.md
	@test -f REPRODUCE.md
	@test -f CONTRIBUTING.md
	@test -f SECURITY.md
	@test -f ARTIFACT.md
	@test -f LICENSE
