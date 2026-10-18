from __future__ import annotations

import unittest

import torch

from .fixtures import (
    AERO_POSTERIOR_FIXTURE,
    ALL_CORRECT_FIXTURE,
    ALL_WRONG_FIXTURE,
    BASE_FIXTURE,
    DAPO_CLIP_FIXTURE,
    GRADED_FIXTURE,
    LOW_CLIP_FIXTURE,
    TRANSLATED_FIXTURE,
    ZERO_MASK_FIXTURE,
)
from .verl_adapter import (
    CORE_ALGOS_SHA256,
    METADATA_SHA256,
    TORCH_FUNCTIONAL_SHA256,
    TRANSFORMERS_VERSION,
    VERL_VERSION,
    VerlUnsupportedObjective,
    evaluate_fixture,
    evaluate_intended_fixture,
    verl_trace,
)
import pytest


class VerlAdapterTests(unittest.TestCase):
    def test_runtime_and_source_provenance_are_exactly_pinned(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        provenance = result.provenance
        assert provenance.verl_version == VERL_VERSION
        assert provenance.transformers_version == TRANSFORMERS_VERSION
        assert provenance.core_algos_sha256 == CORE_ALGOS_SHA256
        assert provenance.torch_functional_sha256 == TORCH_FUNCTIONAL_SHA256
        assert provenance.metadata_sha256 == METADATA_SHA256
        assert provenance.device == "cpu"
        assert "site-packages/verl/" in provenance.core_algos_source

    def test_grpo_trace_maps_every_contract_field(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        assert result.actual is not None
        trace = result.actual
        assert trace is not None
        assert trace.advantages.shape == torch.Size([4])
        assert trace.ratios.shape == torch.Size([4, 2])
        assert trace.mask.shape == torch.Size([4, 2])
        assert trace.surrogate.shape == torch.Size([4, 2])
        assert trace.loss.shape == torch.Size([])
        assert trace.gradient.shape == torch.Size([8])
        assert torch.isfinite(trace.gradient).all()

    def test_native_grpo_mismatch_is_material_and_explained(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        by_field = {field.field: field for field in result.fields}
        assert result.verdict == "MATERIAL_DIFFERENCE"
        assert not by_field["advantages"].agrees
        assert by_field["ratios"].agrees
        assert by_field["mask"].agrees
        assert any("sample_std + 1e-6" in note for note in result.formula_notes)
        assert any("global masked-token mean" in note for note in result.formula_notes)

    def test_group_mapping_is_stable_and_preserves_pair_signs(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        assert result.actual is not None
        assert result.config is not None
        assert result.config.group_id_type == "stable_python_string"
        assert result.actual.advantages[0] < 0
        assert result.actual.advantages[1] > 0
        assert result.actual.advantages[2] < 0
        assert result.actual.advantages[3] > 0

    def test_zero_mask_row_exposes_upstream_advantage_ordering(self) -> None:
        result = evaluate_fixture(ZERO_MASK_FIXTURE, "grpo")
        by_field = {field.field: field for field in result.fields}
        assert not by_field["advantages"].agrees
        assert result.verdict == "MATERIAL_DIFFERENCE"

    def test_unsupported_native_arms_are_not_tested(self) -> None:
        for arm in ("dapo", "gspo", "drgrpo", "aero"):
            with self.subTest(arm=arm):
                result = evaluate_fixture(BASE_FIXTURE, arm)
                assert result.verdict == "NOT_TESTED"
                assert arm in (result.not_tested_reason or "")

    def test_low_level_unsupported_arm_fails_closed(self) -> None:
        with pytest.raises(VerlUnsupportedObjective, match="no native verl"):
            verl_trace(
                arm="gspo",
                rewards=BASE_FIXTURE.rewards,
                logps=BASE_FIXTURE.logps,
                old_logps=BASE_FIXTURE.old_logps,
                mask=BASE_FIXTURE.mask,
            )

    def test_intended_adapter_conforms_for_every_objective_arm(self) -> None:
        fixtures = {
            "grpo": BASE_FIXTURE,
            "dapo": DAPO_CLIP_FIXTURE,
            "gspo": BASE_FIXTURE,
            "drgrpo": BASE_FIXTURE,
            "aero": AERO_POSTERIOR_FIXTURE,
        }
        for arm, fixture in fixtures.items():
            with self.subTest(arm=arm):
                result = evaluate_intended_fixture(fixture, arm)
                assert result.verdict == "PASS"
                assert result.actual_semantics == "intended_verl_s1_adapter"
                assert all(field.agrees for field in result.fields)

    def test_intended_adapter_preserves_selected_completion_mapping(self) -> None:
        selected = (0, 1, 3)
        result = evaluate_intended_fixture(BASE_FIXTURE, "grpo", selected_indices=selected)
        assert result.verdict == "PASS"
        assert result.actual is not None
        assert result.actual.selected_indices == selected
        assert torch.equal(result.actual.mask[2], torch.zeros(2, dtype=torch.float64))

    def test_intended_adapter_conforms_on_reward_and_clip_edge_cases(self) -> None:
        cases = (
            ("grpo", ALL_WRONG_FIXTURE),
            ("grpo", ALL_CORRECT_FIXTURE),
            ("grpo", GRADED_FIXTURE),
            ("grpo", TRANSLATED_FIXTURE),
            ("grpo", LOW_CLIP_FIXTURE),
            ("dapo", LOW_CLIP_FIXTURE),
            ("gspo", ZERO_MASK_FIXTURE),
            ("drgrpo", TRANSLATED_FIXTURE),
        )
        for arm, fixture in cases:
            with self.subTest(arm=arm, fixture=fixture.name):
                assert evaluate_intended_fixture(fixture, arm).verdict == "PASS"


if __name__ == "__main__":
    unittest.main()
