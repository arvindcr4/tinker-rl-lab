from __future__ import annotations

import unittest
from importlib.util import find_spec

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
from .trl_adapter import (
    GRPO_CONFIG_SHA256,
    GRPO_TRAINER_SHA256,
    TRANSFORMERS_VERSION,
    TRL_VERSION,
    TRL_WHEEL_SHA256,
    TRLUnsupportedObjective,
    evaluate_fixture,
    evaluate_intended_fixture,
    trl_trace,
)
import pytest


TRL_AVAILABLE = find_spec("trl") is not None


@unittest.skipUnless(TRL_AVAILABLE, "requires the pinned TRL fixture-test environment")
class TRLAdapterTests(unittest.TestCase):
    def test_runtime_and_source_provenance_are_exactly_pinned(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        provenance = result.provenance
        assert provenance.trl_version == TRL_VERSION
        assert provenance.transformers_version == TRANSFORMERS_VERSION
        assert provenance.locked_wheel_sha256 == TRL_WHEEL_SHA256
        assert provenance.trainer_source_sha256 == GRPO_TRAINER_SHA256
        assert provenance.config_source_sha256 == GRPO_CONFIG_SHA256
        assert provenance.device == "cpu"
        assert provenance.exercised_api == "trl.GRPOTrainer._compute_loss"
        assert provenance.advantage_source.endswith("grpo_trainer.py:2133-2155")

        summary = result.summary()
        assert summary["actual_semantics"] == "native_trl_1.2.0"
        assert summary["expected_semantics"] == "canonical_s1_reference"
        assert summary["verdict"] == "MATERIAL_DIFFERENCE"

    def test_grpo_trace_maps_every_contract_field(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        trace = result.actual
        assert trace.arm == "grpo"
        assert trace.advantages.shape == torch.Size([4])
        assert trace.ratios.shape == torch.Size([4, 2])
        assert trace.mask.shape == torch.Size([4, 2])
        assert trace.surrogate.shape == torch.Size([4, 2])
        assert trace.loss.shape == torch.Size([])
        assert trace.gradient.shape == torch.Size([8])
        assert trace.selected_indices == (0, 1, 2, 3)
        assert torch.isfinite(trace.gradient).all()

    def test_grpo_records_the_advantage_epsilon_mismatch(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        by_field = {field.field: field for field in result.fields}
        assert not result.conforms
        assert result.verdict == "MATERIAL_DIFFERENCE"
        assert not by_field["advantages"].agrees
        assert by_field["ratios"].agrees
        assert by_field["mask"].agrees
        assert "sample_std + 1e-4" in result.formula_notes[0]

    def test_dapo_exercises_native_global_token_reduction(self) -> None:
        result = evaluate_fixture(DAPO_CLIP_FIXTURE, "dapo")
        by_field = {field.field: field for field in result.fields}
        assert result.config.loss_type == "dapo"
        assert result.config.epsilon_high == 0.28
        assert not by_field["loss"].agrees
        assert not by_field["gradient"].agrees
        assert any("global active tokens" in note for note in result.formula_notes)

    def test_gspo_uses_sequence_importance_sampling(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "gspo")
        assert result.config.loss_type == "grpo"
        assert result.config.importance_sampling_level == "sequence"
        assert torch.equal(result.actual.ratios[:, 0], result.actual.ratios[:, 1])

    def test_drgrpo_exercises_native_constant_reduction(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "drgrpo")
        by_field = {field.field: field for field in result.fields}
        assert result.config.loss_type == "dr_grpo"
        assert result.config.scale_rewards == "none"
        assert by_field["advantages"].agrees
        assert not by_field["loss"].agrees
        assert any("batch_size * max_completion_length" in note for note in result.formula_notes)

    def test_zero_mask_fixture_preserves_upstream_ordering_mismatch(self) -> None:
        result = evaluate_fixture(ZERO_MASK_FIXTURE, "grpo")
        by_field = {field.field: field for field in result.fields}
        assert not by_field["advantages"].agrees
        assert any("before applying completion masks" in note for note in result.formula_notes)

    def test_aero_is_explicitly_unsupported_by_native_trl(self) -> None:
        with pytest.raises(TRLUnsupportedObjective, match="no native AERO"):
            trl_trace(
                arm="aero",
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
                assert result.actual_semantics == "intended_trl_s1_adapter"
                assert all(field.agrees for field in result.fields)

    def test_intended_adapter_preserves_selected_completion_mapping(self) -> None:
        selected = (0, 1, 3)
        result = evaluate_intended_fixture(BASE_FIXTURE, "grpo", selected_indices=selected)
        assert result.verdict == "PASS"
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
