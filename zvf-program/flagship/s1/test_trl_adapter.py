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


TRL_AVAILABLE = find_spec("trl") is not None


@unittest.skipUnless(TRL_AVAILABLE, "requires the pinned TRL fixture-test environment")
class TRLAdapterTests(unittest.TestCase):
    def test_runtime_and_source_provenance_are_exactly_pinned(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        provenance = result.provenance
        self.assertEqual(provenance.trl_version, TRL_VERSION)
        self.assertEqual(provenance.transformers_version, TRANSFORMERS_VERSION)
        self.assertEqual(provenance.locked_wheel_sha256, TRL_WHEEL_SHA256)
        self.assertEqual(provenance.trainer_source_sha256, GRPO_TRAINER_SHA256)
        self.assertEqual(provenance.config_source_sha256, GRPO_CONFIG_SHA256)
        self.assertEqual(provenance.device, "cpu")
        self.assertEqual(provenance.exercised_api, "trl.GRPOTrainer._compute_loss")
        self.assertTrue(provenance.advantage_source.endswith("grpo_trainer.py:2133-2155"))

        summary = result.summary()
        self.assertEqual(summary["actual_semantics"], "native_trl_1.2.0")
        self.assertEqual(summary["expected_semantics"], "canonical_s1_reference")
        self.assertEqual(summary["verdict"], "MATERIAL_DIFFERENCE")

    def test_grpo_trace_maps_every_contract_field(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        trace = result.actual
        self.assertEqual(trace.arm, "grpo")
        self.assertEqual(trace.advantages.shape, torch.Size([4]))
        self.assertEqual(trace.ratios.shape, torch.Size([4, 2]))
        self.assertEqual(trace.mask.shape, torch.Size([4, 2]))
        self.assertEqual(trace.surrogate.shape, torch.Size([4, 2]))
        self.assertEqual(trace.loss.shape, torch.Size([]))
        self.assertEqual(trace.gradient.shape, torch.Size([8]))
        self.assertEqual(trace.selected_indices, (0, 1, 2, 3))
        self.assertTrue(torch.isfinite(trace.gradient).all())

    def test_grpo_records_the_advantage_epsilon_mismatch(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        by_field = {field.field: field for field in result.fields}
        self.assertFalse(result.conforms)
        self.assertEqual(result.verdict, "MATERIAL_DIFFERENCE")
        self.assertFalse(by_field["advantages"].agrees)
        self.assertTrue(by_field["ratios"].agrees)
        self.assertTrue(by_field["mask"].agrees)
        self.assertIn("sample_std + 1e-4", result.formula_notes[0])

    def test_dapo_exercises_native_global_token_reduction(self) -> None:
        result = evaluate_fixture(DAPO_CLIP_FIXTURE, "dapo")
        by_field = {field.field: field for field in result.fields}
        self.assertEqual(result.config.loss_type, "dapo")
        self.assertEqual(result.config.epsilon_high, 0.28)
        self.assertFalse(by_field["loss"].agrees)
        self.assertFalse(by_field["gradient"].agrees)
        self.assertTrue(any("global active tokens" in note for note in result.formula_notes))

    def test_gspo_uses_sequence_importance_sampling(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "gspo")
        self.assertEqual(result.config.loss_type, "grpo")
        self.assertEqual(result.config.importance_sampling_level, "sequence")
        self.assertTrue(torch.equal(result.actual.ratios[:, 0], result.actual.ratios[:, 1]))

    def test_drgrpo_exercises_native_constant_reduction(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "drgrpo")
        by_field = {field.field: field for field in result.fields}
        self.assertEqual(result.config.loss_type, "dr_grpo")
        self.assertEqual(result.config.scale_rewards, "none")
        self.assertTrue(by_field["advantages"].agrees)
        self.assertFalse(by_field["loss"].agrees)
        self.assertTrue(any("batch_size * max_completion_length" in note for note in result.formula_notes))

    def test_zero_mask_fixture_preserves_upstream_ordering_mismatch(self) -> None:
        result = evaluate_fixture(ZERO_MASK_FIXTURE, "grpo")
        by_field = {field.field: field for field in result.fields}
        self.assertFalse(by_field["advantages"].agrees)
        self.assertTrue(any("before applying completion masks" in note for note in result.formula_notes))

    def test_aero_is_explicitly_unsupported_by_native_trl(self) -> None:
        with self.assertRaisesRegex(TRLUnsupportedObjective, "no native AERO"):
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
                self.assertEqual(result.verdict, "PASS")
                self.assertEqual(result.actual_semantics, "intended_trl_s1_adapter")
                self.assertTrue(all(field.agrees for field in result.fields))

    def test_intended_adapter_preserves_selected_completion_mapping(self) -> None:
        selected = (0, 1, 3)
        result = evaluate_intended_fixture(BASE_FIXTURE, "grpo", selected_indices=selected)
        self.assertEqual(result.verdict, "PASS")
        self.assertEqual(result.actual.selected_indices, selected)
        self.assertTrue(torch.equal(result.actual.mask[2], torch.zeros(2, dtype=torch.float64)))

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
                self.assertEqual(evaluate_intended_fixture(fixture, arm).verdict, "PASS")


if __name__ == "__main__":
    unittest.main()
