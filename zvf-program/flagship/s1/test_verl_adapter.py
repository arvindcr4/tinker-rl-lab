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


class VerlAdapterTests(unittest.TestCase):
    def test_runtime_and_source_provenance_are_exactly_pinned(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        provenance = result.provenance
        self.assertEqual(provenance.verl_version, VERL_VERSION)
        self.assertEqual(provenance.transformers_version, TRANSFORMERS_VERSION)
        self.assertEqual(provenance.core_algos_sha256, CORE_ALGOS_SHA256)
        self.assertEqual(provenance.torch_functional_sha256, TORCH_FUNCTIONAL_SHA256)
        self.assertEqual(provenance.metadata_sha256, METADATA_SHA256)
        self.assertEqual(provenance.device, "cpu")
        self.assertIn("site-packages/verl/", provenance.core_algos_source)

    def test_grpo_trace_maps_every_contract_field(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        self.assertIsNotNone(result.actual)
        trace = result.actual
        assert trace is not None
        self.assertEqual(trace.advantages.shape, torch.Size([4]))
        self.assertEqual(trace.ratios.shape, torch.Size([4, 2]))
        self.assertEqual(trace.mask.shape, torch.Size([4, 2]))
        self.assertEqual(trace.surrogate.shape, torch.Size([4, 2]))
        self.assertEqual(trace.loss.shape, torch.Size([]))
        self.assertEqual(trace.gradient.shape, torch.Size([8]))
        self.assertTrue(torch.isfinite(trace.gradient).all())

    def test_native_grpo_mismatch_is_material_and_explained(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        by_field = {field.field: field for field in result.fields}
        self.assertEqual(result.verdict, "MATERIAL_DIFFERENCE")
        self.assertFalse(by_field["advantages"].agrees)
        self.assertTrue(by_field["ratios"].agrees)
        self.assertTrue(by_field["mask"].agrees)
        self.assertTrue(any("sample_std + 1e-6" in note for note in result.formula_notes))
        self.assertTrue(any("global masked-token mean" in note for note in result.formula_notes))

    def test_group_mapping_is_stable_and_preserves_pair_signs(self) -> None:
        result = evaluate_fixture(BASE_FIXTURE, "grpo")
        assert result.actual is not None and result.config is not None
        self.assertEqual(result.config.group_id_type, "stable_python_string")
        self.assertLess(result.actual.advantages[0], 0)
        self.assertGreater(result.actual.advantages[1], 0)
        self.assertLess(result.actual.advantages[2], 0)
        self.assertGreater(result.actual.advantages[3], 0)

    def test_zero_mask_row_exposes_upstream_advantage_ordering(self) -> None:
        result = evaluate_fixture(ZERO_MASK_FIXTURE, "grpo")
        by_field = {field.field: field for field in result.fields}
        self.assertFalse(by_field["advantages"].agrees)
        self.assertEqual(result.verdict, "MATERIAL_DIFFERENCE")

    def test_unsupported_native_arms_are_not_tested(self) -> None:
        for arm in ("dapo", "gspo", "drgrpo", "aero"):
            with self.subTest(arm=arm):
                result = evaluate_fixture(BASE_FIXTURE, arm)
                self.assertEqual(result.verdict, "NOT_TESTED")
                self.assertIn(arm, result.not_tested_reason or "")

    def test_low_level_unsupported_arm_fails_closed(self) -> None:
        with self.assertRaisesRegex(VerlUnsupportedObjective, "no native verl"):
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
                self.assertEqual(result.verdict, "PASS")
                self.assertEqual(result.actual_semantics, "intended_verl_s1_adapter")
                self.assertTrue(all(field.agrees for field in result.fields))

    def test_intended_adapter_preserves_selected_completion_mapping(self) -> None:
        selected = (0, 1, 3)
        result = evaluate_intended_fixture(BASE_FIXTURE, "grpo", selected_indices=selected)
        self.assertEqual(result.verdict, "PASS")
        assert result.actual is not None
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
