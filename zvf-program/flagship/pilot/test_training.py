from __future__ import annotations

import copy
import unittest
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch

from pilot.training import (
    ReplayBatch,
    TrainingContractError,
    _bounded_cosine,
    _compare_gradients,
    _finite_norm,
    _gradients,
    completion_logps,
    run_replay_step,
)


class TinyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 17, hidden_size: int = 12) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def batch(model: TinyLM) -> ReplayBatch:
    prompt_ids = torch.tensor([[1, 2, 3]] * 8, dtype=torch.long)
    prompt_mask = torch.ones_like(prompt_ids)
    completion_ids = torch.tensor(
        [
            [4, 5, 6, 0, 0, 0],
            [4, 7, 8, 9, 0, 0],
            [5, 6, 7, 8, 9, 0],
            [4, 4, 0, 0, 0, 0],
            [9, 8, 7, 6, 5, 4],
            [6, 6, 6, 6, 0, 0],
            [8, 8, 8, 0, 0, 0],
            [7, 7, 7, 7, 7, 0],
        ],
        dtype=torch.long,
    )
    active = torch.tensor([True, True, True, False, True, True, False, True])
    mask = torch.zeros_like(completion_ids, dtype=torch.float32)
    lengths = [3, 4, 5, 0, 6, 4, 0, 5]
    for index, length in enumerate(lengths):
        mask[index, :length] = 1.0
    provisional = ReplayBatch(
        group_fingerprint="f" * 64,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=mask,
        rewards=torch.tensor([1, 0, 1, 0, 0, 1, 1, 0], dtype=torch.float32),
        active_rows=active,
        old_logps=torch.zeros_like(mask),
    )
    with torch.no_grad():
        old_logps = completion_logps(model, provisional)
    return replace(provisional, old_logps=old_logps)


class ReplayTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyLM()
        self.batch = batch(self.model)

    def test_completion_logps_have_expected_shape_and_are_finite(self) -> None:
        values = completion_logps(self.model, self.batch)
        self.assertEqual(values.shape, self.batch.completion_ids.shape)
        self.assertTrue(torch.isfinite(values).all())

    def test_completion_logps_pin_deterministic_math_attention_backend(self) -> None:
        from torch.nn.attention import SDPBackend

        recorded = []

        @contextmanager
        def recorder(backends):
            recorded.append(list(backends))
            yield

        with mock.patch("pilot.training.sdpa_kernel", recorder):
            values = completion_logps(self.model, self.batch)
        self.assertEqual(recorded, [[SDPBackend.MATH]])
        self.assertTrue(torch.isfinite(values).all())

    def test_checkpoint_backward_pins_deterministic_math_attention_backend(self) -> None:
        from torch.nn.attention import SDPBackend

        recorded = []

        @contextmanager
        def recorder(backends):
            recorded.append(list(backends))
            yield

        parameters = tuple(self.model.parameters())
        loss = sum(parameter.square().sum() for parameter in parameters)
        with mock.patch("pilot.training.sdpa_kernel", recorder):
            gradients = _gradients(loss, parameters, retain_graph=False)
        self.assertEqual(recorded, [[SDPBackend.MATH]])
        self.assertEqual(len(gradients), len(parameters))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_low_precision_identical_vector_cosine_is_bounded(self) -> None:
        low_precision = torch.linspace(0.1, 1.0, 1000, dtype=torch.float16)
        vector = low_precision.double()
        norm = _finite_norm(vector, label="test")
        cosine = _bounded_cosine(
            vector,
            vector,
            left_norm=norm,
            right_norm=norm,
            label="test-identical",
        )
        self.assertAlmostEqual(cosine, 1.0, places=15)
        self.assertTrue(-1.0 <= cosine <= 1.0)

    def test_exact_identical_vectors_bypass_roundoff_prone_cosine_reduction(self) -> None:
        vector = torch.linspace(-1.0, 1.0, 100_003, dtype=torch.float64)
        norm = _finite_norm(vector, label="test")
        with mock.patch(
            "pilot.training._bounded_cosine",
            side_effect=AssertionError("exact identity must not use a reduction"),
        ):
            comparison = _compare_gradients(
                vector,
                vector.clone(),
                left_norm=norm,
                right_norm=norm,
                left_zero_relation="left_zero",
                right_zero_relation="right_zero",
                label="test-identical",
            )
        self.assertEqual(comparison.relation, "nonzero")
        self.assertEqual(comparison.cosine, 1.0)
        self.assertEqual(comparison.relative_l2, 0.0)

    def test_replay_step_updates_model_and_emits_complete_receipt(self) -> None:
        before = copy.deepcopy(self.model.state_dict())
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=2
        )
        receipt = run_replay_step(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            batch=self.batch,
            condition="intended_full",
            step=1,
        )
        self.assertEqual(receipt.step, 1)
        self.assertEqual(receipt.active_rows, 6)
        self.assertEqual(receipt.gradient_relation, "nonzero")
        self.assertTrue(-1.0 <= receipt.gradient_cosine <= 1.0)
        self.assertGreater(receipt.gradient_relative_l2, 0.0)
        self.assertEqual(receipt.selected_vs_intended_relation, "nonzero")
        self.assertAlmostEqual(receipt.selected_vs_intended_cosine, 1.0, places=6)
        self.assertAlmostEqual(receipt.selected_vs_intended_relative_l2, 0.0, places=7)
        self.assertEqual(receipt.optimizer_update, "applied")
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in self.model.state_dict().items()
            )
        )

    def test_each_condition_can_apply_a_finite_update_from_identical_weights(self) -> None:
        initial = copy.deepcopy(self.model.state_dict())
        receipts = []
        for condition in ("intended_full", "native_trl", "epsilon_only", "reduction_only"):
            model = TinyLM()
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            receipts.append(
                run_replay_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    batch=batch(model),
                    condition=condition,
                    step=1,
                )
            )
        self.assertEqual(
            {receipt.condition for receipt in receipts},
            {"intended_full", "native_trl", "epsilon_only", "reduction_only"},
        )
        self.assertTrue(all(receipt.selected_gradient_norm > 0 for receipt in receipts))

    def test_joint_zero_receipt_is_explicit_and_optimizer_is_a_true_no_op(self) -> None:
        zero_batch = replace(self.batch, rewards=torch.zeros(8, dtype=torch.float32))
        before = copy.deepcopy(self.model.state_dict())
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=2
        )
        receipt = run_replay_step(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            batch=zero_batch,
            condition="intended_full",
            step=1,
        )
        self.assertEqual(receipt.gradient_relation, "joint_zero")
        self.assertIsNone(receipt.gradient_cosine)
        self.assertIsNone(receipt.gradient_relative_l2)
        self.assertEqual(receipt.selected_vs_intended_relation, "joint_zero")
        self.assertIsNone(receipt.selected_vs_intended_cosine)
        self.assertIsNone(receipt.selected_vs_intended_relative_l2)
        self.assertEqual(receipt.intended_gradient_norm, 0.0)
        self.assertEqual(receipt.native_gradient_norm, 0.0)
        self.assertEqual(receipt.selected_gradient_norm, 0.0)
        self.assertEqual(receipt.optimizer_update, "no_op_zero_gradient")
        self.assertEqual(optimizer.state, {})
        self.assertEqual(scheduler.last_epoch, 1)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)

    def test_batch_shape_and_step_fail_closed(self) -> None:
        malformed = replace(self.batch, active_rows=torch.ones(7, dtype=torch.bool))
        with self.assertRaisesRegex(TrainingContractError, "eight entries"):
            completion_logps(self.model, malformed)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        with self.assertRaisesRegex(TrainingContractError, "step must be positive"):
            run_replay_step(
                model=self.model,
                optimizer=optimizer,
                scheduler=None,
                batch=self.batch,
                condition="intended_full",
                step=0,
            )

    def test_phase_context_exposes_separate_flop_accounting_boundaries(self) -> None:
        observed: list[str] = []

        @contextmanager
        def phase(name: str):
            observed.append(name)
            yield

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        run_replay_step(
            model=self.model,
            optimizer=optimizer,
            scheduler=None,
            batch=self.batch,
            condition="reduction_only",
            step=1,
            phase_context=phase,
        )
        self.assertEqual(
            observed,
            ["policy_forward", "optimizer_backward", "diagnostic_backward", "optimizer_step"],
        )


if __name__ == "__main__":
    unittest.main()
