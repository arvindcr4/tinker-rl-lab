from __future__ import annotations

import copy
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from pilot.entropic_gating import (
    EntropicGatingError,
    GivensEntropyGate,
    apply_givens_rotation_pair,
    compute_attention_entropy,
    compute_entropy_density,
    eliminate_noise_components,
    givens_rotation_angle,
)
from pilot.objective import (
    CONDITIONS,
    Condition,
    ConditionTrace,
    GradientDiagnostic,
    ObjectiveContractError,
    condition_loss,
    condition_trace,
    entropic_givens_grpo,
    gradient_diagnostic,
    spectral_legendre_grpo,
)
from pilot.replay import (
    ACTIVE_FILTERED_ROWS,
    FILTERED_CANDIDATE_POOL_SIZE,
    GROUP_SIZE,
    ReplayCandidate,
    ReplayContractError,
    ReplayGroup,
    ReplayLedger,
    balanced_equal_length_group,
    canonical_fingerprint,
    filtered_variable_length_group,
    filtered_variable_length_pool,
    length_cv,
)
from pilot.spectral_attention import (
    LegendreSpectralRouting,
    SpectralAttentionError,
    compute_legendre_polynomials,
    legendre_basis,
    legendre_grid,
    legendre_spectral_projection,
    spectral_pairwise_distance,
)
from pilot.training import (
    ReplayBatch,
    StepReceipt,
    TrainingContractError,
    _bounded_cosine,
    _compare_gradients,
    completion_logps,
    run_replay_step,
)


class TinyLM(torch.nn.Module):
    """Minimal model for replay step verification."""

    def __init__(self, vocab_size: int = 17, hidden_size: int = 12) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def make_objective_fixture(
    *,
    lengths: list[int] | None = None,
    rewards: list[float] | None = None,
    active: list[bool] | None = None,
) -> dict[str, torch.Tensor]:
    row_lengths = lengths if lengths is not None else [4, 6, 8, 8, 12, 16, 20, 24]
    row_rewards = rewards if rewards is not None else [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    row_active = active if active is not None else [True, True, True, False, True, True, False, True]

    width = max(row_lengths)
    mask = torch.zeros((8, width), dtype=torch.float64)
    for index, (length, is_active) in enumerate(zip(row_lengths, row_active, strict=True)):
        if is_active:
            mask[index, :length] = 1.0
    old = torch.zeros((8, width), dtype=torch.float64)
    offsets = torch.linspace(-0.08, 0.08, steps=8, dtype=torch.float64)[:, None]
    positions = torch.linspace(0.2, 1.0, steps=width, dtype=torch.float64)[None, :]
    logps = old + offsets * positions

    return {
        "rewards": torch.tensor(row_rewards, dtype=torch.float64),
        "logps": logps,
        "old_logps": old,
        "completion_mask": mask,
        "active_rows": torch.tensor(row_active, dtype=torch.bool),
    }


def make_replay_batch(model: TinyLM) -> ReplayBatch:
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
        group_fingerprint="a" * 64,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=mask,
        rewards=torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32),
        active_rows=active,
        old_logps=torch.zeros_like(mask),
    )
    with torch.no_grad():
        old_logps = completion_logps(model, provisional)
    return replace(provisional, old_logps=old_logps)


class SpectralGatingGradientReceiptTests(unittest.TestCase):
    """Verification suite for gradient receipt outputs under spectral and entropic GRPO objectives."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = TinyLM()
        self.batch = make_replay_batch(self.model)

    def test_spectral_legendre_receipt_structure_and_norms(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        receipt = run_replay_step(
            model=self.model,
            optimizer=optimizer,
            scheduler=None,
            batch=self.batch,
            condition="spectral_legendre",
            step=1,
        )
        self.assertIsInstance(receipt, StepReceipt)
        self.assertEqual(receipt.step, 1)
        self.assertEqual(receipt.condition, "spectral_legendre")
        self.assertEqual(receipt.active_rows, 6)
        self.assertEqual(receipt.optimizer_update, "applied")
        self.assertTrue(math.isfinite(receipt.selected_loss))
        self.assertTrue(math.isfinite(receipt.intended_loss))
        self.assertTrue(math.isfinite(receipt.native_loss))
        self.assertGreater(receipt.selected_gradient_norm, 0.0)
        self.assertGreater(receipt.intended_gradient_norm, 0.0)
        self.assertGreater(receipt.native_gradient_norm, 0.0)
        self.assertEqual(receipt.gradient_relation, "nonzero")
        self.assertTrue(-1.0 <= receipt.gradient_cosine <= 1.0)
        self.assertGreaterEqual(receipt.gradient_relative_l2, 0.0)

    def test_entropic_givens_receipt_structure_and_norms(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        receipt = run_replay_step(
            model=self.model,
            optimizer=optimizer,
            scheduler=None,
            batch=self.batch,
            condition="entropic_givens",
            step=1,
        )
        self.assertIsInstance(receipt, StepReceipt)
        self.assertEqual(receipt.step, 1)
        self.assertEqual(receipt.condition, "entropic_givens")
        self.assertEqual(receipt.active_rows, 6)
        self.assertEqual(receipt.optimizer_update, "applied")
        self.assertTrue(math.isfinite(receipt.selected_loss))
        self.assertGreater(receipt.selected_gradient_norm, 0.0)
        self.assertEqual(receipt.selected_vs_intended_relation, "nonzero")
        self.assertTrue(-1.0 <= receipt.selected_vs_intended_cosine <= 1.0)
        self.assertGreaterEqual(receipt.selected_vs_intended_relative_l2, 0.0)

    def test_spectral_condition_trace_receipt_output(self) -> None:
        data = make_objective_fixture()
        trace = condition_trace(condition="spectral_legendre", **data)
        self.assertIsInstance(trace, ConditionTrace)
        self.assertEqual(trace.condition, "spectral_legendre")
        self.assertEqual(len(trace.advantages), 8)
        self.assertTrue(math.isfinite(trace.loss))
        self.assertTrue(all(math.isfinite(val) for val in trace.advantages))
        self.assertTrue(all(math.isfinite(val) for val in trace.gradient))
        self.assertEqual(len(trace.active_rows), 6)
        self.assertGreater(trace.active_tokens, 0)

    def test_entropic_givens_condition_trace_receipt_output(self) -> None:
        data = make_objective_fixture()
        trace = condition_trace(condition="entropic_givens", **data)
        self.assertIsInstance(trace, ConditionTrace)
        self.assertEqual(trace.condition, "entropic_givens")
        self.assertEqual(len(trace.advantages), 8)
        self.assertTrue(math.isfinite(trace.loss))
        self.assertTrue(all(math.isfinite(val) for val in trace.advantages))
        self.assertTrue(all(math.isfinite(val) for val in trace.gradient))
        self.assertEqual(len(trace.active_rows), 6)

    def test_gradient_diagnostics_receipt_comparison(self) -> None:
        data = make_objective_fixture()
        intended = condition_trace(condition="intended_full", **data)
        native = condition_trace(condition="native_trl", **data)
        diag = gradient_diagnostic(intended, native)
        self.assertIsInstance(diag, GradientDiagnostic)
        self.assertEqual(diag.relation, "nonzero")
        self.assertTrue(-1.0 <= diag.cosine <= 1.0)
        self.assertGreaterEqual(diag.relative_l2, 0.0)
        self.assertGreater(diag.intended_gradient_norm, 0.0)
        self.assertGreater(diag.native_gradient_norm, 0.0)

    def test_all_six_conditions_executable_and_emit_receipts(self) -> None:
        initial = copy.deepcopy(self.model.state_dict())
        receipts = []
        for cond in CONDITIONS:
            model = TinyLM()
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            receipt = run_replay_step(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                batch=make_replay_batch(model),
                condition=cond,
                step=1,
            )
            receipts.append(receipt)

        self.assertEqual(len(receipts), len(CONDITIONS))
        self.assertEqual({r.condition for r in receipts}, set(CONDITIONS))
        self.assertTrue(all(r.selected_gradient_norm > 0 for r in receipts))


class SpectralReplayHashingTests(unittest.TestCase):
    """Verification suite for deterministic replay hashing and content-addressed fingerprints."""

    def test_canonical_fingerprint_determinism_and_sensitivity(self) -> None:
        payload_a = {"alpha": 1, "beta": [0.1, 0.2], "gamma": "spectral"}
        payload_b = {"gamma": "spectral", "alpha": 1, "beta": [0.1, 0.2]}
        payload_c = {"alpha": 1, "beta": [0.1, 0.200000001], "gamma": "spectral"}

        fp_a = canonical_fingerprint(payload_a)
        fp_b = canonical_fingerprint(payload_b)
        fp_c = canonical_fingerprint(payload_c)

        self.assertEqual(len(fp_a), 64)
        self.assertEqual(fp_a, fp_b)
        self.assertNotEqual(fp_a, fp_c)

    def test_replay_candidate_sha256_hashing(self) -> None:
        candidate = ReplayCandidate.from_tokens(
            candidate_id="c-101",
            token_ids=[10, 20, 30, 40],
            reward=0.95,
        )
        self.assertEqual(candidate.candidate_id, "c-101")
        self.assertEqual(len(candidate.completion_sha256), 64)
        same_candidate = ReplayCandidate.from_tokens(
            candidate_id="c-101",
            token_ids=[10, 20, 30, 40],
            reward=0.95,
        )
        self.assertEqual(candidate.completion_sha256, same_candidate.completion_sha256)

    def test_balanced_group_deterministic_hashing(self) -> None:
        cand_list = [
            ReplayCandidate.from_tokens(
                candidate_id=f"cand-{i}",
                token_ids=[i + 1] * (i + 4),
                reward=float(i % 2),
            )
            for i in range(GROUP_SIZE)
        ]
        g1 = balanced_equal_length_group(cand_list, pad_token_id=0)
        g2 = balanced_equal_length_group(cand_list, pad_token_id=0)
        self.assertEqual(g1.fingerprint, g2.fingerprint)
        self.assertEqual(g1.source_pool_fingerprint, g2.source_pool_fingerprint)
        self.assertEqual(len(g1.fingerprint), 64)
        self.assertEqual(len(g1.active_indices), GROUP_SIZE)

    def test_filtered_group_deterministic_hashing_and_cv(self) -> None:
        cand_list = [
            ReplayCandidate.from_tokens(
                candidate_id=f"cand-{i}",
                token_ids=[i + 1] * (2 ** (i + 1)),
                reward=float(i % 2),
            )
            for i in range(GROUP_SIZE)
        ]
        g1 = filtered_variable_length_group(cand_list, pad_token_id=0)
        g2 = filtered_variable_length_group(cand_list, pad_token_id=0)
        self.assertEqual(g1.fingerprint, g2.fingerprint)
        self.assertEqual(len(g1.active_indices), ACTIVE_FILTERED_ROWS)
        self.assertGreaterEqual(g1.selected_length_cv, 0.35)

    def test_filtered_pool_hashing_and_source_pool_fingerprint(self) -> None:
        pool_list = [
            ReplayCandidate.from_tokens(
                candidate_id=f"pool-{i}",
                token_ids=[i + 1] * (i + 2),
                reward=float(i % 2),
            )
            for i in range(FILTERED_CANDIDATE_POOL_SIZE)
        ]
        g1 = filtered_variable_length_pool(pool_list, pad_token_id=0)
        g2 = filtered_variable_length_pool(pool_list, pad_token_id=0)
        self.assertEqual(g1.fingerprint, g2.fingerprint)
        self.assertEqual(g1.source_pool_fingerprint, g2.source_pool_fingerprint)
        self.assertEqual(len(g1.candidates), GROUP_SIZE)
        self.assertEqual(len(g1.active_indices), ACTIVE_FILTERED_ROWS)

    def test_replay_ledger_fingerprint_and_token_accounting(self) -> None:
        cand_list = [
            ReplayCandidate.from_tokens(
                candidate_id=f"cand-{i}",
                token_ids=[i + 1] * 5,
                reward=1.0,
            )
            for i in range(GROUP_SIZE)
        ]
        group = balanced_equal_length_group(cand_list, pad_token_id=0)
        ledger_1 = ReplayLedger.build([group], rejected_generated_tokens=10, rejected_candidate_count=1)
        ledger_2 = ReplayLedger.build([group], rejected_generated_tokens=10, rejected_candidate_count=1)
        ledger_3 = ReplayLedger.build([group], rejected_generated_tokens=20, rejected_candidate_count=2)

        self.assertEqual(ledger_1.fingerprint, ledger_2.fingerprint)
        self.assertNotEqual(ledger_1.fingerprint, ledger_3.fingerprint)
        self.assertEqual(ledger_1.charged_generated_tokens, 10 + 40)
        self.assertEqual(ledger_3.charged_generated_tokens, 20 + 40)


class SpectralErrorBoundsAndGRPOObjectiveTests(unittest.TestCase):
    """Verification suite for mathematical error bounds and contract invariants of new GRPO objectives."""

    def test_spectral_legendre_objective_zero_scalar_reward_variance(self) -> None:
        data = make_objective_fixture(
            rewards=[0.0] * 8,
            lengths=[2, 4, 8, 12, 16, 20, 24, 28],
            active=[True] * 8,
        )
        base_loss, base_adv = condition_loss(condition="intended_full", **data)
        spec_loss, spec_adv = condition_loss(condition="spectral_legendre", **data)

        self.assertTrue(torch.allclose(base_adv, torch.zeros(8, dtype=torch.float64)))
        self.assertFalse(torch.allclose(spec_adv, torch.zeros(8, dtype=torch.float64)))
        self.assertGreater(float(spec_adv.std()), 0.0)

        trace = condition_trace(condition="spectral_legendre", **data)
        grad_norm = float(torch.linalg.vector_norm(torch.tensor(trace.gradient)))
        self.assertGreater(grad_norm, 0.0)

    def test_entropic_givens_objective_zero_scalar_reward_variance(self) -> None:
        data = make_objective_fixture(
            rewards=[1.0] * 8,
            lengths=[2, 4, 8, 12, 16, 20, 24, 28],
            active=[True] * 8,
        )
        base_loss, base_adv = condition_loss(condition="intended_full", **data)
        givens_loss, givens_adv = condition_loss(condition="entropic_givens", **data)

        self.assertTrue(torch.allclose(base_adv, torch.zeros(8, dtype=torch.float64)))
        self.assertFalse(torch.allclose(givens_adv, torch.zeros(8, dtype=torch.float64)))
        self.assertGreater(float(givens_adv.std()), 0.0)

        trace = condition_trace(condition="entropic_givens", **data)
        grad_norm = float(torch.linalg.vector_norm(torch.tensor(trace.gradient)))
        self.assertGreater(grad_norm, 0.0)

    def test_legendre_spectral_projection_l2_error_bounds(self) -> None:
        c1 = torch.randn(4, 8, 16, dtype=torch.float64)
        c2 = torch.randn(4, 8, 16, dtype=torch.float64)

        d_self = spectral_pairwise_distance(c1, c1)
        self.assertTrue(torch.allclose(d_self, torch.zeros(4, dtype=torch.float64), atol=1e-12))

        d_12 = spectral_pairwise_distance(c1, c2)
        d_21 = spectral_pairwise_distance(c2, c1)
        self.assertTrue(torch.allclose(d_12, d_21, atol=1e-12))
        self.assertTrue((d_12 >= 0.0).all())

        c3 = torch.randn(4, 8, 16, dtype=torch.float64)
        d_13 = spectral_pairwise_distance(c1, c3)
        d_23 = spectral_pairwise_distance(c2, c3)
        sqrt_12 = torch.sqrt(d_12)
        sqrt_23 = torch.sqrt(d_23)
        sqrt_13 = torch.sqrt(d_13)
        self.assertTrue((sqrt_13 <= sqrt_12 + sqrt_23 + 1e-10).all())

    def test_givens_unitary_rotation_exact_norm_preservation(self) -> None:
        x = torch.randn(10, 16, dtype=torch.float64)
        norm_orig = torch.linalg.vector_norm(x, dim=-1)

        theta = torch.tensor(0.7853981633974483, dtype=torch.float64)
        x_rot = apply_givens_rotation_pair(x, i=2, j=5, theta=theta)
        norm_rot = torch.linalg.vector_norm(x_rot, dim=-1)

        self.assertTrue(torch.allclose(norm_orig, norm_rot, atol=1e-12))

    def test_shannon_entropy_density_bounds(self) -> None:
        k = 16
        prob_uniform = torch.full((k,), 1.0 / k, dtype=torch.float64)
        h_max = float(compute_entropy_density(prob_uniform))
        self.assertAlmostEqual(h_max, math.log(k), places=5)

        prob_peaked = torch.zeros(k, dtype=torch.float64)
        prob_peaked[0] = 1.0
        h_min = float(compute_entropy_density(prob_peaked))
        self.assertAlmostEqual(h_min, 0.0, places=5)

    def test_noise_elimination_hard_projection_bound(self) -> None:
        x = torch.randn(5, 12, dtype=torch.float64)
        x_proj, x_rot = eliminate_noise_components(x, n_noise_dims=2)

        self.assertTrue(torch.allclose(x_proj[:, -2:], torch.zeros(5, 2, dtype=torch.float64)))

        norm_orig = torch.linalg.vector_norm(x, dim=-1)
        norm_rot = torch.linalg.vector_norm(x_rot, dim=-1)
        norm_proj = torch.linalg.vector_norm(x_proj, dim=-1)

        self.assertTrue(torch.allclose(norm_orig, norm_rot, atol=1e-12))
        self.assertTrue((norm_proj <= norm_rot + 1e-12).all())

    def test_objective_contract_error_bounds_and_invalid_inputs(self) -> None:
        data = make_objective_fixture()

        bad_rewards = data["rewards"].clone()
        bad_rewards[0] = float("nan")
        with self.assertRaisesRegex(ObjectiveContractError, "must be finite"):
            condition_loss(condition="spectral_legendre", **{**data, "rewards": bad_rewards})

        bad_rows = torch.tensor([True] * 4 + [False] * 4, dtype=torch.bool)
        with self.assertRaisesRegex(ObjectiveContractError, "exactly six or eight rows"):
            condition_loss(condition="entropic_givens", **{**data, "active_rows": bad_rows})

        with self.assertRaisesRegex(ObjectiveContractError, "unknown pilot condition"):
            condition_loss(condition="invalid_condition", **data)


if __name__ == "__main__":
    unittest.main()
