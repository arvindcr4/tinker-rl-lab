from __future__ import annotations

import unittest

from pilot.replay import (
    ACTIVE_FILTERED_ROWS,
    FILTERED_CANDIDATE_POOL_SIZE,
    GROUP_SIZE,
    ReplayCandidate,
    ReplayContractError,
    ReplayLedger,
    balanced_equal_length_group,
    filtered_variable_length_group,
    filtered_variable_length_pool,
    length_cv,
)


def candidates(lengths: list[int]) -> list[ReplayCandidate]:
    return [
        ReplayCandidate.from_tokens(
            candidate_id=f"candidate-{index}",
            token_ids=[index + 1] * length,
            reward=float(index % 2),
        )
        for index, length in enumerate(lengths)
    ]


class ReplayContractTests(unittest.TestCase):
    def test_balanced_regime_has_eight_equal_active_lengths(self) -> None:
        group = balanced_equal_length_group(
            candidates([2, 3, 4, 5, 6, 7, 8, 9]), pad_token_id=0
        )
        active_lengths = [sum(mask) for mask in group.optimization_masks]
        self.assertEqual(len(group.active_indices), GROUP_SIZE)
        self.assertEqual(len(set(active_lengths)), 1)
        self.assertEqual(active_lengths, [9] * GROUP_SIZE)
        self.assertEqual(group.charged_generated_tokens, 44)
        self.assertEqual(group.active_optimization_tokens, 72)

    def test_filtered_regime_selects_exactly_six_rows_at_required_cv(self) -> None:
        group = filtered_variable_length_group(
            candidates([2, 3, 4, 8, 16, 32, 64, 128]), pad_token_id=0
        )
        self.assertEqual(len(group.active_indices), ACTIVE_FILTERED_ROWS)
        self.assertGreaterEqual(group.selected_length_cv, 0.35)
        self.assertEqual(sum(sum(mask) > 0 for mask in group.optimization_masks), 6)
        selected_lengths = [
            len(group.candidates[index].token_ids) for index in group.active_indices
        ]
        self.assertAlmostEqual(group.selected_length_cv, length_cv(selected_lengths))

    def test_filtered_selection_is_deterministic_under_a_tie(self) -> None:
        first = filtered_variable_length_group(
            candidates([2, 2, 2, 2, 20, 20, 20, 20]), pad_token_id=0
        )
        second = filtered_variable_length_group(
            candidates([2, 2, 2, 2, 20, 20, 20, 20]), pad_token_id=0
        )
        self.assertEqual(first.active_indices, second.active_indices)
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_filtered_pool_charges_all_sixteen_candidates_but_optimizes_eight(self) -> None:
        pool = candidates([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32, 48, 64, 128])
        group = filtered_variable_length_pool(pool, pad_token_id=0)
        self.assertEqual(len(pool), FILTERED_CANDIDATE_POOL_SIZE)
        self.assertEqual(len(group.candidates), GROUP_SIZE)
        self.assertEqual(len(group.active_indices), ACTIVE_FILTERED_ROWS)
        self.assertEqual(group.charged_generated_tokens, sum(len(row.token_ids) for row in pool))
        self.assertEqual(
            group.source_pool_fingerprint,
            filtered_variable_length_pool(pool, pad_token_id=0).source_pool_fingerprint,
        )

    def test_filtered_pool_requires_sixteen_unique_candidates(self) -> None:
        with self.assertRaisesRegex(ReplayContractError, "exactly 16"):
            filtered_variable_length_pool(candidates([2] * 8), pad_token_id=0)
        pool = candidates([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32, 48, 64, 128])
        pool[-1] = pool[0]
        with self.assertRaisesRegex(ReplayContractError, "unique"):
            filtered_variable_length_pool(pool, pad_token_id=0)

    def test_filtered_regime_fails_closed_when_length_cv_is_too_low(self) -> None:
        with self.assertRaisesRegex(ReplayContractError, "below 0.350000"):
            filtered_variable_length_group(candidates([10] * GROUP_SIZE), pad_token_id=0)

    def test_group_requires_eight_unique_nonempty_candidates(self) -> None:
        with self.assertRaisesRegex(ReplayContractError, "exactly 8"):
            balanced_equal_length_group(candidates([2] * 7), pad_token_id=0)
        duplicated = candidates([2] * GROUP_SIZE)
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(ReplayContractError, "unique"):
            balanced_equal_length_group(duplicated, pad_token_id=0)
        with self.assertRaisesRegex(ReplayContractError, "cannot be empty"):
            ReplayCandidate.from_tokens(candidate_id="empty", token_ids=[], reward=0.0)

    def test_ledger_charges_rejected_generation_and_is_content_addressed(self) -> None:
        group = balanced_equal_length_group(candidates([2] * GROUP_SIZE), pad_token_id=0)
        ledger = ReplayLedger.build(
            [group], rejected_generated_tokens=17, rejected_candidate_count=2
        )
        self.assertEqual(ledger.charged_generated_tokens, 33)
        changed = ReplayLedger.build(
            [group], rejected_generated_tokens=18, rejected_candidate_count=2
        )
        self.assertNotEqual(ledger.fingerprint, changed.fingerprint)


if __name__ == "__main__":
    unittest.main()
