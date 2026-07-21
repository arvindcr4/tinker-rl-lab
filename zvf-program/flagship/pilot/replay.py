from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any, Iterable, Sequence


GROUP_SIZE = 8
FILTERED_CANDIDATE_POOL_SIZE = 16
ACTIVE_FILTERED_ROWS = 6
FILTERED_MIN_LENGTH_CV = 0.35


class ReplayContractError(RuntimeError):
    """A generated group cannot satisfy the frozen replay contract."""


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def length_cv(lengths: Sequence[int]) -> float:
    if not lengths or any(length <= 0 for length in lengths):
        raise ReplayContractError("completion lengths must be positive")
    mean = fmean(lengths)
    return 0.0 if len(lengths) == 1 else pstdev(lengths) / mean


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    candidate_id: str
    token_ids: tuple[int, ...]
    reward: float
    completion_sha256: str

    @classmethod
    def from_tokens(
        cls,
        *,
        candidate_id: str,
        token_ids: Sequence[int],
        reward: float,
    ) -> "ReplayCandidate":
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ReplayContractError("a replay completion cannot be empty")
        if not math.isfinite(float(reward)):
            raise ReplayContractError("a replay reward must be finite")
        completion_sha256 = hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            candidate_id=candidate_id,
            token_ids=tokens,
            reward=float(reward),
            completion_sha256=completion_sha256,
        )


@dataclass(frozen=True, slots=True)
class ReplayGroup:
    regime: str
    candidates: tuple[ReplayCandidate, ...]
    active_indices: tuple[int, ...]
    padded_token_ids: tuple[tuple[int, ...], ...]
    optimization_masks: tuple[tuple[int, ...], ...]
    selected_length_cv: float
    charged_generated_tokens: int
    active_optimization_tokens: int
    padded_optimization_tokens: int
    source_pool_fingerprint: str
    fingerprint: str

    def as_record(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "candidate_ids": [candidate.candidate_id for candidate in self.candidates],
            "completion_sha256": [candidate.completion_sha256 for candidate in self.candidates],
            "rewards": [candidate.reward for candidate in self.candidates],
            "raw_lengths": [len(candidate.token_ids) for candidate in self.candidates],
            "active_indices": list(self.active_indices),
            "selected_length_cv": self.selected_length_cv,
            "charged_generated_tokens": self.charged_generated_tokens,
            "active_optimization_tokens": self.active_optimization_tokens,
            "padded_optimization_tokens": self.padded_optimization_tokens,
            "source_pool_fingerprint": self.source_pool_fingerprint,
            "fingerprint": self.fingerprint,
        }


def _validate_candidates(candidates: Sequence[ReplayCandidate]) -> tuple[ReplayCandidate, ...]:
    frozen = tuple(candidates)
    if len(frozen) != GROUP_SIZE:
        raise ReplayContractError(f"a replay group requires exactly {GROUP_SIZE} candidates")
    ids = [candidate.candidate_id for candidate in frozen]
    if len(set(ids)) != GROUP_SIZE:
        raise ReplayContractError("candidate IDs must be unique within a replay group")
    return frozen


def _pad(
    candidates: Sequence[ReplayCandidate],
    active_indices: Iterable[int],
    *,
    pad_token_id: int,
    equal_active_length: bool,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    active = set(active_indices)
    width = max(len(candidate.token_ids) for candidate in candidates)
    padded: list[tuple[int, ...]] = []
    masks: list[tuple[int, ...]] = []
    for index, candidate in enumerate(candidates):
        missing = width - len(candidate.token_ids)
        padded.append(candidate.token_ids + (int(pad_token_id),) * missing)
        if index not in active:
            masks.append((0,) * width)
        elif equal_active_length:
            masks.append((1,) * width)
        else:
            masks.append((1,) * len(candidate.token_ids) + (0,) * missing)
    return tuple(padded), tuple(masks)


def _build_group(
    *,
    regime: str,
    candidates: tuple[ReplayCandidate, ...],
    active_indices: tuple[int, ...],
    pad_token_id: int,
    equal_active_length: bool,
    selected_cv: float,
    source_pool_fingerprint: str,
    charged_generated_tokens: int,
) -> ReplayGroup:
    padded, masks = _pad(
        candidates,
        active_indices,
        pad_token_id=pad_token_id,
        equal_active_length=equal_active_length,
    )
    record = {
        "regime": regime,
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "token_ids": [list(candidate.token_ids) for candidate in candidates],
        "rewards": [candidate.reward for candidate in candidates],
        "active_indices": list(active_indices),
        "padded_token_ids": [list(row) for row in padded],
        "optimization_masks": [list(row) for row in masks],
        "selected_length_cv": selected_cv,
        "source_pool_fingerprint": source_pool_fingerprint,
        "charged_generated_tokens": charged_generated_tokens,
    }
    return ReplayGroup(
        regime=regime,
        candidates=candidates,
        active_indices=active_indices,
        padded_token_ids=padded,
        optimization_masks=masks,
        selected_length_cv=selected_cv,
        charged_generated_tokens=charged_generated_tokens,
        active_optimization_tokens=sum(sum(row) for row in masks),
        padded_optimization_tokens=len(candidates) * len(padded[0]),
        source_pool_fingerprint=source_pool_fingerprint,
        fingerprint=canonical_fingerprint(record),
    )


def _pool_fingerprint(candidates: Sequence[ReplayCandidate]) -> str:
    return canonical_fingerprint(
        [
            {
                "candidate_id": candidate.candidate_id,
                "completion_sha256": candidate.completion_sha256,
                "reward": candidate.reward,
                "length": len(candidate.token_ids),
            }
            for candidate in candidates
        ]
    )


def balanced_equal_length_group(
    candidates: Sequence[ReplayCandidate], *, pad_token_id: int
) -> ReplayGroup:
    """Keep all rows and charge right-padding as active optimization tokens."""
    frozen = _validate_candidates(candidates)
    source_pool_fingerprint = _pool_fingerprint(frozen)
    return _build_group(
        regime="balanced_equal_length",
        candidates=frozen,
        active_indices=tuple(range(GROUP_SIZE)),
        pad_token_id=pad_token_id,
        equal_active_length=True,
        selected_cv=0.0,
        source_pool_fingerprint=source_pool_fingerprint,
        charged_generated_tokens=sum(len(candidate.token_ids) for candidate in frozen),
    )


def filtered_variable_length_group(
    candidates: Sequence[ReplayCandidate],
    *,
    pad_token_id: int,
    minimum_cv: float = FILTERED_MIN_LENGTH_CV,
) -> ReplayGroup:
    """Select the unique lexicographically first six-row subset with maximal CV."""
    frozen = _validate_candidates(candidates)
    lengths = tuple(len(candidate.token_ids) for candidate in frozen)
    scored = [
        (length_cv([lengths[index] for index in indices]), indices)
        for indices in itertools.combinations(range(GROUP_SIZE), ACTIVE_FILTERED_ROWS)
    ]
    best_cv = max(score for score, _ in scored)
    best_indices = min(indices for score, indices in scored if math.isclose(score, best_cv))
    if best_cv < minimum_cv:
        raise ReplayContractError(
            f"filtered group maximum selected-row length CV {best_cv:.6f} is below {minimum_cv:.6f}"
        )
    return _build_group(
        regime="filtered_variable_length",
        candidates=frozen,
        active_indices=best_indices,
        pad_token_id=pad_token_id,
        equal_active_length=False,
        selected_cv=best_cv,
        source_pool_fingerprint=_pool_fingerprint(frozen),
        charged_generated_tokens=sum(len(candidate.token_ids) for candidate in frozen),
    )


def filtered_variable_length_pool(
    candidates: Sequence[ReplayCandidate],
    *,
    pad_token_id: int,
    minimum_cv: float = FILTERED_MIN_LENGTH_CV,
) -> ReplayGroup:
    """Choose six active and two inactive rows from a fixed 16-candidate pool."""
    pool = tuple(candidates)
    if len(pool) != FILTERED_CANDIDATE_POOL_SIZE:
        raise ReplayContractError(
            f"filtered replay generation requires exactly {FILTERED_CANDIDATE_POOL_SIZE} candidates"
        )
    ids = [candidate.candidate_id for candidate in pool]
    if len(set(ids)) != FILTERED_CANDIDATE_POOL_SIZE:
        raise ReplayContractError("candidate IDs must be unique within a filtered source pool")

    lengths = tuple(len(candidate.token_ids) for candidate in pool)
    scored = [
        (length_cv([lengths[index] for index in indices]), indices)
        for indices in itertools.combinations(range(FILTERED_CANDIDATE_POOL_SIZE), ACTIVE_FILTERED_ROWS)
    ]
    best_cv = max(score for score, _ in scored)
    active_pool_indices = min(
        indices for score, indices in scored if math.isclose(score, best_cv)
    )
    if best_cv < minimum_cv:
        raise ReplayContractError(
            f"filtered pool maximum selected-row length CV {best_cv:.6f} is below {minimum_cv:.6f}"
        )
    inactive_pool_indices = tuple(
        index for index in range(FILTERED_CANDIDATE_POOL_SIZE) if index not in active_pool_indices
    )[: GROUP_SIZE - ACTIVE_FILTERED_ROWS]
    chosen_pool_indices = tuple(sorted(active_pool_indices + inactive_pool_indices))
    chosen = tuple(pool[index] for index in chosen_pool_indices)
    active_group_indices = tuple(
        group_index
        for group_index, pool_index in enumerate(chosen_pool_indices)
        if pool_index in active_pool_indices
    )
    return _build_group(
        regime="filtered_variable_length",
        candidates=chosen,
        active_indices=active_group_indices,
        pad_token_id=pad_token_id,
        equal_active_length=False,
        selected_cv=best_cv,
        source_pool_fingerprint=_pool_fingerprint(pool),
        charged_generated_tokens=sum(len(candidate.token_ids) for candidate in pool),
    )


@dataclass(frozen=True, slots=True)
class ReplayLedger:
    accepted_groups: tuple[ReplayGroup, ...]
    rejected_generated_tokens: int
    rejected_candidate_count: int
    fingerprint: str

    @property
    def charged_generated_tokens(self) -> int:
        return self.rejected_generated_tokens + sum(
            group.charged_generated_tokens for group in self.accepted_groups
        )

    @classmethod
    def build(
        cls,
        groups: Sequence[ReplayGroup],
        *,
        rejected_generated_tokens: int = 0,
        rejected_candidate_count: int = 0,
    ) -> "ReplayLedger":
        frozen = tuple(groups)
        if rejected_generated_tokens < 0 or rejected_candidate_count < 0:
            raise ReplayContractError("rejected replay accounting cannot be negative")
        record = {
            "group_fingerprints": [group.fingerprint for group in frozen],
            "rejected_generated_tokens": rejected_generated_tokens,
            "rejected_candidate_count": rejected_candidate_count,
        }
        return cls(
            accepted_groups=frozen,
            rejected_generated_tokens=rejected_generated_tokens,
            rejected_candidate_count=rejected_candidate_count,
            fingerprint=canonical_fingerprint(record),
        )
