"""Frozen synthetic inputs shared by the S1 reference and stack adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ObjectiveFixture:
    name: str
    rewards: tuple[tuple[float, ...], ...]
    logps: tuple[tuple[float, ...], ...]
    old_logps: tuple[tuple[float, ...], ...]
    mask: tuple[tuple[float, ...], ...]
    aero_successes: tuple[int, ...] | None = None
    aero_observations: tuple[int, ...] | None = None

    def with_updates(self, **changes: object) -> "ObjectiveFixture":
        return replace(self, **changes)


BASE_FIXTURE = ObjectiveFixture(
    name="base",
    rewards=((0.0, 1.0), (0.0, 1.0)),
    logps=((-0.1, -0.2), (-0.2, -0.4), (-0.3, -0.5), (-0.4, -0.6)),
    old_logps=((-0.2, -0.2), (-0.2, -0.3), (-0.3, -0.4), (-0.5, -0.6)),
    mask=((1.0, 1.0), (1.0, 0.0), (1.0, 1.0), (1.0, 1.0)),
)

DAPO_CLIP_FIXTURE = BASE_FIXTURE.with_updates(
    name="dapo_clip",
    logps=((-0.2, -0.2), (0.2, -0.4), (-0.3, -0.5), (-0.4, -0.6)),
)

ZERO_MASK_FIXTURE = BASE_FIXTURE.with_updates(
    name="zero_mask",
    mask=((1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0)),
)

AERO_POSTERIOR_FIXTURE = BASE_FIXTURE.with_updates(
    name="aero_posterior",
    rewards=((0.0, 0.0), (1.0, 1.0)),
    aero_successes=(1, 9),
    aero_observations=(10, 10),
)

ALL_WRONG_FIXTURE = BASE_FIXTURE.with_updates(
    name="all_wrong",
    rewards=((0.0, 0.0), (0.0, 0.0)),
)

ALL_CORRECT_FIXTURE = BASE_FIXTURE.with_updates(
    name="all_correct",
    rewards=((1.0, 1.0), (1.0, 1.0)),
)

GRADED_FIXTURE = BASE_FIXTURE.with_updates(
    name="graded",
    rewards=((0.25, 0.75), (0.10, 0.90)),
)

TRANSLATED_FIXTURE = BASE_FIXTURE.with_updates(
    name="translated",
    rewards=((10.0, 11.0), (-4.0, -3.0)),
)

LOW_CLIP_FIXTURE = BASE_FIXTURE.with_updates(
    name="low_clip",
    logps=((-0.5566749439387324, -0.5566749439387324), (-0.2, -0.4), (-0.3, -0.5), (-0.4, -0.6)),
)

FIXTURES = {
    fixture.name: fixture
    for fixture in (
        BASE_FIXTURE,
        DAPO_CLIP_FIXTURE,
        ZERO_MASK_FIXTURE,
        AERO_POSTERIOR_FIXTURE,
        ALL_WRONG_FIXTURE,
        ALL_CORRECT_FIXTURE,
        GRADED_FIXTURE,
        TRANSLATED_FIXTURE,
        LOW_CLIP_FIXTURE,
    )
}
