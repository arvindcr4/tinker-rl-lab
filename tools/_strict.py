"""Shared fail-closed assertion for the standard-library-only checkers in tools/."""

from __future__ import annotations


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)
