"""Shared validation for the ``opacity`` field of style objects."""

from __future__ import annotations

from ..exceptions import InvalidParameterError


def check_opacity(owner: str, value: float | None) -> None:
    """Raise unless *value* is None or within [0, 1]."""
    if value is not None and not 0 <= value <= 1:
        raise InvalidParameterError(f"{owner} opacity must be between 0 and 1, got {value!r}")
