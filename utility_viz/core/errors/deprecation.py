"""Deprecation warning type and message helper for the econ-viz -> utility-viz migration."""

from __future__ import annotations

import warnings

DEPRECATED_SINCE = "2.0.0"
REMOVAL_VERSION = "3.0.0"


class UtilityVizDeprecationWarning(FutureWarning):
    """Warns that a legacy (econ-viz 1.x) name is deprecated.

    Subclasses :class:`FutureWarning` so it is visible to end users by default.
    Every message states the version the deprecation started in, the version
    that removes it, and the replacement API.
    """


def deprecation_message(old: str, replacement: str) -> str:
    """Build the standard message: deprecated since, removal in, replacement."""
    return (
        f"{old} is deprecated since {DEPRECATED_SINCE} and will be removed in {REMOVAL_VERSION}; "
        f"use {replacement} instead."
    )


def warn_deprecated(old: str, replacement: str, *, stacklevel: int = 2) -> None:
    """Emit a :class:`UtilityVizDeprecationWarning`.

    *stacklevel* is relative to the caller of this function (``2`` blames the
    caller's caller, mirroring :func:`warnings.warn` called from the caller).
    """
    warnings.warn(deprecation_message(old, replacement), UtilityVizDeprecationWarning, stacklevel=stacklevel + 1)
