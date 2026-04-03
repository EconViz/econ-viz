"""CLI-specific exceptions."""

from __future__ import annotations


class CliConfigError(ValueError):
    """Raised when CLI arguments cannot be resolved into a valid config."""

