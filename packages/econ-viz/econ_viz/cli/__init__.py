"""Legacy ``econ-viz`` command: warns, then forwards to ``utility-viz`` (removed in 3.0.0)."""

from __future__ import annotations

import warnings

from utility_viz.core.errors.deprecation import UtilityVizDeprecationWarning, deprecation_message

__path__ = []  # namespace-like: sub-paths resolve through the legacy finder


def main() -> None:
    """Entry point of the ``econ-viz`` console script."""
    warnings.warn(
        deprecation_message("The `econ-viz` command", "the `utility-viz` command (same arguments)"),
        UtilityVizDeprecationWarning,
        stacklevel=2,
    )
    from utility_viz.cli import main as _main

    _main()
