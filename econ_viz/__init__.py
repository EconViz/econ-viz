"""econ_viz -- deprecated compatibility alias for :mod:`utility_viz`.

The project was renamed to ``utility-viz`` in 2.0.0. This package keeps the
documented 1.x public API importable throughout the 2.x series and **will be
removed in 3.0.0**.

* Names unchanged in 2.x are re-exported as the very same objects.
* ``Canvas``, ``Figure`` and ``Animator`` are thin subclasses that emit one
  :class:`utility_viz.UtilityVizDeprecationWarning` (a ``FutureWarning``) per
  construction; accessing ``Layout`` warns at the access site.
* Documented sub-module paths (``econ_viz.models``, ``econ_viz.optimizer``,
  ``econ_viz.themes``, ...) resolve to their ``utility_viz`` equivalents.
  Undocumented deep paths work on a best-effort basis only.

Migrate with ``import utility_viz`` / ``from utility_viz import ...``.
"""

from __future__ import annotations

from typing import Any

import utility_viz as _utility_viz
from econ_viz import _legacy

_legacy.install_finder()

__all__ = list(_utility_viz.__all__)


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return _legacy.legacy_attr(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"Animator", "Layout"})
