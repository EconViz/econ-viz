"""
econ_viz.interactive
====================

Jupyter-notebook slider controls for econ_viz diagrams.

This submodule is intentionally **not** imported by the top-level
``econ_viz`` package, so a missing ``ipywidgets`` installation never
causes an ``ImportError`` for terminal users.

Usage::

    from econ_viz.interactive import WidgetViewer
    from econ_viz import Canvas
    from econ_viz.models import CobbDouglas

    def draw(p1: float, income: float) -> Canvas:
        c = Canvas(x_max=10, y_max=10, x_label="X_1", y_label="X_2")
        c.add_budget(px=p1, py=2.0, income=income)
        c.add_utility(CobbDouglas(0.5, 0.5), levels=5)
        return c

    viewer = WidgetViewer(draw, p1=(1.0, 8.0, 0.5), income=(10.0, 40.0, 5.0))
    viewer.show()

Requires ``ipywidgets``::

    pip install econ-viz[interactive]
    # or: pip install ipywidgets
"""

from econ_viz.interactive.widgets import WidgetViewer

__all__ = ["WidgetViewer"]
