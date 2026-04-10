"""
econ_viz.animation
==================

Parameter-sweep GIF animation for econ_viz diagrams.

Usage::

    from econ_viz.animation import Animator
    import numpy as np

    def draw(p1: float):
        from econ_viz import Canvas
        from econ_viz.models import CobbDouglas
        c = Canvas(x_max=10, y_max=10, x_label="X_1", y_label="X_2")
        c.add_budget(px=p1, py=2.0, income=20.0)
        return c

    Animator(draw, frames=np.linspace(1, 8, 40)).save("sweep.gif", fps=12)

Requires ``Pillow``::

    pip install econ-viz[animation]
    # or: pip install Pillow
"""

from econ_viz.animation.animator import Animator

__all__ = ["Animator"]
