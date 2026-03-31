"""
econ_viz.analysis — Analytical tools for utility and demand theory.

Submodules
----------
homogeneity
    Homogeneity degree detection, Euler's theorem verification,
    homotheticity testing, and demand degree-0 verification.
"""

from . import levels
from .homogeneity import HomogeneityAnalyzer, HomogeneityResult

__all__ = ["levels", "HomogeneityAnalyzer", "HomogeneityResult"]
