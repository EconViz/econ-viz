"""
econ_viz.models.analysis — Analytical tools for utility and demand theory.

Submodules
----------
homogeneity
    Homogeneity degree detection, Euler's theorem verification,
    homotheticity testing, and demand degree-0 verification.
"""

from econ_viz.models.analysis import levels
from econ_viz.models.analysis.homogeneity import HomogeneityAnalyzer, HomogeneityResult

__all__ = ["levels", "HomogeneityAnalyzer", "HomogeneityResult"]
