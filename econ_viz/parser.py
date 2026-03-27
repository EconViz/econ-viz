"""
LaTeX math string → callable utility function.

This module uses *sympy* to parse a LaTeX expression into a symbolic
representation and then compiles it into a fast NumPy-vectorised callable
that satisfies the :class:`~econ_viz.models.protocol.UtilityFunction`
protocol.

Typical usage::

    from econ_viz.parser import parse_latex

    model = parse_latex(r"x^{0.5} y^{0.5}")          # Cobb-Douglas
    model = parse_latex(r"\\min(x, y)")                # Leontief
    model = parse_latex(r"2x + 3y")                    # Perfect subs

The returned object can be passed directly to :meth:`Canvas.add_utility`
and :func:`~econ_viz.optimizer.solve`.
"""

from __future__ import annotations

import re

import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex as _sympy_parse

from .enums import UtilityType
from .exceptions import ParseError
from .logging import get_logger

logger = get_logger(__name__)

# Canonical free variables — every parsed expression must use these.
_x, _y = sp.symbols("x y", positive=True)


# ------------------------------------------------------------------
# Pre-processing helpers
# ------------------------------------------------------------------

def _preprocess(latex: str) -> str:
    """Normalise a user-supplied LaTeX string before sympy parsing.

    Handles common patterns that sympy's LaTeX parser does not support
    natively:

    * Strips a leading ``U(x,y) =`` or ``u(x,y) =`` preamble.
    * Converts ``\\min(...)`` / ``\\max(...)`` into sympy-friendly forms.
    * Inserts explicit multiplication where juxtaposition implies it
      (e.g. ``2x`` → ``2 x``, ``xy`` → ``x y``).
    """
    s = latex.strip()

    # Strip optional preamble: U(x, y) = ...  or  u = ...
    s = re.sub(r'^[Uu]\s*(\([^)]*\)\s*)?=\s*', '', s)

    # \min(...) → Min(...)   \max(...) → Max(...)
    s = s.replace(r'\min', r'\operatorname{Min}')
    s = s.replace(r'\max', r'\operatorname{Max}')

    # Insert space for implicit multiplication: "2x" → "2 x", "xy" → "x y"
    s = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', s)

    return s


# ------------------------------------------------------------------
# Utility-type inference
# ------------------------------------------------------------------

def _infer_type(expr: sp.Expr) -> UtilityType:
    """Infer the qualitative shape category from the symbolic expression."""
    if expr.has(sp.Min):
        return UtilityType.KINKED
    if expr.is_polynomial(_x, _y) and sp.degree(expr, _x) <= 1 and sp.degree(expr, _y) <= 1:
        # ax + by + c  →  LINEAR
        return UtilityType.LINEAR
    return UtilityType.SMOOTH


# ------------------------------------------------------------------
# Ray / kink helpers
# ------------------------------------------------------------------

def _compute_ray_slopes(expr: sp.Expr, utype: UtilityType) -> list[float]:
    """Derive economically significant ray slopes from the expression."""
    if utype is UtilityType.KINKED:
        # For min(a*x, b*y): kink locus is a*x == b*y  →  y/x = a/b
        mins = list(expr.atoms(sp.Min))
        if mins:
            args = mins[0].args
            if len(args) == 2:
                a_coeff = args[0].coeff(_x) if args[0].has(_x) else 0
                b_coeff = args[1].coeff(_y) if args[1].has(_y) else 0
                if a_coeff and b_coeff:
                    return [float(a_coeff / b_coeff)]
    if utype is UtilityType.LINEAR:
        a_coeff = expr.coeff(_x)
        b_coeff = expr.coeff(_y)
        if b_coeff:
            return [float(a_coeff / b_coeff)]
    return []


def _compute_kink_points(
    expr: sp.Expr, utype: UtilityType, levels: list[float],
) -> list[tuple[float, float]]:
    """Return (x, y) kink coordinates for each contour level."""
    if utype is not UtilityType.KINKED:
        return []
    mins = list(expr.atoms(sp.Min))
    if not mins or len(mins[0].args) != 2:
        return []
    args = mins[0].args
    a_coeff = float(args[0].coeff(_x)) if args[0].has(_x) else 1.0
    b_coeff = float(args[1].coeff(_y)) if args[1].has(_y) else 1.0
    # At kink: a*x == b*y == U  →  x = U/a, y = U/b
    return [(u / a_coeff, u / b_coeff) for u in levels]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class LaTeXUtility:
    """A utility function parsed from a LaTeX math string.

    Conforms to the :class:`~econ_viz.models.protocol.UtilityFunction`
    protocol and can be used anywhere a built-in model is accepted.

    Parameters
    ----------
    latex : str
        LaTeX math expression in ``x`` and ``y``.
    """

    def __init__(self, latex: str) -> None:
        self._latex = latex
        preprocessed = _preprocess(latex)
        logger.debug("Parsing LaTeX: %r → preprocessed: %r", latex, preprocessed)

        try:
            self._expr: sp.Expr = _sympy_parse(preprocessed)
        except Exception as exc:
            raise ParseError(f"Cannot parse LaTeX expression: {latex!r}") from exc

        # Replace sympy Max/Min function names if operatorname was used
        self._expr = self._expr.replace(
            sp.Function('Min'), lambda *a: sp.Min(*a),
        ).replace(
            sp.Function('Max'), lambda *a: sp.Max(*a),
        )

        # Validate that the expression uses only x and y as free symbols
        free = self._expr.free_symbols - {_x, _y}
        if free:
            raise ParseError(
                f"Expression contains unknown symbols {free}; "
                f"only 'x' and 'y' are allowed."
            )

        self._utype = _infer_type(self._expr)
        self._ray_slopes = _compute_ray_slopes(self._expr, self._utype)

        # Compile to a fast NumPy lambda
        self._fn = sp.lambdify((_x, _y), self._expr, modules=["numpy"])
        logger.info("Parsed LaTeX utility: type=%s, expr=%s", self._utype.name, self._expr)

    # -- UtilityFunction protocol ----------------------------------

    @property
    def utility_type(self) -> UtilityType:
        return self._utype

    def __call__(self, x, y):
        return self._fn(x, y)

    def ray_slopes(self) -> list[float]:
        return self._ray_slopes

    def kink_points(self, levels: list[float]) -> list[tuple[float, float]]:
        return _compute_kink_points(self._expr, self._utype, levels)

    # -- Convenience -----------------------------------------------

    @property
    def expr(self) -> sp.Expr:
        """The underlying sympy symbolic expression."""
        return self._expr

    def __repr__(self) -> str:
        return f"LaTeXUtility({self._latex!r})"


def parse_latex(latex: str) -> LaTeXUtility:
    """Parse a LaTeX math string into a callable utility function.

    Parameters
    ----------
    latex : str
        A LaTeX expression in variables ``x`` and ``y``.
        An optional ``U(x,y) =`` preamble is stripped automatically.

    Returns
    -------
    LaTeXUtility
        A callable conforming to the ``UtilityFunction`` protocol.

    Raises
    ------
    ParseError
        If the expression cannot be parsed or contains invalid symbols.

    Examples
    --------
    >>> from econ_viz.parser import parse_latex
    >>> cd = parse_latex(r"x^{0.5} y^{0.5}")
    >>> cd(4, 9)
    6.0
    >>> leon = parse_latex(r"\\min(2x, 3y)")
    >>> leon(1, 1)
    2
    """
    return LaTeXUtility(latex)
