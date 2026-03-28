"""
LaTeX math string → concrete UtilityFunction model.

Uses regex pattern matching to identify the utility function family and
extract its parameters, returning an instantiated model from
:mod:`econ_viz.models.core`.

Supported forms
---------------
**Cobb-Douglas** ``x^{\\alpha} y^{\\beta}``::

    parse_latex(r"x^{0.3} y^{0.7}")      -> CobbDouglas(alpha=0.3, beta=0.7)
    parse_latex(r"U(x,y) = x^0.5 y^0.5") -> CobbDouglas(alpha=0.5, beta=0.5)

**Leontief** ``\\min(ax, by)``::

    parse_latex(r"\\min(2x, y)")          -> Leontief(a=2.0, b=1.0)
    parse_latex(r"U = min(x, 3y)")        -> Leontief(a=1.0, b=3.0)

**Perfect Substitutes** ``ax + by``::

    parse_latex(r"3x + 1.5y")             -> PerfectSubstitutes(a=3.0, b=1.5)
    parse_latex(r"U(x,y) = 2x + y")      -> PerfectSubstitutes(a=2.0, b=1.0)
"""

from __future__ import annotations

import re

from .exceptions import ParseError
from .logging import get_logger
from .models.core import CobbDouglas, Leontief, PerfectSubstitutes

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Number literal: integer or decimal, optionally signed
_NUM = r"(\d+(?:\.\d+)?)"

# Optional coefficient before a variable: "2x", "0.5 x", or just "x" (→ 1)
_COEFF_X = r"(?:(\d+(?:\.\d+)?)\s*\*?\s*)?x"
_COEFF_Y = r"(?:(\d+(?:\.\d+)?)\s*\*?\s*)?y"

# Exponent: ^{n} or ^n
_EXP = r"\^\{?" + _NUM + r"\}?"


def _strip_preamble(s: str) -> str:
    """Remove leading 'U(x,y) =' or 'U =' preamble."""
    return re.sub(r"^[Uu]\s*(?:\([^)]*\)\s*)?=\s*", "", s)


def _coeff(raw: str | None) -> float:
    """Convert a captured coefficient group to float (None → 1.0)."""
    return float(raw) if raw is not None else 1.0


# ---------------------------------------------------------------------------
# Pattern matchers — each returns a model or None
# ---------------------------------------------------------------------------

# Cobb-Douglas: x^{a} y^{b}  or  x^a y^b  (with optional * between)
_CD_RE = re.compile(
    r"^x\s*\^\{?" + _NUM + r"\}?"   # x^{alpha}
    r"\s*\*?\s*"
    r"y\s*\^\{?" + _NUM + r"\}?"    # y^{beta}
    r"\s*$",
    re.IGNORECASE,
)


def _try_cobb_douglas(s: str) -> CobbDouglas | None:
    m = _CD_RE.match(s)
    if not m:
        return None
    alpha, beta = float(m.group(1)), float(m.group(2))
    logger.debug("Matched Cobb-Douglas: alpha=%s, beta=%s", alpha, beta)
    return CobbDouglas(alpha=alpha, beta=beta)


# Leontief: \min(ax, by)  or  min(ax, by)  (order of x/y args can swap)
_MIN_RE = re.compile(
    r"^\\?min\s*\(\s*"
    + _COEFF_X + r"\s*,\s*" + _COEFF_Y +
    r"\s*\)\s*$",
    re.IGNORECASE,
)
_MIN_RE_YX = re.compile(  # y before x
    r"^\\?min\s*\(\s*"
    + _COEFF_Y + r"\s*,\s*" + _COEFF_X +
    r"\s*\)\s*$",
    re.IGNORECASE,
)


def _try_leontief(s: str) -> Leontief | None:
    m = _MIN_RE.match(s)
    if m:
        a, b = _coeff(m.group(1)), _coeff(m.group(2))
        logger.debug("Matched Leontief: a=%s, b=%s", a, b)
        return Leontief(a=a, b=b)
    m = _MIN_RE_YX.match(s)
    if m:
        b, a = _coeff(m.group(1)), _coeff(m.group(2))
        logger.debug("Matched Leontief (yx): a=%s, b=%s", a, b)
        return Leontief(a=a, b=b)
    return None


# Perfect Substitutes: ax + by  (coefficients optional → 1)
_PS_RE = re.compile(
    r"^" + _COEFF_X + r"\s*\+\s*" + _COEFF_Y + r"\s*$",
    re.IGNORECASE,
)


def _try_perfect_substitutes(s: str) -> PerfectSubstitutes | None:
    m = _PS_RE.match(s)
    if not m:
        return None
    a, b = _coeff(m.group(1)), _coeff(m.group(2))
    logger.debug("Matched PerfectSubstitutes: a=%s, b=%s", a, b)
    return PerfectSubstitutes(a=a, b=b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_latex(latex: str) -> CobbDouglas | Leontief | PerfectSubstitutes:
    """Parse a LaTeX utility-function string into a concrete model instance.

    Parameters
    ----------
    latex : str
        A LaTeX math expression in variables ``x`` and ``y``.
        An optional ``U(x,y) =`` or ``U =`` preamble is stripped
        automatically.  The following families are recognised:

        * **Cobb-Douglas** — ``x^{alpha} y^{beta}``
        * **Leontief** — ``\\\\min(ax, by)``
        * **Perfect Substitutes** — ``ax + by``

    Returns
    -------
    CobbDouglas | Leontief | PerfectSubstitutes
        An instantiated model conforming to the ``UtilityFunction`` protocol.

    Raises
    ------
    ParseError
        If the string does not match any recognised pattern.

    Examples
    --------
    >>> from econ_viz.parser import parse_latex
    >>> parse_latex(r"x^{0.3} y^{0.7}")
    CobbDouglas(alpha=0.3, beta=0.7)
    >>> parse_latex(r"\\min(2x, y)")
    Leontief(a=2.0, b=1.0)
    >>> parse_latex(r"3x + 1.5y")
    PerfectSubstitutes(a=3.0, b=1.5)
    """
    # Normalise: strip preamble, collapse whitespace
    cleaned = _strip_preamble(latex.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    logger.debug("parse_latex: %r → cleaned: %r", latex, cleaned)

    for matcher in (_try_cobb_douglas, _try_leontief, _try_perfect_substitutes):
        result = matcher(cleaned)
        if result is not None:
            logger.info("parse_latex: %r → %r", latex, result)
            return result

    raise ParseError(
        f"Unrecognised LaTeX utility function: {latex!r}\n"
        "Supported forms:\n"
        "  Cobb-Douglas      : x^{alpha} y^{beta}\n"
        r"  Leontief          : \min(ax, by)"
        "\n"
        "  Perfect Substitutes: ax + by"
    )
