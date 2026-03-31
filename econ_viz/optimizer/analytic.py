"""Closed-form Marshallian demand formulas rendered as LaTeX text."""

from __future__ import annotations

from ..models import CobbDouglas, Leontief, PerfectSubstitutes, StoneGeary


def _tex_atom(value: str | float | int) -> str:
    if isinstance(value, str):
        return value
    return f"{value:g}"


def _frac(num: str, den: str) -> str:
    return rf"\frac{{{num}}}{{{den}}}"


def _mul(left: str, right: str) -> str:
    return rf"{left}\,{right}"


def solution_tex(
    func,
    *,
    px: str | float = r"p_x",
    py: str | float = r"p_y",
    income: str | float = "I",
    symbolic_params: bool = False,
) -> str:
    """Return a LaTeX-text closed-form Marshallian demand for supported models.

    The returned string is plain TeX text and is not wrapped in ``$...$``.
    """
    px_t = _tex_atom(px)
    py_t = _tex_atom(py)
    income_t = _tex_atom(income)

    if isinstance(func, CobbDouglas):
        alpha = r"\alpha" if symbolic_params else _tex_atom(func.alpha)
        beta = r"\beta" if symbolic_params else _tex_atom(func.beta)
        total = f"{alpha}+{beta}"
        return (
            rf"x^* = {_frac(alpha, total)} {_frac(income_t, px_t)}, \quad "
            rf"y^* = {_frac(beta, total)} {_frac(income_t, py_t)}"
        )

    if isinstance(func, StoneGeary):
        alpha = r"\alpha" if symbolic_params else _tex_atom(func.alpha)
        beta = r"\beta" if symbolic_params else _tex_atom(func.beta)
        bar_x = r"\bar{x}" if symbolic_params else _tex_atom(func.bar_x)
        bar_y = r"\bar{y}" if symbolic_params else _tex_atom(func.bar_y)
        total = f"{alpha}+{beta}"
        supernumerary = f"{income_t} - {_mul(px_t, bar_x)} - {_mul(py_t, bar_y)}"
        return (
            rf"x^* = {bar_x} + {_frac(alpha, total)} {_frac(supernumerary, px_t)}, \quad "
            rf"y^* = {bar_y} + {_frac(beta, total)} {_frac(supernumerary, py_t)}"
        )

    if isinstance(func, Leontief):
        a = "a" if symbolic_params else _tex_atom(func.a)
        b = "b" if symbolic_params else _tex_atom(func.b)
        den = f"{b}{px_t} + {a}{py_t}"
        return (
            rf"x^* = {_frac(f'{b}{income_t}', den)}, \quad "
            rf"y^* = {_frac(f'{a}{income_t}', den)}"
        )

    if isinstance(func, PerfectSubstitutes):
        a = "a" if symbolic_params else _tex_atom(func.a)
        b = "b" if symbolic_params else _tex_atom(func.b)
        mu_x = _frac(a, px_t)
        mu_y = _frac(b, py_t)
        x_corner = _frac(income_t, px_t)
        y_corner = _frac(income_t, py_t)
        return (
            r"(x^*, y^*) = "
            r"\begin{cases}"
            rf"\left({x_corner}, 0\right), & {mu_x} > {mu_y} \\"
            rf"\left(0, {y_corner}\right), & {mu_x} < {mu_y} \\"
            rf"\{{(x,y)\mid {px_t}x + {py_t}y = {income_t},\ x \ge 0,\ y \ge 0\}},"
            rf" & {mu_x} = {mu_y}"
            r"\end{cases}"
        )

    raise NotImplementedError(
        f"Closed-form TeX solution is not implemented for {func.__class__.__name__}."
    )
