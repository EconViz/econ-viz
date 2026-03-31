"""Tests for econ_viz.optimizer.solver — equilibrium solver."""

import numpy as np
import pytest

from econ_viz.exceptions import InvalidParameterError, OptimizationError
from econ_viz.models import CobbDouglas, Leontief, PerfectSubstitutes, CES, QuasiLinear, StoneGeary
from econ_viz.optimizer import (
    Equilibrium,
    solve,
    solution_tex,
    ComparativeStatics,
    SlutskyMatrix,
    comparative_statics,
    slutsky_matrix,
)


def _on_budget(eq: Equilibrium, px, py, income, rtol=1e-4):
    """Return True when the bundle satisfies the budget equation within relative tolerance."""
    return abs(px * eq.x + py * eq.y - income) <= rtol * income


class TestSolveCobbDouglas:
    """Interior (SLSQP) solution for smooth Cobb-Douglas preferences."""

    def setup_method(self):
        self.model = CobbDouglas(alpha=0.5, beta=0.5)
        self.px, self.py, self.income = 2.0, 3.0, 30.0

    def test_on_budget(self):
        eq = solve(self.model, self.px, self.py, self.income)
        assert _on_budget(eq, self.px, self.py, self.income)

    def test_bundle_type(self):
        eq = solve(self.model, self.px, self.py, self.income)
        assert eq.bundle_type == "interior"

    def test_utility_positive(self):
        eq = solve(self.model, self.px, self.py, self.income)
        assert eq.utility > 0

    def test_analytic_optimum(self):
        """For CD(0.5, 0.5) the analytic optimum is x*=αI/px, y*=βI/py."""
        eq = solve(self.model, self.px, self.py, self.income)
        x_star = 0.5 * self.income / self.px
        y_star = 0.5 * self.income / self.py
        assert eq.x == pytest.approx(x_star, rel=1e-4)
        assert eq.y == pytest.approx(y_star, rel=1e-4)

    def test_equilibrium_is_frozen(self):
        """Equilibrium is a frozen dataclass; mutation must raise."""
        eq = solve(self.model, self.px, self.py, self.income)
        with pytest.raises(Exception):
            eq.x = 99.0


class TestSolveLeontief:
    """Analytic kink solution for Leontief preferences."""

    def test_bundle_type(self):
        eq = solve(Leontief(a=1.0, b=1.0), 2.0, 3.0, 30.0)
        assert eq.bundle_type == "kink"

    def test_on_budget(self):
        eq = solve(Leontief(a=2.0, b=3.0), 2.0, 3.0, 30.0)
        assert _on_budget(eq, 2.0, 3.0, 30.0)

    def test_kink_condition(self):
        """At the optimum the kink condition a*x = b*y must hold exactly."""
        lf = Leontief(a=2.0, b=1.0)
        eq = solve(lf, 1.0, 1.0, 10.0)
        assert lf.a * eq.x == pytest.approx(lf.b * eq.y, rel=1e-6)


class TestSolvePerfectSubstitutes:
    """Corner solution for linear (perfect substitutes) preferences."""

    def test_bundle_type(self):
        eq = solve(PerfectSubstitutes(a=1.0, b=2.0), 2.0, 1.0, 10.0)
        assert eq.bundle_type == "corner"

    def test_on_budget(self):
        ps = PerfectSubstitutes(a=1.0, b=2.0)
        eq = solve(ps, 2.0, 1.0, 10.0)
        assert _on_budget(eq, 2.0, 1.0, 10.0)

    def test_all_y_corner(self):
        """MU_y/py = 2 > MU_x/px = 0.5, so all income is spent on y."""
        ps = PerfectSubstitutes(a=1.0, b=2.0)
        eq = solve(ps, 2.0, 1.0, 10.0)
        assert eq.x == pytest.approx(0.0, abs=1e-9)
        assert eq.y == pytest.approx(10.0, rel=1e-6)

    def test_all_x_corner(self):
        """MU_x/px = 3 > MU_y/py = 1/3, so all income is spent on x."""
        ps = PerfectSubstitutes(a=3.0, b=1.0)
        eq = solve(ps, 1.0, 3.0, 9.0)
        assert eq.y == pytest.approx(0.0, abs=1e-9)
        assert eq.x == pytest.approx(9.0, rel=1e-6)


class TestSolveCES:
    """Interior solution for CES preferences."""

    def test_on_budget(self):
        eq = solve(CES(0.5, 0.5, 0.5), 2.0, 3.0, 30.0)
        assert _on_budget(eq, 2.0, 3.0, 30.0)

    def test_bundle_type(self):
        eq = solve(CES(0.5, 0.5, 0.5), 2.0, 3.0, 30.0)
        assert eq.bundle_type == "interior"


class TestSolveInvalidParams:
    """Parameter validation in solve()."""

    def test_negative_px(self):
        with pytest.raises(InvalidParameterError):
            solve(CobbDouglas(), px=-1.0, py=1.0, income=10.0)

    def test_zero_py(self):
        with pytest.raises(InvalidParameterError):
            solve(CobbDouglas(), px=1.0, py=0.0, income=10.0)

    def test_negative_income(self):
        with pytest.raises(InvalidParameterError):
            solve(CobbDouglas(), px=1.0, py=1.0, income=-5.0)


class TestEquilibrium:
    """Structural tests for the Equilibrium frozen dataclass."""

    def test_fields(self):
        eq = Equilibrium(x=3.0, y=4.0, utility=5.0, bundle_type="interior")
        assert eq.x == 3.0
        assert eq.y == 4.0
        assert eq.utility == 5.0
        assert eq.bundle_type == "interior"


class TestSolutionTex:
    def test_cobb_douglas_tex_contains_expected_terms(self):
        tex = solution_tex(CobbDouglas(alpha=0.4, beta=0.6))
        assert r"x^*" in tex
        assert r"\frac{0.4}{0.4+0.6}" in tex or r"\frac{0.4}{0.6+0.4}" in tex
        assert r"\frac{I}{p_x}" in tex
        assert r"\frac{0.6}{0.4+0.6}" in tex or r"\frac{0.6}{0.6+0.4}" in tex
        assert r"\frac{I}{p_y}" in tex

    def test_leontief_tex(self):
        tex = solution_tex(Leontief(a=2.0, b=3.0))
        assert tex == (
            r"x^* = \frac{3I}{3p_x + 2p_y}, \quad "
            r"y^* = \frac{2I}{3p_x + 2p_y}"
        )

    def test_perfect_substitutes_tex_is_piecewise(self):
        tex = solution_tex(PerfectSubstitutes(a=2.0, b=1.0))
        assert r"\begin{cases}" in tex
        assert r"\left(\frac{I}{p_x}, 0\right)" in tex
        assert r"\left(0, \frac{I}{p_y}\right)" in tex
        assert r"p_xx + p_yy = I" in tex

    def test_stone_geary_tex(self):
        tex = solution_tex(StoneGeary(alpha=2.0, beta=3.0, bar_x=1.0, bar_y=4.0))
        assert r"x^* = 1 +" in tex
        assert r"\frac{2}{2+3}" in tex
        assert r"\frac{I - p_x\,1 - p_y\,4}{p_x}" in tex
        assert r"y^* = 4 +" in tex
        assert r"\frac{3}{2+3}" in tex
        assert r"\frac{I - p_x\,1 - p_y\,4}{p_y}" in tex

    def test_custom_symbols(self):
        tex = solution_tex(CobbDouglas(alpha=1.0, beta=1.0), px=r"P_x", py=r"P_y", income="M")
        assert r"\frac{M}{P_x}" in tex
        assert r"\frac{M}{P_y}" in tex

    def test_symbolic_cobb_douglas_params(self):
        tex = solution_tex(CobbDouglas(alpha=0.4, beta=0.6), symbolic_params=True)
        assert r"\frac{\alpha}{\alpha+\beta}" in tex
        assert r"\frac{\beta}{\alpha+\beta}" in tex
        assert r"\frac{I}{p_x}" in tex
        assert r"\frac{I}{p_y}" in tex

    def test_symbolic_stone_geary_params(self):
        tex = solution_tex(StoneGeary(alpha=2.0, beta=3.0, bar_x=1.0, bar_y=4.0), symbolic_params=True)
        assert r"x^* = \bar{x} +" in tex
        assert r"\frac{\alpha}{\alpha+\beta}" in tex
        assert r"I - p_x\,\bar{x} - p_y\,\bar{y}" in tex
        assert r"y^* = \bar{y} +" in tex

    def test_unsupported_model_raises(self):
        with pytest.raises(NotImplementedError):
            solution_tex(CES(alpha=0.5, beta=0.5, rho=0.5))


class TestSolveEdgeCases:
    """Edge-case branches that are hard to hit with standard inputs."""

    def test_slsqp_failure_raises(self, monkeypatch):
        """A non-converging SLSQP result must raise OptimizationError."""
        from unittest.mock import MagicMock
        import econ_viz.optimizer.solver as solver_mod

        fake_result = MagicMock()
        fake_result.success = False
        fake_result.message = "Iteration limit exceeded"
        monkeypatch.setattr(solver_mod, "minimize", lambda **kw: fake_result)

        with pytest.raises(OptimizationError, match="SLSQP"):
            solve(CobbDouglas(), px=1.0, py=1.0, income=10.0)

    def test_slsqp_minimize_called_with_positional_args(self, monkeypatch):
        """Ensure solve() passes args to minimize correctly (regression guard)."""
        calls = []
        import econ_viz.optimizer.solver as solver_mod
        original = solver_mod.minimize

        def recording_minimize(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(solver_mod, "minimize", recording_minimize)
        solve(CobbDouglas(), px=1.0, py=1.0, income=10.0)
        assert len(calls) == 1

    def test_kinked_no_slopes_falls_back_to_interior(self):
        """A KINKED model with no ray slopes must fall back to the interior solver."""
        class NoSlopeKinked:
            from econ_viz.enums import UtilityType as _UT
            utility_type = _UT.KINKED

            def __call__(self, x, y):
                return np.minimum(x, y)

            def ray_slopes(self):
                return []

            def kink_points(self, levels):
                return []

        m = NoSlopeKinked()
        eq = solve(m, px=1.0, py=1.0, income=10.0)
        assert eq.x + eq.y == pytest.approx(10.0, rel=1e-4)


class TestSolveCobbDouglasAnalytic:
    """Verify SLSQP output matches closed-form Marshallian demands.

    SLSQP's ftol=1e-6 controls the *function value* residual, not the
    variable residual.  Variable errors are typically ~1e-3 to ~3e-4 in
    relative terms, so we use rel=1e-3 to stay robust across scipy versions.
    """

    @pytest.mark.parametrize("alpha,beta,px,py,income", [
        (0.5, 0.5, 2.0, 3.0, 30.0),
        (0.4, 0.6, 2.0, 3.0, 30.0),
        (0.3, 0.7, 1.0, 2.0, 20.0),
        (0.8, 0.2, 5.0, 1.0, 50.0),
        (0.5, 0.5, 1.0, 1.0, 100.0),
    ])
    def test_marshallian_demands(self, alpha, beta, px, py, income):
        """x* = α/(α+β)·I/pₓ and y* = β/(α+β)·I/p_y."""
        eq = solve(CobbDouglas(alpha=alpha, beta=beta), px=px, py=py, income=income)
        x_star = alpha / (alpha + beta) * income / px
        y_star = beta  / (alpha + beta) * income / py
        assert eq.x == pytest.approx(x_star, rel=1e-3)
        assert eq.y == pytest.approx(y_star, rel=1e-3)

    @pytest.mark.parametrize("alpha,beta,px,py,income", [
        (0.5, 0.5, 2.0, 3.0, 30.0),
        (0.4, 0.6, 2.0, 3.0, 30.0),
        (0.3, 0.7, 1.0, 2.0, 20.0),
    ])
    def test_budget_exhausted(self, alpha, beta, px, py, income):
        """The optimal bundle must exhaust the budget within relative tolerance."""
        eq = solve(CobbDouglas(alpha=alpha, beta=beta), px=px, py=py, income=income)
        assert _on_budget(eq, px, py, income)


class TestSolveQuasiLinear:
    """Verify that x* is independent of income (the hallmark of quasi-linear utility)."""

    @pytest.mark.parametrize("income", [10.0, 20.0, 40.0, 100.0])
    def test_x_star_independent_of_income(self, income):
        """For U=log(x)+y with pₓ=2, p_y=1 the FOC gives x*=p_y/pₓ=0.5.

        x* must be the same for every income level — this is the defining
        property of quasi-linear utility (no income effect on the non-linear good).
        """
        ql = QuasiLinear(v_func=np.log, linear_in="y")
        eq = solve(ql, px=2.0, py=1.0, income=income)
        assert eq.x == pytest.approx(0.5, rel=1e-3)

    def test_budget_exhausted(self):
        ql = QuasiLinear(v_func=np.log, linear_in="y")
        eq = solve(ql, px=2.0, py=1.0, income=20.0)
        assert _on_budget(eq, 2.0, 1.0, 20.0)


class TestSolveLeontiefAnalytic:
    """Verify analytic kink-solution demands."""

    @pytest.mark.parametrize("a,b,px,py,income", [
        (1.0, 1.0, 2.0, 3.0, 30.0),
        (1.0, 2.0, 2.0, 3.0, 30.0),
        (2.0, 3.0, 1.0, 1.0, 10.0),
    ])
    def test_marshallian_demands(self, a, b, px, py, income):
        """x* = I/(pₓ + a/b·p_y),  y* = a/b·x*.

        Leontief uses an analytic solution (no numerical solver), so a tight
        rel=1e-6 tolerance is appropriate here.
        """
        eq = solve(Leontief(a=a, b=b), px=px, py=py, income=income)
        x_star = income / (px + (a / b) * py)
        y_star = (a / b) * x_star
        assert eq.x == pytest.approx(x_star, rel=1e-6)
        assert eq.y == pytest.approx(y_star, rel=1e-6)

    @pytest.mark.parametrize("a,b,px,py,income", [
        (1.0, 1.0, 2.0, 3.0, 30.0),
        (1.0, 2.0, 2.0, 3.0, 30.0),
    ])
    def test_budget_exhausted(self, a, b, px, py, income):
        eq = solve(Leontief(a=a, b=b), px=px, py=py, income=income)
        assert _on_budget(eq, px, py, income)


class TestSolveStoneGeary:
    """Interior solution for Stone-Geary preferences."""

    def test_bundle_type(self):
        eq = solve(StoneGeary(alpha=0.5, beta=0.5, bar_x=1.0, bar_y=1.0), 2.0, 3.0, 30.0)
        assert eq.bundle_type == "interior"

    def test_on_budget(self):
        eq = solve(StoneGeary(alpha=0.5, beta=0.5, bar_x=1.0, bar_y=1.0), 2.0, 3.0, 30.0)
        assert _on_budget(eq, 2.0, 3.0, 30.0)

    def test_above_subsistence(self):
        """Optimal bundle must strictly exceed the subsistence point."""
        sg = StoneGeary(alpha=0.5, beta=0.5, bar_x=1.0, bar_y=1.0)
        eq = solve(sg, 2.0, 3.0, 30.0)
        assert eq.x > sg.bar_x
        assert eq.y > sg.bar_y

    def test_insufficient_income_raises(self):
        """Income below subsistence expenditure must raise InvalidParameterError."""
        sg = StoneGeary(alpha=0.5, beta=0.5, bar_x=5.0, bar_y=5.0)
        with pytest.raises(InvalidParameterError, match="subsistence"):
            solve(sg, px=2.0, py=2.0, income=10.0)  # need >20 to cover subsistence


class TestSolveStoneGearyAnalytic:
    """Verify SLSQP output matches closed-form Stone-Geary Marshallian demands.

    Analytic Marshallian demands:
        supernumerary income m = I - px*γ_x - py*γ_y
        x* = γ_x + alpha * m / px
        y* = γ_y + beta  * m / py
    """

    @pytest.mark.parametrize("alpha,beta,bx,by,px,py,income", [
        (0.5, 0.5, 1.0, 1.0, 2.0, 3.0, 30.0),
        (0.4, 0.6, 2.0, 1.0, 1.0, 2.0, 20.0),
        (0.3, 0.7, 0.5, 0.5, 3.0, 1.0, 25.0),
    ])
    def test_marshallian_demands(self, alpha, beta, bx, by, px, py, income):
        sg = StoneGeary(alpha=alpha, beta=beta, bar_x=bx, bar_y=by)
        eq = solve(sg, px=px, py=py, income=income)
        m = income - px * bx - py * by
        x_star = bx + alpha * m / px
        y_star = by + beta  * m / py
        assert eq.x == pytest.approx(x_star, rel=1e-3)
        assert eq.y == pytest.approx(y_star, rel=1e-3)

    @pytest.mark.parametrize("alpha,beta,bx,by,px,py,income", [
        (0.5, 0.5, 1.0, 1.0, 2.0, 3.0, 30.0),
        (0.4, 0.6, 2.0, 1.0, 1.0, 2.0, 20.0),
    ])
    def test_budget_exhausted(self, alpha, beta, bx, by, px, py, income):
        sg = StoneGeary(alpha=alpha, beta=beta, bar_x=bx, bar_y=by)
        eq = solve(sg, px=px, py=py, income=income)
        assert _on_budget(eq, px, py, income)

    def test_zero_subsistence_matches_cobb_douglas(self):
        """Stone-Geary with bar=0 must match Cobb-Douglas Marshallian demands exactly."""
        alpha, beta, px, py, income = 0.5, 0.5, 2.0, 3.0, 30.0
        eq_sg = solve(StoneGeary(alpha=alpha, beta=beta, bar_x=0.0, bar_y=0.0),
                      px=px, py=py, income=income)
        eq_cd = solve(CobbDouglas(alpha=alpha, beta=beta), px=px, py=py, income=income)
        assert eq_sg.x == pytest.approx(eq_cd.x, rel=1e-3)
        assert eq_sg.y == pytest.approx(eq_cd.y, rel=1e-3)


class TestSolvePerfectSubstitutesAnalytic:
    """Verify corner-solution demands."""

    @pytest.mark.parametrize("a,b,px,py,income,expect_x", [
        (1.0, 2.0, 2.0, 1.0, 10.0, False),   # b/py > a/px → all on y
        (3.0, 1.0, 1.0, 3.0, 9.0,  True),    # a/px > b/py → all on x
        (1.0, 1.0, 1.0, 2.0, 10.0, True),    # a/px > b/py → all on x
    ])
    def test_corner_demand(self, a, b, px, py, income, expect_x):
        eq = solve(PerfectSubstitutes(a=a, b=b), px=px, py=py, income=income)
        if expect_x:
            assert eq.y == pytest.approx(0.0, abs=1e-9)
            assert eq.x == pytest.approx(income / px, rel=1e-4)
        else:
            assert eq.x == pytest.approx(0.0, abs=1e-9)
            assert eq.y == pytest.approx(income / py, rel=1e-4)


class TestComparativeStaticsCobbDouglas:
    """Verify numerical derivatives against Cobb-Douglas closed-form expressions.

    Analytic Marshallian demands for CD(α, β) with α+β normalised:
        x* = α/(α+β) · I/pₓ
        y* = β/(α+β) · I/pᵧ

    Partial derivatives:
        ∂x*/∂pₓ = -α/(α+β) · I/pₓ²    ∂x*/∂pᵧ = 0       ∂x*/∂I = α/(α+β)/pₓ
        ∂y*/∂pₓ = 0                    ∂y*/∂pᵧ = -β/(α+β)·I/pᵧ²  ∂y*/∂I = β/(α+β)/pᵧ
    """

    def setup_method(self):
        self.alpha, self.beta = 0.4, 0.6
        self.model = CobbDouglas(alpha=self.alpha, beta=self.beta)
        self.px, self.py, self.income = 2.0, 3.0, 60.0
        self.cs = comparative_statics(self.model, self.px, self.py, self.income)
        s = self.alpha + self.beta
        self._dx_dpx = -(self.alpha / s) * self.income / self.px ** 2
        self._dx_dpy = 0.0
        self._dx_dI  = (self.alpha / s) / self.px
        self._dy_dpx = 0.0
        self._dy_dpy = -(self.beta  / s) * self.income / self.py ** 2
        self._dy_dI  = (self.beta  / s) / self.py

    def test_dx_dpx(self):
        assert self.cs.dx_dpx == pytest.approx(self._dx_dpx, rel=2e-2)

    def test_dx_dpy(self):
        assert self.cs.dx_dpy == pytest.approx(self._dx_dpy, abs=1e-3)

    def test_dx_dI(self):
        assert self.cs.dx_dI == pytest.approx(self._dx_dI, rel=2e-2)

    def test_dy_dpx(self):
        assert self.cs.dy_dpx == pytest.approx(self._dy_dpx, abs=1e-3)

    def test_dy_dpy(self):
        assert self.cs.dy_dpy == pytest.approx(self._dy_dpy, rel=2e-2)

    def test_dy_dI(self):
        assert self.cs.dy_dI == pytest.approx(self._dy_dI, rel=2e-2)

    def test_returns_comparative_statics_instance(self):
        assert isinstance(self.cs, ComparativeStatics)

    def test_frozen(self):
        with pytest.raises(Exception):
            self.cs.dx_dpx = 0.0


class TestComparativeStaticsPerfectSubstitutes:
    """Corner-solution comparative statics for perfect substitutes.

    With a=3, b=1, pₓ=1, pᵧ=3 the consumer spends everything on x:
        x* = I/pₓ  →  ∂x*/∂pₓ = -I/pₓ², ∂x*/∂pᵧ = 0, ∂x*/∂I = 1/pₓ
        y* = 0      →  all y-derivatives ≈ 0
    """

    def setup_method(self):
        self.model = PerfectSubstitutes(a=3.0, b=1.0)
        self.px, self.py, self.income = 1.0, 3.0, 9.0
        self.cs = comparative_statics(self.model, self.px, self.py, self.income)

    def test_dx_dpx_negative(self):
        expected = -self.income / self.px ** 2
        assert self.cs.dx_dpx == pytest.approx(expected, rel=1e-3)

    def test_dx_dI_positive(self):
        expected = 1.0 / self.px
        assert self.cs.dx_dI == pytest.approx(expected, rel=1e-3)

    def test_dy_dI_near_zero(self):
        assert abs(self.cs.dy_dI) < 1e-4


class TestComparativeStaticsLeontief:
    """Leontief comparative statics via the analytic kink-solution path.

    Marshallian demands:  x* = I/(pₓ + a/b·pᵧ),  y* = a/b·x*

    With a=b=1: x* = I/(pₓ+pᵧ), y* = I/(pₓ+pᵧ)
        ∂x*/∂pₓ = -I/(pₓ+pᵧ)²
        ∂x*/∂I  =  1/(pₓ+pᵧ)
    """

    def setup_method(self):
        self.model = Leontief(a=1.0, b=1.0)
        self.px, self.py, self.income = 2.0, 3.0, 30.0
        self.cs = comparative_statics(self.model, self.px, self.py, self.income)
        denom = self.px + self.py
        self._dx_dpx = -self.income / denom ** 2
        self._dx_dI  =  1.0 / denom

    def test_dx_dpx(self):
        assert self.cs.dx_dpx == pytest.approx(self._dx_dpx, rel=2e-2)

    def test_dx_dI(self):
        assert self.cs.dx_dI == pytest.approx(self._dx_dI, rel=2e-2)


class TestComparativeStaticsInvalidParams:
    """Parameter validation in comparative_statics()."""

    def test_zero_px(self):
        with pytest.raises(InvalidParameterError):
            comparative_statics(CobbDouglas(), px=0.0, py=1.0, income=10.0)

    def test_negative_py(self):
        with pytest.raises(InvalidParameterError):
            comparative_statics(CobbDouglas(), px=1.0, py=-1.0, income=10.0)

    def test_zero_income(self):
        with pytest.raises(InvalidParameterError):
            comparative_statics(CobbDouglas(), px=1.0, py=1.0, income=0.0)


class TestSlutskyMatrixCobbDouglas:
    """Verify Slutsky matrix entries against Cobb-Douglas closed form."""

    def setup_method(self):
        self.alpha, self.beta = 0.4, 0.6
        self.model = CobbDouglas(alpha=self.alpha, beta=self.beta)
        self.px, self.py, self.income = 2.0, 3.0, 60.0
        self.sm = slutsky_matrix(self.model, self.px, self.py, self.income)
        s = self.alpha + self.beta
        ax = self.alpha / s
        by = self.beta / s
        self._s_xx = -ax * (1 - ax) * self.income / self.px ** 2
        self._s_xy = ax * by * self.income / (self.px * self.py)
        self._s_yx = ax * by * self.income / (self.px * self.py)
        self._s_yy = -by * (1 - by) * self.income / self.py ** 2

    def test_returns_slutsky_matrix_instance(self):
        assert isinstance(self.sm, SlutskyMatrix)

    def test_s_xx(self):
        assert self.sm.s_xx == pytest.approx(self._s_xx, rel=2e-2)

    def test_s_xy(self):
        assert self.sm.s_xy == pytest.approx(self._s_xy, rel=2e-2)

    def test_s_yx(self):
        assert self.sm.s_yx == pytest.approx(self._s_yx, rel=2e-2)

    def test_s_yy(self):
        assert self.sm.s_yy == pytest.approx(self._s_yy, rel=2e-2)

    def test_symmetry(self):
        assert self.sm.s_xy == pytest.approx(self.sm.s_yx, rel=2e-2, abs=1e-4)

    def test_as_array(self):
        assert np.allclose(
            self.sm.as_array(),
            np.array(
                [
                    [self.sm.s_xx, self.sm.s_xy],
                    [self.sm.s_yx, self.sm.s_yy],
                ]
            ),
        )

    def test_is_symmetric(self):
        assert self.sm.is_symmetric()

    def test_is_negative_semidefinite(self):
        assert self.sm.is_negative_semidefinite()

    def test_satisfies_homogeneity(self):
        assert self.sm.satisfies_homogeneity(px=self.px, py=self.py, tol=5e-2)


class TestSlutskyMatrixInvalidParams:
    def test_zero_px(self):
        with pytest.raises(InvalidParameterError):
            slutsky_matrix(CobbDouglas(), px=0.0, py=1.0, income=10.0)


class TestSlutskyMatrixValidationWarnings:
    def test_validation_failures_reported(self):
        sm = SlutskyMatrix(s_xx=1.0, s_xy=2.0, s_yx=0.0, s_yy=1.0)
        failures = sm.validation_failures(px=2.0, py=3.0)
        assert "symmetry" in failures
        assert "negative semidefinite" in failures
        assert "homogeneity" in failures

    def test_failed_checks_emit_warning(self, monkeypatch):
        import econ_viz.optimizer.slutsky as slutsky_mod

        original_cs = slutsky_mod.comparative_statics
        original_solve = slutsky_mod.solve

        def fake_cs(func, px, py, income, h):
            return ComparativeStatics(
                dx_dpx=1.0,
                dx_dpy=0.0,
                dx_dI=0.0,
                dy_dpx=0.0,
                dy_dpy=1.0,
                dy_dI=0.0,
            )

        def fake_solve(func, px, py, income):
            return Equilibrium(x=1.0, y=1.0, utility=1.0, bundle_type="interior")

        monkeypatch.setattr(slutsky_mod, "comparative_statics", fake_cs)
        monkeypatch.setattr(slutsky_mod, "solve", fake_solve)
        try:
            with pytest.warns(UserWarning, match="Slutsky matrix theoretical checks failed"):
                slutsky_matrix(CobbDouglas(), px=2.0, py=3.0, income=10.0)
        finally:
            monkeypatch.setattr(slutsky_mod, "comparative_statics", original_cs)
            monkeypatch.setattr(slutsky_mod, "solve", original_solve)


class TestComparativeStaticsSignWarnings:
    """UserWarning is emitted for economically anomalous signs."""

    def test_giffen_good_warns(self):
        """A model with upward-sloping demand must trigger a UserWarning."""
        class GiffenX:
            """Toy model where x* rises with pₓ (pathological)."""
            from econ_viz.enums import UtilityType as _UT
            utility_type = _UT.SMOOTH

            def __call__(self, x, y):
                # x demand = pₓ*I so derivative is positive — purely for testing
                return float(x * y)

            def ray_slopes(self):
                return []

            def kink_points(self, levels):
                return []

            def lower_bounds(self):
                return (0.0, 0.0)

        # Patch solve to return a Giffen-like pattern
        import econ_viz.optimizer.comparative as comparative_mod
        original_solve = comparative_mod.solve

        call_count = [0]

        def fake_solve(func, px, py, income):
            call_count[0] += 1
            # Make x* increase with pₓ by returning x = pₓ * 2
            return Equilibrium(x=px * 2, y=1.0, utility=1.0, bundle_type="interior")

        comparative_mod.solve = fake_solve
        try:
            with pytest.warns(UserWarning, match="Giffen"):
                comparative_statics(GiffenX(), px=2.0, py=1.0, income=10.0)
        finally:
            comparative_mod.solve = original_solve
