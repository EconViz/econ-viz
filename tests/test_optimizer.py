"""Tests for econ_viz.optimizer.solver — equilibrium solver."""

import numpy as np
import pytest

from econ_viz.exceptions import InvalidParameterError, OptimizationError
from econ_viz.models import CobbDouglas, Leontief, PerfectSubstitutes, CES, QuasiLinear
from econ_viz.optimizer import Equilibrium, solve


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
