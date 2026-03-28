"""Tests for econ_viz.optimizer.solver — equilibrium solver."""

import numpy as np
import pytest

from econ_viz.exceptions import InvalidParameterError, OptimizationError
from econ_viz.models import CobbDouglas, Leontief, PerfectSubstitutes, CES
from econ_viz.optimizer import Equilibrium, solve


def _on_budget(eq: Equilibrium, px, py, income, tol=1e-6):
    """Return True when the bundle satisfies the budget equation within tolerance."""
    return abs(px * eq.x + py * eq.y - income) < tol


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
