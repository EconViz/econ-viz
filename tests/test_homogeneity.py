"""Tests for econ_viz.analysis.homogeneity — HomogeneityAnalyzer."""

import math
import pytest
import numpy as np

from econ_viz.analysis import HomogeneityAnalyzer, HomogeneityResult
from econ_viz.enums import ReturnsToScale
from econ_viz.models import CobbDouglas, Leontief, PerfectSubstitutes, CES, StoneGeary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ShiftedCD:
    """Stone-Geary-like shifted Cobb-Douglas — NOT homogeneous."""
    def __call__(self, x, y):
        return (x - 1) ** 0.5 * (y - 1) ** 0.5


class _CRSProduction:
    """f(x,y) = x^0.3 * y^0.7 — homogeneous of degree 1."""
    def __call__(self, x, y):
        return x ** 0.3 * y ** 0.7


class _IRS:
    """f(x,y) = x^0.8 * y^0.8 — homogeneous of degree 1.6."""
    def __call__(self, x, y):
        return x ** 0.8 * y ** 0.8


class _DRS:
    """f(x,y) = x^0.2 * y^0.3 — homogeneous of degree 0.5."""
    def __call__(self, x, y):
        return x ** 0.2 * y ** 0.3


# ---------------------------------------------------------------------------
# HomogeneityResult dataclass
# ---------------------------------------------------------------------------

class TestHomogeneityResult:
    def test_frozen(self):
        r = HomogeneityResult(degree=1.0, is_homogeneous=True,
                              returns_to_scale=ReturnsToScale.CONSTANT)
        with pytest.raises(Exception):
            r.degree = 2.0


# ---------------------------------------------------------------------------
# ReturnsToScale.from_degree
# ---------------------------------------------------------------------------

class TestReturnsToScaleFromDegree:
    def test_none_returns_not_homogeneous(self):
        assert ReturnsToScale.from_degree(None) is ReturnsToScale.NOT_HOMOGENEOUS

    def test_exactly_one(self):
        assert ReturnsToScale.from_degree(1.0) is ReturnsToScale.CONSTANT

    def test_within_tol_of_one(self):
        assert ReturnsToScale.from_degree(1.0 + 5e-5) is ReturnsToScale.CONSTANT

    def test_increasing(self):
        assert ReturnsToScale.from_degree(1.5) is ReturnsToScale.INCREASING

    def test_decreasing(self):
        assert ReturnsToScale.from_degree(0.5) is ReturnsToScale.DECREASING


# ---------------------------------------------------------------------------
# HomogeneityAnalyzer.degree()
# ---------------------------------------------------------------------------

class TestDegree:
    def test_cobb_douglas_degree(self):
        cd = CobbDouglas(alpha=0.4, beta=0.6)
        result = HomogeneityAnalyzer(cd).degree()
        assert result.is_homogeneous
        assert result.degree == pytest.approx(1.0, abs=1e-3)

    def test_cobb_douglas_returns_to_scale(self):
        cd = CobbDouglas(alpha=0.4, beta=0.6)
        result = HomogeneityAnalyzer(cd).degree()
        assert result.returns_to_scale is ReturnsToScale.CONSTANT

    def test_irs_degree(self):
        result = HomogeneityAnalyzer(_IRS()).degree()
        assert result.is_homogeneous
        assert result.degree == pytest.approx(1.6, abs=1e-3)
        assert result.returns_to_scale is ReturnsToScale.INCREASING

    def test_drs_degree(self):
        result = HomogeneityAnalyzer(_DRS()).degree()
        assert result.is_homogeneous
        assert result.degree == pytest.approx(0.5, abs=1e-3)
        assert result.returns_to_scale is ReturnsToScale.DECREASING

    def test_leontief_degree_one(self):
        result = HomogeneityAnalyzer(Leontief(a=1.0, b=1.0)).degree()
        assert result.is_homogeneous
        assert result.degree == pytest.approx(1.0, abs=1e-3)

    def test_perfect_substitutes_degree_one(self):
        result = HomogeneityAnalyzer(PerfectSubstitutes(a=1.0, b=1.0)).degree()
        assert result.is_homogeneous
        assert result.degree == pytest.approx(1.0, abs=1e-3)

    def test_ces_degree_one(self):
        result = HomogeneityAnalyzer(CES(alpha=0.5, beta=0.5, rho=0.5)).degree()
        assert result.is_homogeneous
        assert result.degree == pytest.approx(1.0, abs=1e-3)

    def test_not_homogeneous(self):
        result = HomogeneityAnalyzer(_ShiftedCD()).degree()
        assert not result.is_homogeneous
        assert result.degree is None
        assert result.returns_to_scale is ReturnsToScale.NOT_HOMOGENEOUS

    @pytest.mark.parametrize("alpha,beta", [(0.3, 0.7), (0.5, 0.5), (0.2, 0.8)])
    def test_cobb_douglas_parametric(self, alpha, beta):
        cd = CobbDouglas(alpha=alpha, beta=beta)
        result = HomogeneityAnalyzer(cd).degree()
        assert result.degree == pytest.approx(alpha + beta, abs=1e-3)


# ---------------------------------------------------------------------------
# HomogeneityAnalyzer.euler_check()
# ---------------------------------------------------------------------------

class TestEulerCheck:
    def test_cobb_douglas_euler_near_zero(self):
        cd = CobbDouglas(alpha=0.4, beta=0.6)
        residual = HomogeneityAnalyzer(cd).euler_check(x=3.0, y=4.0)
        assert residual == pytest.approx(0.0, abs=1e-4)

    def test_euler_multiple_bundles(self):
        az = HomogeneityAnalyzer(CobbDouglas(alpha=0.5, beta=0.5))
        for x, y in [(1.0, 1.0), (2.0, 3.0), (5.0, 0.5)]:
            assert az.euler_check(x, y) == pytest.approx(0.0, abs=1e-4)

    def test_not_homogeneous_returns_nan(self):
        az = HomogeneityAnalyzer(_ShiftedCD())
        assert math.isnan(az.euler_check(x=2.0, y=2.0))


# ---------------------------------------------------------------------------
# HomogeneityAnalyzer.is_homothetic()
# ---------------------------------------------------------------------------

class TestIsHomothetic:
    def test_cobb_douglas_is_homothetic(self):
        assert HomogeneityAnalyzer(CobbDouglas()).is_homothetic()

    def test_ces_is_homothetic(self):
        assert HomogeneityAnalyzer(CES(0.5, 0.5, 0.5)).is_homothetic()

    def test_leontief_is_homothetic(self):
        assert HomogeneityAnalyzer(Leontief(1.0, 1.0)).is_homothetic()

    def test_stone_geary_not_homothetic(self):
        sg = StoneGeary(alpha=0.5, beta=0.5, bar_x=1.0, bar_y=1.0)
        assert not HomogeneityAnalyzer(sg).is_homothetic()


# ---------------------------------------------------------------------------
# HomogeneityAnalyzer.demand_degree_zero()
# ---------------------------------------------------------------------------

class TestDemandDegreeZero:
    def test_cobb_douglas(self):
        cd = CobbDouglas(alpha=0.5, beta=0.5)
        assert HomogeneityAnalyzer(cd).demand_degree_zero(px=2.0, py=3.0, income=60.0)

    def test_leontief(self):
        lf = Leontief(a=1.0, b=1.0)
        assert HomogeneityAnalyzer(lf).demand_degree_zero(px=2.0, py=3.0, income=60.0)

    def test_ces(self):
        ces = CES(alpha=0.5, beta=0.5, rho=0.5)
        assert HomogeneityAnalyzer(ces).demand_degree_zero(px=2.0, py=3.0, income=60.0)

    @pytest.mark.parametrize("scale", [0.5, 2.0, 5.0, 10.0])
    def test_scale_invariance(self, scale):
        cd = CobbDouglas(alpha=0.4, beta=0.6)
        assert HomogeneityAnalyzer(cd).demand_degree_zero(
            px=2.0, py=3.0, income=60.0, scales=(scale,)
        )
