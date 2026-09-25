"""Tests for econ_viz.models.core — all utility function classes."""

import numpy as np
import pytest

from econ_viz.enums import UtilityType
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import (
    CobbDouglas,
    Leontief,
    PerfectSubstitutes,
    CES,
    Satiation,
    QuasiLinear,
    StoneGeary,
    Translog,
)


class TestCobbDouglas:
    """Unit tests for the CobbDouglas utility model."""

    def test_scalar(self):
        cd = CobbDouglas(alpha=0.5, beta=0.5)
        assert cd(4.0, 9.0) == pytest.approx(6.0)

    def test_array(self):
        cd = CobbDouglas(alpha=1.0, beta=1.0)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(cd(x, y), x * y)

    def test_utility_type(self):
        assert CobbDouglas().utility_type is UtilityType.SMOOTH

    def test_ray_slopes(self):
        cd = CobbDouglas(alpha=0.4, beta=0.6)
        assert cd.ray_slopes() == pytest.approx([0.6 / 0.4])

    def test_kink_points_empty(self):
        assert CobbDouglas().kink_points([1.0, 2.0]) == []

    def test_defaults(self):
        cd = CobbDouglas()
        assert cd.alpha == 0.5
        assert cd.beta == 0.5


class TestLeontief:
    """Unit tests for the Leontief (perfect complements) utility model."""

    def test_scalar(self):
        lf = Leontief(a=1.0, b=1.0)
        assert lf(3.0, 5.0) == pytest.approx(3.0)

    def test_array(self):
        """min(2x, y) over x=[1,2], y=[3,3] → [min(2,3), min(4,3)] = [2, 3]."""
        lf = Leontief(a=2.0, b=1.0)
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 3.0])
        np.testing.assert_allclose(lf(x, y), [2.0, 3.0])

    def test_utility_type(self):
        assert Leontief().utility_type is UtilityType.KINKED

    def test_ray_slopes(self):
        lf = Leontief(a=2.0, b=3.0)
        assert lf.ray_slopes() == pytest.approx([2.0 / 3.0])

    def test_kink_points(self):
        lf = Leontief(a=2.0, b=4.0)
        pts = lf.kink_points([4.0, 8.0])
        assert pts == pytest.approx([(2.0, 1.0), (4.0, 2.0)])

    def test_kink_points_empty_levels(self):
        assert Leontief().kink_points([]) == []


class TestPerfectSubstitutes:
    """Unit tests for the PerfectSubstitutes (linear) utility model."""

    def test_scalar(self):
        ps = PerfectSubstitutes(a=2.0, b=3.0)
        assert ps(1.0, 1.0) == pytest.approx(5.0)

    def test_array(self):
        ps = PerfectSubstitutes(a=1.0, b=2.0)
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        np.testing.assert_allclose(ps(x, y), [3.0, 6.0])

    def test_utility_type(self):
        assert PerfectSubstitutes().utility_type is UtilityType.LINEAR

    def test_ray_slopes(self):
        ps = PerfectSubstitutes(a=3.0, b=1.5)
        assert ps.ray_slopes() == pytest.approx([2.0])

    def test_kink_points_empty(self):
        assert PerfectSubstitutes().kink_points([1.0]) == []


class TestCES:
    """Unit tests for the CES (Constant Elasticity of Substitution) utility model."""

    def test_scalar(self):
        ces = CES(alpha=0.5, beta=0.5, rho=0.5)
        assert ces(1.0, 1.0) == pytest.approx(1.0)

    def test_utility_type(self):
        assert CES().utility_type is UtilityType.SMOOTH

    def test_ray_slopes_general(self):
        """Symmetric CES (alpha=beta) must yield a slope of exactly 1."""
        ces = CES(alpha=0.5, beta=0.5, rho=0.5)
        slopes = ces.ray_slopes()
        assert len(slopes) == 1
        assert slopes[0] == pytest.approx(1.0)

    def test_ray_slopes_cd_limit(self):
        """At rho → 0 the CES collapses to Cobb-Douglas; slope must equal beta/alpha."""
        ces = CES(alpha=0.4, beta=0.6, rho=1e-10)
        slopes = ces.ray_slopes()
        assert slopes[0] == pytest.approx(0.6 / 0.4, rel=1e-3)

    def test_ray_slopes_rho_one_raises(self):
        ces = CES(rho=1.0)
        with pytest.raises(InvalidParameterError):
            ces.ray_slopes()

    def test_kink_points_empty(self):
        assert CES().kink_points([1.0]) == []


class TestSatiation:
    """Unit tests for the Satiation (bliss-point) utility model."""

    def test_bliss_point_is_maximum(self):
        """Utility at the bliss point must equal 0 and exceed any other bundle."""
        s = Satiation(bliss_x=5.0, bliss_y=5.0, a=1.0, b=1.0)
        u_bliss = s(5.0, 5.0)
        u_away = s(3.0, 3.0)
        assert u_bliss == pytest.approx(0.0)
        assert u_bliss > u_away

    def test_scalar(self):
        s = Satiation(bliss_x=5.0, bliss_y=5.0, a=1.0, b=1.0)
        assert s(4.0, 4.0) == pytest.approx(-2.0)

    def test_array(self):
        s = Satiation(bliss_x=0.0, bliss_y=0.0, a=1.0, b=1.0)
        x = np.array([1.0, 2.0])
        y = np.array([0.0, 0.0])
        np.testing.assert_allclose(s(x, y), [-1.0, -4.0])

    def test_utility_type(self):
        assert Satiation().utility_type is UtilityType.SMOOTH

    def test_ray_slope(self):
        s = Satiation(bliss_x=4.0, bliss_y=8.0)
        assert s.ray_slopes() == pytest.approx([2.0])

    def test_ray_slope_bliss_x_zero(self):
        """No meaningful ray slope when the bliss point lies on the y-axis."""
        s = Satiation(bliss_x=0.0, bliss_y=5.0)
        assert s.ray_slopes() == []

    def test_kink_points_empty(self):
        assert Satiation().kink_points([1.0]) == []

    def test_invalid_a(self):
        with pytest.raises(InvalidParameterError):
            Satiation(a=0.0)

    def test_invalid_b(self):
        with pytest.raises(InvalidParameterError):
            Satiation(b=-1.0)

    def test_defaults(self):
        s = Satiation()
        assert s.bliss_x == 5.0
        assert s.bliss_y == 5.0


class TestStoneGeary:
    """Unit tests for the StoneGeary utility model."""

    def test_scalar(self):
        """With bar_x=bar_y=0 Stone-Geary reduces to Cobb-Douglas."""
        sg = StoneGeary(alpha=0.5, beta=0.5, bar_x=0.0, bar_y=0.0)
        assert sg(4.0, 9.0) == pytest.approx(6.0)

    def test_array(self):
        sg = StoneGeary(alpha=1.0, beta=1.0, bar_x=1.0, bar_y=1.0)
        x = np.array([2.0, 3.0])
        y = np.array([2.0, 4.0])
        np.testing.assert_allclose(sg(x, y), [1.0, 6.0])

    def test_utility_type(self):
        assert StoneGeary().utility_type is UtilityType.SMOOTH

    def test_ray_slopes_empty(self):
        """Expansion path does not pass through the origin; no ray to draw."""
        assert StoneGeary().ray_slopes() == []

    def test_kink_points_empty(self):
        assert StoneGeary().kink_points([1.0, 2.0]) == []

    def test_lower_bounds(self):
        sg = StoneGeary(bar_x=2.0, bar_y=3.0)
        assert sg.lower_bounds() == (2.0, 3.0)

    def test_subsistence_lines(self):
        sg = StoneGeary(bar_x=2.0, bar_y=3.0)
        assert sg.subsistence_lines() == (2.0, 3.0)

    def test_outside_domain_returns_nan(self):
        """Bundles at or below subsistence must return NaN."""
        sg = StoneGeary(bar_x=1.0, bar_y=1.0)
        assert np.isnan(sg(0.5, 5.0))
        assert np.isnan(sg(5.0, 0.5))
        assert np.isnan(sg(1.0, 5.0))   # x exactly at bar_x

    def test_defaults(self):
        sg = StoneGeary()
        assert sg.alpha == 0.5
        assert sg.beta == 0.5
        assert sg.bar_x == 1.0
        assert sg.bar_y == 1.0

    def test_invalid_alpha(self):
        with pytest.raises(InvalidParameterError):
            StoneGeary(alpha=0.0)

    def test_invalid_beta(self):
        with pytest.raises(InvalidParameterError):
            StoneGeary(beta=-0.5)

    def test_invalid_bar_x(self):
        with pytest.raises(InvalidParameterError):
            StoneGeary(bar_x=-1.0)

    def test_invalid_bar_y(self):
        with pytest.raises(InvalidParameterError):
            StoneGeary(bar_y=-0.1)


class TestQuasiLinear:
    """Unit tests for the QuasiLinear utility model and its v_func validator."""

    def test_linear_in_y_default(self):
        """Default form is U = f(x) + y with f = log."""
        ql = QuasiLinear()
        assert ql(1.0, 2.0) == pytest.approx(np.log(1.0) + 2.0)

    def test_linear_in_x(self):
        """When linear_in='x' the form becomes U = x + f(y)."""
        ql = QuasiLinear(v_func=np.log, linear_in="x")
        assert ql(2.0, 1.0) == pytest.approx(2.0 + np.log(1.0))

    def test_array_output(self):
        ql = QuasiLinear(v_func=np.sqrt, linear_in="y")
        x = np.array([1.0, 4.0, 9.0])
        y = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(ql(x, y), [1.0, 2.0, 3.0])

    def test_utility_type(self):
        assert QuasiLinear().utility_type is UtilityType.SMOOTH

    def test_ray_slopes_empty(self):
        assert QuasiLinear().ray_slopes() == []

    def test_kink_points_empty(self):
        assert QuasiLinear().kink_points([1.0]) == []

    def test_invalid_linear_in(self):
        with pytest.raises(InvalidParameterError):
            QuasiLinear(linear_in="z")

    def test_convex_v_func_raises(self):
        """z**2 has f'' > 0 — must be rejected as non-concave."""
        with pytest.raises(ValueError, match="diminishing marginal utility violated"):
            QuasiLinear(v_func=lambda z: z ** 2)

    def test_decreasing_v_func_raises(self):
        """A decreasing function violates the monotonicity assumption."""
        with pytest.raises(ValueError, match="monotonicity violated"):
            QuasiLinear(v_func=lambda z: -z)

    def test_custom_concave_func(self):
        """sqrt is valid: f' > 0 and f'' < 0 everywhere."""
        ql = QuasiLinear(v_func=np.sqrt)
        assert ql(4.0, 1.0) == pytest.approx(3.0)


class TestTranslog:
    """Unit tests for the Translog utility model."""

    def test_reduces_to_cobb_douglas(self):
        """With all beta=0 and alpha_0=0, Translog = x^alpha_x * y^alpha_y."""
        tl = Translog(alpha_x=0.4, alpha_y=0.6)
        cd = CobbDouglas(alpha=0.4, beta=0.6)
        assert tl(3.0, 5.0) == pytest.approx(cd(3.0, 5.0), rel=1e-6)

    def test_scalar_positive(self):
        tl = Translog(alpha_x=0.5, alpha_y=0.5)
        assert tl(2.0, 3.0) > 0

    def test_array_input(self):
        tl = Translog(alpha_x=0.5, alpha_y=0.5)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = tl(x, y)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_alpha_0_scales_utility(self):
        """alpha_0 > 0 multiplies utility by exp(alpha_0)."""
        tl0 = Translog(alpha_x=0.5, alpha_y=0.5, alpha_0=0.0)
        tl1 = Translog(alpha_x=0.5, alpha_y=0.5, alpha_0=1.0)
        assert tl1(2.0, 3.0) == pytest.approx(tl0(2.0, 3.0) * np.e, rel=1e-6)

    def test_beta_xx_increases_curvature(self):
        """Positive beta_xx increases utility when ln(x) > 0 (x > 1)."""
        tl_flat = Translog(alpha_x=0.5, alpha_y=0.5, beta_xx=0.0)
        tl_curved = Translog(alpha_x=0.5, alpha_y=0.5, beta_xx=0.2)
        # At x=3 > 1: ln(3)>0, so positive beta_xx increases utility
        assert tl_curved(3.0, 2.0) > tl_flat(3.0, 2.0)

    def test_utility_type_smooth(self):
        assert Translog().utility_type is UtilityType.SMOOTH

    def test_ray_slopes_empty(self):
        assert Translog().ray_slopes() == []

    def test_kink_points_empty(self):
        assert Translog().kink_points([1.0, 2.0]) == []

    def test_invalid_alpha_x(self):
        with pytest.raises(InvalidParameterError):
            Translog(alpha_x=0.0, alpha_y=0.5)

    def test_invalid_alpha_y(self):
        with pytest.raises(InvalidParameterError):
            Translog(alpha_x=0.5, alpha_y=-0.1)

    def test_defaults(self):
        tl = Translog()
        assert tl.alpha_x == 0.5
        assert tl.alpha_y == 0.5
        assert tl.beta_xx == 0.0
        assert tl.beta_yy == 0.0
        assert tl.beta_xy == 0.0
        assert tl.alpha_0 == 0.0
