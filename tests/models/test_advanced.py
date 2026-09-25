"""Tests for econ_viz.models.advanced — CustomUtility and MultiGoodCD."""

import numpy as np
import pytest

from econ_viz.enums import UtilityType
from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import CustomUtility, MultiGoodCD


class TestCustomUtility:
    """Unit tests for CustomUtility — wrapping and validating arbitrary callables."""

    def test_callable_is_stored(self):
        fn = lambda x, y: x * y
        m = CustomUtility(func=fn)
        assert m.func is fn

    def test_scalar_eval(self):
        m = CustomUtility(func=lambda x, y: x + y)
        assert m(2.0, 3.0) == pytest.approx(5.0)

    def test_array_eval(self):
        m = CustomUtility(func=lambda x, y: x * y)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(m(x, y), x * y)

    def test_utility_type(self):
        m = CustomUtility(func=lambda x, y: x + y)
        assert m.utility_type is UtilityType.SMOOTH

    def test_ray_slopes_empty(self):
        m = CustomUtility(func=lambda x, y: x + y)
        assert m.ray_slopes() == []

    def test_kink_points_empty(self):
        m = CustomUtility(func=lambda x, y: x + y)
        assert m.kink_points([1.0, 2.0]) == []

    def test_custom_name(self):
        m = CustomUtility(func=lambda x, y: x + y, name="my_func")
        assert m.name == "my_func"
        assert "my_func" in repr(m)

    def test_default_name(self):
        m = CustomUtility(func=lambda x, y: x + y)
        assert m.name == "custom"

    def test_numpy_log_sum(self):
        m = CustomUtility(func=lambda x, y: np.log(x) + np.log(y))
        assert m(np.e, np.e) == pytest.approx(2.0)

    def test_non_vectorised_raises(self):
        """A function that calls float() on an array must raise ValueError."""
        with pytest.raises(ValueError, match="NumPy"):
            CustomUtility(func=lambda x, y: float(x) + float(y))

    def test_func_that_crashes_raises(self):
        """A function that raises internally must be caught and re-raised as ValueError."""
        def bad(x, y):
            raise RuntimeError("intentional crash")

        with pytest.raises(ValueError, match="NumPy"):
            CustomUtility(func=bad)

    def test_wrong_shape_raises(self):
        """A function that returns a scalar array of wrong ndim must raise ValueError."""
        def wrong_shape(x, y):
            return np.array([1.0, 2.0])

        with pytest.raises(ValueError):
            CustomUtility(func=wrong_shape)


class TestMultiGoodCD:
    """Unit tests for MultiGoodCD — N-dimensional Cobb-Douglas with freeze projection."""

    def test_repr(self):
        m = MultiGoodCD({'x': 0.5, 'y': 0.5})
        assert "MultiGoodCD" in repr(m)

    def test_utility_type(self):
        m = MultiGoodCD({'x': 0.5, 'y': 0.5})
        assert m.utility_type is UtilityType.SMOOTH

    def test_ray_slopes_empty(self):
        m = MultiGoodCD({'x': 0.5, 'y': 0.5})
        assert m.ray_slopes() == []

    def test_kink_points_empty(self):
        m = MultiGoodCD({'x': 0.5, 'y': 0.5})
        assert m.kink_points([1.0]) == []

    def test_too_few_goods_raises(self):
        with pytest.raises(InvalidParameterError):
            MultiGoodCD({'x': 1.0})

    def test_non_positive_exponent_raises(self):
        with pytest.raises(InvalidParameterError):
            MultiGoodCD({'x': 0.5, 'y': 0.0})

    def test_negative_exponent_raises(self):
        with pytest.raises(InvalidParameterError):
            MultiGoodCD({'x': 0.5, 'y': -0.3})

    def test_evaluate_two_goods(self):
        m = MultiGoodCD({'x': 0.5, 'y': 0.5})
        assert m.evaluate(x=4.0, y=9.0) == pytest.approx(6.0)

    def test_evaluate_three_goods(self):
        m = MultiGoodCD({'x': 1.0, 'y': 1.0, 'z': 1.0})
        assert m.evaluate(x=2.0, y=3.0, z=4.0) == pytest.approx(24.0)

    def test_evaluate_missing_good_raises(self):
        m = MultiGoodCD({'x': 0.5, 'y': 0.5})
        with pytest.raises(ValueError, match="Missing"):
            m.evaluate(x=1.0)

    def test_freeze_returns_custom_utility(self):
        m = MultiGoodCD({'x': 0.3, 'y': 0.3, 'z': 0.4})
        flat = m.freeze(z=10.0)
        assert isinstance(flat, CustomUtility)

    def test_freeze_scalar_eval(self):
        """freeze(z=1) on U=x^1 y^1 z^1 must give U(x,y) = x*y*1."""
        m = MultiGoodCD({'x': 1.0, 'y': 1.0, 'z': 1.0})
        flat = m.freeze(z=1.0)
        assert flat(2.0, 3.0) == pytest.approx(6.0)

    def test_freeze_array_eval(self):
        m = MultiGoodCD({'x': 1.0, 'y': 1.0, 'z': 2.0})
        flat = m.freeze(z=1.0)
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        np.testing.assert_allclose(flat(x, y), x * y * 1.0 ** 2)

    def test_freeze_two_variables_raises(self):
        """Freezing y and z from a 3-good model leaves only x — must raise."""
        m = MultiGoodCD({'x': 0.3, 'y': 0.3, 'z': 0.4})
        with pytest.raises(ValueError, match="2"):
            m.freeze(y=5.0, z=10.0)

    def test_freeze_no_variables_raises(self):
        """Freezing nothing on a 4-good model leaves 4 active — must raise."""
        m = MultiGoodCD({'x': 0.25, 'y': 0.25, 'z': 0.25, 'w': 0.25})
        with pytest.raises(ValueError, match="2"):
            m.freeze()

    def test_freeze_unknown_good_raises(self):
        m = MultiGoodCD({'x': 0.5, 'y': 0.5})
        with pytest.raises(InvalidParameterError):
            m.freeze(q=1.0)

    def test_freeze_four_goods_two_fixed(self):
        """Freezing 2 of 4 goods must leave exactly 2 active variables."""
        m = MultiGoodCD({'x': 0.25, 'y': 0.25, 'z': 0.25, 'w': 0.25})
        flat = m.freeze(z=5.0, w=8.0)
        assert isinstance(flat, CustomUtility)

    def test_freeze_active_variable_order(self):
        """The first two dict keys not in fixed become (x_axis, y_axis)."""
        m = MultiGoodCD({'a': 1.0, 'b': 1.0, 'c': 1.0})
        flat = m.freeze(c=1.0)
        assert flat(2.0, 3.0) == pytest.approx(6.0)

    def test_freeze_contribution_is_correct(self):
        """U = x^0.5 * y^0.5 * z^1 frozen at z=4 should give 4 * x^0.5 * y^0.5."""
        m = MultiGoodCD({'x': 0.5, 'y': 0.5, 'z': 1.0})
        flat = m.freeze(z=4.0)
        assert flat(4.0, 9.0) == pytest.approx(4.0 * 6.0)
