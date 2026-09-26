"""Tests for the Haagsma (2012) utility with an inferior / Giffen good."""

import warnings

import numpy as np
import pytest

from econ_viz.exceptions import InvalidParameterError
from econ_viz.models import Haagsma
from econ_viz.optimizer import decompose_price_effect, solve
from econ_viz.optimizer.slutsky import slutsky_matrix


@pytest.fixture(autouse=True)
def quiet_good_type_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*(inferior|Giffen) good.*")
        yield


class TestValues:
    def test_utility_and_domain(self):
        m = Haagsma(alpha_x=1.0, alpha_y=2.0, gamma_x=2.0, gamma_y=27.0)
        assert m(3.0, 26.0) == pytest.approx(0.0)
        assert np.isnan(m(2.0, 10.0))
        assert np.isnan(m(5.0, 27.0))
        grid = m(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
        assert np.isnan(grid[0]) and np.isfinite(grid[1])

    @pytest.mark.parametrize("kwargs", [
        {"alpha_x": 2.0, "alpha_y": 1.0},
        {"alpha_x": 1.0, "alpha_y": 1.0},
        {"gamma_x": 0.0},
        {"gamma_y": -1.0},
        {"alpha_x": float("nan")},
    ])
    def test_rejects_invalid_parameters(self, kwargs):
        with pytest.raises(InvalidParameterError):
            Haagsma(**kwargs)

    def test_bounds(self):
        m = Haagsma(gamma_x=2.0, gamma_y=27.0)
        assert m.lower_bounds() == (2.0, 0.0)
        assert m.upper_bounds() == (np.inf, 27.0)


class TestDemand:
    @pytest.mark.parametrize("px, income", [(2.0, 28.0), (1.0, 28.0), (2.0, 20.0), (1.5, 25.0)])
    def test_solver_matches_closed_form(self, px, income):
        m = Haagsma()
        eq = solve(m, px=px, py=1.0, income=income)
        assert (eq.x, eq.y) == pytest.approx(m.demand(px, 1.0, income), abs=1e-4)

    def test_x_is_inferior(self):
        m = Haagsma()
        low, high = m.demand(2.0, 1.0, 20.0)[0], m.demand(2.0, 1.0, 24.0)[0]
        assert high < low

    def test_giffen_condition(self):
        m = Haagsma(gamma_x=2.0, gamma_y=27.0)
        assert m.is_giffen(px=2.0, py=1.0, income=28.0)
        assert not m.is_giffen(px=2.0, py=1.0, income=26.0)
        assert not m.is_giffen(px=2.0, py=1.0, income=31.0)
        # Giffen: x rises with its own price.
        assert m.demand(2.0, 1.0, 28.0)[0] > m.demand(1.0, 1.0, 28.0)[0]

    def test_unbounded_utility_is_rejected(self):
        with pytest.raises(InvalidParameterError, match="no maximum"):
            solve(Haagsma(gamma_x=2.0, gamma_y=27.0), px=2.0, py=1.0, income=31.0)

    def test_slutsky_matrix_matches_theory(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Slutsky matrix theoretical checks failed.*")
            matrix = slutsky_matrix(Haagsma(), px=2.0, py=1.0, income=28.0)
        np.testing.assert_allclose(matrix.as_array(), [[-1.5, 3.0], [3.0, -6.0]], atol=1e-2)

    def test_giffen_decomposition(self):
        dec = decompose_price_effect(Haagsma(), px=(2.0, 1.0), py=1.0, income=28.0, method="hicks")
        se, ie, te = dec.substitution_effect[0], dec.income_effect[0], dec.total_effect[0]
        assert se > 0 and ie < 0 and te < 0
        assert abs(ie) > se
