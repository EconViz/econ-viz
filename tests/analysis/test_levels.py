"""Tests for econ_viz.levels — utility level spacing strategies."""

import numpy as np
import pytest

from econ_viz import levels


class TestAround:
    """Tests for levels.around() — equilibrium-anchored level spacing."""

    def test_anchor_always_included(self):
        lvls = levels.around(anchor=10.0, n=5)
        assert 10.0 in lvls

    def test_length(self):
        for n in range(1, 8):
            assert len(levels.around(anchor=5.0, n=n)) == n

    def test_n_one(self):
        assert levels.around(anchor=7.0, n=1) == [7.0]

    def test_sorted_ascending(self):
        lvls = levels.around(anchor=6.0, n=5, spread=0.5)
        assert lvls == sorted(lvls)

    def test_all_positive(self):
        lvls = levels.around(anchor=1.0, n=7, spread=0.9)
        assert all(v > 0 for v in lvls)

    def test_spread_range(self):
        anchor, spread = 10.0, 0.5
        lvls = levels.around(anchor, n=5, spread=spread)
        assert min(lvls) >= anchor * (1 - spread) - 1e-9
        assert max(lvls) <= anchor * (1 + spread) + 1e-9

    def test_even_n(self):
        lvls = levels.around(anchor=4.0, n=4)
        assert len(lvls) == 4
        assert 4.0 in lvls


class TestPercentile:
    """Tests for levels.percentile() — percentile-based level spacing."""

    def test_length(self):
        x = np.linspace(0.1, 10, 100)
        y = np.linspace(0.1, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = X * Y
        for n in [3, 5, 7]:
            assert len(levels.percentile(Z, n=n)) == n

    def test_values_within_data_range(self):
        Z = np.array([[1.0, 2.0], [3.0, 4.0]])
        lvls = levels.percentile(Z, n=3)
        assert all(1.0 <= v <= 4.0 for v in lvls)

    def test_sorted_ascending(self):
        Z = np.random.default_rng(0).random((50, 50))
        lvls = levels.percentile(Z, n=5)
        assert lvls == sorted(lvls)
