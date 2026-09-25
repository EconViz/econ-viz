"""Shared pytest configuration and fixtures."""

import matplotlib

matplotlib.use("Agg")

import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Prevent figures from leaking between tests or exhausting CI resources."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
