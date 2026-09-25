"""Behavioral tests for consumption-path rendering branches."""

import pytest

from econ_viz import Canvas
from econ_viz.consumer.paths import LinearBudget, PricePath
from econ_viz.models import CobbDouglas


@pytest.fixture()
def price_path():
    return PricePath(
        CobbDouglas(),
        budget=LinearBudget(px=2.0, py=2.0, income=40.0),
        price="px",
        price_range=(1.0, 3.0),
        n=3,
    )


def test_path_can_render_curves_budgets_equilibria_and_legend(price_path):
    canvas = Canvas(x_max=25, y_max=25)

    canvas.add_path(
        price_path,
        label="PCC",
        show_curves=True,
        show_budgets=True,
        show_equilibria=True,
        smooth_curve=True,
    ).show_legend()

    assert canvas.ax.get_legend() is not None
    assert len(canvas.ax.lines) > len(price_path.equilibria)


def test_inverted_path_can_render_raw_points(price_path):
    canvas = Canvas(x_max=25, y_max=25)

    canvas.add_path(
        price_path,
        show_points=True,
        show_budgets=False,
        show_curves=False,
        show_equilibria=False,
        invert_axes=True,
        smooth_curve=False,
    )

    point_lines = [line for line in canvas.ax.lines if line.get_marker() == "o"]
    assert len(point_lines) == len(price_path.equilibria)
    assert point_lines[0].get_xdata()[0] == pytest.approx(price_path.parameter_values[0])
