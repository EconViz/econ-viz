"""Shared constants are the single source for defaults used across modules."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from econ_viz import Canvas, Figure, Layout
from econ_viz.constants.canvas import DEFAULT_DPI, MAX_DPI, MIN_DPI
from econ_viz.constants.logging import LIBRARY_ROOT
from econ_viz.consumer.edgeworth import EdgeworthBox
from econ_viz.models import CobbDouglas
from econ_viz.utils.logging import get_logger


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _edgeworth(**kwargs) -> EdgeworthBox:
    return EdgeworthBox(CobbDouglas(), CobbDouglas(), total_x=10.0, total_y=10.0, **kwargs)


def test_dpi_defaults_are_shared():
    assert Canvas().dpi == DEFAULT_DPI
    assert Figure(Layout.SINGLE)[0].dpi == DEFAULT_DPI
    assert _edgeworth().dpi == DEFAULT_DPI


@pytest.mark.parametrize("requested, expected", [(0, MIN_DPI), (10**6, MAX_DPI)])
def test_dpi_is_clamped_to_shared_bounds(requested, expected):
    assert Canvas(dpi=requested).dpi == expected
    assert _edgeworth(dpi=requested).dpi == expected


def test_logger_namespace_uses_library_root():
    assert get_logger("canvas").name == f"{LIBRARY_ROOT}.canvas"
    assert get_logger(f"{LIBRARY_ROOT}.io").name == f"{LIBRARY_ROOT}.io"
