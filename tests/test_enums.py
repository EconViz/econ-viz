"""Tests for econ_viz.enums — ExportFormat and UtilityType."""

import pytest

from econ_viz.enums import ExportFormat, UtilityType
from econ_viz.exceptions import ExportError


class TestExportFormat:
    """Tests for ExportFormat.from_path() — extension inference and validation."""

    def test_from_path_png(self):
        assert ExportFormat.from_path("plot.png") is ExportFormat.PNG

    def test_from_path_pdf(self):
        assert ExportFormat.from_path("output/fig.pdf") is ExportFormat.PDF

    def test_from_path_svg(self):
        assert ExportFormat.from_path("chart.svg") is ExportFormat.SVG

    def test_from_path_uppercase(self):
        assert ExportFormat.from_path("fig.PNG") is ExportFormat.PNG

    def test_from_path_unknown_raises(self):
        with pytest.raises(ExportError, match="Unsupported"):
            ExportFormat.from_path("figure.bmp")

    def test_from_path_tex_raises(self):
        with pytest.raises(ExportError, match="Unsupported"):
            ExportFormat.from_path("diagram.tex")

    def test_from_path_no_extension_raises(self):
        with pytest.raises(ExportError):
            ExportFormat.from_path("noextension")


class TestUtilityType:
    """Sanity checks for the UtilityType enum members."""

    def test_members_exist(self):
        assert UtilityType.SMOOTH
        assert UtilityType.KINKED
        assert UtilityType.LINEAR

    def test_distinct(self):
        assert UtilityType.SMOOTH is not UtilityType.KINKED
        assert UtilityType.KINKED is not UtilityType.LINEAR
