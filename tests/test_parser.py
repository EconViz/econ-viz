"""Tests for econ_viz.parser — LaTeX string → model instance."""

import pytest

from econ_viz.exceptions import ParseError
from econ_viz.models import CobbDouglas, Leontief, PerfectSubstitutes, CES
from econ_viz.parser import parse_latex


class TestParseCobbDouglas:
    """Verify that Cobb-Douglas LaTeX patterns are recognised and parsed correctly."""

    def test_basic(self):
        m = parse_latex(r"x^{0.3} y^{0.7}")
        assert isinstance(m, CobbDouglas)
        assert m.alpha == pytest.approx(0.3)
        assert m.beta == pytest.approx(0.7)

    def test_without_braces(self):
        m = parse_latex(r"x^0.5 y^0.5")
        assert isinstance(m, CobbDouglas)
        assert m.alpha == pytest.approx(0.5)

    def test_with_preamble(self):
        m = parse_latex(r"U(x,y) = x^{0.4} y^{0.6}")
        assert isinstance(m, CobbDouglas)
        assert m.alpha == pytest.approx(0.4)
        assert m.beta == pytest.approx(0.6)

    def test_integer_exponents(self):
        m = parse_latex(r"x^2 y^3")
        assert isinstance(m, CobbDouglas)
        assert m.alpha == pytest.approx(2.0)
        assert m.beta == pytest.approx(3.0)

    def test_evaluates_correctly(self):
        m = parse_latex(r"x^{0.5} y^{0.5}")
        assert m(4.0, 9.0) == pytest.approx(6.0)


class TestParseLeontief:
    """Verify that Leontief (min) LaTeX patterns are recognised and parsed correctly."""

    def test_basic(self):
        m = parse_latex(r"\min(2x, y)")
        assert isinstance(m, Leontief)
        assert m.a == pytest.approx(2.0)
        assert m.b == pytest.approx(1.0)

    def test_both_coefficients(self):
        m = parse_latex(r"\min(x, 3y)")
        assert isinstance(m, Leontief)
        assert m.a == pytest.approx(1.0)
        assert m.b == pytest.approx(3.0)

    def test_u_preamble(self):
        m = parse_latex(r"U = \min(x, 3y)")
        assert isinstance(m, Leontief)
        assert m.b == pytest.approx(3.0)

    def test_no_backslash(self):
        m = parse_latex(r"min(2x, y)")
        assert isinstance(m, Leontief)
        assert m.a == pytest.approx(2.0)

    def test_y_before_x(self):
        """The parser handles the reversed argument order min(by, ax) correctly."""
        m = parse_latex(r"\min(3y, 2x)")
        assert isinstance(m, Leontief)
        assert m.a == pytest.approx(2.0)
        assert m.b == pytest.approx(3.0)

    def test_evaluates_correctly(self):
        """min(2*3, 4) = 4."""
        m = parse_latex(r"\min(2x, y)")
        assert m(3.0, 4.0) == pytest.approx(4.0)


class TestParsePerfectSubstitutes:
    """Verify that linear (ax + by) LaTeX patterns are recognised and parsed correctly."""

    def test_basic(self):
        m = parse_latex(r"3x + 1.5y")
        assert isinstance(m, PerfectSubstitutes)
        assert m.a == pytest.approx(3.0)
        assert m.b == pytest.approx(1.5)

    def test_unit_coefficients(self):
        m = parse_latex(r"x + y")
        assert isinstance(m, PerfectSubstitutes)
        assert m.a == pytest.approx(1.0)
        assert m.b == pytest.approx(1.0)

    def test_with_preamble(self):
        m = parse_latex(r"U(x,y) = 2x + y")
        assert isinstance(m, PerfectSubstitutes)
        assert m.a == pytest.approx(2.0)
        assert m.b == pytest.approx(1.0)

    def test_evaluates_correctly(self):
        m = parse_latex(r"3x + 2y")
        assert m(1.0, 1.0) == pytest.approx(5.0)


class TestParseCES:
    """Verify that CES LaTeX patterns are recognised and parsed correctly."""

    def test_basic_decimal_outer(self):
        """Acceptance-criteria example from issue #3."""
        m = parse_latex(r"(0.5 x^{-0.5} + 0.5 y^{-0.5})^{-2}")
        assert isinstance(m, CES)
        assert m.alpha == pytest.approx(0.5)
        assert m.beta == pytest.approx(0.5)
        assert m.rho == pytest.approx(-0.5)

    def test_fraction_outer(self):
        """Outer exponent expressed as the fraction 1/rho."""
        m = parse_latex(r"(0.4 x^{0.5} + 0.6 y^{0.5})^{1/0.5}")
        assert isinstance(m, CES)
        assert m.alpha == pytest.approx(0.4)
        assert m.beta == pytest.approx(0.6)
        assert m.rho == pytest.approx(0.5)

    def test_positive_rho(self):
        m = parse_latex(r"(0.3 x^{0.5} + 0.7 y^{0.5})^{2}")
        assert isinstance(m, CES)
        assert m.rho == pytest.approx(0.5)

    def test_without_braces(self):
        m = parse_latex(r"(0.5 x^0.5 + 0.5 y^0.5)^2")
        assert isinstance(m, CES)
        assert m.rho == pytest.approx(0.5)

    def test_with_preamble(self):
        m = parse_latex(r"U(x,y) = (0.5 x^{-0.5} + 0.5 y^{-0.5})^{-2}")
        assert isinstance(m, CES)
        assert m.rho == pytest.approx(-0.5)

    def test_mismatched_inner_exponents_raises(self):
        """Inner exponents for x and y must be equal — mismatch must raise ParseError."""
        with pytest.raises(ParseError):
            parse_latex(r"(0.5 x^{0.5} + 0.5 y^{0.3})^{2}")

    def test_evaluates_correctly(self):
        """CES(0.5, 0.5, 0.5) at (1,1) must equal 1."""
        m = parse_latex(r"(0.5 x^{0.5} + 0.5 y^{0.5})^{2}")
        assert m(1.0, 1.0) == pytest.approx(1.0)


class TestParseErrors:
    """Verify that unrecognised patterns raise ParseError."""

    def test_unknown_form_raises(self):
        with pytest.raises(ParseError):
            parse_latex(r"x^2 + y^2")

    def test_empty_string_raises(self):
        with pytest.raises(ParseError):
            parse_latex("")

    def test_gibberish_raises(self):
        with pytest.raises(ParseError):
            parse_latex(r"\frac{x}{y}")
