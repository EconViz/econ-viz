"""Tests for CLI parsing and the solve-tex command."""

from __future__ import annotations

import argparse

import pytest

from econ_viz.cli.main import build_parser, main
from econ_viz.cli.resolve import build_model
from econ_viz.models import QuasiLinear, StoneGeary, Translog


class TestSolveTexCLI:
    def test_build_parser_registers_solve_tex(self):
        _, subparsers = build_parser()
        assert "solve-tex" in subparsers

    def test_solve_tex_prints_formula(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["econ-viz", "solve-tex", "--model", "cobb-douglas", "--alpha", "0.4", "--beta", "0.6"],
        )
        main()
        out = capsys.readouterr().out.strip()
        assert r"x^*" in out
        assert r"\frac{I}{p_x}" in out
        assert r"\frac{I}{p_y}" in out

    def test_solve_tex_accepts_custom_symbols(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "econ-viz", "solve-tex",
                "--model", "cobb-douglas",
                "--px-symbol", "P_x",
                "--py-symbol", "P_y",
                "--income-symbol", "M",
            ],
        )
        main()
        out = capsys.readouterr().out.strip()
        assert r"\frac{M}{P_x}" in out
        assert r"\frac{M}{P_y}" in out

    def test_solve_tex_can_use_symbolic_params(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "econ-viz", "solve-tex",
                "--model", "cobb-douglas",
                "--symbolic-params",
            ],
        )
        main()
        out = capsys.readouterr().out.strip()
        assert r"\frac{\alpha}{\alpha+\beta}" in out
        assert r"\frac{\beta}{\alpha+\beta}" in out

    def test_solve_tex_unsupported_model_exits(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["econ-viz", "solve-tex", "--model", "ces"],
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Closed-form TeX solution is not implemented for CES" in err


class TestBuildModelCLI:
    def test_build_quasi_linear(self):
        args = argparse.Namespace(
            latex=None,
            model="quasi-linear",
            linear_in="x",
            v_func="sqrt",
        )
        model = build_model(args)
        assert isinstance(model, QuasiLinear)
        assert model.linear_in == "x"

    def test_build_stone_geary(self):
        args = argparse.Namespace(
            latex=None,
            model="stone-geary",
            alpha=0.4,
            beta=0.6,
            bar_x=2.0,
            bar_y=3.0,
        )
        model = build_model(args)
        assert isinstance(model, StoneGeary)
        assert model.bar_x == pytest.approx(2.0)
        assert model.bar_y == pytest.approx(3.0)

    def test_build_translog(self):
        args = argparse.Namespace(
            latex=None,
            model="translog",
            alpha_0=0.1,
            alpha_x=0.4,
            alpha_y=0.6,
            beta_xx=0.2,
            beta_yy=-0.1,
            beta_xy=0.05,
        )
        model = build_model(args)
        assert isinstance(model, Translog)
        assert model.alpha_0 == pytest.approx(0.1)
        assert model.beta_xy == pytest.approx(0.05)

    def test_models_command_lists_new_models(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["econ-viz", "models"])
        main()
        out = capsys.readouterr().out
        assert "QuasiLinear" in out
        assert "StoneGeary" in out
        assert "Translog" in out
