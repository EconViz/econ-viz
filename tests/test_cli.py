"""Tests for CLI parsing and the solve-tex command."""

from __future__ import annotations

import pytest

from econ_viz.cli.main import build_parser, main


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
