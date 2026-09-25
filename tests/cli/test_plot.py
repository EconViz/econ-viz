"""Behavioral coverage for the plot and help CLI commands."""

from __future__ import annotations

import pytest

from econ_viz import Canvas
from econ_viz.cli.main import main


def _run(monkeypatch, *args: str) -> None:
    monkeypatch.setattr("sys.argv", ["econ-viz", *args])
    main()


class TestPlotCommand:
    def test_saves_complete_equilibrium_diagram(self, monkeypatch, tmp_path, capsys):
        output = tmp_path / "equilibrium.png"

        _run(
            monkeypatch,
            "plot",
            "--model", "cobb-douglas",
            "--px", "2",
            "--py", "3",
            "--income", "30",
            "--x-max", "20",
            "--y-max", "20",
            "--fill",
            "--show-ray",
            "--output", str(output),
        )

        assert output.exists()
        assert f"Saved to {output}" in capsys.readouterr().out

    def test_saves_curves_without_budget_inputs(self, monkeypatch, tmp_path):
        output = tmp_path / "curves.svg"

        _run(
            monkeypatch,
            "plot",
            "--latex", r"x^{0.5} y^{0.5}",
            "--n-curves", "3",
            "--output", str(output),
        )

        assert output.exists()

    def test_can_display_minimal_plot(self, monkeypatch):
        shown = []
        monkeypatch.setattr(Canvas, "show", lambda self: shown.append(self))

        _run(
            monkeypatch,
            "plot",
            "--model", "leontief",
            "--no-curves",
            "--no-budget",
            "--no-equilibrium",
        )

        assert len(shown) == 1

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (("plot", "--output", "unused.png"), "provide --model"),
            (("plot", "--model", "missing"), "unknown model"),
            (("plot", "--model", "cobb-douglas", "--theme", "missing"), "unknown theme"),
        ],
    )
    def test_reports_configuration_errors(self, monkeypatch, capsys, args, message):
        monkeypatch.setattr("sys.argv", ["econ-viz", *args])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        assert message in capsys.readouterr().err


class TestHelpCommand:
    def test_prints_root_help(self, monkeypatch, capsys):
        _run(monkeypatch, "help")
        assert "Produce publication-quality" in capsys.readouterr().out

    def test_prints_command_help(self, monkeypatch, capsys):
        _run(monkeypatch, "help", "plot")
        assert "--n-curves" in capsys.readouterr().out

    def test_unknown_topic_reports_available_commands(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["econ-viz", "help", "missing"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2
        assert "unknown command 'missing'" in capsys.readouterr().err
