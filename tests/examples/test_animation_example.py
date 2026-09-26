"""Smoke-mode coverage for the animation example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_animation_example():
    path = Path(__file__).parents[2] / "examples" / "scripts" / "animation.py"
    spec = importlib.util.spec_from_file_location("animation_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_mode_renders_one_short_animation(monkeypatch, tmp_path):
    animation = _load_animation_example()
    saved: list[tuple[str, list[float]]] = []

    def record_save(self, path, **kwargs):
        saved.append((str(path), list(self._frames)))

    monkeypatch.setattr(animation, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(animation, "PARAMETER_DIR", str(tmp_path / "parameter_sweeps"))
    monkeypatch.setattr(animation, "PRICE_DIR", str(tmp_path / "price_sweeps"))
    monkeypatch.setattr(animation, "INCOME_DIR", str(tmp_path / "income_sweeps"))
    monkeypatch.setattr(animation.Animator, "save", record_save)

    animation.main(smoke=True)

    assert len(saved) == 1
    assert saved[0][0].endswith("budget_only_price_sweep.gif")
    assert len(saved[0][1]) == 3
