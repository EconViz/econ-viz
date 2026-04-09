"""WidgetViewer — ipywidgets-based interactive slider viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Union

if TYPE_CHECKING:  # pragma: no cover
    from econ_viz.canvas.base import Canvas
    from econ_viz.canvas.figure import Figure

    AnyFigure = Union[Canvas, Figure]

# Slider spec: (min, max, step)
SliderSpec = tuple[float, float, float]


def _require_widgets() -> None:
    """Raise a helpful ImportError when ipywidgets / IPython are absent."""
    missing = []
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        missing.append("ipywidgets")
    try:
        import IPython  # noqa: F401
    except ImportError:
        missing.append("IPython")

    if missing:
        pkg_list = " ".join(missing)
        raise ImportError(
            f"Interactive widgets require: {pkg_list}\n"
            "Install them with:  pip install econ-viz[interactive]\n"
            "                or:  pip install ipywidgets"
        )


class WidgetViewer:
    """Render an econ_viz diagram inside a Jupyter notebook with live sliders.

    Pass a drawing function and a slider specification for each parameter.
    When a slider is moved the old figure is cleared
    (``clear_output(wait=True)``) and the diagram is redrawn immediately,
    so the notebook cell never accumulates a stack of repeated images.

    Parameters
    ----------
    draw_func : Callable[..., Canvas | Figure]
        A function whose keyword-argument names match the keys in
        *slider_specs* and that returns a fully populated Canvas or Figure.
    **slider_specs : tuple[float, float, float]
        One ``(min, max, step)`` triple per parameter.  Every key must
        correspond to a keyword argument accepted by *draw_func*.

    Examples
    --------
    .. code-block:: python

        from econ_viz.interactive import WidgetViewer
        from econ_viz import Canvas
        from econ_viz.models import CobbDouglas

        def draw(p1: float, income: float) -> Canvas:
            c = Canvas(x_max=10, y_max=10, x_label="X_1", y_label="X_2")
            c.add_budget(px=p1, py=2.0, income=income)
            c.add_utility(CobbDouglas(0.5, 0.5), levels=5)
            return c

        WidgetViewer(draw, p1=(1.0, 8.0, 0.5), income=(10.0, 40.0, 5.0)).show()
    """

    def __init__(
        self,
        draw_func: Callable[..., AnyFigure],
        **slider_specs: SliderSpec,
    ) -> None:
        if not slider_specs:
            raise ValueError(
                "Provide at least one slider spec, e.g. p1=(1.0, 8.0, 0.5)."
            )
        self._draw_func = draw_func
        self._slider_specs: dict[str, SliderSpec] = slider_specs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Render the widget panel in the active Jupyter output cell.

        Raises
        ------
        ImportError
            If ``ipywidgets`` or ``IPython`` are not installed.
        RuntimeError
            If called outside of a Jupyter / IPython environment.
        """
        _require_widgets()

        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        sliders = self._build_sliders(widgets)
        out = widgets.Output()

        # Draw the initial frame before any slider interaction.
        self._redraw(
            out=out,
            values={name: s.value for name, s in sliders.items()},
            plt=plt,
            clear_output=clear_output,
        )

        # Wire each slider to the redraw callback.
        def _on_change(_change: Any) -> None:
            self._redraw(
                out=out,
                values={name: s.value for name, s in sliders.items()},
                plt=plt,
                clear_output=clear_output,
            )

        for slider in sliders.values():
            slider.observe(_on_change, names="value")

        # Lay out: sliders on top, diagram output below.
        slider_box = widgets.VBox(
            list(sliders.values()),
            layout=widgets.Layout(margin="0 0 12px 0"),
        )
        display(widgets.VBox([slider_box, out]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sliders(self, widgets) -> dict[str, Any]:
        """Construct a FloatSlider for every slider spec."""
        sliders: dict[str, Any] = {}
        for name, (lo, hi, step) in self._slider_specs.items():
            mid = lo + (hi - lo) / 2
            # Snap mid to the nearest step boundary.
            mid = round(round((mid - lo) / step) * step + lo, 10)
            sliders[name] = widgets.FloatSlider(
                value=mid,
                min=lo,
                max=hi,
                step=step,
                description=f"{name}:",
                continuous_update=False,
                style={"description_width": "initial"},
                layout=widgets.Layout(width="400px"),
            )
        return sliders

    def _redraw(
        self,
        out,
        values: dict[str, float],
        plt,
        clear_output: Callable,
    ) -> None:
        """Clear the output widget and render a fresh diagram."""
        with out:
            clear_output(wait=True)
            canvas_or_fig = self._draw_func(**values)
            display(canvas_or_fig.fig)
            plt.close(canvas_or_fig.fig)
