"""Endowment, Walrasian equilibrium and allocation-check methods of the Edgeworth box."""

from __future__ import annotations

from typing import TYPE_CHECKING

from utility_viz.core.diagrams.consumer.edgeworth.exchange import check_allocation, locate_walrasian
from utility_viz.core.diagrams.consumer.edgeworth.methods.lines import LinesMixin
from utility_viz.core.diagrams.consumer.edgeworth.plotting import plot_endowment, plot_equilibrium_marker
from utility_viz.core.rendering.stroke import styled
from utility_viz.core.styles.label import Label, split_label
from utility_viz.core.styles.marker import Marker
from utility_viz.core.styles.stroke import Stroke

if TYPE_CHECKING:
    from utility_viz.core.diagrams.consumer.edgeworth.box import EdgeworthBox


class PointsMixin(LinesMixin):
    """``add_endowment``, ``add_walrasian_equilibrium`` and ``check_point``."""

    def add_endowment(
        self,
        x_endowment: float,
        y_endowment: float,
        *,
        label: str | Label = "e",
        color: str | None = None,
        marker: Marker | None = None,
    ) -> EdgeworthBox:
        """Mark the initial endowment point E.

        *label* is the text, or a :class:`Label` for its text, position,
        colour, and size (default ``theme.edgeworth_label``).
        """
        text, style = split_label(label, self.theme.edgeworth_label, "e")
        assert text is not None
        with styled(self, {}, markers={"endowment": marker}, labels={"endowment_label": style}):
            if not (0.0 <= x_endowment <= self.total_x and 0.0 <= y_endowment <= self.total_y):
                raise ValueError("Endowment must lie inside the Edgeworth box.")

            self.endowment = (float(x_endowment), float(y_endowment))
            endowment_marker = self.theme.endowment_marker
            c = color or endowment_marker.color or self.theme.eq_color
            size = endowment_marker.size if endowment_marker.size is not None else self.theme.eq_markersize
            plot_endowment(
                self.ax,
                x=x_endowment,
                y=y_endowment,
                total_x=self.total_x,
                total_y=self.total_y,
                color=c,
                markersize=size,
                label=text,
            )
        return self._as_box()

    def _walrasian_marker_shape(self, marker: str | Marker | None) -> str:
        """Matplotlib shape from a legacy string, a :class:`Marker`, or the theme default."""
        default_shape = self.theme.walrasian_marker.shape or "*"
        if isinstance(marker, Marker):
            return marker.shape or default_shape
        if isinstance(marker, str):
            return marker
        return default_shape

    def add_walrasian_equilibrium(
        self,
        px: float,
        py: float,
        *,
        color: str | None = None,
        marker: str | Marker | None = None,
        markersize: float | None = None,
        label: str | Label = r"X^*",
        contract_stroke: Stroke | None = None,
    ) -> EdgeworthBox:
        """Approximate Walrasian equilibrium on the budget line and contract curve.

        Parameters
        ----------
        color : str, optional
            Marker colour. *None* → ``theme.walrasian_color``.
        markersize : float, optional
            Marker size. *None* → ``theme.walrasian_markersize``.
        contract_stroke : Stroke, optional
            Line style for the contract curve (default ``theme.contract_stroke``).
        marker : str or Marker, optional
            Marker shape (legacy), or a :class:`Marker` for colour, size, and
            shape (default ``theme.walrasian_marker``).
        label : str or Label
            Label text, or a :class:`Label` for its text, position, colour,
            and size (default ``theme.edgeworth_label``).
        """
        t = self.theme
        text, label_style = split_label(label, self.theme.edgeworth_label, r"X^*")
        assert text is not None
        marker_style = marker if isinstance(marker, Marker) else None
        shape = self._walrasian_marker_shape(marker)
        with styled(
            self,
            {"contract": contract_stroke},
            markers={"walrasian": marker_style},
            labels={"walrasian_label": label_style},
        ):
            x_star, y_star = locate_walrasian(self._as_box(), px, py)

            self.walrasian_equilibrium = (x_star, y_star)
            plot_equilibrium_marker(
                self.ax,
                x=x_star,
                y=y_star,
                total_x=self.total_x,
                total_y=self.total_y,
                color=color or t.walrasian_color,
                marker=shape,
                markersize=markersize if markersize is not None else t.walrasian_markersize,
                label=text,
            )
        return self._as_box()

    def check_point(
        self,
        x: float,
        y: float,
        *,
        px: float | None = None,
        py: float | None = None,
        tol: float = 1e-3,
    ) -> dict[str, bool]:
        """Return key checklist conditions at a candidate allocation."""
        return check_allocation(self._as_box(), x, y, px, py, tol)
