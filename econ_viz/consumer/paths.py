"""Reusable equilibrium sweeps for PCC, ICC, and derived demand plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..optimizer import Equilibrium, solve

SweepParameter = Literal["px", "py", "income"]
PriceParameter = Literal["px", "py"]
QuantityAxis = Literal["x", "y"]


@dataclass(frozen=True)
class LinearBudget:
    """Linear budget specification ``px*x + py*y = income``."""

    px: float
    py: float
    income: float

    def with_update(self, **kwargs: float) -> "LinearBudget":
        """Return a copy with one or more fields replaced."""
        return LinearBudget(
            px=kwargs.get("px", self.px),
            py=kwargs.get("py", self.py),
            income=kwargs.get("income", self.income),
        )


@dataclass(frozen=True)
class ConsumptionPath:
    """A solved sequence of equilibrium bundles under a one-parameter sweep."""

    func: object
    base_budget: LinearBudget
    parameter_name: SweepParameter
    parameter_values: tuple[float, ...]
    equilibria: tuple[Equilibrium, ...]
    budgets: tuple[LinearBudget, ...]

    @property
    def x_values(self) -> tuple[float, ...]:
        return tuple(eq.x for eq in self.equilibria)

    @property
    def y_values(self) -> tuple[float, ...]:
        return tuple(eq.y for eq in self.equilibria)

    @property
    def px_values(self) -> tuple[float, ...]:
        return tuple(budget.px for budget in self.budgets)

    @property
    def py_values(self) -> tuple[float, ...]:
        return tuple(budget.py for budget in self.budgets)

    @property
    def income_values(self) -> tuple[float, ...]:
        return tuple(budget.income for budget in self.budgets)

    def quantity_values(self, axis: QuantityAxis) -> tuple[float, ...]:
        """Return the swept optimal quantities for one good."""
        return self.x_values if axis == "x" else self.y_values


def _linspace(range_: tuple[float, float], n: int) -> tuple[float, ...]:
    return tuple(float(v) for v in np.linspace(*range_, n))


class PricePath(ConsumptionPath):
    """Sweep one price in a budget and collect the resulting equilibrium path."""

    def __init__(
        self,
        func,
        budget: LinearBudget,
        price: PriceParameter,
        price_range: tuple[float, float],
        n: int = 30,
    ):
        parameter_values = _linspace(price_range, n)
        budgets = tuple(budget.with_update(**{price: value}) for value in parameter_values)
        equilibria = tuple(
            solve(func, px=item.px, py=item.py, income=item.income)
            for item in budgets
        )
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "base_budget", budget)
        object.__setattr__(self, "parameter_name", price)
        object.__setattr__(self, "parameter_values", parameter_values)
        object.__setattr__(self, "equilibria", equilibria)
        object.__setattr__(self, "budgets", budgets)


class IncomePath(ConsumptionPath):
    """Sweep income in a fixed-price budget and collect the resulting ICC."""

    def __init__(
        self,
        func,
        budget: LinearBudget,
        income_range: tuple[float, float],
        n: int = 30,
    ):
        parameter_values = _linspace(income_range, n)
        budgets = tuple(budget.with_update(income=value) for value in parameter_values)
        equilibria = tuple(
            solve(func, px=item.px, py=item.py, income=item.income)
            for item in budgets
        )
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "base_budget", budget)
        object.__setattr__(self, "parameter_name", "income")
        object.__setattr__(self, "parameter_values", parameter_values)
        object.__setattr__(self, "equilibria", equilibria)
        object.__setattr__(self, "budgets", budgets)
