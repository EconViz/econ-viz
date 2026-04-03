"""Model registry for CLI and other dynamic model resolution flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import (
    CES,
    CobbDouglas,
    Leontief,
    PerfectSubstitutes,
    QuasiLinear,
    Satiation,
    StoneGeary,
    Translog,
)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and constructor for one registered utility model."""

    name: str
    builder: Callable[[object], object]


def _build_cobb_douglas(args) -> CobbDouglas:
    return CobbDouglas(
        alpha=args.alpha if args.alpha is not None else 0.5,
        beta=args.beta if args.beta is not None else 0.5,
    )


def _build_leontief(args) -> Leontief:
    return Leontief(
        a=args.a if args.a is not None else 1.0,
        b=args.b if args.b is not None else 1.0,
    )


def _build_perfect_substitutes(args) -> PerfectSubstitutes:
    return PerfectSubstitutes(
        a=args.a if args.a is not None else 1.0,
        b=args.b if args.b is not None else 1.0,
    )


def _build_ces(args) -> CES:
    return CES(
        rho=args.rho if args.rho is not None else 0.5,
        alpha=args.alpha if args.alpha is not None else 0.5,
        beta=args.beta if args.beta is not None else 0.5,
    )


def _build_satiation(args) -> Satiation:
    return Satiation(
        bliss_x=args.bliss_x if args.bliss_x is not None else 5.0,
        bliss_y=args.bliss_y if args.bliss_y is not None else 5.0,
        a=args.a if args.a is not None else 1.0,
        b=args.b if args.b is not None else 1.0,
    )


def _build_quasi_linear(args) -> QuasiLinear:
    v_func_name = args.v_func
    if v_func_name in (None, "log"):
        v_func = np.log
    elif v_func_name == "sqrt":
        v_func = np.sqrt
    else:
        raise ValueError(f"unknown QuasiLinear v_func '{v_func_name}'. Available: log, sqrt")

    return QuasiLinear(
        v_func=v_func,
        linear_in=args.linear_in if args.linear_in is not None else "y",
    )


def _build_stone_geary(args) -> StoneGeary:
    return StoneGeary(
        alpha=args.alpha if args.alpha is not None else 0.5,
        beta=args.beta if args.beta is not None else 0.5,
        bar_x=args.bar_x if args.bar_x is not None else 1.0,
        bar_y=args.bar_y if args.bar_y is not None else 1.0,
    )


def _build_translog(args) -> Translog:
    return Translog(
        alpha_0=args.alpha_0 if args.alpha_0 is not None else 0.0,
        alpha_x=args.alpha_x if args.alpha_x is not None else 0.5,
        alpha_y=args.alpha_y if args.alpha_y is not None else 0.5,
        beta_xx=args.beta_xx if args.beta_xx is not None else 0.0,
        beta_yy=args.beta_yy if args.beta_yy is not None else 0.0,
        beta_xy=args.beta_xy if args.beta_xy is not None else 0.0,
    )


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "cobb-douglas": ModelSpec(name="cobb-douglas", builder=_build_cobb_douglas),
    "leontief": ModelSpec(name="leontief", builder=_build_leontief),
    "perfect-substitutes": ModelSpec(name="perfect-substitutes", builder=_build_perfect_substitutes),
    "ces": ModelSpec(name="ces", builder=_build_ces),
    "satiation": ModelSpec(name="satiation", builder=_build_satiation),
    "quasi-linear": ModelSpec(name="quasi-linear", builder=_build_quasi_linear),
    "stone-geary": ModelSpec(name="stone-geary", builder=_build_stone_geary),
    "translog": ModelSpec(name="translog", builder=_build_translog),
}


def get_model_registry() -> dict[str, ModelSpec]:
    """Return a copy of the registered model map."""
    return dict(_MODEL_REGISTRY)


def build_registered_model(name: str, args):
    """Build a model from the registry by canonical name."""
    spec = _MODEL_REGISTRY.get(name)
    if spec is None:
        raise KeyError(name)
    return spec.builder(args)

