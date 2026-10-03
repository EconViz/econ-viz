"""Economic models: utility functions, curves, consumer paths, optimisation and analysis.

This facade re-exports the documented model API. Implementation lives in the
``utility``, ``curves``, ``consumer``, ``optimization`` and ``analysis``
subpackages. Models never import from :mod:`econ_viz.core` (drawing layer).
"""

from econ_viz.models.analysis import HomogeneityAnalyzer, HomogeneityResult
from econ_viz.models.consumer import (
    ConsumptionPath,
    EdgeworthState,
    IncomePath,
    LinearBudget,
    PricePath,
)
from econ_viz.models.curves import Layer, around_anchor_levels, percentile_levels
from econ_viz.models.optimization import (
    ComparativeStatics,
    DecompositionMethod,
    Equilibrium,
    PriceEffectDecomposition,
    SlutskyMatrix,
    comparative_statics,
    decompose_price_effect,
    slutsky_matrix,
    solution_tex,
    solve,
)
from econ_viz.models.utility import (
    CES,
    CobbDouglas,
    CustomUtility,
    Haagsma,
    Leontief,
    MultiGoodCD,
    PerfectSubstitutes,
    QuasiLinear,
    Satiation,
    StoneGeary,
    Translog,
    UtilityFunction,
    build_registered_model,
    get_model_registry,
    parse_latex,
)

__all__ = [
    "CES",
    "CobbDouglas",
    "ComparativeStatics",
    "ConsumptionPath",
    "CustomUtility",
    "DecompositionMethod",
    "EdgeworthState",
    "Equilibrium",
    "Haagsma",
    "HomogeneityAnalyzer",
    "HomogeneityResult",
    "IncomePath",
    "Layer",
    "Leontief",
    "LinearBudget",
    "MultiGoodCD",
    "PerfectSubstitutes",
    "PriceEffectDecomposition",
    "PricePath",
    "QuasiLinear",
    "Satiation",
    "SlutskyMatrix",
    "StoneGeary",
    "Translog",
    "UtilityFunction",
    "around_anchor_levels",
    "build_registered_model",
    "comparative_statics",
    "decompose_price_effect",
    "get_model_registry",
    "parse_latex",
    "percentile_levels",
    "slutsky_matrix",
    "solution_tex",
    "solve",
]
