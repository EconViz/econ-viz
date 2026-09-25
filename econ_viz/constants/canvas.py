"""Drawing defaults shared by Canvas, Figure, and other diagram types."""

MIN_DPI = 1
MAX_DPI = 1200
DEFAULT_DPI = 300

# Characters that mark a label fragment as LaTeX math.
MATH_CHARS = frozenset({"^", "_", "{", "}", "\\"})

# Lower bound of the utility contour grid; keeps log- and power-based models finite.
CONTOUR_DOMAIN_MIN = 0.1

# Path smoothing and endpoint extension for PCC/ICC curves.
SMOOTH_SAMPLES = 200
ENDPOINT_EXTENSION_FRAC = 0.025

# Arrowhead patches cover only the head, except the wedge, whose shape runs
# along the whole patch (fractions of the line or axis length).
ARROW_HEAD_ONLY_FRAC = 0.001
ARROW_WEDGE_FRAC = 0.04
