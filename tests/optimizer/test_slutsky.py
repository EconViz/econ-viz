"""Slutsky matrix accuracy."""

import numpy as np

from econ_viz.optimizer.slutsky import slutsky_matrix


def test_slutsky_matrix_is_accurate_for_haagsma_inferior_demand():
    """Solver precision regression: Haagsma demand has a closed form."""
    import warnings

    from econ_viz.models.advanced import CustomUtility

    def utility(x, y, d=27.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((x > 2.0) & (y < d), np.log(x - 2.0) - 2.0 * np.log(d - y), np.nan)

    model = CustomUtility(func=utility, name="Haagsma")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*(inferior|Giffen) good.*")
        warnings.filterwarnings("error", message="Slutsky matrix theoretical checks failed.*")
        matrix = slutsky_matrix(model, px=2.0, py=1.0, income=28.0)
    # x = (27 p_y - I) / p_x + 4 gives S = [[-1.5, 3], [3, -6]] at (2, 1, 28).
    np.testing.assert_allclose(matrix.as_array(), [[-1.5, 3.0], [3.0, -6.0]], atol=1e-2)
