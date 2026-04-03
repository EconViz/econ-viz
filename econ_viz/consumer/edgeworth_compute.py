"""Pure compute helpers for Edgeworth box analytics."""

from __future__ import annotations

import numpy as np


def mrs(func, x: float, y: float, *, x_max: float, y_max: float, eps: float, h: float = 1e-4) -> float:
    """Approximate marginal rate of substitution by central differences."""
    x0 = min(max(float(x), eps), x_max - eps)
    y0 = min(max(float(y), eps), y_max - eps)
    ux = (float(func(x0 + h, y0)) - float(func(x0 - h, y0))) / (2.0 * h)
    uy = (float(func(x0, y0 + h)) - float(func(x0, y0 - h))) / (2.0 * h)
    if not np.isfinite(ux) or not np.isfinite(uy) or abs(uy) < 1e-9:
        return np.nan
    return ux / uy


def unique_points(points: list[tuple[float, float]], *, digits: int = 4) -> np.ndarray:
    """Sort and deduplicate point pairs."""
    if not points:
        return np.empty((0, 2), dtype=float)
    rounded = {(round(float(x), digits), round(float(y), digits)) for x, y in points}
    return np.array(sorted(rounded, key=lambda p: p[0]), dtype=float)


def focus_levels(
    *,
    anchor: float,
    u_min: float,
    u_max: float,
    n: int,
    spread: float,
    extra: float | None = None,
) -> list[float]:
    """Generate robust contour levels around an anchor utility."""
    anchor_f = float(anchor)
    u_min_f = float(u_min)
    u_max_f = float(u_max)
    if not np.isfinite(anchor_f) or not np.isfinite(u_min_f) or not np.isfinite(u_max_f):
        return [float(anchor)]
    if u_max_f <= u_min_f:
        return [float(anchor_f)]

    pad = max(abs(anchor_f), 1.0) * 1e-6
    lo_bound = u_min_f + pad
    hi_bound = u_max_f - pad
    if hi_bound <= lo_bound:
        return [float(np.clip(anchor_f, u_min_f, u_max_f))]

    width = max(abs(anchor_f), 1.0) * max(spread, 1e-6)
    lo = max(lo_bound, anchor_f - width)
    hi = min(hi_bound, anchor_f + width)
    if hi - lo <= 1e-12:
        full = hi_bound - lo_bound
        if full <= 1e-12:
            return [float(np.clip(anchor_f, u_min_f, u_max_f))]
        interval = min(max(2.0 * width, full * 0.2), full)
        lo = min(max(anchor_f - 0.5 * interval, lo_bound), hi_bound - interval)
        hi = lo + interval

    levels = np.linspace(lo, hi, n).tolist()
    if extra is not None and np.isfinite(extra):
        levels.append(float(np.clip(extra, lo_bound, hi_bound)))

    unique_levels: list[float] = []
    for lv in sorted(float(v) for v in levels if np.isfinite(v)):
        if not unique_levels or not np.isclose(unique_levels[-1], lv, rtol=1e-6, atol=1e-9):
            unique_levels.append(lv)
    return unique_levels


def contract_curve_mrs(
    *,
    utility_a,
    utility_b,
    total_x: float,
    total_y: float,
    n: int,
    tolerance: float,
    eps: float,
) -> np.ndarray:
    """Compute contract-curve points by MRS matching on a grid."""
    x_grid = np.linspace(eps, total_x - eps, n)
    y_grid = np.linspace(eps, total_y - eps, max(3 * n, 180))
    points: list[tuple[float, float]] = []

    for x in x_grid:
        best_y: float | None = None
        best_score = np.inf
        for y in y_grid:
            mrs_a = mrs(utility_a, x, y, x_max=total_x, y_max=total_y, eps=eps)
            mrs_b = mrs(
                utility_b,
                total_x - x,
                total_y - y,
                x_max=total_x,
                y_max=total_y,
                eps=eps,
            )
            if not np.isfinite(mrs_a) or not np.isfinite(mrs_b) or mrs_a <= 0 or mrs_b <= 0:
                continue
            score = abs(np.log(mrs_a) - np.log(mrs_b))
            if score < best_score:
                best_score = score
                best_y = float(y)
        if best_y is not None and best_score <= tolerance:
            points.append((float(x), best_y))
    return unique_points(points)


def contract_curve_pareto(
    *,
    eval_ua,
    eval_ub,
    total_x: float,
    total_y: float,
    n: int,
    eps: float,
) -> np.ndarray:
    """Compute contract-curve points by weighted Pareto optimizations."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return np.empty((0, 2), dtype=float)

    lambdas = np.linspace(0.01, 0.99, n)
    points: list[tuple[float, float]] = []
    prev = np.array([total_x * 0.5, total_y * 0.5], dtype=float)
    bounds = [(eps, total_x - eps), (eps, total_y - eps)]

    for lam in lambdas:
        starts = [
            prev,
            np.array([total_x * 0.25, total_y * 0.25]),
            np.array([total_x * 0.75, total_y * 0.75]),
        ]
        best = None
        best_obj = np.inf

        def obj(v: np.ndarray) -> float:
            x, y = float(v[0]), float(v[1])
            ua = eval_ua(x, y)
            ub = eval_ub(x, y)
            if not np.isfinite(ua) or not np.isfinite(ub):
                return 1e9
            return -(lam * ua + (1.0 - lam) * ub)

        for x0 in starts:
            res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds)
            if res.success and float(res.fun) < best_obj:
                best_obj = float(res.fun)
                best = res.x

        if best is not None:
            prev = np.asarray(best, dtype=float)
            points.append((float(best[0]), float(best[1])))

    return unique_points(points)


def line_box_intersections(*, px: float, py: float, income: float, total_x: float, total_y: float) -> list[tuple[float, float]]:
    """Return unique boundary intersections for a price line and box."""
    pts: list[tuple[float, float]] = []
    y0 = income / py
    if 0.0 <= y0 <= total_y:
        pts.append((0.0, y0))
    yw = (income - px * total_x) / py
    if 0.0 <= yw <= total_y:
        pts.append((total_x, yw))
    x0 = income / px
    if 0.0 <= x0 <= total_x:
        pts.append((x0, 0.0))
    xh = (income - py * total_y) / px
    if 0.0 <= xh <= total_x:
        pts.append((xh, total_y))
    unique = unique_points(pts, digits=8)
    return [(float(x), float(y)) for x, y in unique]


def walrasian_equilibrium_point(
    *,
    candidates: np.ndarray,
    px: float,
    py: float,
    income: float,
    mrs_a_fn,
    mrs_b_fn,
) -> tuple[float, float]:
    """Select Walrasian equilibrium candidate from contract-curve points."""
    if len(candidates) == 0:
        raise ValueError("Unable to compute contract curve for equilibrium search.")

    price_ratio = px / py

    def score(p: np.ndarray) -> float:
        x, y = float(p[0]), float(p[1])
        budget_err = abs(px * x + py * y - income) / max(income, 1.0)
        mrs_a = mrs_a_fn(x, y)
        mrs_b = mrs_b_fn(x, y)
        mrs_err = 0.0
        if np.isfinite(mrs_a) and np.isfinite(mrs_b) and mrs_a > 0 and mrs_b > 0:
            mrs_err = abs(np.log(mrs_a / price_ratio)) + abs(np.log(mrs_b / price_ratio))
        return budget_err + 0.25 * mrs_err

    residuals = np.array([px * p[0] + py * p[1] - income for p in candidates], dtype=float)
    crossing_idx = np.where(residuals[:-1] * residuals[1:] <= 0.0)[0]
    if len(crossing_idx) > 0:
        i = int(crossing_idx[np.argmin(np.abs(residuals[crossing_idx]))])
        p0 = candidates[i]
        p1 = candidates[i + 1]
        r0 = float(residuals[i])
        r1 = float(residuals[i + 1])
        if abs(r1 - r0) > 1e-12:
            t = -r0 / (r1 - r0)
            t = min(max(t, 0.0), 1.0)
            p = p0 + t * (p1 - p0)
            return float(p[0]), float(p[1])
        return float(p0[0]), float(p0[1])

    idx = int(np.argmin([score(p) for p in candidates]))
    return float(candidates[idx, 0]), float(candidates[idx, 1])

