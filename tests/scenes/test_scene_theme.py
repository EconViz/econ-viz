from mosaickit import DashStyle, ThemeRegistry

from utility_viz.core.scenes import (
    UTILITY_THEME,
    Budget,
    Equilibrium,
    Indifference,
    register_utility_theme,
    utility_roles,
)
from utility_viz.core.themes import Theme as LegacyTheme


def _resolve(role, category):
    return UTILITY_THEME.resolve(role.value, fallback_category=category)


def test_theme_matches_legacy_defaults():
    legacy = LegacyTheme(name="legacy")
    ic = _resolve(Indifference.MAIN, "primary").stroke
    assert (ic.color.to_hex().upper(), ic.width) == (legacy.ic_color.upper(), legacy.ic_linewidth)
    budget = _resolve(Budget.MAIN, "primary").stroke
    assert (budget.color.to_hex().upper(), budget.width) == (legacy.budget_color.upper(), legacy.budget_linewidth)
    eq = _resolve(Equilibrium.MAIN, "point").marker
    assert eq.color.to_hex().upper() == legacy.eq_color.upper()
    assert eq.size == legacy.eq_markersize**2
    fill = _resolve(Budget.FILL, "primary").fill
    assert fill.opacity == legacy.budget_fill_alpha
    assert _resolve(Equilibrium.DROP, "primary").stroke.dash is DashStyle.DOTTED
    assert _resolve(Equilibrium.RAY, "primary").stroke.width == legacy.ray_linewidth


def test_compensated_budget_inherits_from_budget_through_dotted_fallback():
    compensated = _resolve(Budget.COMPENSATED, "primary")
    assert compensated.stroke.dash is DashStyle.DASHED
    assert compensated.stroke.color.to_hex().upper() == "#777777"


def test_secondary_label_and_curve_inherit_and_dim():
    label = _resolve(Indifference.SECONDARY_LABEL, "text").text
    assert label.opacity == 0.45 and label.size == 9
    assert _resolve(Indifference.SECONDARY, "primary").stroke.opacity == 0.45


def test_all_roles_are_dotted_utility_names():
    assert all(role.startswith("utility.") for role in utility_roles())


def test_register_into_a_theme_registry():
    registry = ThemeRegistry()
    theme = register_utility_theme(registry)
    assert registry.get("default") is theme
    assert "utility.budget" in theme.roles and "axes" in theme.roles
