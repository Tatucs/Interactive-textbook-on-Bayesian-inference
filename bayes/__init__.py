"""Shared math and exercise checkers for the interactive Bayesian textbook.

Runs both locally (pytest) and in the browser (Pyodide); numpy/scipy only.
"""
from .models import (
    beta_binomial_posterior,
    beta_mean_var,
    binomial_likelihood,
    grid_posterior,
    normal_normal_update,
)

__all__ = [
    "beta_binomial_posterior",
    "beta_mean_var",
    "binomial_likelihood",
    "grid_posterior",
    "normal_normal_update",
]

# Match figures to the site's cream theme (only when matplotlib is available).
try:
    import matplotlib as _mpl

    _mpl.rcParams.update({
        "figure.facecolor": "#f1e5c2",
        "axes.facecolor": "#fffdf5",
        "axes.edgecolor": "#5b4636",
        "axes.labelcolor": "#5b4636",
        "axes.titlecolor": "#5b4636",
        "xtick.color": "#5b4636",
        "ytick.color": "#5b4636",
        "text.color": "#5b4636",
        "legend.facecolor": "#fffaf0",
        "axes.prop_cycle": _mpl.cycler(color=["#c0601a", "#c34a4a", "#6f9c99", "#8c6d4f", "#e3884a"]),
    })
except ImportError:  # pragma: no cover
    pass
