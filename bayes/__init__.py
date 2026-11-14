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
