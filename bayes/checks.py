"""Exercise checkers: pure functions returning (ok, message).

Called from the quarto-live grading blocks in the browser and from pytest.
"""
import math

import numpy as np

from .models import beta_binomial_posterior, beta_mean_var, grid_posterior


def _pair(answer):
    a, b = (float(v) for v in answer)
    return a, b


def check_prior_choice(answer, target_mean=0.7, target_strength=10, tol=0.05):
    """Priors page: choose Beta(alpha, beta) with given mean and alpha + beta."""
    try:
        a, b = _pair(answer)
    except (TypeError, ValueError):
        return False, "Answer should be a pair (alpha, beta)."
    if a <= 0 or b <= 0:
        return False, "Both alpha and beta must be positive."
    mean, _ = beta_mean_var(a, b)
    mean_ok = abs(mean - target_mean) <= tol
    strength_ok = abs((a + b) - target_strength) <= tol * target_strength
    if mean_ok and strength_ok:
        return True, f"Correct: Beta({a:g}, {b:g}) has mean {mean:.2f} and strength alpha + beta = {a + b:g}."
    if mean_ok:
        return False, f"Mean is right ({mean:.2f}), but alpha + beta = {a + b:g}, not {target_strength}."
    if strength_ok:
        return False, f"Strength is right, but the mean is {mean:.2f}, not {target_mean}."
    return False, f"Beta({a:g}, {b:g}) has mean {mean:.2f} and alpha + beta = {a + b:g}. Use mean = alpha / (alpha + beta)."


def check_posterior_params(answer, alpha, beta, k, n, tol=1e-9):
    """Check a learner's (alpha', beta') for a Beta-Binomial update."""
    try:
        a, b = _pair(answer)
    except (TypeError, ValueError):
        return False, "Answer should be a pair (alpha_post, beta_post)."
    a_true, b_true = beta_binomial_posterior(alpha, beta, k, n)
    if math.isclose(a, a_true, abs_tol=tol) and math.isclose(b, b_true, abs_tol=tol):
        return True, f"Correct: posterior is Beta({a_true:g}, {b_true:g})."
    if math.isclose(a, b_true, abs_tol=tol) and math.isclose(b, a_true, abs_tol=tol):
        return False, "Swapped: heads go to alpha, tails to beta."
    return False, f"Not quite. Got Beta({a:g}, {b:g}); count heads and tails again."


def check_grid_posterior(answer, theta, alpha, beta, k, n, tol=1e-6):
    """Posteriors page: learner's grid posterior vs. the reference one."""
    try:
        post = np.asarray(answer, dtype=float)
    except (TypeError, ValueError):
        return False, "Answer should be an array of posterior values on the grid."
    if post.shape != np.shape(theta):
        return False, f"Expected {np.size(theta)} values (one per grid point), got shape {post.shape}."
    if not np.all(np.isfinite(post)):
        return False, "The posterior contains NaN or inf values."
    total = post.sum()
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        return False, f"Not normalized: the values sum to {total:.4g}, not 1. Divide by the sum."
    ref = grid_posterior(theta, alpha, beta, k, n)
    err = np.max(np.abs(post - ref))
    if err <= tol:
        return True, "Correct: your grid posterior matches prior x likelihood, normalized."
    return False, f"Normalized, but the shape is off (max difference {err:.2g}). Multiply prior by likelihood."


def check_conjugate_update(answer, alpha, beta, k, n, tol=1e-3):
    """Conjugacy page: learner gives (alpha_post, beta_post, posterior_mean)."""
    try:
        a, b, m = (float(v) for v in answer)
    except (TypeError, ValueError):
        return False, "Answer should be three numbers: (alpha_post, beta_post, post_mean)."
    ok, msg = check_posterior_params((a, b), alpha, beta, k, n)
    if not ok:
        return ok, msg
    true_mean = a / (a + b)
    if abs(m - true_mean) <= tol:
        return True, f"Correct: Beta({a:g}, {b:g}) with posterior mean {true_mean:.3f}."
    if abs(m - k / n) <= tol:
        return False, "That is the data proportion k/n. The posterior mean also includes the prior pseudo-counts."
    if abs(m - alpha / (alpha + beta)) <= tol:
        return False, "That is the prior mean. Use the posterior parameters."
    return False, f"Parameters are right, but the mean {m:g} is off. Use alpha / (alpha + beta)."
