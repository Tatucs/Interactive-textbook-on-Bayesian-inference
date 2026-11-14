"""Exercise checkers: pure functions returning (ok, message)."""
import math

from .models import beta_binomial_posterior


def check_posterior_params(answer, alpha, beta, k, n, tol=1e-9):
    """Check a learner's (alpha', beta') for a Beta-Binomial update."""
    try:
        a, b = (float(v) for v in answer)
    except (TypeError, ValueError):
        return False, "Answer should be a pair (alpha_post, beta_post)."
    a_true, b_true = beta_binomial_posterior(alpha, beta, k, n)
    if math.isclose(a, a_true, abs_tol=tol) and math.isclose(b, b_true, abs_tol=tol):
        return True, f"Correct: posterior is Beta({a_true:g}, {b_true:g})."
    if math.isclose(a, b_true, abs_tol=tol) and math.isclose(b, a_true, abs_tol=tol):
        return False, "Swapped: heads go to alpha, tails to beta."
    return False, f"Not quite. Got Beta({a:g}, {b:g}); count heads and tails again."
