"""Model math used by the chapters (coin-flip / Beta-Binomial running example)."""
import numpy as np
from scipy import stats


def beta_mean_var(alpha: float, beta: float) -> tuple[float, float]:
    """Mean and variance of a Beta(alpha, beta) distribution."""
    s = alpha + beta
    return alpha / s, alpha * beta / (s**2 * (s + 1))


def beta_binomial_posterior(alpha: float, beta: float, k: int, n: int) -> tuple[float, float]:
    """Conjugate update: Beta(alpha, beta) prior + k heads in n flips.

    Posterior is Beta(alpha + k, beta + n - k).
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    return alpha + k, beta + n - k


def binomial_likelihood(theta, k: int, n: int):
    """Likelihood p(k | theta, n), up to the constant binomial coefficient."""
    theta = np.asarray(theta, dtype=float)
    return theta**k * (1 - theta) ** (n - k)


def grid_posterior(theta, alpha: float, beta: float, k: int, n: int):
    """Grid approximation: prior * likelihood, normalized to sum to 1 over the grid."""
    unnorm = stats.beta.pdf(theta, alpha, beta) * binomial_likelihood(theta, k, n)
    return unnorm / unnorm.sum()


def normal_normal_update(mu0: float, tau0: float, sigma: float, data) -> tuple[float, float]:
    """Conjugate update for a Normal mean with known noise sd sigma.

    Prior mu ~ N(mu0, tau0^2), data x_i ~ N(mu, sigma^2).
    Returns posterior (mean, sd): precisions add, the mean is precision-weighted.
    """
    x = np.asarray(data, dtype=float)
    prec = 1 / tau0**2 + x.size / sigma**2
    mean = (mu0 / tau0**2 + x.sum() / sigma**2) / prec
    return mean, prec**-0.5
