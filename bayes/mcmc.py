"""Random-walk Metropolis, written out by hand in numpy (no PPL needed in Pyodide)."""
import numpy as np


def log_coin_posterior_unnorm(theta, alpha: float, beta: float, k: int, n: int):
    """Unnormalized log posterior of the coin model: log[prior * likelihood].

    log p~(theta) = (alpha + k - 1) log theta + (beta + n - k - 1) log(1 - theta).
    The normalizing constant (the evidence) is never needed by Metropolis.
    Returns -inf outside (0, 1), so proposals there are always rejected.
    """
    theta = np.asarray(theta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (alpha + k - 1) * np.log(theta) + (beta + n - k - 1) * np.log1p(-theta)
    return np.where((theta > 0) & (theta < 1), out, -np.inf)


def metropolis(log_target, x0: float, tau: float, n_iter: int, seed: int = 0):
    """Random-walk Metropolis for a 1-D target.

    Proposal q(x'|x) = N(x, tau^2): symmetric, so q cancels in alpha.
    Accept with probability A = min(1, alpha), alpha = p~(x') / p~(x); computed in log space.
    Returns (samples, acceptance_rate); samples has length n_iter (x0 excluded).
    """
    rng = np.random.default_rng(seed)
    x, logp = float(x0), float(log_target(x0))
    samples = np.empty(n_iter)
    accepted = 0
    for t in range(n_iter):
        x_prop = x + tau * rng.standard_normal()
        logp_prop = float(log_target(x_prop))
        if np.log(rng.random()) <= logp_prop - logp:  # u <= A  <=>  log u <= log alpha
            x, logp = x_prop, logp_prop
            accepted += 1
        samples[t] = x
    return samples, accepted / n_iter


def effective_sample_size(x) -> float:
    """ESS = N / (1 + 2 * sum of autocorrelations), summed until they first go negative."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3 or np.var(x) == 0:
        return 1.0
    xc = x - x.mean()
    acf = np.correlate(xc, xc, mode="full")[n - 1:] / (np.arange(n, 0, -1) * np.var(x))
    s = 0.0
    for rho in acf[1:]:
        if rho < 0:
            break
        s += rho
    return float(n / (1 + 2 * s))
