"""Model math used by the chapters."""


def beta_binomial_posterior(alpha: float, beta: float, k: int, n: int) -> tuple[float, float]:
    """Conjugate update: Beta(alpha, beta) prior + k heads in n flips.

    Posterior is Beta(alpha + k, beta + n - k).
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    return alpha + k, beta + n - k
