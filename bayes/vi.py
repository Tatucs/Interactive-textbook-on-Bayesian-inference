"""Mean-field variational inference on a 2-D correlated Gaussian target, in closed form."""
import numpy as np


def target_cov(rho: float, s1: float = 1.0, s2: float = 1.0) -> np.ndarray:
    """Covariance of the 2-D Gaussian target p(θ) = N(m, Σ) with correlation rho."""
    return np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])


def kl_gaussian(mu_q, cov_q, mu_p, cov_p) -> float:
    """KL(q || p) for multivariate Gaussians q = N(mu_q, cov_q), p = N(mu_p, cov_p).

    KL = 1/2 [ tr(Σp⁻¹ Σq) + (mp - mq)ᵀ Σp⁻¹ (mp - mq) - d + log(det Σp / det Σq) ].
    """
    mu_q, mu_p = np.asarray(mu_q, float), np.asarray(mu_p, float)
    cov_q, cov_p = np.asarray(cov_q, float), np.asarray(cov_p, float)
    d = mu_q.size
    prec_p = np.linalg.inv(cov_p)
    diff = mu_p - mu_q
    _, logdet_p = np.linalg.slogdet(cov_p)
    _, logdet_q = np.linalg.slogdet(cov_q)
    return float(0.5 * (np.trace(prec_p @ cov_q) + diff @ prec_p @ diff - d + logdet_p - logdet_q))


def meanfield_optimum(mu_p, cov_p):
    """Optimal factorized q(θ) = q1(θ1) q2(θ2) under KL(q || p) for a Gaussian p.

    Means match p; variances are 1 / Λ_ii with Λ = Σ⁻¹ (the precision matrix),
    which is never larger than the true marginal variance Σ_ii.
    Returns (mu, sigmas).
    """
    prec = np.linalg.inv(np.asarray(cov_p, float))
    return np.asarray(mu_p, float).copy(), 1.0 / np.sqrt(np.diag(prec))


def cavi_means(mu_p, cov_p, mu_init, n_steps: int):
    """Coordinate-ascent (CAVI) path of the mean-field means for a Gaussian target.

    Update for factor i:  m_i <- mu_i - (Λ_ij / Λ_ii) (m_j - mu_j).
    Returns an array of shape (2 * n_steps + 1, 2): the state after each single-factor update.
    """
    mu_p = np.asarray(mu_p, float)
    prec = np.linalg.inv(np.asarray(cov_p, float))
    m = np.asarray(mu_init, float).copy()
    path = [m.copy()]
    for _ in range(n_steps):
        for i, j in ((0, 1), (1, 0)):
            m[i] = mu_p[i] - prec[i, j] / prec[i, i] * (m[j] - mu_p[j])
            path.append(m.copy())
    return np.array(path)
