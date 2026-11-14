import numpy as np
import pytest
from scipy import stats

from bayes.vi import cavi_means, kl_gaussian, meanfield_optimum, target_cov


def test_kl_zero_for_identical_and_positive_otherwise():
    cov = target_cov(0.5)
    assert kl_gaussian([0, 0], cov, [0, 0], cov) == pytest.approx(0.0, abs=1e-12)
    assert kl_gaussian([1, 0], np.eye(2), [0, 0], cov) > 0


def test_kl_matches_monte_carlo():
    rng = np.random.default_rng(0)
    mq, Sq = np.array([0.3, -0.2]), np.diag([0.5, 0.8]) ** 2
    mp, Sp = np.zeros(2), target_cov(0.7)
    x = rng.multivariate_normal(mq, Sq, size=200_000)
    mc = np.mean(stats.multivariate_normal.logpdf(x, mq, Sq) - stats.multivariate_normal.logpdf(x, mp, Sp))
    assert kl_gaussian(mq, Sq, mp, Sp) == pytest.approx(mc, abs=0.01)


def test_meanfield_optimum_is_sqrt_one_minus_rho_squared():
    mu, sig = meanfield_optimum([0, 0], target_cov(0.8))
    assert np.allclose(mu, 0)
    assert np.allclose(sig, 0.6)


def test_meanfield_optimum_minimizes_kl_over_diagonal_q():
    cov = target_cov(0.8)
    _, sig = meanfield_optimum([0, 0], cov)
    best = kl_gaussian([0, 0], np.diag(sig**2), [0, 0], cov)
    for s1 in np.linspace(0.3, 1.2, 10):
        for s2 in np.linspace(0.3, 1.2, 10):
            assert kl_gaussian([0, 0], np.diag([s1**2, s2**2]), [0, 0], cov) >= best - 1e-12


def test_cavi_converges_to_target_mean():
    path = cavi_means([1.0, -1.0], target_cov(0.9), [-2.0, 2.0], n_steps=200)
    assert np.allclose(path[-1], [1.0, -1.0], atol=1e-6)
    assert path.shape == (401, 2)
