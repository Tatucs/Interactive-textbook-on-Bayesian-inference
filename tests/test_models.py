import numpy as np
import pytest
from scipy import stats

from bayes import beta_binomial_posterior, beta_mean_var, grid_posterior


def test_beta_binomial_update():
    assert beta_binomial_posterior(2, 2, k=7, n=10) == (9, 5)


def test_rejects_k_greater_than_n():
    with pytest.raises(ValueError):
        beta_binomial_posterior(1, 1, k=5, n=3)


def test_beta_mean_var_matches_scipy():
    m, v = beta_mean_var(3, 5)
    assert m == pytest.approx(stats.beta.mean(3, 5))
    assert v == pytest.approx(stats.beta.var(3, 5))


def test_grid_posterior_matches_conjugate_answer():
    theta = np.linspace(0, 1, 2001)
    post = grid_posterior(theta, 2, 2, k=7, n=10)
    assert post.sum() == pytest.approx(1.0)
    exact = stats.beta.pdf(theta, 9, 5)
    assert np.max(np.abs(post - exact / exact.sum())) < 1e-12
    # posterior mean from grid agrees with Beta(9, 5) mean
    assert np.sum(theta * post) == pytest.approx(9 / 14, abs=1e-6)


def test_normal_normal_update_against_grid():
    from bayes import normal_normal_update

    data = [1.2, 0.8, 1.5, 0.9]
    mean, sd = normal_normal_update(0.0, 2.0, 1.0, data)
    mu = np.linspace(-5, 5, 20001)
    logp = stats.norm.logpdf(mu, 0, 2) + sum(stats.norm.logpdf(x, mu, 1) for x in data)
    w = np.exp(logp - logp.max()); w /= w.sum()
    grid_mean = np.sum(mu * w)
    assert mean == pytest.approx(grid_mean, abs=1e-6)
    assert sd == pytest.approx(np.sqrt(np.sum((mu - grid_mean) ** 2 * w)), abs=1e-6)
