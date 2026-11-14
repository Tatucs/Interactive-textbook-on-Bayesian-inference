import numpy as np
import pytest
from scipy import stats

from bayes.mcmc import effective_sample_size, log_coin_posterior_unnorm, metropolis


def test_log_target_matches_beta_up_to_constant():
    theta = np.array([0.1, 0.4, 0.8])
    diff = log_coin_posterior_unnorm(theta, 2, 2, 7, 10) - stats.beta.logpdf(theta, 9, 5)
    assert np.allclose(diff, diff[0])
    assert np.isneginf(log_coin_posterior_unnorm(1.2, 2, 2, 7, 10))
    assert np.isneginf(log_coin_posterior_unnorm(0.0, 2, 2, 7, 10))


def test_metropolis_recovers_beta_posterior():
    target = lambda t: log_coin_posterior_unnorm(t, 2, 2, 7, 10)
    samples, acc = metropolis(target, 0.5, tau=0.2, n_iter=20000, seed=1)
    kept = samples[1000:]
    assert kept.mean() == pytest.approx(9 / 14, abs=0.01)
    assert kept.std() == pytest.approx(stats.beta.std(9, 5), abs=0.01)
    assert 0.2 < acc < 0.9


def test_metropolis_is_reproducible():
    target = lambda t: log_coin_posterior_unnorm(t, 2, 2, 7, 10)
    a, _ = metropolis(target, 0.5, 0.2, 500, seed=3)
    b, _ = metropolis(target, 0.5, 0.2, 500, seed=3)
    assert np.array_equal(a, b)


def test_ess_iid_close_to_n_and_correlated_smaller():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(5000)
    assert effective_sample_size(iid) > 4000
    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.95 * ar[t - 1] + rng.standard_normal()
    assert effective_sample_size(ar) < 500
