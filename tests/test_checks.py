import numpy as np
import pytest
from scipy import stats

from bayes.checks import check_grid_posterior, check_posterior_params, check_prior_choice


@pytest.mark.parametrize(
    "answer, ok, fragment",
    [
        ((7, 3), True, "Correct"),
        ((7.0, 3.0), True, "Correct"),
        ((14, 6), False, "Mean is right"),
        ((3, 7), False, "mean is"),
        ((0, 10), False, "positive"),
        ("x", False, "pair"),
    ],
)
def test_check_prior_choice(answer, ok, fragment):
    res, msg = check_prior_choice(answer)
    assert res is ok and fragment in msg


@pytest.mark.parametrize(
    "answer, ok",
    [((9, 5), True), ((9.0, 5.0), True), ((5, 9), False), ((8, 5), False), ("x", False), (None, False)],
)
def test_check_posterior_params(answer, ok):
    assert check_posterior_params(answer, 2, 2, 7, 10)[0] is ok


THETA = np.linspace(0, 1, 501)
PRIOR = stats.beta.pdf(THETA, 2, 2)
LIK = THETA**7 * (1 - THETA) ** 3


@pytest.mark.parametrize(
    "answer, ok, fragment",
    [
        ((PRIOR * LIK) / (PRIOR * LIK).sum(), True, "Correct"),
        (PRIOR * LIK, False, "Not normalized"),
        (LIK / LIK.sum(), False, "shape is off"),
        ((PRIOR + LIK) / (PRIOR + LIK).sum(), False, "shape is off"),
        (np.ones(10), False, "Expected"),
        (None, False, ""),
    ],
)
def test_check_grid_posterior(answer, ok, fragment):
    res, msg = check_grid_posterior(answer, THETA, 2, 2, 7, 10)
    assert res is ok and fragment in msg


@pytest.mark.parametrize(
    "answer, ok, fragment",
    [
        ((9, 5, 9 / 14), True, "Correct"),
        ((9, 5, 0.643), True, "Correct"),
        ((9, 5, 0.7), False, "data proportion"),
        ((9, 5, 0.5), False, "prior mean"),
        ((9, 5, 0.9), False, "off"),
        ((5, 9, 0.357), False, "Swapped"),
        ((9, 5), False, "three numbers"),
    ],
)
def test_check_conjugate_update(answer, ok, fragment):
    from bayes.checks import check_conjugate_update

    res, msg = check_conjugate_update(answer, 2, 2, 7, 10)
    assert res is ok and fragment in msg


def _sampler(rule, n_iter=5000, seed=0):
    """Learner-style Metropolis with a pluggable accept rule(log_u, log_r)."""
    from bayes.mcmc import log_coin_posterior_unnorm

    lt = lambda t: float(log_coin_posterior_unnorm(t, 2, 2, 7, 10))
    rng = np.random.default_rng(seed)
    x, samples, acc = 0.5, [], 0
    for _ in range(n_iter):
        xp = x + 0.2 * rng.standard_normal()
        with np.errstate(invalid="ignore"):
            log_r = lt(xp) - lt(x)
        if rule(np.log(rng.random()), log_r, lt, xp, x):
            x, acc = xp, acc + 1
        samples.append(x)
    return np.array(samples), acc / n_iter


@pytest.mark.parametrize(
    "rule, ok, fragment",
    [
        (lambda lu, lr, lt, xp, x: lu < lr, True, "Correct"),
        (lambda lu, lr, lt, xp, x: True, False, "Every proposal"),
        (lambda lu, lr, lt, xp, x: False, False, "No proposal"),
        (lambda lu, lr, lt, xp, x: lu < lt(x) - lt(xp), False, "proposal over current"),
    ],
)
def test_check_metropolis(rule, ok, fragment):
    from bayes.checks import check_metropolis

    res, msg = check_metropolis(_sampler(rule), 2, 2, 7, 10)
    assert res is ok and fragment in msg, msg


def test_check_metropolis_bad_shape():
    from bayes.checks import check_metropolis

    assert check_metropolis("x", 2, 2, 7, 10)[0] is False
    assert "Too few" in check_metropolis((np.full(50, 0.5), 0.5), 2, 2, 7, 10)[1]
