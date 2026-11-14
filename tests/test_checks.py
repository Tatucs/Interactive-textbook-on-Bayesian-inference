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
