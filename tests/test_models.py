import pytest

from bayes import beta_binomial_posterior
from bayes.checks import check_posterior_params


def test_beta_binomial_update():
    assert beta_binomial_posterior(2, 2, k=7, n=10) == (9, 5)


def test_rejects_k_greater_than_n():
    with pytest.raises(ValueError):
        beta_binomial_posterior(1, 1, k=5, n=3)


@pytest.mark.parametrize(
    "answer, ok",
    [((9, 5), True), ((9.0, 5.0), True), ((5, 9), False), ((8, 5), False), ("x", False), (None, False)],
)
def test_check_posterior_params(answer, ok):
    assert check_posterior_params(answer, 2, 2, 7, 10)[0] is ok
