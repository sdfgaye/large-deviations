import numpy as np
from scipy.stats import binom

from large_deviations.credit import (
    conditional_binomial_tail_probability,
    credit_tail_probability_quadrature,
    factor_threshold_for_conditional_default_probability,
    loss_count_threshold,
)


def test_loss_count_threshold_uses_ceiling():
    assert loss_count_threshold(10, 0.21) == 3
    assert loss_count_threshold(100, 0.9) == 90


def test_factor_threshold_solves_conditional_default_level():
    from large_deviations.credit import conditional_default_probability

    p = 0.02
    rho = 0.4
    q = 0.75

    z_q = factor_threshold_for_conditional_default_probability(
        q=q,
        p=p,
        rho=rho,
    )

    assert np.isclose(
        conditional_default_probability(z_q, p=p, rho=rho),
        q,
        atol=1e-12,
    )


def test_conditional_binomial_tail_probability_matches_scipy():
    n = 50
    p = 0.02
    rho = 0.3
    q = 0.2
    z = 1.5

    from large_deviations.credit import conditional_default_probability

    conditional_p = conditional_default_probability(z, p=p, rho=rho)
    threshold = loss_count_threshold(n, q)

    expected = binom.sf(threshold - 1, n, conditional_p)
    observed = conditional_binomial_tail_probability(
        z,
        n=n,
        p=p,
        rho=rho,
        q=q,
    )

    assert np.isclose(observed, expected)


def test_quadrature_reduces_to_independent_binomial_when_rho_is_zero():
    n = 100
    p = 0.02
    rho = 0.0
    q = 0.1

    result = credit_tail_probability_quadrature(
        n=n,
        p=p,
        rho=rho,
        q=q,
    )

    threshold = loss_count_threshold(n, q)
    expected = binom.sf(threshold - 1, n, p)

    assert np.isclose(result.probability, expected)
    assert result.factor_split is None


def test_quadrature_probability_is_valid_for_dependent_case():
    result = credit_tail_probability_quadrature(
        n=100,
        p=0.02,
        rho=0.3,
        q=0.5,
    )

    assert 0.0 < result.probability < 1.0
    assert np.isfinite(result.log_probability)
    assert result.absolute_error >= 0.0
    assert result.factor_split is not None