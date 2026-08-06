import numpy as np
import pytest
from scipy.stats import norm

from large_deviations.credit import (
    asset_threshold,
    conditional_default_probability,
    gaussian_copula_large_loss_decay_rate,
    large_loss_threshold_qn,
    sample_gaussian_copula_losses,
)


def test_asset_threshold_matches_default_probability():
    p = 0.02
    threshold = asset_threshold(p)

    assert np.isclose(1.0 - norm.cdf(threshold), p)


def test_conditional_default_probability_reduces_to_p_when_rho_is_zero():
    p = 0.02
    z_values = np.array([-3.0, 0.0, 2.0])

    conditional_p = conditional_default_probability(z_values, p=p, rho=0.0)

    assert np.allclose(conditional_p, p)


def test_conditional_default_probability_is_increasing_in_factor():
    p = 0.02
    rho = 0.4

    low = conditional_default_probability(-2.0, p=p, rho=rho)
    mid = conditional_default_probability(0.0, p=p, rho=rho)
    high = conditional_default_probability(2.0, p=p, rho=rho)

    assert low < mid < high


def test_sample_gaussian_copula_losses_has_expected_shapes():
    rng = np.random.default_rng(123)

    losses, factors, conditional_probabilities = sample_gaussian_copula_losses(
        n=100,
        p=0.02,
        rho=0.3,
        sample_size=1_000,
        rng=rng,
    )

    assert losses.shape == (1_000,)
    assert factors.shape == (1_000,)
    assert conditional_probabilities.shape == (1_000,)
    assert np.all(losses >= 0)
    assert np.all(losses <= 100)
    assert np.all((conditional_probabilities > 0.0) & (conditional_probabilities < 1.0))


def test_large_loss_threshold_qn_goes_to_one():
    q_100 = large_loss_threshold_qn(100, a=0.5)
    q_10_000 = large_loss_threshold_qn(10_000, a=0.5)

    assert 0.0 < q_100 < q_10_000 < 1.0


def test_sample_gaussian_copula_losses_mean_matches_marginal_probability():
    # Unconditionally each obligor defaults with probability p, so the mean
    # portfolio loss must be n * p. The tolerance is a 4-sigma band computed
    # from the empirical standard deviation of the losses.
    rng = np.random.default_rng(123)

    n = 50
    p = 0.02
    sample_size = 20_000

    losses, _, _ = sample_gaussian_copula_losses(
        n=n,
        p=p,
        rho=0.3,
        sample_size=sample_size,
        rng=rng,
    )

    mean_loss = float(np.mean(losses))
    standard_error = float(np.std(losses, ddof=1) / np.sqrt(sample_size))

    assert abs(mean_loss - n * p) < 4.0 * standard_error


def test_conditional_default_probability_approaches_step_function_for_high_rho():
    # As rho -> 1, the systematic factor dominates: conditionally on a bad
    # factor the portfolio defaults almost surely, on a good factor almost
    # never.
    p = 0.02
    rho = 0.99

    assert conditional_default_probability(3.0, p=p, rho=rho) > 0.999
    assert conditional_default_probability(0.0, p=p, rho=rho) < 1e-10


@pytest.mark.parametrize(
    ("a", "rho", "expected"),
    [
        (1.0, np.sqrt(0.5), 1.0),
        (0.5, 0.5, 1.5),
        (0.75, 0.2, 18.0),
    ],
)
def test_gaussian_copula_decay_rate_matches_closed_form_values(a, rho, expected):
    # gamma = a (1 - rho^2) / rho^2 evaluated at hand-computed points.
    observed = gaussian_copula_large_loss_decay_rate(a=a, rho=rho)

    assert np.isclose(observed, expected)


def test_gaussian_copula_decay_rate_decreases_with_rho():
    a = 0.75

    weak_dependence = gaussian_copula_large_loss_decay_rate(a=a, rho=0.2)
    stronger_dependence = gaussian_copula_large_loss_decay_rate(a=a, rho=0.6)

    assert weak_dependence > stronger_dependence


def test_gaussian_copula_decay_rate_rejects_zero_rho():
    with pytest.raises(ValueError, match="rho must be strictly positive"):
        gaussian_copula_large_loss_decay_rate(a=0.5, rho=0.0)