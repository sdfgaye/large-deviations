import math

import numpy as np
import pytest

from large_deviations.distributions import bernoulli_ld


def test_bernoulli_distribution_metadata():
    dist = bernoulli_ld(p=0.2)

    assert dist.name == "Bernoulli"
    assert dist.parameters == {"p": 0.2}


def test_bernoulli_cgf_at_zero_is_zero():
    dist = bernoulli_ld(p=0.2)

    assert np.isclose(dist.cgf(0.0), 0.0)


def test_bernoulli_domain_contains_real_values():
    dist = bernoulli_ld(p=0.2)

    assert dist.domain_contains(0.0)
    assert dist.domain_contains(100.0)
    assert dist.domain_contains(-100.0)


def test_bernoulli_tilted_parameter_at_zero_is_original_p():
    dist = bernoulli_ld(p=0.2)

    assert np.isclose(dist.tilted_parameter(0.0), 0.2)


def test_bernoulli_tilted_parameter_increases_for_positive_theta():
    dist = bernoulli_ld(p=0.2)

    assert dist.tilted_parameter(1.0) > 0.2


def test_bernoulli_mean_under_tilt_matches_tilted_parameter():
    dist = bernoulli_ld(p=0.2)

    theta = 1.0
    assert np.isclose(dist.mean_under_tilt(theta), dist.tilted_parameter(theta))


def test_bernoulli_rate_function_is_zero_at_mean():
    dist = bernoulli_ld(p=0.2)

    assert np.isclose(dist.rate_function(0.2), 0.0)


def test_bernoulli_rate_function_is_positive_away_from_mean():
    dist = bernoulli_ld(p=0.2)

    assert dist.rate_function(0.7) > 0.0


def test_bernoulli_rate_function_is_infinite_outside_support():
    dist = bernoulli_ld(p=0.2)

    assert np.isinf(dist.rate_function(-0.1))
    assert np.isinf(dist.rate_function(1.1))


def test_bernoulli_rejects_invalid_probability():
    with pytest.raises(ValueError, match="0 < p < 1"):
        bernoulli_ld(p=0.0)

    with pytest.raises(ValueError, match="0 < p < 1"):
        bernoulli_ld(p=1.0)


def test_bernoulli_rate_function_matches_legendre_at_tilted_mean():
    dist = bernoulli_ld(p=0.2)

    theta = 0.7
    x = dist.mean_under_tilt(theta)

    expected = theta * x - dist.cgf(theta)

    assert np.isclose(dist.rate_function(x), expected)


@pytest.mark.parametrize(
    ("p", "x"),
    [(0.2, 0.7), (0.02, 0.10), (0.5, 0.9)],
)
def test_bernoulli_rate_function_matches_kl_closed_form(p, x):
    # The Bernoulli rate function is the relative entropy
    # I_p(x) = x log(x / p) + (1 - x) log((1 - x) / (1 - p)).
    dist = bernoulli_ld(p=p)

    expected = x * math.log(x / p) + (1.0 - x) * math.log((1.0 - x) / (1.0 - p))

    assert np.isclose(dist.rate_function(x), expected, atol=1e-12)


def test_bernoulli_rate_function_symmetry_under_relabeling():
    # Swapping success and failure labels leaves the divergence invariant:
    # I_p(x) = I_{1-p}(1-x).
    p = 0.2
    x = 0.7

    assert np.isclose(
        bernoulli_ld(p=p).rate_function(x),
        bernoulli_ld(p=1.0 - p).rate_function(1.0 - x),
    )


def test_bernoulli_cgf_derivative_matches_mean_under_tilt():
    dist = bernoulli_ld(p=0.2)

    theta = 0.7
    eps = 1e-6

    numerical_derivative = (dist.cgf(theta + eps) - dist.cgf(theta - eps)) / (2 * eps)

    assert np.isclose(
        numerical_derivative,
        dist.mean_under_tilt(theta),
        rtol=1e-5,
    )