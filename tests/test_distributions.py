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