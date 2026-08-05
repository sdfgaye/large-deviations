import numpy as np
import pytest

from large_deviations.distributions import bernoulli_ld
from large_deviations.importance_sampling.core import (
    exponential_tilting_sum_estimate,
    log_likelihood_ratio_sum,
    naive_sum_estimate,
    summarize_monte_carlo_samples,
)


def test_summarize_monte_carlo_samples_returns_expected_values():
    samples = np.array([0.0, 1.0, 1.0, 0.0])

    estimate = summarize_monte_carlo_samples(samples)

    assert np.isclose(estimate.estimate, 0.5)
    assert estimate.standard_error > 0.0
    assert estimate.relative_error > 0.0
    assert estimate.sample_size == 4


def test_summarize_monte_carlo_samples_rejects_too_small_sample():
    with pytest.raises(ValueError, match="sample_size must be at least 2"):
        summarize_monte_carlo_samples(np.array([1.0]))


def test_summarize_monte_carlo_samples_rejects_non_1d_array():
    samples = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="samples must be a one-dimensional array"):
        summarize_monte_carlo_samples(samples)


def test_log_likelihood_ratio_matches_bernoulli_measure_change():
    # For a single Bernoulli variable, the likelihood ratio dP/dP_theta can be
    # computed directly from the two probability masses:
    #     P(X=1)/P_theta(X=1) = p / p_theta,
    #     P(X=0)/P_theta(X=0) = (1 - p) / (1 - p_theta).
    p = 0.2
    theta = 0.7

    dist = bernoulli_ld(p)
    gamma_theta = dist.cgf(theta)
    tilted_p = float(dist.tilted_parameter(theta))

    log_weights = log_likelihood_ratio_sum(
        theta=theta,
        sums=np.array([0.0, 1.0]),
        n=1,
        gamma_theta=gamma_theta,
    )

    assert np.isclose(np.exp(log_weights[0]), (1.0 - p) / (1.0 - tilted_p))
    assert np.isclose(np.exp(log_weights[1]), p / tilted_p)


def test_log_likelihood_ratio_sum_is_zero_for_identity_tilt():
    # theta = 0 and Gamma(0) = 0 leave the measure unchanged.
    log_weights = log_likelihood_ratio_sum(
        theta=0.0,
        sums=np.array([0.0, 3.0, 7.0]),
        n=10,
        gamma_theta=0.0,
    )

    assert np.allclose(log_weights, 0.0)


def test_log_likelihood_ratio_sum_rejects_non_1d_sums():
    sums = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="sums must be a one-dimensional array"):
        log_likelihood_ratio_sum(
            theta=0.5,
            sums=sums,
            n=10,
            gamma_theta=0.2,
        )


def test_naive_sum_estimate_with_deterministic_sampler():
    def sample_sums(sample_size, rng):
        return np.array([0.0, 1.0, 2.0, 3.0])

    def event(sums):
        return sums >= 2.0

    estimate = naive_sum_estimate(
        sample_sums=sample_sums,
        event=event,
        sample_size=4,
        rng=np.random.default_rng(123),
    )

    assert np.isclose(estimate.estimate, 0.5)
    assert estimate.sample_size == 4


def test_naive_sum_estimate_rejects_wrong_sampler_size():
    def sample_sums(sample_size, rng):
        return np.array([0.0, 1.0])

    def event(sums):
        return sums >= 1.0

    with pytest.raises(ValueError, match="sample_sums must return exactly sample_size"):
        naive_sum_estimate(
            sample_sums=sample_sums,
            event=event,
            sample_size=4,
            rng=np.random.default_rng(123),
        )


def test_exponential_tilting_sum_estimate_with_zero_theta_matches_naive_case():
    def sample_sums_under_tilt(sample_size, rng):
        return np.array([0.0, 1.0, 2.0, 3.0])

    def event(sums):
        return sums >= 2.0

    estimate = exponential_tilting_sum_estimate(
        n=4,
        theta=0.0,
        gamma_theta=0.0,
        sample_sums_under_tilt=sample_sums_under_tilt,
        event=event,
        sample_size=4,
        rng=np.random.default_rng(123),
    )

    assert np.isclose(estimate.estimate, 0.5)
    assert estimate.sample_size == 4