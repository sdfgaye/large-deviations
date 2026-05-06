import numpy as np
import pytest

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


def test_log_likelihood_ratio_sum_formula():
    theta = 0.7
    sums = np.array([0.0, 2.0, 5.0])
    n = 10
    gamma_theta = 0.3

    result = log_likelihood_ratio_sum(
        theta=theta,
        sums=sums,
        n=n,
        gamma_theta=gamma_theta,
    )

    expected = -theta * sums + n * gamma_theta

    assert np.allclose(result, expected)


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