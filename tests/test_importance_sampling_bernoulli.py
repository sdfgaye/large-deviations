import numpy as np
import pytest
from scipy.stats import binom

from large_deviations.distributions import bernoulli_ld
from large_deviations.importance_sampling.bernoulli import (
    bernoulli_tail_naive_mc,
    bernoulli_tail_tilted_mc,
    binomial_tail_event,
    binomial_tail_threshold,
    sample_binomial_sums,
)
from large_deviations.tilting import theta_for_tilted_mean


def test_binomial_tail_threshold():
    assert binomial_tail_threshold(n=100, q=0.10) == 10
    assert binomial_tail_threshold(n=101, q=0.10) == 11


def test_binomial_tail_event():
    event = binomial_tail_event(n=100, q=0.10)

    sums = np.array([0, 9, 10, 11])
    result = event(sums)

    expected = np.array([False, False, True, True])

    assert np.array_equal(result, expected)


def test_binomial_tail_threshold_rejects_invalid_q():
    with pytest.raises(ValueError, match="q must satisfy 0 < q < 1"):
        binomial_tail_threshold(n=100, q=0.0)

    with pytest.raises(ValueError, match="q must satisfy 0 < q < 1"):
        binomial_tail_threshold(n=100, q=1.0)


def test_sample_binomial_sums_has_expected_shape():
    rng = np.random.default_rng(123)
    sampler = sample_binomial_sums(n=20, p=0.3)

    samples = sampler(100, rng)

    assert samples.shape == (100,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 20)


def test_bernoulli_tail_naive_mc_is_reasonable_for_non_extreme_event():
    rng = np.random.default_rng(123)

    n = 20
    p = 0.30
    q = 0.40
    sample_size = 50_000

    estimate = bernoulli_tail_naive_mc(
        n=n,
        p=p,
        q=q,
        sample_size=sample_size,
        rng=rng,
    )

    threshold = binomial_tail_threshold(n=n, q=q)
    exact = float(binom.sf(threshold - 1, n, p))

    assert abs(estimate.estimate - exact) < 4.0 * estimate.standard_error


def test_bernoulli_tail_tilted_mc_uses_saddlepoint_tilt():
    p = 0.02
    q = 0.10

    dist = bernoulli_ld(p)
    theta = theta_for_tilted_mean(dist, target_mean=q)

    assert np.isclose(dist.tilted_parameter(theta), q, atol=1e-8)


def test_bernoulli_tail_tilted_mc_is_reasonable_for_rare_event():
    rng = np.random.default_rng(123)

    n = 100
    p = 0.02
    q = 0.10
    sample_size = 50_000

    estimate = bernoulli_tail_tilted_mc(
        n=n,
        p=p,
        q=q,
        sample_size=sample_size,
        rng=rng,
    )

    threshold = binomial_tail_threshold(n=n, q=q)
    exact = float(binom.sf(threshold - 1, n, p))

    assert estimate.estimate > 0.0
    assert estimate.standard_error > 0.0
    assert estimate.relative_error < 0.20
    assert abs(estimate.estimate - exact) < 4.0 * estimate.standard_error