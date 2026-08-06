"""Bernoulli/binomial importance sampling wrappers.

This module specializes the generic importance sampling tools from
``importance_sampling.core`` to Bernoulli default indicators.

If

    X_i ~ Bernoulli(p),
    S_n = X_1 + ... + X_n,

then S_n is Binomial(n, p). Under exponential tilting, Bernoulli(p)
remains Bernoulli with tilted parameter

    p_theta = p exp(theta) / (1 - p + p exp(theta)).

Therefore, under the tilted law, S_n is Binomial(n, p_theta).
"""

from __future__ import annotations
from collections.abc import Callable

import numpy as np

from large_deviations.distributions import bernoulli_ld
from large_deviations.foundations import validate_probability
from large_deviations.importance_sampling.core import (
    Array,
    EventFunction,
    MonteCarloEstimate,
    exponential_tilting_sum_estimate,
    naive_sum_estimate,
)
from large_deviations.tilting import theta_for_tilted_mean


def _validate_positive_integer(name: str, value: int) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_tail_level(q: float) -> None:
    """Validate a Bernoulli empirical tail level."""
    if not np.isfinite(q):
        raise ValueError("q must be finite.")
    if not 0.0 < q < 1.0:
        raise ValueError("q must satisfy 0 < q < 1.")


def binomial_tail_threshold(n: int, q: float) -> int:
    """Return the integer threshold k for the event S_n / n >= q.

    The event

        S_n / n >= q

    is equivalent to

        S_n >= ceil(n q).

    Parameters
    ----------
    n:
        Number of Bernoulli variables.
    q:
        Tail level.

    Returns
    -------
    int
        Threshold k.
    """
    _validate_positive_integer("n", n)
    _validate_tail_level(q)

    return int(np.ceil(n * q - 1e-12))


def binomial_tail_event(n: int, q: float) -> EventFunction:
    """Return the event function for S_n / n >= q.

    Parameters
    ----------
    n:
        Number of Bernoulli variables.
    q:
        Tail level.

    Returns
    -------
    EventFunction
        Function mapping simulated sums to boolean indicators.
    """
    threshold = binomial_tail_threshold(n=n, q=q)

    def event(sums: Array) -> Array:
        return np.asarray(sums) >= threshold

    return event


def sample_binomial_sums(
    *,
    n: int,
    p: float,
)   -> Callable[[int, np.random.Generator], Array]:
    """Return a sampler for Binomial(n, p) sums.

    The returned function has the signature expected by the generic Monte Carlo
    utilities:

        sample_sums(sample_size, rng)
    """
    _validate_positive_integer("n", n)
    validate_probability(p)

    def sample_sums(sample_size: int, rng: np.random.Generator) -> Array:
        return rng.binomial(n=n, p=p, size=sample_size)

    return sample_sums


def bernoulli_tail_naive_mc(
    n: int,
    p: float,
    q: float,
    sample_size: int,
    rng: np.random.Generator | None = None,
) -> MonteCarloEstimate:
    """Estimate P[Binomial(n, p) / n >= q] by naive Monte Carlo.

    Parameters
    ----------
    n:
        Number of Bernoulli variables.
    p:
        Bernoulli default probability.
    q:
        Tail level.
    sample_size:
        Number of Monte Carlo replications.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    MonteCarloEstimate
        Naive Monte Carlo estimate.
    """
    event = binomial_tail_event(n=n, q=q)
    sample_sums = sample_binomial_sums(n=n, p=p)

    return naive_sum_estimate(
        sample_sums=sample_sums,
        event=event,
        sample_size=sample_size,
        rng=rng,
    )


def bernoulli_tail_tilted_mc(
    n: int,
    p: float,
    q: float,
    sample_size: int,
    theta: float | None = None,
    rng: np.random.Generator | None = None,
) -> MonteCarloEstimate:
    """Estimate P[Binomial(n, p) / n >= q] by exponential tilting.

    If theta is not supplied, it is chosen by solving

        Gamma'(theta) = q,

    so that the tilted Bernoulli mean is exactly q.

    Parameters
    ----------
    n:
        Number of Bernoulli variables.
    p:
        Bernoulli default probability under the original law.
    q:
        Tail level.
    sample_size:
        Number of Monte Carlo replications.
    theta:
        Optional tilting parameter. If None, the saddle-point tilt is used.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    MonteCarloEstimate
        Exponential-tilting importance sampling estimate.
    """
    _validate_positive_integer("n", n)
    validate_probability(p)
    _validate_tail_level(q)

    dist = bernoulli_ld(p)

    if theta is None:
        theta = theta_for_tilted_mean(dist, target_mean=q)

    gamma_theta = dist.cgf(theta)
    tilted_p = float(dist.tilted_parameter(theta))

    event = binomial_tail_event(n=n, q=q)
    sample_sums_under_tilt = sample_binomial_sums(n=n, p=tilted_p)

    return exponential_tilting_sum_estimate(
        n=n,
        theta=theta,
        gamma_theta=gamma_theta,
        sample_sums_under_tilt=sample_sums_under_tilt,
        event=event,
        sample_size=sample_size,
        rng=rng,
    )