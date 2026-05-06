"""Generic Monte Carlo and importance sampling tools.

This module contains distribution-agnostic utilities for rare-event estimation
of iid sums.

For an iid sum

    S_n = X_1 + ... + X_n,

exponential tilting gives the likelihood ratio

    dP_theta / dP = exp(theta S_n - n Gamma(theta)).

Therefore, when sampling under the tilted law P_theta, the correction factor is

    dP / dP_theta = exp(-theta S_n + n Gamma(theta)).

The functions in this file do not know whether the underlying distribution is
Bernoulli, Poisson, Gaussian, or anything else. Distribution-specific modules
only need to provide a sampler for S_n under the tilted law.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
EventFunction = Callable[[Array], Array]
SampleSumsFunction = Callable[[int, np.random.Generator], Array]


@dataclass(frozen=True, slots=True)
class MonteCarloEstimate:
    """Summary of a Monte Carlo probability estimate."""

    estimate: float
    standard_error: float
    relative_error: float
    sample_size: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a dictionary representation, useful for pandas."""
        return asdict(self)


def _validate_sample_size(sample_size: int) -> None:
    """Validate a Monte Carlo sample size."""
    if not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer.")
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2.")


def _validate_positive_integer(name: str, value: int) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_finite_scalar(name: str, value: float) -> None:
    """Validate that a scalar is finite."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _as_1d_float_array(values: Array, name: str) -> Array:
    """Convert values to a one-dimensional float array."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    return array


def summarize_monte_carlo_samples(samples: Array) -> MonteCarloEstimate:
    """Summarize Monte Carlo samples.

    Parameters
    ----------
    samples:
        One-dimensional array of Monte Carlo replications. For a naive estimator,
        these are usually indicators. For an importance sampling estimator, these
        are usually weighted indicators.

    Returns
    -------
    MonteCarloEstimate
        Estimate, standard error, relative error, and sample size.
    """
    samples = _as_1d_float_array(samples, name="samples")
    sample_size = int(samples.size)

    _validate_sample_size(sample_size)

    estimate = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / np.sqrt(sample_size))

    relative_error = (
        float(standard_error / abs(estimate))
        if estimate != 0.0
        else np.inf
    )

    return MonteCarloEstimate(
        estimate=estimate,
        standard_error=standard_error,
        relative_error=relative_error,
        sample_size=sample_size,
    )


def log_likelihood_ratio_sum(
    theta: float,
    sums: Array,
    n: int,
    gamma_theta: float,
) -> Array:
    """Return log dP/dP_theta for an iid sum S_n.

    Under exponential tilting,

        dP_theta / dP = exp(theta S_n - n Gamma(theta)).

    Therefore,

        log(dP / dP_theta) = -theta S_n + n Gamma(theta).

    Parameters
    ----------
    theta:
        Exponential tilting parameter.
    sums:
        One-dimensional array of simulated values of S_n under the tilted law.
    n:
        Number of iid variables in the sum.
    gamma_theta:
        Cumulant generating function Gamma(theta).

    Returns
    -------
    np.ndarray
        Log likelihood ratios.
    """
    _validate_finite_scalar("theta", theta)
    _validate_finite_scalar("gamma_theta", gamma_theta)
    _validate_positive_integer("n", n)

    sums = _as_1d_float_array(sums, name="sums")

    return -theta * sums + n * gamma_theta


def naive_sum_estimate(
    *,
    sample_sums: SampleSumsFunction,
    event: EventFunction,
    sample_size: int,
    rng: np.random.Generator | None = None,
) -> MonteCarloEstimate:
    """Estimate P(event(S_n)) by naive Monte Carlo.

    This function is distribution-agnostic. The caller provides a function that
    samples S_n under the original probability measure.

    Parameters
    ----------
    sample_sums:
        Function with signature ``sample_sums(sample_size, rng)`` returning
        simulated values of S_n under the original law.
    event:
        Function mapping simulated sums to boolean indicators.
    sample_size:
        Number of Monte Carlo replications.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    MonteCarloEstimate
        Naive Monte Carlo estimate.
    """
    _validate_sample_size(sample_size)

    if rng is None:
        rng = np.random.default_rng()

    sums = _as_1d_float_array(
        sample_sums(sample_size, rng),
        name="sampled sums",
    )

    if sums.size != sample_size:
        raise ValueError("sample_sums must return exactly sample_size values.")

    indicators = np.asarray(event(sums), dtype=float)

    if indicators.shape != sums.shape:
        raise ValueError("event must return an array with the same shape as sums.")

    return summarize_monte_carlo_samples(indicators)


def exponential_tilting_sum_estimate(
    *,
    n: int,
    theta: float,
    gamma_theta: float,
    sample_sums_under_tilt: SampleSumsFunction,
    event: EventFunction,
    sample_size: int,
    rng: np.random.Generator | None = None,
) -> MonteCarloEstimate:
    """Estimate P(event(S_n)) using exponential tilting.

    This is the generic iid-sum importance sampling estimator:

        E_theta[
            1_event(S_n) exp(-theta S_n + n Gamma(theta))
        ].

    The function does not assume a specific distribution. The caller only needs
    to provide a sampler for S_n under the tilted law.

    Parameters
    ----------
    n:
        Number of iid variables in the sum.
    theta:
        Exponential tilting parameter.
    gamma_theta:
        Cumulant generating function Gamma(theta).
    sample_sums_under_tilt:
        Function with signature ``sample_sums_under_tilt(sample_size, rng)``
        returning simulated values of S_n under P_theta.
    event:
        Function mapping simulated sums to boolean indicators.
    sample_size:
        Number of Monte Carlo replications.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    MonteCarloEstimate
        Importance sampling estimate.
    """
    _validate_positive_integer("n", n)
    _validate_finite_scalar("theta", theta)
    _validate_finite_scalar("gamma_theta", gamma_theta)
    _validate_sample_size(sample_size)

    if rng is None:
        rng = np.random.default_rng()

    sums = _as_1d_float_array(
        sample_sums_under_tilt(sample_size, rng),
        name="sampled tilted sums",
    )

    if sums.size != sample_size:
        raise ValueError(
            "sample_sums_under_tilt must return exactly sample_size values."
        )

    indicators = np.asarray(event(sums), dtype=float)

    if indicators.shape != sums.shape:
        raise ValueError("event must return an array with the same shape as sums.")

    log_weights = log_likelihood_ratio_sum(
        theta=theta,
        sums=sums,
        n=n,
        gamma_theta=gamma_theta,
    )

    weighted_samples = indicators * np.exp(log_weights)

    return summarize_monte_carlo_samples(weighted_samples)