"""
Elementary distributions for large-deviation theory.

This module provides closed-form large-deviation objects for classical
probability distributions: Bernoulli, Poisson, Gaussian, and Exponential.

References
----------
Pham, H. (2010), Large Deviations in Mathematical Finance, Sections 2.1--2.2.
"""

from __future__ import annotations

import numpy as np

from large_deviations.foundations import (
    DistributionLD,
    safe_xlogy,
    validate_probability,
)


def bernoulli_ld(p: float) -> DistributionLD:
    """Large-deviation objects for a Bernoulli(p) distribution.

    For X ~ Bernoulli(p):

        Γ(θ) = log(1 - p + p exp(θ))

    Under exponential tilting, X remains Bernoulli with parameter:

        p_θ = p exp(θ) / (1 - p + p exp(θ))

    The rate function is:

        Γ*(x) = x log(x / p) + (1 - x) log((1 - x) / (1 - p))

    for x in [0, 1], and +∞ otherwise.
    """
    validate_probability(p)
    
    log_p = np.log(p)
    log_q = np.log1p(-p)

    def cgf(theta: float) -> float:
        """Cumulant generating function Γ(θ), computed stably."""
        return float(np.logaddexp(log_q, log_p + theta))

    def domain_contains(theta: float) -> bool:
        """Check if a value is in the domain of the cumulant generating function."""
        return np.isfinite(theta)
    
    def tilted_parameter(theta: float) -> float:
        """Tilted Bernoulli parameter p_θ, computed stably."""
        return float(np.exp(log_p + theta - cgf(theta)))

    def mean_under_tilt(theta: float) -> float:
        """Mean of the tilted distribution."""
        return tilted_parameter(theta)

    def rate_function(x: float) -> float:
        """Rate function Γ*(x), computed stably."""
        if x < 0.0 or x > 1.0:
            return np.inf

        return safe_xlogy(x, p) + safe_xlogy(1.0 - x, 1.0 - p)

    return DistributionLD(
        name="Bernoulli",
        parameters={"p": p},
        cgf=cgf,
        rate_function=rate_function,
        tilted_parameter=tilted_parameter,
        domain_contains=domain_contains,
        mean_under_tilt=mean_under_tilt,
    )