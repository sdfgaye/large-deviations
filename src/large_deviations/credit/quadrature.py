"""Quadrature benchmarks for one-factor Gaussian copula credit losses.

This module implements the semi-analytic benchmark

    P[L_n >= n q]
    = E[ P[Binomial(n, p(Z)) >= ceil(n q) | Z] ],

where

    p(Z) = Phi((rho Z + Phi^{-1}(p)) / sqrt(1 - rho^2)).

This is the primary benchmark for the dependent credit portfolio model before
using Monte Carlo or two-step importance sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.stats import binom, norm

from large_deviations.credit.gaussian_copula import (
    conditional_default_probability,
    validate_rho,
)
from large_deviations.foundations import (
    noise_tolerant_ceil,
    validate_positive_integer,
    validate_probability,
)


@dataclass(frozen=True)
class CreditTailQuadratureResult:
    """Result returned by the one-dimensional quadrature benchmark.

    Attributes
    ----------
    probability:
        Estimated tail probability P[L_n >= ceil(n q)].
    log_probability:
        Logarithm of the estimated tail probability.
    absolute_error:
        Absolute integration error reported by scipy.integrate.quad.
    threshold:
        Integer loss-count threshold ceil(n q).
    factor_split:
        Value z_q such that p(z_q) = q. For rho = 0, this is None.
    """

    probability: float
    log_probability: float
    absolute_error: float
    threshold: int
    factor_split: float | None


def loss_count_threshold(n: int, q: float) -> int:
    """Return the integer threshold ceil(n q).

    The rare event is written mathematically as L_n >= n q, but L_n is an
    integer-valued loss count. Numerically, we therefore use

        L_n >= ceil(n q).
    """
    validate_positive_integer("n", n)
    validate_probability(q, name="q")

    return noise_tolerant_ceil(n * q)


def factor_threshold_for_conditional_default_probability(
    *,
    q: float,
    p: float,
    rho: float,
) -> float:
    """Return z_q such that p(z_q) = q.

    In the one-factor Gaussian copula model,

        p(z) = Phi((rho z + Phi^{-1}(p)) / sqrt(1 - rho^2)).

    Solving p(z_q) = q gives

        z_q = [sqrt(1 - rho^2) Phi^{-1}(q) - Phi^{-1}(p)] / rho.

    This threshold is useful for splitting the quadrature integral around the
    transition region where the conditional default probability reaches q.
    """
    validate_probability(q, name="q")
    validate_probability(p)
    validate_rho(rho)

    if rho == 0.0:
        raise ValueError(
            "rho must be strictly positive to define a finite factor threshold."
        )

    numerator = np.sqrt(1.0 - rho**2) * norm.ppf(q) - norm.ppf(p)
    return float(numerator / rho)


def conditional_binomial_tail_probability(
    z: float,
    *,
    n: int,
    p: float,
    rho: float,
    q: float,
) -> float:
    """Return P[Binomial(n, p(z)) >= ceil(n q)].

    This is the conditional loss tail probability given the systematic factor
    Z = z.
    """
    threshold = loss_count_threshold(n, q)
    conditional_p = float(conditional_default_probability(z, p=p, rho=rho))

    return float(binom.sf(threshold - 1, n, conditional_p))


def log_conditional_binomial_tail_probability(
    z: float,
    *,
    n: int,
    p: float,
    rho: float,
    q: float,
) -> float:
    """Return log P[Binomial(n, p(z)) >= ceil(n q)]."""
    threshold = loss_count_threshold(n, q)
    conditional_p = float(conditional_default_probability(z, p=p, rho=rho))

    return float(binom.logsf(threshold - 1, n, conditional_p))


def credit_tail_integrand(
    z: float,
    *,
    n: int,
    p: float,
    rho: float,
    q: float,
) -> float:
    """Return the quadrature integrand.

    The integrand is

        P[Binomial(n, p(z)) >= ceil(n q)] * phi(z),

    where phi is the standard normal density.
    """
    conditional_tail = conditional_binomial_tail_probability(
        z,
        n=n,
        p=p,
        rho=rho,
        q=q,
    )

    return float(conditional_tail * norm.pdf(z))


def log_credit_tail_integrand(
    z: float,
    *,
    n: int,
    p: float,
    rho: float,
    q: float,
) -> float:
    """Return the log of the quadrature integrand."""
    return float(
        log_conditional_binomial_tail_probability(
            z,
            n=n,
            p=p,
            rho=rho,
            q=q,
        )
        + norm.logpdf(z)
    )


def credit_tail_probability_quadrature(
    *,
    n: int,
    p: float,
    rho: float,
    q: float,
    epsabs: float = 0.0,
    epsrel: float = 1e-10,
    limit: int = 300,
) -> CreditTailQuadratureResult:
    """Compute P[L_n >= ceil(n q)] by one-dimensional quadrature.

    Parameters
    ----------
    n:
        Number of obligors.
    p:
        Marginal default probability.
    rho:
        Factor loading in the one-factor Gaussian copula model.
    q:
        Loss fraction threshold.
    epsabs:
        Absolute tolerance passed to scipy.integrate.quad. The default is zero
        because the target probabilities may be very small.
    epsrel:
        Relative tolerance passed to scipy.integrate.quad.
    limit:
        Maximum number of subintervals used by scipy.integrate.quad.

    Returns
    -------
    CreditTailQuadratureResult
        Probability, log-probability, integration error and diagnostic metadata.
    """
    validate_positive_integer("n", n)
    validate_probability(p)
    validate_rho(rho)
    validate_probability(q, name="q")

    threshold = loss_count_threshold(n, q)

    if rho == 0.0:
        probability = float(binom.sf(threshold - 1, n, p))
        log_probability = float(binom.logsf(threshold - 1, n, p))

        return CreditTailQuadratureResult(
            probability=probability,
            log_probability=log_probability,
            absolute_error=0.0,
            threshold=threshold,
            factor_split=None,
        )

    factor_split = factor_threshold_for_conditional_default_probability(
        q=q,
        p=p,
        rho=rho,
    )

    def integrand(z: float) -> float:
        return credit_tail_integrand(
            z,
            n=n,
            p=p,
            rho=rho,
            q=q,
        )

    left_value, left_error = quad(
        integrand,
        -np.inf,
        factor_split,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )

    right_value, right_error = quad(
        integrand,
        factor_split,
        np.inf,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )

    probability = float(left_value + right_value)
    absolute_error = float(left_error + right_error)

    log_probability = float(np.log(probability)) if probability > 0.0 else -np.inf

    return CreditTailQuadratureResult(
        probability=probability,
        log_probability=log_probability,
        absolute_error=absolute_error,
        threshold=threshold,
        factor_split=factor_split,
    )