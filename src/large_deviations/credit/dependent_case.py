"""Asymptotic diagnostics for dependent Gaussian copula credit losses.

This module validates the large-loss asymptotic regime from Pham's Theorem 6.1:

    log P[L_n >= n q_n] ~ - beta log n,

where

    q_n = 1 - scale * n^{-a}
    beta = a * (1 - rho^2) / rho^2.

The probability P[L_n >= n q_n] is computed through the quadrature benchmark,
not by naive Monte Carlo.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from large_deviations.credit.gaussian_copula import (
    gaussian_copula_large_loss_decay_rate,
    large_loss_threshold_qn,
)
from large_deviations.credit.quadrature import credit_tail_probability_quadrature


@dataclass(frozen=True)
class LargeLossAsymptoticPoint:
    """One numerical point in the large-loss asymptotic validation."""

    n: int
    q_n: float
    threshold: int
    probability: float
    log_probability: float
    log_n: float
    empirical_decay_ratio: float
    factor_split: float | None
    quadrature_error: float


@dataclass(frozen=True)
class LargeLossAsymptoticResult:
    """Full diagnostic result for a grid of portfolio sizes."""

    points: list[LargeLossAsymptoticPoint]
    theoretical_decay_rate: float
    fitted_decay_rate: float
    fitted_intercept: float


def estimate_loglog_decay_rate(
    log_n_values: Iterable[float],
    log_probability_values: Iterable[float],
) -> tuple[float, float]:
    """Estimate beta in log P_n ≈ intercept - beta log n.

    The theorem predicts:

        log P[L_n >= n q_n] / log n -> - beta.

    A linear regression of log_probability against log_n gives:

        log_probability ≈ intercept + slope * log_n.

    Therefore:

        beta_hat = -slope.
    """
    x = np.asarray(list(log_n_values), dtype=float)
    y = np.asarray(list(log_probability_values), dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("inputs must be one-dimensional.")

    if x.size != y.size:
        raise ValueError("inputs must have the same length.")

    if x.size < 2:
        raise ValueError("at least two points are required to estimate a slope.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("inputs must contain only finite values.")

    slope, intercept = np.polyfit(x, y, deg=1)

    return float(-slope), float(intercept)


def compute_large_loss_asymptotic_point(
    *,
    n: int,
    p: float,
    rho: float,
    a: float,
    scale: float = 1.0,
    epsabs: float = 0.0,
    epsrel: float = 1e-10,
    limit: int = 300,
) -> LargeLossAsymptoticPoint:
    """Compute one point of the dependent large-loss asymptotic regime."""
    q_n = large_loss_threshold_qn(n, a=a, scale=scale)

    quadrature_result = credit_tail_probability_quadrature(
        n=n,
        p=p,
        rho=rho,
        q=q_n,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )

    log_n = float(np.log(n))

    empirical_decay_ratio = float(
        -quadrature_result.log_probability / log_n
    )

    return LargeLossAsymptoticPoint(
        n=n,
        q_n=q_n,
        threshold=quadrature_result.threshold,
        probability=quadrature_result.probability,
        log_probability=quadrature_result.log_probability,
        log_n=log_n,
        empirical_decay_ratio=empirical_decay_ratio,
        factor_split=quadrature_result.factor_split,
        quadrature_error=quadrature_result.absolute_error,
    )


def compute_large_loss_asymptotic_curve(
    *,
    n_values: Iterable[int],
    p: float,
    rho: float,
    a: float,
    scale: float = 1.0,
    epsabs: float = 0.0,
    epsrel: float = 1e-10,
    limit: int = 300,
) -> LargeLossAsymptoticResult:
    """Compute the full asymptotic diagnostic curve.

    Parameters
    ----------
    n_values:
        Portfolio sizes.
    p:
        Marginal default probability.
    rho:
        Gaussian factor loading.
    a:
        Exponent in q_n = 1 - scale * n^{-a}.
    scale:
        Positive multiplicative constant in the threshold definition.

    Returns
    -------
    LargeLossAsymptoticResult
        Points, theoretical decay rate and fitted log-log decay rate.
    """
    points = [
        compute_large_loss_asymptotic_point(
            n=int(n),
            p=p,
            rho=rho,
            a=a,
            scale=scale,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        for n in n_values
    ]

    fitted_decay_rate, fitted_intercept = estimate_loglog_decay_rate(
        [point.log_n for point in points],
        [point.log_probability for point in points],
    )

    theoretical_decay_rate = gaussian_copula_large_loss_decay_rate(
        a=a,
        rho=rho,
    )

    return LargeLossAsymptoticResult(
        points=points,
        theoretical_decay_rate=theoretical_decay_rate,
        fitted_decay_rate=fitted_decay_rate,
        fitted_intercept=fitted_intercept,
    )


def asymptotic_points_to_records(
    points: Iterable[LargeLossAsymptoticPoint],
) -> list[dict[str, float | int | None]]:
    """Convert asymptotic points to plain records for notebooks or DataFrames."""
    return [
        {
            "n": point.n,
            "q_n": point.q_n,
            "threshold": point.threshold,
            "probability": point.probability,
            "log_probability": point.log_probability,
            "log_n": point.log_n,
            "empirical_decay_ratio": point.empirical_decay_ratio,
            "factor_split": point.factor_split,
            "quadrature_error": point.quadrature_error,
        }
        for point in points
    ]