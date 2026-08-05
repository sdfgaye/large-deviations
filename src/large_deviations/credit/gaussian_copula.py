"""One-factor Gaussian copula tools for homogeneous credit portfolios."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from large_deviations.foundations import (
    validate_positive_integer,
    validate_probability,
)

Array = np.ndarray


def validate_rho(rho: float) -> None:
    """Validate a one-factor Gaussian copula factor loading."""
    if not np.isfinite(rho):
        raise ValueError("rho must be finite.")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must satisfy 0 <= rho < 1.")


def _validate_threshold_exponent(a: float) -> None:
    if not np.isfinite(a) or not 0.0 < a <= 1.0:
        raise ValueError("a must satisfy 0 < a <= 1.")


def asset_threshold(p: float) -> float:
    """Return x_p such that P[N(0,1) > x_p] = p."""
    validate_probability(p)
    return float(norm.ppf(1.0 - p))


def conditional_default_probability(
    z: float | Array,
    *,
    p: float,
    rho: float,
) -> float | Array:
    """Return P[default | Z = z] in the one-factor Gaussian copula model.

    The model is:

        X_k = rho Z + sqrt(1 - rho^2) epsilon_k

    and default occurs when:

        X_k > Phi^{-1}(1 - p).

    Therefore:

        p(z) = Phi((rho z + Phi^{-1}(p)) / sqrt(1 - rho^2)).
    """
    validate_probability(p)
    validate_rho(rho)

    z_array = np.asarray(z)
    denominator = np.sqrt(1.0 - rho**2)

    conditional_p = norm.cdf((rho * z_array + norm.ppf(p)) / denominator)

    if np.isscalar(z):
        return float(conditional_p)

    return conditional_p


def sample_gaussian_copula_losses(
    *,
    n: int,
    p: float,
    rho: float,
    sample_size: int,
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array, Array]:
    """Sample losses from a homogeneous one-factor Gaussian copula portfolio.

    Returns
    -------
    losses:
        Binomial conditional losses L_n.
    factors:
        Simulated systematic factors Z.
    conditional_probabilities:
        Conditional default probabilities p(Z).
    """
    validate_positive_integer("n", n)
    validate_positive_integer("sample_size", sample_size)
    validate_probability(p)
    validate_rho(rho)

    if rng is None:
        rng = np.random.default_rng()

    factors = rng.normal(loc=0.0, scale=1.0, size=sample_size)
    conditional_probabilities = conditional_default_probability(
        factors,
        p=p,
        rho=rho,
    )

    losses = rng.binomial(
        n=n,
        p=conditional_probabilities,
        size=sample_size,
    )

    return losses, factors, conditional_probabilities


def large_loss_threshold_qn(
    n: int,
    *,
    a: float,
    scale: float = 1.0,
) -> float:
    """Return q_n = 1 - scale * n^{-a}.

    This is the large-loss threshold regime used later for the final theorem.
    """
    validate_positive_integer("n", n)
    _validate_threshold_exponent(a)

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive.")

    q_n = 1.0 - scale * n ** (-a)

    if not 0.0 < q_n < 1.0:
        raise ValueError("q_n must satisfy 0 < q_n < 1.")

    return float(q_n)


def gaussian_copula_large_loss_decay_rate(
    *,
    a: float,
    rho: float,
) -> float:
    """Return a(1 - rho^2) / rho^2.

    This is the polynomial decay exponent in the dependent Gaussian copula
    large-loss regime.
    """
    _validate_threshold_exponent(a)
    validate_rho(rho)

    if rho == 0.0:
        raise ValueError("rho must be strictly positive for the dependent asymptotic.")

    return float(a * (1.0 - rho**2) / rho**2)