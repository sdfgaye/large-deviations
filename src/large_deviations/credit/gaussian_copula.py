"""One-factor Gaussian copula tools for homogeneous credit portfolios."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

Array = np.ndarray


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_probability(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} < 1.")


def _validate_rho(rho: float) -> None:
    if not np.isfinite(rho):
        raise ValueError("rho must be finite.")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must satisfy 0 <= rho < 1.")


def asset_threshold(p: float) -> float:
    """Return x_p such that P[N(0,1) > x_p] = p."""
    _validate_probability("p", p)
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
    _validate_probability("p", p)
    _validate_rho(rho)

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
    _validate_positive_integer("n", n)
    _validate_positive_integer("sample_size", sample_size)
    _validate_probability("p", p)
    _validate_rho(rho)

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
    _validate_positive_integer("n", n)

    if not np.isfinite(a) or not 0.0 < a <= 1.0:
        raise ValueError("a must satisfy 0 < a <= 1.")

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
    if not np.isfinite(a) or not 0.0 < a <= 1.0:
        raise ValueError("a must satisfy 0 < a <= 1.")

    _validate_rho(rho)

    if rho == 0.0:
        raise ValueError("rho must be strictly positive for the dependent asymptotic.")

    return float(a * (1.0 - rho**2) / rho**2)