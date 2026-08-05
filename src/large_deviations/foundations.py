"""Foundational abstractions for large deviations.

This module contains generic objects and numerical tools used across
the project: cumulant generating functions, rate functions, domains,
exponential tilting interfaces, and shared validation helpers.

References
----------
Pham, H. (2010), Large Deviations in Mathematical Finance, Sections 2.1--2.2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


ScalarFunction = Callable[[float], float]
TiltedParameter = float | tuple[float, ...]
TiltedParameterFunction = Callable[[float], TiltedParameter]

CEILING_FLOAT_NOISE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class DistributionLD:
    """Large-deviation objects associated with a probability distribution."""

    name: str
    parameters: dict[str, float]
    cgf: ScalarFunction
    rate_function: ScalarFunction
    tilted_parameter: TiltedParameterFunction
    domain_contains: Callable[[float], bool]
    mean_under_tilt: ScalarFunction


def validate_probability(value: float, name: str = "p") -> None:
    """Validate that a value is a probability strictly between 0 and 1."""
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} < 1.")


def validate_positive(name: str, value: float) -> None:
    """Validate that a parameter is strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")


def validate_positive_integer(name: str, value: int) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def validate_finite(name: str, value: float) -> None:
    """Validate that a scalar value is finite."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def noise_tolerant_ceil(value: float) -> int:
    """Return ceil(value), ignoring floating-point noise just above integers.

    Products such as 100 * 0.1 evaluate to 10.000000000000002 in floating
    point; a plain ceiling would round this up to 11. The tolerance is far
    larger than float rounding noise yet far smaller than any meaningful
    fraction of an integer count.
    """
    return int(np.ceil(value - CEILING_FLOAT_NOISE_TOLERANCE))


def safe_xlogy(x: float, y: float) -> float:
    """Return x log(x / y), with the convention 0 log(0 / y) = 0."""
    if x < 0.0:
        raise ValueError("x must be non-negative.")
    if y <= 0.0:
        raise ValueError("y must be positive.")
    if x == 0.0:
        return 0.0
    return float(x * np.log(x / y))
