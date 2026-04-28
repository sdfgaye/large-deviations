"""Foundational abstractions for large deviations.

This module contains generic objects and numerical tools used across
the project: cumulant generating functions, rate functions, domains,
and exponential tilting interfaces.

References
----------
Pham, H. (2010), Large Deviations in Mathematical Finance, Sections 2.1--2.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ScalarFunction = Callable[[float], float]
TiltedParameter = float | tuple[float, ...]
TiltedParameterFunction = Callable[[float], TiltedParameter]

@dataclass(frozen=True, slots=True)
class DistributionLD:
    """Large-deviation objects associated with a probability distribution."""

    name: str
    parameters:dict[str, float]
    cgf: ScalarFunction
    rate_function: ScalarFunction
    tilted_parameter: TiltedParameterFunction
    domain_contains: Callable[[float], bool]
    mean_under_tilt: ScalarFunction


def validate_probability(p: float) -> None:
    """Validate that p is a probability strictly between 0 and 1."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must satisfy 0 < p < 1.")
    
def validate_positive(name: str, value: float) -> None:
    """Validate that a parameter is strictly positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")


def safe_xlogy(x: float, y: float) -> float:
    """Return x log(x / y), with the convention 0 log(0 / y) = 0."""
    if x < 0.0:
        raise ValueError("x must be non-negative.")
    if y <= 0.0:
        raise ValueError("y must be positive.")
    if x == 0.0:
        return 0.0
    return float(x * np.log(x / y))