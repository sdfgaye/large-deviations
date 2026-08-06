"""Reusable helpers for exponential tilting.

This module is distribution-agnostic: it works with any DistributionLD object
providing a cumulant generating function, a tilted parameter, and a tilted mean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from large_deviations.foundations import (
    DistributionLD,
    TiltedParameter,
    validate_finite,
)


@dataclass(frozen=True, slots=True)
class TiltingSummary:
    """Summary of an exponentially tilted distribution."""

    distribution: str
    theta: float
    unit_weight_multiplier: float
    cgf: float
    mean_under_tilt: float
    tilted_parameter: TiltedParameter

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation, useful for pandas."""
        return asdict(self)


def unit_weight_multiplier(theta: float) -> float:
    """Return exp(theta), the multiplier applied per one-unit increase in x.

    Under exponential tilting,

        new weight(x) is proportional to exp(theta * x) old weight(x).

    Therefore exp(theta) is the multiplicative factor for increasing x by one.
    For Bernoulli variables, this is also the odds multiplier.
    """
    validate_finite("theta", theta)
    return float(np.exp(theta))


def tilting_summary(dist: DistributionLD, theta: float) -> TiltingSummary:
    """Return the main quantities associated with exponential tilting."""
    validate_finite("theta", theta)

    if not dist.domain_contains(theta):
        raise ValueError(
            f"theta={theta} is outside the CGF domain for {dist.name}."
        )

    return TiltingSummary(
        distribution=dist.name,
        theta=float(theta),
        unit_weight_multiplier=unit_weight_multiplier(theta),
        cgf=dist.cgf(theta),
        mean_under_tilt=dist.mean_under_tilt(theta),
        tilted_parameter=dist.tilted_parameter(theta),
    )


def evaluate_tilting_curve(
    dist: DistributionLD,
    theta_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate Gamma(theta) and the tilted mean on a theta grid.

    Values outside the CGF domain are returned as NaN.
    """
    theta_grid = np.asarray(theta_grid, dtype=float)

    cgf_values = np.full_like(theta_grid, fill_value=np.nan, dtype=float)
    mean_values = np.full_like(theta_grid, fill_value=np.nan, dtype=float)

    for index, theta in enumerate(theta_grid):
        theta_float = float(theta)

        if not np.isfinite(theta_float):
            continue

        if not dist.domain_contains(theta_float):
            continue

        cgf_values[index] = dist.cgf(theta_float)
        mean_values[index] = dist.mean_under_tilt(theta_float)

    return {
        "theta": theta_grid,
        "cgf": cgf_values,
        "mean_under_tilt": mean_values,
    }


def _tilted_mean_offset_grid(
    dist: DistributionLD,
    theta_grid: np.ndarray,
    target_mean: float,
) -> np.ndarray:
    """Evaluate mean_under_tilt - target_mean on a grid, NaN where undefined."""
    offsets = np.full_like(theta_grid, fill_value=np.nan, dtype=float)

    for index, theta in enumerate(theta_grid):
        theta_float = float(theta)

        if not dist.domain_contains(theta_float):
            continue

        try:
            mean = dist.mean_under_tilt(theta_float)
        except Exception:
            continue

        if np.isfinite(mean):
            offsets[index] = mean - target_mean

    return offsets


def _find_sign_change_bracket(
    thetas: np.ndarray,
    offsets: np.ndarray,
) -> tuple[float, float] | None:
    """Return consecutive grid points where the offset changes sign, if any."""
    for i in range(len(thetas) - 1):
        if offsets[i] * offsets[i + 1] < 0.0:
            return float(thetas[i]), float(thetas[i + 1])

    return None


def _bisect_root(
    function,
    bracket: tuple[float, float],
    tolerance: float,
    max_iterations: int,
) -> float:
    """Find a root of function by bisection inside a sign-changing bracket."""
    left, right = bracket
    left_value = function(left)

    for _ in range(max_iterations):
        mid = 0.5 * (left + right)
        mid_value = function(mid)

        if abs(mid_value) <= tolerance:
            return float(mid)

        if left_value * mid_value <= 0.0:
            right = mid
        else:
            left = mid
            left_value = mid_value

        if abs(right - left) <= tolerance:
            break

    return float(0.5 * (left + right))


def theta_for_tilted_mean(
    dist: DistributionLD,
    target_mean: float,
    theta_range: tuple[float, float] = (-10.0, 10.0),
    grid_size: int = 2001,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Find theta such that the tilted mean equals target_mean.

    This solves numerically

        Gamma'(theta) = target_mean.

    It works for any DistributionLD whose tilted mean is monotone in theta,
    which is the usual case for one-dimensional exponential families.

    Parameters
    ----------
    dist:
        DistributionLD object.
    target_mean:
        Desired mean under the tilted distribution.
    theta_range:
        Search interval for theta.
    grid_size:
        Number of points used to find an initial sign-changing bracket.
    tolerance:
        Bisection tolerance.
    max_iterations:
        Maximum number of bisection iterations.

    Returns
    -------
    float
        The tilt parameter theta.
    """
    validate_finite("target_mean", target_mean)

    lower, upper = theta_range

    if not lower < upper:
        raise ValueError("theta_range must satisfy lower < upper.")

    theta_grid = np.linspace(lower, upper, grid_size)
    offsets = _tilted_mean_offset_grid(dist, theta_grid, target_mean)

    valid = np.isfinite(offsets)

    if not np.any(valid):
        raise ValueError(
            f"No valid theta values found for {dist.name} in range {theta_range}."
        )

    valid_thetas = theta_grid[valid]
    valid_offsets = offsets[valid]

    on_target = np.where(np.abs(valid_offsets) <= tolerance)[0]

    if on_target.size > 0:
        return float(valid_thetas[on_target[0]])

    bracket = _find_sign_change_bracket(valid_thetas, valid_offsets)

    if bracket is None:
        reachable_means = valid_offsets + target_mean

        raise ValueError(
            f"Could not bracket a solution for Gamma'(theta) = {target_mean}. "
            f"On theta_range={theta_range}, valid tilted means range from "
            f"{float(np.min(reachable_means)):.6g} to "
            f"{float(np.max(reachable_means)):.6g}. "
            "Try a wider theta_range or check that the target mean is reachable."
        )

    def tilted_mean_offset(theta: float) -> float:
        if not dist.domain_contains(theta):
            raise ValueError(f"theta={theta} is outside the CGF domain.")

        return float(dist.mean_under_tilt(theta) - target_mean)

    return _bisect_root(tilted_mean_offset, bracket, tolerance, max_iterations)
