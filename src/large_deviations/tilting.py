"""Reusable helpers for exponential tilting.

This module is distribution-agnostic: it works with any DistributionLD object
providing a cumulant generating function, a tilted parameter, and a tilted mean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from large_deviations.foundations import DistributionLD


TiltedParameter = float | tuple[float, ...]


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


def validate_finite(name: str, value: float) -> None:
    """Validate that a scalar value is finite."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


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

    centered_means = np.full_like(theta_grid, fill_value=np.nan, dtype=float)

    for index, theta in enumerate(theta_grid):
        theta_float = float(theta)

        if not dist.domain_contains(theta_float):
            continue

        try:
            mean = dist.mean_under_tilt(theta_float)
        except Exception:
            continue

        if np.isfinite(mean):
            centered_means[index] = mean - target_mean

    valid = np.isfinite(centered_means)

    if not np.any(valid):
        raise ValueError(
            f"No valid theta values found for {dist.name} in range {theta_range}."
        )

    valid_thetas = theta_grid[valid]
    valid_values = centered_means[valid]

    exact_match_index = np.where(np.abs(valid_values) <= tolerance)[0]

    if exact_match_index.size > 0:
        return float(valid_thetas[exact_match_index[0]])

    bracket: tuple[float, float] | None = None

    for i in range(len(valid_thetas) - 1):
        left_value = valid_values[i]
        right_value = valid_values[i + 1]

        if left_value == 0.0:
            return float(valid_thetas[i])

        if right_value == 0.0:
            return float(valid_thetas[i + 1])

        if left_value * right_value < 0.0:
            bracket = (float(valid_thetas[i]), float(valid_thetas[i + 1]))
            break

    if bracket is None:
        min_mean = float(np.nanmin(valid_values + target_mean))
        max_mean = float(np.nanmax(valid_values + target_mean))

        raise ValueError(
            f"Could not bracket a solution for Gamma'(theta) = {target_mean}. "
            f"On theta_range={theta_range}, valid tilted means range from "
            f"{min_mean:.6g} to {max_mean:.6g}. "
            "Try a wider theta_range or check that the target mean is reachable."
        )

    left, right = bracket

    def centered_mean(theta: float) -> float:
        if not dist.domain_contains(theta):
            raise ValueError(f"theta={theta} is outside the CGF domain.")

        return float(dist.mean_under_tilt(theta) - target_mean)

    left_value = centered_mean(left)
    right_value = centered_mean(right)

    for _ in range(max_iterations):
        mid = 0.5 * (left + right)
        mid_value = centered_mean(mid)

        if abs(mid_value) <= tolerance:
            return float(mid)

        if left_value * mid_value <= 0.0:
            right = mid
            right_value = mid_value
        else:
            left = mid
            left_value = mid_value

        if abs(right - left) <= tolerance:
            return float(0.5 * (left + right))

    return float(0.5 * (left + right))