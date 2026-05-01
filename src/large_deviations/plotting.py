"""Plotting utilities for exponential tilting."""

from __future__ import annotations

import numpy as np

from large_deviations.foundations import DistributionLD
from large_deviations.tilting import (
    evaluate_tilting_curve,
    theta_for_tilted_mean,
)


def tilting_curve_data(
    dist: DistributionLD,
    theta_grid: np.ndarray | None = None,
    theta_range: tuple[float, float] = (-6.0, 6.0),
    target_mean: float | None = None,
    num_points: int = 600,
) -> dict[str, np.ndarray | float | None]:
    """Return curve data for exponential tilting.

    Parameters
    ----------
    dist:
        DistributionLD object.
    theta_grid:
        Optional grid of theta values.
    theta_range:
        Range used when theta_grid is not provided.
    target_mean:
        Optional target mean. If provided, theta_star is computed numerically.
    num_points:
        Number of grid points when theta_grid is not provided.

    Returns
    -------
    dict
        Dictionary containing theta values, CGF values, tilted means,
        and optionally theta_star.
    """
    theta_star: float | None = None

    if target_mean is not None:
        theta_star = theta_for_tilted_mean(
            dist=dist,
            target_mean=target_mean,
            theta_range=(-10.0, 10.0),
        )

    if theta_grid is None:
        lower, upper = theta_range

        if theta_star is not None:
            lower = min(lower, theta_star) - 0.5
            upper = max(upper, theta_star) + 0.5

        theta_grid = np.linspace(lower, upper, num_points)

    curve = evaluate_tilting_curve(
        dist=dist,
        theta_grid=theta_grid,
    )

    return {
        **curve,
        "theta_star": theta_star,
        "target_mean": target_mean,
    }


def plot_tilting(
    dist: DistributionLD,
    target_mean: float | None = None,
    theta: float | None = None,
    theta_grid: np.ndarray | None = None,
    theta_range: tuple[float, float] = (-6.0, 6.0),
    ax=None,
):
    """Plot Gamma(theta) and the tilted mean Gamma'(theta).

    The plot uses two vertical axes:

    - left axis: cumulant generating function Gamma(theta)
    - right axis: tilted mean Gamma'(theta)

    Parameters
    ----------
    dist:
        DistributionLD object.
    target_mean:
        Optional target tilted mean.
    theta:
        Optional theta value to highlight.
    theta_grid:
        Optional theta grid.
    theta_range:
        Theta range used when theta_grid is not provided.
    ax:
        Optional matplotlib axis.

    Returns
    -------
    tuple
        ``(fig, (ax1, ax2))``.
    """
    import matplotlib.pyplot as plt

    data = tilting_curve_data(
        dist=dist,
        theta_grid=theta_grid,
        theta_range=theta_range,
        target_mean=target_mean,
    )

    theta_values = data["theta"]
    cgf_values = data["cgf"]
    mean_values = data["mean_under_tilt"]
    theta_star = data["theta_star"]

    if theta is None and theta_star is not None:
        theta = float(theta_star)

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(10, 5))
    else:
        ax1 = ax
        fig = ax1.figure

    ax1.plot(
        theta_values,
        cgf_values,
        label=r"$\Gamma(\theta)$",
    )

    ax1.set_xlabel(r"Tilt parameter $\theta$")
    ax1.set_ylabel(r"CGF $\Gamma(\theta)$")

    ax2 = ax1.twinx()

    ax2.plot(
        theta_values,
        mean_values,
        linestyle="--",
        label=r"$\Gamma'(\theta)$",
    )

    ax2.set_ylabel(r"Tilted mean $\Gamma'(\theta)$")

    if target_mean is not None:
        ax2.axhline(
            target_mean,
            linestyle=":",
            label=rf"target mean ${target_mean:.2f}$",
        )

    if theta_star is not None:
        ax1.axvline(
            theta_star,
            linestyle=":",
            label=rf"$\theta^*={theta_star:.2f}$",
        )

        if dist.domain_contains(float(theta_star)):
            ax2.scatter(
                [theta_star],
                [dist.mean_under_tilt(float(theta_star))],
                zorder=5,
            )

    if theta is not None:
        ax1.axvline(
            theta,
            linestyle="--",
            label=rf"chosen $\theta={theta:.2f}$",
        )

        if dist.domain_contains(float(theta)):
            ax2.scatter(
                [theta],
                [dist.mean_under_tilt(float(theta))],
                zorder=5,
            )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
    )

    ax1.set_title(f"Exponential tilting: {dist.name}")
    fig.tight_layout()

    return fig, (ax1, ax2)