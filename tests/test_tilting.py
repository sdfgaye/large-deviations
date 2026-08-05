import math

import numpy as np
import pytest

from large_deviations.distributions import bernoulli_ld
from large_deviations.foundations import DistributionLD
from large_deviations.tilting import (
    evaluate_tilting_curve,
    theta_for_tilted_mean,
    tilting_summary,
    unit_weight_multiplier,
)


def gaussian_toy_distribution(domain_half_width: float = np.inf) -> DistributionLD:
    """Standard Gaussian large-deviation objects, optionally with a truncated domain."""
    return DistributionLD(
        name="GaussianToy",
        parameters={},
        cgf=lambda theta: 0.5 * theta**2,
        rate_function=lambda x: 0.5 * x**2,
        tilted_parameter=lambda theta: theta,
        domain_contains=lambda theta: abs(theta) <= domain_half_width,
        mean_under_tilt=lambda theta: theta,
    )


@pytest.mark.parametrize(
    ("p", "q"),
    [(0.02, 0.10), (0.20, 0.50), (0.50, 0.10), (0.30, 0.90)],
)
def test_theta_for_tilted_mean_matches_bernoulli_closed_form(p, q):
    # For Bernoulli(p), solving Gamma'(theta) = q has the closed form
    # theta = log(q (1 - p) / (p (1 - q))).
    dist = bernoulli_ld(p)

    theta = theta_for_tilted_mean(dist, target_mean=q)
    expected = math.log(q * (1.0 - p) / (p * (1.0 - q)))

    assert np.isclose(theta, expected, atol=1e-8)


def test_theta_for_tilted_mean_recovers_untilted_mean():
    dist = bernoulli_ld(0.3)

    theta = theta_for_tilted_mean(dist, target_mean=0.3)

    assert np.isclose(theta, 0.0, atol=1e-8)


def test_theta_for_tilted_mean_rejects_unreachable_target():
    dist = bernoulli_ld(0.2)

    with pytest.raises(ValueError, match="Could not bracket"):
        theta_for_tilted_mean(dist, target_mean=1.5)


def test_unit_weight_multiplier_is_exp_theta():
    assert np.isclose(unit_weight_multiplier(0.7), math.exp(0.7))
    assert np.isclose(unit_weight_multiplier(0.0), 1.0)


def test_unit_weight_multiplier_rejects_non_finite_theta():
    with pytest.raises(ValueError, match="theta must be finite"):
        unit_weight_multiplier(np.inf)


def test_tilting_summary_is_consistent_with_distribution():
    p = 0.2
    theta = 0.7
    dist = bernoulli_ld(p)

    summary = tilting_summary(dist, theta)

    assert summary.distribution == "Bernoulli"
    assert np.isclose(summary.theta, theta)
    assert np.isclose(summary.unit_weight_multiplier, math.exp(theta))
    assert np.isclose(summary.cgf, dist.cgf(theta))
    assert np.isclose(summary.mean_under_tilt, dist.mean_under_tilt(theta))
    assert np.isclose(summary.tilted_parameter, dist.tilted_parameter(theta))


def test_tilting_summary_rejects_theta_outside_domain():
    dist = gaussian_toy_distribution(domain_half_width=1.0)

    with pytest.raises(ValueError, match="outside the CGF domain"):
        tilting_summary(dist, 2.0)


def test_tilting_summary_to_dict_round_trip():
    summary = tilting_summary(bernoulli_ld(0.2), 0.5)

    as_dict = summary.to_dict()

    assert as_dict["distribution"] == "Bernoulli"
    assert np.isclose(as_dict["theta"], 0.5)


def test_evaluate_tilting_curve_returns_exact_values_inside_domain():
    dist = gaussian_toy_distribution()
    theta_grid = np.array([-1.0, 0.0, 0.5, 2.0])

    curve = evaluate_tilting_curve(dist, theta_grid)

    assert np.allclose(curve["theta"], theta_grid)
    assert np.allclose(curve["cgf"], 0.5 * theta_grid**2)
    assert np.allclose(curve["mean_under_tilt"], theta_grid)


def test_evaluate_tilting_curve_marks_out_of_domain_values_as_nan():
    dist = gaussian_toy_distribution(domain_half_width=1.0)
    theta_grid = np.array([-2.0, -0.5, 0.5, 2.0])

    curve = evaluate_tilting_curve(dist, theta_grid)

    assert np.isnan(curve["cgf"][0])
    assert np.isnan(curve["cgf"][3])
    assert np.allclose(curve["cgf"][1:3], 0.5 * theta_grid[1:3] ** 2)
