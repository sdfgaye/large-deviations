import numpy as np
import pytest

from large_deviations.foundations import (
    DistributionLD,
    safe_xlogy,
    validate_positive,
    validate_probability,
)


def test_safe_xlogy_uses_zero_convention():
    assert np.isclose(safe_xlogy(0.0, 0.2), 0.0)


def test_safe_xlogy_returns_expected_value():
    x = 0.5
    y = 0.2
    expected = x * np.log(x / y)

    assert np.isclose(safe_xlogy(x, y), expected)


def test_safe_xlogy_rejects_negative_x():
    with pytest.raises(ValueError, match="x must be non-negative"):
        safe_xlogy(-0.1, 0.2)


def test_safe_xlogy_rejects_non_positive_y():
    with pytest.raises(ValueError, match="y must be positive"):
        safe_xlogy(0.1, 0.0)

    with pytest.raises(ValueError, match="y must be positive"):
        safe_xlogy(0.1, -0.2)


def test_validate_probability_accepts_strict_probability():
    validate_probability(0.2)


def test_validate_probability_rejects_boundary_and_invalid_values():
    with pytest.raises(ValueError, match="0 < p < 1"):
        validate_probability(0.0)

    with pytest.raises(ValueError, match="0 < p < 1"):
        validate_probability(1.0)

    with pytest.raises(ValueError, match="0 < p < 1"):
        validate_probability(-0.1)

    with pytest.raises(ValueError, match="0 < p < 1"):
        validate_probability(1.1)


def test_validate_positive_accepts_positive_value():
    validate_positive("lam", 3.0)


def test_validate_positive_rejects_zero_and_negative_values():
    with pytest.raises(ValueError, match="lam must be strictly positive"):
        validate_positive("lam", 0.0)

    with pytest.raises(ValueError, match="lam must be strictly positive"):
        validate_positive("lam", -1.0)


def test_distribution_ld_container_exposes_expected_fields():
    dist = DistributionLD(
        name="Toy",
        parameters={"alpha": 1.0},
        cgf=lambda theta: theta**2,
        rate_function=lambda x: x**2,
        tilted_parameter=lambda theta: theta,
        domain_contains=lambda theta: np.isfinite(theta),
        mean_under_tilt=lambda theta: 2.0 * theta,
    )

    assert dist.name == "Toy"
    assert dist.parameters == {"alpha": 1.0}
    assert np.isclose(dist.cgf(2.0), 4.0)
    assert np.isclose(dist.rate_function(3.0), 9.0)
    assert np.isclose(dist.tilted_parameter(1.5), 1.5)
    assert dist.domain_contains(0.0)
    assert np.isclose(dist.mean_under_tilt(2.0), 4.0)