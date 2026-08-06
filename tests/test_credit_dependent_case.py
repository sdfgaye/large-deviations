import numpy as np

from large_deviations.credit import (
    compute_large_loss_asymptotic_curve,
    compute_large_loss_asymptotic_point,
    estimate_loglog_decay_rate,
    gaussian_copula_large_loss_decay_rate,
)


def test_estimate_loglog_decay_rate_recovers_known_slope():
    log_n = np.log(np.array([100, 200, 500, 1_000], dtype=float))
    beta = 2.5
    intercept = 0.7
    log_p = intercept - beta * log_n

    observed_beta, observed_intercept = estimate_loglog_decay_rate(
        log_n,
        log_p,
    )

    assert np.isclose(observed_beta, beta)
    assert np.isclose(observed_intercept, intercept)


def test_compute_large_loss_asymptotic_point_returns_valid_values():
    point = compute_large_loss_asymptotic_point(
        n=100,
        p=0.02,
        rho=0.5,
        a=0.5,
        scale=1.0,
    )

    assert point.n == 100
    assert 0.0 < point.q_n < 1.0
    assert 0 <= point.threshold <= point.n
    assert 0.0 < point.probability < 1.0
    assert np.isfinite(point.log_probability)
    assert np.isfinite(point.empirical_decay_ratio)
    assert point.factor_split is not None
    assert point.quadrature_error >= 0.0


def test_compute_large_loss_asymptotic_curve_has_theoretical_rate():
    rho = 0.5
    a = 0.5

    result = compute_large_loss_asymptotic_curve(
        n_values=[50, 100, 200],
        p=0.02,
        rho=rho,
        a=a,
        scale=1.0,
    )

    expected_rate = gaussian_copula_large_loss_decay_rate(
        a=a,
        rho=rho,
    )

    assert len(result.points) == 3
    assert np.isclose(result.theoretical_decay_rate, expected_rate)
    assert np.isfinite(result.fitted_decay_rate)
    assert np.isfinite(result.fitted_intercept)