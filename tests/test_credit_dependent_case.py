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


def test_compute_large_loss_asymptotic_point_returns_consistent_values():
    n = 100
    a = 0.5

    point = compute_large_loss_asymptotic_point(
        n=n,
        p=0.02,
        rho=0.5,
        a=a,
        scale=1.0,
    )

    assert point.n == n
    assert np.isclose(point.q_n, 1.0 - n ** (-a))
    assert point.threshold == int(np.ceil(n * point.q_n))
    assert 0.0 < point.probability < 1.0
    assert np.isclose(point.log_probability, np.log(point.probability))
    assert np.isclose(point.log_n, np.log(n))
    assert np.isclose(
        point.empirical_decay_ratio,
        -point.log_probability / np.log(n),
    )
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


def test_fitted_decay_rate_converges_toward_theoretical_rate_from_above():
    # Theorem 6.1 gives log P[L_n >= n q_n] ~ -beta log n with subpolynomial
    # corrections that vanish like 1/log n, so the fitted slope converges very
    # slowly (it is still ~2.4 at n = 1e8 for beta = 1.5). Asserting closeness
    # would be meaningless at test-sized n; what the theorem does pin down is
    # the direction: the fitted rate stays above beta and decreases toward it
    # as the grid of portfolio sizes moves up.
    p = 0.02
    rho = 0.5
    a = 0.5

    small_grid = compute_large_loss_asymptotic_curve(
        n_values=[50, 100, 200],
        p=p,
        rho=rho,
        a=a,
        scale=1.0,
    )
    large_grid = compute_large_loss_asymptotic_curve(
        n_values=[1_000, 2_000, 5_000, 10_000],
        p=p,
        rho=rho,
        a=a,
        scale=1.0,
    )

    beta = gaussian_copula_large_loss_decay_rate(a=a, rho=rho)

    assert small_grid.fitted_decay_rate > large_grid.fitted_decay_rate > beta

    # Along a single grid, the point-wise ratio -log P_n / log n must decrease
    # monotonically toward beta as well.
    ratios = [point.empirical_decay_ratio for point in large_grid.points]
    assert all(earlier > later for earlier, later in zip(ratios, ratios[1:]))
    assert all(ratio > beta for ratio in ratios)


def test_fitted_decay_rate_is_smaller_for_more_systemic_portfolios():
    # Higher rho means a smaller theoretical exponent, i.e. extreme losses are
    # less rare. The fitted finite-n slope must preserve this ordering.
    a = 0.5
    grid = [500, 1_000, 2_000]

    weakly_dependent = compute_large_loss_asymptotic_curve(
        n_values=grid,
        p=0.02,
        rho=0.3,
        a=a,
        scale=1.0,
    )
    strongly_dependent = compute_large_loss_asymptotic_curve(
        n_values=grid,
        p=0.02,
        rho=0.7,
        a=a,
        scale=1.0,
    )

    assert (
        strongly_dependent.fitted_decay_rate < weakly_dependent.fitted_decay_rate
    )