"""Generate the README preview figure for notebook 01.

The figure is intentionally generated from quantitative objects used in the
project: exponential tilting, binomial rare-event probabilities, and Cramer's
log-asymptotic scale.
"""

from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "assets" / "notebook_01_bernoulli_preview.svg"


def bernoulli_cgf(theta: np.ndarray | float, p: float) -> np.ndarray | float:
    """Bernoulli cumulant generating function."""
    return np.log1p(p * (np.exp(theta) - 1.0))


def bernoulli_tilted_mean(theta: np.ndarray | float, p: float) -> np.ndarray | float:
    """Mean under the exponentially tilted Bernoulli law."""
    gamma = bernoulli_cgf(theta, p)
    return p * np.exp(theta - gamma)


def bernoulli_rate(x: float, p: float) -> float:
    """Bernoulli Cramer rate function."""
    if not 0.0 <= x <= 1.0:
        return math.inf

    def xlogy(a: float, b: float) -> float:
        if a == 0.0:
            return 0.0
        return a * math.log(a / b)

    return xlogy(x, p) + xlogy(1.0 - x, 1.0 - p)


def binomial_tail_threshold(n: int, level: float) -> int:
    """Count threshold for {S_n / n >= level}."""
    return int(math.ceil(n * level - 1e-12))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    p = 0.20
    ell = 0.60
    n = 80

    theta_star = math.log(ell * (1.0 - p) / (p * (1.0 - ell)))
    rate_ell = bernoulli_rate(ell, p)

    plt.rcParams.update(
        {
            "figure.figsize": (13, 6),
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )

    fig = plt.figure(figsize=(13.5, 6.2))
    fig.patch.set_facecolor("white")

    title = (
        "Notebook 01 — Bernoulli Exponential Tilting\n"
        "CGF, tilted measure, Cramer's rate function, and rare-event simulation"
    )
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    axes = fig.subplots(1, 3)

    # Panel 1 — tilted mean and saddle point
    theta_grid = np.linspace(-3.0, 4.0, 500)
    tilted_means = bernoulli_tilted_mean(theta_grid, p)

    axes[0].plot(theta_grid, tilted_means, linewidth=2.4, label=r"$\Gamma'(\theta)$")
    axes[0].axhline(ell, linestyle="--", linewidth=1.6, label=rf"target $\ell={ell:.2f}$")
    axes[0].axvline(theta_star, linestyle=":", linewidth=1.8, label=rf"$\theta^*={theta_star:.2f}$")
    axes[0].scatter([theta_star], [ell], s=65, zorder=5)
    axes[0].set_title("Saddle-point tilt")
    axes[0].set_xlabel(r"tilt parameter $\theta$")
    axes[0].set_ylabel(r"tilted mean $\Gamma'(\theta)$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=True, fontsize=9)

    # Panel 2 — original vs tilted binomial mass
    counts = np.arange(0, n + 1)
    empirical_means = counts / n
    original_pmf = binom.pmf(counts, n, p)
    tilted_pmf = binom.pmf(counts, n, ell)

    axes[1].plot(empirical_means, original_pmf, linewidth=2.2, label=rf"original $p={p:.2f}$")
    axes[1].plot(empirical_means, tilted_pmf, linewidth=2.2, label=rf"tilted $p_\theta={ell:.2f}$")
    axes[1].axvline(ell, linestyle="--", linewidth=1.6, label=rf"rare level $\ell={ell:.2f}$")
    axes[1].fill_between(
        empirical_means,
        0.0,
        original_pmf,
        where=empirical_means >= ell,
        alpha=0.25,
    )
    axes[1].set_title("Rare event made typical")
    axes[1].set_xlabel(r"empirical mean $S_n/n$")
    axes[1].set_ylabel("binomial probability mass")
    axes[1].legend(frameon=True, fontsize=9)

    # Panel 3 — exact log tail versus Cramer scale
    n_grid = np.arange(20, 241, 10)
    exact_log_scale = []

    for n_value in n_grid:
        threshold = binomial_tail_threshold(int(n_value), ell)
        tail_probability = binom.sf(threshold - 1, int(n_value), p)
        exact_log_scale.append(math.log(tail_probability) / n_value)

    exact_log_scale = np.array(exact_log_scale)
    cramer_scale = -rate_ell * np.ones_like(n_grid, dtype=float)

    axes[2].plot(n_grid, exact_log_scale, marker="o", linewidth=2.2, label=r"exact $\frac{1}{n}\log P$")
    axes[2].plot(n_grid, cramer_scale, linestyle="--", linewidth=2.0, label=rf"$-\Gamma^*(\ell)={-rate_ell:.3f}$")
    axes[2].set_title("Log-asymptotic convergence")
    axes[2].set_xlabel("sample size n")
    axes[2].set_ylabel(r"log probability per sample")
    axes[2].legend(frameon=True, fontsize=9)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(OUTPUT_PATH, format="svg", bbox_inches="tight")
    print(f"Saved preview to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()