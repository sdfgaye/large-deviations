"""Generate the README preview SVG for notebook 02.

Run from the repository root:

    python scripts/generate_notebook_02_preview.py

It writes:

    assets/notebook_02_cramer_preview.svg
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom


P_DEFAULT = 0.02
Q_TAIL = 0.10
N_VALUES = np.array([50, 100, 200, 500, 1_000, 2_000, 5_000], dtype=int)
OUTPUT_PATH = Path("assets/notebook_02_cramer_preview.svg")


def bernoulli_rate_function(x: np.ndarray | float, p: float) -> np.ndarray | float:
    """Bernoulli Cramer rate function.

    I(x) = x log(x/p) + (1-x) log((1-x)/(1-p)) for x in [0, 1].
    The implementation handles x=0 and x=1 by continuity.
    """
    x_arr = np.asarray(x, dtype=float)
    out = np.full_like(x_arr, np.inf, dtype=float)

    mask = (0.0 <= x_arr) & (x_arr <= 1.0)
    xm = x_arr[mask]

    values = np.zeros_like(xm)

    positive = xm > 0.0
    values[positive] += xm[positive] * np.log(xm[positive] / p)

    below_one = xm < 1.0
    values[below_one] += (1.0 - xm[below_one]) * np.log(
        (1.0 - xm[below_one]) / (1.0 - p)
    )

    out[mask] = values

    if np.isscalar(x):
        return float(out)
    return out


def bernoulli_cgf(theta: float, p: float) -> float:
    """Cumulant generating function Gamma(theta) for Bernoulli(p)."""
    return float(np.log1p(p * np.expm1(theta)))


def theta_for_bernoulli_mean(target_mean: float, p: float) -> float:
    """Solve Gamma'(theta)=target_mean for Bernoulli(p)."""
    if not 0.0 < target_mean < 1.0:
        raise ValueError("target_mean must lie in (0, 1).")
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1).")
    return math.log(target_mean * (1.0 - p) / (p * (1.0 - target_mean)))


def binomial_tail_threshold(n: int, level: float) -> int:
    """Integer k for the event S_n / n >= level."""
    return int(math.ceil(n * level - 1e-12))


def exact_binomial_log_tail(n: int, p: float, level: float) -> float:
    """log P[Binomial(n, p) >= ceil(n * level)]."""
    k = binomial_tail_threshold(n, level)
    return float(binom.logsf(k - 1, n, p))


def build_preview(output_path: Path = OUTPUT_PATH) -> None:
    """Build and save the notebook 02 preview SVG."""
    p = P_DEFAULT
    q = Q_TAIL

    theta_q = theta_for_bernoulli_mean(q, p)
    rate_q = float(bernoulli_rate_function(q, p))

    log_tails = np.array([exact_binomial_log_tail(int(n), p, q) for n in N_VALUES])
    empirical_rates = -log_tails / N_VALUES
    cramer_log_approx = -N_VALUES * rate_q

    x_grid = np.linspace(0.0, 0.20, 500)
    rate_grid = bernoulli_rate_function(x_grid, p)

    plt.rcParams.update(
        {
            "figure.figsize": (13.6, 6.6),
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, axes = plt.subplots(1, 3)

    # 1) Bernoulli rate function
    ax = axes[0]
    ax.plot(x_grid, rate_grid, label=r"$\Gamma^*(x)$", linewidth=2.0)
    ax.axvline(p, linestyle="--", label=r"$p=2\%$")
    ax.axvline(q, linestyle="--", label=r"$q=10\%$")
    ax.scatter([q], [rate_q], zorder=3)
    ax.set_title("Bernoulli rate function")
    ax.set_xlabel("realized default rate x")
    ax.set_ylabel(r"$\Gamma^*(x)$")
    ax.set_xlim(0.0, 0.20)
    ax.set_ylim(0.0, 0.313)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    # 2) Convergence of exact logarithmic rate to Gamma*(q)
    ax = axes[1]
    ax.semilogx(
        N_VALUES,
        empirical_rates,
        marker="o",
        linewidth=2.0,
        label=r"exact $-\log P/n$",
    )
    ax.axhline(rate_q, linestyle="--", linewidth=1.8, label=r"$\Gamma^*(q)$")
    ax.set_title("Cramer convergence")
    ax.set_xlabel("portfolio size n")
    ax.set_ylabel(r"$-\frac{1}{n}\log P(S_n/n \geq q)$")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    # 3) Exact log tail versus Cramer leading-order approximation
    ax = axes[2]
    ax.semilogx(
        N_VALUES,
        log_tails,
        marker="o",
        linewidth=2.0,
        label="exact binomial tail",
    )
    ax.semilogx(
        N_VALUES,
        cramer_log_approx,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label=r"$-n\Gamma^*(q)$",
    )
    ax.set_title("Log tail probability")
    ax.set_xlabel("portfolio size n")
    ax.set_ylabel(r"$\log P(S_n/n \geq q)$")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    fig.suptitle("Notebook 02 — Cramer's Theorem: Bernoulli Tail Risk", fontsize=18, y=0.97)
    fig.text(
        0.5,
        0.045,
        rf"p={p:.2%}, q={q:.2%}, $\theta_q$={theta_q:.4f}, "
        rf"$\Gamma^*(q)$={rate_q:.4f} — exact binomial benchmark before importance sampling",
        ha="center",
        fontsize=11,
    )

    fig.tight_layout(rect=[0.02, 0.07, 1.0, 0.94], w_pad=2.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_preview()
    print(f"Wrote {OUTPUT_PATH}")
