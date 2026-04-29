# large-deviations

A research-style quantitative finance project on rare events, exponential tilting, and large loss asymptotics.

## Overview

This repository studies the core ideas of **large deviations theory** with a final focus on **credit portfolio risk**.

The project is based on Huyen Pham's *Large Deviations in Mathematical Finance* and follows a simple workflow:

**Derive -> Code -> Apply**

The objective is to move from foundational tools such as:
- cumulant generating functions,
- exponential change of measure,
- Fenchel-Legendre transforms,
- Cramer's theorem,
- importance sampling,

to a final implementation of **extreme portfolio loss asymptotics** in a **one-factor Gaussian copula model**.

## Why this project?

Rare events are exactly where standard intuition and naive Monte Carlo begin to fail.

In quantitative finance, these questions appear naturally in:
- tail risk,
- stress scenarios,
- rare default events,
- variance reduction,
- and risk management of large portfolios.

Large deviations provide the mathematical language for understanding how these probabilities decay, while exponential tilting provides the computational mechanism for estimating them efficiently.

This project is also designed as a GitHub showcase for quantitative research / quant developer roles:
the goal is not only to reproduce formulas, but to build a clean bridge between theory, numerical experiments, and implementation.

## Final target

The long-term goal of the repository is to reproduce and implement the large-loss credit risk result for a homogeneous portfolio in a one-factor Gaussian copula setting:

$$
\lim_{n \to \infty} \frac{1}{\ln n}\ln \mathbb{P}(L_n \ge n q_n)
=
-a \frac{1-\rho^2}{\rho^2},
\qquad
q_n \uparrow 1,\quad 1-q_n = O(n^{-a}),\ 0<a\le 1.
$$

This result highlights a key phenomenon in credit risk:

> when dependence becomes more systemic, extreme losses become much less rare.

## Roadmap

### Module 0 - Foundations
- cumulant generating function $\Gamma(\theta)$
- exponential change of measure
- convexity and saddle-point intuition
- Fenchel-Legendre transform
- Bernoulli / Poisson / Gaussian / Exponential examples

### Module 1 - Cramer's theorem
- exponential decay of rare-event probabilities
- upper bound and lower bound
- empirical verification of the log-asymptotic regime

### Module 2 - Importance Sampling
- rare-event estimation
- exponential tilting
- asymptotically optimal importance sampling

### Module 3 - Gartner-Ellis
- extension beyond i.i.d. settings
- limiting cumulant generating functions
- bridge toward dependent models

### Module 4 - Credit Portfolio Risk
- one-factor Gaussian copula
- homogeneous default portfolio
- extreme loss asymptotics
- two-step importance sampling

## Theory notes

### Foundations

- [Exponential tilting and importance sampling](docs/foundations/exponential_tilting.md)

### Distributions

- [Bernoulli distribution](docs/distributions/bernoulli.md)

## Repository structure

```text
large-deviations/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- notebooks/
|-- src/
|   `-- large_deviations/
|-- tests/
|-- docs/
`-- assets/
```

## Current status

- [x] Project initialized
- [x] Packaging scaffold in place
- [x] Foundations module
- [x] Bernoulli implementation
- [x] Bernoulli tests
- [x] Exponential tilting theory note
- [x] Bernoulli theory note
- [ ] Poisson implementation
- [ ] Cramer experiments
- [ ] Importance sampling module
- [ ] Credit portfolio final module

## Tech stack

- Python 3.10+
- NumPy
- SciPy
- Pandas
- Matplotlib
- JupyterLab
- ipywidgets
- pytest

## Project philosophy

This is not meant to be a loose collection of notebooks.

The goal is to build a repository that is:
- mathematically rigorous,
- readable,
- reproducible,
- modular,
- and useful as a long-term research / portfolio project.

Each piece of code should trace back to a clear mathematical object, and each notebook should help explain why the implementation matters.

## Reference

H. Pham, *Large Deviations in Mathematical Finance* (2010).

## Author

**Souleymane Gaye**
