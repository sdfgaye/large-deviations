# One-factor Gaussian copula model

## 1. Model

We model a homogeneous credit portfolio with $n$ obligors.

The default indicator of obligor $k$ is

$$
Y_k = 1_{\{X_k>x_p\}},
$$

where

$$
x_p = \Phi^{-1}(1-p).
$$

The latent variable is

$$
X_k =
\rho Z+\sqrt{1-\rho^2}\varepsilon_k,
$$

with independent standard normal variables

$$
Z,\varepsilon_1,\ldots,\varepsilon_n.
$$

The systematic factor is $Z$. The idiosyncratic noise is $\varepsilon_k$.

## 2. Why the threshold is $x_p=\Phi^{-1}(1-p)$

Each $X_k$ is standard normal.

So

$$
\mathbb P(Y_k=1) =
\mathbb P(X_k>x_p).
$$

To make this equal to $p$, choose $x_p$ such that

$$
1-\Phi(x_p)=p.
$$

Therefore,

$$
x_p=\Phi^{-1}(1-p).
$$

## 3. Conditional default probability

Conditionally on $Z=z$,

$$
X_k = \rho z+\sqrt{1-\rho^2}\varepsilon_k.
$$

So

$$
p(z) =
\mathbb P(Y_k=1\mid Z=z) =
\Phi\left(
\frac{\rho z+\Phi^{-1}(p)}
{\sqrt{1-\rho^2}}
\right).
$$

This function is increasing in $z$.

A large positive $z$ is a bad systematic state because it raises the conditional default probability for every obligor at the same time.

## 4. Conditional loss distribution

The portfolio loss is

$$
L_n = Y_1+\cdots+Y_n.
$$

Given $Z=z$, the variables $Y_1,\ldots,Y_n$ are independent Bernoulli variables with common probability $p(z)$.

Hence

$$
L_n\mid Z=z
\sim
\mathrm{Binomial}(n,p(z)).
$$

This gives a simple simulation algorithm.

## 5. Simulation algorithm

> **Input:** $n$, $p$, $\rho$, `sample_size`.
>
> **For each Monte Carlo path:**
>
> 1. Sample $Z \sim \mathcal N(0,1)$.
> 2. Compute $p_Z  = \Phi\left(\frac{\rho Z+\Phi^{-1}(p)}{\sqrt{1-\rho^2}}\right)$
>
> 3. Sample $L_n \sim \mathrm{Binomial}(n,p_Z)$.
>
> 4. Store $L_n/n$.


This is faster than simulating every obligor separately.

## 6. Limit behavior

### Independent Bernoulli portfolio

If $\rho=0$, then

$$
p(Z)=p.
$$

So

$$
\frac{L_n}{n}\to p.
$$

The limiting loss rate is deterministic.

### Dependent one-factor portfolio

If $\rho>0$, then $p(Z)$ is random.

Conditionally,

$$
\frac{L_n}{n}\to p(Z).
$$

Unconditionally,

$$
\frac{L_n}{n}\Rightarrow p(Z).
$$

The limiting loss rate is random.

This is the mathematical reason dependence creates clustered losses.

## 7. Large-loss threshold regime

The extreme-loss theorem studies thresholds

$$
q_n\uparrow 1.
$$

A standard regime is

$$
1-q_n=O(n^{-a}),
\qquad
0<a\le 1.
$$

The event is

$$
\{L_n\ge n q_n\}.
$$

This means that almost the entire portfolio defaults.

## 8. Large-loss asymptotic

In the homogeneous one-factor Gaussian copula model,

$$
\lim_{n\to\infty}
\frac{1}{\log n}
\log \mathbb P(L_n\ge n q_n) =
-a\frac{1-\rho^2}{\rho^2}.
$$

Equivalently,

$$
\mathbb P(L_n\ge n q_n)
\approx
n^{-a(1-\rho^2)/\rho^2}.
$$

The decay is polynomial in $n$, not exponential in $n$.

## 9. Interpretation of the exponent

The exponent is

$$
\gamma =
a\frac{1-\rho^2}{\rho^2}.
$$

If $\rho$ is small, then $\gamma$ is large, so extreme losses are very rare.

If $\rho$ is close to one, then $\gamma$ is small, so extreme losses are much less rare.

The systematic factor dominates the portfolio.

## 10. Connection to importance sampling

The final project target is a two-step importance sampling estimator.

The two steps are:

```text
1. shift the distribution of the systematic factor Z
2. conditionally on Z, tilt the Bernoulli defaults
```

This mirrors the model structure:

$$
\mathbb P(L_n\ge nq_n) =
\mathbb E\left[
\mathbb P(L_n\ge nq_n\mid Z)
\right].
$$

Notebook 04 builds the model.

Notebook 05 should attack the rare-event estimator.
