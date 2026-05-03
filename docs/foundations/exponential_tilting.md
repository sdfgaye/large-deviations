# Exponential Tilting and Importance Sampling

## 1. Rare-event estimation problem

Let $X$ be a real-valued random variable with probability distribution $\mu$.

We are interested in estimating a rare-event probability of the form:

```math
p = \mathbb{P}(X > \ell).
```

Define the rare event:

```math
A = \{X > \ell\}.
```

Then:

```math
p = \mathbb{P}(A).
```

Using the distribution $\mu$, this probability can be written as:

```math
p = \int 1_A(x)\,\mu(dx).
```

This means that we integrate the indicator of the rare event under the original probability law.

In Monte Carlo simulation, the naive estimator is:

```math
\widehat p_N
=
\frac{1}{N}
\sum_{i=1}^N 1_A(X_i),
```

where $X_1,\dots,X_N$ are independent samples from $\mu$.

This estimator is unbiased, but when $A$ is rare, most simulated values satisfy:

```math
1_A(X_i)=0.
```

Therefore, the estimator can have high relative error.

The goal of importance sampling is to simulate from a different probability distribution under which the rare event is more likely, while correcting the estimator so that it remains unbiased.

---

## 2. Change of probability measure

Let $\nu$ be another probability distribution.

Assume that $\nu$ is absolutely continuous with respect to $\mu$, and write its density as:

```math
f(x)
=
\frac{d\nu}{d\mu}(x).
```

Then:

```math
\nu(dx)=f(x)\mu(dx).
```

Whenever $f(x)>0$, we can rewrite:

```math
\mu(dx)=\frac{1}{f(x)}\nu(dx).
```

Starting from:

```math
p = \int 1_A(x)\mu(dx),
```

we substitute $\mu(dx)$ by $\frac{1}{f(x)}\nu(dx)$:

```math
p
=
\int 1_A(x)\frac{1}{f(x)}\nu(dx).
```

Therefore:

```math
p
=
\mathbb{E}_{\nu}
\left[
1_A(X)\frac{1}{f(X)}
\right].
```

Since:

```math
\frac{1}{f(X)}
=
\frac{d\mu}{d\nu}(X),
```

we obtain the importance sampling identity:

```math
p
=
\mathbb{E}_{\nu}
\left[
1_A(X)\frac{d\mu}{d\nu}(X)
\right].
```

The new law $\nu$ is used to simulate more efficiently, and the likelihood ratio $\frac{d\mu}{d\nu}$ corrects the bias introduced by the change of measure.

---

## 3. Why exponential tilting?

Suppose the rare event is of the form:

```math
X > \ell.
```

Then large values of $X$ contribute most to the probability.

A natural way to make large values more likely is to give more weight to large $x$. For $\theta > 0$, the function

```math
e^{\theta x}
```

is increasing in $x$.

So we define a new probability measure proportional to:

```math
e^{\theta x}\mu(dx).
```

Formally:

```math
\nu(dx) \propto e^{\theta x}\mu(dx).
```

This is not yet a probability measure, because its total mass is not necessarily equal to $1$. We need to normalize it.

---

## 4. Cumulant generating function

The normalizing constant is:

```math
\int e^{\theta x}\mu(dx)
=
\mathbb{E}_{\mu}
\left[
e^{\theta X}
\right].
```

The cumulant generating function is defined by:

```math
\Gamma(\theta)
=
\log
\mathbb{E}_{\mu}
\left[
e^{\theta X}
\right].
```

Therefore:

```math
\mathbb{E}_{\mu}
\left[
e^{\theta X}
\right]
=
e^{\Gamma(\theta)}.
```

The set of admissible values of $\theta$ is the effective domain:

```math
D(\Gamma)
=
\left\{
\theta \in \mathbb{R}
\;:\;
\Gamma(\theta)<\infty
\right\}.
```


---

## 5. Definition of the exponentially tilted law

For $\theta \in D(\Gamma)$, the exponentially tilted law is defined by:

```math
\mu_{\theta}(dx)
=
e^{\theta x-\Gamma(\theta)}\mu(dx).
```

This is a probability measure because:

```math
\int \mu_{\theta}(dx)
=
\int e^{\theta x-\Gamma(\theta)}\mu(dx).
```

Since $e^{-\Gamma(\theta)}$ does not depend on $x$:

```math
\int \mu_{\theta}(dx)
=
e^{-\Gamma(\theta)}
\int e^{\theta x}\mu(dx).
```

By definition of $\Gamma$:

```math
\int e^{\theta x}\mu(dx)
=
e^{\Gamma(\theta)}.
```

Hence:

```math
\int \mu_{\theta}(dx)
=
e^{-\Gamma(\theta)}
e^{\Gamma(\theta)}
=
1.
```

So $\mu_\theta$ is a valid probability distribution.

---

## 6. Importance sampling weight under exponential tilting

The density of the tilted law with respect to the original law is:

```math
\frac{d\mu_\theta}{d\mu}(x)
=
e^{\theta x-\Gamma(\theta)}.
```

Therefore, the inverse likelihood ratio is:

```math
\frac{d\mu}{d\mu_\theta}(x)
=
e^{-\theta x+\Gamma(\theta)}.
```

So for any event $A$, we can write:

```math
\mathbb{P}_{\mu}(A)
=
\mathbb{E}_{\mu_\theta}
\left[
1_A(X)
e^{-\theta X+\Gamma(\theta)}
\right].
```

This is the core identity behind exponential-tilting importance sampling.

In words:

> We simulate under the tilted law $\mu_\theta$, but we correct each simulated value by the likelihood ratio $e^{-\theta X+\Gamma(\theta)}$.

---

## 7. Exponential tilting for independent sums

Let $X_1,\dots,X_n$ be independent and identically distributed random variables with common law $\mu$.

Define:

```math
S_n
=
X_1+\cdots+X_n.
```

Under the tilted probability measure, each $X_i$ has tilted law $\mu_\theta$.

The likelihood ratio is:

```math
\frac{dP_\theta}{dP}(X_1,\dots,X_n)
=
\prod_{i=1}^n
e^{\theta X_i-\Gamma(\theta)}.
```

Therefore:

```math
\frac{dP_\theta}{dP}(X_1,\dots,X_n)
=
e^{\theta(X_1+\cdots+X_n)-n\Gamma(\theta)}.
```

Since $S_n=X_1+\cdots+X_n$, this becomes:

```math
\frac{dP_\theta}{dP}(X_1,\dots,X_n)
=
e^{\theta S_n-n\Gamma(\theta)}.
```

The inverse likelihood ratio is:

```math
\frac{dP}{dP_\theta}(X_1,\dots,X_n)
=
e^{-\theta S_n+n\Gamma(\theta)}.
```

Hence, for a rare event of the form:

```math
\left\{
\frac{S_n}{n} \ge x
\right\},
```


we obtain:

```math
\mathbb{P}
\left(
\frac{S_n}{n} \ge x
\right)
=
\mathbb{E}_{\theta}
\left[
1_{\{S_n/n \ge x\}}
e^{-\theta S_n+n\Gamma(\theta)}
\right].
```

This is the importance sampling estimator associated with exponential tilting.

---

## 8. Mean under the tilted law

The cumulant generating function contains information about the tilted distribution.

For $\theta$ in the interior of $D(\Gamma)$:

```math
\Gamma'(\theta)
=
\mathbb{E}_{\theta}[X].
```

This means that the derivative of the CGF gives the mean of $X$ under the tilted law $\mu_\theta$.

So exponential tilting does not only reweight the distribution. It moves its mean.

This is why exponential tilting is useful for rare events.

If the rare event is:

```math
\frac{S_n}{n} \ge x,
```

one typically chooses $\theta$ such that:

```math
\Gamma'(\theta)=x.
```

Then, under $P_\theta$, the empirical mean $S_n/n$ is centered around $x$.

The rare event becomes typical under the tilted measure.

---

## 9. Connection with Cramer's theorem

Cramer's theorem studies the asymptotic probability of rare deviations of empirical means.

The rate function is the Fenchel-Legendre transform of the cumulant generating function:

```math
\Gamma^*(x)
=
\sup_{\theta \in \mathbb{R}}
\left\{
\theta x-\Gamma(\theta)
\right\}.
```

For large $n$, Cramer's theorem gives the logarithmic approximation:

```math
\mathbb{P}
\left(
\frac{S_n}{n} \ge x
\right)
\approx
e^{-n\Gamma^*(x)}.
```

The optimal exponential tilt is linked to the optimizer in:

```math
\Gamma^*(x)
=
\theta x-\Gamma(\theta).
```

When the supremum is attained at $\theta=\theta_x$, the same $\theta_x$ is used to define the tilted law that makes the rare event less rare.

Thus, exponential tilting plays two roles:

1. It is a computational tool for importance sampling.
2. It is a theoretical tool in the proof and interpretation of large deviation results.

---

## 10. Interpretation

Exponential tilting transforms the original law by giving more weight to outcomes that are important for the rare event.

For $\theta>0$, large values of $X$ are amplified:

```math
e^{\theta x}
\quad \text{increases with } x.
```

For $\theta<0$, small values of $X$ are amplified.

The normalizing term $\Gamma(\theta)$ ensures that the tilted law remains a probability distribution.

The likelihood ratio ensures that the estimator remains unbiased.

A useful summary is:

```math
\mu_\theta(dx)
=
e^{\theta x-\Gamma(\theta)}\mu(dx),
```

and:

```math
\frac{d\mu}{d\mu_\theta}(x)
=
e^{-\theta x+\Gamma(\theta)}.
```

So exponential tilting can be summarized as:

> Change the sampling distribution to make the rare event more frequent, then correct the bias with the inverse likelihood ratio.

---

## 11. Code correspondence

In the codebase, each distribution-specific object should expose the following large-deviation ingredients:

| Mathematical object | Meaning | Code role |
|---|---|---|
| $\Gamma(\theta)$ | cumulant generating function | `cgf(theta)` |
| $D(\Gamma)$ | admissible domain | `domain_contains(theta)` |
| $\mu_\theta$ | tilted law | `tilted_parameter(theta)` |
| $\mathbb{E}_\theta[X]$ | mean under tilted law | `mean_under_tilt(theta)` |
| $\Gamma^*(x)$ | rate function | `rate_function(x)` |

The Bernoulli implementation is the first concrete example of this structure.


---

## Next step

The present note introduced the exponential change of measure, the tilted law
$\mu_\theta$, and the likelihood ratio used in importance sampling.

The next note explains how this change of measure is used to prove Cramer's theorem and
why the rate function

$$
\Gamma^*(x)
=
\sup_{\theta\in\mathbb R}
\{\theta x-\Gamma(\theta)\}
$$

governs the logarithmic decay of rare-event probabilities.

Continue with:

```text
docs/foundations/cramers_theorem_from_scratch.md
```
---

## Reference

H. Pham, *Large Deviations in Mathematical Finance*, 2010.

Relevant sections:

- Introduction: rare-event simulation and importance sampling.
- Section 2.1: Laplace transform and change of probability measures.
- Section 2.2: Cramer's theorem and exponential tilting.
