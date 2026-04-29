# Bernoulli Distribution

## 1. Setup

Let

```math
X \sim \mathrm{Bernoulli}(p),
\qquad 0 < p < 1.
```

This means:

```math
\mathbb{P}(X=1)=p,
\qquad
\mathbb{P}(X=0)=1-p.
```

The Bernoulli distribution is the simplest non-trivial example in large deviations.

If $X_1,\dots,X_n$ are i.i.d. Bernoulli random variables, then the empirical mean

```math
\bar X_n
=
\frac{1}{n}
\sum_{i=1}^n X_i
```

is the empirical frequency of successes.

By the law of large numbers:

```math
\bar X_n \to p.
```

Large deviations study the probability that $\bar X_n$ is far from $p$, for example:

```math
\mathbb{P}(\bar X_n \ge x),
\qquad x > p.
```

---

## 2. Cumulant generating function

The cumulant generating function is defined by:

```math
\Gamma(\theta)
=
\log \mathbb{E}
\left[
e^{\theta X}
\right].
```

For a Bernoulli random variable:

```math
\mathbb{E}
\left[
e^{\theta X}
\right]
=
e^{\theta \cdot 0}\mathbb{P}(X=0)
+
e^{\theta \cdot 1}\mathbb{P}(X=1).
```

Therefore:

```math
\mathbb{E}
\left[
e^{\theta X}
\right]
=
(1-p)+p e^\theta.
```

So the cumulant generating function is:

```math
\Gamma(\theta)
=
\log(1-p+p e^\theta).
```

Since this expression is finite for every real $\theta$, the effective domain is:

```math
D(\Gamma)=\mathbb{R}.
```

---

## 3. Exponential tilting

The exponentially tilted law is defined by:

```math
\mu_\theta(x)
=
e^{\theta x-\Gamma(\theta)}
\mu(x).
```

For Bernoulli, the support is only:

```math
x \in \{0,1\}.
```

So we only need to compute the tilted probabilities of $0$ and $1$.

---

## 4. Tilted probability of success

We first compute the probability of observing $1$ under the tilted law.

By definition:

```math
\mathbb{P}_\theta(X=1)
=
e^{\theta \cdot 1-\Gamma(\theta)}
\mathbb{P}(X=1).
```

Since $\mathbb{P}(X=1)=p$, we get:

```math
\mathbb{P}_\theta(X=1)
=
e^{\theta-\Gamma(\theta)}p.
```

Now:

```math
e^{\Gamma(\theta)}
=
1-p+p e^\theta.
```

Therefore:

```math
e^{-\Gamma(\theta)}
=
\frac{1}{1-p+p e^\theta}.
```

Hence:

```math
\mathbb{P}_\theta(X=1)
=
\frac{p e^\theta}{1-p+p e^\theta}.
```

We denote this tilted success probability by:

```math
p_\theta
=
\frac{p e^\theta}{1-p+p e^\theta}.
```

Thus, under exponential tilting:

```math
X \sim \mathrm{Bernoulli}(p)
\quad
\Longrightarrow
\quad
X \sim \mathrm{Bernoulli}(p_\theta)
\text{ under } \mathbb{P}_\theta.
```

---

## 5. Tilted probability of failure

Similarly:

```math
\mathbb{P}_\theta(X=0)
=
e^{\theta \cdot 0-\Gamma(\theta)}
\mathbb{P}(X=0).
```

Since $\mathbb{P}(X=0)=1-p$, this gives:

```math
\mathbb{P}_\theta(X=0)
=
e^{-\Gamma(\theta)}(1-p).
```

Therefore:

```math
\mathbb{P}_\theta(X=0)
=
\frac{1-p}{1-p+p e^\theta}.
```

The two probabilities sum to one:

```math
\frac{1-p}{1-p+p e^\theta}
+
\frac{p e^\theta}{1-p+p e^\theta}
=
1.
```

So the tilted distribution is indeed a Bernoulli distribution.

---

## 6. Mean under the tilted law

For a Bernoulli random variable, the mean is equal to the success probability.

Therefore:

```math
\mathbb{E}_\theta[X]
=
p_\theta.
```

So:

```math
\mathbb{E}_\theta[X]
=
\frac{p e^\theta}{1-p+p e^\theta}.
```

This is also obtained by differentiating the cumulant generating function.

Since:

```math
\Gamma(\theta)
=
\log(1-p+p e^\theta),
```

we have:

```math
\Gamma'(\theta)
=
\frac{p e^\theta}{1-p+p e^\theta}.
```

Hence:

```math
\Gamma'(\theta)
=
\mathbb{E}_\theta[X].
```

This identity is fundamental in exponential tilting.

It means that the parameter $\theta$ moves the mean of the distribution.

---

## 7. Interpretation of the tilting parameter

The tilted success probability is:

```math
p_\theta
=
\frac{p e^\theta}{1-p+p e^\theta}.
```

If $\theta=0$, then:

```math
p_0
=
\frac{p}{1-p+p}
=
p.
```

So there is no change of measure.

If $\theta>0$, then $e^\theta>1$, and the tilted probability satisfies:

```math
p_\theta > p.
```

So successes become more frequent.

If $\theta<0$, then $e^\theta<1$, and:

```math
p_\theta < p.
```

So successes become less frequent.

This is exactly why exponential tilting is useful for rare events.

If the rare event is "many successes", then we choose $\theta>0$.

If the rare event is "very few successes", then we choose $\theta<0$.

---

## 8. Saddle-point equation

Suppose we want to make the empirical mean typical around some value $x$.

For Bernoulli variables, this means we want the tilted mean to satisfy:

```math
\mathbb{E}_\theta[X]=x.
```

Using the identity $\Gamma'(\theta)=\mathbb{E}_\theta[X]$, we need:

```math
\Gamma'(\theta)=x.
```

For Bernoulli:

```math
x
=
\frac{p e^\theta}{1-p+p e^\theta}.
```

We solve for $\theta$.

First:

```math
x(1-p+p e^\theta)
=
p e^\theta.
```

So:

```math
x(1-p)+xp e^\theta
=
p e^\theta.
```

Move the terms involving $e^\theta$ to the same side:

```math
x(1-p)
=
p e^\theta(1-x).
```

Therefore:

```math
e^\theta
=
\frac{x(1-p)}{p(1-x)}.
```

So the saddle-point parameter is:

```math
\theta^*(x)
=
\log
\left(
\frac{x(1-p)}{p(1-x)}
\right),
\qquad x \in (0,1).
```

This is the value of $\theta$ that makes $x$ the typical mean under the tilted law.

---

## 9. Rate function

Cramer's theorem says that the probability of observing an atypical empirical mean decays exponentially.

For Bernoulli random variables:

```math
\mathbb{P}(\bar X_n \approx x)
\approx
e^{-n\Gamma^*(x)}.
```

The rate function is the Fenchel-Legendre transform of the cumulant generating function:

```math
\Gamma^*(x)
=
\sup_{\theta \in \mathbb{R}}
\left\{
\theta x-\Gamma(\theta)
\right\}.
```

For Bernoulli:

```math
\Gamma(\theta)
=
\log(1-p+p e^\theta).
```

So:

```math
\Gamma^*(x)
=
\sup_{\theta \in \mathbb{R}}
\left\{
\theta x-\log(1-p+p e^\theta)
\right\}.
```

The optimizer satisfies:

```math
x=\Gamma'(\theta).
```

Using the saddle-point solution:

```math
\theta^*(x)
=
\log
\left(
\frac{x(1-p)}{p(1-x)}
\right),
```

we obtain the closed-form rate function:

```math
\Gamma^*(x)
=
x\log\left(\frac{x}{p}\right)
+
(1-x)\log\left(\frac{1-x}{1-p}\right),
\qquad x \in [0,1].
```

Outside $[0,1]$, the rate function is infinite:

```math
\Gamma^*(x)=+\infty,
\qquad x \notin [0,1].
```

This is because an empirical mean of Bernoulli random variables cannot be smaller than $0$ or larger than $1$.

---

## 10. Boundary convention

At the boundary $x=0$ or $x=1$, the formula contains terms of the form:

```math
0\log(0).
```

We use the standard convention:

```math
0\log(0)=0.
```

More precisely:

```math
\lim_{x \downarrow 0}
x\log(x)
=
0.
```

So:

```math
\Gamma^*(0)
=
\log\left(\frac{1}{1-p}\right),
```

and:

```math
\Gamma^*(1)
=
\log\left(\frac{1}{p}\right).
```

This convention is implemented in code through a safe version of $x\log(x/y)$.

---

## 11. Key properties of the Bernoulli rate function

The Bernoulli rate function is:

```math
\Gamma^*(x)
=
x\log\left(\frac{x}{p}\right)
+
(1-x)\log\left(\frac{1-x}{1-p}\right).
```

It satisfies:

```math
\Gamma^*(x)\ge 0.
```

It is equal to zero at the mean:

```math
\Gamma^*(p)=0.
```

It is positive away from the mean:

```math
x\neq p
\quad \Longrightarrow \quad
\Gamma^*(x)>0.
```

It is infinite outside the support:

```math
x \notin [0,1]
\quad \Longrightarrow \quad
\Gamma^*(x)=+\infty.
```

Interpretation:

> $\Gamma^*(x)$ measures the exponential cost of forcing a Bernoulli($p$) sample to behave as if its empirical success frequency were $x$.

---

## 12. Importance sampling interpretation

Suppose we want to estimate:

```math
\mathbb{P}
\left(
\bar X_n \ge x
\right),
\qquad x>p.
```

Under the original Bernoulli($p$) law, this event is rare.

We choose $\theta>0$ so that:

```math
\Gamma'(\theta)=x.
```

Then under the tilted law:

```math
X_i \sim \mathrm{Bernoulli}(x).
```

The event $\bar X_n \ge x$ is no longer rare under the tilted measure.

The likelihood ratio for the full sample is:

```math
\frac{dP}{dP_\theta}(X_1,\dots,X_n)
=
e^{-\theta S_n+n\Gamma(\theta)},
```

where:

```math
S_n=X_1+\cdots+X_n.
```

Therefore:

```math
\mathbb{P}
\left(
\bar X_n \ge x
\right)
=
\mathbb{E}_\theta
\left[
1_{\{\bar X_n \ge x\}}
e^{-\theta S_n+n\Gamma(\theta)}
\right].
```


This is the Bernoulli version of exponential-tilting importance sampling.

---

## 13. Numerical example

Let:

```math
p=0.2.
```

The original distribution has success probability $20$%$.

If we choose $\theta=0$, then:

```math
p_\theta=p=0.2.
```

If $\theta>0$, the tilted success probability increases.

For example, with $\theta=1$:

```math
p_\theta
=
\frac{0.2e^1}{0.8+0.2e^1}.
```

This is greater than $0.2$, so successes become more likely.

This is useful when estimating a rare event such as:

```math
\bar X_n \ge 0.6.
```

Instead of waiting a very long time to observe many successes under Bernoulli($0.2$), we simulate from a tilted Bernoulli law where successes are more frequent, and then correct using the likelihood ratio.

---

## 14. Code correspondence

The Bernoulli large-deviation object exposes the main mathematical quantities.

| Mathematical object | Formula | Code role |
|---|---|---|
| Cumulant generating function | $\Gamma(\theta)=\log(1-p+p e^\theta)$ | `cgf(theta)` |
| Effective domain | $D(\Gamma)=\mathbb{R}$ | `domain_contains(theta)` |
| Tilted parameter | $p_\theta=\frac{p e^\theta}{1-p+p e^\theta}$ | `tilted_parameter(theta)` |
| Mean under tilt | $\mathbb{E}_\theta[X]=p_\theta$ | `mean_under_tilt(theta)` |
| Rate function | $\Gamma^*(x)=x\log(x/p)+(1-x)\log((1-x)/(1-p))$ | `rate_function(x)` |

This is the first concrete distribution implemented in the project.

The same pattern will later be reused for:

- Poisson distribution,
- Gaussian distribution,
- Exponential distribution.

---

## 15. Summary

For:

```math
X \sim \mathrm{Bernoulli}(p),
```

the cumulant generating function is:

```math
\Gamma(\theta)
=
\log(1-p+p e^\theta).
```

The exponentially tilted distribution remains Bernoulli:

```math
X \sim \mathrm{Bernoulli}(p_\theta)
\quad \text{under } \mathbb{P}_\theta,
```

where:

```math
p_\theta
=
\frac{p e^\theta}{1-p+p e^\theta}.
```

The mean under the tilted law is:

```math
\mathbb{E}_\theta[X]
=
p_\theta
=
\Gamma'(\theta).
```

The rate function is:

```math
\Gamma^*(x)
=
x\log\left(\frac{x}{p}\right)
+
(1-x)\log\left(\frac{1-x}{1-p}\right),
\qquad x \in [0,1].
```

And:

```math
\Gamma^*(x)=+\infty,
\qquad x \notin [0,1].
```

The key intuition is:

> Exponential tilting changes the Bernoulli success probability from $p$ to $p_\theta$, making rare success frequencies more typical while preserving unbiased estimation through the likelihood ratio.

---

## Reference

H. Pham, *Large Deviations in Mathematical Finance*, 2010.

Relevant sections:

- Section 2.1: Laplace transform and change of probability measures.
- Section 2.2: Cramer's theorem.
