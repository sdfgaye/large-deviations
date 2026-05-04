# Cramer's Theorem from Scratch

This note explains the mathematical logic behind Cramer's theorem for empirical means, with a focus on the exponential change of measure used in large deviations and importance sampling.

It is meant to be read after the note on exponential tilting:

```text
docs/foundations/exponential_tilting.md
```

The goal is to understand why the following approximation is natural:

```math
\mathbb{P}\left(\frac{S_n}{n} \ge x\right)
\approx
\exp\left(-n\Gamma^*(x)\right),
```

where

```math
S_n = X_1 + \cdots + X_n,
```

and

```math
\Gamma^*(x)
=
\sup_{\theta\in\mathbb{R}}\{\theta x-\Gamma(\theta)\}.
```


## Position in the project

This note is the theoretical core of **Module 1 — Cramer's theorem**.

It should be read after:

```text
docs/foundations/exponential_tilting.md
```

The previous note introduces the exponential change of measure, the tilted law
\(\mu_\theta\), the likelihood ratio, and the identity

\[
\Gamma'(\theta)=\mathbb E_\theta[X].
\]

This note explains how those objects are used to prove the logarithmic rare-event
asymptotic

\[
\mathbb P\left(\frac{S_n}{n}\ge x\right)
\approx
\exp(-n\Gamma^*(x)).
\]

It prepares the next pieces of the project:

```text
docs/distributions/bernoulli.md
notebooks/01_bernoulli_exponential_tilting.ipynb
src/large_deviations/importance_sampling.py
```

The role of this note is to connect the theory to the project workflow:

```text
Derive -> Code -> Apply
```

In particular, it explains:

1. why the cumulant generating function \(\Gamma\) appears naturally;
2. why the Fenchel-Legendre transform \(\Gamma^*\) is the exponential cost;
3. why exponential tilting makes a rare empirical mean typical;
4. why the same saddle-point \(\theta\) appears in Cramer's theorem and in importance sampling.

After this note, the Bernoulli case study becomes a concrete implementation of the general theory.

---

## Scope of this note

This note focuses on the clean one-dimensional i.i.d. setting.

We assume that the cumulant generating function is finite on a suitable domain and that,
for the target value \(x\), the saddle-point equation

\[
\Gamma'(\theta)=x
\]

admits an appropriate solution.

This is the right level of generality for the first part of the project, especially for the
Bernoulli, Poisson, Gaussian, and exponential examples.

The goal is not to state Cramer's theorem in its most technical form. The goal is to
understand the mechanism behind the theorem:

```text
exponential moments
-> exponential tilting
-> likelihood ratio
-> rate function
-> importance sampling
```

---

## 1. Setup

Let

```math
X_1,X_2,\dots,X_n
```

be independent and identically distributed real-valued random variables with common law $\mu$.

Define the partial sum:

```math
S_n = X_1 + \cdots + X_n.
```

The empirical mean is:

```math
\frac{S_n}{n}.
```

Let

```math
\bar{x}=\mathbb{E}[X_1].
```

By the law of large numbers,

```math
\frac{S_n}{n}\to \bar{x}
```

in probability as $n\to\infty$.

Therefore, if

```math
x > \bar{x},
```

then the event

```math
\left\{\frac{S_n}{n}\ge x\right\}
```

is rare for large $n$.

Cramer's theorem describes how fast this rare-event probability goes to zero.

---

## 2. The cumulant generating function

The cumulant generating function of $X_1$ is defined by:

```math
\Gamma(\theta)
=
\log\mathbb{E}\left[e^{\theta X_1}\right].
```

Equivalently, if $X_1$ has law $\mu$, then:

```math
\Gamma(\theta)
=
\log\int e^{\theta y}\mu(dy).
```

The effective domain of $\Gamma$ is:

```math
D(\Gamma)
=
\{\theta\in\mathbb{R}:\Gamma(\theta)<\infty\}.
```

The reason $\Gamma$ appears naturally is that exponentials turn sums into products.

Indeed,

```math
S_n=X_1+\cdots+X_n,
```

so

```math
e^{\theta S_n}
=
e^{\theta X_1}\cdots e^{\theta X_n}.
```

Using independence:

```math
\mathbb{E}\left[e^{\theta S_n}\right]
=
\prod_{i=1}^n\mathbb{E}\left[e^{\theta X_i}\right]
=
\left(\mathbb{E}\left[e^{\theta X_1}\right]\right)^n.
```

Since

```math
\mathbb{E}\left[e^{\theta X_1}\right]
=
e^{\Gamma(\theta)},
```

we get:

```math
\mathbb{E}\left[e^{\theta S_n}\right]
=
e^{n\Gamma(\theta)}.
```

This identity is the first major reason why $\Gamma$ is the right object for sums of independent random variables.

---

## 3. Upper bound: where $\Gamma^*(x)$ first appears

We want to bound:

```math
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
=
\mathbb{P}(S_n\ge nx).
```

Take $\theta\ge 0$.

On the event $S_n\ge nx$, we have:

```math
\theta S_n\ge \theta nx.
```

Because the exponential function is increasing:

```math
e^{\theta S_n}\ge e^{\theta nx}.
```

Therefore, on this event:

```math
1_{\{S_n\ge nx\}}
\le
e^{\theta S_n-\theta nx}.
```

Taking expectations gives:

```math
\mathbb{P}(S_n\ge nx)
=
\mathbb{E}\left[1_{\{S_n\ge nx\}}\right]
\le
\mathbb{E}\left[e^{\theta S_n-\theta nx}\right].
```

Since $e^{-\theta nx}$ is deterministic:

```math
\mathbb{P}(S_n\ge nx)
\le
e^{-\theta nx}\mathbb{E}\left[e^{\theta S_n}\right].
```

Using the identity from the previous section:

```math
\mathbb{E}\left[e^{\theta S_n}\right]
=
e^{n\Gamma(\theta)},
```

we obtain:

```math
\mathbb{P}(S_n\ge nx)
\le
e^{-\theta nx}e^{n\Gamma(\theta)}.
```

So:

```math
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
\le
\exp\left(-n(\theta x-\Gamma(\theta))\right).
```

This is true for every admissible $\theta\ge 0$.

Therefore, to get the strongest possible upper bound, we maximize the quantity

```math
\theta x-\Gamma(\theta).
```

This leads to the definition:

```math
\Gamma^*(x)
=
\sup_{\theta\in\mathbb{R}}
\{\theta x-\Gamma(\theta)\}.
```

So $\Gamma^*(x)$ does not appear by magic.

It appears because we test all exponential bounds and keep the best one.

At this stage, we have proved only the upper bound:

```math
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
\le
\exp\left(-n\Gamma^*(x)\right).
```

Cramer's theorem says that this upper bound is asymptotically exact on the logarithmic scale.

---

## 4. Exponential tilting

To prove the lower bound, we introduce a new probability law.

For $\theta\in D(\Gamma)$, define:

```math
\mu_\theta(dy)
=
e^{\theta y-\Gamma(\theta)}\mu(dy).
```

This is called the exponentially tilted law.

It is a probability measure because:

```math
\int \mu_\theta(dy)
=
\int e^{\theta y-\Gamma(\theta)}\mu(dy)
```

and since $e^{-\Gamma(\theta)}$ is constant in $y$,

```math
\int \mu_\theta(dy)
=
e^{-\Gamma(\theta)}\int e^{\theta y}\mu(dy).
```

By definition of $\Gamma$,

```math
\int e^{\theta y}\mu(dy)
=
e^{\Gamma(\theta)}.
```

Therefore:

```math
\int \mu_\theta(dy)
=
e^{-\Gamma(\theta)}e^{\Gamma(\theta)}
=
1.
```

So $\mu_\theta$ is a valid probability distribution.

Intuitively:

- if $\theta>0$, the tilted law gives more weight to large values of $X$;
- if $\theta<0$, it gives more weight to small values of $X$;
- if $\theta=0$, the law is unchanged.

---

## 5. The tilted probability measure for $n$ variables

For one variable,

```math
\frac{d\mu_\theta}{d\mu}(X_i)
=
e^{\theta X_i-\Gamma(\theta)}.
```

For $n$ independent variables, the likelihood ratio is the product:

```math
\frac{dP_\theta}{dP}(X_1,\dots,X_n)
=
\prod_{i=1}^n e^{\theta X_i-\Gamma(\theta)}.
```

Expanding the product:

```math
\frac{dP_\theta}{dP}(X_1,\dots,X_n)
=
e^{\theta(X_1+\cdots+X_n)-n\Gamma(\theta)}.
```

Since $S_n=X_1+\cdots+X_n$,

```math
\frac{dP_\theta}{dP}
=
e^{\theta S_n-n\Gamma(\theta)}.
```

The inverse likelihood ratio is:

```math
\frac{dP}{dP_\theta}
=
e^{-\theta S_n+n\Gamma(\theta)}.
```

This inverse ratio is the correction factor used when we simulate under $P_\theta$ but want a probability under the original law $P$.

For any event $A$ depending on $X_1,\dots,X_n$:

```math
P(A)
=
\mathbb{E}_\theta
\left[
1_A\frac{dP}{dP_\theta}
\right].
```

Therefore:

```math
P(A)
=
\mathbb{E}_\theta
\left[
1_A e^{-\theta S_n+n\Gamma(\theta)}
\right].
```

---

## 6. Mean under the tilted law

The derivative of $\Gamma$ gives the mean under the tilted law.

Start from:

```math
\Gamma(\theta)
=
\log\mathbb{E}\left[e^{\theta X}\right].
```

Define:

```math
M(\theta)
=
\mathbb{E}\left[e^{\theta X}\right].
```

Then:

```math
\Gamma(\theta)=\log M(\theta).
```

Differentiating:

```math
\Gamma'(\theta)
=
\frac{M'(\theta)}{M(\theta)}.
```

Now:

```math
M'(\theta)
=
\frac{d}{d\theta}\mathbb{E}\left[e^{\theta X}\right]
=
\mathbb{E}\left[Xe^{\theta X}\right].
```

Therefore:

```math
\Gamma'(\theta)
=
\frac{\mathbb{E}\left[Xe^{\theta X}\right]}
{\mathbb{E}\left[e^{\theta X}\right]}.
```

Since

```math
\mathbb{E}\left[e^{\theta X}\right]
=
e^{\Gamma(\theta)},
```

we get:

```math
\Gamma'(\theta)
=
\mathbb{E}\left[Xe^{\theta X-\Gamma(\theta)}\right].
```

But $e^{\theta x-\Gamma(\theta)}\mu(dx)$ is exactly $\mu_\theta(dx)$.

Hence:

```math
\Gamma'(\theta)
=
\int x\mu_\theta(dx)
=
\mathbb{E}_\theta[X].
```

So:

```math
\Gamma'(\theta)=\mathbb{E}_\theta[X].
```

This identity is crucial.

It means that $\theta$ controls the mean of the tilted distribution.

---

## 7. The saddle-point equation

The rate function is defined by:

```math
\Gamma^*(x)
=
\sup_\theta\{\theta x-\Gamma(\theta)\}.
```

To find the optimizer, define:

```math
F(\theta)=\theta x-\Gamma(\theta).
```

If $F$ is differentiable and the maximum is attained at an interior point, then:

```math
F'(\theta)=0.
```

Since:

```math
F'(\theta)=x-\Gamma'(\theta),
```

we obtain the saddle-point equation:

```math
\Gamma'(\theta)=x.
```

Using the previous section:

```math
\Gamma'(\theta)=\mathbb{E}_\theta[X].
```

Therefore:

```math
\mathbb{E}_\theta[X]=x.
```

This has a simple interpretation:

> the optimal tilt is the one that makes the rare target value $x$ become the typical mean under the tilted law.

---

## 8. Statement of Cramer's theorem

Under suitable regularity assumptions, for $x\ge \mathbb{E}[X_1]$, Cramer's theorem states:

```math
\lim_{n\to\infty}
\frac{1}{n}
\log
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
=
-\Gamma^*(x).
```

Equivalently, at logarithmic scale:

```math
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
\approx
\exp\left(-n\Gamma^*(x)\right).
```

This does not mean that the ratio of the two sides converges to $1$.

It means:

```math
\frac{1}{n}\log
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
\to
-\Gamma^*(x).
```

So $\Gamma^*(x)$ is the exponential decay rate of the rare-event probability.

---

## 9. Lower bound: why the upper bound is sharp

The upper bound showed:

```math
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
\le
\exp(-n\Gamma^*(x)).
```

To prove Cramer's theorem, we also need a lower bound showing that the probability is not exponentially smaller than this.

The idea is to focus on the smaller event:

```math
A_{n,\varepsilon}
=
\left\{
\frac{S_n}{n}\in[x,x+\varepsilon)
\right\}.
```

Since

```math
A_{n,\varepsilon}
\subseteq
\left\{
\frac{S_n}{n}\ge x
\right\},
```

we have:

```math
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
\ge
\mathbb{P}(A_{n,\varepsilon}).
```

So it is enough to lower bound $\mathbb{P}(A_{n,\varepsilon})$.

---

## 10. Rewriting the probability under the tilted measure

Using the inverse likelihood ratio:

```math
\frac{dP}{dP_\theta}
=
e^{-\theta S_n+n\Gamma(\theta)},
```

we write:

```math
\mathbb{P}(A_{n,\varepsilon})
=
\mathbb{E}_\theta
\left[
1_{A_{n,\varepsilon}}
e^{-\theta S_n+n\Gamma(\theta)}
\right].
```

This is the key change-of-measure identity.

The factor

```math
e^{-\theta S_n+n\Gamma(\theta)}
```

is the correction factor for going back from the tilted law $P_\theta$ to the original law $P$.

---

## 11. Simplifying the correction factor on $A_{n,\varepsilon}$

We now rewrite the exponent:

```math
-\theta S_n+n\Gamma(\theta).
```

Since

```math
S_n=n\frac{S_n}{n},
```

we have:

```math
-\theta S_n+n\Gamma(\theta)
=
-n\theta\frac{S_n}{n}+n\Gamma(\theta).
```

Factor out $n$:

```math
-\theta S_n+n\Gamma(\theta)
=
n\left(-\theta\frac{S_n}{n}+\Gamma(\theta)\right).
```

Now add and subtract $\theta x$ inside the parentheses:

```math
-\theta\frac{S_n}{n}+\Gamma(\theta)
=
-\theta x+\Gamma(\theta)
-\theta\left(\frac{S_n}{n}-x\right).
```

Therefore:

```math
-\theta S_n+n\Gamma(\theta)
=
-n(\theta x-\Gamma(\theta))
-n\theta\left(\frac{S_n}{n}-x\right).
```

So:

```math
e^{-\theta S_n+n\Gamma(\theta)}
=
e^{-n(\theta x-\Gamma(\theta))}
e^{-n\theta(S_n/n-x)}.
```

On $A_{n,\varepsilon}$, we know:

```math
x\le \frac{S_n}{n}<x+\varepsilon.
```

Thus:

```math
0\le \frac{S_n}{n}-x<\varepsilon.
```

If $\theta>0$, then:

```math
0\le \theta\left(\frac{S_n}{n}-x\right)<\theta\varepsilon.
```

Therefore:

```math
-n\theta\left(\frac{S_n}{n}-x\right)
\ge
-n\theta\varepsilon.
```

So on $A_{n,\varepsilon}$:

```math
e^{-n\theta(S_n/n-x)}
\ge
e^{-n\theta\varepsilon}.
```

Hence:

```math
e^{-\theta S_n+n\Gamma(\theta)}
\ge
e^{-n(\theta x-\Gamma(\theta))}e^{-n\theta\varepsilon}.
```

More generally, one can write $|\theta|\varepsilon$ to cover both signs of $\theta$.

---

## 12. Lower bounding the expectation

Recall:

```math
\mathbb{P}(A_{n,\varepsilon})
=
\mathbb{E}_\theta
\left[
1_{A_{n,\varepsilon}}
e^{-\theta S_n+n\Gamma(\theta)}
\right].
```

On $A_{n,\varepsilon}$, we have the bound:

```math
e^{-\theta S_n+n\Gamma(\theta)}
\ge
e^{-n(\theta x-\Gamma(\theta))}e^{-n|\theta|\varepsilon}.
```

Therefore:

```math
\mathbb{P}(A_{n,\varepsilon})
\ge
\mathbb{E}_\theta
\left[
1_{A_{n,\varepsilon}}
e^{-n(\theta x-\Gamma(\theta))}e^{-n|\theta|\varepsilon}
\right].
```

The exponential terms are deterministic, so they can be taken outside the expectation:

```math
\mathbb{P}(A_{n,\varepsilon})
\ge
e^{-n(\theta x-\Gamma(\theta))}e^{-n|\theta|\varepsilon}
\mathbb{E}_\theta[1_{A_{n,\varepsilon}}].
```

But

```math
\mathbb{E}_\theta[1_{A_{n,\varepsilon}}]
=
P_\theta(A_{n,\varepsilon}).
```

Thus:

```math
\mathbb{P}(A_{n,\varepsilon})
\ge
e^{-n(\theta x-\Gamma(\theta))}e^{-n|\theta|\varepsilon}
P_\theta(A_{n,\varepsilon}).
```

This is the central inequality in the lower bound.

---

## 13. Taking logarithms

Start from:

```math
\mathbb{P}(A_{n,\varepsilon})
\ge
e^{-n(\theta x-\Gamma(\theta))}e^{-n|\theta|\varepsilon}
P_\theta(A_{n,\varepsilon}).
```

Take logarithms:

```math
\log\mathbb{P}(A_{n,\varepsilon})
\ge
-n(\theta x-\Gamma(\theta))
-n|\theta|\varepsilon
+
\log P_\theta(A_{n,\varepsilon}).
```

Divide by $n$:

```math
\frac{1}{n}\log\mathbb{P}(A_{n,\varepsilon})
\ge
-(\theta x-\Gamma(\theta))
-|\theta|\varepsilon
+
\frac{1}{n}\log P_\theta(A_{n,\varepsilon}).
```

---

## 14. Why the last term vanishes

Now choose $\theta$ such that:

```math
\Gamma'(\theta)=x.
```

Since

```math
\Gamma'(\theta)=\mathbb{E}_\theta[X],
```

this means:

```math
\mathbb{E}_\theta[X]=x.
```

Under $P_\theta$, the variables $X_1,\dots,X_n$ are still i.i.d., but their common mean is now $x$.

By the law of large numbers under $P_\theta$,

```math
\frac{S_n}{n}\to x.
```

So the event

```math
A_{n,\varepsilon}
=
\left\{
\frac{S_n}{n}\in[x,x+\varepsilon)
\right\}
```

is no longer exponentially rare under $P_\theta$.

In regular cases, a central limit theorem gives:

```math
P_\theta(A_{n,\varepsilon})\to \frac{1}{2}.
```

The exact constant is not the important point.

The important point is that $P_\theta(A_{n,\varepsilon})$ does not decay like $e^{-cn}$.

Therefore:

```math
\frac{1}{n}\log P_\theta(A_{n,\varepsilon})
\to 0.
```

This is why the last term disappears at the logarithmic scale.

---

## 15. Completing the lower bound

We had:

```math
\frac{1}{n}\log\mathbb{P}(A_{n,\varepsilon})
\ge
-(\theta x-\Gamma(\theta))
-|\theta|\varepsilon
+
\frac{1}{n}\log P_\theta(A_{n,\varepsilon}).
```

Taking $n\to\infty$ gives:

```math
\liminf_{n\to\infty}
\frac{1}{n}\log\mathbb{P}(A_{n,\varepsilon})
\ge
-(\theta x-\Gamma(\theta))
-|\theta|\varepsilon.
```

Then send $\varepsilon\to 0$:

```math
\lim_{\varepsilon\to 0}
\liminf_{n\to\infty}
\frac{1}{n}\log\mathbb{P}(A_{n,\varepsilon})
\ge
-(\theta x-\Gamma(\theta)).
```

If $\theta$ is the optimizer in the Fenchel-Legendre transform, then:

```math
\Gamma^*(x)=\theta x-\Gamma(\theta).
```

Therefore:

```math
\liminf_{n\to\infty}
\frac{1}{n}
\log
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
\ge
-\Gamma^*(x).
```

Together with the upper bound, this gives:

```math
\lim_{n\to\infty}
\frac{1}{n}
\log
\mathbb{P}\left(\frac{S_n}{n}\ge x\right)
=
-\Gamma^*(x).
```

This is Cramer's theorem.

---

## 16. Why the same $\theta$ appears in importance sampling

The probability of interest is:

```math
p_n
=
\mathbb{P}\left(\frac{S_n}{n}\ge x\right).
```

Using exponential tilting:

```math
p_n
=
\mathbb{E}_\theta
\left[
1_{\{S_n/n\ge x\}}
e^{-\theta S_n+n\Gamma(\theta)}
\right].
```

This gives an unbiased importance sampling estimator:

```math
Z_\theta
=
1_{\{S_n/n\ge x\}}
e^{-\theta S_n+n\Gamma(\theta)}.
```

To make simulation efficient, we want the event

```math
\left\{\frac{S_n}{n}\ge x\right\}
```

to be common under $P_\theta$.

The natural choice is to make the tilted mean equal to $x$:

```math
\mathbb{E}_\theta[X]=x.
```

Since

```math
\mathbb{E}_\theta[X]=\Gamma'(\theta),
```

we choose:

```math
\Gamma'(\theta)=x.
```

This is the same saddle-point equation as in Cramer's theorem.

Therefore, the same $\theta$ appears twice:

1. In Cramer's theorem, it identifies the most likely way for the rare event to occur.
2. In importance sampling, it defines the simulation law under which that rare event becomes typical.

---

## 17. Mental summary

The full logic is:

```text
1. We study P(S_n/n >= x), with x above the usual mean.
2. We use exponentials because they detect large sums.
3. Independence gives E[e^{theta S_n}] = e^{n Gamma(theta)}.
4. This gives an upper bound exp(-n(theta x - Gamma(theta))).
5. Optimizing over theta gives Gamma*(x).
6. To prove the lower bound, we tilt the law using mu_theta.
7. Under the tilted law, the mean becomes Gamma'(theta).
8. We choose theta such that Gamma'(theta)=x.
9. Then S_n/n is centered around x under the tilted law.
10. The event is no longer rare under P_theta.
11. The likelihood ratio contributes exp(-n(theta x - Gamma(theta))).
12. At the optimum, this is exp(-n Gamma*(x)).
```

The core intuition is:

> Large deviations identify the exponential cost of forcing an empirical mean to behave as if the underlying distribution had a different mean.

---

## 18. Code correspondence

In the codebase, the mathematical objects should correspond to functions such as:

| Mathematical object | Meaning | Possible code name |
|---|---|---|
| $\Gamma(\theta)$ | cumulant generating function | `cgf(theta)` |
| $\Gamma'(\theta)$ | tilted mean | `mean_under_tilt(theta)` |
| $\Gamma^*(x)$ | rate function | `rate_function(x)` |
| $\theta_x$ | optimal tilt | `saddlepoint(x)` |
| $e^{-\theta S_n+n\Gamma(\theta)}$ | likelihood correction | `importance_weight(theta, s_n, n)` |

For the Bernoulli case, if $X\sim\mathrm{Bernoulli}(p)$, then:

```math
\Gamma(\theta)=\log(1-p+pe^\theta),
```

and the tilted Bernoulli parameter is:

```math
p_\theta
=
\frac{pe^\theta}{1-p+pe^\theta}.
```

The optimal tilt for targeting a rare empirical mean $x$ satisfies:

```math
p_\theta=x.
```

Equivalently:

```math
\theta_x
=
\log\left(\frac{x(1-p)}{p(1-x)}\right).
```

The Bernoulli rate function is:

```math
\Gamma^*(x)
=
x\log\left(\frac{x}{p}\right)
+
(1-x)\log\left(\frac{1-x}{1-p}\right),
\qquad x\in[0,1].
```

This is the relative entropy between a Bernoulli distribution with parameter $x$ and a Bernoulli distribution with parameter $p$.

---

## Reference

Huyen Pham, *Large Deviations in Mathematical Finance*, 2010.

Relevant sections:

- Section 2.1: Laplace transform and change of probability measures.
- Section 2.2: Cramer's theorem.
- Remark 2.3: relation with importance sampling.
