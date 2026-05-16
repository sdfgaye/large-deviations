# Copulas from scratch

## 1. Why do we need copulas?

Suppose every obligor in a credit portfolio has the same marginal default probability:

$$
p = 2\%.
$$

This tells us the risk of one obligor.

It does **not** tell us how defaults happen together.

Two portfolios can have the same marginal default probabilities and very different joint behavior:

```text
Portfolio A:
    defaults are almost independent

Portfolio B:
    defaults are strongly driven by the same macro factor
```

Both can have $p=2\%$, but Portfolio B has much larger crisis risk.

A copula is a way to model this missing piece:

```text
marginal distributions + dependence structure = joint distribution
```

## 2. Marginals versus dependence

A marginal distribution describes one variable alone.

For example:

$$
\mathbb P(Y_k=1)=p
$$

says obligor $k$ defaults with probability $p$.

But a credit portfolio depends on the full vector:

$$
(Y_1,\ldots,Y_n).
$$

To price or simulate portfolio losses, we need the joint law, not just each marginal law.

The loss is

$$
L_n = Y_1+\cdots+Y_n.
$$

If the $Y_k$ are independent, $L_n$ is binomial.

If the $Y_k$ are dependent, large losses can become much more likely.

## 3. The basic copula idea

For a continuous random variable $X$ with cumulative distribution function $F$,

$$
U = F(X)
$$

has a uniform distribution on $[0,1]$.

This means that we can separate two jobs:


> 1. build dependence between uniform variables $U_1,...,U_n$
> 2. transform each $U_k$ into the marginal variable we want


A copula is the joint distribution of those uniforms.

Informally:

$$
C(u_1,\ldots,u_n)
=
\mathbb P(U_1\le u_1,\ldots,U_n\le u_n).
$$

The copula contains the dependence structure.

The marginals are added afterward.

## 4. Gaussian copula recipe

A Gaussian copula starts with a multivariate normal vector:

$$
(G_1,\ldots,G_n)
\sim
N(0,R),
$$

where $R$ is a correlation matrix.

Then define

$$
U_k=\Phi(G_k),
$$

where $\Phi$ is the standard normal CDF.

Each $U_k$ is uniform on $[0,1]$, but the vector $(U_1,\ldots,U_n)$ is dependent because the vector $(G_1,\ldots,G_n)$ is dependent.

That dependent uniform vector is the Gaussian copula.

## 5. Turning the copula into default indicators

For a homogeneous credit portfolio, each obligor has default probability $p$.

A simple default rule is:

$$
Y_k = 1_{\{U_k > 1-p\}}.
$$

Since $U_k$ is uniform,

$$
\mathbb P(U_k>1-p)=p.
$$

So the marginal default probability is correct.

Equivalently, because $U_k=\Phi(G_k)$,

$$
U_k>1-p
\quad\Longleftrightarrow\quad
G_k>\Phi^{-1}(1-p).
$$

So we can write

$$
Y_k = 1_{\{G_k>x_p\}},
\qquad
x_p=\Phi^{-1}(1-p).
$$

This is the threshold representation used in the Gaussian copula credit model.

## 6. One-factor Gaussian copula

The one-factor model writes the Gaussian latent variable as

$$
X_k
=
\rho Z+\sqrt{1-\rho^2}\varepsilon_k,
$$

where:

- $Z\sim N(0,1)$ is the systematic factor,
- $\varepsilon_k\sim N(0,1)$ is the idiosyncratic factor,
- $Z,\varepsilon_1,\ldots,\varepsilon_n$ are independent,
- $0\le \rho <1$ is the factor loading.

Each $X_k$ is standard normal because

$$
\mathrm{Var}(X_k)=\rho^2+(1-\rho^2)=1.
$$

For $i\ne j$,

$$
\mathrm{Corr}(X_i,X_j)=\rho^2.
$$

Important wording:

> 1. $\rho$ is the factor loading 
> 2. $\rho^2$ is the pairwise latent asset correlation in this homogeneous setup.


Default is

$$
Y_k=1_{\{X_k>x_p\}},
\qquad
x_p=\Phi^{-1}(1-p).
$$

## 7. Conditional default probability

Condition on the systematic factor $Z=z$.

Then

$$
X_k = \rho z+\sqrt{1-\rho^2}\varepsilon_k.
$$

Default occurs when

$$
\rho z+\sqrt{1-\rho^2}\varepsilon_k>x_p.
$$

So

$$
\varepsilon_k
>
\frac{x_p-\rho z}{\sqrt{1-\rho^2}}.
$$

Therefore

$$
p(z)
=
\mathbb P(Y_k=1\mid Z=z)
=
1-\Phi\left(
\frac{x_p-\rho z}{\sqrt{1-\rho^2}}
\right).
$$

Using $x_p=\Phi^{-1}(1-p)=-\Phi^{-1}(p)$, this becomes

$$
p(z)
=
\Phi\left(
\frac{\rho z+\Phi^{-1}(p)}
{\sqrt{1-\rho^2}}
\right).
$$

This is the key formula.

## 8. Conditional binomial loss

Given $Z=z$, the defaults are independent Bernoulli variables with probability $p(z)$.

So

$$
L_n\mid Z=z
\sim
\mathrm{Binomial}(n,p(z)).
$$

This is the main computational advantage of the one-factor model.

We can simulate losses in two steps:


> 1. sample $Z$
> 2. sample $L_n | Z$ from $Binomial(n, p(Z))$


No need to simulate all $n$ individual defaults when the portfolio is homogeneous.

## 9. What changes versus independent defaults?

### Independent case

If $\rho=0$, then

$$
p(Z)=p.
$$

So

$$
L_n\sim \mathrm{Binomial}(n,p),
$$

and

$$
\frac{L_n}{n}\to p.
$$

A fixed threshold $q>p$ is a rare event:

$$
\mathbb P(L_n/n\ge q)
$$

decays exponentially in $n$.

### Dependent one-factor case

If $\rho>0$, then $p(Z)$ is random.

Conditionally,

$$
\frac{L_n}{n}\approx p(Z)
$$

for large $n$.

So the limiting loss rate is random:

$$
\frac{L_n}{n}\Rightarrow p(Z).
$$

This is the core difference.

A bad systematic factor can make the conditional default probability high for many obligors at once.

## 10. Why large losses become polynomial

For extreme thresholds $q_n\uparrow 1$, Pham's large-loss result in the homogeneous one-factor Gaussian copula model states:

$$
\lim_{n\to\infty}
\frac{1}{\log n}
\log \mathbb P(L_n\ge nq_n)
=
-a\frac{1-\rho^2}{\rho^2},
$$

when

$$
1-q_n=O(n^{-a}),
\qquad
0<a\le 1.
$$

So the probability behaves like a power of $n$:

$$
\mathbb P(L_n\ge nq_n)
\approx
n^{-a(1-\rho^2)/\rho^2}.
$$

This is very different from the independent Bernoulli case, where large deviations usually produce exponential decay in $n$.

## 11. Common pitfalls

### Pitfall 1 — confusing default probability and dependence

The marginal default probability $p$ says how likely one name is to default.

The copula controls how defaults cluster.

### Pitfall 2 — confusing $\rho$ and pairwise correlation

In the one-factor model

$$
X_k=\rho Z+\sqrt{1-\rho^2}\varepsilon_k,
$$

the pairwise latent correlation is $\rho^2$, not $\rho$.

### Pitfall 3 — thinking the Gaussian copula is perfect

The Gaussian copula is mathematically convenient, but it is not a universal model of crisis dependence.

In this project, it is used because it gives a clean bridge from:

```text
Bernoulli rare events
to dependent portfolio losses
to large-loss asymptotics
to two-step importance sampling
```

## 12. Project map

```text
Notebook 01:
    Bernoulli exponential tilting

Notebook 02:
    Cramer's theorem for independent defaults

Notebook 03:
    Importance sampling for independent rare default tails

Notebook 04:
    One-factor Gaussian copula dependent defaults

Notebook 05:
    Extreme-loss asymptotics and two-step importance sampling
```
