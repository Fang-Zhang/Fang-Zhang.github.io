---
title: "Maths for LLMs, Tested in Code (5/10): Statistics, Evaluation Noise and Scaling Laws"
date: 2027-01-18 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, statistics, evaluation, scaling-laws, series]
math: true
description: "Part 5 of 10. Two places where LLM work is statistics: a benchmark score is a sample proportion with a confidence interval, and a scaling law is a regression of loss on size. Simulations show how many questions separate two models, and a learning-curve fit shows how a tight bootstrap interval can sit on top of a wrong extrapolation."
---

Chapter 5 of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) covers sampling, estimators, bias and variance, confidence, hypothesis tests and regression. In LLM work it appears in two places: evaluation (is model B better than model A, or is that noise?) and scaling laws (fit loss against size and extrapolate). Both are run here on this machine (CPU, NumPy and SciPy). The general statistical background (confidence intervals, multiple testing, bias-variance) is in the [ML interview series](/posts/ml-engineer-interviews-13-maths-and-statistics/), so this post sticks to the LLM-specific parts. [Part 4](/posts/maths-for-llms-4-probability-softmax-sampling/) built the character model that supplies the learning curve below.

![Three panels. Left: 95 percent half-width of an accuracy of 0.70 against the number of questions, from 8.8 points at 100 questions to 1.3 points at 5000. Middle: for a true gap of 2 points, the fraction of simulated benchmarks in which the better model ranks first and in which a paired or unpaired test is significant, against the number of questions. Right: held-out loss in bits per character of a 3-gram character model against training size, with a fitted curve E plus B over D to the beta that lies systematically off the points.](/assets/pic/maths-llms-5-statistics.png)
*Left: benchmark size and interval width. Middle: ranking is easy, significance is not. Right: a learning-curve fit that looks tight and extrapolates badly.*

## 1. A benchmark score is a binomial proportion

If a model answers each of $n$ independent questions correctly with probability $p$, the measured accuracy $\hat p$ has standard error $\sqrt{p(1-p)/n}$, and a 95% interval of about $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$. The half-width depends on $n$ only, not on the model. For two models scored on the same questions the quantity of interest is the difference; the script below simulates two models with true accuracies 0.70 and 0.72 whose errors are correlated (both fail on the hard questions), and counts how often the better one ranks first and how often a 95% test calls the difference significant. The unpaired test is a $\chi^2$ test on two proportions. The paired test is the exact McNemar test, which uses only the questions on which the models disagree.

```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
# 1. A benchmark score is a binomial proportion: how wide is its interval?
print("95% Wilson interval for accuracy 0.70 on n questions:")
for n in (100, 500, 1000, 5000):
    lo, hi = stats.binomtest(int(0.7 * n), n).proportion_ci(0.95, method='wilson')
    print(f"  n={n:>5}: [{lo:.3f}, {hi:.3f}]  half-width {100*(hi-lo)/2:.1f} points")
# 2. Two models, true accuracies 0.70 and 0.72, scored on the same n questions. How often is the better one ranked better, and
#    how often does a 95% test call the difference significant? Unpaired vs paired (correlated errors: both models fail on the same hard questions).
def trial(n, rho=0.6):
    d = rng.random(n)                                   # shared question difficulty
    u1 = rng.random(n); u2 = rng.random(n)
    # mix a shared and an independent uniform to get correlated outcomes with the right marginals
    s1 = np.where(rng.random(n) < rho, d, u1); s2 = np.where(rng.random(n) < rho, d, u2)
    a = s1 < 0.70; b = s2 < 0.72
    return a, b
def power(n, paired, reps=1000):
    hit = 0; better = 0
    for _ in range(reps):
        a, b = trial(n)
        better += b.mean() > a.mean()
        if paired:
            x = int(((~a) & b).sum()); y = int((a & (~b)).sum())     # discordant pairs -> exact McNemar
            p = stats.binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
        else:
            p = stats.chi2_contingency([[a.sum(), n - a.sum()], [b.sum(), n - b.sum()]], correction=False)[1]
        hit += (p < 0.05) and (b.mean() > a.mean())
    return better / reps, hit / reps
print("\nTrue gap 2 points (0.70 vs 0.72), outcomes correlated across models:")
print(f"{'n':>6} {'P(better ranked better)':>25} {'unpaired test power':>21} {'paired (McNemar) power':>24}")
for n in (500, 1000, 2000, 5000, 10000):
    r, pu = power(n, False); _, pp = power(n, True)
    print(f"{n:>6} {r:>25.2f} {pu:>21.2f} {pp:>24.2f}")
# 3. Sample size for a given gap, unpaired, 80% power: n = 2 (z_a + z_b)^2 p(1-p) / delta^2
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
for delta in (0.05, 0.02, 0.01):
    print(f"gap {100*delta:.0f} points: n per model ~ {2 * z**2 * 0.7 * 0.3 / delta**2:,.0f} (unpaired, 80% power, accuracy ~0.70)")
```

```
95% Wilson interval for accuracy 0.70 on n questions:
  n=  100: [0.604, 0.781]  half-width 8.8 points
  n=  500: [0.658, 0.739]  half-width 4.0 points
  n= 1000: [0.671, 0.728]  half-width 2.8 points
  n= 5000: [0.687, 0.713]  half-width 1.3 points

True gap 2 points (0.70 vs 0.72), outcomes correlated across models:
     n   P(better ranked better)   unpaired test power   paired (McNemar) power
   500                      0.79                  0.06                     0.13
  1000                      0.88                  0.11                     0.22
  2000                      0.96                  0.25                     0.38
  5000                      1.00                  0.63                     0.79
 10000                      1.00                  0.92                     0.98
gap 5 points: n per model ~ 1,319 (unpaired, 80% power, accuracy ~0.70)
gap 2 points: n per model ~ 8,241 (unpaired, 80% power, accuracy ~0.70)
gap 1 points: n per model ~ 32,965 (unpaired, 80% power, accuracy ~0.70)
```

The first block gives the scale: at 1,000 questions the interval for an accuracy of 0.70 is $\pm$2.8 points, and a 1-point difference between two leaderboard entries is inside it. In the second block the true 2-point gap is real by construction. At 1,000 questions the better model ranks first in 88% of simulated benchmarks, but a test calls the gap significant in only 11% (unpaired) or 22% (paired); at 5,000 questions those figures are 63% and 79%, and it takes 10,000 for 92% and 98%. Ranking and significance are different questions: the first is nearly always right in this setup, while the second needs far more data. Pairing helps because it removes the variance shared by the two models, which is why evaluation papers that compare models on the same items use paired tests. The last lines apply the standard sample-size formula $n = 2(z_{\alpha/2}+z_{\beta})^2\,p(1-p)/\delta^2$: about 1,300 questions per model for a 5-point gap, 8,200 for a 2-point gap and 33,000 for 1 point.

What the simulation leaves out is part of the real problem: a benchmark has finitely many questions, but a model's answers also vary with the sampling temperature (Part 4) and with prompt wording, so the true variance is at least the binomial one. The correlated-outcomes model here is a convenience, with an assumed correlation of 0.6; its output is shaped by that assumption.

## 2. Scaling laws are a regression

Scaling-law papers fit a smooth function to the final loss of many training runs. [Kaplan et al.](https://arxiv.org/abs/2001.08361) report that the test loss of a Transformer language model follows a power law when limited by the non-embedding parameters $N$, the dataset size $D$ or the compute budget. [Hoffmann et al.](https://arxiv.org/abs/2203.15556) model the loss as

$$\hat L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},$$

where, in their description, the first term is the loss of an ideal generative process, the entropy of natural text; the second is the penalty for a finite model; and the third the penalty for finite training. This is non-linear least squares. The experiment fits only the data term, $L(D)=E+B/D^{\beta}$, on the character model of Part 4 (a model with a fixed number of parameters per context, so the model-size term is absent). It trains on the first $D$ characters of a novel for nine values of $D$, measures the loss in bits per character on a held-out tenth, and bootstraps over 20 blocks of the held-out text to get an interval.

```python
# Learning curve of a count-based character model: held-out loss against training-set size D, fitted with L(D) = E + B / D^beta
# (the data term of the form used by Hoffmann et al. 2022). Then a bootstrap over fit points for the uncertainty of E and beta.
import re, numpy as np, collections
from scipy.optimize import curve_fit
t = open('pg.txt', encoding='utf-8').read(); t = t[t.index('*** START'):t.index('*** END')]
t = re.sub(r'\s+', ' ', t.lower()); t = re.sub(r"[^a-z' .,;!?-]", '', t)
chars = sorted(set(t)); V = len(chars); ix = {c: i for i, c in enumerate(chars)}; a = np.array([ix[c] for c in t])
K = 3; split = int(0.9 * len(a)); test = a[split:]; train_all = a[:split]
def ctx_ids(x): return sum(x[i:len(x) - K + i] * V ** (K - 1 - i) for i in range(K))
def heldout_loss(D, alpha=0.1):
    tr = train_all[:D]; c = np.zeros((V ** K, V)); np.add.at(c, (ctx_ids(tr), tr[K:]), 1)
    p = (c + alpha); p /= p.sum(1, keepdims=True)
    return float(-np.log(p[ctx_ids(test), test[K:]]).mean() / np.log(2))   # bits per character
Ds = np.array([2000, 5000, 10000, 20000, 50000, 100000, 200000, 400000, split])
L = np.array([heldout_loss(int(D)) for D in Ds])
for D, l in zip(Ds, L): print(f"D = {D:>7,}  held-out loss {l:.3f} bits/char")
f = lambda D, E, B, b: E + B / D ** b
p0 = (2.0, 10.0, 0.3)
popt, _ = curve_fit(f, Ds, L, p0=p0, bounds=([0, 0, 0.01], [6, 1e4, 3]), maxfev=20000)
print(f"fit: E = {popt[0]:.3f} bits, B = {popt[1]:.2f}, beta = {popt[2]:.3f};  max |residual| = {np.abs(f(Ds, *popt) - L).max():.3f}")
# bootstrap: refit on resampled held-out blocks (the test text is split into 20 blocks; the loss is recomputed on a resample of blocks)
rng = np.random.default_rng(0); blocks = np.array_split(np.arange(len(test) - K), 20)
tr_c = {}
def block_losses(D, alpha=0.1):
    tr = train_all[:D]; c = np.zeros((V ** K, V)); np.add.at(c, (ctx_ids(tr), tr[K:]), 1)
    p = (c + alpha); p /= p.sum(1, keepdims=True); cid = ctx_ids(test)
    ll = -np.log(p[cid, test[K:]]) / np.log(2)
    return np.array([ll[b].mean() for b in blocks])
BL = np.array([block_losses(int(D)) for D in Ds])            # (points, blocks)
fits = []
for _ in range(200):
    sel = rng.integers(0, len(blocks), len(blocks)); Lb = BL[:, sel].mean(1)
    try: fits.append(curve_fit(f, Ds, Lb, p0=popt, bounds=([0, 0, 0.01], [6, 1e4, 3]), maxfev=20000)[0])
    except Exception: pass
fits = np.array(fits)
for name, j in (("E", 0), ("beta", 2)):
    lo, hi = np.percentile(fits[:, j], [2.5, 97.5]); print(f"bootstrap 95% interval for {name}: [{lo:.3f}, {hi:.3f}]  ({len(fits)} refits)")
# extrapolation: fit on the 5 smallest points, predict the largest
sm = slice(0, 5); p2, _ = curve_fit(f, Ds[sm], L[sm], p0=p0, bounds=([0, 0, 0.01], [6, 1e4, 3]), maxfev=20000)
print(f"fit on D <= {Ds[4]:,} only -> predicts {f(Ds[-1], *p2):.3f} at D = {Ds[-1]:,}; measured {L[-1]:.3f}; E = {p2[0]:.3f}, beta = {p2[2]:.3f}")
np.save('scaling.npy', np.array([Ds, L]))
```

```
D =   2,000  held-out loss 4.335 bits/char
D =   5,000  held-out loss 3.809 bits/char
D =  10,000  held-out loss 3.436 bits/char
D =  20,000  held-out loss 3.052 bits/char
D =  50,000  held-out loss 2.520 bits/char
D = 100,000  held-out loss 2.313 bits/char
D = 200,000  held-out loss 2.169 bits/char
D = 400,000  held-out loss 2.087 bits/char
D = 640,451  held-out loss 2.052 bits/char
fit: E = 1.215 bits, B = 21.52, beta = 0.251;  max |residual| = 0.114
bootstrap 95% interval for E: [1.173, 1.267]  (200 refits)
bootstrap 95% interval for beta: [0.245, 0.258]  (200 refits)
fit on D <= 50,000 only -> predicts 1.719 at D = 640,451; measured 2.052; E = 0.000, beta = 0.162
```

Three observations:

- **The fit has a tight interval.** The bootstrap interval for $\beta$ is $[0.245, 0.258]$ and for the irreducible term $E$ is $[1.17, 1.27]$ bits (right panel shows the fit).
- **The fit is wrong anyway.** The maximum residual is 0.114 bits, the points curve away from the line systematically, and the last line is the test that matters: a fit on the five smallest sizes predicts 1.72 bits at 640,451 characters where 2.05 was measured, with an estimated $E$ of exactly 0. The interval measured the noise of the held-out text only. It did not measure whether the functional form suits the data.
- **The likely reason (the author's reading, not tested separately).** For a count model with a fixed context length, the loss cannot go below the entropy that three characters of context leave unexplained, and the amount of data needed to estimate each context grows with the number of contexts. That is a different mechanism from the one the form assumes, so the form is misspecified here.

The lesson for real scaling-law work is about statistics, not about the law: an interval from resampling is conditional on the model form, and extrapolation outside the fitted range is the test of the form. Hoffmann et al. estimate the allocation by more than one approach, and test their prediction by training a model (Chinchilla, 70B parameters on 1.4 trillion tokens) at the predicted size, which is the same kind of check as the last line above.

## Which maths, which component

- Binomial proportion and its interval: every accuracy on a benchmark.
- Paired tests and sample-size formulas: model comparison on a shared test set.
- Non-linear regression with a bootstrap: scaling laws; the check is out-of-range prediction.
- Bias and variance: a fit can be low-variance (narrow interval) and biased (wrong form) at once.

Next: [Part 6, geometry, dot products and the attention scale](/posts/maths-for-llms-6-geometry-dot-products-attention-scale/).

## Limits

- The evaluation simulation assumes independent questions, a fixed correlation of 0.6 between the models' outcomes, and two fixed true accuracies; real benchmarks have clusters of related questions, label errors and judge variance.
- The learning curve is for a count model on one novel, with one train-test split; it demonstrates a procedure and a failure of extrapolation, and says nothing about the constants in published scaling laws.
- The published laws are described only at the level of what the two papers state in the pages read here; their fitted exponents are not reproduced.
- Not covered: multiple comparisons across many benchmarks, contamination of test sets, and Bayesian alternatives to the bootstrap.
