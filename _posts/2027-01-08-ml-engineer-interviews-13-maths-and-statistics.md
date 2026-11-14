---
title: "ML Engineer Interviews, With Reference Answers (13/18): Mathematics and Statistics"
date: 2027-01-08 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, statistics, linear-algebra, probability, mle, series]
description: "Part 13 of 18. Ten mathematics and statistics questions for ML interviews: least squares, PCA, maximum likelihood, Bayes, confidence intervals, multiple testing, bias-variance, ridge as a prior, heavy tails and numerical error, each checked by simulation."
---

This post covers the mathematics and statistics questions that follow the coding round of [Part 12](/posts/ml-engineer-interviews-12-coding/): linear algebra, probability, maximum likelihood, Bayes, intervals and tests, and numerical issues. Each question has what it tests, a reference answer, a weak answer and a follow-up. Each claim that can be checked by computation was run here with fixed seeds, and the output is shown; claims about books and notes are limited to passages that were read: the [Deep Learning book](https://www.deeplearningbook.org/) chapters on linear algebra, probability and numerical computation, and the [Stanford CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf) (2026 edition). Statements that no source supports are marked as the author's.

![Left: actual coverage of a nominal 95% interval for a 2% rate: Wald 0.643, 0.907, 0.950 and Wilson 0.922, 0.933, 0.947 at n = 50, 200, 1000. Right: interquartile range of the sample mean against sample size for normal, lognormal and Cauchy data; normal and lognormal shrink, Cauchy stays between 1.5 and 2.5.](/assets/pic/ml-interview-13-ci-coverage-clt.png)
*Left: intervals for a small rate (question 5). Right: the sample mean for three distributions (question 9).*

## 1. "Solve linear regression in closed form. When does it break?"

**What it tests.** The normal equations, and the difference between a formula and a numerically sound algorithm.

**Reference answer.** Least squares minimises the squared residual; setting the gradient to zero gives the normal equations `X^T X b = X^T y` (the CS229 notes derive them in their linear-regression chapter). They have a unique solution when the columns of X are linearly independent. Two things go wrong in practice. First, collinear (or nearly collinear) columns make `X^T X` singular or ill-conditioned. The Deep Learning book defines conditioning as how rapidly a function changes with small changes in its input and gives the condition number of a matrix with an eigendecomposition as the ratio of its largest to its smallest eigenvalue in magnitude; large values make inversion sensitive to rounding. Second, forming `X^T X` squares the condition number (checked below), so an explicit inverse of it loses about twice as many digits as a method that works on X directly. The same book describes the pseudoinverse, computed from the SVD, as giving the solution of minimal Euclidean norm when there are many solutions and the closest fit when there is none; [`np.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) works from the singular values (its documentation treats those below `rcond` times the largest as zero), and `np.linalg.pinv` computes the pseudoinverse.

```python
import numpy as np
rng = np.random.default_rng(0); n = 200
x1 = rng.normal(size=n); x2 = x1 + 1e-6 * rng.normal(size=n); X = np.c_[x1, x2, rng.normal(size=n)]       # two almost identical columns
beta = np.array([1.0, 1.0, 2.0]); y = X @ beta + 0.01 * rng.normal(size=n)
print(f"cond(X) = {np.linalg.cond(X):.2e}, cond(X^T X) = {np.linalg.cond(X.T @ X):.2e}  (the square of cond(X) up to rounding: {np.linalg.cond(X) ** 2:.2e})")
b_ne = np.linalg.inv(X.T @ X) @ X.T @ y                                                                  # normal equations with an explicit inverse
b_ls = np.linalg.lstsq(X, y, rcond=None)[0]                                                              # SVD-based least squares
b_pi = np.linalg.pinv(X) @ y
for name, b in (("normal equations", b_ne), ("lstsq (SVD)", b_ls), ("pinv", b_pi)):
    print(f"{name:17s} coefficients {np.round(b, 3)}  residual {np.linalg.norm(X @ b - y):.4f}")
print("sum of the first two coefficients (identifiable): ne %.4f, lstsq %.4f, true 2.0" % (b_ne[:2].sum(), b_ls[:2].sum()))
Xs = [X + 0.0]; rs = np.random.default_rng(1); sp = []
for _ in range(200):
    yy = X @ beta + 0.01 * rs.normal(size=n); sp.append(np.linalg.lstsq(X, yy, rcond=None)[0][:2])
print(f"spread (std) of the first coefficient over 200 noise draws: {np.std([s[0] for s in sp]):.1f} (the true value is 1.0; the data cannot tell the two columns apart)")
```

```
cond(X) = 1.88e+06, cond(X^T X) = 3.53e+12  (the square of cond(X) up to rounding: 3.53e+12)
normal equations  coefficients [-1157.918  1159.918     1.999]  residual 0.1416
lstsq (SVD)       coefficients [-1157.624  1159.625     1.999]  residual 0.1416
pinv              coefficients [-1157.624  1159.625     1.999]  residual 0.1416
sum of the first two coefficients (identifiable): ne 2.0006, lstsq 2.0007, true 2.0
spread (std) of the first coefficient over 200 noise draws: 690.7 (the true value is 1.0; the data cannot tell the two columns apart)
```

The condition number of X is 1.9e6 and that of `X^T X` is 3.5e12, its square. All three methods reach the same residual (0.1416) and the same fit, but the individual coefficients are meaningless: they come out near -1158 and +1160 where the true values are 1 and 1. The sum of the two coefficients is identifiable and is recovered (2.0007 against 2.0), and with new noise draws the first coefficient has a standard deviation of 691. The data cannot separate two columns that differ by 1e-6, which is a statistical problem, not a numerical one; regularisation (question 8) or dropping a column is the remedy. The practical rule: never invert `X^T X` explicitly; use `lstsq`, a QR or SVD-based solver, or a regularised solve.

**Common weak answer.** "`inv(X.T @ X) @ X.T @ y`", with no mention of what happens when it is singular. The inverse exists in the textbook and fails in the dataset.

**Follow-up.** "What if there are more features than examples?" The normal equations have infinitely many solutions; the pseudoinverse picks the minimal-norm one, and a ridge penalty picks a unique one.

## 2. "Explain PCA and its link to the SVD."

**What it tests.** Eigendecomposition and SVD as tools, not memorised definitions.

**Reference answer.** PCA finds orthogonal directions of maximal variance. After centring the data matrix Z (rows are examples), the directions are the eigenvectors of the covariance matrix, and the variances are its eigenvalues. Equivalently, take the SVD `Z = U S V^T`: the rows of `V^T` are the same directions, and the variance along each is `S^2 / (n - 1)`. The SVD route is preferred numerically because it never forms the covariance matrix (the same squaring of the condition number as in question 1). Keeping the top k components gives the best rank-k approximation of Z in squared error, and the error equals the sum of the dropped squared singular values. State the assumptions: PCA is linear, sensitive to the scale of the features (standardise first when units differ), and it keeps variance, which is not the same as keeping what matters for a prediction.

```python
import numpy as np
from sklearn.decomposition import PCA
rng = np.random.default_rng(0); Z = rng.normal(size=(500, 5)) @ rng.normal(size=(5, 5)); Z -= Z.mean(0)
U, S, Vt = np.linalg.svd(Z, full_matrices=False); var_svd = S ** 2 / (len(Z) - 1)
w, V = np.linalg.eigh(np.cov(Z.T)); var_eig = w[::-1]
p = PCA().fit(Z)
print("explained variance, SVD of the centred data :", np.round(var_svd, 3))
print("eigenvalues of the covariance matrix        :", np.round(var_eig, 3))
print("scikit-learn PCA                            :", np.round(p.explained_variance_, 3))
print("components agree up to sign:", np.allclose(np.abs(Vt), np.abs(p.components_), atol=1e-8))
print(f"first two components keep {var_svd[:2].sum() / var_svd.sum():.1%} of the variance; reconstruction error of the rank-2 fit {np.linalg.norm(Z - (U[:, :2] * S[:2]) @ Vt[:2]) ** 2:.1f} = sum of the dropped S^2 {np.sum(S[2:] ** 2):.1f}")
```

```
explained variance, SVD of the centred data : [10.28   8.988  3.242  1.395  0.326]
eigenvalues of the covariance matrix        : [10.28   8.988  3.242  1.395  0.326]
scikit-learn PCA                            : [10.28   8.988  3.242  1.395  0.326]
components agree up to sign: True
first two components keep 79.5% of the variance; reconstruction error of the rank-2 fit 2476.5 = sum of the dropped S^2 2476.5
```

The three computations give identical variances (10.28, 8.99, 3.24, 1.40, 0.33), the components match scikit-learn's up to sign, and the rank-2 reconstruction error (2476.5) equals the sum of the dropped squared singular values (2476.5). The sign ambiguity is real: eigenvectors are defined up to sign, so a test of "the same components" must compare absolute values or the subspaces.

**Common weak answer.** "PCA reduces dimensions" with no statement of what is preserved, or forgetting to centre the data, which turns the first component into the mean.

**Follow-up.** "How many components do you keep?" A cumulative explained-variance threshold, a gap in the spectrum, or a downstream validation score; the first two are heuristics and the third is the one that answers the question being asked (the author's view).

## 3. "What is maximum likelihood estimation? Where does it show up in what you already use?"

**What it tests.** Whether the candidate sees losses as likelihoods, and knows the weaknesses.

**Reference answer.** The CS229 notes state the principle: choose the parameters that make the observed data as probable as possible, and since any strictly increasing function of the likelihood has the same maximiser, maximise the log-likelihood, which turns products into sums. Two standard losses are special cases. Under the assumption of independent Gaussian noise on the target, maximising the likelihood is minimising the sum of squared errors; the notes add that the assumption is not necessary for least squares to be a reasonable method. For classification with a softmax, the notes show that the negative log-likelihood of the training data is the cross-entropy loss. The weaknesses to name are that MLE can be biased in small samples and gives extreme estimates when data are scarce.

```python
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(0)
# 1. Gaussian regression: maximising the likelihood is the same as minimising squared error
X = np.c_[np.ones(100), rng.normal(size=100)]; y = X @ [1.0, 2.0] + rng.normal(scale=0.5, size=100)
nll = lambda t: 0.5 * np.sum((y - X @ t[:2]) ** 2) / np.exp(2 * t[2]) + 100 * t[2]                      # negative log-likelihood, sigma = exp(t[2]); constants dropped
r = minimize(nll, [0, 0, 0]); print("MLE coefficients", np.round(r.x[:2], 4), " least squares", np.round(np.linalg.lstsq(X, y, rcond=None)[0], 4))
# 2. the MLE of a variance divides by n and is biased low
n = 5; draws = rng.normal(size=(200000, n)); print(f"n = {n}: mean of the MLE variance {draws.var(1, ddof=0).mean():.3f}, with n-1 {draws.var(1, ddof=1).mean():.3f} (true 1.000; expected factor (n-1)/n = {(n - 1) / n})")
# 3. Bernoulli: the MLE of the rate is the sample frequency, and says 0 after 0 successes in 5
k, m = 0, 5; print("MLE of a click rate after 0 clicks in 5 views:", k / m)
```

```
MLE coefficients [0.9725 2.0275]  least squares [0.9725 2.0275]
n = 5: mean of the MLE variance 0.802, with n-1 1.002 (true 1.000; expected factor (n-1)/n = 0.8)
MLE of a click rate after 0 clicks in 5 views: 0.0
```

The first line maximises the Gaussian likelihood numerically (with the noise scale also estimated) and recovers the least-squares coefficients to four decimals. The second shows the bias: the maximum-likelihood variance divides by n, and with n = 5 its mean over 200,000 samples is 0.802 against a true 1.0, the expected factor (n-1)/n = 0.8; dividing by n-1 gives 1.002. The third shows the extreme-estimate problem: 0 clicks in 5 views gives an estimated rate of exactly 0, which question 4 repairs with a prior.

**Common weak answer.** "It maximises the probability of the parameters." It maximises the probability of the data given the parameters; the other is a posterior and needs a prior.

**Follow-up.** "Is the MLE of a variance unbiased?" No, as the simulation shows; the correction is n-1, and for large n the difference disappears.

## 4. "A fraud alert fires on 99% of fraud and 2% of legitimate payments; fraud is 0.1% of payments. What is the probability that an alert is fraud?"

**What it tests.** Bayes' rule and the base-rate fallacy, with a number.

**Reference answer.** The Deep Learning book gives Bayes' rule as `P(x|y) = P(x) P(y|x) / P(y)` and notes that `P(y)` can be computed as the sum over x of `P(y|x) P(x)`. Here `P(alert) = 0.99 x 0.001 + 0.02 x 0.999 = 0.02097`, and the posterior is `0.00099 / 0.02097 = 0.0472`: about 5% of alerts are fraud despite a 99% detection rate, because legitimate payments outnumber fraud a thousand to one. The point for ML work is that precision depends on the base rate, so a metric measured on a rebalanced test set does not transfer to production (Part 4). The same rule gives a fix for the scarce-data problem of question 3: put a prior on the rate and update it.

```python
import numpy as np
from scipy.stats import beta
sens, fpr, base = 0.99, 0.02, 0.001                                                                  # alert fires on 99% of fraud and 2% of legitimate payments; fraud is 0.1%
post = sens * base / (sens * base + fpr * (1 - base)); print(f"P(fraud | alert) by Bayes' rule: {post:.4f}")
rng = np.random.default_rng(0); N = 5_000_000; fraud = rng.random(N) < base; alert = np.where(fraud, rng.random(N) < sens, rng.random(N) < fpr)
print(f"simulation of {N:,} payments: {fraud[alert].mean():.4f} of the {alert.sum():,} alerts are fraud")
a0, b0 = 2, 98                                                                                       # prior Beta(2, 98): mean click rate 2%
for k, n in ((0, 5), (3, 100), (30, 1000)):
    mle = k / n; mean = (a0 + k) / (a0 + b0 + n); lo, hi = beta.ppf([0.025, 0.975], a0 + k, b0 + n - k)
    print(f"{k:3d} clicks in {n:4d} views: MLE {mle:.3f}  posterior mean {mean:.4f}  95% credible interval [{lo:.3f}, {hi:.3f}]")
```

```
P(fraud | alert) by Bayes' rule: 0.0472
simulation of 5,000,000 payments: 0.0477 of the 104,612 alerts are fraud
  0 clicks in    5 views: MLE 0.000  posterior mean 0.0190  95% credible interval [0.002, 0.052]
  3 clicks in  100 views: MLE 0.030  posterior mean 0.0250  95% credible interval [0.008, 0.051]
 30 clicks in 1000 views: MLE 0.030  posterior mean 0.0291  95% credible interval [0.020, 0.040]
```

The simulation of 5 million payments agrees (0.0477). The three rows show a Beta(2, 98) prior (mean 2%, an assumption for illustration) updated with data: after 0 clicks in 5 views the posterior mean is 0.019 instead of 0, and the credible interval is wide (0.002 to 0.052); after 1,000 views the prior has little influence and the estimate is close to the frequency (0.0291 against 0.030). The CS229 notes describe the same mechanism for parameters: with a Gaussian prior the maximum a posteriori estimate has a smaller norm than the maximum-likelihood one and is less susceptible to overfitting, the subject of question 8.

**Common weak answer.** "The test is 99% accurate, so 99%." This is the base-rate fallacy, and the usual interviewer follow-up exists to catch it.

**Follow-up.** "What changes if fraud doubles to 0.2%?" The posterior rises to about 9% (0.00198 / 0.02194, the author's arithmetic), which is why alert quality needs monitoring when the base rate drifts (Part 11).

## 5. "Give a confidence interval for a click-through rate measured on 50 impressions. What does the 95% mean?"

**What it tests.** Whether the textbook formula is used blindly, and the meaning of coverage.

**Reference answer.** The 95% describes the procedure: if the experiment were repeated many times and the interval recomputed each time, about 95% of the intervals should contain the true rate. It is not the probability that this interval contains the truth. The default formula, estimate plus or minus 1.96 standard errors (the Wald interval), relies on a normal approximation that is poor for small samples and rates near 0 or 1. The statsmodels documentation for `proportion_confint` lists alternatives, among them the Wilson score, Agresti-Coull, Clopper-Pearson (beta) and Jeffreys intervals, with the normal approximation as the default. For other statistics, the bootstrap is the general tool; [SciPy's `bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) defaults to the BCa method and also offers a percentile method. Whichever is used, check its coverage by simulation, as below.

```python
import numpy as np
from scipy.stats import norm, bootstrap
z = norm.ppf(0.975)
def wald(k, n): p = k / n; h = z * np.sqrt(p * (1 - p) / n); return p - h, p + h
def wilson(k, n):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return c - h, c + h
rng = np.random.default_rng(0)
for n, p in ((50, 0.02), (200, 0.02), (1000, 0.02), (50, 0.3)):
    ks = rng.binomial(n, p, 20000); out = []
    for f in (wald, wilson):
        lo, hi = f(ks, n); out.append(np.mean((lo <= p) & (p <= hi)))
    print(f"n = {n:5d}, true rate {p:.2f}: coverage of the nominal 95% interval  Wald {out[0]:.3f}  Wilson {out[1]:.3f}")
data = (rng.random(300) < 0.1).astype(float); r = bootstrap((data,), np.mean, n_resamples=5000, method="percentile", rng=1)
print(f"300 clicks-or-not observations, rate {data.mean():.3f}: bootstrap {r.confidence_interval.low:.3f}-{r.confidence_interval.high:.3f}, Wilson {wilson(data.sum(), 300)[0]:.3f}-{wilson(data.sum(), 300)[1]:.3f}")
```

```
n =    50, true rate 0.02: coverage of the nominal 95% interval  Wald 0.643  Wilson 0.922
n =   200, true rate 0.02: coverage of the nominal 95% interval  Wald 0.907  Wilson 0.933
n =  1000, true rate 0.02: coverage of the nominal 95% interval  Wald 0.950  Wilson 0.947
n =    50, true rate 0.30: coverage of the nominal 95% interval  Wald 0.939  Wilson 0.959
300 clicks-or-not observations, rate 0.137: bootstrap 0.097-0.177, Wilson 0.102-0.180
```

For a true rate of 2%, the Wald interval covers the truth in only 64.3% of repetitions at n = 50, 90.7% at n = 200 and 95.0% at n = 1000; the Wilson interval is at 92.2%, 93.3% and 94.7%. At a rate of 30% with n = 50 both are close to nominal (93.9% and 95.9%). On a sample of 300 observations with an observed rate of 0.137 the bootstrap (0.097 to 0.177, the percentile method) and Wilson (0.102 to 0.180) agree closely. The lesson is to know where the approximation fails: small counts and rare events, which are exactly the situation of most click-through, conversion and fraud rates.

**Common weak answer.** "We are 95% sure the rate is in this interval", or the Wald formula at n = 50 with 1 click, whose lower bound is negative.

**Follow-up.** "Can you say that the rate is below 5% with 95% confidence?" Use a one-sided interval or an exact method, and say which; with so few events the honest answer may be that the data cannot show it.

## 6. "You test 20 metrics in an A/B test and one is significant at 0.05. Is the feature a success?"

**What it tests.** Multiple comparisons and optional stopping; the experiment-design side is in Part 4.

**Reference answer.** If each of m independent tests has a 5% false-positive rate under no effect, the chance of at least one false positive is `1 - 0.95^m`: 22.6% for 5 metrics and 64.2% for 20. Corrections include Bonferroni (use `0.05 / m` as the threshold) and false-discovery-rate procedures; better is to declare one primary metric before the experiment and treat the others as guardrails or exploration. A second inflation comes from peeking: testing repeatedly and stopping at the first significant result raises the false-positive rate well above the nominal level, which is why fixed sample sizes or sequential methods designed for repeated looks are used.

```python
import numpy as np
from scipy.stats import ttest_ind
rng = np.random.default_rng(0)
print("probability of at least one false positive among m independent tests at 0.05:", {m: round(1 - 0.95 ** m, 3) for m in (1, 5, 20)})
hits = sum(min(ttest_ind(rng.normal(size=500), rng.normal(size=500)).pvalue for _ in range(20)) < 0.05 for _ in range(2000)); print(f"simulation: 20 metrics with no true effect, at least one p < 0.05 in {hits / 2000:.3f} of the experiments")
print(f"Bonferroni (threshold 0.05/20 = {0.05 / 20}), share of experiments with any false positive:", sum(min(ttest_ind(rng.normal(size=500), rng.normal(size=500)).pvalue for _ in range(20)) < 0.05 / 20 for _ in range(2000)) / 2000)
peek = 0                                                                                             # peeking: test after every 100 users per arm up to 2000 users, stop at the first p < 0.05
for _ in range(2000):
    a, b = rng.normal(size=2000), rng.normal(size=2000); peek += any(ttest_ind(a[:k], b[:k]).pvalue < 0.05 for k in range(100, 2001, 100))
print(f"no true effect, 20 looks at the data: declared significant in {peek / 2000:.3f} of the experiments (one look: 0.05)")
```

```
probability of at least one false positive among m independent tests at 0.05: {1: 0.05, 5: 0.226, 20: 0.642}
simulation: 20 metrics with no true effect, at least one p < 0.05 in 0.642 of the experiments
Bonferroni (threshold 0.05/20 = 0.0025), share of experiments with any false positive: 0.0505
no true effect, 20 looks at the data: declared significant in 0.250 of the experiments (one look: 0.05)
```

With no true effect in any of 20 metrics (500 users per arm, 2,000 simulated experiments) at least one p-value falls below 0.05 in 64.2% of the experiments, matching the formula. With the Bonferroni threshold of 0.0025 the share falls to 5.05%. For peeking, a single metric with no true effect and 20 looks at the data, stopping at the first p below 0.05, was declared significant in 25.0% of experiments, five times the nominal 5%.

**Common weak answer.** "p = 0.04, so there is a 96% chance the effect is real." The p-value is the probability of data at least this extreme if there were no effect, not the probability that the effect is real.

**Follow-up.** "Bonferroni seems conservative; what else?" Controlling the false-discovery rate (Benjamini-Hochberg) allows more discoveries when many metrics have real effects; that was not simulated here and is stated from general knowledge.

## 7. "Explain the bias-variance trade-off with a concrete decomposition."

**What it tests.** Precision about what is being decomposed.

**Reference answer.** The CS229 notes set up the regression case with `y = h*(x) + noise`, a model trained on a random training set, and a test example, and decompose the expected test error. The standard result is that expected squared error equals squared bias (how far the average prediction, over training sets, is from the true function) plus variance (how much the prediction moves between training sets) plus the irreducible noise variance. A simple model has high bias and low variance; a flexible one has the opposite. The decomposition is over the randomness of the training set, which makes it estimable only in simulation or with resampling; and the CS229 notes themselves point to the double-descent phenomenon as a limit of the classic picture for modern, heavily over-parameterised models.

```python
import numpy as np
rng = np.random.default_rng(0); f = lambda x: np.sin(2 * np.pi * x); xt = np.linspace(0, 1, 200); sigma = 0.3
for deg in (1, 3, 6):
    preds = []
    for _ in range(500):
        x = rng.random(30); y = f(x) + sigma * rng.normal(size=30); preds.append(np.polyval(np.polyfit(x, y, deg), xt))
    P = np.array(preds); bias2 = ((P.mean(0) - f(xt)) ** 2).mean(); var = P.var(0).mean()
    yt = f(xt)[None] + sigma * rng.normal(size=P.shape); mse = ((P - yt) ** 2).mean()
    print(f"degree {deg}: bias^2 {bias2:.4f}  variance {var:.4f}  noise {sigma ** 2:.4f}  sum {bias2 + var + sigma ** 2:.4f}  measured test MSE {mse:.4f}")
```

```
degree 1: bias^2 0.1999  variance 0.0238  noise 0.0900  sum 0.3137  measured test MSE 0.3154
degree 3: bias^2 0.0047  variance 0.0175  noise 0.0900  sum 0.1123  measured test MSE 0.1122
degree 6: bias^2 0.0003  variance 0.1626  noise 0.0900  sum 0.2529  measured test MSE 0.2543
```

The data are a noisy sine wave (noise standard deviation 0.3, so noise variance 0.09), 30 points per training set, 500 training sets. A straight line has a squared bias of 0.200 and a variance of 0.024; a cubic reduces the bias to 0.005 at a variance of 0.018; a degree-6 polynomial has bias 0.0003 and a variance of 0.163 (largely at the ends of the interval). In each case bias squared plus variance plus noise matches the directly measured test error to within 0.002 (0.3137 against 0.3154, 0.1123 against 0.1122, 0.2529 against 0.2543). The best of the three is the middle one, which is the trade-off in one table.

**Common weak answer.** "Bias is underfitting and variance is overfitting" without saying that variance is the variation across training sets, and that noise is a third term that no model removes.

**Follow-up.** "How does more data change it?" Variance shrinks with more data for a fixed model class; bias does not, which is why a too-simple model plateaus.

## 8. "Why does adding an L2 penalty help? What is it, probabilistically?"

**What it tests.** Linking regularisation to priors, and giving the closed form.

**Reference answer.** The ridge solution is `b = (X^T X + lambda I)^-1 X^T y`. The added `lambda I` also fixes the conditioning problem of question 1, since the matrix is invertible for any `lambda > 0`. Probabilistically, the CS229 notes define the maximum a posteriori estimate as the maximiser of the likelihood times a prior, note that a common choice is a Gaussian prior with zero mean and covariance `tau^2 I`, and that the resulting estimate has smaller norm than the maximum-likelihood one and is less susceptible to overfitting. With Gaussian noise of variance `sigma^2`, the MAP estimate is the ridge solution with `lambda = sigma^2 / tau^2` (derived here from the two densities and checked numerically below): a strong prior (small `tau^2`) means a large penalty. The penalty helps when the prior is roughly right, and hurts when it is too strong.

```python
import numpy as np
from sklearn.linear_model import Ridge
rng = np.random.default_rng(0); X = rng.normal(size=(40, 10)); w = rng.normal(scale=np.sqrt(1 / 5.0), size=10); y = X @ w + rng.normal(scale=1.0, size=40)
lam = 5.0; closed = np.linalg.solve(X.T @ X + lam * np.eye(10), X.T @ y); sk = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
print("closed form (X^T X + lambda I)^-1 X^T y vs sklearn Ridge, max abs difference:", np.abs(closed - sk).max())
sigma2, tau2 = 1.0, 1.0 / lam; print(f"Gaussian noise variance {sigma2} and prior variance {tau2:.2f} per weight give lambda = sigma^2 / tau^2 = {sigma2 / tau2}")
from scipy.optimize import minimize
neg_log_post = lambda t: 0.5 * np.sum((y - X @ t) ** 2) / sigma2 + 0.5 * np.sum(t ** 2) / tau2; print("maximum a posteriori estimate equals the ridge solution:", np.allclose(minimize(neg_log_post, np.zeros(10), tol=1e-12).x, closed, atol=1e-4))
res = {l: [] for l in (0.0, lam, 50.0)}; rr = np.random.default_rng(5)                                  # average over 300 fresh datasets drawn from the same model
for _ in range(300):
    Xa = rr.normal(size=(40, 10)); wa = rr.normal(scale=np.sqrt(tau2), size=10); ya = Xa @ wa + rr.normal(size=40); Xt = rr.normal(size=(500, 10))
    for l in res: b = np.linalg.solve(Xa.T @ Xa + l * np.eye(10), Xa.T @ ya); res[l].append(np.mean((Xt @ (b - wa)) ** 2))
for l, v in res.items(): print(f"lambda {l:5.1f}: mean test MSE (noise-free) over 300 datasets {np.mean(v):.3f}")
```

```
closed form (X^T X + lambda I)^-1 X^T y vs sklearn Ridge, max abs difference: 2.220446049250313e-16
Gaussian noise variance 1.0 and prior variance 0.20 per weight give lambda = sigma^2 / tau^2 = 5.0
maximum a posteriori estimate equals the ridge solution: True
lambda   0.0: mean test MSE (noise-free) over 300 datasets 0.334
lambda   5.0: mean test MSE (noise-free) over 300 datasets 0.276
lambda  50.0: mean test MSE (noise-free) over 300 datasets 0.740
```

The closed form matches scikit-learn's `Ridge` to 2e-16, and the MAP solution from a generic optimiser matches the closed form. The experiment averages the test error over 300 fresh datasets drawn from the model that the prior describes: lambda = 5 gives 0.276, better than no penalty (0.334), and lambda = 50 is too strong and gives 0.740. A prior that does not describe the data would shift this result, so the penalty strength is chosen by validation in practice.

**Common weak answer.** "It stops the weights from getting too large" with no reason that this helps, or applying the penalty to the intercept and to unscaled features.

**Follow-up.** "What does L1 correspond to?" A Laplace prior, which tends to give exact zeros; that mapping is stated from general knowledge and was not tested here.

## 9. "Why does averaging help? When does it not?"

**What it tests.** The central limit theorem, and awareness that its conditions are not automatic.

**Reference answer.** The Deep Learning book states that the central limit theorem shows the sum of many independent random variables is approximately normally distributed, which is one reason the normal distribution models noise well. For the sample mean of independent, identically distributed observations with a finite variance, the spread shrinks like one over the square root of n: 100 times more data gives 10 times less spread. The conditions matter. With heavy tails the convergence is slow, and for a distribution without a mean, such as the Cauchy, averaging does not help at all (a standard result, not taken from the sources above; the experiment shows the effect).

```python
import numpy as np
rng = np.random.default_rng(0)
for name, gen in (("normal", lambda s: rng.normal(size=s)), ("lognormal(0, 1.5)", lambda s: rng.lognormal(0, 1.5, size=s)), ("Cauchy", lambda s: rng.standard_cauchy(size=s))):
    row = []
    for n in (100, 10000, 1000000):
        m = np.array([gen(n).mean() for _ in range(100)]) if n < 1000000 else np.array([gen(n).mean() for _ in range(20)])
        row.append(f"n={n:>7,}: IQR of the sample mean {np.subtract(*np.percentile(m, [75, 25])):9.4f}")
    print(f"{name:18s}", " | ".join(row))
```

```
normal             n=    100: IQR of the sample mean    0.1362 | n= 10,000: IQR of the sample mean    0.0120 | n=1,000,000: IQR of the sample mean    0.0016
lognormal(0, 1.5)  n=    100: IQR of the sample mean    0.9196 | n= 10,000: IQR of the sample mean    0.0978 | n=1,000,000: IQR of the sample mean    0.0105
Cauchy             n=    100: IQR of the sample mean    1.9562 | n= 10,000: IQR of the sample mean    1.4508 | n=1,000,000: IQR of the sample mean    2.4354
```

The interquartile range of the sample mean falls from 0.136 to 0.0120 to 0.0016 for normal data as n goes from 100 to 10,000 to 1,000,000, about a factor of 10 per 100-fold increase in n. For a heavy-tailed lognormal (sigma = 1.5) it falls from 0.92 to 0.098 to 0.011, the same rate but from a much higher starting point: a sample of 100 is a poor guide. For the Cauchy it does not fall at all (1.96, 1.45, 2.44, the variation being noise in a 20-to-100 repetition estimate). In practice this is why the mean of revenue per user, session length or latency can be dominated by a few observations, and why medians, trimmed means or log transforms are used, and why intervals for such metrics need the bootstrap rather than a normal formula (and even then check the coverage, question 5).

**Common weak answer.** "By the central limit theorem everything is normal for n above 30." The number 30 is a rule of thumb, and the lognormal result above shows it fails for skewed data.

**Follow-up.** "How would you estimate the average revenue per user with a few huge customers?" Report the mean with a bootstrap interval and the median alongside it, and say which one the decision depends on.

## 10. "Why is this numerically wrong, and what do you do about it?"

**What it tests.** Floating-point awareness, which separates working code from correct maths. It also covers the cost side (complexity) of the linear algebra used above.

**Reference answer.** The Deep Learning book, in its chapter on numerical computation, treats overflow and underflow with the softmax example (stabilised by subtracting the maximum, as in Part 12, question 8) and poor conditioning as in question 1. Three more cases come up in interviews. The one-pass formula `E[x^2] - E[x]^2` for the variance subtracts two huge, nearly equal numbers and loses the answer; a one-pass algorithm that updates the mean and the sum of squared deviations (Welford's) or a two-pass computation does not. Accumulating many small numbers in float32 into a running total loses the small increments once the total is large. And a linear system should be solved with a solver, not by forming an inverse: the solver is both cheaper and, as the test shows, more accurate. (Dense solves and inverses of an n x n matrix both grow as n cubed; that is general knowledge, not from the sources above, and the timing below is the measurement.)

```python
import numpy as np, time
x = np.random.default_rng(0).normal(1e8, 1.0, 100000)                                                  # tiny variance on a huge offset
naive = np.mean(x ** 2) - np.mean(x) ** 2
mean = m2 = 0.0
for i, v in enumerate(x, 1): d = v - mean; mean += d / i; m2 += d * (v - mean)                         # Welford's one-pass algorithm
print(f"variance of data with mean 1e8 and std 1: E[x^2]-E[x]^2 = {naive:.3f}, Welford {m2 / len(x):.4f}, numpy {x.var():.4f}")
f = np.full(10_000_000, 0.1, dtype=np.float32); run = float(np.cumsum(f[:1000000], dtype=np.float32)[-1])
print(f"float32 values of 0.1: numpy sum of 10 million {f.sum():.1f} (exact 1,000,000); running float32 sum of the first 1 million {run:.1f} (exact 100,000)")
A = np.random.default_rng(1).normal(size=(2000, 2000)); b = A @ np.ones(2000)
t0 = time.perf_counter(); s1 = np.linalg.solve(A, b); t1 = time.perf_counter() - t0
t0 = time.perf_counter(); s2 = np.linalg.inv(A) @ b; t2 = time.perf_counter() - t0
print(f"2000 x 2000 system: solve {t1:.2f} s, error {np.abs(s1 - 1).max():.1e}; inverse then multiply {t2:.2f} s, error {np.abs(s2 - 1).max():.1e}")
```

```
variance of data with mean 1e8 and std 1: E[x^2]-E[x]^2 = 2.000, Welford 1.0003, numpy 1.0003
float32 values of 0.1: numpy sum of 10 million 1000000.1 (exact 1,000,000); running float32 sum of the first 1 million 100958.3 (exact 100,000)
2000 x 2000 system: solve 0.08 s, error 1.4e-12; inverse then multiply 0.25 s, error 9.3e-12
```

For data with a mean of 1e8 and a standard deviation of 1, the one-pass formula returns 2.000 where the true variance is 1.0003, which both Welford and NumPy return. Summing 10 million float32 values of 0.1, NumPy's `sum` gives 1000000.1 (exact: 1,000,000), but a naive running float32 total over only the first 1 million values gives 100958.3 against an exact 100,000, an error of almost 1%. For a 2000 x 2000 system the solver took 0.08 s with an error of 1.4e-12, and the inverse followed by a multiplication took 0.25 s with an error of 9.3e-12. Timings vary between runs.

**Common weak answer.** "Floating-point error is too small to matter." The variance example is off by a factor of two, with no warning.

**Follow-up.** "How do you find such a bug in a pipeline?" Compare against a higher-precision or reference implementation on a small input, check invariants (variances are non-negative, probabilities sum to one) and look at extreme inputs.

## Takeaways

- Do not form `X^T X` and invert it; it squares the condition number (1.9e6 to 3.5e12 in the test). Collinear columns make individual coefficients meaningless (standard deviation 691) while the identifiable combination is fine.
- PCA is the SVD of centred data; compare components up to sign.
- Maximum likelihood explains least squares and cross-entropy; the variance MLE is biased low by (n-1)/n (0.802 at n = 5).
- With a 99% detection rate and a 0.1% base rate, 4.7% of alerts are fraud.
- At a 2% rate and n = 50 the Wald interval covers 64.3% of the time; Wilson covers 92.2%.
- 20 metrics at 0.05 give a 64.2% chance of a false positive; 20 looks at one metric give 25.0%.
- Bias, variance and noise add up to the measured error in the simulation; ridge is MAP under a Gaussian prior and helped (0.276 against 0.334) when the prior matched.
- The mean of heavy-tailed data converges slowly or not at all; the Cauchy sample mean never settled.
- One-pass variance gave 2.000 instead of 1.0003; use solvers, not inverses.

## Limits

- The experiments use synthetic data, fixed seeds and a single machine; they illustrate the claims and do not prove them in general. Coverage figures are from 20,000 repetitions, the test results from 2,000.
- Claims about sources are limited to passages read: the Deep Learning book chapters on linear algebra, probability and numerical computation, the CS229 notes, and the statsmodels and SciPy documentation pages. The statements about the Benjamini-Hochberg procedure, the Laplace prior, the Cauchy distribution, the cubic cost of dense solves and the doubling of the base rate in question 4 are the author's or general knowledge and were not tested against a source.
- The prior in questions 4 and 8 and all rates in the examples are chosen for illustration.
- Not covered: hypothesis tests beyond the t-test, information theory, optimisation theory, and proofs; interviews for research roles ask for derivations that this post does not attempt.
- The candidates in the examples are hypothetical, and the reference answers are one reasonable answer among several.

Next: [Part 14, the take-home assignment and presenting a model](/posts/ml-engineer-interviews-14-take-home-and-presenting-a-model/).
