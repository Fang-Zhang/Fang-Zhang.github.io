---
title: "ML Engineer Interviews, With Reference Answers (3/18): Fundamentals and Classical Algorithms"
date: 2026-12-29 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, machine-learning, scikit-learn, series]
description: "Part 3 of 18. Ten classical-ML interview questions (bias-variance, regularisation, trees, forests, SVM, k-means, feature selection) with reference answers and code that was run, with its output shown."
---

This post covers the first technical round, the fundamentals: bias and variance, overfitting, regularisation, decision trees, random forests, support vector machines, k-means and feature selection. It gives ten questions, and for each one what it tests, a reference answer, a common weak answer and a follow-up. Where an answer depends on a number, the number comes from code that was run for this post, and the code and its output are shown. The answers are the author's, written for hypothetical candidates; they are not any company's questions. The previous post, [Part 2](/posts/ml-engineer-interviews-2-the-recruiter-screen-and-the-role/), covered the recruiter screen.

All code ran on a CPU-only machine with Python 3.14, NumPy 2.5.4 and scikit-learn 1.9.1. Random seeds are fixed, so the numbers repeat. They are results on small synthetic data sets, chosen to show a mechanism; they are not benchmarks, and a different data set will give different values.

![Left: a stacked bar chart of squared bias and variance for polynomial fits of degree 1, 3, 5 and 9. Squared bias falls from 0.182 to about zero and variance rises from 0.008 to 0.039. Right: train and test accuracy of a decision tree against maximum depth; train accuracy rises to 1.0 while test accuracy peaks near depth 6 and then falls.]({{ "/assets/pic/ml-interview-3-bias-variance-depth.png" | relative_url }})
*Left: bias and variance of polynomial fits, measured over 500 repeated training sets (code in question 1). Right: train and test accuracy of a decision tree as depth grows (code in question 5).*

## 1. "Explain the bias-variance trade-off."

**What it tests.** Whether the candidate can connect a formula to behaviour, and whether the candidate knows what is measured and over what. The [Deep Learning book](https://www.deeplearningbook.org/contents/ml.html) (Goodfellow, Bengio and Courville, chapter 5) defines the bias of an estimator as the expected value of the estimate minus the true value, where the expectation is over training sets, and gives the decomposition MSE = Bias² + Var. It ties the pair to capacity, underfitting and overfitting.

**Reference answer.** The expectation is over training sets, not over test points. Imagine drawing many training sets from the same process and fitting the same model to each. Bias is how far the average of those fitted functions is from the truth: it is what the model class cannot represent. Variance is how much the fitted function moves from one training set to the next: it is how much the model is driven by the particular sample. Expected squared error at a point is bias squared plus variance plus the irreducible noise. A model that is too simple has high bias and low variance; a model that is too flexible has the reverse. The trade-off is that capacity moves the two in opposite directions, so the best capacity for a given sample size is in the middle.

The decomposition can be measured by simulation when the truth is known. Here the truth is a sine curve, the training set is 20 points with noise of standard deviation 0.3, and 500 training sets are drawn:

```python
import numpy as np
rng = np.random.default_rng(0)
f = lambda x: np.sin(2*np.pi*x)
x = np.linspace(0, 1, 20)            # fixed design, noise redrawn each repetition
xt = np.linspace(0.05, 0.95, 50)
print("degree  bias^2  variance  bias^2+var  (noise variance 0.09)")
for d in (1, 3, 5, 9):
    P = np.empty((500, xt.size))
    for r in range(500):
        y = f(x) + rng.normal(0, 0.3, x.size)
        P[r] = np.polynomial.Polynomial.fit(x, y, d)(xt)
    b = np.mean((P.mean(0) - f(xt))**2); v = P.var(0).mean()
    print(f"{d:>6} {b:7.3f} {v:9.3f} {b+v:10.3f}")
```

```
degree  bias^2  variance  bias^2+var  (noise variance 0.09)
     1   0.182     0.008      0.190
     3   0.005     0.013      0.019
     5   0.000     0.021      0.021
     9   0.000     0.039      0.039
```

A straight line has almost all of its error in bias (0.182 of 0.190). From degree 3 on, bias is about zero and the error is variance, which rises with degree: 0.013, 0.021, 0.039. The minimum of bias² + variance is at degree 3. The candidate should not read the exact minimum as a rule, because it depends on the sample size and the noise.

**Common weak answer.** "Bias is underfitting and variance is overfitting", with no mention of what is averaged over. The follow-up usually exposes it.

**Follow-up.** "Modern neural networks fit the training data exactly and still generalise. Does the trade-off still hold?" [Belkin et al.](https://arxiv.org/abs/1812.11118) describe a "double descent" curve that contains the textbook U-shaped curve and shows test error falling again as capacity grows beyond the point where the model interpolates the data. A good answer says that the U-shape describes the classical regime, that the second descent is documented for several model families and data sets, and that the trade-off remains a useful way to reason about small models and small data. It should not claim that the classical picture is wrong.

## 2. "How do you tell overfitting from underfitting, and what do you do about each?"

**What it tests.** Diagnosis before treatment. The candidate should reach for the training and validation error before anything else.

**Reference answer.** Compare training error with validation error. High training error and a small gap means underfitting: the model class is too weak, the features are too poor, or training stopped too early. Low training error and a large gap means overfitting: the model is fitting the sample. The fixes are different. For underfitting: add capacity, add features, reduce regularisation, train longer. For overfitting: get more data, reduce capacity, add regularisation, stop early, augment data, or average models. A learning curve (error against training set size) separates the cases further: if validation error is still falling as data grows, more data helps; if the two curves have met at a high error, more data will not help and the model or the features must change.

The decision-tree table in question 5 shows the pattern in numbers: at depth 1 train and test accuracy are both low (0.803 and 0.749), at depth 8 train accuracy is 0.955 and test 0.795, and with no depth limit train is 1.000 and test 0.772.

**Common weak answer.** "Get more data" for every problem. It is the right remedy for variance only.

**Follow-up.** "Training accuracy is 99% and validation accuracy is 99% but production accuracy is 80%. What happened?" The gap is not overfitting in the sense above; the usual causes are leakage, a train and serving mismatch, or a shift in the data. Parts 4 and 5 cover these.

## 3. "What is regularisation? Compare L1 and L2."

**What it tests.** Whether the candidate knows regularisation as a preference built into the objective, and can say what each penalty does to coefficients. The Deep Learning book defines regularisation as any modification made to a learning algorithm that is intended to reduce its generalisation error but not its training error, and gives weight decay as the first example: minimise the training MSE plus λ times the squared L2 norm of the weights, where λ sets the strength of the preference for small weights.

**Reference answer.** Regularisation adds a penalty on the weights to the training loss. L2 (ridge) penalises the sum of squared weights; it shrinks all weights toward zero without setting them exactly to zero, and it handles correlated features by spreading weight among them. L1 (lasso) penalises the sum of absolute values. According to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/linear_model.html), the lasso estimates sparse coefficients, meaning it can set coefficients exactly to zero and prefers solutions with fewer non-zero coefficients. The consequence is that L1 performs feature selection as part of fitting, and L2 does not. The strength is a hyperparameter chosen by cross-validation. Penalised models need features on comparable scales, because the penalty depends on the size of the weights.

Here 60 training rows have 30 features, of which only 5 have any effect, and noise of standard deviation 2:

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error as mse
rng = np.random.default_rng(1)
n, p = 60, 30
X = rng.normal(size=(n+2000, p)); w = np.zeros(p); w[:5] = [3, -2, 1.5, 1, -1]
y = X @ w + rng.normal(0, 2, n+2000)
Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
for name, m in [("OLS", LinearRegression()), ("Ridge a=10", Ridge(alpha=10)), ("Lasso a=0.3", Lasso(alpha=0.3))]:
    m.fit(Xtr, ytr); c = m.coef_
    print(f"{name:12} train MSE {mse(ytr, m.predict(Xtr)):6.2f}  test MSE {mse(yte, m.predict(Xte)):6.2f}  "
          f"nonzero coefs {int((np.abs(c) > 1e-8).sum()):2d}  sum|coef| on the 25 noise features {np.abs(c[5:]).sum():5.2f}")
```

```
OLS          train MSE   2.82  test MSE  10.10  nonzero coefs 30  sum|coef| on the 25 noise features 10.54
Ridge a=10   train MSE   3.38  test MSE   8.45  nonzero coefs 30  sum|coef| on the 25 noise features  7.88
Lasso a=0.3  train MSE   4.96  test MSE   6.04  nonzero coefs 14  sum|coef| on the 25 noise features  2.11
```

Training error goes up as the penalty is applied (2.82, 3.38, 4.96) and test error goes down (10.10, 8.45, 6.04), which is the definition at work. Lasso kept 14 non-zero coefficients, not 5, and the noise features retain a small total weight (2.11). The alphas were not tuned; a candidate who says "the lasso recovers the true features" from a result like this one would be overstating it.

**Common weak answer.** "L1 gives sparsity because of the diamond shape" with no statement of what that means for the fitted model, or "L2 is better because it is smoother" with no condition.

**Follow-up.** "Two features are almost identical. What do L1 and L2 do?" L2 splits the weight between them; L1 tends to keep one and drop the other, and which one is unstable across resamples. This follows from the penalties' shapes; the numbers above do not test it.

## 4. "Why is logistic regression a linear model, and how is it regularised?"

**What it tests.** Whether the candidate sees that the model is linear in the log-odds and knows the practical defaults.

**Reference answer.** Logistic regression models the log-odds of the positive class as a linear function of the features, passes it through the sigmoid, and is fitted by maximising the likelihood, equivalently minimising the cross-entropy. The decision boundary is a hyperplane. It is a classifier although the name says regression. Squared error is a poor loss for it, because the loss is not convex in the weights once a sigmoid is inside it and the gradients vanish for confident wrong predictions; cross-entropy avoids both. The scikit-learn documentation says regularisation is applied by default, which is common in machine learning but not in statistics, and that no regularisation amounts to setting C to a very high value; so a default fit is already regularised, and features should be scaled. The candidate should be able to say that the coefficients are log-odds ratios per unit of the feature, and that the model's probabilities are only useful if they are calibrated, which Part 4 covers.

**Common weak answer.** "It uses a sigmoid so it is nonlinear." The sigmoid is on the output; the boundary is linear.

**Follow-up.** "When would you pick logistic regression over a boosted tree model?" When interpretability, speed, a stable baseline or a small data set matter. The answer is a trade-off, not a ranking.

## 5. "How does a decision tree choose splits, and why do trees overfit?"

**What it tests.** The mechanism of greedy recursive partitioning and the knobs that control it.

**Reference answer.** At each node the tree tries splits on each feature and picks the one that most reduces an impurity measure (Gini or entropy for classification, squared error for regression) in the resulting children. It repeats until a stopping rule fires. The search is greedy, so each split is locally best and the tree is not globally optimal. Without a stopping rule the tree keeps splitting until the leaves are pure, which means it memorises the training set, noise included. The [scikit-learn documentation](https://scikit-learn.org/stable/modules/tree.html) notes that trees tend to overfit on data with a large number of features and that the deeper the tree, the more complex the rules and the fitter the model. The controls are maximum depth, minimum samples per leaf, and cost-complexity pruning (the `ccp_alpha` parameter). Trees need little data preparation, because they do not need feature scaling, and prediction cost is logarithmic in the number of training rows.

The experiment uses 1,500 rows, 20 features of which 5 are informative, and 10% of the labels flipped at random:

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=1500, n_features=20, n_informative=5, flip_y=0.1, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
print("max_depth  train_acc  test_acc")
for d in (1, 2, 3, 5, 8, None):
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    print(f"{str(d):>9} {t.score(Xtr, ytr):10.3f} {t.score(Xte, yte):9.3f}")
```

```
max_depth  train_acc  test_acc
        1      0.803     0.749
        2      0.821     0.749
        3      0.852     0.763
        5      0.916     0.801
        8      0.955     0.795
     None      1.000     0.772
```

Train accuracy rises to 1.000 by memorising the flipped labels. Test accuracy rises to 0.801 at depth 5 and falls to 0.772 for the unlimited tree: this is the right panel of the figure.

**Common weak answer.** Naming Gini and entropy and stopping there. The follow-up is how the tree is kept from overfitting.

**Follow-up.** "Why can't a tree extrapolate?" Its prediction is a constant per leaf, so outside the range of the training data it returns the value of the nearest leaf.

## 6. "Why does a random forest work, and does adding trees cause overfitting?"

**What it tests.** Whether the candidate understands bagging as variance reduction, and can cite the condition under which it works.

**Reference answer.** A single deep tree has low bias and high variance. Bagging trains many trees on bootstrap samples and averages them. The scikit-learn documentation says that bagging methods are used to reduce the variance of a base estimator by introducing randomisation into its construction and then averaging, and that they work best with strong, complex models such as fully grown trees, whereas boosting usually works best with weak models such as shallow trees. A random forest adds a second randomisation: each split considers only a random subset of the features, which lowers the correlation between trees. [Breiman's 2001 paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) states that the generalisation error of a forest depends on the strength of the individual trees and the correlation between them, and that the error converges to a limit as the number of trees grows, which he gives as the reason why random forests do not overfit as more trees are added. The caveat is that this is a statement about the number of trees: a forest of deep trees on a noisy problem can still be a poor model, and the limit is not necessarily a good value.

On the same data as question 5:

```python
from sklearn.ensemble import RandomForestClassifier
for n in (1, 10, 100, 500):
    r = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    print(f"forest n_trees={n:>3}  train {r.score(Xtr, ytr):.3f}  test {r.score(Xte, yte):.3f}")
```

```
forest n_trees=  1  train 0.903  test 0.737
forest n_trees= 10  train 0.987  test 0.801
forest n_trees=100  train 1.000  test 0.823
forest n_trees=500  train 1.000  test 0.821
```

Test accuracy goes from 0.737 for one tree to 0.801 for ten, 0.823 for 100 and 0.821 for 500: it flattens and does not decline, which agrees with the paper's claim. The training accuracy is 1.000 with 100 trees, so a perfect training score does not mean the forest overfits harmfully; the training score of a bagged model is not an honest estimate, because each tree has seen most of the rows.

**Common weak answer.** "It combines many models, so it is more accurate." The reason it works (variance reduction through weakly correlated trees) is the point.

**Follow-up.** "How does boosting differ?" Boosting fits models sequentially, each correcting the errors of the earlier ones, and it reduces bias; it can overfit as rounds are added, so the number of rounds is tuned with early stopping.

## 7. "Explain support vector machines: margin, support vectors, kernels, C and gamma."

**What it tests.** Depth in one classical method, and whether the candidate can state what the hyperparameters do.

**Reference answer.** The linear SVM picks, among the hyperplanes that separate the classes, the one with the largest margin. The training points that lie on the margin, the support vectors, determine the solution; the other points could be removed without changing it. For data that cannot be separated, the soft-margin form of [Cortes and Vapnik (1995)](https://link.springer.com/article/10.1007/BF00994018) allows violations at a cost, which is the role of C: large C punishes violations heavily and fits the training data closely, and small C allows a wider margin with more violations. Kernels let the method work in a high-dimensional feature space without computing the mapping; the scikit-learn documentation lists the linear, polynomial, RBF and sigmoid kernels, and defines the RBF kernel as exp(−γ‖x − x′‖²). Gamma sets how local the RBF kernel is: a large gamma makes each point influence only its close neighbours and the boundary becomes wiggly. The documentation also states that SVMs are not scale invariant, so features should be scaled, and that training cost grows quickly with the number of training vectors (between O(n_features × n_samples²) and O(n_features × n_samples³) for the libsvm-based implementation).

The paper gives a result candidates rarely mention: if the training data are separable, the expected test error is bounded by the expected number of support vectors divided by the number of training vectors, and the bound does not contain the dimension of the space. On a separable two-blob data set, with a large C to approximate a hard margin, the check is:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
X, y = make_blobs(n_samples=120, centers=2, cluster_std=0.8, random_state=4)
m = SVC(kernel="linear", C=1e6).fit(X, y)
loo = 1 - cross_val_score(SVC(kernel="linear", C=1e6), X, y, cv=LeaveOneOut()).mean()
print(f"separable blobs: n={len(y)} n_SV={m.n_support_.sum()}  nSV/n={m.n_support_.sum()/len(y):.3f}  "
      f"leave-one-out error={loo:.3f}  train acc={m.score(X, y):.2f}")
X, y = make_moons(n_samples=600, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
for C, g in [(1, 1), (1, 0.1), (100, 1), (1000, 100), (1000, 1000)]:
    s = SVC(C=C, gamma=g).fit(Xtr, ytr)
    print(f"C={C:<5} gamma={g:<6} nSV={s.n_support_.sum():3d}  train {s.score(Xtr, ytr):.3f}  test {s.score(Xte, yte):.3f}")
```

```
separable blobs: n=120 n_SV=3  nSV/n=0.025  leave-one-out error=0.017  train acc=1.00
C=1     gamma=1      nSV= 79  train 0.937  test 0.880
C=1     gamma=0.1    nSV=125  train 0.883  test 0.850
C=100   gamma=1      nSV= 54  train 0.947  test 0.883
C=1000  gamma=100    nSV=246  train 1.000  test 0.840
C=1000  gamma=1000   nSV=299  train 1.000  test 0.657
```

Three of 120 points are support vectors (0.025), and the leave-one-out error is 0.017, which is below the bound's value. That is one data set and not a proof, and the bound concerns an expectation. The second block shows the two knobs on the two-moons data: with gamma at 1000 the training accuracy is 1.000 and the test accuracy is 0.657, because every point has become its own island and 299 of 300 training points are support vectors. A high support-vector fraction is itself a warning sign of overfitting, which follows from the bound above.

**Common weak answer.** "It finds the best separating hyperplane" with no mention of the margin, the soft margin or scaling.

**Follow-up.** "When would you not use an SVM?" With a very large training set, because of the training cost above, or when calibrated probabilities are needed, because the SVM outputs a score and not a probability.

## 8. "How does k-means work, and what are its failure modes?"

**What it tests.** Whether the candidate treats a simple algorithm as an optimisation problem with known weaknesses.

**Reference answer.** K-means minimises the within-cluster sum of squares, which scikit-learn calls the inertia, over k centroids. Lloyd's algorithm alternates two steps: assign each point to the nearest centroid, then move each centroid to the mean of its points. Every step lowers or keeps the objective, so it converges, but only to a local optimum, which can be poor; [Arthur and Vassilvitskii](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) state this in their paper on k-means++. Their remedy is a seeding rule: pick the first centre uniformly at random, then pick each next centre with probability proportional to D(x)², the squared distance from the point to the nearest centre already chosen. They prove that the expected cost after seeding is within a factor of 8(ln k + 2) of the optimum. scikit-learn uses this seeding by default. The documentation lists other limits: inertia assumes that clusters are convex and isotropic and responds poorly to elongated clusters or irregular shapes, it is not a normalised metric, and in very high dimensions distances inflate, so a dimensionality reduction such as PCA first can help. The number of clusters must be chosen; the documentation points to silhouette analysis for that.

The effect of seeding over 200 runs on 3,000 points in 12 blobs, with one initialisation per run:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=3000, centers=12, cluster_std=0.6, random_state=3)
for init in ("random", "k-means++"):
    s = np.array([KMeans(12, init=init, n_init=1, random_state=i).fit(X).inertia_ for i in range(200)])
    print(f"{init:10} inertia over 200 seeds: min {s.min():8.1f} median {np.median(s):8.1f} max {s.max():8.1f}  "
          f"share within 1% of best {np.mean(s <= s.min()*1.01):.2f}")
```

```
random     inertia over 200 seeds: min   1983.1 median   3506.3 max  13110.3  share within 1% of best 0.03
k-means++  inertia over 200 seeds: min   1983.1 median   2037.9 max   2799.4  share within 1% of best 0.29
```

Both methods find the same best value (1983.1), but random seeding reaches it in 3% of runs and has a median 77% above the best and a worst case above 13,000; k-means++ reaches it in 29% of runs, with a median 3% above and a worst case of 2,799. Even with k-means++, one run is not enough; the usual practice is several restarts and keeping the lowest inertia.

**Common weak answer.** Describing the two steps and nothing else. The questions that follow are about initialisation, scale and the choice of k.

**Follow-up.** "The data has one feature in metres and another in millimetres. What happens?" The distance is dominated by the larger-scale feature, so features should be standardised first.

## 9. "How would you select features, and what can go wrong?"

**What it tests.** Method knowledge, and awareness of the leak that cross-validation does not catch if selection happens outside it.

**Reference answer.** There are three families. Filter methods score each feature on its own (correlation, mutual information, an F-test) and keep the top k. Wrapper methods search subsets using a model (recursive elimination). Embedded methods select as part of fitting: the lasso's zeroed coefficients, or importances from a tree ensemble, which scikit-learn's `SelectFromModel` uses to drop unimportant features. The pitfall is to select features on the whole data set and then cross-validate: the labels have already influenced which features are kept, so the validation folds are not independent. The scikit-learn documentation on [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) says that data leakage is information unavailable at prediction time being used to build the model, that it gives over-optimistic cross-validation estimates, and that transformations must be learned from the training data only. Selection is such a transformation.

The demonstration uses data in which nothing can be learned: 60 rows, 2,000 noise features and random labels, so the true accuracy is 0.5.

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
rng = np.random.default_rng(0)
X = rng.normal(size=(60, 2000)); y = rng.integers(0, 2, 60)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
Xs = SelectKBest(f_classif, k=20).fit_transform(X, y)
wrong = cross_val_score(LogisticRegression(), Xs, y, cv=cv).mean()
right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression()), X, y, cv=cv).mean()
print(f"select-then-CV accuracy {wrong:.2f}; selection inside the CV pipeline {right:.2f}")
```

```
select-then-CV accuracy 0.95; selection inside the CV pipeline 0.48
```

Selecting first gives 0.95 accuracy on labels that are pure noise; selecting inside each training fold gives 0.48, which is chance. The fix is to put every fitted step in a pipeline and cross-validate the whole pipeline. Part 4 returns to leakage.

**Common weak answer.** Listing methods with no mention of where in the pipeline selection happens.

**Follow-up.** "Correlated features share importance in a tree ensemble. How does that affect importance-based selection?" Importance is split among them, so each looks less important than the group is, and dropping by threshold can remove a whole group.

## 10. "A new tabular problem arrives. Which model do you try first, and why?"

**What it tests.** Judgement: whether the candidate starts with a baseline and chooses by the properties of the problem and not by fashion. This question has no source, and the answer is the author's.

**Reference answer.** Start with the simplest thing that could work and measure it: a majority-class or mean prediction, then a regularised linear or logistic model with sensible features. The baseline sets the number a complex model has to beat, and it often reveals data problems early. Then try a tree ensemble (random forest or gradient boosting), which handles mixed feature types and nonlinearities with little preprocessing, as the tree documentation notes. Compare using the same cross-validation, with all preprocessing inside the pipeline. Choose with the constraints in view: interpretability and speed favour the linear model, accuracy on nonlinear tabular data often favours the ensembles, and a small data set limits how much capacity can be used. State what would change the choice, such as a latency limit, a need for calibrated probabilities, or a requirement to explain each prediction.

**Common weak answer.** "I'd use XGBoost" or "I'd use a neural network" before looking at the data.

**Follow-up.** "The baseline has 91% accuracy. Is that good?" It depends on the class balance and the cost of errors, which Part 4 covers.

## What the sources show

These points are the author's reading.

- **Each method has a documented failure mode, and the question usually targets it.** Trees overfit, forests depend on weakly correlated trees, SVMs depend on scaling and gamma, k-means depends on seeding and cluster shape, and selection leaks if done outside cross-validation.
- **The numbers above come from small synthetic sets.** They illustrate mechanisms; the sizes of the gaps will differ elsewhere.
- **The theory has limits.** The U-shaped bias-variance curve is a classical picture, and Belkin et al. document regimes where it does not describe the test error.

## Limits of this summary

- The Deep Learning book and the scikit-learn pages were read online. The Elements of Statistical Learning was not opened for this post, so nothing here is attributed to it.
- The Cortes and Vapnik bound is quoted from the paper as an expectation, and the single data set above illustrates it without proving it.
- Question 4 and question 10 rest on general practice and the scikit-learn defaults and are not tied to a specific source beyond that.
- Boosting is mentioned only in outline; Part 5 and Part 6 return to gradient boosting and deep models.
