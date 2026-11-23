---
title: "From Zero to ML Engineer, Part 7: Making It Better, and Knowing When to Stop"
date: 2027-05-05 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, regularization, hyperparameter tuning, random forest, gradient boosting, ML engineer]
description: "Part 7 of a 12-part series for complete beginners. Error analysis, regularisation, hyperparameter search and tree ensembles, applied in the order a practitioner would try them, with the honest finding that on this dataset the simple model wins and further tuning is inside the noise."
---

## What you will do, and what you get

You will run the standard improvement ladder on the tumour model: look at its mistakes, turn the regularisation dial, search for settings automatically, and try the two strongest classical models, a random forest and gradient boosting. The result is a measured answer, and on this data it is not the answer many people expect. The tuned logistic regression is the same as the default one, and the ensembles score *below* logistic regression (96.0% and 96.5% against 97.5%). What you gain is a method that tells you when an improvement is real, and when to stop.

![Three panels. Left: logistic regression train and cross-validated accuracy against the regularisation setting C; the cross-validated curve peaks at C equal 1 at 98.5 percent while the train curve keeps rising. Middle: bars for decision tree 92.9, logistic regression 97.5, random forest 96.0 and gradient boosting 96.5 percent, with error bars of about 1.6 to 2.5 points. Right: random forest accuracy against the number of trees, rising from 91.8 percent with one tree to a plateau near 96 percent, below the 97.5 percent line of logistic regression]({{ site.baseurl }}/assets/pic/mlpath-7-improving.png){: width="900" }

## How: the steps

1. **Error analysis.** List the rows the model gets wrong and look at them. Are the mistakes near the boundary, or confident? Do they share a pattern?
2. **Regularise.** Add a penalty on large weights, and choose its strength by cross-validation.
3. **Search settings automatically.** Try a grid of values for each *hyperparameter* (a setting you choose, as opposed to a parameter the model learns), scoring each by cross-validation on the training rows only.
4. **Try a stronger model family.** Ensembles of trees are the usual next step for table data.
5. **Compare with a spread, and stop** when the remaining differences are smaller than the noise.

## Why it works: the principle

**Error analysis comes first because it is the cheapest information.** Cross-validated predictions on the 398 training rows give 6 errors: 4 malignant cases missed and 2 false alarms. Three of the six have a probability between 0.2 and 0.8, so they sit near the boundary: a small change might fix them. The other three are confident mistakes. Row 290 is malignant with a worst perimeter of 85, which is typical of benign rows (the benign median is 87, the malignant one 137) and the model gives it a malignancy probability of 0.01. No dial on this model will fix such a row; it needs information the columns do not contain, or it is a labelling error. Knowing which kind of mistake dominates tells you whether to tune or to collect better data.

**Regularisation** limits how large the weights can grow. In scikit-learn's logistic regression the strength is set by `C`, where a *smaller* C means a *stronger* penalty. The left panel is the underfit/overfit picture from Part 6 on a new axis: with a very strong penalty (C = 0.0001) the model scores 63% (it is close to the always-benign baseline), and with a very weak one (C = 1000) the training score reaches 100% while the cross-validated score falls to 96.0%. The peak, 98.5%, is at C = 1, which is the library default. Defaults are often reasonable, which is why you measure before assuming tuning will help.

**Grid search** automates the loop: for each candidate value it runs cross-validation on the training rows, then picks the best. The [scikit-learn guide](https://scikit-learn.org/stable/modules/grid_search.html) covers it. Note that the best score found by the search is itself optimistic, since it is the maximum over many candidates (the same selection effect as Part 4). Here the search over 13 values of C picks C = 1, and the test result is 95.3%, identical to the default model on the same test rows. When tuning returns the default, the model was already near its best.

**Ensembles.** A single decision tree is unstable: small changes in the data change the tree a lot. A *random forest* trains many trees on random resamples of the rows, with random subsets of columns at each split, and averages their votes. Averaging cancels part of the individual noise, which is why the right panel climbs from 91.8% with one tree to about 96% with 25 or more, then flattens: extra trees do not hurt, and do not help either. Random forests were introduced by [Breiman](https://link.springer.com/article/10.1023/A:1010933404324). *Gradient boosting* builds trees one after another, each fitted to the errors left by the previous ones. The [ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html) describes both.

**Why the simple model wins here.** The comparison uses 5-fold cross-validation repeated 4 times, so each model has 20 scores. Logistic regression gets 97.5% (spread 1.6 points), the forest 96.0% (1.8), boosting 96.5% (1.6), and the single depth-5 tree 92.9% (2.5). The ensembles clearly beat the single tree, but the gap between them and logistic regression is about one spread. On a different dataset, such as one with strong interactions between columns, the ranking can reverse; that is why you measure. The practical rule: prefer the simplest model whose score is within the noise of the best.

## Example, worked through

- **Errors:** 6 of 398 training rows are wrong under cross-validation; 3 are confident mistakes.
- **C sweep (train / cross-validated):** C = 0.01: 95.2 / 94.0; C = 1: 99.5 / 98.5; C = 1000: 100.0 / 96.0.
- **Tuned versus default:** best C = 1 (cross-validated 98.5%), test 95.3%; default C = 1, test 95.3%. The 95.3% is on the one test split from Part 6, which sits at the low end of the 95.3% to 98.8% range that different splits give.
- **Ensembles:** tree 92.9%, forest 96.0%, boosting 96.5%, logistic regression 97.5%, each with a spread of 1.6 to 2.5 points.
- **The decision:** keep logistic regression. It is the most accurate, the cheapest to train, and its weights can be read.

## Implementation

Save as `part7.py` and run it (about a minute on a CPU). It needs numpy and scikit-learn.

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

data = load_breast_cancer(); X, y = data.data, 1 - data.target          # 1 = malignant
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
cv5 = StratifiedKFold(5, shuffle=True, random_state=0)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

# 1. Error analysis: which training rows does cross-validation get wrong?
prob = cross_val_predict(lr, Xa, ya, cv=cv5, method="predict_proba")[:, 1]
wrong = np.where((prob >= 0.5) != ya)[0]
print(f"1. cross-validated errors on the {len(ya)} training rows: {len(wrong)}  (missed malignant {int(((prob<0.5)&(ya==1)).sum())}, false alarms {int(((prob>=0.5)&(ya==0)).sum())})")
j = list(data.feature_names).index("worst perimeter"); med = {c: np.median(Xa[ya == c, j]) for c in (0, 1)}
print(f"   'worst perimeter' median: benign {med[0]:.0f}, malignant {med[1]:.0f}")
for i in wrong[:6]:
    print(f"   row {i:3d}: truth {'malignant' if ya[i] else 'benign   '}, predicted P(malignant)={prob[i]:.2f}, worst perimeter {Xa[i, j]:.0f}")
print(f"   mistakes with P between 0.2 and 0.8: {int(((prob[wrong] > .2) & (prob[wrong] < .8)).sum())} of {len(wrong)}   (confident mistakes: {int(((prob[wrong] <= .2) | (prob[wrong] >= .8)).sum())})")

# 2. Regularisation: C is the inverse of the penalty on large weights
Cs = np.logspace(-4, 3, 8)
tr, va = validation_curve(lr, Xa, ya, param_name="logisticregression__C", param_range=Cs, cv=cv5)
print("\n2. logistic regression, C: train / cross-validated accuracy")
for c, a, b in zip(Cs, tr.mean(1), va.mean(1)): print(f"   C={c:9.4f}: {100*a:5.1f}% / {100*b:5.1f}%")

# 3. Tuning on training rows only, then one look at the test rows
grid = GridSearchCV(lr, {"logisticregression__C": np.logspace(-3, 3, 13)}, cv=cv5).fit(Xa, ya)
print(f"\n3. grid search over 13 values of C: best C={grid.best_params_['logisticregression__C']:.3g}, cross-validated {100*grid.best_score_:.1f}%, test (looked at once) {100*grid.score(Xt, yt):.1f}%")
print(f"   default C=1 on the same test rows: {100*lr.fit(Xa, ya).score(Xt, yt):.1f}%")

# 4. Ensembles versus single models: 5-fold cross-validation repeated 4 times on all 569 rows
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
models = {"decision tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=0),
          "logistic regression": lr,
          "random forest (300 trees)": RandomForestClassifier(300, random_state=0, n_jobs=-1),
          "gradient boosting": HistGradientBoostingClassifier(random_state=0)}
print("\n4. 20 folds each (5-fold x 4 repeats)")
for k, m in models.items():
    s = cross_val_score(m, X, y, cv=rcv); print(f"   {k:27s} mean {100*s.mean():.1f}%  std {100*s.std():.1f}")
print("   random forest, number of trees -> cross-validated accuracy")
for n in (1, 5, 25, 100, 300):
    s = cross_val_score(RandomForestClassifier(n, random_state=0, n_jobs=-1), X, y, cv=rcv); print(f"   {n:3d} trees: {100*s.mean():.1f}%")
```

Output:

```
1. cross-validated errors on the 398 training rows: 6  (missed malignant 4, false alarms 2)
   'worst perimeter' median: benign 87, malignant 137
   row  57: truth malignant, predicted P(malignant)=0.17, worst perimeter 110
   row  98: truth benign   , predicted P(malignant)=0.67, worst perimeter 114
   row 265: truth malignant, predicted P(malignant)=0.07, worst perimeter 120
   row 290: truth malignant, predicted P(malignant)=0.01, worst perimeter 85
   row 319: truth benign   , predicted P(malignant)=0.53, worst perimeter 66
   row 323: truth malignant, predicted P(malignant)=0.46, worst perimeter 108
   mistakes with P between 0.2 and 0.8: 3 of 6   (confident mistakes: 3)

2. logistic regression, C: train / cross-validated accuracy
   C=   0.0001:  63.1% /  63.3%
   C=   0.0010:  87.6% /  87.2%
   C=   0.0100:  95.2% /  94.0%
   C=   0.1000:  98.2% /  97.7%
   C=   1.0000:  99.5% /  98.5%
   C=  10.0000:  99.6% /  97.0%
   C= 100.0000:  99.9% /  96.2%
   C=1000.0000: 100.0% /  96.0%

3. grid search over 13 values of C: best C=1, cross-validated 98.5%, test (looked at once) 95.3%
   default C=1 on the same test rows: 95.3%

4. 20 folds each (5-fold x 4 repeats)
   decision tree (depth 5)     mean 92.9%  std 2.5
   logistic regression         mean 97.5%  std 1.6
   random forest (300 trees)   mean 96.0%  std 1.8
   gradient boosting           mean 96.5%  std 1.6
   random forest, number of trees -> cross-validated accuracy
     1 trees: 91.8%
     5 trees: 94.8%
    25 trees: 95.9%
   100 trees: 96.1%
   300 trees: 96.0%
```

## Try it

1. Add `class_weight="balanced"` to the logistic regression. Does recall of the malignant class change, and at what cost in precision?
2. Grid-search `max_depth` and `min_samples_leaf` for the random forest. Does the tuned forest pass logistic regression?
3. Use `GridSearchCV` inside an outer `cross_val_score` (nested cross-validation) and compare the outer score to `best_score_`. How large is the optimism?
