---
title: "From Zero to ML Engineer, Part 5: Features and Your First Real Models"
date: 2027-05-03 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, scikit-learn, logistic regression, decision tree, feature scaling, ML engineer]
description: "Part 5 of a 12-part series for complete beginners. Turning columns into model inputs (scaling, one-hot encoding), fitting logistic regression and a decision tree on real tumour data, reading what each one learned, and why preprocessing belongs inside a Pipeline."
---

## What you will do, and what you get

You will fit your first two real models on the tumour data from Part 2, using the honest split from Part 4: **logistic regression**, which reaches 97.4% test accuracy, and a **decision tree**, which reaches about 93%. You will see why the numbers in a table have to be *prepared* before a model can use them (one change, scaling, lifts a distance-based model from 91.7% to 95.5%), and how to read what each model learned. You will also see a leak hiding inside preprocessing that turns pure noise into an apparent 85% accuracy, and the one-line fix.

![Three panels. Left: test accuracy of nearest-neighbour models, 91.7 and 93.4 on raw columns versus 95.5 and 95.8 on scaled columns. Middle: decision tree accuracy against maximum depth, flat near 93 percent, below the 97.4 percent line of logistic regression. Right: accuracy on pure noise, 85 percent when columns are selected before cross-validation versus 48 percent inside a Pipeline]({{ site.baseurl }}/assets/pic/mlpath-5-models.png){: width="900" }

## How: the steps

1. **Make every column a number.** Text categories become several 0/1 columns (*one-hot encoding*).
2. **Put numeric columns on a common scale.** *Standardising* subtracts each column's mean and divides by its standard deviation, so every column is centred on 0 with a spread of 1.
3. **Chain the steps with the model in a `Pipeline`.** Fit it on the training rows only; it applies the same steps to new rows automatically.
4. **Fit a simple model first.** Logistic regression for a yes/no answer, a decision tree when you want readable rules.
5. **Read what it learned**, as weights or as rules, and check the score on the validation rows.

## Why it works: the principle

**Why scaling matters.** In the tumour data, "area" runs in the hundreds while "smoothness" sits near 0.1. A nearest-neighbour model measures distance between rows, so the large-valued column decides almost everything and the others are ignored. Standardising gives each column an equal say. Gradient descent (Part 3) has the same trouble: the loss landscape becomes a long narrow valley, and the walk down it is slow. Here, logistic regression needs 2,914 solver iterations on raw columns and 17 on scaled ones. Tree models, in contrast, only ask "is this value above a threshold?", so scaling does not change them. The [scikit-learn preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html) covers scalers and encoders.

**Logistic regression** is Part 3's line with a squashing step. It computes a weighted sum of the columns, `z = w1 x1 + w2 x2 + ... + b`, and turns it into a probability between 0 and 1 with the function `1 / (1 + e^-z)`. Training is the same loop as before: pick the weights that make the loss small, by gradient descent. Because it has one weight per column, the weights are the explanation. After scaling, a larger weight (in magnitude) means a more influential column, and the sign gives the direction. See the [linear models guide](https://scikit-learn.org/stable/modules/linear_model.html).

**A decision tree** is a set of if/else questions, chosen greedily: at each step it picks the column and threshold that best separate the classes, then repeats inside each half. The [decision tree guide](https://scikit-learn.org/stable/modules/tree.html) describes the method. The depth-2 tree in the output is the whole model in six lines. Notice that its right-hand branch ends in class 0 on both sides, so that split does not change any prediction: trees can contain redundant questions. Depth is a dial: a deeper tree fits the training rows more closely, but the middle panel shows accuracy flat from depth 5 on. The tree never catches logistic regression here. One plausible reason is that a weighted sum of many columns draws a slanted boundary, which a single tree can only approximate with right-angled cuts; Part 7 returns to trees with ensembles.

**The leak inside preprocessing.** Anything that learns from data is fitted by `fit`: scalers learn means, a feature selector learns which columns look useful. If it sees the test rows, information flows backwards, the same problem as Part 4. The cleanest demonstration is the one in the right panel. With 5,000 columns of random numbers and random labels, nothing can be learned, and the true accuracy is 50%. Choosing the 20 columns that best match the labels *using all rows*, then cross-validating, reports 85%: among 5,000 random columns, some match 100 random labels by chance, and the selection step found them while looking at the answers. Putting the selector inside a `Pipeline` re-selects columns on each training part only, and the score drops to 48%. [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) (Hastie, Tibshirani and Friedman) discusses this exact trap under "the wrong and right way to do cross-validation". A Pipeline makes the right way the default.

## Example, worked through

- **Scaling (left panel, mean test accuracy over 20 splits):** 1-nearest-neighbour 91.7% raw, 95.5% scaled; 15 neighbours 93.4% raw, 95.8% scaled. Same model, same data, one preprocessing step.
- **Logistic regression** (scaled, 20 splits): 97.4%. This is the same number as the Part 1 trial; there it was a single line in a comparison, here you know what is inside it.
- **What it learned:** the three largest weights are `radius error` (-1.08), `worst texture` (-1.07) and `mean concave points` (-1.01). In this dataset the label 1 is benign, so a negative weight pushes toward malignant: larger errors in radius, rougher texture and more concave points all raise the malignancy score. Weights of correlated columns share credit, so read them as a hint rather than as proof that one column causes the outcome.
- **One-hot encoding:** `["Auckland", "Wellington", "Auckland", "Dunedin"]` becomes three columns, `city_Auckland`, `city_Dunedin` and `city_Wellington`, with exactly one 1 per row. A single column numbered 0, 1, 2 would claim that Dunedin sits between Auckland and Wellington.

## Implementation

Save as `part5.py` and run it. It needs numpy, pandas and scikit-learn.

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_selection import SelectKBest, f_classif

data = load_breast_cancer(); X, y = data.data, data.target
def mean_test(model, n=20):
    s = []
    for seed in range(n):
        a, b, ya, yb = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed); s.append(model.fit(a, ya).score(b, yb))
    return 100 * np.mean(s)

# 1. Scaling: the same models with and without it
print("1. scaling (mean test accuracy, 20 splits)")
print(f"   1-nearest-neighbour  raw {mean_test(KNeighborsClassifier(1)):.1f}%   scaled {mean_test(make_pipeline(StandardScaler(), KNeighborsClassifier(1))):.1f}%")
print(f"   15-nearest-neighbours raw {mean_test(KNeighborsClassifier(15)):.1f}%   scaled {mean_test(make_pipeline(StandardScaler(), KNeighborsClassifier(15))):.1f}%")
a, b, ya, yb = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
raw = LogisticRegression(max_iter=100000).fit(a, ya); sc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=100000)).fit(a, ya)
print(f"   logistic regression: solver iterations raw {raw.n_iter_[0]}, scaled {sc[-1].n_iter_[0]}")

# 2. The models themselves
print("\n2. three models (mean test accuracy, 20 splits)")
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
print(f"   logistic regression {mean_test(lr):.1f}%")
for d in (1, 2, 3, 5, 10, None): print(f"   decision tree, max depth {str(d):4s} {mean_test(DecisionTreeClassifier(max_depth=d, random_state=0)):.1f}%")
tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(a, ya)
print("   the depth-2 tree, as rules:\n" + "\n".join("     " + l for l in export_text(tree, feature_names=list(data.feature_names)).splitlines()))
sc.fit(a, ya); w = sc[-1].coef_[0]; order = np.argsort(-np.abs(w))[:3]
print("   logistic regression: three largest weights (on scaled columns, positive = benign):", [(str(data.feature_names[i]), round(float(w[i]), 2)) for i in order])

# 3. Text categories become numbers: one-hot encoding
df = pd.DataFrame({"city": ["Auckland", "Wellington", "Auckland", "Dunedin"]})
enc = OneHotEncoder().fit(df); print("\n3. one-hot encoding of", df.city.tolist(), "\n  ", enc.get_feature_names_out().tolist(), "\n  ", enc.transform(df).toarray().astype(int).tolist())

# 4. A leak inside preprocessing: choose the 20 'best' of 5000 pure-noise columns using ALL rows, then cross-validate
rng = np.random.default_rng(0); Xn = rng.standard_normal((100, 5000)); yn = rng.integers(0, 2, 100)
wrong = cross_val_score(LogisticRegression(max_iter=1000), SelectKBest(f_classif, k=20).fit_transform(Xn, yn), yn, cv=5).mean()
right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)), Xn, yn, cv=5).mean()
print(f"\n4. noise only, labels random (true accuracy 50%): select-then-validate {100*wrong:.0f}%   selection inside a Pipeline {100*right:.0f}%")
```

Output:

```
1. scaling (mean test accuracy, 20 splits)
   1-nearest-neighbour  raw 91.7%   scaled 95.5%
   15-nearest-neighbours raw 93.4%   scaled 95.8%
   logistic regression: solver iterations raw 2914, scaled 17

2. three models (mean test accuracy, 20 splits)
   logistic regression 97.4%
   decision tree, max depth 1    90.3%
   decision tree, max depth 2    92.5%
   decision tree, max depth 3    92.8%
   decision tree, max depth 5    93.4%
   decision tree, max depth 10   93.0%
   decision tree, max depth None 93.0%
   the depth-2 tree, as rules:
     |--- worst perimeter <= 106.10
     |   |--- worst concave points <= 0.16
     |   |   |--- class: 1
     |   |--- worst concave points >  0.16
     |   |   |--- class: 0
     |--- worst perimeter >  106.10
     |   |--- worst concave points <= 0.14
     |   |   |--- class: 0
     |   |--- worst concave points >  0.14
     |   |   |--- class: 0
   logistic regression: three largest weights (on scaled columns, positive = benign): [('radius error', -1.08), ('worst texture', -1.07), ('mean concave points', -1.01)]

3. one-hot encoding of ['Auckland', 'Wellington', 'Auckland', 'Dunedin'] 
   ['city_Auckland', 'city_Dunedin', 'city_Wellington'] 
   [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]

4. noise only, labels random (true accuracy 50%): select-then-validate 85%   selection inside a Pipeline 48%
```

## Try it

1. Add `RandomForestClassifier` to the comparison, or any other classifier from scikit-learn, behind the same `mean_test` function. Where does it land between the tree and logistic regression?
2. In part 4 of the program, set `k=20` to `k=5` and then to `k=200`. How does the wrong-way score move?
3. Fit a `StandardScaler` on all rows before splitting, and compare the test accuracy of logistic regression with fitting it on the training rows only. The difference is small on this dataset; ask why, and why the feature-selection leak is so much larger.
