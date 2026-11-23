---
title: "From Zero to ML Engineer, Part 6: Is the Model Any Good?"
date: 2027-05-04 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, evaluation, precision, recall, cross-validation, overfitting, ML engineer]
description: "Part 6 of a 12-part series for complete beginners. Baselines, the confusion matrix, precision and recall, moving the decision threshold, cross-validation, and reading the gap between training and validation scores to tell underfitting from overfitting."
---

## What you will do, and what you get

You will take the logistic regression from Part 5 and judge it properly. Instead of one accuracy number, you will have a baseline to beat, a confusion matrix, precision and recall, a threshold chosen on purpose, and a cross-validated score with a spread. The result is an answer to "is it good?" that depends on what a mistake costs. On the tumour test rows, 95.3% accuracy hides a more specific fact: 4 malignant cases missed and 4 healthy cases flagged.

![Three panels. Left: confusion matrix with 103 true negatives, 4 false positives, 4 false negatives and 60 true positives. Middle: precision and recall against decision threshold; lowering the threshold from 0.9 to 0.02 raises recall from 87.5 to 100 percent and drops precision from 100 to 68.1 percent. Right: decision tree train accuracy climbing to 100 percent with depth while cross-validated accuracy stays near 92.6 percent]({{ site.baseurl }}/assets/pic/mlpath-6-evaluation.png){: width="900" }

## How: the steps

1. **Set a baseline.** Score a model that ignores the columns (always predict the majority class). Any real model must clear this bar.
2. **Count the four outcomes** in a confusion matrix: true positives (TP), false positives (FP), true negatives (TN), false negatives (FN). "Positive" means the case you care about finding, here malignant.
3. **Compute the metrics that match the cost of mistakes.** Accuracy, precision, recall.
4. **Choose the threshold** on the validation rows, not on the test rows.
5. **Cross-validate** to get a score with a spread, instead of one draw.
6. **Compare training score to validation score.** A large gap means overfitting; two low scores mean underfitting.

## Why it works: the principle

**A baseline gives a number its meaning.** In this data 62.6% of the rows are benign. The model that always says "benign" scores 62.6% accuracy and finds none of the malignant cases (recall 0%). Against that, 95.3% is a real improvement, and 62.6% is the floor. The [scikit-learn DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) exists for this. On a dataset that is 99% one class, a model with 99% accuracy has learned nothing.

**Accuracy counts all mistakes as equal; the confusion matrix does not.** Missing a malignant case (FN) and raising a false alarm (FP) are different mistakes with different costs. Two ratios separate them:

- **Precision** = TP / (TP + FP): when the model says "malignant", how often it is right.
- **Recall** = TP / (TP + FN): of all the malignant cases, how many it finds.

The [scikit-learn model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html) lists these and many more.

**The threshold is a dial.** Logistic regression outputs a probability, and the 0.5 cut-off is a convention. Lower it and the model flags more cases: recall rises and precision falls, as the middle panel shows. At a threshold of 0.02, no malignant case is missed and 30 healthy ones are flagged. At 0.7, there are no false alarms and 5 cases are missed. Which point is right depends on the cost of each mistake, which is a decision about the problem rather than something the data decides. (The program sweeps thresholds on the test rows only to draw the figure. In a real project, make this choice on the validation rows; otherwise it is the test-set peeking from Part 4.)

**One split is one draw.** Across 20 different random splits of the same data, logistic regression scores anywhere from 95.3% to 98.8%. A single split could report either end. *Cross-validation* uses every row for testing exactly once: it cuts the data into 5 parts, trains on 4 and tests on the fifth, and rotates. The five scores, with their mean and spread, are a more honest report: 97.9% with a standard deviation of 1.4 points. If two models differ by less than that spread, the data cannot tell them apart.

**Underfitting and overfitting** are read from the train-versus-validation gap. A decision tree of depth 1 scores 92.7% on training rows and 89.6% on held-out rows: both are low, so the model is too simple (underfitting). By depth 7 the training score is 100% while the held-out score has stayed around 92.5%: the model has memorised the training rows, and the extra depth buys nothing on new ones (overfitting). The best depth here is about 5, where the held-out score peaks at 93.8%, though differences of a point or so are within the noise of cross-validation. The [learning curve guide](https://scikit-learn.org/stable/modules/learning_curve.html) describes this diagnostic.

## Example, worked through

- **Baseline:** always "benign" gives 62.6% accuracy and 0% recall.
- **Logistic regression on the 171 test rows:** TN=103, FP=4, FN=4, TP=60. Accuracy (60+103)/171 = 95.3%. Precision 60/64 = 93.8%. Recall 60/64 = 93.8%.
- **Raising recall:** at threshold 0.1, recall is 98.4% (1 missed case) at the price of 14 false alarms and an 81.8% precision. If follow-up checking is cheap and a miss is serious, that trade is reasonable. If every flag costs a lot, 0.7 is better.
- **Spread:** the five cross-validation scores are 95.6, 97.4, 98.2, 100.0 and 98.2.
- **The gap:** depth 1: 92.7 vs 89.6; depth 5: 99.3 vs 93.8; depth 12: 100.0 vs 92.6. Training accuracy only goes up; the held-out accuracy plateaus.

## Implementation

Save as `part6.py` and run it. It needs numpy and scikit-learn.

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

X, y = load_breast_cancer(return_X_y=True)
y = 1 - y                                   # now 1 = malignant, the case we must not miss
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xa, ya)

# 1. A baseline that never looks at the columns
dummy = DummyClassifier(strategy="most_frequent").fit(Xa, ya); pd_ = dummy.predict(Xt)
print(f"1. baseline 'always benign': accuracy {100*accuracy_score(yt, pd_):.1f}%, recall of malignant {100*recall_score(yt, pd_):.0f}%")

# 2. The confusion matrix and what it contains
p = model.predict(Xt); tn, fp, fn, tp = confusion_matrix(yt, p).ravel()
print(f"\n2. logistic regression on {len(yt)} test rows: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"   accuracy  = (TP+TN)/all = {100*(tp+tn)/len(yt):.1f}%")
print(f"   precision = TP/(TP+FP)  = {tp}/{tp+fp} = {100*tp/(tp+fp):.1f}%   (when it says malignant, how often is it right)")
print(f"   recall    = TP/(TP+FN)  = {tp}/{tp+fn} = {100*tp/(tp+fn):.1f}%   (of the malignant cases, how many it finds)")

# 3. Moving the decision threshold trades one for the other
prob = model.predict_proba(Xt)[:, 1]
print("\n3. threshold on the malignant probability")
for t in (0.9, 0.7, 0.5, 0.3, 0.1, 0.02):
    q = (prob >= t).astype(int); print(f"   threshold {t:4}: precision {100*precision_score(yt, q):5.1f}%  recall {100*recall_score(yt, q):5.1f}%  missed {int(((q==0)&(yt==1)).sum())}  false alarms {int(((q==1)&(yt==0)).sum())}")

# 4. One split is one draw: spread over 20 splits versus 5-fold cross-validation
single = [model.fit(*train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)[::2]).score(*train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)[1::2]) for s in range(20)]
print(f"\n4. accuracy over 20 different splits: min {100*min(single):.1f}%  max {100*max(single):.1f}%  (a single split could report anything in this range)")
cv = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
print(f"   5-fold cross-validation: {[round(float(100*c),1) for c in cv]}  mean {100*cv.mean():.1f}%  std {100*cv.std():.1f}")

# 5. Under- and over-fitting: train score versus cross-validated score as the tree gets deeper
depths = np.arange(1, 13)
tr, va = validation_curve(DecisionTreeClassifier(random_state=0), X, y, param_name="max_depth", param_range=depths, cv=StratifiedKFold(5, shuffle=True, random_state=0))
print("\n5. decision tree, depth: train accuracy / cross-validated accuracy")
for d, a, b in zip(depths, tr.mean(1), va.mean(1)): print(f"   depth {d:2d}: {100*a:5.1f}% / {100*b:5.1f}%")
```

Output:

```
1. baseline 'always benign': accuracy 62.6%, recall of malignant 0%

2. logistic regression on 171 test rows: TN=103 FP=4 FN=4 TP=60
   accuracy  = (TP+TN)/all = 95.3%
   precision = TP/(TP+FP)  = 60/64 = 93.8%   (when it says malignant, how often is it right)
   recall    = TP/(TP+FN)  = 60/64 = 93.8%   (of the malignant cases, how many it finds)

3. threshold on the malignant probability
   threshold  0.9: precision 100.0%  recall  87.5%  missed 8  false alarms 0
   threshold  0.7: precision 100.0%  recall  92.2%  missed 5  false alarms 0
   threshold  0.5: precision  93.8%  recall  93.8%  missed 4  false alarms 4
   threshold  0.3: precision  85.9%  recall  95.3%  missed 3  false alarms 10
   threshold  0.1: precision  81.8%  recall  98.4%  missed 1  false alarms 14
   threshold 0.02: precision  68.1%  recall 100.0%  missed 0  false alarms 30

4. accuracy over 20 different splits: min 95.3%  max 98.8%  (a single split could report anything in this range)
   5-fold cross-validation: [95.6, 97.4, 98.2, 100.0, 98.2]  mean 97.9%  std 1.4

5. decision tree, depth: train accuracy / cross-validated accuracy
   depth  1:  92.7% /  89.6%
   depth  2:  95.0% /  91.7%
   depth  3:  97.5% /  93.2%
   depth  4:  98.3% /  92.6%
   depth  5:  99.3% /  93.8%
   depth  6:  99.6% /  92.6%
   depth  7: 100.0% /  92.3%
   depth  8: 100.0% /  92.6%
   depth  9: 100.0% /  92.6%
   depth 10: 100.0% /  92.6%
   depth 11: 100.0% /  92.6%
   depth 12: 100.0% /  92.6%
```

## Try it

1. Swap the decision tree in part 5 of the program for logistic regression with `C` from 0.001 to 1000 (smaller `C` means stronger regularisation). Which side of the curve is under- and which is overfitting?
2. Replace the tumour labels with a 95/5 imbalance by dropping most malignant rows. What does the baseline score now, and which metric still tells the truth?
3. Pick the threshold that gives at most 2 missed cases using only the training rows (via cross-validation), then report the test result once.
