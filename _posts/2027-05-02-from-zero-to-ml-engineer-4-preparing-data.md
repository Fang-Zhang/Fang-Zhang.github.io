---
title: "From Zero to ML Engineer, Part 4: Preparing Data and Keeping the Test Honest"
date: 2027-05-02 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, data leakage, train test split, validation, ML engineer]
description: "Part 4 of a 12-part series for complete beginners. How to split data into train, validation and test sets, and two ways a split quietly stops being honest: twin rows on both sides, and picking the best of many models on the test set. Both are reproduced with code."
---

## What you will do, and what you get

You will split a dataset into three parts with a clear job for each, and learn to recognise two leaks that make a model look better than it is. After this post you can read any reported accuracy and ask the right question: *what had the model seen when it was scored?* The program reproduces both leaks: one makes the same model score 5.8 points better, the other makes a coin-flipping model look like it knows something.

![Three panels. Left: a bar split into train 60% (341 rows), validation 20% (114 rows) and test 20% (114 rows). Middle: accuracy of 1-nearest-neighbour is 91.7% when rows are unique and 97.5% when every row appears twice. Right: histogram of 200 random models on the test rows, the best scoring 62% against 53% for the same model on fresh rows]({{ site.baseurl }}/assets/pic/mlpath-4-splits.png){: width="900" }

## How: the steps

1. **Clean first, split second.** Remove exact duplicate rows, decide how to treat missing values, and fix obviously wrong entries. Do anything that *learns from the data* (such as computing averages) only after the split, on the training part.
2. **Split into three parts.** *Training* (about 60%): the model learns from it. *Validation* (about 20%): you compare models and settings on it. *Test* (about 20%): you score the final choice on it, once.
3. **Keep the class mix the same in all parts.** `stratify=y` does it.
4. **Make every decision on train and validation only.** That includes which model, which settings, which columns to use.
5. **Open the test set once.** If the test score disappoints, resist the urge to adjust and test again; that turns the test set into a second validation set.

## Why it works: the principle

The [scikit-learn guide](https://scikit-learn.org/stable/modules/cross_validation.html) states the reason in one sentence: "Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data." That is what the red bar in Part 1 showed. The same guide explains why a validation set exists: if you tune settings until the test score is good, "knowledge about the test set can 'leak' into the model and evaluation metrics no longer report on generalization performance."

Both are examples of **data leakage**, defined in the [scikit-learn pitfalls page](https://scikit-learn.org/stable/common_pitfalls.html) as when "information that would not be available at prediction time is used when building the model", giving "overly optimistic performance estimates". The test set is only an honest stand-in for the future if nothing about it has influenced the model. Two ways it gets influenced:

**Leak 1: the same row on both sides.** Real data often contains duplicates or near-duplicates: an event logged twice, the same customer in two exports. A random split puts one copy in training and the other in testing, so the model is tested on rows it has memorised. In the program, every row of the tumour data is duplicated before the split: 1-nearest-neighbour scores 97.5% instead of 91.7%, with the same model and the same underlying data. Nothing about the model improved; only the test got easier. (It does not reach 100% because both copies of some rows land in the test set.) The defence is step 1: deduplicate before splitting, and when rows belong to a group (the same patient, the same customer), keep each group entirely on one side.

**Leak 2: choosing the winner on the test set.** The program creates 200 "models" that guess at random on 100 coin-flip labels. Each has a true accuracy of 50%, yet the best of them scores 62% on the test rows, because the best of 200 noisy numbers is high by chance. On 100 fresh rows, the same model scores 53%. A hundred test rows give a standard deviation of 5 points, so the best of 200 sits about two and a half standard deviations above 50. This happens with real models too, with less obvious numbers: every setting you try and compare on the same rows is one more draw. That is why the validation set takes the repeated comparisons, and the test set sees one final model.

## Example, worked through

Reading the output below:

- **Sizes:** 341 train, 114 validation and 114 test rows, with 37.2%, 37.7% and 36.8% malignant: the class mix is preserved to within a point.
- **Leak 1:** 91.7% on a split of unique rows, 97.5% after duplicating, over 20 random splits each.
- **Leak 2:** 62% on the test rows for the best random model, 53% on fresh ones, where the truth is 50%.

A rule of thumb that falls out of this: when a result looks surprisingly good, check for leakage before celebrating. A second kind of leak, in which the preprocessing step is fitted on all rows before splitting, is handled by the Pipeline tool in Part 5.

## Implementation

Save as `part4.py` and run it. It needs numpy and scikit-learn.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True)

# 1. The three-way split: 60% train, 20% validation (choose between models), 20% test (look once, at the end)
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=0)
for name, part in (("train", y_train), ("validation", y_val), ("test", y_test)):
    print(f"{name:10s} {len(part):3d} rows, {100 * (part == 0).mean():.1f}% malignant")

# 2. Leak 1: duplicate rows on both sides of the split
honest, leaked = [], []
for seed in range(20):
    a, b, ya, yb = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    honest.append(KNeighborsClassifier(1).fit(a, ya).score(b, yb))
    X2, y2 = np.vstack([X, X]), np.r_[y, y]                      # every row logged twice, as happens in real event data
    a, b, ya, yb = train_test_split(X2, y2, test_size=0.3, random_state=seed)
    leaked.append(KNeighborsClassifier(1).fit(a, ya).score(b, yb))
print(f"\n1-nearest-neighbour, split of unique rows:     {100 * np.mean(honest):.1f}%")
print(f"1-nearest-neighbour, split after duplicating:  {100 * np.mean(leaked):.1f}%   (same model, same data)")

# 3. Leak 2: choosing the best of many models on the test set
rng = np.random.default_rng(0)
labels = rng.integers(0, 2, 100)                                 # 100 test rows with coin-flip labels: no model can know them
fresh = rng.integers(0, 2, 100)                                  # 100 new rows with coin-flip labels
guesses = rng.integers(0, 2, (200, 100))                         # 200 "models" that guess at random
test_acc = (guesses == labels).mean(1); fresh_acc = (guesses == fresh).mean(1)
best = test_acc.argmax()
print(f"\n200 random models, best accuracy on the test rows: {100 * test_acc.max():.0f}%   the same model on 100 fresh rows: {100 * fresh_acc[best]:.0f}%")
```

Output:

```
train      341 rows, 37.2% malignant
validation 114 rows, 37.7% malignant
test       114 rows, 36.8% malignant

1-nearest-neighbour, split of unique rows:     91.7%
1-nearest-neighbour, split after duplicating:  97.5%   (same model, same data)

200 random models, best accuracy on the test rows: 62%   the same model on 100 fresh rows: 53%
```

## Try it

1. Change `random_state` in the first split. Do the class shares and sizes change?
2. In leak 2, raise the number of models from 200 to 2000, then lower it to 5. How does the best test score move, and what does that say about how many settings you can compare on one test set?
3. Make the test set 1,000 rows instead of 100 in leak 2. How much smaller is the inflated score?
