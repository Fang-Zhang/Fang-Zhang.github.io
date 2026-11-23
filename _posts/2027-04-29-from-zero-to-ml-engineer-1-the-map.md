---
title: "From Zero to ML Engineer, Part 1: The Map"
date: 2027-04-29 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, roadmap, scikit-learn, python, ML engineer]
description: "Part 1 of a 12-part series that takes a complete beginner to the point of carrying a small machine learning project from data to a served model. This post draws the map and runs a 20-line program that learns from examples, is scored on rows it has never seen, and makes a prediction."
---

## What you will do, and what you get

By the end of this post you will have run a short Python program that learns to tell malignant from benign tumours using 569 labelled examples, and scores about 97% on rows it was never shown. You will also know what an ML engineer does day to day, and what the next eleven posts cover. The tumour data is a teaching dataset for practising the method; nothing here is medical advice.

![Left: the loop of an ML project, data split into a training set and a hidden test set, a model trained on the training set, evaluated on the test set and used to predict new rows. Right: accuracy of five approaches, always guessing benign 62.6%, a hand-written rule 89.1%, one-nearest-neighbour scored on test rows 91.7%, logistic regression 97.4%, and one-nearest-neighbour scored on its own training rows 100%]({{ site.baseurl }}/assets/pic/mlpath-1-map.png){: width="900" }

## How: the steps

An ML engineer's work is a loop with five steps. Every post in this series is a closer look at one part of it.

1. **Get data.** Rows of examples; each row has input columns (called *features*) and, for supervised learning, the answer (called the *label*). Here: 569 tumours, 30 measurements each, and a label saying malignant or benign.
2. **Split it.** Keep a share of the rows, here 30%, hidden in a *test set*. The rest is the *training set*.
3. **Train.** A learning algorithm reads the training set and adjusts the numbers inside a *model* (its *parameters*) so that its answers match the labels.
4. **Evaluate.** Ask the trained model for answers on the hidden test rows and count how many are right. This number, *accuracy*, is your estimate of how it will do on new data.
5. **Predict.** Use the model on rows whose answer nobody knows yet.

## Why it works: the principle

In ordinary programming you write the rules. In machine learning you supply examples and a method for finding the rules: the algorithm searches for parameter values that make the model's answers agree with the labels on the training set, and the result is a function you can call on new rows.

The one idea that separates machine learning from fitting a curve is **generalisation**: the model is useful only if it works on rows it did not train on. This is why step 2 exists. A model can score perfectly on its training rows simply by memorising them, which says nothing about new rows. The red bar in the figure shows it: a one-nearest-neighbour model (it answers with the label of the single most similar training row) is right on 100% of the rows it has been given, and on only 91.7% of the rows it has not. Only the second number tells you anything.

The second idea is the **baseline**: always compare with something trivial. Here 37% of the rows are malignant, so a "model" that always answers benign is right 62.6% of the time without learning anything. A model at 70% would look respectable until you saw that.

The scale of the job is worth knowing too. A well-known paper on machine learning in production, [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html), warns that "it is dangerous to think of these quick wins as coming for free" and that "it is common to incur massive ongoing maintenance costs in real-world ML systems". Training the model is the quick part; the rest of this series is about the other parts.

## An example, worked through

The data is the Wisconsin breast cancer diagnostic dataset: [scikit-learn ships a copy](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) with 569 samples (212 malignant, 357 benign) and 30 numeric features per sample, originally from the [UCI repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). Four approaches were scored on 20 different random 70/30 splits, and the mean accuracy on the hidden rows is reported:

- **Always guess benign: 62.6%.** The baseline.
- **A hand-written rule: 89.1%.** "If the mean radius of the cell nuclei is above 15, call it malignant." A person looked at one column and picked a threshold. Writing rules works, and it stops at what one person can see in one column.
- **Logistic regression: 97.4%.** A model that learns a weight for each of the 30 columns. Part 5 explains it.
- **One-nearest-neighbour: 91.7%** on hidden rows, and 100% on its own training rows.

Learning from the data beat the hand-written rule by about eight points, and the same data looked 8 points better when scored the wrong way. Repeating over 20 splits matters too: any single split is a little lucky or unlucky, and the standard deviation across splits was about one to two points.

## The implementation

Install the two libraries once with `pip install numpy scikit-learn`, save the program as `part1.py` and run it. It needs no GPU, no account and no download.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True)            # 569 rows, 30 measurements per row; y: 0 = malignant, 1 = benign
print("rows, columns:", X.shape)

scores = {"always guess benign": [], "hand-written rule": [], "logistic regression": [],
          "1-nearest-neighbour (test)": [], "1-nearest-neighbour (train)": []}
for seed in range(20):                                # repeat with 20 different random splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    scores["always guess benign"].append(DummyClassifier(strategy="most_frequent").fit(X_train, y_train).score(X_test, y_test))
    scores["hand-written rule"].append(((X_test[:, 0] <= 15).astype(int) == y_test).mean())   # column 0 is mean radius
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    scores["logistic regression"].append(model.score(X_test, y_test))
    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    scores["1-nearest-neighbour (test)"].append(knn.score(X_test, y_test))
    scores["1-nearest-neighbour (train)"].append(knn.score(X_train, y_train))

for name, s in scores.items():
    print(f"{name:30s} accuracy {100 * np.mean(s):5.1f}%")

new_row = X_test[:1]                                  # predict: one row the model has not been trained on
print("prediction for one unseen row:", ["malignant", "benign"][model.predict(new_row)[0]], " true:", ["malignant", "benign"][y_test[0]])
```

Output:

```
rows, columns: (569, 30)
always guess benign            accuracy  62.6%
hand-written rule              accuracy  89.1%
logistic regression            accuracy  97.4%
1-nearest-neighbour (test)     accuracy  91.7%
1-nearest-neighbour (train)    accuracy 100.0%
prediction for one unseen row: benign  true: benign
```

Four lines do the real work: `train_test_split` (step 2), `.fit` (step 3), `.score` (step 4) and `.predict` (step 5). The `StandardScaler` puts all 30 columns on the same scale before logistic regression sees them; Part 5 explains why that matters. The `stratify=y` argument keeps the share of malignant rows the same in both parts of the split.

## What the series covers

1. **This post:** the map and the first working loop.
2. Tools and the first data: reading and inspecting a dataset with numpy and pandas.
3. What learning is: a model written from scratch, with its parameters, loss and gradient descent.
4. Preparing data: cleaning, the three-way split and data leakage.
5. Features and the first real models.
6. Is the model good: baselines, metrics, over- and underfitting.
7. Making it better: error analysis, tuning, forests and boosting.
8. Neural networks, from one neuron to a small network.
9. Repeatable experiments.
10. A model as a service, and watching it after release.
11. A complete small project.
12. Where to go next.

## Try it

1. Change `test_size=0.3` to `0.1` and then `0.5`. Which scores change, and does the 1-nearest-neighbour train score move?
2. Change the hand-written rule's threshold from 15 to 12 and then 18. Which value scores best, and why is picking it by looking at the test rows a mistake? (Part 4 returns to this.)
3. Replace `LogisticRegression` with `KNeighborsClassifier(n_neighbors=15)` and compare.
