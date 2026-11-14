---
title: "ML Engineer Interviews, With Reference Answers (5/18): Data, Sampling and Features"
date: 2026-12-31 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, machine-learning, feature-engineering, data-leakage, series]
description: "Part 5 of 18. Ten interview questions on missing values, encoding, leakage, imbalance, sampling and training-serving skew, with reference answers and experiments whose code and output are shown."
---

This post covers the data questions of the technical round: how to handle missing values and categorical features, where preprocessing leaks, what to do with imbalanced or oversized data, and how training and serving drift apart. It gives ten questions with what each tests, a reference answer, a common weak answer and a follow-up. Numbers in the answers come from code that was run for this post, shown with its output. The answers are the author's, for hypothetical candidates. [Part 4](/posts/ml-engineer-interviews-4-evaluation-metrics-and-experiments/) covered evaluation.

The code ran on a CPU-only machine with Python 3.14.7 and scikit-learn 1.9.1, with fixed seeds. The data are synthetic and chosen to show a mechanism; other data will give other numbers. The main sources are the [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) and Google's [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).

![Left: cross-validated accuracy on pure noise with random labels is 0.87 when feature selection is fitted on all rows before cross-validation and 0.50 when selection sits inside the pipeline. Right: for a 2.5 percent positive problem, class weighting and random oversampling raise recall at threshold 0.5 from 0.27 to about 0.71 and raise the mean predicted score from 0.025 to about 0.37.]({{ "/assets/pic/ml-interview-5-leakage-imbalance.png" | relative_url }})
*Left: preprocessing leakage (question 5). Right: what reweighting does to recall and to the scores (question 6).*

## 1. "How do you handle missing values?"

**What it tests.** Whether the candidate treats missingness as information and keeps the fix inside the training fold.

**Reference answer.** First ask why the value is missing, because the cause decides the fix. A value missing at random can be imputed; a value missing because of the outcome carries signal that imputation erases. Options, in rising effort: let the model handle missing values natively if it can; impute with a constant, the mean or median, or a model-based imputer; and, whatever the imputation, keep a flag that says which values were imputed. The [scikit-learn guide](https://scikit-learn.org/stable/modules/impute.html) notes that when using imputation, preserving the information about which values had been missing can be informative, and that `SimpleImputer` and `IterativeImputer` have an `add_indicator` option, off by default. The imputer's statistics must be learned on the training fold only (question 5). The standard names for the missingness mechanisms are not taken from the sources read for this post.

An experiment where values go missing for half the positives and 5% of the negatives:

```python
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
rng = np.random.default_rng(0)
n = 5000; x = rng.normal(size=n); y = (x + rng.normal(0, 1, n) > 0).astype(int)
miss = rng.random(n) < np.where(y == 1, 0.5, 0.05)
xm = np.where(miss, np.nan, x).reshape(-1, 1)
print(f"missing rate: positives {miss[y==1].mean():.2f}, negatives {miss[y==0].mean():.2f}")
for add in (False, True):
    p = make_pipeline(SimpleImputer(strategy="mean", add_indicator=add), LogisticRegression())
    print(f"mean imputation, add_indicator={add}: CV accuracy {cross_val_score(p, xm, y, cv=5).mean():.3f}")
```

```
missing rate: positives 0.50, negatives 0.05
mean imputation, add_indicator=False: CV accuracy 0.778
mean imputation, add_indicator=True: CV accuracy 0.812
```

The indicator adds 3.4 points, because the fact that the value is missing predicts the label. Mean imputation alone throws that information away.

**Common weak answer.** "Fill with the mean." It does not ask why the value is missing.

**Follow-up.** "The feature is missing for 60% of rows at serving but only 5% in training. What happens?" The imputed value and the indicator both shift; this is training-serving skew (question 9).

## 2. "One-hot encoding or something else for a categorical feature?"

**What it tests.** Cardinality, model family and the cost of each encoding.

**Reference answer.** One-hot encoding turns a feature with n categories into n binary columns; the scikit-learn guide describes it as the one-of-K or dummy encoding. It is the default for low-cardinality features and linear models. For high-cardinality features such as zip code or region, one-hot inflates the feature space and makes the downstream model more expensive; the guide offers `TargetEncoder` for this case, which replaces a category with a blend of the target mean in that category and the overall target mean, weighted by how often the category occurs. Other options are grouping rare categories (`OneHotEncoder` supports `max_categories` and infrequent categories), hashing, and learned embeddings in neural models. What to say about unseen categories: decide the behaviour in advance, because a category that appears only at serving time will otherwise fail or be silently mishandled.

**Common weak answer.** "Label encode it." Integer codes impose an order that a linear model will use, and that is wrong for nominal categories.

**Follow-up.** "A tree-based model, 10,000 categories." One-hot makes sparse, deep splits; target encoding or native categorical support is cheaper, and the leakage in question 3 applies.

## 3. "What goes wrong with target encoding?"

**What it tests.** Whether the candidate sees that an encoding that uses the label leaks it into the training features.

**Reference answer.** If each category is replaced by the mean label of its training rows, the training feature contains the row's own label. For a category with few rows, the encoding is nearly the label itself, and the model learns to trust it. The scikit-learn guide states the remedy: `fit_transform` internally relies on a cross-fitting scheme to prevent target information from leaking into the train-time representation, especially for non-informative high-cardinality categorical variables, where each fold is encoded with the encodings learned from the other folds. It also notes that `fit(X, y).transform(X)` does not equal `fit_transform(X, y)`, so training data should always go through `fit_transform`.

An experiment with 2,000 categories and a label that has nothing to do with the category, so the best possible test accuracy is 0.50:

```python
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0)
n = 6000; cat = rng.integers(0, 2000, n).reshape(-1, 1); y = rng.integers(0, 2, n)
Xtr, Xte, ytr, yte = train_test_split(cat, y, test_size=0.5, random_state=0)
rf = lambda: RandomForestClassifier(200, min_samples_leaf=5, random_state=0, n_jobs=-1)
enc = TargetEncoder(random_state=0).fit(Xtr, ytr)
a, b = enc.transform(Xtr), enc.transform(Xte)
m = rf().fit(a, ytr); print(f"fit().transform(): train acc {m.score(a, ytr):.2f}, test acc {m.score(b, yte):.2f}")
enc2 = TargetEncoder(random_state=0); a2 = enc2.fit_transform(Xtr, ytr); b2 = enc2.transform(Xte)
m = rf().fit(a2, ytr); print(f"fit_transform():   train acc {m.score(a2, ytr):.2f}, test acc {m.score(b2, yte):.2f}")
```

```
fit().transform(): train acc 0.80, test acc 0.51
fit_transform():   train acc 0.52, test acc 0.50
```

With the naive encoding, the forest reaches 0.80 on training data from a feature that carries no information, and 0.51 on test data. With cross fitting, training accuracy is 0.52, which matches reality.

**Common weak answer.** "Target encoding is good for high-cardinality features." True, and incomplete.

**Follow-up.** "How would you notice this in a project?" A large gap between training and validation scores, and a high-importance feature that is a target-derived encoding.

## 4. "When does feature scaling matter?"

**What it tests.** Knowing which models are sensitive to feature scale, and why.

**Reference answer.** Scale matters for models whose objective or distance depends on feature magnitudes: regularised linear models, k-nearest neighbours, support vector machines, k-means, and neural networks trained by gradient descent. The scikit-learn guide says that if a feature has a variance that is orders of magnitude larger than the others, it might dominate the objective function and make the estimator unable to learn from the other features as expected. Regularisation makes this concrete: a penalty applied to weights is only fair across features if the features are on comparable scales. As the author's note, not a cited claim: tree-based models split on one feature at a time and are usually unaffected by monotone rescaling, which is why they are often the first model tried on unscaled tabular data. Whichever scaler is used, fit it on the training fold only and apply the same transformation to validation and serving data.

**Common weak answer.** "Always standardise." Harmless for most models, and it hides the reason.

**Follow-up.** "A feature has heavy outliers. Standardise or something else?" Standardisation is sensitive to outliers; a robust scaler or a transform of the feature is an option, chosen by looking at the distribution.

## 5. "Your pipeline scales and selects features before cross-validation. What is wrong?"

**What it tests.** Preprocessing leakage, the most common way a validation score is quietly inflated.

**Reference answer.** Any step that learns from data (scaling, imputation, feature selection, encoding, dimensionality reduction) is a model. If it is fitted on all rows before the split, the validation rows have influenced the transformation and the score is optimistic. The [scikit-learn guide to common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) defines data leakage as using information that would not be available at prediction time, lists the same rule (never call `fit` on the test data; always split first, before any preprocessing), and recommends the pipeline as the way to prevent it. A pipeline refits every step on each training fold.

The classic demonstration is feature selection on noise: 100 rows, 5,000 random features, random labels, so no model can beat 0.50:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5000)); y = rng.integers(0, 2, 100)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
Xs = SelectKBest(f_classif, k=20).fit_transform(X, y)
print(f"select on all rows, then CV: accuracy {cross_val_score(LogisticRegression(max_iter=1000), Xs, y, cv=cv).mean():.2f}")
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
print(f"selection inside the pipeline: accuracy {cross_val_score(pipe, X, y, cv=cv).mean():.2f}")
```

```
select on all rows, then CV: accuracy 0.87
selection inside the pipeline: accuracy 0.50
```

Selecting the 20 features most correlated with the labels, using all 100 labels, gives 0.87 on data with no signal. The same selection inside the pipeline gives 0.50, the truth.

**Common weak answer.** "It's only scaling, so it doesn't matter much." Scaling leaks little; selection, target encoding and imputation can leak a lot, and the habit is what matters.

**Follow-up.** "How do you guard against this in a team?" Pipelines as the only way to build a model, and a review checklist that asks, for each step, what data it was fitted on.

## 6. "The positive class is 2.5% of the data. What do you do?"

**What it tests.** Whether the candidate knows the options, and what each does to the scores.

**Reference answer.** Start by checking whether there is a problem. A model trained on the raw data ranks perfectly well, and the evaluation advice from Part 4 applies (average precision, recall at a fixed precision, a cost-based threshold). The options, in order of effort:

1. Do nothing to the data and choose the threshold from costs.
2. Weight the classes in the loss (`class_weight`).
3. Resample: oversample the minority class, undersample the majority class, or both. [Chawla et al. (2002)](https://arxiv.org/abs/1106.1813) propose SMOTE, which creates synthetic minority examples, and show that combining minority over-sampling with majority under-sampling can achieve better classifier performance, in ROC space, than under-sampling the majority class alone. This post did not run SMOTE.
4. Collect more minority examples, which is often the best option and the most expensive.

Reweighting and resampling change what the model believes the base rate is. The experiment below uses a 2.5%-positive data set and a logistic regression:

```python
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, recall_score, precision_score, brier_score_loss
X, y = make_classification(n_samples=60000, n_features=20, n_informative=6, weights=[0.98],
                           flip_y=0.01, class_sep=1.0, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
def rep(name, p):
    pred = p >= 0.5
    print(f"{name:26} AP {average_precision_score(yte, p):.3f}  recall@0.5 {recall_score(yte, pred):.3f}  "
          f"precision@0.5 {precision_score(yte, pred, zero_division=0):.3f}  mean score {p.mean():.3f}  "
          f"Brier {brier_score_loss(yte, p):.3f}")
rep("plain", LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
rep("class_weight='balanced'", LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr).predict_proba(Xte)[:, 1])
rng = np.random.default_rng(0); pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
idx = np.r_[neg, rng.choice(pos, len(neg), replace=True)]
rep("random oversampling", LogisticRegression(max_iter=1000).fit(Xtr[idx], ytr[idx]).predict_proba(Xte)[:, 1])
```

```
plain                      AP 0.527  recall@0.5 0.269  precision@0.5 0.990  mean score 0.025  Brier 0.016
class_weight='balanced'    AP 0.434  recall@0.5 0.714  precision@0.5 0.079  mean score 0.367  Brier 0.160
random oversampling        AP 0.432  recall@0.5 0.713  precision@0.5 0.079  mean score 0.366  Brier 0.160
```

At the 0.5 threshold, balancing raises recall from 0.27 to 0.71 and drops precision from 0.99 to 0.08. The average score moves from 0.025, which is the true base rate, to 0.37, so the output is no longer a probability: the Brier score goes from 0.016 to 0.160. The ranking quality, measured by average precision, did not improve in this run; it fell from 0.527 to 0.434. The practical conclusion is that for a model whose probabilities are used, thresholding is the cleaner tool, and if the data are reweighted the scores need recalibrating (Part 4, question 5).

**Common weak answer.** "Use SMOTE." It names a technique and skips the diagnosis and the effect on the probabilities.

**Follow-up.** "You oversampled, and the product needs a probability. What do you do?" Recalibrate on untouched held-out data, or correct for the sampling ratio.

## 7. "There is too much data to train on. How do you sample?"

**What it tests.** Whether subsampling preserves the quantities the model needs.

**Reference answer.** Do not drop data arbitrarily, for example by taking the first files and ignoring the rest. Google's Rules of Machine Learning says to importance-weight sampled data: if example X is sampled with probability 30%, give it a weight of 10/3, and with importance weighting the calibration properties discussed in the rules still hold. The rule adds that data that was never shown to the user can be dropped. The same logic applies to subsampling negatives to speed up training on an imbalanced problem: keep every positive, keep a fraction of the negatives, and weight the kept negatives by the inverse of the fraction.

An experiment on the 2.5%-positive data from question 6, keeping all positives and 10% of the negatives:

```python
rng = np.random.default_rng(0)
keep = (ytr == 1) | (rng.random(len(ytr)) < 0.1)
Xs, ys = Xtr[keep], ytr[keep]; w = np.where(ys == 0, 10.0, 1.0)
for name, m in (("full data", LogisticRegression(max_iter=1000).fit(Xtr, ytr)),
                ("sampled, no weights", LogisticRegression(max_iter=1000).fit(Xs, ys)),
                ("sampled, weights", LogisticRegression(max_iter=1000).fit(Xs, ys, sample_weight=w))):
    p = m.predict_proba(Xte)[:, 1]
    print(f"{name:20} AP {average_precision_score(yte, p):.3f}  mean score {p.mean():.3f}  Brier {brier_score_loss(yte, p):.4f}")
```

```
full data            AP 0.527  mean score 0.025  Brier 0.0164
sampled, no weights  AP 0.499  mean score 0.146  Brier 0.0391
sampled, weights     AP 0.528  mean score 0.025  Brier 0.0164
```

Training on 12% of the rows with weights reproduces the full-data model: the same average precision, the same mean score and the same Brier score. Without the weights, the mean score is 0.146 against a true 0.025 and the Brier score more than doubles.

**Common weak answer.** "Take a random 10%." For a balanced problem that may be fine; for a rare class it removes the positives, and without weights it shifts the probabilities.

**Follow-up.** "The data are logs from many users. Random rows or random users?" It depends on what the model must generalise over; sampling by user keeps each user's rows together, which also matters for the splitting in question 8.

## 8. "How do you split the data into train, validation and test?"

**What it tests.** Whether the split reflects how the model will be used, and whether rare classes survive it.

**Reference answer.** Three rules. First, split before any preprocessing (question 5). Second, keep class proportions in each split when the positive class is rare: the [scikit-learn guide](https://scikit-learn.org/stable/modules/cross_validation.html) says `cross_val_score` uses `StratifiedKFold` by default for classifiers when `cv` is an integer, and stratification keeps the class ratio in each fold. Third, split along the axis that matches deployment: by time for a model that predicts the future, by group when rows from one entity must not straddle the split (Part 4, question 7). Google's Rules of Machine Learning says it directly: if the model is built on data until January 5th, test it on data from January 6th and after, because that better reflects what the system will do in production. The rule adds that performance should be expected to be somewhat worse on the new data, but not radically worse, and that the area under the curve should stay reasonably close even when daily effects move the average click rate.

A small case: 200 rows with 10 positives, sorted by label, five folds:

```python
from sklearn.model_selection import KFold
y = np.zeros(200, int); y[:10] = 1
for name, cv in (("KFold(5), no shuffle", KFold(5)),
                 ("KFold(5, shuffle)", KFold(5, shuffle=True, random_state=0)),
                 ("StratifiedKFold(5)", StratifiedKFold(5, shuffle=True, random_state=0))):
    print(f"{name:22} positives per test fold: {[int(y[te].sum()) for _, te in cv.split(y.reshape(-1,1), y)]}")
```

```
KFold(5), no shuffle   positives per test fold: [10, 0, 0, 0, 0]
KFold(5, shuffle)      positives per test fold: [3, 1, 2, 3, 1]
StratifiedKFold(5)     positives per test fold: [2, 2, 2, 2, 2]
```

Unshuffled k-fold on ordered data puts every positive in one test fold, so four of five folds cannot compute recall at all. Shuffling gives uneven counts; stratification gives exactly two per fold.

**Common weak answer.** "80/20 random split." It does not ask about time, groups or class balance.

**Follow-up.** "How big should the test set be?" Large enough that the metric's uncertainty is smaller than the differences you want to detect; for a rare class, the number of positives matters more than the number of rows.

## 9. "What is training-serving skew, and how do you prevent it?"

**What it tests.** Operational awareness: the same feature must mean the same thing in both places.

**Reference answer.** Skew is a difference between how features are computed or what data are seen at training time and at serving time. Google's Rules of Machine Learning gives a three-part decomposition: the gap between training and holdout performance (always exists, not always bad); the gap between holdout and next-day data (large drops may indicate time-sensitive features); and the gap between next-day data and live data, which for the same example should give exactly the same score, so a discrepancy there probably indicates an engineering error. Prevention, from the same source:

- **Log the features at serving time** and use the logged features for training. The rules say this is the best way to train like you serve, that teams which measured it were sometimes surprised, and that doing it even for a small fraction of examples allows the consistency to be checked.
- **Reuse code** between the training and serving pipelines where possible, because batch and online processing differ.
- **Watch joins against changing tables:** features such as comment counts may change between training and serving, so snapshot the table or log the features. The rules add that a snapshot still does not completely resolve the issue.
- **Measure the skew** as a standing metric, and test the infrastructure independently from the model.

**Common weak answer.** "Monitor the model's accuracy." Accuracy needs labels, which arrive late; feature-level consistency checks catch skew earlier.

**Follow-up.** "The offline AUC is 0.9 and the live AUC is 0.7. Where do you look first?" The third gap in the decomposition: score the same logged examples in both environments, and compare feature values one by one.

## 10. "How do you approach feature engineering?"

**What it tests.** Judgement about which features to add, keep and delete, and the feedback loops that features can create.

**Reference answer.** Several points from Google's Rules of Machine Learning, which the author finds a sound checklist:

- Start with directly observed and reported features, not learned ones. A feature produced by another system has that system's objective, which may be only weakly correlated with yours, and its meaning can change when it is updated (rule 17).
- Combine and modify existing features in ways people can understand (rule 20).
- The number of feature weights a linear model can learn is roughly proportional to the amount of data (rule 21).
- Clean up features no longer used, because unused features are technical debt (rule 22).
- Look for patterns in the model's errors and create new features from them (rule 26).
- Beware feedback loops with positional features: if the model learns that first position gets clicks, and the position is not available when candidates are scored, training and serving differ, so keep positional features separate from the rest of the model (rule 36).

Add the points from this post: a feature must be available at prediction time with the same definition (questions 8 and 9), and a feature derived from the label must be cross-fitted (question 3).

**Common weak answer.** "Create as many features as possible and let the model pick." That is how leakage and skew enter.

**Follow-up.** "You have a feature that improves validation AUC by 0.02 and takes a week to build in production. What do you do?" Estimate the value against the cost, check that it is available at serving time, and consider whether the gain survives a time-based split.

## Takeaways

- Ask why a value is missing before choosing how to fill it, and keep the indicator.
- Anything that learns from data belongs inside the training fold, which a pipeline guarantees.
- Reweighting and resampling change the scores; if the probability matters, correct or recalibrate.
- Subsample with importance weights, not by discarding data.
- Log features at serving time; most training-serving skew is an engineering error.

## Limits

- The experiments use synthetic data and one seed each; they show mechanisms, not general effect sizes.
- SMOTE and the other resampling methods are described from the original paper's abstract and were not run here.
- Hashing and embeddings for categorical features, the scaling behaviour of tree models and the names of missingness mechanisms come from general practice, not from the sources read for this post.
- Image, text and time-series feature pipelines are not covered.
- The reference answers describe one reasonable answer. Interviewers differ, and the candidates in the examples are hypothetical.

Next: Part 6, on deep learning and training.
