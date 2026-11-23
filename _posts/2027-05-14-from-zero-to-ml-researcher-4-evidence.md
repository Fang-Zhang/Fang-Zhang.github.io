---
title: "From Zero to ML Researcher, Part 4: Evidence"
date: 2027-05-14 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, evaluation, baselines, held-out data, statistics, ML researcher]
description: "Part 4 of a 12-part series for complete beginners. Four habits turn a number into evidence: compare with a baseline, judge on held-out data, run a sanity check that catches memorisation, and never pick the best of many on one test set. The post also shows how to tell a real difference from noise."
---

## What you will do, and what you get

You will run four small experiments that show how a good-looking number can mean nothing. A model that memorises its training images scores 100%, and it still scores 100% when the training labels are shuffled and there is nothing to learn. The best of 200 coin-flipping "models" scores 61% on a small test set, and 50.2% on new rows. A difference of 0.94 points between two methods turns out to be real when measured on 30 paired splits. You will leave with a checklist that every later part uses: baseline, held-out data, a sanity check, and a paired comparison with an interval.

![Left: train and test accuracy for four settings. The nearest-average model scores 91 on train and 89 on test; the nearest-neighbour model scores 100 and 98; with shuffled labels the nearest-average scores 16 and 7 and the nearest-neighbour scores 100 on train and 10 on test; a dashed line marks the majority baseline at 9.3 percent. Middle: a histogram of 200 coin-flipping models on 100 test rows, centred on 50 percent, with the best at 61 percent, and a line at 50.2 percent for the same kind of model on 100,000 new rows. Right: 30 grey lines connecting the accuracy of the median-average and mean-average models on the same 30 splits; the mean-average is higher on every split]({{ site.baseurl }}/assets/pic/mlr-4-evidence.png){: width="900" }

## How: the steps

1. **Compute a baseline first:** the score of something trivial (always predict the most common class). Any model must beat it by a margin that means something.
2. **Split the data** into rows the model learns from and rows it is judged on, and report the held-out score.
3. **Run a sanity check:** train the same model on shuffled labels. If it still scores well on the training rows, then a high training score proves nothing about learning.
4. **Choose among models on separate data** from the data used for the final number. Do not try many models and report the best on a single test set.
5. **Compare two methods on the same splits,** many splits, and report the average difference with an interval.

## Why it works: the principle

**A score is meaningful only relative to something.** On the digits data, always predicting the most common digit scores 9.3%. That is the floor: the nearest-average model of Part 1 scores 89.2% on held-out rows, which is far above it.

**Training accuracy measures memory; held-out accuracy measures generalisation.** The 1-NN model predicts the label of the single closest training image. On its own training rows the closest image is itself, so it scores 100% always. On held-out rows it scores 98.4%, and that is the honest number. (This model also answers a question left open in Part 1: the one-average-per-digit model was too simple, since keeping all 1,000 images lifts accuracy from 89.2% to 98.4%.)

**The shuffled-label check.** Shuffle the training labels: now the images and labels are unrelated, so nothing can be learned, and the right held-out accuracy is chance (about 10%). 1-NN still scores 100% on its training rows (it memorises the noise) and 10.0% on held-out rows. The nearest-average model, which has only ten averages to store and so cannot memorise, scores 16.5% on train and 6.6% on test. The same trick, at a much larger scale, was used by [Zhang and colleagues (2017)](https://arxiv.org/abs/1611.03530) to show that large neural networks can fit random labels perfectly: a model that fits the training set has not necessarily learned anything.

**Selecting on the test set leaks it.** Take 200 models that are pure coin flips, and a test set of 100 rows. Each scores about 50%, with a typical spread of 5 points, and the best of 200 reaches 61%. Choosing it and reporting 61% reports luck. A fresh coin-flipper on 100,000 new rows scores 50.2%. In real work the "models" are hyperparameter settings or ideas, and the test set is reused each time one is tried; the same effect applies in a milder form. The remedy is to choose on a *validation* set (or by cross-validation, described in the [scikit-learn guide](https://scikit-learn.org/stable/modules/cross_validation.html)) and touch the test set once at the end; the library lists related traps in its [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

**Noise, and the paired comparison.** One split gives one number, and its value depends on which rows landed where. Over 30 random splits the nearest-average model scores 89.97% with a standard deviation of 0.85 points, and the median variant 89.03% with 0.79. The two ranges overlap heavily, so unpaired, a gap of 0.94 would look like noise. But both methods were scored on *the same* splits, and the gap on each split is what matters: the mean-based model wins on all 30 splits, and the average gap is +0.94 points with a bootstrap 95% interval of +0.73 to +1.16. (A bootstrap resamples the 30 paired gaps with replacement many times, and reports the middle 95% of the averages.) The first split alone said +1.38. That was a fair report of that split and an unreliable guess of the true gap.

**Pitfalls and limits.**
- The 30 splits come from the same 1,797 images, so they are not independent. The interval says how much the result moves across splits, not how it would move on a different dataset.
- "Wins on 30 of 30" would not survive a change of dataset, task or model size. Say what was tested.
- A baseline is not always "most common class". For a regression task it can be the mean of the training targets; for a time series, "same as yesterday".
- Passing the shuffled-label check is necessary, not sufficient. It rules out one failure.

## Example, worked through

1. **Baseline.** The most common training digit is predicted for every test image: 9.3% correct.
2. **Held-out.** Nearest-average: train 91.1%, test 89.2%. 1-NN: train 100.0%, test 98.4%. The small gap between train and test for the first says it is not memorising; the second memorises but also generalises.
3. **Sanity check.** Shuffle the labels of the 1,000 training rows. 1-NN: train 100.0%, test 10.0%. The 100% on train is not evidence of learning.
4. **Selection.** 200 coin-flippers, 100 test rows: median 50%, best 61%. Fresh data: 50.2%.
5. **Noise.** 30 splits, same splits for both methods: mean-average 89.97 +- 0.85, median-average 89.03 +- 0.79, paired gap +0.94 (95% interval +0.73 to +1.16).

## Implementation

Save as `part4.py` and run it (a few seconds on a CPU). It needs numpy and scikit-learn (only for loading the data).

```python
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

def split(seed, n_train=1000):
    order = np.random.default_rng(seed).permutation(len(y)); return order[:n_train], order[n_train:]

def sqdist(A, B):                                   # squared distances between every row of A and every row of B
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T

def fit_centroid(Xtr, ytr, how="mean"):
    f = np.mean if how == "mean" else np.median
    return np.array([f(Xtr[ytr == d], axis=0) for d in range(10)])

def predict_centroid(c, A): return sqdist(A, c).argmin(1)
def predict_1nn(Xtr, ytr, A): return ytr[sqdist(A, Xtr).argmin(1)]
acc = lambda p, t: 100 * (p == t).mean()

# 1. A baseline, and held-out data. Memorising is easy; generalising is the question.
tr, te = split(0); c = fit_centroid(X[tr], y[tr])
majority = np.bincount(y[tr]).argmax()
print("1. baseline and held-out accuracy")
print(f"   always predict the most common digit : test {acc(np.full(len(te), majority), y[te]):.1f}%")
print(f"   nearest average (Part 1)            : train {acc(predict_centroid(c, X[tr]), y[tr]):.1f}%  test {acc(predict_centroid(c, X[te]), y[te]):.1f}%")
print(f"   nearest single training image (1-NN): train {acc(predict_1nn(X[tr], y[tr], X[tr]), y[tr]):.1f}%  test {acc(predict_1nn(X[tr], y[tr], X[te]), y[te]):.1f}%")

# 2. A sanity check that catches memorisation: shuffle the training labels so that nothing can be learned.
ys = np.random.default_rng(1).permutation(y[tr])
cs = fit_centroid(X[tr], ys)
print("\n2. the same models trained on shuffled labels (there is nothing to learn)")
print(f"   nearest average : train {acc(predict_centroid(cs, X[tr]), ys):.1f}%  test {acc(predict_centroid(cs, X[te]), y[te]):.1f}%")
print(f"   1-NN            : train {acc(predict_1nn(X[tr], ys, X[tr]), ys):.1f}%  test {acc(predict_1nn(X[tr], ys, X[te]), y[te]):.1f}%")

# 3. Leakage through selection: choose the best of 200 pure coin-flippers on a small test set.
rng = np.random.default_rng(2); truth = rng.integers(0, 2, 100); fresh = rng.integers(0, 2, 100000)
guessers = rng.integers(0, 2, (200, 100))
scores = 100 * (guessers == truth).mean(1); best = scores.argmax()
fresh_pred = rng.integers(0, 2, 100000)             # a coin-flipper has no memory of the test set: judge a fresh one
print("\n3. best of 200 coin-flipping 'models' on 100 test rows")
print(f"   typical model {np.median(scores):.0f}%, best model {scores.max():.0f}%; the same kind of model on 100,000 new rows: {acc(fresh_pred, fresh):.1f}%")

# 4. Noise across splits: is nearest-average (mean) better than nearest-median? 30 different splits.
A, B = [], []
for seed in range(30):
    tr, te = split(100 + seed)
    A.append(acc(predict_centroid(fit_centroid(X[tr], y[tr], "mean"), X[te]), y[te]))
    B.append(acc(predict_centroid(fit_centroid(X[tr], y[tr], "median"), X[te]), y[te]))
A, B = np.array(A), np.array(B); diff = A - B
boot = [np.random.default_rng(i).choice(diff, 30).mean() for i in range(5000)]
print("\n4. 30 random splits, mean-average vs median-average")
print(f"   mean   : {A.mean():.2f}% +- {A.std():.2f}   median : {B.mean():.2f}% +- {B.std():.2f}")
print(f"   first split alone says: difference {diff[0]:+.2f} points; over 30 splits: {diff.mean():+.2f} points (95% interval {np.percentile(boot, 2.5):+.2f} to {np.percentile(boot, 97.5):+.2f}); mean wins on {(diff > 0).sum()} of 30 splits, loses on {(diff < 0).sum()}")
np.save("res4.npy", np.array([A, B]))
np.save("sel4.npy", scores)
```

Output:

```
1. baseline and held-out accuracy
   always predict the most common digit : test 9.3%
   nearest average (Part 1)            : train 91.1%  test 89.2%
   nearest single training image (1-NN): train 100.0%  test 98.4%

2. the same models trained on shuffled labels (there is nothing to learn)
   nearest average : train 16.5%  test 6.6%
   1-NN            : train 100.0%  test 10.0%

3. best of 200 coin-flipping 'models' on 100 test rows
   typical model 50%, best model 61%; the same kind of model on 100,000 new rows: 50.2%

4. 30 random splits, mean-average vs median-average
   mean   : 89.97% +- 0.85   median : 89.03% +- 0.79
   first split alone says: difference +1.38 points; over 30 splits: +0.94 points (95% interval +0.73 to +1.16); mean wins on 30 of 30 splits, loses on 0
```

## Try it

1. Make the test set 100 rows instead of 797 (change the default `n_train=1000` in `split` to 1697) and rerun experiment 4. How does the width of the interval change?
2. In experiment 3, replace 200 coin-flippers with 20 and with 2,000. How does the best score move, and why?
3. Train 1-NN on 50 images instead of 1,000. Which gap grows, train-versus-test, or baseline-versus-test?
