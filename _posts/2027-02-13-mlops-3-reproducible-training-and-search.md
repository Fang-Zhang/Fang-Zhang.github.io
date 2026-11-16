---
title: "MLOps (3/6): Reproducible Training, and What a Hyperparameter Search Reports"
date: 2027-02-13 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [MLOps, reproducibility, hyperparameter search, experiment tracking, series]
description: "Part 3 of 6. Gradient-boosted trees on CPU repeat bit for bit under a fixed seed, vary by 0.002 AUC across seeds, and a 64-config search reports a validation score 0.005 to 0.017 above test depending on how many rows chose the winner."
---

[Part 2](/posts/mlops-2-data-validation/) checked the data going in. This part covers the training step: whether a run can be repeated, what an experiment record must contain, and how much a hyperparameter search flatters its own result.

## What the source says

[Google's MLOps guide](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) describes an automated training pipeline with data and model validation steps, and stores the resulting model in a registry together with its metadata. [Sculley et al. (2015)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems) list configuration debt among the ML-specific risks. This post measures two things those documents assume: that a recorded configuration reproduces a model, and that the score recorded with it is trustworthy.

## The setup

A synthetic binary classification set (30,000 rows, 20 features, 15% label noise) from scikit-learn's `make_classification`. Training uses 6,000 rows, validation 3,000, test 21,000. The model is scikit-learn's gradient-boosted trees, on CPU.

## Reproducibility

- Three runs with the same configuration and seed gave the same prediction hash (`2f0220413c44` each time).
- Twenty seeds with a configuration that subsamples rows (0.7) gave validation AUC with mean 0.8453 and standard deviation 0.0021 (range 0.8393 to 0.8482). A seed is therefore part of the model's identity: a difference of 0.002 AUC between two runs is within the seed spread.
- The record that reproduced the best model contained the training-data hash, the configuration, the seed and the library name; a rerun from it gave identical predictions. The library version is also part of the record in practice; this single-machine test does not vary it.

## What a search reports

Sixty-four random configurations (trees 20 to 200, depth 1 to 6, learning rate 0.01 to 0.3, subsample 0.5 to 1.0), each fitted once. Validation AUC across them: median 0.8312, maximum 0.8679. For each K, the experiment draws 300 random subsets of K configurations, keeps the one best on validation, and compares its validation AUC with its AUC on the 21,000-row test set.

| Configs searched | Validation of chosen | Test of chosen | Optimism |
|---|---|---|---|
| 1 | 0.8191 | 0.8157 | +0.0034 |
| 4 | 0.8518 | 0.8497 | +0.0021 |
| 16 | 0.8634 | 0.8605 | +0.0029 |
| 64 | 0.8679 | 0.8626 | +0.0053 |

The search is worth doing here: test AUC rises from 0.816 to 0.863. The optimism, the amount by which the reported validation score exceeds test, grows with the number of configurations at the high end, but with 3,000 validation rows it stays near 0.005. The winner also beats the default configuration clearly (test AUC 0.8628 against 0.8391).

The size of the validation set matters more. Repeating the 64-config choice on random subsets of the validation rows:

| Validation rows | Validation of chosen | Test of chosen | Optimism | Regret vs best config |
|---|---|---|---|---|
| 150 | 0.8742 | 0.8574 | +0.0167 | 0.0078 |
| 300 | 0.8720 | 0.8609 | +0.0111 | 0.0044 |
| 1,000 | 0.8691 | 0.8633 | +0.0058 | 0.0019 |
| 3,000 | 0.8679 | 0.8626 | +0.0053 | 0.0027 |

![Two line charts. Left: AUC of the chosen configuration against the number of configurations searched, for validation and test; both rise from about 0.82 to 0.87 and 0.86. Right: optimism and regret against validation size; optimism falls from 0.017 at 150 rows to 0.005 at 3,000, regret from 0.008 to 0.003.](/assets/pic/mlops-3-search.png)
*Figure 1. Left: gains from searching more configurations. Right: how the validation-set size changes the optimism of the reported score and the quality of the choice.*

With 150 validation rows the search reports 0.874 for a model that scores 0.857 on test, and it picks a configuration 0.008 AUC worse than the best in the pool. Regret flattens above 1,000 rows (0.0019 at 1,000 against 0.0027 at 3,000; that difference is within the noise of 200 resamples).

## What to record

The measurements support a minimum record per run: data hash, full configuration, seed, library versions, the validation set identity and size, and the number of configurations tried to reach this one. The last item is the one most often missing, and it is what lets a reader judge the optimism. A test set that was used to choose among models is a validation set, so the final number needs rows no search has touched.

## Limits

- One synthetic dataset, one model family, one machine; GPU training, multi-threaded boosting libraries and distributed training add nondeterminism not tested here.
- The pool of 64 configurations is fitted once each; the K-subset draws share those fits, so the rows of the first table are correlated.
- Optimism and regret are means over resamples of a single validation set of 3,000 rows, not over fresh datasets.
- The search is random, not Bayesian or successive halving; those would change the optimism curve.

Next: [Part 4](/posts/mlops-4-delayed-labels-and-monitoring/) covers delayed labels: how late the alarm is, and what immature labels do to a retrain.
