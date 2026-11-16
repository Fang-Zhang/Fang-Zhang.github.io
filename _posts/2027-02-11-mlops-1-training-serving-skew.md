---
title: "MLOps (1/6): Training-Serving Skew, and a Check That Catches It"
date: 2027-02-11 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [MLOps, training-serving skew, feature pipeline, data quality, series]
description: "Part 1 of 6. Five realistic ways the serving-side feature code can differ from the training-side code, measured on a tabular model: AUC drops from 0.686 to as low as 0.552, one case is invisible to ranking metrics, and a feature-level comparison of logged serving features catches all five."
---

This series covers classic machine-learning operations: the model is trained by the team, the data is tabular, and labels eventually arrive. It is independent of the LLMOps series. Part 1 starts where production ML systems most often go wrong, which is the gap between the code that builds features for training and the code that builds them for serving.

## What the sources say

[Google's Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) defines training-serving skew as a difference between performance during training and performance during serving. Its Rule 29 gives the remedy: save the set of features used at serving time, and pipe those features to a log to use at training time. [Sculley et al. (2015)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems) list data dependencies and configuration among the ML-specific sources of technical debt; a feature computed in two places is both. The [ML Test Score rubric](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) turns this kind of concern into 28 concrete tests and monitoring needs.

## The experiment

Synthetic data, 40,000 rows: age, income, region (4 values), and a 30-day spend that is missing for 10% of rows. The label is drawn from a logistic function of these. A feature function (log income, log spend with missing filled by the training median, one-hot region) is fitted on 30,000 rows. Two models are trained on its output: logistic regression and gradient-boosted trees (scikit-learn defaults). The remaining 10,000 rows are scored through the training pipeline, then through five serving variants that each differ in one way:

- **Units:** income arrives in thousands.
- **Null handling:** a missing spend becomes 0 instead of the training median.
- **Category order:** the one-hot columns are assigned in order of first appearance within a request batch.
- **Stale join:** the spend feature comes from the previous row (a misaligned join).
- **Unseen category:** a new region value appears in 10% of rows.

![Two bar charts over six cases. Left: AUC for logistic regression and boosted trees. Same code 0.686 and 0.685; income in thousands 0.686 and 0.643; null to zero 0.667 and 0.666; one-hot by order seen 0.670 and 0.669; stale join 0.553 and 0.552; unseen category 0.685 and 0.684. Right: mean predicted probability of the logistic model, near 0.57 except 0.997 for income in thousands.](/assets/pic/mlops-1-skew.png)
*Figure 1. Held-out AUC (left) and mean predicted probability (right) for each serving variant. The data is synthetic; the pattern, not the values, is the point.*

## What the numbers show

- **A stale join is the worst case:** AUC falls from 0.686 to 0.553 for both models, close to the 0.5 of a coin. Its mean score (0.567) matches the base rate (0.568), so a check on average score sees nothing.
- **Units are invisible to ranking for the linear model.** Multiplying income by a constant adds a constant to the log feature, which shifts every score equally and leaves the order unchanged. AUC stays at 0.6864. The mean predicted probability goes from 0.572 to 0.997, so every decision made at a fixed threshold changes. The boosted trees lose 0.042 AUC (0.685 to 0.643) because their splits sit on the training scale.
- **Null handling and category order cost about 0.017 AUC each,** and move the mean score by 0.03 and 0.02. These are the changes that survive code review because each looks reasonable alone.
- **An unseen category costs 0.002 AUC.** All one-hot columns are zero for those rows, and 10% of rows is not enough to move the aggregate; the damage is local to that slice.

An aggregate metric alone would pass some of these and fail others, and it could not say which feature to look at.

## The check

Log the feature vector at serving time, recompute the same rows through the training pipeline, and compare per column. On this data the largest absolute difference per column was:

- same code: 0.0 in every column
- units: `log_income` 6.91
- null handling: `log_spend` 1.10
- category order: `r_north`, `r_south`, `r_east` 1.00
- stale join: `log_spend` 2.88
- unseen category: all four region columns 1.00

Every case is flagged and points at the column. The same-code case is exactly zero, so the threshold can be zero for deterministic features. This is Rule 29 in code: the comparison needs the logged features, and the logged features need an owner and a retention period.

## Limits

- The data is synthetic with a linear signal, and AUC is only 0.686; effects on a high-signal model could be larger.
- Each serving variant is injected by the author to mimic a failure type; real incidents combine several.
- The comparison assumes the training pipeline can recompute features for past requests, which needs point-in-time correct data; stale joins in the recomputation would hide the error.
- Two model families and one seed; no confidence intervals on the AUC values.

Next: [Part 2](/posts/mlops-2-data-validation/) covers data validation: a schema, a distribution check and a measured false-alarm rate.
