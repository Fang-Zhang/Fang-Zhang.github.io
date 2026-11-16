---
title: "MLOps (2/6): Data Validation Is a Schema, a Distribution Check and a Choice of Thresholds"
date: 2027-02-12 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [MLOps, data validation, schema, data quality, series]
description: "Part 2 of 6. A 40-line validator (schema, null rates, domains, ranges, distribution distance) run on nine corrupted batches: it flags eight of nine corruptions at a 1-in-200 false-alarm rate, misses a 3% shift, and flags changes that cost the model nothing."
---

[Part 1](/posts/mlops-1-training-serving-skew/) covered the gap between training and serving code. This part covers the data itself: what a pipeline should check on each new batch before it trains on it, and what those checks cost in false alarms.

## What the source says

[Breck et al. (2019)](https://proceedings.mlsys.org/paper_files/paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html) describe the data validation system used in Google's TFX platform. The paper's argument is that training and serving data are a production asset on par with the algorithm and infrastructure. It describes three components: a data analyzer that computes statistics, a data validator that checks them against a schema, and a model unit tester that checks the training code using synthetic data generated from the schema. It also notes that validation runs over the whole batch so that anomalies in small slices are not silently ignored. The [ML Test Score rubric](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) lists data tests among its 28 items.

## The validator

The same synthetic credit-style table as Part 1 (age, income, region, 30-day spend with about 10% null, and a binary label). The reference batch has 30,000 rows. The validator checks, for a new batch:

- required columns present;
- null fraction per column against a limit (1.5 times the reference rate for spend, zero for the others);
- categorical values inside the known domain;
- numeric values inside human-reviewed bounds (age 18 to 80, income and spend 0 to 10^7);
- region frequencies: largest absolute difference from the reference above 0.05;
- age and income median moved by more than 0.10 of the reference interquartile range;
- label rate moved by more than 0.03.

A first version took the bounds from the observed minimum and maximum of the reference batch. It raised alarms on 19 of 200 clean batches of 2,000 rows, because a new batch will often exceed the extremes of an old one. Reviewed bounds brought this to 1 of 200. The schema can be inferred, as the paper describes, but the thresholds need a human decision and a measured false-alarm rate.

## Nine corruptions

Each case is a new batch of 10,000 rows. The bar is the AUC of a logistic model retrained on that batch and scored on a clean test set of 10,000 rows. The reference model scores 0.6936.

![Bar chart over ten batches showing AUC after retraining, with the validator verdict above each. Clean 0.694, not flagged. Age in months 0.672, flagged. Negative incomes: retraining failed, flagged. Spend null 40 percent 0.693, flagged. Region typos 0.694, flagged. Region mix shift 0.683, flagged. Income plus 15 percent 0.694, flagged. Income plus 3 percent 0.694, not flagged. Labels flipped 20 percent 0.693, flagged. Column dropped: retraining failed, flagged.](/assets/pic/mlops-2-validation.png)
*Figure 1. Validator verdict against the AUC of a model retrained on the batch.*

- **Flagged and harmful:** age in months (AUC 0.672), region mix shift (0.683), negative incomes and a dropped column (the retraining code failed outright).
- **Flagged and nearly harmless to this model:** 40% nulls in spend (0.693), typos in 5% of regions (0.694), income +15% (0.694), labels flipped for 20% of rows (0.693). The validator reports that the data changed, not that the model got worse. Whether to block training on such a flag is a policy decision; this experiment cannot make it.
- **Not flagged:** income +3%, which also had no effect (0.694).
- **Clean batch:** not flagged.

The label-flip case is the one the validator found only through the label-rate check (0.533 against 0.566). A flip that preserved the label rate would pass every check here.

## Where the checks stop

The validator compares a batch with a reference. It cannot tell a bug from a real change in the world, and a slow drift that moves under each threshold passes forever. Part 4 returns to this with labels and monitoring. The paper's third component, unit tests for training code with schema-generated data, is not implemented here.

## Limits

- One synthetic table and one model family; harm to a tree model or to a model with a strong income effect would differ.
- The corruptions are chosen by the author and are each applied to one batch, once; the AUC values have no interval.
- False-alarm rates are from 200 clean batches of 2,000 rows; the corruption cases use 10,000 rows, so their sensitivity is higher than a 2,000-row daily batch would give.
- The 1-in-200 figure is for these thresholds on this data and would be re-measured on any other.

Next: [Part 3](/posts/mlops-3-reproducible-training-and-search/) covers reproducible training, experiment records and what a hyperparameter search reports.
