---
title: "MLOps (4/6): Delayed Labels Delay the Alarm and Bias the Retrain"
date: 2027-02-14 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [MLOps, monitoring, delayed feedback, concept drift, series]
description: "Part 4 of 6. When labels arrive days after predictions, a labelled-accuracy monitor lags a concept change by the label delay plus two days, input monitoring sees nothing, and a model trained on immature labels predicts 0.348 where the true rate is 0.448."
---

[Part 3](/posts/mlops-3-reproducible-training-and-search/) covered training. After release, a classic ML system usually gets its labels late: a loan defaults months later, a purchase follows a click by days. This part measures what that delay does to monitoring and to retraining.

## What the sources say

[Chapelle (2014)](https://dl.acm.org/doi/10.1145/2623330.2623634) studies delayed feedback in display advertising, where a conversion can come up to 30 days after the click, and models the delay explicitly alongside the conversion itself. [Rabanser, Günnemann and Lipton (2018)](https://arxiv.org/abs/1810.11953) study dataset-shift detection and open with the observation that ML systems, which depend strongly on properties of their inputs, tend to fail silently; they find that two-sample tests on a pre-trained classifier's outputs perform best across the shifts they test. Their shifts alter the inputs or the label proportions. The case below is different: the input distribution stays put and the relation between inputs and label changes.

## Part A: how late is the alarm

A logistic model is trained on 5 features (20,000 rows, AUC 0.826). From day 30 the label rule changes: the features that matter switch from the first three to the last three, with the same input distribution. Each day brings 500 predictions. The monitor pools 7 days of labelled traffic and alarms when AUC falls more than 0.08 below the training value. The labels for day *t* arrive on day *t* plus the delay.

- **Input monitoring is blind.** The largest daily PSI on the first feature over 90 days was 0.045, and the daily mean score barely moved (0.447 before, 0.442 after, against a day-to-day standard deviation of 0.011).
- **The labelled monitor works but lags.** First alarm, in days after the change: delay 0 gives 2, delay 1 gives 3, delay 7 gives 9, delay 14 gives 16, delay 30 gives 32. The lag is the delay plus about two days, which is the time for changed labels to fill enough of the 7-day window.
- **False alarms:** 0 of 200 simulated 60-day runs with no change (delay 7).

![Two panels. Left: line chart of days from the change to the first alarm against label delay, points at (0,2), (1,3), (7,9), (14,16), (30,32), just above the line lag equals delay. Right: bars of mean predicted probability on fresh data for four training methods: naive 0.348, wait until mature 0.441, drop rows under 7 days old 0.403, delay-aware likelihood 0.452, against a true rate of 0.448.](/assets/pic/mlops-4-delayed-labels.png)
*Figure 1. Left: alarm lag against label delay. Right: calibration of models trained under four ways of handling immature labels.*

When the delay is long, the monitor reports last month's model. Options without labels are a proxy label with a short delay, or a canary set with known answers; neither detects a change in a feature the canaries do not exercise.

## Part B: training on immature labels

Positives convert after an exponentially distributed delay with a 7-day mean; each row is observed at an age between 0 and 30 days. A positive that has not yet converted looks like a negative. The true positive rate is 0.452 and the observed rate is 0.347. Four ways to train, each scored on fresh data with the true label rule:

| Method | Rows | Mean prediction (true 0.448) | AUC |
|---|---|---|---|
| Naive: unconverted rows are 0 | 60,000 | 0.348 | 0.8178 |
| Wait until age 21 days | 18,066 | 0.441 | 0.8176 |
| Drop rows younger than 7 days | 45,897 | 0.403 | 0.8178 |
| Delay-aware likelihood | 56,820 | 0.452 | 0.8177 |

Ranking quality (AUC) is the same to three decimals in every row. The level of the predicted probability is not: the naive model underestimates by 0.10, which matters for any decision at a fixed threshold or any expected-value calculation. Waiting for maturity fixes the level but uses 30% of the rows and trains on data 21 days old. The delay-aware method fits the probability of conversion times the probability that the label has arrived by now, and recovers 0.452 using 95% of the rows; it assumes the delay law is known, which here it is by construction.

## Limits

- The data is synthetic and linear; the delay is exponential and independent of the features, which real delays are not (Chapelle's own model lets the delay depend on features).
- The monitor's threshold (0.08) and window (7 days) were fixed by the author; a more sensitive setting alarms sooner and false-alarms more, and only the false-alarm count at this setting was measured.
- One change type, one run per delay, so the lag figures have no interval.
- The delay-aware fit is given the true delay law. Estimating it adds error that this experiment does not measure.

Next: [Part 5](/posts/mlops-5-feedback-loops/) covers feedback loops: what a model never sees, and what a little exploration buys.
