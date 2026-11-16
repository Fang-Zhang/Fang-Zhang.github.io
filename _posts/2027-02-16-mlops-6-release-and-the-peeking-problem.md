---
title: "MLOps (6/6): Release, Canaries and the Peeking Problem"
date: 2027-02-16 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [MLOps, A/B testing, canary, sequential testing, series]
description: "Part 6 of 6. Checking an A/B test every day and stopping at the first significant result turns a 5% false-positive rate into 25.1%; an always-valid test holds it at 1.5% at the price of power. A 1% canary for an hour catches a doubled error rate and misses a 0.1-point one."
---

[Part 5](/posts/mlops-5-feedback-loops/) covered loops that shape training data. The last step of the pipeline is release: deciding whether a new model is better than the one it replaces, using live traffic. This part measures two things about that step: how often an A/B test lies when it is watched, and what a small canary can see.

## What the source says

[Johari, Pekelis and Walsh](https://arxiv.org/abs/1512.04922) state that A/B tests analysed with ordinary p-values and confidence intervals are unreliable if users choose sample sizes by continuously monitoring the test. They define always-valid p-values and confidence intervals that remain valid whenever the user decides to stop. The simulation below reproduces the problem and a mixture sequential test of that family. The mixture parameter (prior variance 0.0004 on the difference) is the author's choice, not taken from the paper.

## The peeking problem

Two arms with 500 users per arm per day for 20 days. Conversion is 10% in the control. The test is a two-sample z-test at the 5% level. Four thousand simulated experiments per row.

| Setup | False positives (no true difference) | Detected (true lift 10% to 11%) |
|---|---|---|
| One look at day 20 | 4.9% | 62.8% |
| Daily looks, stop at first p below 0.05 | 25.1% | 78.5% |
| Daily looks, mixture sequential test | 1.5% | 32.4% |

![Two bar charts for the three setups. Left: false positives with no true difference: 4.9 percent for one look, 25.1 percent for daily looks stopping at first significance, 1.5 percent for the mixture sequential test; a dashed line marks the nominal 5 percent. Right: detection of a one-point lift: 62.8, 78.5 and 32.4 percent.](/assets/pic/mlops-6-release.png)
*Figure 1. False-positive rate and detection rate of three ways of reading the same experiments.*

Stopping at the first significant day makes one experiment in four a false win, and the median stopping day among those false wins was day 5. The detection gain (78.5% against 62.8%) comes from the same mechanism: it is not extra evidence. The always-valid test keeps the error rate below 5%, at 1.5%, and in this 20-day window it detects the lift 32.4% of the time against 62.8% for a single look. It is conservative here; its advantage is that the experiment can be stopped early for a large effect without invalidating the result, and that was not tested. Fixing the sample size in advance and looking once is the cheapest honest design when the duration can be fixed.

## What size can a test see?

Baseline conversion 10%, two-sided 5% level, 80% power, per arm:

- lift of 2.0 points: 3,842 users (8 days at 500 per arm per day)
- 1.0 point: 14,752 (30 days)
- 0.5 point: 57,764 (116 days)
- 0.2 point: 356,339 (713 days)

Required sample size scales with the inverse square of the lift. A model that is better by 0.2 points is not measurable at this traffic in any practical time, and the decision then rests on offline evaluation or on a larger risk budget.

## What a canary can see

A canary sends 1% of traffic to the new version. At 200 requests per second for one hour that is 7,200 canary requests against 712,800 for the old model. The detection rate of a two-sample z-test at 5%, 3,000 simulated hours per row:

- error rate 0.5% to 1.0%: 99.9%
- error rate 0.5% to 0.6%: 24.0%
- conversion 10.0% to 9.5%: 27.9%

A canary is a guard against gross failure (crashes, doubled error rates, schema mismatches), not a measurement of small quality changes. Shadow mode, where the new model scores live traffic and its output is logged but not used, has the same sample-size arithmetic for quality, since there is no outcome unless the old model's action produced one; its main use is catching serving errors and comparing outputs (Part 1's feature-parity check also fits here).

## The series in one line each

1. [Part 1](/posts/mlops-1-training-serving-skew/): five serving variants cost between 0 and 0.134 AUC, and a per-column check on logged features caught all five.
2. [Part 2](/posts/mlops-2-data-validation/): a validator flagged eight of nine corruptions at a 1-in-200 false-alarm rate, and flagged several that cost the model nothing.
3. [Part 3](/posts/mlops-3-reproducible-training-and-search/): seeded training repeated exactly; a search over 64 configurations reported 0.005 to 0.017 AUC too high depending on validation size.
4. [Part 4](/posts/mlops-4-delayed-labels-and-monitoring/): a labelled monitor lagged a concept change by the label delay plus two days; immature labels biased the level of predictions by 0.10.
5. [Part 5](/posts/mlops-5-feedback-loops/): without exploration, 30 of 200 items ever got data and the ranker found 0.40 of the best 5.
6. This part: daily peeking gave 25.1% false positives; a 1% canary sees a doubled error rate and not a 0.1-point one.

## Limits

- The A/B simulation uses binary conversion with independent users and a fixed daily volume; real tests have novelty effects, weekday cycles and correlated users.
- Only one mixture parameter was tried; a different choice shifts the power of the sequential test.
- Early-stopping benefits of the sequential test, and multiple metrics per experiment, were not simulated.
- Canary numbers assume independent requests and a z-test; they say nothing about rare, severe failures that need different detectors.

This closes the series; an [epilogue](/posts/mlops-vs-llmops-side-by-side/) sets it against LLMOps. It is independent of the [LLMOps series](/posts/llmops-1-from-mlops-to-llmops/), which covers the cases where the model is not trained by the team.
