---
title: "MLOps (5/6): Feedback Loops, and What Exploration Buys"
date: 2027-02-15 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [MLOps, feedback loops, selection bias, exploration, series]
description: "Part 5 of 6. Two simulated loops where a model's decisions shape its next training set: a lender that retrains on approved applicants only, and a ranker that learns from clicks on what it showed. Without exploration the ranker finds 0.40 of the 5 best items; at 30% exploration it finds 2.92."
---

[Part 4](/posts/mlops-4-delayed-labels-and-monitoring/) covered labels that arrive late. This part covers labels that exist only for the cases the model chose to act on. [Sculley et al. (2015)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems) list hidden feedback loops among the ML-specific risks, and [Breck et al. (2019)](https://proceedings.mlsys.org/paper_files/paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html) note that logged predictions are often used to build more training data, so feedback loops can amplify even small data errors. [Chaney, Stewart and Engelhardt (2018)](https://arxiv.org/abs/1710.11214) simulate recommendation systems trained on data from users already exposed to recommendations and report that this homogenises user behaviour without increasing utility. The two simulations below are the author's, built to isolate the mechanism.

## Loop 1: training only on what was approved

A lender scores applicants and approves the safest 30%; only approved applicants get an outcome label. Each round it retrains on all labelled applicants so far and approves from 4,000 new ones. The true default rate rises with |x0| (a U-shape the first model sees only in part), so the applicants with large |x0| are the ones the model stops approving and stops learning about. An exploration share approves a random fraction of applicants regardless of score. Ten runs per setting, 20,000 held-out applicants, gradient-boosted trees. The "far region" is |x0| above 1.5, where the true default rate is 0.923.

| Exploration | AUC overall, round 0 → 7 | AUC in the far region, round 0 → 7 | Predicted default rate in the far region, round 0 → 7 |
|---|---|---|---|
| 0 | 0.799 → 0.806 | 0.660 → 0.675 | 0.891 → 0.875 |
| 0.05 | 0.799 → 0.807 | 0.660 → 0.682 | 0.891 → 0.882 |
| 0.20 | 0.799 → 0.810 | 0.660 → 0.707 | 0.891 → 0.892 |

The overall AUC hides the loop: it moves by less than 0.011 in every setting. Inside the region the lender stopped approving, learning is slow without exploration (0.015 AUC in seven rounds) and the predicted rate drifts further from the true 0.923, from 0.891 to 0.875. With 20% exploration the region improves by 0.047 and the predicted rate holds at 0.892. The effect here is moderate. A model that never sees the blind region does not collapse; it stays where it was and slowly gets worse relative to the world.

## Loop 2: ranking by observed clicks

A ranker shows 5 of 200 items per step for 3,000 steps. Each item has a true click rate drawn from a Beta(2, 8) distribution, so the mean is 0.2. The ranker orders items by observed clicks over shows, starting from a prior of one click in ten shows for each item, and only shown items get new data. An exploration share replaces that fraction of the 5 slots with random items. One hundred runs per setting.

| Exploration | True top 5 found | Rank correlation, estimate vs truth | Items shown beyond the prior | Quality of shown top 5 vs best 5 |
|---|---|---|---|---|
| 0 | 0.40 of 5 | 0.06 | 30 of 200 | 56% |
| 0.02 | 0.75 | 0.32 | 161 | 71% |
| 0.10 | 1.75 | 0.62 | 200 | 85% |
| 0.30 | 2.92 | 0.81 | 200 | 94% |

![Two panels. Left: AUC in the blind region by retraining round for exploration 0, 0.05 and 0.20, ending at 0.675, 0.682 and 0.707. Right: bars of shown top-5 quality against best possible for exploration 0, 0.02, 0.10, 0.30: 56, 71, 85 and 94 percent, with 0.40, 0.75, 1.75 and 2.92 of the 5 true best items found.](/assets/pic/mlops-5-feedback.png)
*Figure 1. Left: the lender loop. Right: the click-ranking loop.*

Without exploration, 170 of the 200 items are never shown past the prior and the ranking has almost no correlation with the truth (0.06). The loop locks onto whichever items got early clicks. Even 2% exploration, one random slot in fifty, moves the shown quality from 56% to 71%.

## What to do about it

- Log the model's decision and score with each outcome, so the selection process can be modelled later.
- Reserve a small randomised slice with logged probabilities; it is the only unbiased sample of the blind region.
- Evaluate on that slice, not on the labelled set the model itself selected.
- Treat the exploration share as a cost: here it buys information at the price of showing some worse items or approving some riskier applicants.

## Limits

- Both simulations are simple and synthetic; real recommenders and lenders have many interacting models, human overrides and delayed labels (Part 4).
- The ranker ignores item position effects and user differences.
- The lender's approval rule (safest 30%) and the U-shaped risk are chosen by the author to make the blind region visible.
- Random exploration is the simplest scheme; bandit methods that explore adaptively were not tested.
- Loop 1 averages ten runs and loop 2 one hundred; no intervals are given.

Next: [Part 6](/posts/mlops-6-release-and-the-peeking-problem/) covers release: the peeking problem in A/B tests, and what a small canary can see.
