---
title: "ML Engineer Interviews, With Reference Answers (10/18): ML System Design II, Fraud, Forecasting and Data Modelling"
date: 2027-01-05 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, system-design, fraud-detection, forecasting, feature-store, series]
description: "Part 10 of 18. The second half of the system-design round: fraud detection with delayed labels, real-time versus batch, point-in-time features, forecasting baselines and evaluation, and data modelling, with runnable code."
---

This post covers the second half of the system-design round, after [Part 9](/posts/ml-engineer-interviews-9-ml-system-design-1-recommendation-and-search/): fraud detection, features computed in real time versus in batch, point-in-time correctness, forecasting, and the table design underneath all of them. It gives ten questions with what each tests, a reference answer, a common weak answer and a follow-up. The numbers in the answers come from code that was run for this post, on synthetic data, with the output shown; claims about papers and documentation are limited to the passages that were read.

The code ran on a CPU-only machine (32 cores) with Python 3.14.7, NumPy, pandas, scikit-learn and the standard-library `sqlite3`, with fixed seeds. The data are synthetic, so the experiments check mechanisms; absolute numbers will differ on real data.

![Left: precision of the top 100 daily fraud alerts when the newest usable labels are 1, 7 and 14 days old: 0.631, 0.423 and 0.211. Right: total cost against decision threshold on a synthetic fraud problem, lowest near the threshold given by the cost ratio, 0.0476.](/assets/pic/ml-interview-10-fraud-delay-cost.png)
*Left: label delay under concept drift (question 2). Right: cost-based thresholding (question 3).*

## 1. "Design a fraud detection system for card payments. How do you structure your answer?"

**What it tests.** Whether the candidate treats fraud as a decision system under constraints, not as a classifier on an imbalanced table.

**Reference answer.** Open with the constraints that change the design, and ask for them if they are not given:

1. **The decision and its deadline.** Approve, decline, or send to a human or a step-up check, within a latency budget set by the payment flow (the author's framing; typical budgets are tens to hundreds of milliseconds, but the number should come from the interviewer).
2. **Costs.** A missed fraud costs roughly the transaction amount plus fees; a false decline costs a customer and revenue; a manual review costs analyst time. These three numbers set the threshold (question 3).
3. **Capacity.** Human investigators can look at a limited number of alerts per day. The Dal Pozzolo et al. paper on [concept-drift adaptation with delayed supervised information](https://dalpozz.github.io/static/pdf/IJCNN2015_final.pdf) states the consequence: the primary concern of a fraud-detection system is to return a small number of very precise alerts, and the optimal system maximises the frauds detected within the budget of alerts that can be reported. Classical metrics such as AUC or the F-measure are, in the paper's words, not necessarily meaningful for alert precision.
4. **Labels.** Which labels exist, and when. Only a few transactions are checked by investigators soon after the fact; the labels for the rest become available days later, when customers have had time to report fraud (same paper). This drives the training-set design (question 2).
5. **Adversaries.** Fraud patterns change because fraudsters react; the paper lists concept drift and class imbalance as the two core difficulties. Plan for retraining and monitoring from the first day.

Then draw the pipeline: event stream, feature computation (some features over recent history, some static), a model score, a rules layer for hard constraints (sanctions, velocity limits, known-bad lists), a decision policy with thresholds, an alert queue for investigators, and a label pipeline that feeds back. A reasonable first model is gradient-boosted trees on aggregate features (counts and amounts per card, device, merchant and time window); a rules-only baseline comes first (the same "ship a heuristic first" logic as in Part 9).

**Common weak answer.** "Train XGBoost on the labelled data, optimise AUC, and use SMOTE for the imbalance." It skips the decision, the costs, the alert budget and the label delay, and it evaluates with a metric that does not match the operating point.

**Follow-up.** "What do you do on day one, with no labels?" Rules and anomaly scores for review, and treat the investigators' decisions as the first labelled data; expect them to be a biased sample, for the reason given in question 2.

## 2. "Labels arrive days after the transaction, and the fraud pattern drifts. How does that change training?"

**What it tests.** Understanding verification latency and why it interacts with concept drift.

**Reference answer.** If the newest usable label is `d` days old, the model is trained on the world as it was `d` days ago. When the fraud pattern is stable, that costs little; when it drifts, the model scores today's fraud with yesterday's pattern. The [IJCNN 2015 paper](https://dalpozz.github.io/static/pdf/IJCNN2015_final.pdf) reports that most fraud-detection literature assumes immediate and accurate labels after each transaction, and describes the realistic setting instead: investigators supply a few recent labels (feedback) on the alerts the system raised, and all other labels arrive with a delay. Two further points from that paper and its [journal follow-up](https://dalpozz.github.io/static/pdf/TNNLS_2017.pdf):

- The feedback samples are selected by the system itself (they are the transactions it found riskiest), so a classifier trained on them alone suffers from sample selection bias. The authors' conclusion is to train separate classifiers on feedback samples and on delayed samples and to aggregate their output probabilities, which improved alert precision on real card data in their experiments.
- The delayed labels cover all transactions, so they are the representative sample, but they are old.

The experiment below isolates the delay. A synthetic fraud pattern rotates slowly over 60 days; each day a logistic regression is retrained on a 14-day window ending `delay` days earlier, and the day's 4,000 transactions are scored. The metric is the precision of the 100 highest-scored transactions, a fixed alert budget. The simulation has no alert-feedback loop, so it shows the delay effect only, not the selection bias.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0); days, per_day, d = 60, 4000, 6
def fraud_dir(day):                                   # the fraud pattern rotates slowly: concept drift
    a = day * 0.06; v = np.zeros(d); v[0], v[1] = np.cos(a), np.sin(a); v[2] = 0.5; return v
X, y, day = [], [], []
for t in range(days):
    x = rng.normal(size=(per_day, d)); logit = 4.0 * (x @ fraud_dir(t)) / 1.1 - 7.5
    X.append(x); y.append(rng.random(per_day) < 1 / (1 + np.exp(-logit))); day += [t] * per_day
X = np.vstack(X); y = np.concatenate(y); day = np.array(day); print(f"fraud rate {y.mean():.4f}")
def run(delay, window=14, budget=100):
    prec = []
    for t in range(30, days):
        lo, hi = t - delay - window, t - delay                    # only labels that have arrived: transactions at least `delay` days old
        m = (day >= lo) & (day < hi); clf = LogisticRegression(max_iter=300).fit(X[m], y[m])
        mt = day == t; s = clf.decision_function(X[mt]); top = np.argsort(-s)[:budget]; prec.append(y[mt][top].mean())
    return np.mean(prec)
for delay in (1, 7, 14): print(f"label delay {delay:2d} days: mean precision of the top 100 alerts per day {run(delay):.3f}")
```

```
fraud rate 0.0456
label delay  1 days: mean precision of the top 100 alerts per day 0.631
label delay  7 days: mean precision of the top 100 alerts per day 0.423
label delay 14 days: mean precision of the top 100 alerts per day 0.211
```

The numbers show the mechanism: with a drift of this size, a 7-day delay costs a third of the precision and a 14-day delay about two thirds. The size of the loss depends on the drift speed that was simulated and says nothing about a real portfolio. The design consequences are to measure the label delay and the drift rate on real data, to prefer features that are stable under drift, to retrain often, and to use the recent feedback labels even though they are biased, for example in a separate model as the authors do.

**Common weak answer.** "Train on all labelled data from the last year." It ignores that the labels for the latest transactions are missing (a naive pipeline then treats them as genuine), and that old data describes old fraud.

**Follow-up.** "How long should you wait before treating an unreported transaction as genuine?" Choose the maturity window from the observed distribution of report delays (for example the point at which most chargebacks have arrived) and exclude younger transactions from training instead of labelling them negative.

## 3. "How do you pick the metric and the threshold for a fraud model?"

**What it tests.** Linking a probability to a decision and a cost, and knowing what AUC hides.

**Reference answer.** Evaluate at the operating point that production will use. Two views are useful: precision at the alert budget (as in question 2), and expected cost. With a calibrated fraud probability `p`, flagging a transaction costs `C_FP` in expectation if it is genuine and saves `C_FN` if it is fraud, so flag when `p * C_FN > (1 - p) * C_FP`, that is when `p > C_FP / (C_FP + C_FN)`. This is the standard cost-sensitive decision rule; it needs calibrated probabilities, so check calibration ([Part 4](/posts/ml-engineer-interviews-4-evaluation-metrics-and-experiments/)) and recalibrate if the training set was resampled.

The experiment uses assumed costs of 100 for a missed fraud and 5 for a false alert, on a synthetic problem with a 2% fraud rate. The predicted optimum is 0.0476.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(1); n = 200000; x = rng.normal(size=(n, 4)); logit = 2.0 * x[:, 0] + 1.0 * x[:, 1] - 6.0; y = rng.random(n) < 1 / (1 + np.exp(-logit))
tr = np.arange(n) < n // 2; clf = LogisticRegression(max_iter=300).fit(x[tr], y[tr]); p = clf.predict_proba(x[~tr])[:, 1]; yt = y[~tr]
print(f"fraud rate {yt.mean():.4f}, test transactions {len(yt)}")
C_FN, C_FP = 100.0, 5.0            # assumed costs: average loss on a missed fraud, cost of reviewing/declining a genuine transaction
print(f"cost-optimal threshold on calibrated probability = C_FP / (C_FP + C_FN) = {C_FP / (C_FP + C_FN):.4f}")
best = None
for th in (0.5, 0.2, 0.1, 0.05, 0.0476, 0.02, 0.01):
    pred = p >= th; fn = (~pred & yt).sum(); fp = (pred & ~yt).sum(); cost = C_FN * fn + C_FP * fp; print(f"threshold {th:6.4f}: alerts {pred.sum():6d}, missed frauds {fn:4d}, false alerts {fp:5d}, total cost {cost:9.0f}")
```

```
fraud rate 0.0195, test transactions 100000
cost-optimal threshold on calibrated probability = C_FP / (C_FP + C_FN) = 0.0476
threshold 0.5000: alerts    365, missed frauds 1715, false alerts   131, total cost    172155
threshold 0.2000: alerts   1888, missed frauds 1229, false alerts  1168, total cost    128740
threshold 0.1000: alerts   4275, missed frauds  844, false alerts  3170, total cost    100250
threshold 0.0500: alerts   8364, missed frauds  564, false alerts  6979, total cost     91295
threshold 0.0476: alerts   8729, missed frauds  549, false alerts  7329, total cost     91545
threshold 0.0200: alerts  17109, missed frauds  272, false alerts 15432, total cost    104360
threshold 0.0100: alerts  26294, missed frauds  140, false alerts 24485, total cost    136425
```

The total cost is lowest around 0.05, in line with the rule; the curve is flat near the minimum (0.0476 and 0.05 differ by less than 1%), so the exact cutoff matters less than being in the right region. A threshold of 0.5, the default of many libraries, nearly doubles the cost here. AUC would give the same value for every row of this table, because it does not depend on the threshold, which is the reason it cannot choose one.

**Common weak answer.** "Maximise F1, or accuracy." On a 2% base rate a model that approves everything has 98% accuracy; F1 gives equal weight to precision and recall, which the real costs rarely do.

**Follow-up.** "The investigators can review only 500 alerts a day, whatever the cost ratio says." Then the budget is the binding constraint: rank by score and take the top 500, and use the cost-based threshold to decide whether to use the whole budget.

## 4. "When do features and predictions need to be real time, and when is batch enough?"

**What it tests.** Matching the freshness of each signal to the decision, and knowing the cost of two code paths.

**Reference answer.** Ask how fast the signal decays and how fast the decision is needed, per feature and not per system (the author's rule of thumb). Three tiers usually cover a fraud or recommendation system:

- **Request-time.** Computed from the request itself: amount, merchant, device, time of day. No history needed.
- **Near-real-time aggregates.** Counts and sums over short windows ("transactions on this card in the last 10 minutes"). These carry the strongest fraud signal and must be fresh to the second or minute, so they are maintained by a stream processor in a low-latency store.
- **Batch aggregates and embeddings.** Slowly changing history ("average spend over 90 days", a learned user embedding). Recomputed hourly or daily and loaded to an online store; staleness of hours does not change the score much.

The architecture question underneath is how to compute the same feature in the stream and in the batch used for training. Jay Kreps' [Questioning the Lambda Architecture](https://www.oreilly.com/radar/questioning-the-lambda-architecture/) describes the Lambda approach as running a batch system and a stream system in parallel, implementing the transformation logic twice, and stitching the results together at query time. His objection is that maintaining code that must produce the same result in two complex distributed systems is as painful as it sounds, and that the operational complexity is the one thing everyone who runs it agrees on. His alternative (he suggests the name Kappa Architecture) uses one stream-processing code path and reprocesses by replaying the retained input log with a new version of the same code, only when the code changes. This is a 2014 opinion post by an engineer who built Kafka- and Samza-based stream infrastructure at LinkedIn (his own description), so read it as an argument, not a measurement. Engines that run batch and streaming jobs from the same code attack the same problem from another side (the author's observation).

**Common weak answer.** "Real time is always better, so use streaming everywhere." Streaming costs engineering and operations, and a feature that changes slowly gains nothing from it. The opposite weak answer, "batch is cheaper, so batch only", fails when the signal is a burst.

**Follow-up.** "The online and offline versions of a feature disagree. How do you find out?" Log the features that were served for each prediction and compare them with the offline recomputation (question 5).

## 5. "A feature is computed by a streaming job in production and by SQL in training. What can go wrong?"

**What it tests.** Training-serving skew, one of the most common failures of deployed models.

**Reference answer.** The two implementations agree on clean data and diverge on the data's irregularities: late and out-of-order events, duplicates, time zones, window edge conventions (is the boundary inclusive?), null handling, and restated values. Rule 37 of the Rules of Machine Learning (read in Part 9) recommends measuring training-serving skew; the simplest control is to log the served feature values and diff them against the offline recomputation.

The test below computes a 7-day event count per user two ways: a batch recomputation that scans history, and a streaming version that keeps a per-user queue and drops expired timestamps. With events processed in order they agree exactly; then 5% of events are delayed by two hours before the streaming job sees them.

```python
import numpy as np, pandas as pd
from collections import deque
rng = np.random.default_rng(0); n = 20000
ev = pd.DataFrame({"user": rng.integers(0, 50, n), "ts": np.sort(rng.integers(0, 14 * 86400, n))})
W = 7 * 86400
# batch: count of the user's events in the previous 7 days (strictly before the event)
ev["batch"] = [((ev.user.values[:i] == u) & (ev.ts.values[:i] >= t - W)).sum() for i, (u, t) in enumerate(zip(ev.user.values[:3000], ev.ts.values[:3000]))] + [np.nan] * (n - 3000)
dq = {}; stream = []
for u, t in zip(ev.user.values, ev.ts.values):
    q = dq.setdefault(u, deque())
    while q and q[0] < t - W: q.popleft()
    stream.append(len(q)); q.append(t)
ev["stream"] = stream; chk = ev.iloc[:3000]
print(f"first 3000 events: batch and streaming 7-day counts identical: {(chk.batch == chk.stream).all()}")
# late arrival: 5% of events reach the streaming job 2 hours late (processed out of order) -> its counts drift from the batch truth
late = rng.random(n) < 0.05; order = np.argsort(ev.ts.values + late * 7200, kind="stable")
dq = {}; s2 = np.zeros(n, dtype=int)
for i in order:
    u, t = ev.user.values[i], ev.ts.values[i]; q = dq.setdefault(u, deque())
    while q and q[0] < t - W: q.popleft()
    s2[i] = len(q); q.append(t)
true_counts = np.array(stream)
print(f"with 5% of events arriving 2 h late: {(s2 != true_counts).mean():.1%} of streaming feature values differ from the in-order values")
```

```
first 3000 events: batch and streaming 7-day counts identical: True
with 5% of events arriving 2 h late: 17.2% of streaming feature values differ from the in-order values
```

With ordered events the two implementations are identical, which is the case the unit test usually covers. A 5% late-arrival rate is enough to change 17.2% of the feature values (the counts seen by later events depend on the processing order, and a late event is missing from the counts of events processed before it). The mitigations are to compute windows on event time with a watermark that bounds the allowed lateness, to define the offline feature with the same lateness rule instead of an idealised one, and to log served values so that the training data can use what the model actually saw.

**Common weak answer.** "Write both in the same SQL dialect and they will match." Matching syntax does not control for data arrival order, and a feature that depends on arrival order differs between replay and production.

**Follow-up.** "Which is the better source of truth for training: recomputed or logged features?" Logged values reproduce what the model saw but can be sparse and are tied to the old feature definitions; recomputed values allow new features but must replicate production's lateness behaviour. Many teams keep both (the author's observation).

## 6. "What is point-in-time correctness, and how do you build a training set that respects it?"

**What it tests.** Preventing label and future leakage when joining features to events.

**Reference answer.** Each training row is an event at time `t` with a label. Its features must be the values that would have been available at `t`, not the latest values and not values computed with later data. The [Feast documentation](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) describes the feature store's job in these terms: it joins features onto an entity dataframe of timestamps and entities in a point-in-time correct way, meaning it can reproduce the state of the features at a specific point in the past. The standard implementation is an as-of join: for each event, take the most recent feature row with a timestamp before the event (and, where features have a publication delay, before the event minus that delay).

The experiment builds a daily per-user feature (the running mean of the user's past outcomes, which by construction includes that day's outcomes) and joins it to the events twice with `pandas.merge_asof`: with the first snapshot at or after the event (leaky) and with the last snapshot before it (correct). A logistic regression is trained on the first 30 days and evaluated on the rest.

```python
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0); n_users, days = 2000, 40
# daily feature snapshot: the user's average spend that day; it is updated at end of day and reacts to an event that happens on that day
users = np.arange(n_users); base = rng.normal(size=n_users)
ev = pd.DataFrame({"user": rng.choice(users, 20000), "ts": rng.uniform(0, days, 20000)}).sort_values("ts")
ev["y"] = (rng.random(len(ev)) < 1 / (1 + np.exp(-(1.5 * base[ev.user.values] - 1)))).astype(int)
# feature table: one row per user per day at end of day; value = running mean of the user's labels up to and including that day (so it contains the day's labels)
rows = []
for u, g in ev.groupby("user"):
    for d in range(days):
        past = g[g.ts < d + 1].y; rows.append((u, float(d + 1), past.mean() if len(past) else np.nan))
feat = pd.DataFrame(rows, columns=["user", "avail", "f"]).dropna().sort_values("avail")
ev = ev.sort_values("ts")
naive = pd.merge_asof(ev, feat, left_on="ts", right_on="avail", by="user", direction="forward")            # first snapshot at or after the event: includes the event's own label
pit = pd.merge_asof(ev, feat, left_on="ts", right_on="avail", by="user", direction="backward")           # last snapshot available before the event
for name, df in (("naive join (snapshot taken after the event)", naive), ("point-in-time join (last snapshot before the event)", pit)):
    df = df.dropna(subset=["f"]); tr = df.ts < 30; clf = LogisticRegression().fit(df[tr][["f"]], df[tr].y)
    print(f"{name}: test AUC {roc_auc_score(df[~tr].y, clf.predict_proba(df[~tr][['f']])[:, 1]):.3f} (n={len(df)})")
```

```
naive join (snapshot taken after the event): test AUC 0.864 (n=20000)
point-in-time join (last snapshot before the event): test AUC 0.777 (n=17710)
```

The leaky join reports an AUC of 0.864 against 0.777 for the correct one, an overstatement of 0.087, and nothing in the offline numbers reveals the problem; it appears when the model reaches production, where only the earlier snapshot exists. The correct join also has fewer rows (17,710 against 20,000), because events with no earlier snapshot have no feature value, which is itself the honest situation at serving time.

**Common weak answer.** "Join on user ID and take the latest feature value." That is the leak above, often invisible because the join runs and the model trains.

**Follow-up.** "How do you detect leakage after the fact?" A single feature with implausibly high importance, an offline-to-online metric gap, and a check that every feature timestamp precedes its event timestamp.

## 7. "How do you build a forecasting model for daily demand? Where do you start?"

**What it tests.** Baselines before models, and recognising structure (trend, seasonality) in the data.

**Reference answer.** Start by looking at the series and defining the horizon and the loss, then fit the baselines. [Forecasting: Principles and Practice](https://otexts.com/fpp3/simple-methods.html) (Hyndman and Athanasopoulos) lists the benchmark methods: the mean of the history; the naive method, where every forecast is the last observation; and the seasonal naive method, where each forecast equals the last observed value from the same season, `y(T+h) = y(T+h-m(k+1))` with `m` the seasonal period and `k` the integer part of `(h-1)/m`. The book notes that the naive method works remarkably well for many economic and financial series. A candidate model is only useful if it beats these under the same evaluation.

The experiment uses 104 weeks of synthetic daily data with a weekly pattern, a trend and noise, and evaluates 14-day forecasts from 50 forecast origins (rolling, one origin per week, covered in question 8).

```python
import numpy as np
rng = np.random.default_rng(0); n = 7 * 104                       # 104 weeks of daily data, weekly seasonality, trend, noise
t = np.arange(n); y = 100 + 0.05 * t + 15 * np.sin(2 * np.pi * t / 7) + 8 * (t % 7 == 5) + rng.normal(0, 4, n)
m = 7; h = 14
def mae(a, b): return np.abs(a - b).mean()
def forecasts(train):
    T = len(train); mean = np.full(h, train.mean()); naive = np.full(h, train[-1])
    snaive = np.array([train[T + hh - m * ((hh - 1) // m + 1) - 1] for hh in range(1, h + 1)])   # y_{T+h-m(k+1)}, k = floor((h-1)/m)
    return {"mean": mean, "naive": naive, "seasonal naive": snaive}
origins = list(range(7 * 52, n - h, 7)); res = {k: [] for k in ("mean", "naive", "seasonal naive")}
for o in origins:
    f = forecasts(y[:o]); actual = y[o:o + h]
    for k in res: res[k].append(mae(f[k], actual))
print(f"rolling-origin evaluation, {len(origins)} origins, 14-day horizon (MAE):")
for k, v in res.items(): print(f"  {k:15s} {np.mean(v):6.2f}")
# a regression with calendar features, fitted at each origin, versus a random split
def design(tt): return np.column_stack([np.ones(len(tt)), tt] + [(tt % 7 == j).astype(float) for j in range(1, 7)])
errs = []
for o in origins:
    beta = np.linalg.lstsq(design(t[:o]), y[:o], rcond=None)[0]; errs.append(mae(design(t[o:o + h]) @ beta, y[o:o + h]))
print(f"  regression: trend + day-of-week {np.mean(errs):6.2f}")
```

```
rolling-origin evaluation, 50 origins, 14-day horizon (MAE):
  mean             14.25
  naive            14.26
  seasonal naive    4.47
  regression: trend + day-of-week   3.14
```

Mean and naive are equally poor here, 14.25 and 14.26, because the weekly pattern dominates; seasonal naive cuts the error to 4.47 with no fitting at all; a regression with a trend and day-of-week indicators reaches 3.14, because it also captures the trend that seasonal naive ignores. In an interview the order of this table is the answer: simple structure first, then models, then the question of whether the additional complexity (gradient boosting with calendar and lag features, or a neural forecaster) earns its cost on the same rolling evaluation.

**Common weak answer.** "Use an LSTM (or a transformer)." Without a baseline there is nothing to show that the extra model is needed, and on small series it often is not.

**Follow-up.** "Demand is intermittent, with many zeros." Then errors like MAPE break (zeros in the denominator); use a scale-free metric that tolerates zeros or evaluate the quantity that drives the decision, such as stock-outs at a service level (the author's suggestion).

## 8. "How do you evaluate a forecasting model? Why not a random train-test split?"

**What it tests.** Respecting time order in validation.

**Reference answer.** Use a split that mirrors deployment: train on the past, test on the future. The book's [time series cross-validation](https://otexts.com/fpp3/tscv.html) generalises this: there is a series of test sets, each training set contains only observations that occurred before its test set, so no future observations are used to construct a forecast; the earliest observations are not used as test sets because a small training set gives unreliable forecasts; accuracy is averaged over the test sets. The procedure is also called evaluation on a rolling forecasting origin. If multi-step forecasts matter more than one-step forecasts, the test sets use the multi-step errors at the horizon in use (the book's example is a 4-step horizon), as in question 7, where the horizon was 14 days.

A random split lets the model train on points on both sides of each test point, so it can interpolate in time, a service that does not exist in production. The size of the effect depends on the model. The test below uses gradient boosting on the time index and day of week, on a series with a steady upward trend, and compares shuffled 5-fold cross-validation with the rolling-origin evaluation of 14-day forecasts.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
rng = np.random.default_rng(0); n = 7 * 104; t = np.arange(n); y = 100 + 0.15 * t + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 4, n)
X = np.column_stack([t, t % 7]); mk = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
rand = [np.abs(mk().fit(X[a], y[a]).predict(X[b]) - y[b]).mean() for a, b in KFold(5, shuffle=True, random_state=0).split(X)]
fwd = []
for o in range(7 * 52, n - 14, 7): fwd.append(np.abs(mk().fit(X[:o], y[:o]).predict(X[o:o + 14]) - y[o:o + 14]).mean())
print(f"gradient boosting on (time index, day of week), steady upward trend:")
print(f"  shuffled 5-fold CV MAE        {np.mean(rand):6.2f}")
print(f"  rolling-origin, 14-day MAE    {np.mean(fwd):6.2f}")
```

```
gradient boosting on (time index, day of week), steady upward trend:
  shuffled 5-fold CV MAE          3.58
  rolling-origin, 14-day MAE      4.41
```

The shuffled split reports an error of 3.58 against 4.41 for the honest evaluation, an understatement of about a fifth. The gap here is moderate because the model has few features; with lag features, or models that memorise neighbouring points, it can be much larger, and those cases were not tested here. Other points to state: keep any hyperparameter tuning inside the training window of each origin, and hold out a final block that is touched once.

**Common weak answer.** "5-fold cross-validation, as usual." Standard K-fold assumes exchangeable rows, which a time series is not.

**Follow-up.** "You have only two years of data and strong yearly seasonality." Then there are only one or two examples of each seasonal position; keep the rolling evaluation but expect high variance in the estimate, report the spread across origins, and prefer simple seasonal structure over flexible models.

## 9. "A stakeholder wants a forecast. What do you give them besides a number?"

**What it tests.** Communicating uncertainty and checking that it is calibrated.

**Reference answer.** A point forecast together with a prediction interval, and the interval's coverage checked on held-out origins. The forecasting book explains the dependence: the usual formulas for prediction intervals assume that the residuals have constant variance and are normally distributed, which makes the calculation easier, but a method that does not satisfy these properties is not necessarily improvable, and an alternative way of obtaining the intervals is then needed (the book's section 5.5 covers bootstrapped residuals). Two practical routes follow: an interval from the spread of past forecast errors at the same horizon (empirical quantiles), or quantile regression (training the model to output the 10th and 90th percentiles directly, for example with a pinball loss).

The test builds an 80% interval for the seasonal naive forecast one week ahead, in two ways, from the standard deviation of the in-sample seasonal differences (normal assumption) and from their empirical 10% and 90% quantiles, and measures how often the realised values fall inside across many origins.

```python
import numpy as np
rng = np.random.default_rng(0); n = 7 * 104; t = np.arange(n); y = 100 + 0.05 * t + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 4, n); m, h = 7, 7
cov = {"normal": [], "empirical": []}
res_all = []
for o in range(7 * 40, n - h, 3):
    tr = y[:o]; res = tr[m:] - tr[:-m]                                 # seasonal naive residuals (in-sample)
    sd = res.std(ddof=1); f = tr[o - m:o][:h]; actual = y[o:o + h]
    cov["normal"].append(np.mean(np.abs(actual - f) <= 1.28 * sd))                       # 80% interval from residual sd
    lo, hi = np.quantile(res, [0.1, 0.9]); cov["empirical"].append(np.mean((actual >= f + lo) & (actual <= f + hi)))
for k, v in cov.items(): print(f"{k:13s} nominal 80% interval, seasonal naive, 1-week horizon: actual coverage {np.mean(v):.3f}")
```

```
normal        nominal 80% interval, seasonal naive, 1-week horizon: actual coverage 0.799
empirical     nominal 80% interval, seasonal naive, 1-week horizon: actual coverage 0.779
```

Both intervals cover close to the nominal 80% (0.799 and 0.779), as they should here, because the simulated noise is Gaussian with constant variance. On real data with heavy tails or variance that changes with the level, the normal interval is the one expected to fail; the check itself is the reusable part: compute the coverage on held-out origins, and if it is far from nominal, change the method before presenting the interval. Also note that the interval should widen with the horizon, and that decisions (stock levels, staffing) usually depend on a quantile, not on the mean.

**Common weak answer.** "The model's RMSE is the uncertainty." RMSE is an average over cases and horizons, not an interval for a particular forecast.

**Follow-up.** "The business needs a 95% service level on stock." Forecast the 95th percentile of demand over the replenishment lead time directly, instead of the mean plus a multiple of an error estimate.

## 10. "How would you model the data for these systems? What tables do you need?"

**What it tests.** Turning the leakage and delay issues above into a schema.

**Reference answer.** Keep four things separate, each with its own timestamps. This is the author's design, consistent with the point-in-time idea in the Feast documentation, not a prescription from a source:

- **Events** (transactions, impressions, orders): immutable and append-only, with the time the event happened (`event_ts`) and the time the system received it (`ingested_ts`). The gap between the two is what late-arrival handling (question 5) needs.
- **Labels**: a separate table keyed by event ID, with the time the label became known (`labeled_ts`). A label is a fact about the event that is learned later, so storing it on the event row would erase when it was known.
- **Feature snapshots**: keyed by entity and feature timestamp, written by the pipeline that serves the feature, never updated in place, so any past state can be reproduced.
- **Entity tables** for slowly changing attributes, with valid-from and valid-to ranges if history matters.

A training set "as of time `T`" is then a query: events with their last feature snapshot before the event, and labels with `labeled_ts <= T`. The example below builds this in `sqlite3` with three transactions of one user, one of which is later confirmed as fraud, and builds the training set at two dates.

```python
import sqlite3
db = sqlite3.connect(":memory:"); c = db.cursor()
c.executescript("""
CREATE TABLE txn (txn_id INTEGER PRIMARY KEY, user_id INT, amount REAL, event_ts INT, ingested_ts INT);
CREATE TABLE label (txn_id INT PRIMARY KEY, is_fraud INT, labeled_ts INT);
CREATE TABLE user_feat (user_id INT, feat_ts INT, avg_amount_7d REAL, PRIMARY KEY (user_id, feat_ts));
INSERT INTO txn VALUES (1,7,40,100,101),(2,7,900,200,201),(3,7,55,300,301);
INSERT INTO label VALUES (2,1,500);
INSERT INTO user_feat VALUES (7,50,45.0),(7,150,45.0),(7,250,328.3),(7,350,331.7);
""")
q = """SELECT t.txn_id, t.event_ts, f.feat_ts, f.avg_amount_7d, l.is_fraud, l.labeled_ts
FROM txn t
JOIN user_feat f ON f.user_id = t.user_id AND f.feat_ts = (SELECT MAX(feat_ts) FROM user_feat WHERE user_id = t.user_id AND feat_ts < t.event_ts)
LEFT JOIN label l ON l.txn_id = t.txn_id AND l.labeled_ts <= :asof
ORDER BY t.txn_id"""
for asof in (400, 600):
    print(f"training set as of t={asof}: (txn, event_ts, feat_ts, avg_amount_7d, is_fraud, labeled_ts)")
    for r in c.execute(q, {"asof": asof}): print("  ", r)
```

```
training set as of t=400: (txn, event_ts, feat_ts, avg_amount_7d, is_fraud, labeled_ts)
   (1, 100, 50, 45.0, None, None)
   (2, 200, 150, 45.0, None, None)
   (3, 300, 250, 328.3, None, None)
training set as of t=600: (txn, event_ts, feat_ts, avg_amount_7d, is_fraud, labeled_ts)
   (1, 100, 50, 45.0, None, None)
   (2, 200, 150, 45.0, 1, 500)
   (3, 300, 250, 328.3, None, None)
```

At `T = 400` the fraud label of transaction 2 (known at 500) is not available, and the row appears with no label; at `T = 600` it is. The transactions that never get a label show `None` at both dates: that is "unknown", which is different from "genuine", and the training code should either wait for the maturity window of question 2 or exclude them. The feature used for transaction 3 is the snapshot at 250 (328.3), which already reflects the large transaction 2, because that happened before it.

Operational points to add in an interview: make backfills idempotent (re-running a day overwrites the same partition), version feature definitions, keep entity IDs stable across systems, and test the join with a synthetic case that has a known answer, such as the one above.

**Common weak answer.** "One wide table with the latest features and the label as a column." It cannot answer what the model would have seen at a past time, and it mixes the label-delay problem into every row.

**Follow-up.** "A feature definition changes. What do you do with the history?" Backfill the new definition from the raw events into a new feature version, train on it, and keep the old version served until the new model is live.

## Takeaways

- Fraud is a decision problem with an alert budget and costs; evaluate at the operating point, and set thresholds from costs on calibrated probabilities. In the toy test, the cost-ratio threshold of 0.0476 sat at the minimum and the default 0.5 cost nearly twice as much.
- Labels arrive late and the pattern drifts: in the simulation, a 7-day label delay cut alert precision from 0.631 to 0.423. The real-data paper recommends handling recent feedback labels and delayed labels separately.
- Decide freshness per feature, not per system; two code paths for the same feature are a maintenance and a skew risk.
- A streaming feature that matched the batch version on ordered data differed on 17.2% of values once 5% of events arrived late.
- Join features to events as of the event time; the leaky join overstated AUC by 0.087 in the experiment.
- For forecasts: seasonal naive first (4.47 against 14.25 for the mean in the test), then richer models, evaluated on rolling origins, with intervals whose coverage has been checked.
- Store events, labels, and feature snapshots separately with their own timestamps, so any past training set can be rebuilt.

## Limits

- All experiments use synthetic data on one CPU machine; the numbers show mechanisms (delay under drift, skew from late events, leakage, split leakage) and do not transfer to real portfolios. The fraud simulation omits the alert-feedback loop, adversarial adaptation, and the cost of the alerts themselves.
- Claims about papers and documentation are limited to the passages read: the two Dal Pozzolo et al. papers (abstract and introduction passages), Kreps' 2014 post, the Feast point-in-time documentation page, and chapter 5 of Forecasting: Principles and Practice. The cited positions are those authors' at the time of writing.
- Items marked as the author's own (the answer framework for fraud, the latency tiers, the schema design, the quantile and service-level suggestions, the late-event mitigations) have no source behind them.
- Not covered: graph-based fraud detection, anomaly detection without labels beyond a mention, hierarchical and probabilistic forecasting methods, deep forecasting models, forecast reconciliation, and the full design of a feature store (backfills, online store consistency).
- The candidates in the examples are hypothetical, and the reference answers are one reasonable answer among several.

Next: Part 11, deployment and MLOps.
