---
title: "ML Engineer Interviews, With Reference Answers (11/18): Deployment and MLOps"
date: 2027-01-06 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, mlops, serving, monitoring, drift, series]
description: "Part 11 of 18. Ten questions on putting models into production: tail latency, batching, rollouts, monitoring, drift detection, retraining schedules and technical debt, with runnable code."
---

This post covers the deployment and operations questions of the ML interview, after [Part 10](/posts/ml-engineer-interviews-10-ml-system-design-2-fraud-forecasting-and-data-modelling/): what to test before release, latency and batching, rollouts, monitoring, drift, retraining and technical debt. It gives ten questions with what each tests, a reference answer, a common weak answer and a follow-up. The numbers in the answers come from code that was run for this post, on synthetic data, with the output shown; claims about papers and documentation are limited to the passages that were read. Timing numbers (milliseconds, requests per second) depend on the machine and vary by a few percent between runs.

The code ran on a CPU-only machine (32 cores, 4 threads used for the PyTorch timing) with Python 3.14.7, NumPy, SciPy, scikit-learn and PyTorch 2.15.0, with fixed seeds. The data are synthetic, so the experiments check mechanisms; absolute numbers will differ on real systems.

![Left: measured throughput of a three-layer network on CPU against batch size: 11.2, 26.6, 69.6, 109.1 and 109.4 thousand requests per second for batches of 1, 4, 16, 64 and 256, with latency per call from 0.09 to 2.34 ms. Right: AUC over the last 30 days under slow drift for retraining schedules: never 0.754, every 30 days 0.896, every 7 days 0.908, every day 0.910.](/assets/pic/ml-interview-11-batching-retraining.png)
*Left: the throughput and latency trade-off of batching (question 3). Right: retraining frequency under drift (question 7).*

## 1. "What has to be true before a model goes to production? How do you test an ML system?"

**What it tests.** Whether the candidate knows that most failures in production are in the system around the model.

**Reference answer.** The model is one component; the data pipeline, the features, the serving code and the monitoring fail more often than the algorithm. The [ML Test Score paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf) from Google (Breck et al.) offers a rubric of 28 tests and monitoring needs drawn from production systems, grouped into tests for features and data, tests for model development, tests for the ML infrastructure, and monitoring tests. Its framing is that training is like compilation where the source is both code and data, so training data needs testing like code, and a trained model needs production practices like a binary: debuggability, rollbacks and monitoring. Each test is written as an assertion to be checked regularly, and the paper notes that the rubric leaves out generic software practice (unit tests, release process), which remains necessary.

A compact version for an interview, in the order of the pipeline:

1. **Data**: input schemas and invariants are checked, and features have owners and documented meaning.
2. **Training**: the pipeline is reproducible, hyperparameters are tuned, and the model beats a very simple baseline (a "simpler model is not better" test in the paper); quality is checked on important slices and not only overall.
3. **Offline to online**: the relation between the offline proxy metric and the online business metric is known (the paper suggests measuring it with small A/B experiments, for example with an intentionally degraded model).
4. **Serving**: the model is tested end to end in the serving binary, is rolled out gradually, and can be rolled back quickly (questions 4 and 9).
5. **Monitoring**: inputs, predictions and quality are watched after launch (question 5).

**Common weak answer.** "Check the accuracy on the test set and deploy." It addresses the model in isolation, and the paper's point is that this is the easy part.

**Follow-up.** "You have one day. Which three checks do you add first?" A schema and range check on the inputs, a comparison of served and offline feature values (Part 10, question 5), and a one-command rollback that has been practised; the choice is the author's judgment, but each of the three is also one of the paper's recommended tests.

## 2. "The service must answer in 100 ms at the 99th percentile and calls many backends. What do you worry about?"

**What it tests.** Tail latency and fan-out, which averages hide.

**Reference answer.** A request that waits for many parallel calls is as slow as the slowest of them, so rare slow responses become common. Dean and Barroso's [The Tail at Scale](https://www.barroso.org/publications/TheTailAtScale.pdf) gives the example: if a server answers in 10 ms typically but has a 99th-percentile latency of one second, then a request served by one such server is slow one time in 100; if a request must collect responses from 100 such servers in parallel, 63% of requests take more than one second. They list causes of variability (shared resources, background daemons, global resource sharing and maintenance activities, among others) and a family of techniques that tolerate it, including hedged requests: send the request to the replica believed best, send a second copy to another replica after a short delay if there is no answer yet, use whichever answers first and cancel the other. The paper warns that naive versions add unacceptable extra load and that variations exist that obtain most of the gain with a small increase.

The simulation reproduces the paper's arithmetic and tests hedging. Each server call takes about 10 ms (log-normal), except that 1% of calls take 1,000 ms.

```python
import numpy as np
rng = np.random.default_rng(0); N, S = 20000, 100                          # requests, servers per request
def server(size):                                                           # 10 ms typical, 1% of calls take 1 s
    return np.where(rng.random(size) < 0.01, 1000.0, rng.lognormal(np.log(10), 0.3, size))
t1 = server((N, S)); single = t1[:, 0]
print(f"one server:  p50 {np.percentile(single, 50):6.1f} ms  p99 {np.percentile(single, 99):7.1f} ms  slower than 1 s: {np.mean(single >= 1000):.3f}")
fan = t1.max(axis=1)
print(f"100 servers: p50 {np.percentile(fan, 50):6.1f} ms  p99 {np.percentile(fan, 99):7.1f} ms  slower than 1 s: {np.mean(fan >= 1000):.3f}   (1 - 0.99**100 = {1 - 0.99**100:.3f})")
h = 20.0; t2 = server((N, S)); hedged = np.minimum(t1, h + t2).max(axis=1)  # resend to a second replica if no reply after 20 ms, keep the first answer
print(f"hedged after {h:.0f} ms: p50 {np.percentile(hedged, 50):6.1f} ms  p99 {np.percentile(hedged, 99):7.1f} ms  slower than 1 s: {np.mean(hedged >= 1000):.3f}   extra requests: {np.mean(t1 > h):.1%}")
```

```
one server:  p50   10.0 ms  p99  1000.0 ms  slower than 1 s: 0.011
100 servers: p50 1000.0 ms  p99  1000.0 ms  slower than 1 s: 0.631   (1 - 0.99**100 = 0.634)
hedged after 20 ms: p50   28.7 ms  p99    43.5 ms  slower than 1 s: 0.008   extra requests: 2.0%
```

With one server, 1.1% of calls are slow; with 100 in parallel, 63.1% of requests are slow, close to the analytic 63.4%. Hedging after 20 ms cuts the share of slow requests to 0.8% and brings the 99th percentile to 43.5 ms, at the price of 2.0% extra requests. These figures come from an invented latency distribution in which the slowness is independent between replicas; if the slow calls are caused by a shared dependency, the second copy is slow too and hedging does not help.

**Common weak answer.** "Average latency is 12 ms, so we are fine." The average and even the median describe almost none of the requests that the user experiences as slow when fan-out is large.

**Follow-up.** "What do you measure?" Percentiles per endpoint and per backend, with the request fan-out recorded, since the percentile of the whole request is not the percentile of its parts.

## 3. "How do you raise the throughput of a model server? When does batching help, and what does it cost?"

**What it tests.** The throughput and latency trade-off, and dynamic batching.

**Reference answer.** A model call has a fixed cost (loading weights from memory, launching kernels, Python and framework overhead) and a cost per request, so combining requests raises throughput. The [NVIDIA Triton documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) describes dynamic batching in these terms: the server combines inference requests into a batch created dynamically, which typically increases throughput, for stateless models; the settings control the preferred batch sizes and the maximum time a request may be delayed in the scheduler to let others join. The cost is latency: a request that waits for companions waits, and a bigger batch takes longer to run.

The measurements below time a three-layer network (256 to 1024 to 1024 to 10) on CPU at several batch sizes, fit a linear cost model (fixed plus per-request time), and simulate a single server with Poisson arrivals under two policies: no batching, and batching of up to 64 requests with a wait of at most 2 ms.

```python
import time, numpy as np, torch
torch.manual_seed(0); torch.set_num_threads(4)
net = torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 10)).eval()
cost = {}
with torch.no_grad():
    for b in (1, 4, 16, 64, 256):
        x = torch.randn(b, 256); [net(x) for _ in range(20)]
        ts = []
        for _ in range(200): t0 = time.perf_counter(); net(x); ts.append(time.perf_counter() - t0)
        cost[b] = np.median(ts) * 1000; print(f"batch {b:3d}: {cost[b]:6.3f} ms per call, {b / cost[b] * 1000:9.0f} requests/s")
bs = np.array(list(cost)); ms = np.array(list(cost.values())); slope, icpt = np.polyfit(bs, ms, 1)
print(f"linear fit: {icpt:.3f} ms fixed + {slope:.4f} ms per extra request")
rng = np.random.default_rng(0)
def sim(rate, max_b, max_wait, n=40000):                                    # one server, Poisson arrivals (rate in requests/s), times in ms
    arr = np.cumsum(rng.exponential(1000 / rate, n)); lat = np.zeros(n); free = 0.0; i = 0
    while i < n:
        start = max(free, arr[i]); deadline = arr[i] + max_wait; j = i + 1   # the oldest waiting request may be held until its deadline
        while j < n and j - i < max_b and arr[j] <= max(start, deadline): j += 1
        launch = max(start, arr[j - 1]) if j - i == max_b else max(start, deadline)
        done = launch + icpt + slope * (j - i); lat[i:j] = done - arr[i:j]; free = done; i = j
    return lat
for rate in (3000, 9000, 30000):
    for name, mb, mw in (("no batching", 1, 0.0), ("batch up to 64, wait up to 2 ms", 64, 2.0)):
        l = sim(rate, mb, mw); print(f"{rate:6d} req/s, {name:32s}: p50 {np.percentile(l, 50):9.2f} ms, p99 {np.percentile(l, 99):9.2f} ms")
```

```
batch   1:  0.090 ms per call,     11166 requests/s
batch   4:  0.150 ms per call,     26605 requests/s
batch  16:  0.230 ms per call,     69571 requests/s
batch  64:  0.587 ms per call,    109079 requests/s
batch 256:  2.341 ms per call,    109363 requests/s
linear fit: 0.081 ms fixed + 0.0088 ms per extra request
  3000 req/s, no batching                     : p50      0.09 ms, p99      0.24 ms
  3000 req/s, batch up to 64, wait up to 2 ms : p50      1.32 ms, p99      2.18 ms
  9000 req/s, no batching                     : p50      0.22 ms, p99      1.15 ms
  9000 req/s, batch up to 64, wait up to 2 ms : p50      1.31 ms, p99      2.29 ms
 30000 req/s, no batching                     : p50   1132.79 ms, p99   2228.92 ms
 30000 req/s, batch up to 64, wait up to 2 ms : p50      1.58 ms, p99      2.59 ms
```

Throughput rises about 10 times from batch 1 to batch 64 (11.2 to 109.1 thousand requests per second) and then flattens, while the time per call grows from 0.09 to 0.59 ms. In the simulation, at light and moderate load (3,000 and 9,000 requests per second) no batching is better, because batching adds up to the 2 ms wait; at 30,000 requests per second, above the 11,000 that unbatched serving can sustain, the unbatched queue grows without bound (the 1.1 s median is an artefact of the length of the run), while the batched server stays at 1.6 ms median and 2.6 ms at the 99th percentile. The practical rule is to set the maximum wait from the latency budget (question 2), measure the knee of the throughput curve, and do not batch when the load is low. On GPUs the same logic holds with a larger fixed cost and a larger knee. This experiment ran on a CPU and does not show GPU numbers.

**Common weak answer.** "Always use the largest batch." Past the knee throughput does not improve, latency does get worse, and the tail suffers.

**Follow-up.** "Other levers when batching is not enough?" Smaller or distilled models, quantisation, caching of repeated requests, more replicas, and moving work to precomputation (Part 9, question 5 and Part 10, question 4).

## 4. "How do you release a new model safely?"

**What it tests.** Release discipline: staged exposure, and statistical power for the comparison.

**Reference answer.** Stage the exposure. The ML Test Score paper recommends turning up new models gradually, running old and new models concurrently, with the new model seeing a small fraction of traffic that is increased as it is observed to behave sanely; it also recommends a model canary before serving, because a model can include code paths that are not live in the separate serving binary, causing havoc at serving time. A common sequence (the author's framing, with the paper's two ideas inside it): offline evaluation on a temporal split, then shadow mode (the new model scores live traffic and its outputs are logged and not used), then a small canary percentage, then a controlled experiment, then full rollout with the old version kept for rollback (question 9).

The statistical question is how much traffic the canary needs. To detect a rise in the error rate from 5.0% to 5.5% with 80% power at a 5% level, the standard two-proportion sample size is computed and checked by simulation (one-sided tests at the 2.5% level, so the false alarm rate should be about 2.5%).

```python
import numpy as np
from scipy.stats import norm
p0, p1, alpha, power = 0.050, 0.055, 0.05, 0.80                                # error rate of the live model, and a regression worth catching
za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power); pb = (p0 + p1) / 2
n = ((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2) / (p1 - p0) ** 2
print(f"requests per arm to detect {p0:.1%} -> {p1:.1%} (two-sided alpha {alpha}, power {power:.0%}): {int(np.ceil(n))}")
rng = np.random.default_rng(0)
def detect(n, a, b, trials=4000):
    x, y = rng.binomial(n, a, trials), rng.binomial(n, b, trials); pp = (x + y) / (2 * n)
    z = (y - x) / n / np.sqrt(2 * pp * (1 - pp) / n); return np.mean(z > za)
for m in (int(n), int(n) // 4):
    print(f"n = {m:6d} per arm: regression detected in {detect(m, p0, p1):.1%} of trials; false alarm rate with no change {detect(m, p0, p0):.1%}")
```

```
requests per arm to detect 5.0% -> 5.5% (two-sided alpha 0.05, power 80%): 31234
n =  31233 per arm: regression detected in 80.2% of trials; false alarm rate with no change 2.6%
n =   7808 per arm: regression detected in 27.3% of trials; false alarm rate with no change 2.3%
```

The answer is about 31,000 requests per arm. A canary sized at a quarter of that detects the regression in only 27% of trials, so "nothing alarmed during the canary" would be weak evidence of safety. Small regressions need large samples; the canary therefore watches large effects (crashes, latency, null predictions, a collapse of a key rate) and the experiment watches small ones.

**Common weak answer.** "Deploy to 1% of users for an hour and see if the metrics look fine." Without a sample size, "fine" cannot be told from "too few observations to see a problem".

**Follow-up.** "The quality metric needs labels that arrive days later." Then the canary uses proxies available immediately (score distributions, latency, error codes, agreement with the old model) and the label-based check happens afterwards; keep the old model available until that check passes.

## 5. "What do you monitor after launch?"

**What it tests.** Knowing which signals catch which failures.

**Reference answer.** The ML Test Score paper lists seven monitoring tests: dependency changes result in notification; data invariants hold for inputs; training and serving are not skewed; models are not too stale; models are numerically stable; computing performance has not regressed; prediction quality has not regressed. Grouping them by what they detect:

- **System health**: latency percentiles, error rates, throughput, resource use (question 2).
- **Input data**: schema, ranges, null rates and category frequencies for each feature, and the age of upstream tables (the paper's example is a top-N table whose refresh process stops, which is only noticed if its age is monitored).
- **Predictions**: the distribution of scores and the rate of each predicted class; shifts here are cheap to detect because no labels are needed.
- **Quality**: the real metric once labels arrive, per slice, with the label delay in mind (Part 10, question 2).
- **Skew**: served versus offline feature values.

Alerts should lead to a runbook with an owner (the author's addition), since a dashboard that nobody acts on is not monitoring. For drift in the inputs specifically, the choice of test matters (question 6).

**Common weak answer.** "Track accuracy on a dashboard." Accuracy needs labels, which may take days or never arrive; the cheaper input and prediction monitors catch most problems earlier.

**Follow-up.** "Accuracy is stable but revenue fell." The offline proxy and the business metric have diverged, or a slice (a segment, a market) degraded while the average held; check slices and the offline-to-online relation (question 1).

## 6. "How do you detect data drift? What goes wrong with naive approaches?"

**What it tests.** Choosing a two-sample test sensibly in many dimensions.

**Reference answer.** Drift detection is a two-sample problem: are the production inputs drawn from the same distribution as the training data? The paper [Failing Loudly](https://arxiv.org/abs/1810.11953) (Rabanser, Günnemann and Lipton) states that for simple univariate distributions such hypothesis testing is a mature science, while best practice for two-sample tests on high-dimensional data remains an open question; kernel-based multivariate tests scale badly with data size and lose statistical power in high dimensions. Their empirical study compares dimensionality-reduction methods followed by tests, and finds that across the shifts they explore, a two-sample test applied after dimensionality reduction with pre-trained classifiers performs best, while domain-classifying approaches help to characterise a shift and judge whether it is harmful.

Two pitfalls follow, both shown below. First, with many features, per-feature tests raise false alarms. Second, per-feature tests cannot see a change in the relation between features. The experiment runs 100 per-feature Kolmogorov-Smirnov tests on data with no shift, then compares a per-feature test with a domain classifier (a model trained to tell reference rows from current rows) on a shift that changes only the correlation between two features.

```python
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
rng = np.random.default_rng(0)
# 1) many features, no real shift: how many per-feature tests fire?
ref, cur = rng.normal(size=(5000, 100)), rng.normal(size=(5000, 100))
p = np.array([ks_2samp(ref[:, j], cur[:, j]).pvalue for j in range(100)])
print(f"no shift, 100 features: {np.sum(p < 0.05)} KS tests with p < 0.05; with Bonferroni (p < 0.05/100): {np.sum(p < 0.0005)}")
# 2) a shift that changes the correlation between two features but leaves every marginal unchanged
cov0, cov1 = np.eye(5), np.eye(5); cov1[0, 1] = cov1[1, 0] = 0.8
a, b = rng.multivariate_normal(np.zeros(5), cov0, 5000), rng.multivariate_normal(np.zeros(5), cov1, 5000)
pm = [ks_2samp(a[:, j], b[:, j]).pvalue for j in range(5)]
print("correlation shift, per-feature KS p-values:", " ".join(f"{v:.2f}" for v in pm))
X = np.vstack([a, b]); y = np.r_[np.zeros(5000), np.ones(5000)]
auc = cross_val_score(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0), X, y, cv=5, scoring="roc_auc").mean()
print(f"domain classifier (reference vs current), cross-validated AUC: {auc:.3f}  (0.5 means indistinguishable)")
```

```
no shift, 100 features: 3 KS tests with p < 0.05; with Bonferroni (p < 0.05/100): 0
correlation shift, per-feature KS p-values: 0.96 0.37 0.56 0.05 0.59
domain classifier (reference vs current), cross-validated AUC: 0.715  (0.5 means indistinguishable)
```

With no shift, 3 of 100 tests fire at the 5% level (about 5 are expected by chance), and none after a Bonferroni correction. For the correlation shift, the five marginal tests show nothing meaningful (one borderline value at 0.05 among five, consistent with chance), while the domain classifier separates the two samples with a cross-validated AUC of 0.715. In practice use both: cheap per-feature statistics (PSI or KS with a multiple-testing rule, and a threshold on effect size and not only the p-value, since large samples make tiny differences significant) for explanation, and a classifier-based test for detection; and ask whether a shift is harmful by looking at the prediction distribution and the quality metrics, since a shift can be benign.

**Common weak answer.** "Run a KS test on every feature every day and alert on p < 0.05." On 100 features that alerts about 5 times a day by chance alone, and misses joint changes.

**Follow-up.** "Drift is detected. Now what?" Check whether it is real (a pipeline bug is more common than a real change), whether it hurts quality, then retrain on recent data or fix the upstream cause (question 7).

## 7. "How often should you retrain?"

**What it tests.** Treating the retraining schedule as a measurable trade-off.

**Reference answer.** Measure how fast quality decays with model age, then trade it against the cost of retraining. The ML Test Score paper describes how to measure it: test the effect of an old model on live metrics with a small A/B experiment, and test a range of ages to obtain an age-versus-quality curve that shows how much staleness is tolerable; it recommends monitoring the age of the production model against that threshold. It also notes that infrequently updated models carry a maintenance cost of their own: a model retrained by hand once or twice a year may be hard to retrain when the engineer who knew the steps leaves.

The simulation uses a classifier whose decision direction rotates slowly (about 62 degrees over the 90 evaluated days), a 14-day training window, and four schedules.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0); days, per_day, d = 120, 1500, 6
def w(day): a = day * 0.012; v = np.zeros(d); v[0], v[1] = np.cos(a), np.sin(a); v[2] = 0.5; return v
X = rng.normal(size=(days * per_day, d)); day = np.repeat(np.arange(days), per_day)
y = np.concatenate([rng.random(per_day) < 1 / (1 + np.exp(-(2.5 * X[day == t] @ w(t) - 1))) for t in range(days)])
def run(every, window=14, first=30):
    clf, aucs, trains = None, [], 0
    for t in range(first, days):
        if clf is None or (every and (t - first) % every == 0):
            m = (day >= t - window) & (day < t); clf = LogisticRegression(max_iter=300).fit(X[m], y[m]); trains += 1
        m = day == t; aucs.append(roc_auc_score(y[m], clf.decision_function(X[m])))
    return np.mean(aucs), np.mean(aucs[-30:]), trains
for name, ev in (("never retrained after day 30", 0), ("every 30 days", 30), ("every 7 days", 7), ("every day", 1)):
    a, last, n = run(ev); print(f"{name:28s}: mean AUC {a:.3f}, last 30 days {last:.3f}, trainings {n}")
```

```
never retrained after day 30: mean AUC 0.832, last 30 days 0.754, trainings 1
every 30 days               : mean AUC 0.897, last 30 days 0.896, trainings 3
every 7 days                : mean AUC 0.909, last 30 days 0.908, trainings 13
every day                   : mean AUC 0.910, last 30 days 0.910, trainings 90
```

Never retraining loses 15 points of AUC over the last 30 days (0.754 against 0.908 for weekly retraining), monthly retraining recovers most of the loss (0.896), and weekly or daily retraining are within 0.002 of each other although daily costs seven times as many training runs (90 against 13). So the answer for this drift speed is weekly; for a faster-drifting problem such as fraud (Part 10, question 2) the answer shifts toward daily, and for a slow one toward monthly. Automate retraining, validate each new model against the current one before promotion (an unattended retrain on corrupted data is a failure mode), and trigger extra retraining from the monitors of question 6.

**Common weak answer.** "Retrain every night, to be safe." It costs compute and operational risk (each retrain can ship a bad model) for no measured gain, as the daily row shows.

**Follow-up.** "What is the window?" It is a second knob: a short window tracks drift but has more variance, and a long window is stable but slow; treat it as a hyperparameter selected on rolling evaluation (Part 10, question 8).

## 8. "What is technical debt in ML systems?"

**What it tests.** Knowing the failure modes that appear only after months of maintenance.

**Reference answer.** The reference is Sculley et al., [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf). Its opening observation is that developing and deploying ML systems is relatively fast and cheap, but maintaining them over time is difficult and expensive. The paper lists risk factors specific to ML: boundary erosion, entanglement, hidden feedback loops, undeclared consumers, data dependencies, configuration issues, and changes in the external world. Four to be able to explain:

- **Entanglement (CACE).** Changing Anything Changes Everything: changing the distribution of one input, or adding or removing a feature, can change the behaviour of all the others, and the paper says the same applies to hyperparameters, sampling and thresholds.
- **Unstable data dependencies.** Input signals produced by other systems can change behaviour over time, for example when an input is itself a model that is updated, or when its owner is another team; the paper's common mitigation is to consume a versioned copy of the signal, which does not change until the model is deliberately moved to a new version.
- **Underutilised data dependencies.** Input signals that add little modelling benefit still make the system vulnerable to change and could be removed with no loss.
- **Dead experimental codepaths.** Experiments implemented as conditional branches inside production code accumulate and become hard to reason about.

The small test below shows entanglement in its simplest form. Two features share an upstream signal, and removing one changes the weight of the other.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0); n = 50000
z = rng.normal(size=n); x1, x2 = z + rng.normal(size=n), z + rng.normal(size=n)
y = rng.random(n) < 1 / (1 + np.exp(-(0.6 * x1 + 0.6 * x2 + 0.0)))
def fit(*cols): c = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack(cols), y); return " ".join(f"{v:6.3f}" for v in c.coef_[0])
x3 = z + rng.normal(size=n)                                        # a new feature built from the same upstream signal
print("coefficients (x1, x2, ...):")
print("  x1, x2          :", fit(x1, x2)); print("  x3 added        :", fit(x1, x2, x3)); print("  x2 removed      :", fit(x1))
```

```
coefficients (x1, x2, ...):
  x1, x2          :  0.600  0.607
  x3 added        :  0.601  0.608 -0.003
  x2 removed      :  0.820
```

A coefficient of 0.600 for the first feature becomes 0.820 when the second is removed, because the first now has to carry what both carried. The model is the same family and the data are the same; only the feature set changed. That is the reason features are removed through a retraining and evaluation, never by a quiet edit of the serving code.

**Common weak answer.** "Technical debt is messy code." The paper's point is that ML adds debt at the system level (data dependencies, feedback loops, entanglement), where code quality does not reach.

**Follow-up.** "How do you pay it down?" Remove features whose ablation shows no loss, version and document the signals you consume, delete dead experiment branches, and add the tests and monitors of questions 1 and 5.

## 9. "A model's predictions influence its own future training data. What is the risk, and how do you manage it? What about rollback?"

**What it tests.** Feedback loops, and the discipline of reversibility.

**Reference answer.** Sculley et al. describe direct feedback loops: a model may directly influence the selection of its own future training data. They note that the theoretically correct solution would be a bandit algorithm, which does not necessarily scale to the action spaces of real problems, and that the effect can be mitigated by some amount of randomisation, or by isolating parts of the data from being influenced by the model. A hidden feedback loop is harder to see: two systems influence each other indirectly through the world, for example when one system selects the products shown on a page and another selects the related reviews, so that improving one changes how users behave toward the other. Examples in practice: a recommender trains on clicks on what it showed (Part 9, question 8), and a fraud system trains on labels from the transactions it flagged (Part 10, question 2). Mitigations: log the propensity or a randomised holdout slice, keep an exploration share, and evaluate on data the model did not choose.

Rollback is the other half of safety. The ML Test Score paper lists as an infrastructure test that models can be quickly and safely rolled back to a previous serving version, calls rollback a key part of incident response, and says that because it is an emergency procedure, operators should practise it. This needs the previous model, its feature definitions, and its configuration to be retained and deployable, which is a reason to version them together (the author's inference).

**Common weak answer.** "Retrain on the latest data." If the latest data was shaped by the model's own decisions, retraining reinforces its blind spots.

**Follow-up.** "How big should the randomised slice be?" Large enough to measure the quantities you care about with the precision of question 4's calculation, and small enough that the cost of the lost optimisation is acceptable; it is a business decision.

## 10. "Choose between batch prediction, online serving and on-device inference."

**What it tests.** Matching the serving pattern to the requirements.

**Reference answer.** This answer is the author's reasoning, built on the measurements above; no source is cited for the framework. The three patterns differ in when the prediction is computed:

- **Batch prediction**: compute predictions for all entities on a schedule and store them. Cheap per prediction, simple, and every output can be inspected before release; stale by the schedule, and wasteful when most entities are never queried. Fits nightly recommendations, churn lists and risk scores that change slowly.
- **Online serving**: compute on request. Fresh, and able to use request-time context; needs the latency engineering of questions 2 and 3, and a feature pipeline that is consistent between training and serving (Part 10, question 5).
- **On-device**: run inside the app or device. No network round trip, works offline, keeps raw data on the device; the model must be small, updates reach users slowly, and monitoring is harder.

A simple cost comparison goes first: if the table has 5 million users and 200,000 are active per day, batch scoring does 25 times the work of scoring on request, but at 0.09 ms per call (the measurement of question 3, for a small model) both are trivial, and the decision is about freshness and complexity, not compute. For a model that takes 50 ms per call the same arithmetic would favour computing on request for the active users only. Many systems combine them: batch for the slow part (user embeddings), online for the fast part (the request context and recent actions).

**Common weak answer.** "Online is better because it is real time." It adds latency risk, on-call burden and a skew risk for a freshness that the decision may not need.

**Follow-up.** "The product wants a score within 10 ms of a click, with a large model." Precompute what can be precomputed, serve a smaller model on the hot path, and move the large model to an asynchronous path whose output arrives later.

## Takeaways

- Test the system around the model: data, features, serving, rollback and monitoring; the 28-test rubric of the ML Test Score paper is a good checklist.
- Tail latency dominates with fan-out: 1% slow calls make 63% of 100-way requests slow, and a hedged request cut the slow share to 0.8% for 2.0% extra requests in the simulation.
- Batching raised throughput about 10 times (batch 1 to 64) at 0.59 ms per call, but hurts at low load; set the wait from the latency budget.
- Size the canary: about 31,000 requests per arm to detect 5.0% to 5.5% errors; a quarter of that catches it 27% of the time.
- Per-feature drift tests fire by chance and miss joint changes; use a classifier-based test for detection and per-feature statistics to explain.
- Measure the age-versus-quality curve and choose the retraining schedule from it: weekly was within 0.002 AUC of daily at a seventh of the runs in the simulation.
- ML technical debt is systemic: entanglement, unstable and underused data dependencies, feedback loops; keep a practised rollback.

## Limits

- All experiments use synthetic data on one CPU machine; they show mechanisms (fan-out arithmetic, batching trade-offs, sample sizes, test behaviour, retraining schedules) and not what any real system would measure. Timings vary between runs; the batching simulation assumes Poisson arrivals and a linear cost model fitted to a small network, and no GPU numbers were measured.
- Claims about papers and documentation are limited to the passages read: the ML Test Score paper, The Tail at Scale, Hidden Technical Debt in Machine Learning Systems, Failing Loudly, and the Triton dynamic-batcher page, at the time they were written. Failing Loudly studies image and text shifts with specific detectors; its conclusions are summarised, not extended.
- Items marked as the author's own (the release sequence with shadow mode, the three first checks, the monitoring grouping and runbooks, the serving-pattern comparison, the interpretation of the randomised slice) have no source behind them in this post.
- Not covered: GPU serving and LLM-specific serving (continuous batching, KV-cache management), feature store internals, CI/CD for ML pipelines, model registries, cost optimisation in depth, privacy and security in deployment, and monitoring of fairness.
- The candidates in the examples are hypothetical, and the reference answers are one reasonable answer among several.

Next: [Part 12, coding questions](/posts/ml-engineer-interviews-12-coding/).
