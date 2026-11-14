---
title: "ML Engineer Interviews, With Reference Answers (15/18): The Project Deep Dive"
date: 2027-01-10 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, project-deep-dive, ablation, technical-debt, distribution-shift, series]
description: "Part 15 of 18. Ten questions an interviewer asks about one project, from why it mattered to what it would take to maintain it, each answered with a reference answer and demonstrated on real code: trade-offs, ablations, paired intervals, entanglement and underspecification."
---

The final round often takes one project from the candidate's resume and asks about it for 45 minutes. [Part 14](/posts/ml-engineer-interviews-14-take-home-and-presenting-a-model/) covered presenting a take-home; this post covers the harder case in which the interviewer chooses the direction. It gives ten questions that follow a project from motivation to maintenance, each with what it tests, a reference answer, a weak answer and a follow-up. The projects in the examples are hypothetical. Where an answer rests on a published claim, the paper was read: [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) (Sculley et al., 2015), [Underspecification Presents Challenges for Credibility in Modern Machine Learning](https://arxiv.org/abs/2011.03395) (D'Amour et al., 2020) and [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434) (Gulrajani and Lopez-Paz, 2020). The numbers come from code run on a 32-core CPU machine with fixed seeds, on synthetic or bundled data; they show that an effect exists and how it behaves, not how large it will be on a real project. Advice that no source supports is marked as the author's.

![Left: ten neural networks trained with different seeds all reach a held-out AUC of about 0.757, but their AUC after the second input stops tracking the first ranges from 0.620 to 0.753. Right: the fitted coefficient of feature a is 0.86, 0.93 and 1.55 for three different feature sets, against a true value of 1.0.](/assets/pic/ml-interview-15-underspecification-cace.png)
*Left: models equal on held-out data and unequal after a shift (question 7). Right: one coefficient under three feature sets (question 4).*

## 1. "Pick the project you are proudest of and give me the one-minute version."

**What it tests.** Whether the candidate chose a project that can survive a deep dive, and can state it compactly.

**Reference answer.** Choose before the interview, and choose by what the questions below will need, not by how impressive the title sounds. The criteria (the author's): the candidate made real decisions in it; there was a measurable result; something went wrong that was understood; and the candidate can still explain the data, the metric and the baseline from memory. A smaller project that is known to the last detail beats a larger one known at the level of a slide. The one-minute version has five parts in this order: the problem and who needed it, the candidate's role, the approach in a sentence, the result with one number and its comparison, and the one thing that was hard. A hypothetical example: "A payments team needed to rank manual reviews because reviewers could check about 5,000 of 400,000 monthly cases. I owned the model and the offline evaluation. We replaced a rule list with a gradient-boosted ranker trained on three years of review outcomes. Precision in the reviewed top 5,000 went from 0.12 to 0.31 on a holdout from the latest quarter. The difficult part was that review outcomes only exist for cases that were reviewed." The last sentence gives the interviewer a place to dig, which is the point of the summary.

**Common weak answer.** A list of technologies, or a summary of the team's work with no "I" and no number.

**Follow-up.** "What was your part, specifically?" See question 10.

## 2. "Why was this worth doing, and how did you know it worked?"

**What it tests.** Linking the model to a decision and a cost, with honest uncertainty on the claimed value.

**Reference answer.** State the decision the model changes, the unit economics, and how the number was measured. The conversion from a metric to money needs two kinds of input: quantities measured with error, such as precision on a reviewed sample, and assumptions, such as the gain from a true positive. Give an interval for the first and say which assumption the answer is most sensitive to. In the hypothetical payments example, 5,000 cases are flagged per month, 400 of them were reviewed by hand and 124 were true positives; a true positive is assumed to save 150 and each review to cost 4.

```python
import numpy as np
flagged, reviewed, hits = 5000, 400, 124                       # hypothetical: 5,000 cases flagged per month, 400 reviewed by hand, 124 true positives
p = hits / reviewed; z = 1.96; c = (p + z**2 / (2 * reviewed)) / (1 + z**2 / reviewed); h = z * np.sqrt(p * (1 - p) / reviewed + z**2 / (4 * reviewed**2)) / (1 + z**2 / reviewed); lo, hi = c - h, c + h; gain_per_hit, cost_per_review = 150.0, 4.0
f = lambda q: flagged * q * gain_per_hit - flagged * cost_per_review
print(f"precision on the reviewed sample {p:.3f}, 95% Wilson interval {lo:.3f} to {hi:.3f}")
print(f"monthly net value = {flagged} x precision x {gain_per_hit:.0f} - {flagged} x {cost_per_review:.0f}: point {f(p):,.0f}; interval {f(lo):,.0f} to {f(hi):,.0f}")
print(f"break-even precision = {cost_per_review / gain_per_hit:.3f}")
```

```
precision on the reviewed sample 0.310, 95% Wilson interval 0.267 to 0.357
monthly net value = 5000 x precision x 150 - 5000 x 4: point 212,500; interval 179,997 to 247,714
break-even precision = 0.027
```

Precision is 0.310 with a 95% Wilson interval of 0.267 to 0.357 (the interval method of [Part 13](/posts/ml-engineer-interviews-13-maths-and-statistics/), question 5). The monthly net value is 212,500 at the point estimate and 179,997 to 247,714 across the interval. The break-even precision is 0.027, so the sign of the value is not in doubt at any plausible sampling error; the sizes of the gain per hit and of the cost per review are the assumptions that carry the result, and they should be the ones defended. A caveat to state: precision measured on cases that the old process also flagged may not hold on the rest, which is the selection problem in the one-minute example.

**Common weak answer.** "It improved AUC by 3 points", with no link to a decision, or "it saved 2 million" with no account of how that was computed.

**Follow-up.** "Was there an online test?" If not, say what the offline number does not cover (Part 4, offline and online gaps) and how the launch was staged.

## 3. "Why this approach? What else did you consider?"

**What it tests.** Trade-off reasoning that includes cost, latency and maintainability, not only accuracy.

**Reference answer.** Present two or three real alternatives with their measured differences on the same data, then the criterion that decided. Typical axes are accuracy, training cost, prediction latency, model size, interpretability and how hard the model is to keep correct. Rule 4 of Google's [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) says the first model should be simple and the infrastructure right; the argument for a heavier model has to be the measured gain.

```python
import pickle, time, numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X, y = make_classification(60000, 30, n_informative=10, n_redundant=4, n_clusters_per_class=2, flip_y=0.03, class_sep=0.9, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print(f"{len(Xtr)} training rows, {X.shape[1]} features; CPU, single process; prediction time is for 10,000 rows (best of 5)")
for name, m in {"logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), "gradient boosting": HistGradientBoostingClassifier(random_state=0), "random forest (100 trees)": RandomForestClassifier(100, min_samples_leaf=3, random_state=0, n_jobs=1)}.items():
    t = time.perf_counter(); m.fit(Xtr, ytr); fit = time.perf_counter() - t; best = min((lambda t0: (m.predict_proba(Xte[:10000]), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(5))
    print(f"{name:26s} AUC {roc_auc_score(yte, m.predict_proba(Xte)[:, 1]):.4f} | fit {fit:6.1f} s | predict 10k rows {best * 1000:7.1f} ms | pickled size {len(pickle.dumps(m)) / 1e3:8.1f} kB")
```

```
48000 training rows, 30 features; CPU, single process; prediction time is for 10,000 rows (best of 5)
logistic regression        AUC 0.7979 | fit    0.1 s | predict 10k rows     1.0 ms | pickled size      2.0 kB
gradient boosting          AUC 0.9754 | fit    0.4 s | predict 10k rows     4.8 ms | pickled size    419.2 kB
random forest (100 trees)  AUC 0.9774 | fit   16.0 s | predict 10k rows    85.7 ms | pickled size  41902.0 kB
```

On a synthetic problem built to contain interactions (48,000 training rows, 30 features), logistic regression has an AUC of 0.798, gradient boosting 0.975 and a random forest of 100 trees 0.977. The forest buys 0.002 of AUC over boosting at forty times the fit time (16.0 s against 0.4 s), eighteen times the prediction time on 10,000 rows (85.7 ms against 4.8 ms) and a pickled model one hundred times larger (41.9 MB against 419 kB). The linear model is the fastest by far and the least accurate, because this data was built with interactions; on a problem that is nearly linear the order would change (see the paired comparison in question 6). The point of the table is that each choice is defended with numbers, and that the winner depends on the constraint: with a 10 ms budget per batch the forest is out regardless of its AUC.

**Common weak answer.** "I tried several models and XGBoost was the best", with no costs, no baseline and no reason the others lost.

**Follow-up.** "Would the choice change at ten times the traffic?" Latency and model size scale with traffic; accuracy does not.

## 4. "What was the hardest technical problem?"

**What it tests.** Depth: a real problem, a diagnosis, a fix and a lesson, as opposed to a list of difficulties.

**Reference answer.** Pick a problem with a clear shape: symptom, hypothesis, test, cause, fix, what was learned. Entanglement is a good example of a cause. Sculley et al. state that machine learning systems mix signals together: if the input distribution of one feature changes, the importance, weights or use of the remaining features may all change, and adding or removing a feature can cause similar changes; they call it the CACE principle, "Changing Anything Changes Everything", and say it applies to hyper-parameters, learning settings, sampling methods, convergence thresholds and data selection too. In practice this makes "we improved one component" an unreliable statement, because the evaluation of the whole system is what changed.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0); n = 5000
a = rng.normal(size=n); b = 0.9 * a + 0.44 * rng.normal(size=n); c = 0.7 * b + 0.7 * rng.normal(size=n)
y = rng.random(n) < 1 / (1 + np.exp(-(1.0 * a + 0.8 * b + 0.6 * c)))
for cols, names in ((np.c_[a, b], "a, b"), (np.c_[a, b, c], "a, b, c"), (np.c_[a, c], "a, c")):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(cols, y); print(f"features {names:8s} coefficients {np.round(m.coef_[0], 2)}")
print("true coefficients used to generate the labels: a 1.0, b 0.8, c 0.6")
```

```
features a, b     coefficients [0.86 1.33]
features a, b, c  coefficients [0.93 0.87 0.73]
features a, c     coefficients [1.55 0.92]
true coefficients used to generate the labels: a 1.0, b 0.8, c 0.6
```

The labels are generated from three correlated features with true coefficients 1.0, 0.8 and 0.6. With features a and b the fitted coefficient of a is 0.86; adding c moves it to 0.93; dropping b and keeping a and c moves it to 1.55, because a absorbs the effect of the missing correlated feature. A team that removed feature b "because it was redundant" and saw a's coefficient jump would have reason to ask what else moved. A good deep-dive answer would add what was done: freeze the evaluation set, change one thing at a time, compare per-slice metrics before and after, and log the configuration with every run.

**Common weak answer.** "The data was messy", or a problem that was solved by a library call, with no diagnosis.

**Follow-up.** "How did you know that was the cause?" The answer should be an experiment that could have come out differently, not a story.

## 5. "Which parts of the system were actually necessary?"

**What it tests.** Ablation: the discipline of removing components and measuring, with attention to noise.

**Reference answer.** For each claimed contribution (a feature group, a trick, a loss term, a data source), the evidence is the score with it removed, on the same splits, with a spread. Paired differences on the same folds are more informative than two separate means. Report the contribution as a difference with an uncertainty, and say what the unremarkable results mean: some components do nothing, and some hurt.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
groups = {"mean": [c for c in X if c.startswith("mean")], "error": [c for c in X if c.endswith("error")], "worst": [c for c in X if c.startswith("worst")]}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0); base = None
def score(cols): return cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X[cols], y, cv=cv, scoring="roc_auc")
runs = {"all 30 features": list(X.columns)}
for g in groups: runs[f"without '{g}' group"] = [c for c in X if c not in groups[g]]
for g in groups: runs[f"only '{g}' group"] = groups[g]
print("logistic regression, ROC AUC over 10 x 5-fold CV (mean, std over the 50 folds, paired difference to the full model: mean and standard error)")
for k, cols in runs.items():
    s = score(cols); base = s if base is None else base; d = s - base
    print(f"{k:24s} {s.mean():.4f}  std {s.std():.4f}  diff {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
```

```
logistic regression, ROC AUC over 10 x 5-fold CV (mean, std over the 50 folds, paired difference to the full model: mean and standard error)
all 30 features          0.9947  std 0.0056  diff +0.0000 (se 0.0000)
without 'mean' group     0.9955  std 0.0054  diff +0.0008 (se 0.0002)
without 'error' group    0.9924  std 0.0084  diff -0.0023 (se 0.0007)
without 'worst' group    0.9861  std 0.0087  diff -0.0087 (se 0.0008)
only 'mean' group        0.9850  std 0.0087  diff -0.0097 (se 0.0008)
only 'error' group       0.9424  std 0.0186  diff -0.0523 (se 0.0026)
only 'worst' group       0.9922  std 0.0092  diff -0.0026 (se 0.0009)
```

For logistic regression on the 30 breast-cancer features (grouped as the dataset names them: mean, error and worst values), the full model has a cross-validated AUC of 0.9947. Removing the "worst" group costs 0.0087 (standard error 0.0008), removing the "error" group costs 0.0023, and removing the "mean" group raises the score by 0.0008 (standard error 0.0002), so the mean features were not helping once the others were present. The "worst" group alone reaches 0.9922, close to the full model and above the "mean" group alone (0.9850). These standard errors treat the 50 folds as independent; they are not, because the folds share data, so the true uncertainty is larger and differences of 0.001 should not be read as findings. An answer in an interview would say that as well.

**Common weak answer.** "Every feature I added helped", with no comparison, or a feature-importance chart offered as an ablation (Part 14, question 7).

**Follow-up.** "If the 'mean' group is useless, would you drop it?" Possibly, for simplicity and cost, but the change is within noise; the decision rests on maintenance cost, not on the 0.0008.

## 6. "You say the new model is better. How sure are you?"

**What it tests.** Comparing two models fairly, and knowing why overlapping intervals do not settle it.

**Reference answer.** Evaluate both models on the same test examples and bootstrap the examples jointly, so that each resample gives a paired difference. The interval of the difference is narrower than the intervals of the two scores, because the shared difficulty of the examples cancels. Two overlapping separate intervals therefore do not show that the difference is zero. Say the size of the test set, give the difference with its interval, and say what the smallest difference is that would change the decision.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
rng0 = np.random.default_rng(1); X = rng0.normal(size=(60000, 10)); logit = X @ np.linspace(0.8, 0.1, 10) + 0.12 * X[:, 0] * X[:, 1]; y = (rng0.random(60000) < 1 / (1 + np.exp(-logit))).astype(int)   # mostly linear, a small interaction
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=30000, random_state=0, stratify=y)
a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]; b = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
rng = np.random.default_rng(0)
for n in (500, 5000, 30000):
    idx = rng.choice(len(yte), n, replace=False); ys, sa, sb = yte[idx], a[idx], b[idx]; d, ra, rb = [], [], []
    for _ in range(1000):
        i = rng.integers(0, n, n)
        if ys[i].min() == ys[i].max(): continue
        x, z = roc_auc_score(ys[i], sa[i]), roc_auc_score(ys[i], sb[i]); ra.append(x); rb.append(z); d.append(z - x)
    pct = lambda v: np.percentile(v, [2.5, 97.5])
    print(f"test rows {n:6d}: AUC logistic {roc_auc_score(ys, sa):.4f} {np.round(pct(ra), 3)}  boosting {roc_auc_score(ys, sb):.4f} {np.round(pct(rb), 3)} | paired difference {roc_auc_score(ys, sb) - roc_auc_score(ys, sa):+.4f} {np.round(pct(d), 4)}")
```

```
test rows    500: AUC logistic 0.8138 [0.776 0.851]  boosting 0.8012 [0.763 0.84 ] | paired difference -0.0126 [-0.0258 -0.0003]
test rows   5000: AUC logistic 0.8210 [0.809 0.833]  boosting 0.8142 [0.802 0.826] | paired difference -0.0069 [-0.0104 -0.0032]
test rows  30000: AUC logistic 0.8186 [0.814 0.823]  boosting 0.8103 [0.805 0.815] | paired difference -0.0083 [-0.0098 -0.0068]
```

The data are generated from a mostly linear rule with a small interaction, so the comparison is close. With the logistic model on one side and gradient boosting on the other, the separate 95% bootstrap intervals overlap heavily at 5,000 test rows (0.809 to 0.833 and 0.802 to 0.826), yet the paired difference (boosting minus logistic) is -0.0069 with an interval of -0.0104 to -0.0032, which excludes zero. At 500 rows the paired interval is -0.0258 to -0.0003, barely excluding zero, and at 30,000 rows it tightens to -0.0098 to -0.0068 around -0.0083. The simpler model is better here by a small amount that is visible only with the paired method. The same logic is why Part 14 reported paired differences over repeated splits.

**Common weak answer.** "0.821 against 0.814, so the first is better", or "the intervals overlap, so there is no difference".

**Follow-up.** "The paired interval excludes zero, but the difference is 0.007. Does it matter?" Only if 0.007 is worth something against the cost and risk of the change; statistical significance is not the same as relevance.

## 7. "How do you know it will hold up in production?"

**What it tests.** Awareness that held-out performance does not determine behaviour under change.

**Reference answer.** The held-out score is measured under the training conditions. D'Amour et al. define a pipeline as underspecified when it can return many predictors with equivalently strong held-out performance in the training domain, and show that such predictors can behave very differently in deployment domains. Their protocol detects it by stress tests that probe the model's inductive biases, in which performance is sensitive to arbitrary choices that preserve held-out performance, such as the random seed; the stress tests produce variation between predictors, not a uniform degradation. A practical answer has three parts: train several seeds and compare them under shifted conditions, not only on the held-out set; monitor inputs and outputs after launch (Part 11); and stage the rollout. Gulrajani and Lopez-Paz add a caution about methods: under fair, controlled comparison, a carefully implemented empirical risk minimisation baseline performed at the state of the art across the datasets they tested, so that elaborate robustness methods need to beat a strong baseline under the same model selection.

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
def make(n, rng, copy):
    y = rng.integers(0, 2, n); c = y * 1.0 + rng.normal(0, 1.0, n)                                       # one real signal
    p = c + rng.normal(0, 0.05, n) if copy else rng.normal(0, 1.4, n)                                      # a near-copy of it in training; unrelated at deployment
    return np.c_[c, p], y
rng = np.random.default_rng(0); Xtr, ytr = make(4000, rng, True); Xid, yid = make(20000, rng, True); Xsh, ysh = make(20000, rng, False)
auc_id, auc_sh = [], []
for seed in range(10):
    m = MLPClassifier((32,), max_iter=500, random_state=seed).fit(Xtr, ytr); auc_id.append(roc_auc_score(yid, m.predict_proba(Xid)[:, 1])); auc_sh.append(roc_auc_score(ysh, m.predict_proba(Xsh)[:, 1]))
print(f"10 networks, same data and recipe, different seeds: held-out AUC {min(auc_id):.4f} to {max(auc_id):.4f}")
print(f"same networks when the second input stops tracking the first: AUC {min(auc_sh):.3f} to {max(auc_sh):.3f} (the best any model can do on the first input alone is about 0.76)")
```

```
10 networks, same data and recipe, different seeds: held-out AUC 0.7572 to 0.7573
same networks when the second input stops tracking the first: AUC 0.620 to 0.753 (the best any model can do on the first input alone is about 0.76)
```

In this synthetic problem two inputs carry the signal in training: the second is a near-copy of the first. Ten small neural networks that differ only in the random seed reach held-out AUCs of 0.7572 to 0.7573, indistinguishable. When the second input stops tracking the first (a change the held-out set does not contain), their AUCs range from 0.620 to 0.753. Any of the ten would be accepted on the held-out score, and they depend on the two inputs to different extents. The shifted score of 0.753 is about what the first input alone supports; the 0.620 network relies more on the second (the author's reading of the numbers; the weights were not inspected). This is a constructed case, and real shifts are less clean; the lesson is that agreement on held-out data says nothing about which solution the training found.

**Common weak answer.** "The test set was from the same period, so it will generalise." Or "we will monitor it" with no statement of what would be monitored or the threshold for action.

**Follow-up.** "What shift would hurt this model most?" The answer names a real dependency: a feature produced by another system, a population change, a seasonal pattern.

## 8. "What happened after launch? What does it cost to keep this running?"

**What it tests.** Whether the candidate has seen a model live, and knows that most of the cost comes after the first version.

**Reference answer.** Sculley et al. observe that developing and deploying ML systems is relatively fast and cheap but maintaining them over time is difficult and expensive, and list ML-specific risk factors: boundary erosion, entanglement, hidden feedback loops, undeclared consumers, data dependencies, configuration issues and changes in the external world. Four are worth having a story for. Unstable data dependencies: consuming a signal produced by another system that changes behaviour over time, for instance another model that retrains. Undeclared consumers: predictions that are logged or written to files can be used silently as inputs to other systems, so that a change to the model changes something the owner did not know existed. Hidden feedback loops: two systems that influence each other through the world, such that improving one changes the data of the other. Glue code and pipeline jungles: the code that connects a general-purpose package to its surroundings, which can lock in assumptions. For each, the reference answer names what was put in place: owners for upstream signals, access control and an inventory of consumers, randomisation or holdouts to measure the loop, and tests on the pipeline end to end.

**Common weak answer.** "It is running fine" with no metrics, or a launch that ended the story.

**Follow-up.** "When would you retire it?" When the maintenance cost exceeds the measured value from question 2, or the decision it supports has changed.

## 9. "What would you do differently?"

**What it tests.** Reflection: whether the candidate can identify specific errors and a changed practice.

**Reference answer.** A good answer names a specific decision, the evidence that it was wrong and what is done now. The candidates for such a decision are the failure modes shown in this series: a baseline built late, so that the gain of the complex model was never measured (Part 14, question 1); preprocessing fitted before the split, which inflated an estimate (Part 14, question 3); a single split treated as the truth (Part 14, question 4); an aggregate metric that hid a slice (Part 14, question 5); a launch without monitoring the inputs (Part 11). Say what the mistake cost in a sentence, and what the check is now, so that it reads as a change in practice and not as a confession. For example: "The first offline estimate used a random split, but the labels had a delay, so the test period overlapped the training period. Now I split by time and report the delay."

**Common weak answer.** "I would have started earlier" or a strength disguised as a flaw ("I care too much about quality").

**Follow-up.** "Is there a decision you would make the same way?" It tests whether the candidate can separate a bad outcome from a bad decision.

## 10. "What exactly was your contribution, and what do you not remember?"

**What it tests.** Honest attribution, and what the candidate does when the questions go beyond what they know.

**Reference answer.** Separate what was done by the candidate, what was done with others, and what was done by others and only understood. Use "I" for the first, "we" with a named role for the second, and name the colleague's work for the third. Interviewers probe this by following a detail down several levels (why that learning rate, why that data source, what the label definition was) until the candidate's knowledge ends, and the useful behaviour at that point is to say where it ends: "that part was owned by the data team; what I know is the interface and the failure I saw." Then say how the answer could be checked. An invented detail is worse than a missing one, because the interviewer can ask one more question (the author's observation). For details that are simply forgotten, state what was true at the time and what is uncertain now, and do not fill the gap. Prepare by writing a one-page fact sheet for the project: the data sources and sizes, the label definition, the split, the baseline and the final metrics with their intervals, the ablations, the costs, and the three things that went wrong.

**Common weak answer.** Claiming the whole team's work, or answering every probe with the same confidence including the ones that are guesses.

**Follow-up.** "Who would disagree with your account?" The question tests whether the candidate can represent the view of a colleague fairly.

## Takeaways

- Choose the project by what a deep dive needs: real decisions, a measurable result, a problem understood, details remembered.
- Convert a metric to money with an interval and name the assumptions: precision 0.310 (0.267 to 0.357) gave a monthly value of 212,500 (179,997 to 247,714) under assumed unit values.
- Defend a choice with a table of accuracy, cost, latency and size: the forest added 0.002 AUC over boosting at forty times the fit time and a hundred times the model size.
- "Changing Anything Changes Everything": the coefficient of feature a was 0.86, 0.93 or 1.55 depending on which other features were present.
- Ablate with paired differences, and remember that fold-based standard errors are optimistic.
- Compare models with a paired bootstrap: separate intervals overlapped while the paired difference excluded zero.
- Equal held-out scores do not mean equal behaviour: ten seeds with AUC 0.7572 to 0.7573 ranged from 0.620 to 0.753 after a shift.
- Launch is the start of the cost: unstable inputs, undeclared consumers and feedback loops.
- Say where your knowledge ends; do not fill the gap.

## Limits

- The experiments use synthetic data (and one bundled dataset) with fixed seeds; the models, rates and costs in the examples are hypothetical and were chosen to make each effect visible. Timings vary between runs and between machines.
- The papers were read for the passages cited: the technical-debt paper's list of risk factors and its entanglement principle, the underspecification definition and stress-test protocol, and the domain-generalisation paper's abstract and its finding on empirical risk minimisation. The criteria for choosing a project, the structure of the one-minute version and the advice on attribution are the author's.
- A deep dive differs by company and by role: research roles probe derivations and papers, applied roles probe data, evaluation and operations.
- Not covered: confidentiality limits on describing a real project, which depend on the employer, and the preparation of a project from academic or open-source work.
- The candidates are hypothetical, and the reference answers are one reasonable answer among several.

Next: [Part 16, behavioural and values questions](/posts/ml-engineer-interviews-16-behavioural-and-values/).
