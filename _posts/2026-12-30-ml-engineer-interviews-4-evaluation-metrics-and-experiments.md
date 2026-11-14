---
title: "ML Engineer Interviews, With Reference Answers (4/18): Evaluation, Metrics and Experiments"
date: 2026-12-30 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, machine-learning, evaluation, ab-testing, series]
description: "Part 4 of 18. Ten interview questions on metrics, thresholds, calibration, cross-validation, leakage and A/B tests, with reference answers and simulations whose code and output are shown."
---

This post covers the evaluation questions of the technical round: which metric to use and why, how to pick a threshold, what calibration is, how to cross-validate without leaking, and how to run an A/B test. It gives ten questions with what each tests, a reference answer, a common weak answer and a follow-up. Numbers in the answers come from code that was run for this post, shown with its output. The answers are the author's, for hypothetical candidates. [Part 3](/posts/ml-engineer-interviews-3-fundamentals-and-classical-algorithms/) covered fundamentals.

The code ran on a CPU-only machine with Python 3.14, NumPy, SciPy and scikit-learn 1.9.1, with fixed seeds. The data are synthetic and chosen to show a mechanism; other data will give other numbers.

![Left: false positive rate of an A/A test as a function of the number of looks at the data when stopping at the first p below 0.05: 0.051 for one look, 0.086, 0.144, 0.198 and 0.261 for 2, 5, 10 and 20 looks. Right: simulated power against users per arm for a 5 percent baseline conversion and a 5 percent relative lift; power crosses 0.8 near 121,600 users per arm.]({{ "/assets/pic/ml-interview-4-peeking-power.png" | relative_url }})
*Left: peeking inflates the false positive rate (question 10). Right: power against sample size for a small effect (question 10).*

## 1. "The model has 98.5% accuracy. Is that good?"

**What it tests.** Whether the candidate asks for the class balance before judging a number.

**Reference answer.** It cannot be judged without the base rate. If 98.5% of rows are negative, a model that always predicts negative scores 98.5%. Accuracy also hides which errors are made, and the two kinds of error usually cost different amounts. The fix is to report the confusion matrix and metrics that name the error: recall, precision, balanced accuracy, and a ranking metric such as average precision.

An experiment with 1.5% positives (454 of 30,000 test rows):

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, recall_score, precision_score, roc_auc_score,
                             average_precision_score, balanced_accuracy_score)
X, y = make_classification(n_samples=60000, n_features=20, n_informative=6, weights=[0.99],
                           flip_y=0.01, class_sep=0.8, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
base = np.zeros_like(yte)
print(f"always-negative: accuracy {accuracy_score(yte, base):.4f}  recall {recall_score(yte, base):.2f}  "
      f"balanced accuracy {balanced_accuracy_score(yte, base):.2f}")
s = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
print(f"logistic: ROC AUC {roc_auc_score(yte, s):.3f}  average precision {average_precision_score(yte, s):.3f}"
      f"  (random scorer: {yte.mean():.3f})")
pred = (s >= 0.5).astype(int)
print(f"threshold 0.5: accuracy {accuracy_score(yte, pred):.4f} recall {recall_score(yte, pred):.3f} "
      f"precision {precision_score(yte, pred, zero_division=0):.3f}")
```

```
always-negative: accuracy 0.9849  recall 0.00  balanced accuracy 0.50
logistic: ROC AUC 0.694  average precision 0.331  (random scorer: 0.015)
threshold 0.5: accuracy 0.9866 recall 0.115 precision 1.000
```

The trivial model has 0.9849 accuracy and finds no positive case. The logistic model at 0.5 has 0.9866, 0.17 points higher, and finds 11.5% of the positives. Balanced accuracy puts the trivial model at 0.50, which is chance. Average precision of 0.331 against a 0.015 floor is the number that shows the model has real ranking signal.

**Common weak answer.** "Yes, it's high." Or the opposite habit, quoting F1 without saying what the positive class is.

**Follow-up.** "What would you report to the product owner?" The metric tied to the decision: for example, how many true cases are caught at a fixed review budget.

## 2. "Precision, recall and F1: how do you choose?"

**What it tests.** Linking the metric to the cost of each error.

**Reference answer.** Precision is the share of flagged cases that are real; recall is the share of real cases that are flagged. They trade off through the threshold. Choose by what a false negative and a false positive cost: when a miss is expensive (a missed fraud case, a missed diagnosis) favour recall and accept more false alarms; when a false alarm is expensive (each one consumes a reviewer's hour, or blocks a legitimate customer) favour precision. F1 is the harmonic mean of the two and weights them equally, so it is the right summary only if that equal weighting reflects the costs; otherwise use a cost-weighted measure, or report precision at a fixed recall, or recall at a fixed precision. For multiclass problems state the averaging (micro, macro or weighted), because the choice changes the number when classes are imbalanced.

**Common weak answer.** "F1 balances them, so I use F1." It skips the cost argument.

**Follow-up.** "Macro or micro F1?" Macro treats each class equally and exposes poor performance on rare classes; micro is dominated by frequent classes.

## 3. "ROC AUC or PR AUC?"

**What it tests.** Whether the candidate knows when ROC curves mislead.

**Reference answer.** ROC AUC measures how well the scores rank a random positive above a random negative. It does not depend on the class ratio, which is a strength for comparing models and a weakness when negatives vastly outnumber positives, because a small false positive rate can still mean many false positives in absolute terms. [Saito and Rehmsmeier (2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) show this on imbalanced data: ROC plots can be deceptive about reliability because of an intuitive but wrong interpretation of specificity, whereas precision-recall plots evaluate the fraction of true positives among positive predictions and give a more accurate picture of future performance. When the positive class is rare and the cost sits with the flagged cases, use precision-recall and average precision, and compare against the prevalence, which is the average precision of a random scorer. ROC AUC remains fine for balanced problems and for comparing rankers on a common data set.

In the experiment above, the ROC AUC is 0.694 and the average precision 0.331. The second number is only interpretable next to the 0.015 base rate; it is about 22 times the random level, which the ROC number does not convey.

**Common weak answer.** "AUC is threshold independent, so it is always better." It is threshold independent; that is not the same as being informative for the decision.

**Follow-up.** "Can a model have a better ROC AUC and a worse PR AUC than another?" Yes, because the two weight regions of the ranking differently; the PR curve is concentrated at the top of the ranking.

## 4. "How do you choose the decision threshold?"

**What it tests.** Separating the model from the decision.

**Reference answer.** The model produces a score; the threshold is a business decision. Write down the cost of a false negative and a false positive, compute the expected cost on a validation set across thresholds, and pick the minimum. Do not default to 0.5, and do not tune the threshold on the test set. The threshold should be revisited when costs or prevalence change.

On a 5%-positive problem (a fresh data set, 20,000 test rows), with the false-positive cost fixed at 1:

```python
from sklearn.metrics import f1_score
X, y = make_classification(n_samples=40000, n_features=20, n_informative=6, weights=[0.95],
                           flip_y=0.01, class_sep=0.9, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=1)
s = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
ths = np.linspace(0.02, 0.9, 89)
def cost(th, c_fn, c_fp):
    pr = s >= th; fn = ((~pr) & (yte == 1)).sum(); fp = (pr & (yte == 0)).sum()
    return c_fn*fn + c_fp*fp
for c_fn in (1, 10, 50):
    cs = [cost(th, c_fn, 1) for th in ths]
    print(f"cost FN={c_fn:>2} : FP=1 -> best threshold {ths[int(np.argmin(cs))]:.2f}")
f1s = [f1_score(yte, s >= th) for th in ths]
print(f"max-F1 threshold {ths[int(np.argmax(f1s))]:.2f} (F1 {max(f1s):.3f})")
```

```
cost FN= 1 : FP=1 -> best threshold 0.44
cost FN=10 : FP=1 -> best threshold 0.12
cost FN=50 : FP=1 -> best threshold 0.03
max-F1 threshold 0.26 (F1 0.598)
```

The same model has a best threshold of 0.44, 0.12 or 0.03 depending on the cost ratio, and 0.26 if F1 is the target. The threshold moves by more than a factor of ten with the costs, which is why the question cannot be answered without them.

**Common weak answer.** "0.5", or "whichever maximises F1" without justifying equal weights.

**Follow-up.** "The prevalence changes in production. What happens to your threshold?" The score distribution and the optimal threshold both move; recalibrate and re-derive the threshold.

## 5. "What is calibration, and when does it matter?"

**What it tests.** Whether the candidate separates ranking quality from the meaning of the probability.

**Reference answer.** A model is calibrated if, among the cases it scores at 0.8, about 80% are positive. Calibration matters when the probability itself is used: in an expected-cost calculation, when combining scores from several models, when showing a probability to a person, or when the threshold is derived from costs (question 4). It does not matter for pure ranking. The metrics that measure it are proper scoring rules: the [scikit-learn documentation](https://scikit-learn.org/stable/modules/model_evaluation.html) lists the Brier score and log loss as the scoring functions for predicted probabilities. A reliability diagram, or a binned expected calibration error, shows where the model is off.

Different model families fail differently. The [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) quotes Niculescu-Mizil and Caruana on why bagging and random forests have difficulty making predictions near 0 and 1: variance in the base models biases predictions that should be near 0 or 1 away from those values. [Guo et al. (2017)](https://arxiv.org/abs/1706.04599) report that modern neural networks are poorly calibrated and that temperature scaling, a single-parameter variant of Platt scaling, is surprisingly effective on most of their data sets. The two post-hoc methods in scikit-learn are sigmoid and isotonic; the guide says isotonic performs as well as or better than sigmoid when there is enough data (more than about 1,000 samples), and is more prone to overfitting on small data sets.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
X, y = make_classification(n_samples=20000, n_features=20, n_informative=8, flip_y=0.05,
                           class_sep=0.8, random_state=2)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=2)
def ece(y, p, bins=10):
    idx = np.minimum((p*bins).astype(int), bins-1); e = 0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean()*abs(y[m].mean() - p[m].mean())
    return e
rf = lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1)
models = {"logistic": LogisticRegression(max_iter=1000), "random forest": rf(),
          "forest + isotonic": CalibratedClassifierCV(rf(), method="isotonic", cv=5),
          "forest + sigmoid": CalibratedClassifierCV(rf(), method="sigmoid", cv=5)}
for n, m in models.items():
    p = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    print(f"{n:18} AUC {roc_auc_score(yte, p):.3f}  Brier {brier_score_loss(yte, p):.3f}  "
          f"logloss {log_loss(yte, p):.3f}  ECE {ece(yte, p):.3f}")
```

```
logistic           AUC 0.880  Brier 0.140  logloss 0.438  ECE 0.020
random forest      AUC 0.958  Brier 0.079  logloss 0.285  ECE 0.083
forest + isotonic  AUC 0.957  Brier 0.070  logloss 0.260  ECE 0.008
forest + sigmoid   AUC 0.957  Brier 0.070  logloss 0.250  ECE 0.008
```

The forest ranks much better than logistic regression (AUC 0.958 against 0.880), but its raw probabilities are the worst calibrated (ECE 0.083 against 0.020). Calibrating it with either method cuts the ECE to 0.008 and improves Brier and log loss, with the AUC unchanged to within 0.001. The scikit-learn guide notes that isotonic can change ranking metrics slightly because it introduces ties.

**Common weak answer.** "Calibration makes the model more accurate." It does not change the ranking in any meaningful way; it makes the probabilities mean what they say.

**Follow-up.** "On what data do you fit the calibrator?" Held-out data, or cross-validation as in `CalibratedClassifierCV`, never the data the base model was trained on.

## 6. "Which loss or metric for which kind of problem?"

**What it tests.** Matching the metric to the quantity the business needs.

**Reference answer.** Match the metric to the question the prediction answers. The scikit-learn guide to model evaluation tabulates scoring functions by the functional being predicted: the Brier score and log loss for class probabilities, zero-one loss for the most likely class, mean squared error for the mean, median absolute error for the median, and the quantile pinball loss for a quantile. The practical consequence: squared error rewards predicting the mean and is sensitive to outliers; absolute error targets the median and is more robust; if the decision needs a high quantile (a delivery-time promise, a capacity plan), train and evaluate with the pinball loss at that quantile. A mismatch, for example evaluating a mean-predicting model by median absolute error, rewards the wrong behaviour. For ranking and recommendation problems the metric should reflect what the user sees, such as the top of the list; this answer does not go deeper because the sources read for this post do not cover those metrics.

**Common weak answer.** "RMSE for regression, accuracy for classification." Names defaults without naming the decision.

**Follow-up.** "The target has a long tail. What changes?" Outliers dominate squared error; consider a robust loss or a transform, and say which quantity the business needs.

## 7. "How do you validate a model properly?"

**What it tests.** Cross-validation as a way to estimate performance on the data the model will see, which means the split must mimic deployment.

**Reference answer.** Hold out data that mimics deployment, and keep every step that learns from data (scaling, imputation, feature selection, target encoding) inside the training fold; a pipeline object does this. Random k-fold assumes rows are independent and identically distributed. When they are not, the split has to respect the structure. The [scikit-learn guide](https://scikit-learn.org/stable/modules/cross_validation.html) says that if the data come from groups (different subjects, experiments, measurement devices), it is safer to use group-wise cross-validation, and it provides `TimeSeriesSplit` for ordered data. Use a separate test set once, at the end; repeated peeking at it turns it into a validation set.

Two experiments with a random forest. In the first, 100 subjects have 20 rows each, the features are a noisy fingerprint of the subject, and the labels are random coin flips per subject, so nothing can generalise to a new subject:

```python
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit, cross_val_score
rng = np.random.default_rng(0)
S, R = 100, 20
subj = np.repeat(np.arange(S), R); fp = rng.normal(size=(S, 5)); lab = rng.integers(0, 2, S)
X = fp[subj] + rng.normal(0, 0.3, (S*R, 5)); y = lab[subj]
rf = RandomForestClassifier(100, random_state=0, n_jobs=-1)
print(cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean(),
      cross_val_score(rf, X, y, groups=subj, cv=GroupKFold(5)).mean())
```

```
shuffled KFold accuracy 0.87, GroupKFold accuracy 0.46
```

Shuffled k-fold reports 0.87 on a problem that is at chance for new subjects; group k-fold reports 0.46, the honest answer. In the second, the relation between the feature and the label flips sign halfway through the series:

```python
n = 6000; t = np.arange(n); x1 = rng.normal(size=n)
y2 = ((np.where(t < n//2, 1, -1)*x1 + rng.normal(0, 0.5, n)) > 0).astype(int)
X2 = np.c_[x1, t/n]
tr, te = np.arange(n//2), np.arange(n//2, n)
print(cross_val_score(rf, X2, y2, cv=KFold(5, shuffle=True, random_state=0)).mean(),
      rf.fit(X2[tr], y2[tr]).score(X2[te], y2[te]),
      cross_val_score(rf, X2, y2, cv=TimeSeriesSplit(5)).mean())
```

```
shuffled KFold 0.83, train first half / test second half 0.18, TimeSeriesSplit(5) 0.60
```

Shuffled k-fold, which mixes past and future, says 0.83. Training on the first half and testing on the second gives 0.18, worse than chance, because the relation has reversed. `TimeSeriesSplit` averages 0.60 since its early folds still train and test within the first regime. Which number to believe depends on the deployment question, and that is the point: a random split answers a question nobody asked.

**Common weak answer.** "5-fold cross-validation." It names the tool without checking whether rows are independent.

**Follow-up.** "The data are user events over a year. How do you split?" By time, with groups by user if the same user appears on both sides and the model could memorise users.

## 8. "What is data leakage? Give examples."

**What it tests.** Whether the candidate can find the information in the training data that will not exist at prediction time.

**Reference answer.** Leakage is any route by which the training or validation data contain information that is not available when the model is used. The usual forms:

- **Target leakage.** A feature is a consequence of the label, for example a "refund issued" flag used to predict fraud, or a field filled in by a human after the outcome is known.
- **Preprocessing leakage.** Scaling, imputation, feature selection or target encoding fitted on the whole data set before splitting, so validation rows influenced the transformation.
- **Group leakage.** The same user, device or patient appears on both sides of the split (the first experiment in question 7).
- **Temporal leakage.** Future rows in the training set, or features computed with information from after the prediction time (the second experiment in question 7).
- **Duplicates.** Near-identical rows across the split.

The diagnostics are the author's practice rather than a cited standard: a validation score that is too good to be true; one feature with outsized importance; a large gap between offline and live performance; and a check, for each feature, of when its value becomes known. The remedies are a pipeline that fits transformations inside each training fold, a split that follows the deployment structure, and a feature definition that carries an "as of" time.

**Common weak answer.** "Don't train on the test set." True, and only the simplest form.

**Follow-up.** "Your model scored 0.99 AUC in validation. What do you do?" Treat it as a suspected leak first: inspect the top features, then reproduce the split by group and by time.

## 9. "Offline metrics improved but the online metric did not. Why?"

**What it tests.** Seeing the evaluation chain from the metric to the product.

**Reference answer.** The author's reasoning; the questions are ordered from cheapest to check to hardest:

1. **The pipeline.** Is the model served with the same features and preprocessing as in training (training-serving skew), and is the new model actually the one serving traffic?
2. **The offline data.** Does the validation set mimic live traffic? Leakage (question 8) and a stale or filtered sample both inflate offline numbers.
3. **The metric.** An offline metric is a proxy. Accuracy, AUC or log loss on logged data need not move the quantity the product cares about; the logged data were also generated by the old system, which biases what the new model is evaluated on.
4. **The experiment.** Is it powered to detect the effect (question 10)? An offline gain of a fraction of a percent can be real and invisible in an online test of ordinary size.
5. **The change in behaviour.** Users adapt to a new system; effects can be larger or smaller than the first days suggest. [Kohavi et al. (2007)](https://exp-platform.com/Documents/GuideControlledExperiments.pdf) note primacy and newness effects and say that some experiments need to run for multiple weeks.

The practical answer is to decide before the experiment which online metric is the criterion. Kohavi et al. call it the Overall Evaluation Criterion (OEC), the quantitative measure of the experiment's objective, and they recommend a single metric, possibly a weighted combination, because a single metric forces trade-offs to be made explicit.

**Common weak answer.** "The A/B test is noisy." Possibly, but it is the last item on the list, not the first.

**Follow-up.** "How do you choose the online metric for a recommender?" A metric tied to the product goal that moves within the experiment's duration, plus guardrail metrics for harm; the sources read for this post do not give a recipe, so the candidate should reason from the product.

## 10. "Design an A/B test for a new model."

**What it tests.** The statistics and the process of controlled experiments, and the traps in each.

**Reference answer.**

1. **Hypothesis and metric.** Fix the OEC and guardrail metrics in advance (question 9).
2. **Unit and randomisation.** Randomise by a unit that matches the experience (usually the user, not the page view), with a stable hash so a user always sees the same variant. Kohavi et al. warn that randomisation and hashing are not as simple in practice as often assumed, and that some schemes fail the conditional-independence tests needed for validity.
3. **Check the system with an A/A test.** Kohavi et al. define it: assign users to two groups but give them exactly the same experience; the null hypothesis should be rejected about 5% of the time at the 95% confidence level. It tests the experimentation system and gives the variance for the power calculation.
4. **Size the test before running it.** The inputs are the baseline, the smallest effect worth detecting, the confidence level and the power. Kohavi et al. say power is commonly desired to be around 80 to 95%. Their formula, which assumes 95% confidence and 90% power, is n = (4rσ/Δ)², with r the number of variants, σ the standard deviation of the metric and Δ the difference to detect. [Evan Miller](https://www.evanmiller.org/how-not-to-run-an-ab-test.html) gives a rule of thumb of 16σ²/δ² per group. The experiment below compares the two.
5. **Ramp up.** Kohavi et al. suggest starting with a small share of users for the treatment, for example 0.1%, then 0.5%, 2.5%, 10% and 50%, checking for egregious problems at each step.
6. **Run for the planned duration and analyse once.** Do not stop at the first significant result.

For a 5% baseline conversion and a 5% relative lift (0.0025 absolute), with a fixed seed:

```python
import numpy as np
from scipy.stats import norm
p, rel = 0.05, 0.05
d = p*rel; sigma2 = p*(1-p)
n_koh = (4*2*np.sqrt(sigma2)/d)**2          # both variants together
n_em = 16*sigma2/d**2                        # per group
print(f"Kohavi formula, both variants together: {n_koh:,.0f}  (per variant {n_koh/2:,.0f})")
print(f"Miller 16*sigma^2/delta^2, per group: {n_em:,.0f}")
rng = np.random.default_rng(0)
def power(n, reps=4000):
    a = rng.binomial(n, p, reps); b = rng.binomial(n, p*(1+rel), reps)
    pa, pb = a/n, b/n; pp = (a+b)/(2*n); se = np.sqrt(2*pp*(1-pp)/n)
    return np.mean(np.abs(pb-pa)/se > norm.ppf(0.975))
for n in (int(n_koh/2), int(n_em), int(n_em/2)):
    print(f"simulated power with {n:>8,} per arm: {power(n):.3f}")
```

```
Kohavi formula, both variants together: 486,400  (per variant 243,200)
Miller 16*sigma^2/delta^2, per group: 121,600
simulated power with  243,199 per arm: 0.976
simulated power with  121,599 per arm: 0.794
simulated power with   60,799 per arm: 0.503
```

The two rules differ by a factor of two per arm. Kohavi et al. state that their formula assumes 95% confidence and 90% power and that the factor 4 may overestimate by 25% for large n; the simulation gives 97.6% power at their figure and 79.4% at Miller's. Halving the sample again drops the power to 50%, a coin flip. A 5% relative change on a 5% conversion rate needs on the order of a hundred thousand users per arm. That is the main reason small model improvements often cannot be verified online.

**The peeking trap.** Miller's article states that significance calculations assume the sample size was fixed in advance; if the experiment runs until a significant difference appears, the reported significance levels become meaningless. A simulation of A/A tests, where the true difference is zero, shows the size of the effect (20,000 runs, 2,000 observations each, stopping at the first p below 0.05 among equally spaced looks):

```
fixed sample size 2000: false positive rate 0.051
peek  2 times: 0.086
peek  5 times: 0.144
peek 10 times: 0.198
peek 20 times: 0.261
test after every observation from the 10th on: 0.512
```

With ten looks the nominal 5% error rate becomes about 20%, and with continuous monitoring the experiment declares a winner in about half of the runs where there is none. The legitimate options are a fixed sample size and a single analysis, or a sequential method designed for repeated looks; the sources read for this post do not describe sequential methods, so the candidate should say that they exist and when they are needed, and not improvise one.

**Common weak answer.** "Split traffic 50/50, wait until it is significant." It skips the sample size and invites the peeking problem.

**Follow-up.** "You ran ten metrics and one is significant." With many metrics some will cross the threshold by chance; fix the primary metric beforehand, treat the others as exploratory, and correct or replicate.

## Takeaways

- Name the base rate before judging any metric; accuracy alone is not evidence.
- Choose the metric and threshold from the cost of each error, and calibrate when the probability is used.
- A validation split must mimic deployment; random k-fold is a choice with a hidden assumption.
- Offline gains are proxies; decide the online criterion first.
- An A/B test needs a sample size computed in advance and one analysis at the end.

## Limits

- The experiments use synthetic data and one seed each; they show mechanisms, and the exact numbers will differ elsewhere.
- Ranking and recommendation metrics, sequential testing and variance-reduction methods are not covered here, because the sources read for this post do not treat them in enough detail to cite.
- Items marked as the author's reasoning (questions 8 and 9, and the order of checks) have no single source behind them.
- The reference answers describe one reasonable answer. Interviewers differ, and the candidates in the examples are hypothetical.

Next: Part 5, on data, sampling and features.
