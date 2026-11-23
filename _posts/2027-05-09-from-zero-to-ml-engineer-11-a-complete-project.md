---
title: "From Zero to ML Engineer, Part 11: A Complete Small Project, Including Its Failure"
date: 2027-05-09 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, end-to-end project, evaluation, bootstrap, model card, ML engineer]
description: "Part 11 of a 12-part series for complete beginners. One script runs the whole workflow: split, baseline, candidates, simplest-within-noise model choice, threshold chosen for a requirement, a single look at the test set, a bootstrap interval, and a model card. The requirement is met on validation and missed on test, and the post reports that as it is."
---

## What you will do, and what you get

You will run the whole workflow from the previous ten parts as a single script of about 70 lines, on a stated requirement: *find at least 98% of the malignant cases*. The output is a saved model with its threshold, a model card, a test result with a confidence interval, and two automatic self-checks. The honest result is mixed. The model reaches 97.4% accuracy against a 63.2% baseline and raises no false alarms, but the recall requirement is met on the validation rows (where it was tuned) and missed on the test rows: 39 of 42 malignant cases found, 92.9%. The script reports this and does not adjust anything to hide it. Understanding why this happens is the most useful thing the project teaches.

![Three panels. Left: the six stages of the project as a flow, each labelled with the part of the series that covers it. Middle: cross-validated accuracy of the baseline at 62.8 percent, logistic regression with C equal 1 at 96.8, with C equal 0.1 at 96.5, random forest at 95.6 and gradient boosting at 94.4 percent. Right: recall of 100 percent on validation versus 92.9 percent on test with an interval from 84.6 to 100, against a dashed requirement line at 98 percent]({{ site.baseurl }}/assets/pic/mlpath-11-project.png){: width="900" }

## How: the steps

1. **State the requirement before touching the data.** Here: recall of malignant cases of at least 98%, with as few false alarms as possible.
2. **Split into train, validation and test** (60/20/20, stratified), as in Part 4.
3. **Score a baseline and several candidates** by cross-validation on the training rows only (Parts 5 to 7).
4. **Choose the simplest model within the noise of the best** (Part 7).
5. **Choose the decision threshold on the validation rows** so that the requirement is met (Part 6).
6. **Refit on train plus validation** with everything frozen, then **open the test set once**, and report the result with an interval.
7. **Save the model, the threshold, a model card and the checks** (Part 9), ready for the service of Part 10.

## Why it works: the principle

**The order protects the numbers.** Each step uses only data that earlier steps have not used for the same decision. The training rows fit the weights; cross-validation on them picks the model family; the validation rows pick the threshold; the test rows are opened once, last. If any decision peeks at a later set, the final number is optimistic (Part 4).

**Choosing the simplest model within the noise.** Cross-validation gives logistic regression with C = 1 96.8% (spread 3.1 points), C = 0.1 96.5%, the random forest 95.6%, gradient boosting 94.4%, and the baseline 62.8%. The ranking among the top four is smaller than the spread, so the script takes the simplest model that is within 1.5 points of the best, which is also the best one here: logistic regression, whose weights can be read (Part 5).

**A bootstrap interval says how much a test number could move.** The test set is one sample of 114 rows. The bootstrap, introduced by [Efron in 1979](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full), resamples the rows (with replacement) many times and recomputes the metric each time; the middle 95% of those values is an interval. Here 2,000 resamples give accuracy 97.4% (interval 94.7 to 100.0) and recall 92.9% (84.6 to 100.0). The recall interval is wide because it is computed from only the 42 malignant rows.

**Why the requirement was missed.** The requirement was 98% recall. With 43 malignant rows in the validation set, 98% means *no* misses (42 of 43 is 97.7%). The script chooses the *highest* threshold that meets it, 0.64, which is the edge: just above it, one malignant validation case would be lost. A threshold sitting on the edge of a set of 43 cases has been fitted to those 43 cases. On the 42 malignant test rows, the model at that threshold misses 3. Three things are true at once: the validation result was real, the test result is the honest estimate, and the test interval (84.6 to 100.0) is too wide to say whether the true recall is above or below 98%. What the data support is: *recall is probably in the low 90s, and the requirement is not demonstrated*.

**What not to do next.** Lowering the threshold now until the test set shows 98% would turn the test set into a validation set, and the claim would no longer be tested (Part 4). The legitimate repairs are made before the test set is opened again, or with a new test set: choose the threshold from out-of-fold predictions on all 455 train and validation rows (170 malignant cases, a steadier basis than 43), require a margin (for example 100% recall on validation rather than exactly the minimum), or negotiate the requirement.

## Example, worked through

- **Split:** 341 train, 114 validation, 114 test rows.
- **Choice:** logistic regression, C = 1, scaled inputs, in a Pipeline.
- **Threshold:** 0.64, giving recall 100% and 0 false alarms on the validation rows.
- **Test, looked at once:** TN = 72, FP = 0, FN = 3, TP = 39. Accuracy 97.4% (interval 94.7 to 100.0), baseline 63.2%. Recall 92.9% (interval 84.6 to 100.0). Precision 100.0%.
- **Self-checks (passed):** the reloaded model reproduces the predictions exactly, and beats the baseline by more than 20 points. A check like this, written as an `assert`, protects a model you retrain later: if a future version fails it, you find out before a user does.
- **Model card:** a small JSON file next to the model: which model and threshold, the requirement, the seed, a data fingerprint, library versions, the cross-validation table, the test counts, and the known limits (569 rows from one source, no range checks on inputs, not for real medical use).

## Implementation

Save as `project.py` and run it (under a minute on a CPU). It needs numpy, scikit-learn and joblib. It writes a folder `project_run/` with `model.joblib` and `model_card.json`.

```python
import warnings; warnings.filterwarnings("ignore")
import hashlib, json, platform, pathlib
import numpy as np, sklearn, joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

SEED = 0; RECALL_TARGET = 0.98                          # the requirement: find at least 98% of malignant cases on the validation rows
out = pathlib.Path("project_run"); out.mkdir(exist_ok=True)

# 1. Data and the three-way split (Part 4)
X, y = load_breast_cancer(return_X_y=True); y = 1 - y   # 1 = malignant
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=SEED)
print(f"1. split: train {len(y_train)}, validation {len(y_val)}, test {len(y_test)}")

# 2. Baseline and candidates, compared by cross-validation on the training rows (Parts 5-7)
candidates = {
    "baseline: always benign": DummyClassifier(strategy="most_frequent"),
    "logistic regression C=1": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000)),
    "logistic regression C=0.1": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    "random forest": RandomForestClassifier(300, random_state=SEED, n_jobs=-1),
    "gradient boosting": HistGradientBoostingClassifier(random_state=SEED)}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED); scores = {}
print("\n2. 5-fold cross-validation on the training rows")
for name, m in candidates.items():
    s = cross_val_score(m, X_train, y_train, cv=cv); scores[name] = s.mean(); print(f"   {name:28s} {100*s.mean():5.1f}% +- {100*s.std():.1f}")
best = max((k for k in scores if not k.startswith("baseline")), key=scores.get)
# prefer the simplest model within one spread (about 1.5 points) of the best
simple = [k for k in scores if k.startswith("logistic") and scores[k] >= scores[best] - 0.015]
chosen = simple[0] if simple else best
print(f"   best score: {best}; chosen (simplest within 1.5 points): {chosen}")

# 3. Choose the decision threshold on the validation rows to meet the recall requirement (Part 6)
model = candidates[chosen].fit(X_train, y_train); p_val = model.predict_proba(X_val)[:, 1]
ok = [t for t in np.linspace(0.01, 0.99, 99) if ((p_val >= t) & (y_val == 1)).sum() / (y_val == 1).sum() >= RECALL_TARGET]
threshold = round(float(max(ok)), 2)                    # the highest threshold that still meets the requirement: fewest false alarms
q = p_val >= threshold
print(f"\n3. threshold chosen on validation: {threshold}  (recall {100*((q)&(y_val==1)).sum()/(y_val==1).sum():.1f}%, false alarms {int(((q)&(y_val==0)).sum())})")

# 4. Refit on train + validation with the frozen settings, then open the test set once (Part 4)
final = candidates[chosen].fit(X_rest, y_rest); p = final.predict_proba(X_test)[:, 1]; pred = (p >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
rng = np.random.default_rng(SEED); n = len(y_test); acc_b, rec_b = [], []
for _ in range(2000):                                   # bootstrap: resample the test rows to see how much the numbers could move
    i = rng.integers(0, n, n); yt, pr = y_test[i], pred[i]; acc_b.append((yt == pr).mean()); rec_b.append(((pr == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1))
ci = lambda v: f"{100*np.percentile(v, 2.5):.1f} to {100*np.percentile(v, 97.5):.1f}"
base_acc = (y_test == 0).mean()
print(f"\n4. test set (looked at once), {n} rows: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"   accuracy {100*(tp+tn)/n:.1f}%  (95% interval {ci(acc_b)}),  baseline {100*base_acc:.1f}%")
print(f"   recall   {100*tp/(tp+fn):.1f}%  (95% interval {ci(rec_b)}),  precision {100*tp/(tp+fp):.1f}%")

# 5. Save everything a colleague needs (Part 9) and self-check
card = {"model": chosen, "threshold": threshold, "recall_target_on_validation": RECALL_TARGET, "seed": SEED,
        "data_sha256": hashlib.sha256(X.tobytes()).hexdigest()[:16], "sklearn": sklearn.__version__, "python": platform.python_version(),
        "cv_scores_train": {k: round(v, 4) for k, v in scores.items()},
        "test": {"n": int(n), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "known_limits": "569 rows from one source; inputs outside the training ranges are not checked; not a medical device"}
joblib.dump({"model": final, "threshold": threshold}, out / "model.joblib"); (out / "model_card.json").write_text(json.dumps(card, indent=1))
back = joblib.load(out / "model.joblib")
assert ((back["model"].predict_proba(X_test)[:, 1] >= back["threshold"]).astype(int) == pred).all(), "saved model disagrees"
assert (tp + tn) / n > base_acc + 0.20, "does not clearly beat the baseline"
print("\n5. saved:", sorted(f.name for f in out.iterdir()), "| self-checks passed: reload gives identical predictions; beats the baseline by more than 20 points")
json.dump({"scores": scores, "cm": [int(tn), int(fp), int(fn), int(tp)], "ci_rec": [float(np.percentile(rec_b, 2.5)), float(np.percentile(rec_b, 97.5))], "ci_acc": [float(np.percentile(acc_b, 2.5)), float(np.percentile(acc_b, 97.5))], "threshold": threshold, "val": [int(((q)&(y_val==0)).sum())]}, open("proj11.json", "w"))
```

Output:

```
1. split: train 341, validation 114, test 114

2. 5-fold cross-validation on the training rows
   baseline: always benign       62.8% +- 0.6
   logistic regression C=1       96.8% +- 3.1
   logistic regression C=0.1     96.5% +- 4.4
   random forest                 95.6% +- 2.3
   gradient boosting             94.4% +- 4.1
   best score: logistic regression C=1; chosen (simplest within 1.5 points): logistic regression C=1

3. threshold chosen on validation: 0.64  (recall 100.0%, false alarms 0)

4. test set (looked at once), 114 rows: TN=72 FP=0 FN=3 TP=39
   accuracy 97.4%  (95% interval 94.7 to 100.0),  baseline 63.2%
   recall   92.9%  (95% interval 84.6 to 100.0),  precision 100.0%

5. saved: ['model.joblib', 'model_card.json'] | self-checks passed: reload gives identical predictions; beats the baseline by more than 20 points
```

## Try it

1. Replace the threshold step with out-of-fold probabilities: `cross_val_predict(model, X_rest, y_rest, cv=5, method="predict_proba")`, then pick the threshold from those 455 rows. How does it change, and what does the test set say now? (Run it as a new experiment and keep the old result in the model card; do not just overwrite it.)
2. Change `RECALL_TARGET` to 0.95 and rerun. Which numbers move, and by how much?
3. Add a third self-check: the model's accuracy on each of the 5 cross-validation folds must stay above 90%.
