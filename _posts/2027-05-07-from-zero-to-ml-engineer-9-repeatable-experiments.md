---
title: "From Zero to ML Engineer, Part 9: Experiments You Can Repeat"
date: 2027-05-07 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, reproducibility, experiment tracking, random seed, joblib, ML engineer]
description: "Part 9 of a 12-part series for complete beginners. Why the same code gives different numbers, how a random seed fixes that, how to tell a real improvement from seed noise, and a 40-line run folder that records settings, data fingerprint, metrics and the trained model."
---

## What you will do, and what you get

You will turn "I ran something and the number was good" into a record that anyone, including you in a month, can re-run and compare. The program fixes the randomness, runs five experiments, writes each one to its own folder (settings, a fingerprint of the data, library versions, metrics, the trained model), prints a ranked comparison from those files, and checks that a saved model loads back identical. You will also see the reason this matters in numbers: a difference of half a point between two settings turns out to be about the size of what the random seed alone changes.

![Three panels. Left: histograms of test accuracy over 30 seeds for two random forest settings, overlapping heavily, with means 94.39 and 93.80 percent. Middle: a diagram of one run folder containing config.json, metrics.json and model.joblib. Right: five runs ranked by cross-validated accuracy, logistic regression with C equal 1 on top at 98.8 percent]({{ site.baseurl }}/assets/pic/mlpath-9-experiments.png){: width="900" }

## How: the steps

1. **Set the random seed** of everything that uses randomness: the data split, the model, the cross-validation folds.
2. **Describe each experiment as one settings dictionary**, and make a single function that runs it from that dictionary.
3. **Write every run to its own folder:** the settings, a data fingerprint, library versions, the metrics, and the trained model.
4. **Compare runs by reading the files**, never from memory or from terminal scrollback.
5. **Repeat across seeds before believing a small difference.**

## Why it works: the principle

**Randomness is everywhere, and invisible.** A random forest picks random rows and columns; a split shuffles rows; a neural network starts from random weights. Run the same forest three times without a seed on the same split and you get 94.7%, 95.3% and 93.0%. With `random_state=0` you get 94.7% every time. Fixing seeds does not make the model better; it makes the run *replayable*, so that a change in the number can only come from a change you made. The [scikit-learn pitfalls page](https://scikit-learn.org/stable/common_pitfalls.html) has a section on controlling randomness and explains how the `random_state` argument behaves.

**A seed is not a fix for noise.** A fixed seed gives one draw from a distribution of results. The left panel shows the distribution: the same random forest on the same split, with 30 different seeds, ranges from 93.6% to 95.9%. A second setting (`max_features=0.5`) scores 0.58 points lower on average, and the two histograms overlap heavily: the seed-to-seed standard deviation is 0.65 and 0.54 points. A single run of each, with the seed you happened to pick, can show either setting ahead. The rule: **a difference counts only when it is clearly larger than the spread across seeds** (or across cross-validation folds, as in Part 6).

**A run folder is a small, boring database.** Experiment-tracking tools such as [MLflow](https://mlflow.org/docs/latest/ml/tracking/) exist for teams and add a web interface; the idea underneath is the folder in the middle panel. Each file answers a question you will ask later:

- *What did I run?* `config.json`: settings plus the random seed.
- *On which data?* A SHA-256 hash of the data's bytes: if the hash differs, the data differs, whatever the file is called.
- *In which environment?* Library and Python versions: the same code can give different numbers on a different version.
- *What happened?* `metrics.json`.
- *What did it produce?* `model.joblib`, the whole pipeline (scaler and model together), so nothing can be applied in a different way at prediction time.

**Saving models has a catch.** `joblib` (like `pickle`) stores Python objects, and the [scikit-learn persistence guide](https://scikit-learn.org/stable/model_persistence.html) states that pickle-based persistence may trigger execution of malicious code when loading a model from an untrusted source. Only load files you or your team produced, and load them with the same library versions that saved them: which is another reason to record the versions.

## Example, worked through

- **Seeds on, seeds off:** unseeded forest `[94.7, 95.3, 93.0]`; seeded `[94.7, 94.7, 94.7]`.
- **Seed spread:** 30 seeds, setting A mean 94.39% (range 93.6 to 95.9), setting B 93.80% (range 93.0 to 94.7). The gap is 0.58 points, about one standard deviation of either, so the data do not support calling A better.
- **Five runs, ranked by 5-fold cross-validation on the training rows:** `lr_C1` 98.8% (std 1.1), `lr_C100` 96.7%, `rf_300` 95.7%, `rf_50` 95.5%, `lr_C0.01` 95.0%. (The 98.8% differs slightly from the 98.5% in Part 7 because the folds are drawn with a different seed: one more demonstration of seed spread.)
- **Cost of tracking:** each run takes 0.02 to 1.5 seconds and writes three small files.
- **Round trip:** the saved pipeline loads back and predicts exactly like a freshly trained one (`True`), with a test accuracy of 95.9%.

## Implementation

Save as `part9.py` and run it. It needs numpy, scikit-learn and joblib (installed with scikit-learn). The folder `runs/` appears next to the script; open the JSON files in it.

```python
import warnings; warnings.filterwarnings("ignore")
import hashlib, json, platform, time, pathlib, shutil
import numpy as np, sklearn, joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# 1. Randomness: the same code, run three times
print("1. random forest, random_state=None, same split:", [round(100 * RandomForestClassifier(50).fit(Xa, ya).score(Xt, yt), 1) for _ in range(3)])
print("   random forest, random_state=0,    same split:", [round(100 * RandomForestClassifier(50, random_state=0).fit(Xa, ya).score(Xt, yt), 1) for _ in range(3)])

# 2. The seed spread: how much do two settings differ, compared with what a seed alone changes?
def spread(**kw): return np.array([100 * RandomForestClassifier(100, random_state=s, **kw).fit(Xa, ya).score(Xt, yt) for s in range(30)])
A, B = spread(), spread(max_features=0.5)
print(f"\n2. 30 seeds, same split: setting A (default) mean {A.mean():.2f}% range {A.min():.1f}-{A.max():.1f};  setting B (max_features=0.5) mean {B.mean():.2f}% range {B.min():.1f}-{B.max():.1f}")
print(f"   difference of means {B.mean()-A.mean():+.2f} points; spread across seeds (std) A {A.std():.2f} B {B.std():.2f}")

# 3. A run is a folder: config, data fingerprint, environment, metrics, model
runs = pathlib.Path("runs"); shutil.rmtree(runs, ignore_errors=True)
def run_experiment(name, config):
    out = runs / name; out.mkdir(parents=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **config["lr"])) if config["model"] == "lr" else RandomForestClassifier(**config["rf"])
    t0 = time.time()
    cv = cross_val_score(model, Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=config["seed"]))
    model.fit(Xa, ya)
    meta = {"config": config, "data_sha256": hashlib.sha256(X.tobytes()).hexdigest()[:16], "sklearn": sklearn.__version__, "python": platform.python_version()}
    metrics = {"cv_mean": round(float(cv.mean()), 4), "cv_std": round(float(cv.std()), 4), "seconds": round(time.time() - t0, 2)}
    (out / "config.json").write_text(json.dumps(meta, indent=1)); (out / "metrics.json").write_text(json.dumps(metrics)); joblib.dump(model, out / "model.joblib")
    return out
configs = {"lr_C0.01": {"model": "lr", "seed": 0, "lr": {"C": 0.01}}, "lr_C1": {"model": "lr", "seed": 0, "lr": {"C": 1}},
           "lr_C100": {"model": "lr", "seed": 0, "lr": {"C": 100}},
           "rf_50": {"model": "rf", "seed": 0, "rf": {"n_estimators": 50, "random_state": 0}}, "rf_300": {"model": "rf", "seed": 0, "rf": {"n_estimators": 300, "random_state": 0}}}
for k, c in configs.items(): run_experiment(k, c)
print("\n3. files in one run:", sorted(p.name for p in (runs / "lr_C1").iterdir()))
print("   config.json:", json.loads((runs / "lr_C1" / "config.json").read_text()))
rows = [(d.name, json.loads((d / "metrics.json").read_text())) for d in runs.iterdir()]
print("   all runs, best first:")
for n, m in sorted(rows, key=lambda r: -r[1]["cv_mean"]): print(f"   {n:9s} cv {100*m['cv_mean']:.1f}% +- {100*m['cv_std']:.1f}   {m['seconds']}s")

# 4. The saved model comes back identical
loaded = joblib.load(runs / "lr_C1" / "model.joblib"); fresh = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xa, ya)
print("\n4. saved model and a freshly trained one predict the same on all test rows:", bool((loaded.predict(Xt) == fresh.predict(Xt)).all()), "| test accuracy", round(100 * loaded.score(Xt, yt), 1), "%")
np.save("seed9.npy", np.stack([A, B])); json.dump({n: m for n, m in rows}, open("rows9.json", "w"))
```

Output:

```
1. random forest, random_state=None, same split: [94.7, 95.3, 93.0]
   random forest, random_state=0,    same split: [94.7, 94.7, 94.7]

2. 30 seeds, same split: setting A (default) mean 94.39% range 93.6-95.9;  setting B (max_features=0.5) mean 93.80% range 93.0-94.7
   difference of means -0.58 points; spread across seeds (std) A 0.65 B 0.54

3. files in one run: ['config.json', 'metrics.json', 'model.joblib']
   config.json: {'config': {'model': 'lr', 'seed': 0, 'lr': {'C': 1}}, 'data_sha256': '6b202a2072f9a038', 'sklearn': '1.9.1', 'python': '3.14.7'}
   all runs, best first:
   lr_C1     cv 98.8% +- 1.1   0.02s
   lr_C100   cv 96.7% +- 1.5   0.03s
   rf_300    cv 95.7% +- 0.6   1.5s
   rf_50     cv 95.5% +- 1.5   0.26s
   lr_C0.01  cv 95.0% +- 1.1   0.02s

4. saved model and a freshly trained one predict the same on all test rows: True | test accuracy 95.9 %
```

The versions in `config.json` are the ones this run used; yours will show your own.

## Try it

1. Add the library versions and a start time to `metrics.json`, and add a `notes` field to the settings. When would you want each?
2. Change one byte of the data (for example `X[0, 0] += 1`) and rerun: what happens to the data hash?
3. Write a ten-line `compare.py` that reads every `runs/*/metrics.json` and prints the best run, without importing any model code.
