---
title: "From Zero to ML Engineer, Part 12: Watching a Model After It Ships, and Where to Go Next"
date: 2027-05-10 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, monitoring, data drift, learning path, ML engineer]
description: "Part 12 of a 12-part series for complete beginners. A deployed model fails silently when its inputs change. A 30-line check detects an input change before any labels arrive, then a short, sourced list says what to learn next."
---

## What you will do, and what you get

You will run a 30-line input check against a model that is "live". In the simulation, the measuring device is recalibrated at batch 6 and reports two columns 30% too large. The check raises an alert on every one of the five changed batches and on none of the five healthy ones, using nothing but the inputs. Accuracy over the same batches fell only from 95.2% to 91.8%, and each batch has just 100 rows, so accuracy alone would have been hard to read as a signal. After the code, a short list says what to learn next, with sources.

![Left: bars show the largest shift of any column's mean, in standard errors, for ten batches of 100 rows. Batches 1 to 5 stay under 2.8; batches 6 to 10, where two area columns are 30 percent too large, are between 5.4 and 6.4, above a dashed alert limit at 3.14. A black line shows batch accuracy between 90 and 97 percent. Right: five boxes summarise the series: foundations, the honest workflow, networks, shipping, and what comes next]({{ site.baseurl }}/assets/pic/mlpath-12-next.png){: width="900" }

## How: the steps

1. **Keep the training statistics.** Save the mean and standard deviation of every input column when the model is trained (they fit in the model card from Part 11).
2. **For each batch of live inputs, compare each column's mean with the training mean**, in units of the standard error (the training standard deviation divided by the square root of the batch size).
3. **Alert when the largest shift passes a limit** chosen in advance: 3.14 standard errors here.
4. **When an alert fires, look before retraining.** The cause may be a broken sensor, a changed unit, a new data source, or a real change in the world, and each needs a different fix.
5. **Keep collecting true labels** for a sample of the traffic. The input check is an early warning; only labels measure accuracy.

## Why it works: the principle

**A model is a function fitted to one distribution of inputs.** Everything in Parts 4 to 7 measured it on rows resembling the training rows. After shipping, the inputs come from the world, and the world changes. When the inputs move away from what the model saw, its accuracy can fall without any error being raised: the service of Part 10 still returns a probability for any 30 numbers. [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml), a guide from Google, includes a rule on watching for silent failures, and [Sculley and colleagues (NeurIPS 2015)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) list changes in the external world and unstable data dependencies among the maintenance costs of deployed models.

**Why check the inputs.** Labels often arrive late or never: a loan defaults months later, and a cancer diagnosis is confirmed by a different test. Inputs arrive immediately, and a shift in them is visible at once. The check does not say the model is wrong; it says the model is now being asked questions unlike the ones it learned from, so its earlier test results no longer describe it.

**Where the limit 3.14 comes from.** If nothing has changed, the mean of a batch of 100 rows lies within about 2 standard errors of the training mean 95% of the time. With 30 columns checked at once, a few will wander past 2 by chance in nearly every batch, so the limit is raised until a false alarm is rare: a two-sided 5% level divided by 30 columns (a Bonferroni correction) gives 3.14. The ten batches here are drawn from the same 171 held-out rows, so they are not independent; the five healthy batches staying below 2.8 is a demonstration, not a measured false-alarm rate.

**Pitfalls and limits.**
- The check sees one column at a time. Two columns that move together while each stays in range can pass it. Multi-column checks exist, but need more care.
- With a small batch, small real drifts hide inside the noise; with a very large batch, harmless drifts trigger alerts. Choose the batch size and limit with the cost of a false alarm in mind.
- The accuracy drop in this demonstration is moderate (95.2% to 91.8%) because the model uses 30 columns and only two moved. A drift in the inputs the model leans on most would hurt more.
- Some drift is normal: seasons, product changes. An alert starts an investigation; it does not prove the model is broken.

## Example, worked through

Take batch 7. Its mean for "mean area" is about 6.4 standard errors from the training mean: the training mean is about 650, the standard deviation about 350, and a 100-row batch has a standard error of 350 / 10 = 35. A mean shifted by 30% is about 195 units, which is more than 5 standard errors, so the check fires. The healthy batch 5 had a largest shift of 2.7 (on a different column), below 3.14, so no alert. Batch accuracies in the drift period were 90, 93, 92, 90 and 94%, against 94, 96, 95, 94 and 97% before: lower on average, but the gap is about the size of the batch-to-batch noise on 100 rows, which is why the label-free check is useful.

## Implementation

Save as `part12.py` and run it (a few seconds on a CPU). It needs numpy, scikit-learn and scipy (which scikit-learn installs).

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer(); X, y = data.data, 1 - data.target
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xa, ya)
mu, sd = Xa.mean(0), Xa.std(0)
LIMIT = norm.ppf(1 - 0.05 / 2 / 30)       # two-sided 5% level, Bonferroni-corrected for 30 columns: about 3.2 standard errors
names = list(data.feature_names); area_cols = [names.index("mean area"), names.index("worst area")]

# "Live traffic": 10 batches of 100 rows drawn from the held-out rows. From batch 6 on, the measuring
# device is recalibrated and reports the two area columns 30% too large. The labels are unchanged.
rng = np.random.default_rng(0); rows = []
for b in range(1, 11):
    i = rng.choice(len(Xt), 100, replace=False); xb = Xt[i].copy()
    if b >= 6: xb[:, area_cols] *= 1.3
    acc = 100 * model.score(xb, yt[i])
    shift = np.abs((xb.mean(0) - mu) / (sd / np.sqrt(len(xb))))          # how many standard errors each column's mean has moved
    worst = int(shift.argmax()); alert = shift.max() > LIMIT
    rows.append((b, acc, shift.max(), names[worst], alert))
print(f"alert limit: {LIMIT:.2f} standard errors\nbatch  accuracy  largest mean shift (std. errors)  column               alert")
for b, acc, s, c, a in rows: print(f"{b:5d}  {acc:7.1f}%  {s:10.1f}                        {c:20s} {'YES' if a else 'no'}")
healthy = [r for r in rows if r[0] <= 5]; drift = [r for r in rows if r[0] >= 6]
print(f"\nbatches 1-5 (healthy): mean accuracy {np.mean([r[1] for r in healthy]):.1f}%, alerts {sum(r[4] for r in healthy)}")
print(f"batches 6-10 (area columns x1.3): mean accuracy {np.mean([r[1] for r in drift]):.1f}%, alerts {sum(r[4] for r in drift)}")
np.save("batches12.npy", np.array([[r[0], r[1], r[2], r[4]] for r in rows], dtype=float))
```

Output:

```
alert limit: 3.14 standard errors
batch  accuracy  largest mean shift (std. errors)  column               alert
    1     94.0%         1.8                        concavity error      no
    2     96.0%         1.9                        compactness error    no
    3     95.0%         2.2                        compactness error    no
    4     94.0%         1.5                        concavity error      no
    5     97.0%         2.7                        compactness error    no
    6     90.0%         5.4                        mean area            YES
    7     93.0%         6.4                        mean area            YES
    8     92.0%         5.4                        mean area            YES
    9     90.0%         6.2                        mean area            YES
   10     94.0%         6.2                        mean area            YES

batches 1-5 (healthy): mean accuracy 95.2%, alerts 0
batches 6-10 (area columns x1.3): mean accuracy 91.8%, alerts 5
```

## Where to go next

The path of this series was: data and tools, a model learned by hand, an honest split, features and models, metrics, tuning, a neural network, repeatable runs, a service, a full project, and monitoring. The open edges, with a source for each:

- **More classical methods and the details behind each one:** the [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) is the reference for every estimator used here.
- **Deep learning from the ground up:** the [Deep Learning book](https://www.deeplearningbook.org/) (Goodfellow, Bengio and Courville) covers the theory; the [PyTorch tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html) cover practice, building on Part 8.
- **Production practice:** [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) and the [hidden technical debt paper](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html).
- **A habit that outweighs any list:** take one real dataset, state a requirement first, write the one-script project of Part 11 for it, and report the result with an interval, including the parts that fail.

## The whole series

1. [The Map]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-1-the-map/)
2. [Tools and the First Data]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-2-tools-and-first-data/)
3. [What Learning Is]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-3-what-learning-is/)
4. [Preparing Data and Keeping the Test Honest]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-4-preparing-data/)
5. [Features and Your First Real Models]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-5-features-and-first-models/)
6. [Is the Model Any Good?]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-6-is-the-model-good/)
7. [Making It Better, and Knowing When to Stop]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-7-making-it-better/)
8. [Your First Neural Network]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-8-first-neural-network/)
9. [Experiments You Can Repeat]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-9-repeatable-experiments/)
10. [Serving a Model Over HTTP]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-10-serving-a-model/)
11. [A Complete Small Project, Including Its Failure]({{ site.baseurl }}/posts/from-zero-to-ml-engineer-11-a-complete-project/)
12. Watching a Model After It Ships, and Where to Go Next (this post)

## Try it

1. Change the drift factor from 1.3 to 1.1 and rerun. At which factor does the alert stop firing, and what happens to the accuracy at that point?
2. Apply the drift to a different column, such as "mean texture", and compare the accuracy drop.
3. Add a second alert: the share of rows with any value outside the training minimum and maximum for its column.
