---
title: "From Zero to ML Researcher, Part 1: The Map"
date: 2027-05-11 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, research, roadmap, python, ML researcher]
description: "Part 1 of a 12-part series for complete beginners. A researcher produces evidence, not just working programs. A 20-line program learns to read handwritten digits, one experiment shows how its accuracy depends on the amount of data, and the plateau it hits becomes the first research question."
---

## What you will do, and what you get

You will run a 20-line Python program that learns to recognise handwritten digits from examples, and scores 89.2% on 797 images it never saw (guessing would score about 10%). Then you will do the step that separates research from programming: turn a curiosity into a question, and run an experiment that answers it. The question: *how does accuracy depend on the number of training images?* The answer comes back as a curve with a plateau at 89.3%. The plateau is the real result of this post, because it raises the next question, and the next eleven parts follow that thread from this toy to a small language model studied with the same discipline.

![Left: the research loop, from curiosity to question, experiment, evidence and claim, with an arrow back to a new question. Middle: ten small grey images, the average picture the program learned for each digit from 0 to 9. Right: accuracy against training images per digit, from 65.9 percent with 1 image to 89.3 percent with 80 images, with a shaded band showing the spread over 30 random draws and a dotted line at 10 percent for guessing]({{ site.baseurl }}/assets/pic/mlr-1-map.png){: width="900" }

## How: the steps

You need Python 3 ([download](https://www.python.org/downloads/)) and two libraries: `pip install numpy scikit-learn`. No accounts, no GPU, no downloads beyond the install; the data ships with scikit-learn.

1. **Load the data.** 1,797 images of handwritten digits, each 8 by 8 pixels (the [UCI optical recognition dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), through scikit-learn's [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)).
2. **Split.** Shuffle, keep 1,000 images for learning and 797 for judging.
3. **Learn.** For each digit, average all of its training images into one picture.
4. **Predict.** For a new image, find which of the ten average pictures it is closest to.
5. **Score** on the 797 images the program never saw.
6. **Ask a question, then design the experiment.** Repeat the whole thing with 1, 2, 5, 10, 20, 40 and 80 training images per digit, 30 times each with different random draws, and look at the spread, not only the mean.

## Why it works: the principle

**Learning, here, means computing something from examples that you can reuse on new examples.** The thing computed is ten average images (they are the grey pictures in the figure). Nothing else is stored. The rule for a new image is: pick the digit whose average is nearest, where *nearest* means the smallest sum of squared pixel differences. A learner of this kind is called a *nearest-centroid classifier*; scikit-learn documents a version of it among its [neighbour methods](https://scikit-learn.org/stable/modules/neighbors.html).

**Why judge on images the program never saw.** A program that memorised its training images would score 100% on them and tell you nothing about new ones. Only unseen images measure whether what was learned carries over. (Part 4 builds this carefully; for now, keep the two sets apart.)

**What a researcher adds: the loop.** A programmer asks "does it work?" A researcher asks a question that can be answered by an experiment, changes *one* thing, collects evidence with its spread, and states only what the evidence supports. The loop in the figure is the working method of the whole series.

**Reading the experiment.**
- Accuracy rises quickly (65.9% with one image per digit, 83.4% with five), then flattens: 89.3% at 40 images per digit, 89.3% at 80.
- The spread shrinks as data grows: the standard deviation over 30 draws is 6.1 points with one image per digit and 0.3 with 80. With one image per digit the *worst* draw scored 53.2% and the *best* 80.2%. Had you run it once, you could have reported either number.
- **The plateau is the finding.** More data of the same kind stopped helping long before the data ran out. A hypothesis, not a conclusion: the model is too simple. One average picture per digit cannot represent that a 7 can be written with or without a bar, or a 1 with or without a foot. The next parts build models that can, and test that hypothesis instead of assuming it.

**Pitfalls and limits.**
- One run proves little. The spread across draws (6.1 points at the smallest size) is as large as many differences people report as improvements.
- The 89.3% is specific to this dataset (small 8 by 8 images). It does not carry over to other kinds of images.
- The test images were reused across all experiments. For a toy this is acceptable; Part 4 explains why a long series of decisions made on the same test set slowly inflates the score.

## Example, worked through

Take the first unseen image. Its true digit is 1. Its first row of pixel values (0 means white, 16 means black) is:

pixel values (0-16), first row of the image: [ 0.  0.  0. 12. 16.  9.  0.  0.]

The program measures the squared distance to each of the ten average pictures:

```
distance to the average 0:  2843.7
distance to the average 1:   683.3
distance to the average 2:  1460.5
distance to the average 3:  2019.4
distance to the average 4:  1604.4
distance to the average 5:  1679.2
distance to the average 6:  1900.2
distance to the average 7:  1264.8
distance to the average 8:  1076.4
distance to the average 9:  2415.8
```

The smallest distance is to the average 1 (683.3), so the prediction is 1. Correct. The runner-up is the average 8 (1,076.4), which shows the geometry: a thin vertical stroke is somewhat near a blob that has a vertical middle.

Across all 797 images the program makes 86 mistakes. The most common confusions are 5 predicted as 9 (8 times), 1 as 8 (8 times), 2 as 8 (6 times) and 8 as 9 (6 times): digits that share strokes. Looking at *which* mistakes occur is itself a research habit, and it is where the next hypotheses come from.

## Implementation

Save as `part1.py` and run it (a few seconds on a CPU). It prints the accuracy and then the experiment table.

```python
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)          # 1,797 images of handwritten digits, 8x8 pixels, flattened to 64 numbers
rng = np.random.default_rng(0)
order = rng.permutation(len(y))
train, test = order[:1000], order[1000:]     # learn from 1,000 images, be judged on 797 the program never saw

# Learn: one average image per digit.
centroids = np.array([X[train][y[train] == d].mean(axis=0) for d in range(10)])

# Predict: the digit whose average image is closest.
def predict(images):
    distances = ((images[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)

print(f"accuracy on 797 unseen images: {100 * (predict(X[test]) == y[test]).mean():.1f}%")

# The research step: turn a curiosity into a question, and the question into an experiment.
# Question: how does accuracy depend on the number of training images per digit?
# Experiment: for each size, repeat 30 times with a different random draw, then look at the spread.
print("\nimages per digit   mean accuracy   std    worst   best")
results = {}
for k in [1, 2, 5, 10, 20, 40, 80]:
    scores = []
    for repeat in range(30):
        r = np.random.default_rng(1000 + repeat)
        chosen = np.concatenate([r.choice(train[y[train] == d], k, replace=False) for d in range(10)])
        c = np.array([X[chosen][y[chosen] == d].mean(axis=0) for d in range(10)])
        d2 = ((X[test][:, None, :] - c[None]) ** 2).sum(axis=2)
        scores.append(100 * (d2.argmin(axis=1) == y[test]).mean())
    results[k] = scores
    print(f"{k:16d}   {np.mean(scores):12.1f}%   {np.std(scores):4.1f}   {np.min(scores):5.1f}  {np.max(scores):5.1f}")
np.save("curve1.npy", np.array([results[k] for k in results]))
np.save("centroids1.npy", centroids)
```

Output:

```
accuracy on 797 unseen images: 89.2%

images per digit   mean accuracy   std    worst   best
               1           65.9%    6.1    53.2   80.2
               2           75.4%    4.4    64.9   81.6
               5           83.4%    2.1    78.8   88.3
              10           87.2%    1.7    83.6   90.2
              20           88.7%    0.7    87.5   90.3
              40           89.3%    0.7    87.8   90.2
              80           89.3%    0.3    88.7   90.1
```

To reproduce the worked example, save this as `example.py`:

```python
import numpy as np
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
rng = np.random.default_rng(0); order = rng.permutation(len(y)); train, test = order[:1000], order[1000:]
centroids = np.array([X[train][y[train] == d].mean(axis=0) for d in range(10)])
i = test[0]
print("true digit:", y[i]); print("pixel values (0-16), first row of the image:", X[i][:8])
dist = ((X[i] - centroids) ** 2).sum(axis=1)
for d in range(10): print(f"distance to the average {d}: {dist[d]:7.1f}")
print("closest:", dist.argmin())
wrong = [j for j in test if ((X[j] - centroids) ** 2).sum(axis=1).argmin() != y[j]]
from collections import Counter
print("most common mistakes (true -> predicted):", Counter((int(y[j]), int(((X[j]-centroids)**2).sum(axis=1).argmin())) for j in wrong).most_common(4), "of", len(wrong), "errors")
```

Output:

```
true digit: 1
pixel values (0-16), first row of the image: [ 0.  0.  0. 12. 16.  9.  0.  0.]
distance to the average 0:  2843.7
distance to the average 1:   683.3
distance to the average 2:  1460.5
distance to the average 3:  2019.4
distance to the average 4:  1604.4
distance to the average 5:  1679.2
distance to the average 6:  1900.2
distance to the average 7:  1264.8
distance to the average 8:  1076.4
distance to the average 9:  2415.8
closest: 1
most common mistakes (true -> predicted): [((5, 9), 8), ((1, 8), 8), ((2, 8), 6), ((8, 9), 6)] of 86 errors
```

## The road ahead

Twelve parts, one thread, each runnable on a CPU with no accounts:

1. The map (this post).
2. Python, arrays and shapes: the minimum of numpy.
3. What learning is: fit a line by gradient descent.
4. Evidence: baselines, held-out data, leakage, seeds and noise.
5. A neural network from scratch, with backpropagation checked numerically.
6. Language as prediction: a character-level baseline.
7. Attention from scratch.
8. A small transformer trained on a CPU.
9. Hypothesis and ablation: is a difference real?
10. Scaling laws in miniature.
11. Reading and reproducing a paper.
12. When it fails, and writing it up.

## Try it

1. Change the 797/1,000 split to 1,497/300 and rerun. Which numbers move, and by how much?
2. Run the experiment with only digits 3, 5 and 8 (the confusable ones). Is the plateau higher or lower?
3. Replace "closest average" with "closest single training image" (nearest neighbour). Does it plateau at the same level?
