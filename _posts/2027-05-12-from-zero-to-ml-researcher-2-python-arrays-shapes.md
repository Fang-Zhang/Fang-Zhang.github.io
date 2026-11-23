---
title: "From Zero to ML Researcher, Part 2: Python, Arrays and Shapes"
date: 2027-05-12 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, numpy, broadcasting, python, ML researcher]
description: "Part 2 of a 12-part series for complete beginners. Almost every model in later parts is arithmetic on arrays, and almost every bug is a wrong shape. Learn axes, reshaping and broadcasting, replace a double loop with one expression, and catch a bug that reports 10% instead of 89%."
---

## What you will do, and what you get

You will learn the small part of Python and numpy that the rest of the series uses: arrays, shapes, axes, reshaping, boolean selection and broadcasting. The payoff is concrete. The distance computation from Part 1 (797 images against 10 average pictures) becomes one line instead of a double loop, gives the same numbers, and runs about 16 times faster on the test machine (126 ms against 8 ms). You will also reproduce a silent bug, where one wrongly shaped array turns an 89.2% accuracy into a plausible-looking 10.0%, and learn the habit that catches it.

![Left: a 3 by 4 array with its axes labelled, and the results of summing over axis 0 (12, 15, 18, 21) and over axis 1 (6, 22, 38). Middle: a column of three numbers plus a row of four numbers broadcast to a 3 by 4 grid. Right: the shapes in the Part 1 computation: 797 by 64 and 10 by 64 become 797 by 1 by 64 and 1 by 10 by 64, a difference of shape 797 by 10 by 64, and after summing the last axis 797 by 10; below it, a warning box about a shape bug that reports 10 percent accuracy]({{ site.baseurl }}/assets/pic/mlr-2-shapes.png){: width="900" }

## How: the steps

You need Python 3 and `pip install numpy scikit-learn`. The official [Python tutorial](https://docs.python.org/3/tutorial/index.html) and NumPy's [absolute beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html) cover the rest of the basics; this post covers what the series needs.

1. **Make an array and read its `shape`.** Every other step starts from this.
2. **Select** a row (`a[1]`), a column (`a[:, 2]`), or the rows that satisfy a condition (`X[y == 7]`).
3. **Collapse an axis** with `sum`, `mean`, `argmin` and the `axis=` argument.
4. **Reshape**: the same numbers seen with a different shape (64 numbers as an 8 by 8 image).
5. **Broadcast** two arrays of different shapes in one arithmetic expression.
6. **Replace a loop with a broadcast expression,** and check that both give the same numbers.
7. **State the shape you expect** with an `assert`, before you trust a result.

## Why it works: the principle

**A shape is the type of an array.** `(797, 64)` means 797 examples with 64 numbers each. Axis 0 runs down the rows and axis 1 across the columns; `a.sum(axis=0)` collapses the rows and leaves one number per column, `a.sum(axis=1)` collapses the columns and leaves one number per row (see [`numpy.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)). A rule that removes most confusion: *the axis you name is the axis that disappears.*

**Broadcasting is the rule for mixed shapes.** According to the [NumPy documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html), two arrays are compared axis by axis starting from the last one, and two sizes are compatible when they are equal or one of them is 1; the axis of size 1 is stretched to match the other. So `(3,1) + (1,4)` gives `(3,4)`: every row-number is added to every column-number. A missing leading axis counts as 1. The distance computation uses exactly this: `images[:, None, :]` has shape `(797, 1, 64)`, `centroids[None, :, :]` has shape `(1, 10, 64)`, their difference has shape `(797, 10, 64)` (every image minus every average), and summing the last axis leaves `(797, 10)`.

**Why a model is arithmetic on arrays.** A neural network, a loss, a gradient step: in Parts 3 to 8 each is a sequence of operations on arrays. Reading a line of such code means reading the shapes. Writing it means choosing them. Most beginner time is lost on a shape that is not what the author believed.

**Why vectorised code is faster.** The loop version makes about 510,000 Python-level operations (797 images, 10 averages, 64 pixels each, one at a time). The broadcast version hands the whole block to compiled code in one call. The test machine took 126 ms for the loop and 8 ms for the broadcast (your numbers will differ; the ratio matters). For a toy this is irrelevant; for training a model, which repeats the same computation thousands of times, it decides what is feasible.

**Pitfalls and limits.**
- **Silent broadcasting is the commonest source of wrong-but-plausible results.** Comparing predictions of shape `(797,)` with labels of shape `(797, 1)` does not fail. It broadcasts to `(797, 797)`, compares every prediction with every label, and the mean is about 10%, because about a tenth of all pairs match by chance. No error is raised. The habit that catches it: write down the shape you expect, and `assert` it.
- **Memory:** a broadcast intermediate is as large as its biggest shape. `(797, 10, 64)` is 510,080 numbers, harmless; the same trick on a million images would need a different approach.
- **Views versus copies:** slicing (`a[1]`, `a[:, 2]`) usually returns a view of the same memory, so changing the slice changes the original. When you need an independent array, call `.copy()`.

## Example, worked through

Take `a = np.arange(12).reshape(3, 4)`: the numbers 0 to 11 laid out in 3 rows and 4 columns.

- `a[1]` is row 1: `[4 5 6 7]`. `a[:, 2]` is column 2: `[2 6 10]`.
- `a.sum(axis=0)` collapses the rows: `[12 15 18 21]` (0+4+8, 1+5+9, ...). `a.sum(axis=1)` collapses the columns: `[6 22 38]` (0+1+2+3, 4+5+6+7, 8+9+10+11).
- Broadcasting `[[10],[20],[30]] + [[1,2,3,4]]` gives a 3 by 4 array whose entry in row i, column j is the i-th column value plus the j-th row value: `[[11 12 13 14], [21 22 23 24], [31 32 33 34]]`.
- In the digits data, `X.shape` is `(1797, 64)`; `X[y == 7]` keeps the 179 rows whose label is 7, shape `(179, 64)`; `X[0].reshape(8, 8)` is the first image as a grid.
- The distance calculation, step by step in shapes, is in the figure and in the printed output below. The loop and the broadcast result agree (`np.allclose` is `True`).

## Implementation

Save as `part2.py` and run it (a few seconds on a CPU). It needs numpy and scikit-learn.

```python
import time
import numpy as np
from sklearn.datasets import load_digits

# 1. An array is a block of numbers with a shape. Axis 0 runs down the rows, axis 1 across the columns.
a = np.arange(12).reshape(3, 4)
print("a =\n", a, "\nshape:", a.shape)
print("a[1]       =", a[1], "  (row 1)")
print("a[:, 2]    =", a[:, 2], "  (column 2)")
print("a.sum(axis=0) =", a.sum(axis=0), "  (collapse the rows: one number per column)")
print("a.sum(axis=1) =", a.sum(axis=1), "  (collapse the columns: one number per row)")

# 2. A dataset is an array: one row per example. An image is the same 64 numbers seen as 8x8.
X, y = load_digits(return_X_y=True)
print("\nX.shape:", X.shape, " y.shape:", y.shape, " one image reshaped:", X[0].reshape(8, 8).shape)
print("rows where the digit is 7:", (y == 7).sum(), "-> X[y == 7].shape:", X[y == 7].shape)

# 3. Broadcasting: size-1 axes are stretched to match. (3,1) + (1,4) -> (3,4)
col = np.array([[10], [20], [30]]); row = np.array([[1, 2, 3, 4]])
print("\n(3,1) + (1,4) =\n", col + row, "\nshape:", (col + row).shape)

# 4. The distance computation of Part 1: 797 images x 10 averages, as a loop and as one broadcast expression.
rng = np.random.default_rng(0); order = rng.permutation(len(y)); train, test = order[:1000], order[1000:]
centroids = np.array([X[train][y[train] == d].mean(axis=0) for d in range(10)])
images = X[test]
t0 = time.perf_counter(); loop = np.zeros((len(images), 10))
for i in range(len(images)):
    for d in range(10):
        loop[i, d] = sum((images[i, p] - centroids[d, p]) ** 2 for p in range(64))
t1 = time.perf_counter()
fast = ((images[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
t2 = time.perf_counter()
print(f"\nloop result == broadcast result: {np.allclose(loop, fast)}")
print(f"shapes: images {images.shape} -> images[:, None, :] {images[:, None, :].shape}; centroids[None] {centroids[None].shape};")
print(f"        difference {(images[:, None, :] - centroids[None]).shape} -> sum(axis=2) {fast.shape}")
print(f"time: loop {1000*(t1-t0):.0f} ms, broadcast {1000*(t2-t1):.1f} ms")

# 5. The classic silent bug: (n,) against (n,1) broadcasts to (n,n) and still gives an answer.
pred = fast.argmin(axis=1); truth = y[test]
print(f"\ncorrect: accuracy {100*(pred == truth).mean():.1f}%")
wrong = (pred == truth.reshape(-1, 1))
print(f"bug:     pred (797,) == truth (797,1) has shape {wrong.shape}, 'accuracy' {100*wrong.mean():.1f}%")

# 6. The habit that catches it: state the shape you expect.
assert (pred == truth).shape == (797,), "unexpected shape"
```

Output:

```
a =
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]] 
shape: (3, 4)
a[1]       = [4 5 6 7]   (row 1)
a[:, 2]    = [ 2  6 10]   (column 2)
a.sum(axis=0) = [12 15 18 21]   (collapse the rows: one number per column)
a.sum(axis=1) = [ 6 22 38]   (collapse the columns: one number per row)

X.shape: (1797, 64)  y.shape: (1797,)  one image reshaped: (8, 8)
rows where the digit is 7: 179 -> X[y == 7].shape: (179, 64)

(3,1) + (1,4) =
 [[11 12 13 14]
 [21 22 23 24]
 [31 32 33 34]] 
shape: (3, 4)

loop result == broadcast result: True
shapes: images (797, 64) -> images[:, None, :] (797, 1, 64); centroids[None] (1, 10, 64);
        difference (797, 10, 64) -> sum(axis=2) (797, 10)
time: loop 126 ms, broadcast 8.0 ms

correct: accuracy 89.2%
bug:     pred (797,) == truth (797,1) has shape (797, 797), 'accuracy' 10.0%
```

## Try it

1. Predict the shape of `X[:5].mean(axis=1)` and of `X[:5][:, None, :] - X[:3][None, :, :]` before running them. Then run them.
2. Rewrite the distance as `np.linalg.norm(images[:, None, :] - centroids[None], axis=2)` and compare with `fast`. What is different, and why does `argmin` give the same predictions?
3. Add an `assert fast.shape == (len(images), 10)` and deliberately break it by writing `centroids[:, None, :]`. What shape does the error message report?
