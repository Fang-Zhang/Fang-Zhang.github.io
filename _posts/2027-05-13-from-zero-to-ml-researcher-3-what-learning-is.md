---
title: "From Zero to ML Researcher, Part 3: What Learning Is"
date: 2027-05-13 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, gradient descent, loss function, numpy, ML researcher]
description: "Part 3 of a 12-part series for complete beginners. Learning means adjusting numbers to reduce a loss. A line is fitted to data with a known answer by gradient descent in 15 lines of numpy, the gradient is checked numerically, and three learning rates show when learning works and when it explodes."
---

## What you will do, and what you get

You will fit a straight line to 50 noisy points by gradient descent, in plain numpy, with every step visible. The data are generated from a known rule (y = 3x + 2, plus noise), so you can check the method against the truth. After 500 steps the method finds w = 3.11 and b = 1.50 with loss 3.97, against 3.105, 1.515 and 3.97 from the exact least-squares solution. You will also watch a learning rate of 0.05 push the loss to infinity, and see why. This is all of "learning" in miniature; every later model repeats the same loop with more numbers.

![Left: 50 grey points around a rising line, with the fitted line at step 0 (flat at zero), step 10 and step 500, and the dashed true line. Middle: loss against step on a log scale for three learning rates: 0.001 falls slowly, 0.02 falls fast and settles near 4, 0.05 rises to a clipped value of one million. Right: a contour plot of the loss over w and b, a long narrow valley, with the path of learning rate 0.02 jumping first to w equal 4.8 and then zigzagging into the valley]({{ site.baseurl }}/assets/pic/mlr-3-learning.png){: width="900" }

## How: the steps

1. **Choose a model with adjustable numbers.** Here, a line: `y_hat = w * x + b`. The numbers `w` and `b` are the *parameters*.
2. **Choose a loss:** one number that says how wrong the model is. Here, the mean squared error: the average of `(y_hat - y)**2` over all points.
3. **Compute the gradient:** how much the loss changes when each parameter is nudged. For this model it has a closed form (below).
4. **Step downhill:** `w = w - lr * dL/dw`, `b = b - lr * dL/db`, where `lr` is the learning rate.
5. **Repeat** until the loss stops falling.
6. **Check:** compare the gradient with a numerical estimate, and the result with an exact solution and with the known truth.

## Why it works: the principle

**Learning is minimising a number.** The loss `L(w, b)` is a surface over the parameters: each pair `(w, b)` has a height. The right panel of the figure shows it for this problem. The best line is at the bottom of the valley. The gradient of `L` at a point is the direction of steepest ascent, so stepping against it goes downhill. Calculus gives the gradient for the line:

```
dL/dw = mean( 2 * (w*x + b - y) * x )
dL/db = mean( 2 * (w*x + b - y) )
```

**Trust but check the formula.** A derivative can be estimated without calculus: nudge the parameter by a tiny amount in both directions and divide the change in loss by the size of the nudge (a central finite difference). At w = 1, b = 1 the formula gives (-158.015, -23.206) and the numerical estimate gives the same numbers to four decimals. In Part 5 this check is used on a neural network, where a hand-derived gradient is easy to get wrong.

**The learning rate is a speed limit.** Too small and learning crawls (0.001 is still creeping at step 100, with loss 4.2). Too large and each step overshoots the valley by more than it fixes; the error grows each step and the loss reaches infinity (0.05). Here is the arithmetic. Near the bottom, the loss curves up with a steepness that depends on the data: for this data the steepest direction has a curvature of about 74.0 and the shallowest of about 0.46 (these are the eigenvalues of the loss's second-derivative matrix, also called the Hessian; see the treatment in the [Deep Learning book](https://www.deeplearningbook.org/contents/numerical.html)). A step of size `lr` multiplies the error along a direction by `1 - lr * curvature`. For the error to shrink in the steepest direction, `|1 - lr * 74.0|` must stay below 1, which needs `lr < 2 / 74.0 = 0.027`. So 0.02 works and 0.05 does not.

**Why 0.02 still zigzags, and why b ends at 1.50 and not 2.** With lr = 0.02 the factor in the steep direction is `1 - 0.02 * 74.0 = -0.48`: the error flips sign and shrinks by half each step. That is the zigzag in the right panel, including the first step that jumps from w = 0 to w = 4.82. Along the shallow direction the factor is `1 - 0.02 * 0.46 = 0.991`, so the error there shrinks by under 1% per step. After 500 steps, a bit of it remains: b = 1.50 against the exact least-squares value 1.515. The two directions need very different step sizes, which is a reason later parts rescale their inputs and use adaptive optimisers.

**Why the answer is not exactly the truth.** The data have noise with standard deviation 2, so the best possible loss is near 4 (the noise variance), and the best fitted line is w = 3.105, b = 1.515, not (3, 2). Fifty noisy points do not pin the true rule down exactly. Comparing against the truth is possible here only because the data were generated; real data come without a truth, which is why Part 4 builds evidence from held-out data instead.

**Pitfalls and limits.**
- A falling loss does not mean a correct model; it means a better fit on these points.
- The loss, not the accuracy, is what is minimised. They are different numbers, and Part 6 returns to this.
- For a line there is an exact answer (`np.polyfit`, [documentation](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)). Gradient descent matters because larger models have no exact answer, only the downhill walk.
- A learning rate that works on one problem may fail on another; the safe limit depends on the data.

## Example, worked through

Three points that lie exactly on `y = 3x + 2`: (1, 5), (2, 8), (3, 11). Start at `w = 0, b = 0`, with `lr = 0.05`.

- Predictions are all 0, so the errors `y_hat - y` are `-5, -8, -11` and the loss is `(25 + 64 + 121) / 3 = 70`.
- `dL/dw = 2 * mean(err * x) = 2 * (-5 - 16 - 33) / 3 = -36`, and `dL/db = 2 * mean(err) = 2 * (-24 / 3) = -16`.
- Step: `w = 0 - 0.05 * (-36) = 1.80`, `b = 0 - 0.05 * (-16) = 0.80`.
- New predictions: 2.6, 4.4, 6.2. Errors: -2.4, -3.6, -4.8. Loss: `(5.76 + 12.96 + 23.04) / 3 = 13.92`.

One step took the loss from 70 to 13.92. The code below reproduces these numbers.

## Implementation

Save as `part3.py` and run it (about a second on a CPU). It needs only numpy.

```python
import numpy as np
np.seterr(all="ignore")

# Data where the truth is known: y = 3x + 2 plus noise. A method can be tested on data like this, because the right answer is known.
rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 50)
y = 3 * x + 2 + rng.normal(0, 2, 50)

def loss(w, b):                      # mean squared error of the line  y_hat = w*x + b
    return np.mean((w * x + b - y) ** 2)

def gradient(w, b):                  # derivatives of the loss with respect to w and b
    err = w * x + b - y
    return 2 * np.mean(err * x), 2 * np.mean(err)

def descend(lr, steps, w=0.0, b=0.0):
    path = [(w, b, loss(w, b))]
    for _ in range(steps):
        gw, gb = gradient(w, b)
        w, b = w - lr * gw, b - lr * gb      # the whole of "learning": step against the gradient
        path.append((w, b, loss(w, b)))
    return np.array(path)

# 1. Check the gradient formulas against a numerical estimate (a habit worth keeping for every formula).
w0, b0, h = 1.0, 1.0, 1e-6
num_w = (loss(w0 + h, b0) - loss(w0 - h, b0)) / (2 * h); num_b = (loss(w0, b0 + h) - loss(w0, b0 - h)) / (2 * h)
print("formula  :", np.round(gradient(w0, b0), 4)); print("numerical:", np.round([num_w, num_b], 4))

# 2. Three learning rates.
print("\nlearning rate   loss after 1 / 10 / 100 / 500 steps            final w, b")
for lr in (0.001, 0.02, 0.05):
    p = descend(lr, 500)
    print(f"{lr:<14}  " + " / ".join(f"{p[s, 2]:.3g}" for s in (1, 10, 100, 500)) + f"      {p[-1, 0]:.3g}, {p[-1, 1]:.3g}")

# 3. Compare with the exact least-squares answer and with the truth.
wl, bl = np.polyfit(x, y, 1)
p = descend(0.02, 500)
print(f"\ngradient descent (lr 0.02, 500 steps): w = {p[-1,0]:.3f}, b = {p[-1,1]:.3f}, loss {p[-1,2]:.3f}")
print(f"exact least squares (np.polyfit)     : w = {wl:.3f}, b = {bl:.3f}, loss {loss(wl, bl):.3f}")
print("truth used to make the data          : w = 3, b = 2, noise std 2 (so the best possible loss is near 4)")

# 4. One step by hand on three points that lie exactly on y = 3x + 2.
xs, ys = np.array([1., 2., 3.]), np.array([5., 8., 11.]); w, b = 0.0, 0.0
err = w * xs + b - ys; gw, gb = 2 * np.mean(err * xs), 2 * np.mean(err)
print(f"\nby hand: errors {err}, loss {np.mean(err**2):.2f}, dw {gw:.1f}, db {gb:.1f}")
w, b = w - 0.05 * gw, b - 0.05 * gb
print(f"after one step with lr 0.05: w = {w:.2f}, b = {b:.2f}, loss {np.mean((w*xs+b-ys)**2):.2f}")
np.save("paths3.npy", np.array([descend(lr, 500) for lr in (0.001, 0.02)]), allow_pickle=False)
np.save("data3.npy", np.array([x, y]))
```

Output (the last learning rate prints astronomically large numbers because it diverges; that is the point):

```
formula  : [-158.015   -23.2062]
numerical: [-158.015   -23.2062]

learning rate   loss after 1 / 10 / 100 / 500 steps            final w, b
0.001           348 / 90.3 / 4.2 / 4.13      3.23, 0.689
0.02            96.9 / 4.18 / 4.01 / 3.97      3.11, 1.5
0.05            2.93e+03 / 1.73e+11 / 8.76e+88 / inf      -2.29e+216, -3.36e+215

gradient descent (lr 0.02, 500 steps): w = 3.106, b = 1.505, loss 3.974
exact least squares (np.polyfit)     : w = 3.105, b = 1.515, loss 3.973
truth used to make the data          : w = 3, b = 2, noise std 2 (so the best possible loss is near 4)

by hand: errors [ -5.  -8. -11.], loss 70.00, dw -36.0, db -16.0
after one step with lr 0.05: w = 1.80, b = 0.80, loss 13.92
```

## Try it

1. Rerun with `lr = 0.027` and `lr = 0.028`. Does the boundary computed above show up in the loss at step 100?
2. Standardise x before fitting: `x = (x - x.mean()) / x.std()`. Which learning rate now works, how many steps does 0.05 need, and what do the new `w` and `b` mean?
3. Add a second noise level (`rng.normal(0, 6, 50)`) and compare the final loss with the noise variance. Is the relationship the same?
