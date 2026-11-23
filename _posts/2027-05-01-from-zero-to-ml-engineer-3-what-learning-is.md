---
title: "From Zero to ML Engineer, Part 3: What Learning Is"
date: 2027-05-01 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, gradient descent, linear regression, numpy, ML engineer]
description: "Part 3 of a 12-part series for complete beginners. A model is a function with adjustable numbers; learning is adjusting them to make a loss smaller. A line is fitted from scratch in 20 lines of numpy with gradient descent, including the failure that happens when the steps are too large."
---

## What you will do, and what you get

You will write a learning algorithm yourself, from nothing but numpy, and watch it find a rule from examples. The data is 50 delivery jobs: distance in kilometres and delivery time in minutes (generated, so that the true rule is known: 3 minutes per kilometre plus 2, with noise). The program starts knowing nothing, and ends with `w = 3.105, b = 1.515`, exactly the answer the textbook formula gives. Every model in this series, up to neural networks with millions of numbers, is trained by the same loop.

![Three panels. Left: 50 scattered points and a fitted line that moves from flat at zero toward the data over steps 0, 10, 60 and 1999. Middle: loss against step for learning rates 0.001, 0.02 and 0.05, the last exploding upward. Right: a contour map of the loss over slope and intercept, a narrow valley with the training path going downhill to the minimum]({{ site.baseurl }}/assets/pic/mlpath-3-learning.png){: width="900" }

## How: the steps

1. **Choose a model.** Here: `prediction = w * x + b`, a line. The numbers `w` (slope) and `b` (intercept) are the *parameters*. They are what gets learned.
2. **Choose a loss.** A single number that says how wrong the model is. Here: the *mean squared error*, the average of (prediction - answer)^2 over all rows. Zero would be perfect.
3. **Start anywhere.** Set both parameters to 0.
4. **Measure the slope of the loss.** For each parameter, work out how much the loss goes up or down if that parameter increases a little. This is the *gradient*.
5. **Step downhill.** Subtract `learning_rate x gradient` from each parameter.
6. **Repeat** steps 4 and 5 until the loss stops falling.

## Why it works: the principle

Think of the loss as a landscape. Every pair (w, b) is a location, and the loss is its height; the right graph in the figure draws it as a map, with yellow for low ground. Learning is walking downhill. You cannot see the whole landscape, but at your current location you can measure the slope, and the slope tells you which way is down. That is all a gradient is.

For the mean squared error, the slopes have a short closed form (they come from the derivative of the squared term):

- slope with respect to `w`: `2 x mean(error x x)`
- slope with respect to `b`: `2 x mean(error)`

where `error = prediction - answer`. A worked step with three rows, x = (1, 2, 3) and answers (3, 5, 7), starting at w = b = 0, learning rate 0.05:

```
errors = (-3, -5, -7)    loss = (9 + 25 + 49) / 3 = 27.67
slope for w = 2 x mean(-3, -10, -21) = -22.67      slope for b = 2 x mean(-3, -5, -7) = -10
new w = 0 - 0.05 x (-22.67) = 1.133      new b = 0 - 0.05 x (-10) = 0.5
loss after the step: 5.49
```

The loss fell from 27.67 to 5.49 in one step. The signs matter: both slopes are negative (raising w or b lowers the loss), so subtracting a negative number raises the parameters.

**The learning rate** is how big a step to take. Too small and learning is slow; too large and you overshoot the valley, land higher on the other side, and the next step is bigger still. The middle panel shows all three cases on the same data: 0.001 is still creeping after 60 steps, 0.02 converges in about 15, and 0.05 makes the loss explode (it reaches about 10^174 after 200 steps). When a loss goes up or turns into `inf` or `nan`, the first thing to try is a smaller learning rate.

**Why does the path in the right panel bend?** The valley is long and narrow: `x` is always positive and runs from 0 to 10 while `b` is multiplied by the constant 1, so raising `w` and lowering `b` almost cancel, and the loss changes slowly along one diagonal and quickly across it. Narrow valleys make gradient descent slow. Rescaling columns, which Part 5 does, makes the valley rounder and the walk faster.

**Why not just use the formula?** For a line there is one, and the program compares against it. For almost every other model there is not, and gradient descent is how they are trained. The scikit-learn documentation describes the [same method at scale](https://scikit-learn.org/stable/modules/sgd.html) for linear models, and NumPy's [polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) is the closed-form reference used below.

## Example, worked through

Reading the output of the program:

- **Learning rate 0.001, 200 steps:** `w = 3.243, b = 0.567`, loss 4.18. Still on its way: `b` should end near 1.5.
- **Learning rate 0.02, 200 steps:** `w = 3.129, b = 1.351`, loss 3.98. Nearly there.
- **Learning rate 0.05:** parameters of order 10^86 and a loss of 10^174. Diverged.
- **2000 steps at 0.02:** `w = 3.105, b = 1.515`, identical to `np.polyfit` to three decimals.
- **The loss floor:** the fitted loss is 3.97, close to 4.00, the variance of the noise added when the data was generated. A perfect model of the rule cannot beat the noise it cannot see, so a loss near the noise level means the model has learned everything there is to learn. The baseline of always predicting the mean has a loss of 85.85, so the model removes about 95% of the error.
- **The recovered rule:** the true rule was `3x + 2`; the learned one is `3.10x + 1.52`. The gap comes from having only 50 noisy rows, not from a flaw in the method.

## Implementation

Save as `part3.py` and run it. It needs only numpy.

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 50)                          # e.g. distance in km
y = 3.0 * x + 2.0 + rng.normal(0, 2.0, 50)          # e.g. delivery minutes: a true rule plus noise the model cannot know

def train(x, y, lr, steps):
    w, b = 0.0, 0.0                                 # the two parameters, starting from nothing
    history = []
    for step in range(steps):
        pred = w * x + b
        error = pred - y
        loss = (error ** 2).mean()                  # mean squared error: the number we want to make small
        grad_w = 2 * (error * x).mean()             # how the loss changes if w goes up a little
        grad_b = 2 * error.mean()                   # ... and if b goes up a little
        w -= lr * grad_w                            # step downhill
        b -= lr * grad_b
        history.append((w, b, loss))
    return w, b, history

np.seterr(all="ignore")                              # the diverging run overflows on purpose; hide the warnings
for lr in (0.001, 0.02, 0.05):
    w, b, h = train(x, y, lr, 200)
    print(f"learning rate {lr:5.3f}: after 200 steps w={w:.4g} b={b:.4g} loss={h[-1][2]:.4g}")

w, b, h = train(x, y, 0.02, 2000)
w_exact, b_exact = np.polyfit(x, y, 1)              # the textbook closed-form answer, for comparison
print(f"gradient descent, 2000 steps: w={w:.3f} b={b:.3f}")
print(f"closed form (np.polyfit):     w={w_exact:.3f} b={b_exact:.3f}")
print(f"baseline loss (always predict the mean): {y.var():.2f}   fitted loss: {h[-1][2]:.2f}   (noise variance is 4.00)")
```

Output:

```
learning rate 0.001: after 200 steps w=3.243 b=0.5674 loss=4.184
learning rate 0.020: after 200 steps w=3.129 b=1.351 loss=3.98
learning rate 0.050: after 200 steps w=-7.112e+86 b=-1.04e+86 loss=2.619e+174
gradient descent, 2000 steps: w=3.105 b=1.515
closed form (np.polyfit):     w=3.105 b=1.515
baseline loss (always predict the mean): 85.85   fitted loss: 3.97   (noise variance is 4.00)
```

## Try it

1. Change the noise (`rng.normal(0, 2.0, 50)`) to 5.0. Does the fitted slope move? Does the loss floor move to 25?
2. Find the largest learning rate between 0.02 and 0.05 that still converges. How does it change if you divide `x` by 10 first?
3. Add a second input column and a second weight, and extend the update rule. (The pattern is identical: one slope per parameter.)
