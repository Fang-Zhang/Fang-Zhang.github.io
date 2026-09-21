---
title: "Learning Rate: The One Number That Controls How AI Learns"
date: 2026-08-09 09:00:00 +1200
categories: [AI/ML]
tags: [deep-learning, gradient-descent, pytorch, learning-rate]
---

Every time a neural network trains, it's really doing one repeated thing: guess, measure the error, and nudge itself to be less wrong. The learning rate — usually just written `lr` — is the dial that controls how big each nudge is. Get it right and the network converges smoothly. Get it wrong and it either creeps forward for days or blows up entirely.

## The one line that does all the work

```python
x = x - lr * x.grad
```

That's the entire update rule for a single weight. Three pieces:

- **`x`** — the weight (a parameter the model is trying to tune)
- **`x.grad`** — the gradient: how much the loss changes if you nudge `x`, and in which direction
- **`lr`** — the learning rate: how far you actually move, scaled down from that gradient

In plain English: *move the weight a little in the opposite direction of the gradient, scaled by lr.*

## A concrete walk-through

```python
x = torch.tensor(3.0, requires_grad=True)  # weight = 3.0, loss = x^2 = 9.0
loss = x ** 2

loss.backward()       # gradient = 2x = 6.0
print(x.grad)         # tensor(6.0)

lr = 0.1
# Optimizer does:
# x = 3.0 - 0.1 * 6.0 = 2.4   (new position)
# loss = 2.4^2 = 5.76         (lower! less error)
# repeat until x -> 0, loss -> 0
```

Each step, the weight edges closer to the value that minimizes loss.

## The hill analogy

Picture yourself blindfolded on a hilly landscape, trying to reach the lowest valley — that valley is zero loss:

- **Tensor** — the ground you're standing on (the data)
- **Loss** — your current altitude (how wrong you are)
- **Gradient** — the slope under your feet (which way is downhill, and how steep)
- **Learning rate** — how big a stride you take
- **Optimizer** — the process of actually taking that stride, over and over

## Why not just take giant steps?

If you're blindfolded on a hill, huge steps feel efficient — until you stride right over the valley and up the other side.

```python
# Too large: loss bounces and may explode
optimizer = optim.Adam(model.parameters(), lr=1.0)

# Too small: loss barely moves
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# Just right: a sane default
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

- **Too large** — you overshoot the valley, loss oscillates or diverges
- **Too small** — you inch forward so slowly training takes forever
- **Just right** — steady, efficient progress toward the minimum

Typical values in practice: `0.01`, `0.001`, `0.0001`, `0.00001` — smaller is safer but slower; most training starts around `0.001` and adjusts from there.

## Where it lives in real code

You set `lr` once, when you create the optimizer — and it governs every single weight update for the rest of training:

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

for X_batch, y_batch in loader:
    pred = model(X_batch)            # forward pass
    loss = criterion(pred, y_batch)  # how wrong are we (the hill height)
    optimizer.zero_grad()            # clear old gradients
    loss.backward()                  # compute gradients (the slope)
    optimizer.step()                 # apply: weight = weight - lr * gradient
```

## How it all connects

```
Tensors (data) --> Model --> Loss (how wrong)
                                  |
                             loss.backward()
                                  |
                        Gradients computed for every weight
                                  |
                       optimizer.step()
                       weight = weight - lr * gradient
                                  |
                            Weights updated
                                  |
                       Loop again -> loss gets smaller
```

**The one-liner:** Tensors carry the data. The model transforms it. Loss measures the error. Gradients point uphill. The learning rate decides the step size. The optimizer walks downhill. Repeat until done.
