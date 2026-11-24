---
title: "Inside the Math, Part 4: The Gradient of LayerNorm Is a Projection"
date: 2027-05-26 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, layer normalization, backpropagation, gradient, jacobian, transformer, inside the math]
description: "Part 4 of an 8-part series on the classic maths inside language models. Backpropagating through LayerNorm gives a closed form: the incoming gradient, with its mean and its component along the normalised vector removed, divided by the vector's std. Derived once, verified against autograd to 1.3e-15 on 512 real vectors from a trained transformer."
---

## What you will do, and what you get

Part 3 wrote LayerNorm's forward pass. Here you derive its backward pass, the formula that turns the gradient on the output into the gradient on the input, and check it on real data. Four results, in float64, on 512 vectors from a trained 110,384-parameter character transformer:

- **The closed form matches PyTorch autograd** to 1.3e-15 (largest difference over all 512 x 64 entries).
- **The Jacobian (the 64 x 64 matrix of all output-versus-input derivatives) has two blocked directions.** 62 of its 64 singular values are of order 1; the other two are 4.2e-06 and 9.0e-18. The blocked directions are "add the same number to every feature" and "stretch the vector along its own direction", because LayerNorm undoes both.
- **The gradient on the input always sums to zero over the features** (5.3e-15 in the worst case), which is the same fact seen from the backward side.
- **The gradient is divided by the vector's std.** A vector three times larger gets a gradient 0.333 times as large, and across the 512 real vectors the correlation of log(std) with log(gradient ratio) is -0.93.

![Left: gradient on the 64 features of one vector, arriving from above (dy, grey) and sent below (dx, red): similar shapes, with sum of dy equal to -6.09 and sum of dx equal to 0. Middle: the 64 singular values of the Jacobian on a log scale: 62 near 1, one at 4e-6 and one at 9e-18. Right: log-log scatter of the ratio of gradient norms against the std of the input vector for 512 real vectors, falling along the line 1 over std]({{ site.baseurl }}/assets/pic/mm-4-layernorm-backward.png){: width="900" }

## How: the steps

1. **Write the forward pass as a chain** (gain `g`, bias `b`, per vector of `n = 64` features): `mu = mean(x)`, `var = mean((x - mu)^2)`, `r = 1 / sqrt(var + eps)`, `xhat = (x - mu) * r`, `y = g * xhat + b`.
2. **Start from the gradient `dy` on the output** (whatever the layers above send) and pass it through the gain: `dxhat = g * dy`.
3. **Follow every path from `xhat` back to `x`.** `x` reaches `xhat` three ways: directly (`xhat = (x - mu) r`), through the mean `mu`, and through the scale `r`, which depends on all features through `var`.
4. **Add the three contributions:** `dx = r * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))`.
5. **Read it as a projection:** from `dxhat`, remove its average and its component along `xhat`, then multiply by `r`.
6. **Check three ways:** against autograd, against finite differences on one Jacobian column, and against the closed-form Jacobian `r * diag(g) * (I - 11^T / n - xhat xhat^T / n)`.
7. **Measure** how the gradient size depends on the input std.

## Why it works: the principle

**Three paths.** With `dxhat` known, the direct path contributes `r * dxhat`. Through the mean, each `x_j` moves `mu` by `1/n`, which moves every `xhat_i` by `-r`, so this path contributes `-r * mean(dxhat)` to every feature. Through the scale, a change of `x_j` changes `var` by `2 (x_j - mu) / n`, hence `r` by `-r^3 (x_j - mu) / n`, and then every `xhat_i = (x_i - mu) r` by `-r * xhat_i * xhat_j / n`; weighting with `dxhat_i` and summing over `i` gives `-r * xhat_j * mean(dxhat * xhat)` for feature `j`. The three sum to the formula in step 4.

**Why it is a projection.** `xhat` has mean 0 (so it is perpendicular to the all-ones vector) and `sum(xhat^2) = n` (when `eps` is ignored). The expression `dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)` therefore removes the component of `dxhat` along the all-ones direction and along the `xhat` direction. What is left is perpendicular to both. Two consequences, both measured: the result sums to zero, and the Jacobian has two (near-)zero singular values. The `4.2e-06` is not exactly zero because of `eps`, which makes the second removed direction very slightly imperfect.

**What the two blocked directions mean.** Adding the same constant to all features does not change the output, so no loss can depend on it: the gradient in that direction is zero. Multiplying the vector by a positive number does not change `xhat` either, so the gradient along the vector's own direction is zero too. These are the symmetries of Part 3 (re-centering and re-scaling invariance) seen from the gradient side. They are also visible in the paper that introduced LayerNorm: [Ba, Kiros and Hinton (2016)](https://arxiv.org/abs/1607.06450) analyse its invariance properties under rescaling and re-centering.

**The `1 / std` factor.** The overall factor `r = 1 / sqrt(var + eps)` multiplies the whole gradient. A larger input vector (larger std) gets a proportionally smaller gradient: the right panel shows the 1/std law on real vectors. In Part 3 the residual stream grew with depth, from std 1.29 to 3.11; the gradient that passes *through* the normalisation of a later layer is smaller by that factor, compared with the same gradient at an early layer. [Xiong and colleagues (2020)](https://arxiv.org/abs/2002.04745) analyse the gradients of the two placements of LayerNorm in a transformer at initialisation; this post measures only the factor of one LayerNorm, not a whole network.

**Pitfalls and limits.**
- **Do not forget the two correction terms.** `r * dxhat` alone is the gradient of "divide by a constant std" and is wrong; the autograd comparison catches it immediately.
- **`eps` breaks exactness.** With `eps` in the denominator the dot product `dx . xhat` is 1.6e-04 instead of exactly 0 here, since `sum(xhat^2)` is slightly below `n`. This is a property of the formula with `eps`, not of rounding error.
- **The gain's own gradients.** The parameters get `dg = sum over vectors of dy * xhat` and `db = sum over vectors of dy`. They are not part of the Jacobian of `x` and not computed above; they follow from `y = g * xhat + b`.
- **Float64.** The finite-difference check with `eps = 1e-6` agrees to 2.4e-10 in float64; in float32 the same check is far noisier.
- **One layer, one set of vectors.** The std range of the 512 vectors is 1.09 to 1.64; the slope of the right panel is a property of the formula, but the size of the spread is of this model.

## Example, worked through

The vector `v = [1, 2, 3, 4]`, gain 1, `eps` ignored, and an incoming gradient `dy = [1, 0, 0, 0]` (only the first output matters).

- Forward: `mu = 2.5`, `var = 1.25`, `r = 1 / 1.118 = 0.894`, `xhat = [-1.342, -0.447, 0.447, 1.342]`.
- `dxhat = dy`, so `mean(dxhat) = 0.25` and `mean(dxhat * xhat) = -1.342 / 4 = -0.3354`.
- `dx = 0.894 * ([1, 0, 0, 0] - 0.25 + 0.3354 * xhat) = 0.894 * [0.30, -0.40, -0.10, 0.20] = [0.268, -0.358, -0.089, 0.179]`.
- The entries sum to 0, and the dot product with `xhat` is `-0.360 + 0.160 - 0.040 + 0.240 = 0`.
- Autograd gives the same four numbers.

```python
import torch
torch.set_default_dtype(torch.float64)
v = torch.tensor([1., 2., 3., 4.], requires_grad=True); dy = torch.tensor([1., 0., 0., 0.]); eps = 0.0         # eps ignored to keep the arithmetic exact
mu = v.mean(); r = 1 / torch.sqrt(((v - mu) ** 2).mean() + eps); xhat = ((v - mu) * r).detach(); r = r.detach()
dxhat = dy                                                                  # gain 1
dx = r * (dxhat - dxhat.mean() - xhat * (dxhat * xhat).mean())
print("xhat =", xhat.numpy().round(3), " r =", round(r.item(), 3))
print("mean(dxhat) =", dxhat.mean().item(), " mean(dxhat * xhat) =", round((dxhat * xhat).mean().item(), 4))
print("dx by the formula   =", dx.numpy().round(3), " sum =", round(dx.sum().item(), 12), " dot with xhat =", round((dx * xhat).sum().item(), 12))
y = (v - v.mean()) / torch.sqrt(((v - v.mean()) ** 2).mean()); (y * dy).sum().backward()
print("dx by autograd      =", v.grad.numpy().round(3))
```

Output:

```
xhat = [-1.342 -0.447  0.447  1.342]  r = 0.894
mean(dxhat) = 0.25  mean(dxhat * xhat) = -0.3354
dx by the formula   = [ 0.268 -0.358 -0.089  0.179]  sum = -0.0  dot with xhat = 0.0
dx by autograd      = [ 0.268 -0.358 -0.089  0.179]
```

## Implementation

`part4.py` loads the model trained by `part8.py` (see Part 1 of this series; about 4 minutes on a CPU), captures 512 vectors entering the first LayerNorm, and runs the four checks. Keep `part8.py`, `tiny8.pt` and the script in one folder.

```python
import json, numpy as np, torch
from part8 import Tiny, chars, windows, test, D
torch.set_default_dtype(torch.float64)
m = Tiny().double(); m.load_state_dict({k: v.double() for k, v in torch.load("tiny8.pt").items()}); m.eval()
ln = m.blocks[0].ln1; g = ln.weight.detach(); b = ln.bias.detach(); eps = ln.eps
xb, yb = windows(test, list(range(0, 64 * 8, 64)))
store = []; h = ln.register_forward_hook(lambda mod, i, o: store.append(i[0].detach()))
with torch.no_grad(): m(xb)
X = store[0].reshape(-1, D)                                                # 512 vectors of 64 numbers entering the first LayerNorm
dy = torch.randn_like(X)                                                    # an arbitrary upstream gradient dL/dy for each vector

def forward(x):
    mu = x.mean(-1, keepdim=True); var = ((x - mu) ** 2).mean(-1, keepdim=True); r = 1 / torch.sqrt(var + eps); xhat = (x - mu) * r
    return xhat * g + b, xhat, r

def backward(dy, xhat, r):                                                  # the closed form: project out the mean and the xhat direction
    dxhat = dy * g
    return r * (dxhat - dxhat.mean(-1, keepdim=True) - xhat * (dxhat * xhat).mean(-1, keepdim=True))

y, xhat, r = forward(X); dx = backward(dy, xhat, r)
xa = X.clone().requires_grad_(True); (ln(xa) * dy).sum().backward()
print(f"1. closed form vs PyTorch autograd on {X.shape[0]} real vectors: largest difference {(dx - xa.grad).abs().max().item():.1e}")

print("\n2. two exact properties of the gradient that comes out")
print(f"   sum of dx over the 64 features of each vector: largest |sum| = {dx.sum(-1).abs().max().item():.1e}   (shifting all features together changes nothing)")
print(f"   dx . xhat for each vector: largest |dot| = {(dx * xhat).sum(-1).abs().max().item():.1e}   (growing the vector along its own direction changes nothing; only exact when eps is ignored)")

print("\n3. finite-difference check of one row of the Jacobian dy/dx")
i = 0; j = 5; e = 1e-6; d = torch.zeros(D); d[j] = e
fd = (forward(X[i] + d)[0] - forward(X[i] - d)[0]) / (2 * e)               # how every output changes when input feature j changes
Jfull = torch.autograd.functional.jacobian(lambda v: forward(v)[0], X[i])  # 64 x 64
print(f"   column {j}: finite differences vs autograd Jacobian, largest difference {(fd - Jfull[:, j]).abs().max().item():.1e}")
P = torch.eye(D) - torch.ones(D, D) / D                                      # projector that removes the mean
xh = xhat[i]; Jclosed = r[i] * (torch.diag(g) @ (P - torch.outer(xh, xh) / D))
print(f"   closed-form Jacobian r * diag(g) * (I - 11^T/n - xhat xhat^T/n) vs autograd: largest difference {(Jclosed - Jfull).abs().max().item():.1e}")
sv = torch.linalg.svdvals(Jfull / (g.abs().mean() * r[i]))
print(f"   singular values of the Jacobian (divided by its overall scale r * mean|g|): {(sv > 1e-4).sum().item()} of {D} are clearly non-zero (the two smallest are {sv[-2].item():.1e} and {sv[-1].item():.1e}); the missing directions are the mean direction and the xhat direction")

print("\n4. sensitivity: the Jacobian scales with 1/std of the input vector")
sd = X.std(-1, unbiased=False); gn = dx.norm(dim=-1) / dy.norm(dim=-1)
c = np.corrcoef(np.log(sd.numpy()), np.log(gn.numpy()))[0, 1]
print(f"   input std ranges {sd.min():.2f} to {sd.max():.2f}; correlation of log(std) with log(gradient norm ratio) = {c:.2f}")
big = X * 3; yb_, xhb, rb = forward(big)
print(f"   a vector three times larger gets a gradient {(backward(dy, xhb, rb).norm(dim=-1) / dx.norm(dim=-1)).mean().item():.3f} times as large (1/3 = 0.333)")
json.dump({"sd": sd.tolist(), "gn": gn.tolist(), "sv": sv.tolist(), "xh": xh.tolist(), "dxrow": dx[0].tolist(), "dyrow": dy[0].tolist()}, open("results4.json", "w"))
```

Output:

```
1. closed form vs PyTorch autograd on 512 real vectors: largest difference 1.3e-15

2. two exact properties of the gradient that comes out
   sum of dx over the 64 features of each vector: largest |sum| = 5.6e-15   (shifting all features together changes nothing)
   dx . xhat for each vector: largest |dot| = 1.3e-04   (growing the vector along its own direction changes nothing; only exact when eps is ignored)

3. finite-difference check of one row of the Jacobian dy/dx
   column 5: finite differences vs autograd Jacobian, largest difference 2.4e-10
   closed-form Jacobian r * diag(g) * (I - 11^T/n - xhat xhat^T/n) vs autograd: largest difference 2.2e-16
   singular values of the Jacobian (divided by its overall scale r * mean|g|): 62 of 64 are clearly non-zero (the two smallest are 4.2e-06 and 9.0e-18); the missing directions are the mean direction and the xhat direction

4. sensitivity: the Jacobian scales with 1/std of the input vector
   input std ranges 1.09 to 1.64; correlation of log(std) with log(gradient norm ratio) = -0.94
   a vector three times larger gets a gradient 0.333 times as large (1/3 = 0.333)
```

## Try it

1. Remove the `xhat` term from `backward` and measure the difference to autograd; then remove the mean term instead.
2. Add the gain and bias gradients `dg` and `db` and compare them with `ln.weight.grad` and `ln.bias.grad`.
3. Write the backward pass for RMSNorm from Part 3 (no mean) and check which of the two blocked directions remains.
