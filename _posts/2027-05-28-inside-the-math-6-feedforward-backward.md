---
title: "Inside the Math, Part 6: Backpropagation Through the Feed-Forward Block"
date: 2027-05-28 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, backpropagation, gradient, feed-forward network, gelu, transformer, inside the math]
description: "Part 6 of an 8-part series on the classic maths inside language models. The backward pass of a feed-forward block is five lines of matrix products, each with the shape of the thing it differentiates. Written by hand and matched to autograd to 2.1e-14 on 256 real vectors from a trained transformer; the backward pass costs twice the forward pass."
---

## What you will do, and what you get

Part 5 wrote the forward pass of the feed-forward block. Here you write its backward pass by hand: given the gradient `dY` on the output, produce the gradients of both weight matrices, both biases and the input. The test bed is the last block of a trained 110,384-parameter character transformer, in float64, with 256 real vectors and a random upstream gradient.

- **Five lines of matrix products are enough.** The hand-written gradients differ from PyTorch autograd by at most 2.1e-14 (`dW1`), 1.1e-14 (`db1`), 2.7e-15 (`dX`) and 0 (`dW2`, `db2`, which use the same operations as autograd).
- **Shapes give the formulas.** Each gradient has the shape of the thing it is the gradient of, and that fixes which matrices to multiply.
- **The nonlinearity only multiplies.** The gradient passes through GELU by element-wise multiplication with its derivative, which is between about -0.13 and 1.13 (it exceeds 1 for moderate positive inputs).
- **No unit is dead in this model.** Every one of the 256 hidden units receives gradient: row norms of `dW1` run from 64.2 to 200.7 with a median of 115.0. 34.4 percent of (vector, unit) pairs have a GELU derivative below 0.05 in size.
- **Backward costs twice forward:** 65,536 against 32,768 multiply-adds per vector.

![Top: the forward chain X, U, H, Y with their shapes, and the backward chain from dY to dW2, db2, dH, dU, dW1, db1 and dX. Bottom left: histogram of the GELU derivative over all (vector, unit) pairs of the last block. Bottom right: the norm of each row of dW1, sorted, showing that every one of the 256 hidden units receives gradient]({{ site.baseurl }}/assets/pic/mm-6-feedforward-backward.png){: width="900" }

## How: the steps

1. **Store what the forward pass computed:** the input `X` (one row per vector), `U = X W1^T + b1`, `H = GELU(U)`, `Y = H W2^T + b2`.
2. **Receive `dY`,** the gradient of the loss with respect to `Y`, same shape as `Y`.
3. **Second layer:** `dW2 = dY^T H`, `db2 = sum of the rows of dY`, `dH = dY W2`.
4. **Through the nonlinearity:** `dU = dH * GELU'(U)`, an element-wise product.
5. **First layer:** `dW1 = dU^T X`, `db1 = sum of the rows of dU`, `dX = dU W1`.
6. **Hand `dX` to the layer below** (the LayerNorm in front, whose backward pass is Part 4).
7. **Check each result against autograd** and against the hand-count of one single weight.

## Why it works: the principle

**One rule, applied twice.** For `Y = H W^T` with a row per vector, the chain rule gives `dH = dY W` and `dW = dY^T H`: the gradient on the weights is the sum, over vectors, of the outer product of the gradient on the output and the input that produced it. Item 3 of the measurements writes this out for one weight: `dW2[3, 10]` is the sum over the 256 vectors of `dY[n, 3] * H[n, 10]`, 21.305881 by the loop and 21.305881 by the matrix product.

**Shapes as a guide.** The [CS231n notes on backpropagation](https://cs231n.github.io/optimization-2/) point out that you do not need to remember the expressions for `dW` and `dX`, because they are easy to re-derive from dimensions: the gradient on the weights must be the same size as the weights. Here `dW1` must be 256 x 64, and the only product of `dU` (256 x 256) and `X` (256 x 64) with that shape is `dU^T X`.

**Through GELU.** Because GELU acts on each number separately, its Jacobian is diagonal and the backward step is an element-wise product with the derivative `GELU'(u) = Phi(u) + u * phi(u)`, where `Phi` and `phi` are the Gaussian cumulative distribution and density. Very negative inputs get a derivative near 0 (the unit blocks the gradient), large positive inputs near 1, and moderate positive ones slightly above 1 (1.083 at `u = 1`), which the histogram shows as a small bump above 1. ReLU would give exactly 0 or 1. The chain rule over the whole network is the subject of the [backpropagation chapter of the Deep Learning book](https://www.deeplearningbook.org/contents/mlp.html), which presents it as a computation on a graph.

**Why backward costs twice forward.** Forward has two matrix products. Backward has, for each matrix, one product for the weight gradient and one for the input gradient: four products of the same size, hence twice the work. This is a count of multiply-adds, not a timing: real speed also depends on memory traffic.

**Pitfalls and limits.**
- **Transposes.** Mixing up `dY^T H` and `H^T dY` produces a matrix of the wrong shape, so shape errors show up immediately; a square matrix would hide the mistake, so test with unequal sizes (here 64 and 256).
- **Sum the bias gradient over vectors.** `db` is the sum of the rows of `dY`, not one row.
- **Use the stored `U`, not `H`,** inside `GELU'`.
- **Dead units are possible, not present.** A unit whose pre-activation stays in the flat negative region for every input gets almost no gradient. In this trained model no unit has a derivative below 0.05 for more than 99 percent of the 256 vectors (0 of 256), and the gradient norms span only a factor of 3. Models trained with other settings may differ.
- **A random `dY` is a probe,** not a real loss gradient: the norms printed are specific to this draw, while the agreement with autograd is not.
- **Memory.** The backward pass needs `X`, `U` and `H` from the forward pass. Keeping them is a main reason training needs far more memory than running the model.

## Example, worked through

The block of Part 5 again: `x = [1, 2]`, three hidden units, `dy = [1, 0]`.

- Forward: `u = [1, 2, 3.5]`, `h = [0.841, 1.954, 3.499]`, `GELU'(u) = [1.083, 1.085, 1.003]`.
- `dh = W2^T dy = [1, 0, 1]` (only the units whose value touches the first output).
- `du = dh * GELU'(u) = [1.083, 0, 1.003]`. The middle unit receives no gradient because its value writes only to the second output, which has zero gradient.
- `dW2 = outer(dy, h)`: only the first row is nonzero, `[0.841, 1.954, 3.499]`.
- `dW1 = outer(du, x)`: rows `[1.083, 2.167]`, `[0, 0]`, `[1.003, 2.006]`.
- `dx = W1^T du = [2.086, 1.003]`.
- Autograd agrees exactly.

```python
import torch, math
torch.set_default_dtype(torch.float64)
gelu = lambda u: 0.5 * u * (1 + torch.erf(u / math.sqrt(2)))
dgelu = lambda u: 0.5 * (1 + torch.erf(u / math.sqrt(2))) + u * torch.exp(-u ** 2 / 2) / math.sqrt(2 * math.pi)
x = torch.tensor([1., 2.]); W1 = torch.tensor([[1., 0.], [0., 1.], [1., 1.]]); b1 = torch.tensor([0., 0., 0.5])
W2 = torch.tensor([[1., 0., 1.], [0., 1., -1.]]); dy = torch.tensor([1., 0.])        # only the first output matters
u = W1 @ x + b1; h = gelu(u)
dW2 = torch.outer(dy, h); dh = W2.T @ dy; du = dh * dgelu(u); dW1 = torch.outer(du, x); dx = W1.T @ du
print("u =", u.numpy(), " h =", h.numpy().round(3), " GELU'(u) =", dgelu(u).numpy().round(3))
print("dh = W2^T dy =", dh.numpy(), " du = dh * GELU'(u) =", du.numpy().round(3))
print("dW2 =\n", dW2.numpy().round(3)); print("dW1 =\n", dW1.numpy().round(3)); print("dx = W1^T du =", dx.numpy().round(3))
a, c = W1.clone().requires_grad_(), W2.clone().requires_grad_(); xx = x.clone().requires_grad_()
(c @ gelu(a @ xx + b1) @ dy).backward()
print("autograd: dW1 diff", (a.grad - dW1).abs().max().item(), " dW2 diff", (c.grad - dW2).abs().max().item(), " dx diff", (xx.grad - dx).abs().max().item())
```

Output:

```
u = [1.  2.  3.5]  h = [0.841 1.954 3.499]  GELU'(u) = [1.083 1.085 1.003]
dh = W2^T dy = [1. 0. 1.]  du = dh * GELU'(u) = [1.083 0.    1.003]
dW2 =
 [[0.841 1.954 3.499]
 [0.    0.    0.   ]]
dW1 =
 [[1.083 2.167]
 [0.    0.   ]
 [1.003 2.006]]
dx = W1^T du = [2.086 1.003]
autograd: dW1 diff 0.0  dW2 diff 0.0  dx diff 0.0
```

## Implementation

`part6.py` loads the model trained by `part8.py` (see Part 1 of this series; about 4 minutes on a CPU), captures 256 vectors entering the last block's MLP, runs the hand-written backward pass and compares it to autograd. Keep `part8.py`, `tiny8.pt` and the script in one folder.

```python
import json, math, numpy as np, torch, torch.nn.functional as F
from part8 import Tiny, chars, windows, test, T, D, V
torch.set_default_dtype(torch.float64)
m = Tiny().double(); m.load_state_dict({k: v.double() for k, v in torch.load("tiny8.pt").items()}); m.eval()
blk = m.blocks[1]; mlp = blk.mlp
xb, yb = windows(test, list(range(0, 64 * 4, 64)))                          # 4 windows x 64 positions = 256 vectors
store = {}; blk.ln2.register_forward_hook(lambda mod, i, o: store.update(x=o.detach()))
with torch.no_grad(): m(xb)
X = store["x"].reshape(-1, D)                                               # the 256 vectors entering the MLP of the last block
W1, b1, W2, b2 = (mlp[0].weight.detach(), mlp[0].bias.detach(), mlp[2].weight.detach(), mlp[2].bias.detach())
gelu = lambda u: 0.5 * u * (1 + torch.erf(u / math.sqrt(2)))
dgelu = lambda u: 0.5 * (1 + torch.erf(u / math.sqrt(2))) + u * torch.exp(-u ** 2 / 2) / math.sqrt(2 * math.pi)       # derivative of GELU

U = X @ W1.T + b1; Hh = gelu(U); Y = Hh @ W2.T + b2                          # forward, all 256 vectors at once
dY = torch.randn_like(Y)                                                    # an arbitrary upstream gradient, one row per vector
print(f"forward shapes: X {tuple(X.shape)}, U/H {tuple(U.shape)}, Y {tuple(Y.shape)}")

dW2 = dY.T @ Hh; db2 = dY.sum(0)                                            # gradients of the second layer's parameters
dH = dY @ W2                                                                # gradient flowing back into the hidden layer
dU = dH * dgelu(U)                                                          # through the nonlinearity: elementwise multiplication
dW1 = dU.T @ X; db1 = dU.sum(0); dX = dU @ W1                               # first layer's parameters and the gradient passed on to the layer below

p = [W1.clone().requires_grad_(), b1.clone().requires_grad_(), W2.clone().requires_grad_(), b2.clone().requires_grad_()]; Xg = X.clone().requires_grad_()
out = gelu(Xg @ p[0].T + p[1]) @ p[2].T + p[3]; (out * dY).sum().backward()
print("\n1. hand-written backward vs PyTorch autograd (largest absolute difference)")
for name, mine, theirs in (("dW2", dW2, p[2].grad), ("db2", db2, p[3].grad), ("dW1", dW1, p[0].grad), ("db1", db1, p[1].grad), ("dX ", dX, Xg.grad)):
    print(f"   {name}: shape {tuple(mine.shape)}, difference {(mine - theirs).abs().max().item():.1e}")

print("\n2. shapes tell you the formula: each gradient has the shape of the thing it is the gradient of")
print("   dW2 (64 x 256) = dY^T (64 x 256 vectors) @ H (256 vectors x 256)   -> sum over vectors of outer products dY_n (x) H_n")
print("   dW1 (256 x 64) = dU^T @ X")

print("\n3. gradient of ONE weight by hand: dW2[3, 10] = sum over the 256 vectors of dY[n, 3] * H[n, 10]")
print(f"   {(dY[:, 3] * Hh[:, 10]).sum().item():.6f} vs matrix product {dW2[3, 10].item():.6f}")

print("\n4. dead and sleepy units: a unit whose GELU input is negative for every vector gets (almost) no gradient")
g = dgelu(U); frac_small = (g.abs() < 0.05).double().mean(0)
dead = (frac_small > 0.99).sum().item()
print(f"   GELU derivative below 0.05 in size for {100 * (g.abs() < 0.05).double().mean().item():.1f}% of (vector, unit) pairs; units with this for more than 99% of the vectors: {dead} of 256")
print(f"   gradient norm per row of W1 (one hidden unit each): smallest {dW1.norm(dim=1).min().item():.3f}, median {dW1.norm(dim=1).median().item():.3f}, largest {dW1.norm(dim=1).max().item():.3f}")

print("\n5. cost: multiply-adds for forward vs backward of this block, per vector")
fwd = 2 * D * 4 * D; bwd = 2 * fwd
print(f"   forward {fwd:,} (two matrix products); backward {bwd:,} (for each matrix: one product for dW and one for dX) -> backward is {bwd // fwd}x the forward cost")
json.dump({"dW2_row": dW2[3].tolist(), "grad_norm_rows": dW1.norm(dim=1).tolist(), "frac_small": frac_small.tolist(), "gprime_hist": np.histogram(g.flatten().numpy(), bins=40, range=(-0.2, 1.2))[0].tolist()}, open("results6.json", "w"))
```

Output:

```
forward shapes: X (256, 64), U/H (256, 256), Y (256, 64)

1. hand-written backward vs PyTorch autograd (largest absolute difference)
   dW2: shape (64, 256), difference 0.0e+00
   db2: shape (64,), difference 0.0e+00
   dW1: shape (256, 64), difference 2.1e-14
   db1: shape (256,), difference 1.1e-14
   dX : shape (256, 64), difference 2.7e-15

2. shapes tell you the formula: each gradient has the shape of the thing it is the gradient of
   dW2 (64 x 256) = dY^T (64 x 256 vectors) @ H (256 vectors x 256)   -> sum over vectors of outer products dY_n (x) H_n
   dW1 (256 x 64) = dU^T @ X

3. gradient of ONE weight by hand: dW2[3, 10] = sum over the 256 vectors of dY[n, 3] * H[n, 10]
   21.305881 vs matrix product 21.305881

4. dead and sleepy units: a unit whose GELU input is negative for every vector gets (almost) no gradient
   GELU derivative below 0.05 in size for 34.4% of (vector, unit) pairs; units with this for more than 99% of the vectors: 0 of 256
   gradient norm per row of W1 (one hidden unit each): smallest 64.172, median 115.005, largest 200.711

5. cost: multiply-adds for forward vs backward of this block, per vector
   forward 32,768 (two matrix products); backward 65,536 (for each matrix: one product for dW and one for dX) -> backward is 2x the forward cost
```

## Try it

1. Replace GELU by ReLU in forward and backward and check that the hand-written gradient still matches autograd.
2. Stack two such blocks and write the backward pass through both, passing `dX` of the second as `dY` of the first.
3. Train the model with the learning rate 100 times larger and count again how many units have a GELU derivative below 0.05 on more than 99 percent of the vectors.
