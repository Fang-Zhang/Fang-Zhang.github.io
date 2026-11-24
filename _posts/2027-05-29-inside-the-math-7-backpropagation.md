---
title: "Inside the Math, Part 7: Backpropagation Is the Chain Rule on a Graph, in 41 Lines"
date: 2027-05-29 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, backpropagation, automatic differentiation, autograd, chain rule, inside the math]
description: "Part 7 of an 8-part series on the classic maths inside language models. A 41-line automatic-differentiation engine in NumPy reproduces PyTorch's loss and every parameter gradient (largest relative difference 1.2e-15) through a real LayerNorm, feed-forward block, final LayerNorm, output layer and cross-entropy of a trained transformer."
---

## What you will do, and what you get

You will write an automatic-differentiation engine (a mini autograd: it records operations and replays them backward to compute gradients) in 41 lines of NumPy, and run it on a real piece of a trained 110,384-parameter character transformer: the last block's LayerNorm and feed-forward part, the final LayerNorm, the output layer and the cross-entropy loss, for 256 vectors in float64.

- **Same loss, same gradients as PyTorch.** The loss is 1.687224206127 in both (2.434 bits per character on this slice). All ten parameter gradients and the input gradient differ from PyTorch by at most 6.9e-17 in absolute size, at most 1.2e-15 relative to the largest gradient entry (a few units of float64 rounding).
- **The graph has 43 nodes and 30 operations.** LayerNorm is built from small primitives (mean, subtract, multiply, power), so Part 4's closed form is not needed: the chain rule through the pieces gives the same answer.
- **Nothing builds a Jacobian.** The Jacobian of GELU on the 256 x 256 hidden tensor would have 65,536 x 65,536 = 4.29e9 entries (34 GB in float64). The backward step is instead one element-wise product of 65,536 numbers.
- **Backward needs the forward values.** The intermediate values in this slice take 3.8 MB for 256 vectors, 14.8 kB per vector.

![Left: a computation graph for f = (x*y + x)^2 at x = 2, y = 3 with forward values in blue and gradients in red: x has gradient 64 because it is used twice, y has 32, a and b have 16. Right: bar chart on a log scale of the largest difference to PyTorch, relative to the largest gradient entry, for each of the ten parameter gradients: all near 1e-15, the float64 rounding level]({{ site.baseurl }}/assets/pic/mm-7-backprop.png){: width="900" }

## How: the steps

1. **Wrap every value in a node** that stores its data, a gradient buffer (initially zeros), the nodes it was computed from (its parents), and a small function for the backward step.
2. **Run the forward computation normally.** Each operation (`+`, `-`, `*`, `@`, power, mean, GELU, cross-entropy) creates a new node and thereby records the graph.
3. **Seed the loss:** its gradient with respect to itself is 1.
4. **Order the graph:** a depth-first walk from the loss lists every node after all of its parents; reversing that list puts every node before its parents.
5. **Walk backward.** For each node, call its backward function with the node's finished gradient; it multiplies by the operation's local derivative and **adds** the result into each parent's gradient buffer.
6. **Undo broadcasting.** If a parameter of shape (64,) was added to a (256, 64) tensor, its gradient is the sum over the 256 rows.
7. **Compare** every gradient to PyTorch's.

| operation | backward step, with `g` the gradient arriving from above |
|---|---|
| `a + b` | `g` goes to both |
| `a * b` | `g * b` to `a`, `g * a` to `b` |
| `A @ B` | `g @ B^T` to `A`, `A^T @ g` to `B` (Part 6) |
| `a ** p` | `g * p * a^(p-1)` |
| `mean` | `g / n` to every entry |
| GELU | `g * GELU'(a)` (Part 6) |
| softmax + cross-entropy | `(p - y) / N` (Part 2) |

## Why it works: the principle

**Chain rule.** If `f` is a composition of simple steps, its derivative is the product of the steps' derivatives. For a function with many inputs and many steps, that product is a chain of Jacobian matrices. Backpropagation evaluates it from the output end: it starts with a row vector (the gradient of the loss, a scalar, with respect to the last value) and multiplies it by one Jacobian at a time, a *vector-Jacobian product*. Every step in the table is such a product, written without ever forming the matrix. The [back-propagation chapter of the Deep Learning book (section 6.5)](https://www.deeplearningbook.org/contents/mlp.html) formalises exactly this with computational graphs, in which each node is a variable.

**Why from the output end.** The loss is a single number and the parameters are many. Going backward gives the gradient with respect to *all* of them in one pass, whose cost is a small multiple of the forward pass (Part 6: twice, for matrix products). Going forward would need one pass per parameter, here 110,384 of them. [Baydin and colleagues (2015)](https://arxiv.org/abs/1502.05767) survey automatic differentiation as a family of techniques that is more general than backpropagation; backpropagation is its reverse mode applied to neural networks.

**Fan-out means addition.** In the left panel `x` feeds two nodes (`a = x*y` and `b = a + x`), so its gradient is the sum of two contributions: `16 * y = 48` through `a` and `16` directly, 64 in total. This is why step 5 uses `+=`. In the transformer the same happens at every residual connection: the residual stream feeds both the sublayer and the addition after it, and the two gradients add.

**Why the graph checks out on a real model.** LayerNorm appears here in its primitive form (`(x - mean) * (var + eps)^(-1/2) * g + b`), with `var` itself made of a mean of squares. Part 4 derived the closed form for this function; the engine never needs it, because the chain rule through the primitives yields the same numbers, and they match PyTorch's own LayerNorm gradient.

**What is stored.** Several backward steps need forward values: multiplication needs both factors, GELU needs its input, cross-entropy needs the probabilities. The 3.8 MB are those values for this slice only: one feed-forward part and the head for 256 vectors. A whole network keeps them for every layer, which is why training uses far more memory than generation (Part 8 puts numbers on it).

**Pitfalls and limits.**
- **Assigning instead of adding.** Writing `parent.grad = ...` instead of `+=` silently breaks every node with more than one consumer. The two-path example above catches it.
- **Order matters.** A node's backward function may only run once all of its consumers have added their contributions, which the reversed depth-first order guarantees.
- **Broadcasting.** A forward step that copies a value (a bias added to every row) needs a backward step that sums.
- **In-place edits.** Overwriting a stored value after the forward pass corrupts the backward pass; real frameworks track versions to catch it.
- **This engine is a teaching tool.** It runs on the CPU in NumPy, supports only the operations above, takes the mean only over the last axis, and uses `np.vectorize` for the error function, which is slow. The agreement with PyTorch shows that the logic is right, not that it is fast.
- **The test covers one slice** of the model (the last feed-forward part and the head), not the attention layers or the embeddings, whose operations (softmax over rows, indexing) would need new entries in the table.

## Example, worked through

The scalar function `f = (x * y + x)^2` at `x = 2`, `y = 3`, as in the figure.

- Forward: `a = x*y = 6`, `b = a + x = 8`, `f = b*b = 64`.
- Seed `df/df = 1`. The node `f = b*b` uses `b` twice, so `b` receives `1 * b + 1 * b = 16`.
- `b = a + x`: the gradient 16 goes to both `a` and `x`.
- `a = x*y`: `x` receives `16 * y = 48` (so `x` now holds 16 + 48 = 64) and `y` receives `16 * x = 32`.
- By hand: `f = (xy + x)^2`, so `df/dx = 2(xy + x)(y + 1) = 2 * 8 * 4 = 64` and `df/dy = 2(xy + x) * x = 32`. The engine agrees.

```python
from miniad import Tensor
x, y = Tensor(2.0), Tensor(3.0)
a = x * y                      # a = 6
b = a + x                      # b = 8   (x is used twice: its gradient has two contributions)
f = b * b                      # f = 64
order = f.backward()
names = {id(x): "x", id(y): "y", id(a): "a = x*y", id(b): "b = a+x", id(f): "f = b*b"}
for t in order: print(f"{names[id(t)]:8s} value {float(t.data):5.1f}   df/d(this) = {float(t.grad):5.1f}")
print("by hand: df/db = 2b = 16; df/da = 16; df/dy = 16*x = 32; df/dx = 16*y (via a) + 16 (direct) = 64")
```

Output:

```
x        value   2.0   df/d(this) =  64.0
y        value   3.0   df/d(this) =  32.0
a = x*y  value   6.0   df/d(this) =  16.0
b = a+x  value   8.0   df/d(this) =  16.0
f = b*b  value  64.0   df/d(this) =   1.0
by hand: df/db = 2b = 16; df/da = 16; df/dy = 16*x = 32; df/dx = 16*y (via a) + 16 (direct) = 64
```

## Implementation

Three files in one folder with `part8.py` and `tiny8.pt` from Part 1 of this series: the engine `miniad.py` (41 lines), the experiment `part7.py`, and the example above. Everything runs on a CPU.

`miniad.py`:

```python
import math, numpy as np
from math import erf
verf = np.vectorize(erf)

class Tensor:
    def __init__(s, data, parents=(), back=None):
        s.data = np.asarray(data, dtype=np.float64); s.grad = np.zeros_like(s.data); s.parents = parents; s.back = back   # back(g): pushes g into the parents
    def __add__(a, b): b = lift(b); return Tensor(a.data + b.data, (a, b), lambda g: (a.acc(g), b.acc(g)))
    def __sub__(a, b): b = lift(b); return Tensor(a.data - b.data, (a, b), lambda g: (a.acc(g), b.acc(-g)))
    def __mul__(a, b): b = lift(b); return Tensor(a.data * b.data, (a, b), lambda g: (a.acc(g * b.data), b.acc(g * a.data)))
    def __matmul__(a, b): return Tensor(a.data @ b.data, (a, b), lambda g: (a.acc(g @ b.data.T), b.acc(a.data.T @ g)))
    @property
    def T(a): return Tensor(a.data.T, (a,), lambda g: a.acc(g.T))
    def __pow__(a, p): return Tensor(a.data ** p, (a,), lambda g: a.acc(g * p * a.data ** (p - 1)))
    def mean(a): return Tensor(a.data.mean(-1, keepdims=True), (a,), lambda g: a.acc(np.broadcast_to(g, a.data.shape) / a.data.shape[-1]))
    def gelu(a):
        Phi = 0.5 * (1 + verf(a.data / math.sqrt(2))); phi = np.exp(-a.data ** 2 / 2) / math.sqrt(2 * math.pi)
        return Tensor(a.data * Phi, (a,), lambda g: a.acc(g * (Phi + a.data * phi)))
    def acc(a, g):                                                       # add into .grad, summing over broadcast dimensions
        while g.ndim > a.data.ndim: g = g.sum(0)
        for ax, n in enumerate(a.data.shape):
            if n == 1 and g.shape[ax] != 1: g = g.sum(ax, keepdims=True)
        a.grad += g
    def backward(a):
        order, seen = [], set()
        def visit(t):
            if id(t) not in seen: seen.add(id(t)); [visit(p) for p in t.parents]; order.append(t)
        visit(a); a.grad = np.ones_like(a.data)
        for t in reversed(order):                                       # children before parents
            if t.back: t.back(t.grad)
        return order

lift = lambda x: x if isinstance(x, Tensor) else Tensor(x)

def cross_entropy(logits, targets):                                     # mean over rows of -log softmax[target]; its backward is (p - y) / N (Part 2)
    z = logits.data - logits.data.max(-1, keepdims=True); p = np.exp(z) / np.exp(z).sum(-1, keepdims=True); n = len(targets)
    loss = -np.log(p[np.arange(n), targets]).mean(); y = np.zeros_like(p); y[np.arange(n), targets] = 1
    return Tensor(loss, (logits,), lambda g: logits.acc(g * (p - y) / n))

def layernorm(x, g, b, eps=1e-5):
    c = x - x.mean(); var = (c * c).mean(); return c * (var + eps) ** -0.5 * g + b
```

`part7.py`:

```python
import json, numpy as np, torch, torch.nn.functional as F
from part8 import Tiny, windows, test, T, D, V
from miniad import Tensor, cross_entropy, layernorm
torch.set_default_dtype(torch.float64)
m = Tiny().double(); m.load_state_dict({k: v.double() for k, v in torch.load("tiny8.pt").items()}); m.eval()
xb, yb = windows(test, list(range(0, 64 * 4, 64)))                          # 4 windows x 64 positions = 256 vectors
# residual stream just before the last feed-forward part: captured at the input of ln2 in the last block
store2 = {}; m.blocks[1].ln2.register_forward_pre_hook(lambda mod, i: store2.update(x=i[0].detach()))
with torch.no_grad(): m(xb)
R0 = store2["x"].reshape(-1, D); tgt = yb.reshape(-1)                       # residual stream entering the last feed-forward part, and the true next characters
blk = m.blocks[1]; P = lambda t: t.detach().numpy()
params = {"ln2.g": blk.ln2.weight, "ln2.b": blk.ln2.bias, "W1": blk.mlp[0].weight, "b1": blk.mlp[0].bias, "W2": blk.mlp[2].weight, "b2": blk.mlp[2].bias,
          "lnf.g": m.ln.weight, "lnf.b": m.ln.bias, "Wo": m.out.weight, "bo": m.out.bias}

# the same computation in the mini autograd
T_ = {k: Tensor(P(v)) for k, v in params.items()}; x = Tensor(P(R0))
h = layernorm(x, T_["ln2.g"], T_["ln2.b"]); h = (h @ T_["W1"].T + T_["b1"]).gelu() @ T_["W2"].T + T_["b2"]
r = x + h; z = layernorm(r, T_["lnf.g"], T_["lnf.b"]) @ T_["Wo"].T + T_["bo"]
loss = cross_entropy(z, tgt.numpy()); order = loss.backward()

# the same computation in PyTorch autograd
xt = R0.clone().requires_grad_(); pt = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
ln = lambda u, g, b: F.layer_norm(u, (D,), g, b, 1e-5)
ht = F.linear(F.gelu(F.linear(ln(xt, pt["ln2.g"], pt["ln2.b"]), pt["W1"], pt["b1"])), pt["W2"], pt["b2"])
zt = F.linear(ln(xt + ht, pt["lnf.g"], pt["lnf.b"]), pt["Wo"], pt["bo"]); lt = F.cross_entropy(zt, tgt); lt.backward()

print(f"loss: mini autograd {loss.data:.12f}, PyTorch {lt.item():.12f}  (bits per character {loss.data / np.log(2):.3f})")
print(f"graph: {len(order)} nodes in topological order; {sum(1 for t in order if t.back)} operations\n")
print("gradient check, largest absolute difference to PyTorch (relative to the largest gradient entry)")
rows = []
for k in params:
    a, b = T_[k].grad, pt[k].grad.numpy(); rows.append((k, np.abs(a - b).max(), np.abs(b).max())); print(f"   d{k:7s} shape {str(a.shape):12s} diff {np.abs(a - b).max():.1e}   (largest entry {np.abs(b).max():.2e})")
a, b = x.grad, xt.grad.numpy(); print(f"   dx       shape {str(a.shape):12s} diff {np.abs(a - b).max():.1e}   (largest entry {np.abs(b).max():.2e})")

print("\nwhy vector-Jacobian products: the size of the Jacobian we never build")
n = 256 * 256
print(f"   GELU on the (256 x 256) hidden tensor: full Jacobian {n} x {n} = {n * n:.2e} entries ({n * n * 8 / 1e9:.0f} GB in float64); the product with an upstream gradient is one element-wise multiply of {n:,} numbers")
print(f"   the Jacobian is diagonal, but a general framework does not know that; each op stores the few numbers its backward needs instead")

mem = sum(t.data.nbytes for t in order if t.back) + sum(t.data.nbytes for t in order if not t.back); act = sum(t.data.nbytes for t in order if t.back)
print(f"\nmemory kept for the backward pass in this graph: {act / 1e6:.1f} MB of intermediate values for 256 vectors ({act / 256 / 1e3:.1f} kB per vector); parameters and input {(mem - act) / 1e6:.2f} MB")
json.dump({"rows": rows, "n_nodes": len(order)}, open("results7.json", "w"))
```

Output:

```
loss: mini autograd 1.687224206127, PyTorch 1.687224206127  (bits per character 2.434)
graph: 43 nodes in topological order; 30 operations

gradient check, largest absolute difference to PyTorch (relative to the largest gradient entry)
   dln2.g   shape (64,)        diff 3.5e-17   (largest entry 3.22e-02)
   dln2.b   shape (64,)        diff 4.2e-17   (largest entry 3.52e-02)
   dW1      shape (256, 64)    diff 3.5e-17   (largest entry 5.29e-02)
   db1      shape (256,)       diff 1.4e-17   (largest entry 1.82e-02)
   dW2      shape (64, 256)    diff 4.2e-17   (largest entry 4.01e-02)
   db2      shape (64,)        diff 1.6e-17   (largest entry 1.32e-02)
   dlnf.g   shape (64,)        diff 1.9e-17   (largest entry 2.64e-02)
   dlnf.b   shape (64,)        diff 2.1e-17   (largest entry 2.85e-02)
   dWo      shape (48, 64)     diff 6.9e-17   (largest entry 9.68e-02)
   dbo      shape (48,)        diff 2.1e-17   (largest entry 2.08e-02)
   dx       shape (256, 64)    diff 6.9e-18   (largest entry 4.73e-03)

why vector-Jacobian products: the size of the Jacobian we never build
   GELU on the (256 x 256) hidden tensor: full Jacobian 65536 x 65536 = 4.29e+09 entries (34 GB in float64); the product with an upstream gradient is one element-wise multiply of 65,536 numbers
   the Jacobian is diagonal, but a general framework does not know that; each op stores the few numbers its backward needs instead

memory kept for the backward pass in this graph: 3.8 MB of intermediate values for 256 vectors (14.8 kB per vector); parameters and input 0.42 MB
```

## Try it

1. Change `acc` to assign instead of add, run the two-path example, and watch `x` get the wrong gradient.
2. Add an `exp` operation and a `log` operation to the engine, rebuild the softmax from them, and compare with the `(p - y) / N` shortcut.
3. Extend `mean` to take an axis, then build the attention softmax and check the gradient of the attention weights against PyTorch.
