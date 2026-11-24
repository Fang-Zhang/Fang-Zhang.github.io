---
title: "Inside the Math, Part 8: What Training Holds in Memory, and How Adam Steps"
date: 2027-05-30 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, adam, optimizer, backpropagation, memory, transformer, inside the math]
description: "Part 8, the last of an 8-part series on the classic maths inside language models. Backpropagation through a whole transformer holds 575 kB of activations per 64-character window, 21 times the parameter, gradient and Adam state at batch 64. Adam's update and bias correction, written by hand, match torch.optim.Adam to 3.2e-12."
---

## What you will do, and what you get

Parts 1 to 7 covered the pieces: softmax, cross-entropy, normalization, the feed-forward block and backpropagation. This part puts them together on a whole 110,384-parameter character transformer and answers two questions: what must a training step hold in memory, and how does Adam turn a gradient into a step?

- **Activations dominate the memory of a training step.** At batch 64 (windows of 64 characters) the values saved for the backward pass take 36.8 MB, 575 kB per window. Parameters (0.44 MB), gradients (0.44 MB) and Adam's two extra numbers per parameter (0.88 MB) together take 1.77 MB, so the activations are 21 times larger, and they grow linearly with the batch: 4.6 MB at batch 8, 73.6 MB at batch 128.
- **Adam written by hand matches `torch.optim.Adam`.** After 50 steps from the trained model, in float64, the largest parameter difference is 3.2e-12.
- **Bias correction matters in the first steps.** The mean size of the first update, in units of the learning rate, is 0.974 with the correction and 3.074 without. For a constant gradient, the uncorrected step is 3.16 times too large at step 1 and 6.53 times at step 10, and is within 1.26 times only after about 1,000 steps.
- **Adam beats plain SGD here by a wide margin.** From the same random start, same batches, 300 steps: validation loss 2.380 bits per character for Adam (learning rate 3e-3) against 3.264 for the best plain SGD tried (learning rate 2.0); SGD with learning rate 5 or more diverges to `nan`.
- **The embedding table gets its gradient by scatter-add.** In a batch of 512 characters with 34 distinct ones, exactly 34 rows of the token table receive a nonzero gradient, and each equals the sum of the output gradients at the positions where its character occurs.

![Left: stacked bars of training memory against batch size 8, 16, 32, 64, 128: a fixed 1.77 MB for parameters, gradients and Adam state, plus activations growing from 4.6 MB to 73.6 MB, totals 6.4, 11.0, 20.2, 38.6 and 75.4 MB. Middle: the factor by which the step size is too large without bias correction, for a constant gradient, rising from 3.2 at step 1 to 6.5 at step 10 and falling to 1.0 by step 3000, against a flat line at 1 with the correction. Right: training loss over 300 steps for Adam and for SGD at three learning rates, Adam falling to 2.77 while SGD stays above 3.6]({{ site.baseurl }}/assets/pic/mm-8-memory-adam.png){: width="900" }

## How: the steps

**The backward pass through the whole model** reuses the parts of this series, in reverse order of the forward pass:

1. Loss to logits: softmax and cross-entropy give `(p - y) / N` (Part 2).
2. Through the output layer and the final LayerNorm: matrix products as in Part 6 and the projection of Part 4.
3. For each block, last to first: the feed-forward part (Part 6), its LayerNorm (Part 4), then the attention part, where the softmax over each row of scores contributes its Jacobian `diag(p) - p p^T` (Part 1 and 2) and the matrix products contribute their usual `dW` and `dX`.
4. At every residual connection the gradient splits and the two branches add up again (Part 7).
5. At the embeddings, the gradient is scattered back: row `c` of the token table receives the sum of the gradients at all positions holding character `c`; the other rows get zero.

**Then Adam updates every parameter**, with `g` the gradient, `t` the step number and, for the learning rate `lr`, the decay rates `b1 = 0.9` and `b2 = 0.999`, and `eps = 1e-8` (the defaults of [Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980)):

1. `m = b1 * m + (1 - b1) * g` (a running average of the gradient)
2. `v = b2 * v + (1 - b2) * g^2` (a running average of its square)
3. `m_hat = m / (1 - b1^t)` and `v_hat = v / (1 - b2^t)` (bias correction)
4. `theta = theta - lr * m_hat / (sqrt(v_hat) + eps)`

All operations are element-wise: every parameter has its own `m` and `v`.

## Why it works: the principle

**Why activations are saved.** The backward steps of Parts 4 to 7 need the forward values: multiplication needs both factors, GELU needs its input, LayerNorm needs the normalised vector and its scale, softmax needs the probabilities. The count above is measured by hooking every tensor PyTorch saves for the backward pass, counting each stored buffer once and leaving out the parameters themselves. It scales with batch size times positions, and in the plain formulation of attention also with the square of the context length, since each head scores every position against every other. The model here calls PyTorch's fused attention function, and this post does not separate how much of the count belongs to the attention scores. At this scale the whole training step takes 38.6 MB at batch 64; the parameters and Adam's state are a small part, because the model is small. In a large model the proportions change, since parameters grow with the model while activations grow with the model and the batch. This post measures only the small model.

**What Adam does.** The first moment `m` smooths the noisy gradient. The second moment `v` estimates the typical squared size of each parameter's gradient, and dividing by its root makes the step roughly `lr` in size whatever the scale of the gradient: the authors list invariance to diagonal rescaling of the gradients among the method's properties. In the worked example below, the first step is exactly `lr` in size for a gradient of 1; it would be the same for a gradient of 100.

**Why the bias correction.** `m` and `v` start at 0. After `t` steps with a constant gradient `g`, `m = (1 - b1^t) g` and `v = (1 - b2^t) g^2`, both too small. Dividing by `(1 - b1^t)` and `(1 - b2^t)` removes this exactly. Without it, the step is `(1 - b1^t) / sqrt(1 - b2^t)` times the right size, the curve in the middle panel: `v` is biased much more than `m` early on because `b2` is much closer to 1, so the denominator is too small and the step too large. On the first step the measured mean size is 3.074 times `lr` without the correction (the formula gives 3.16 for a constant gradient) and 0.974 with it (slightly below 1; entries with very small gradients, where `eps` matters, are one possible reason).

**Why not plain SGD.** The SGD comparison is one seed and 300 steps with a coarse learning-rate grid (0.1, 0.5, 2, 5, 10, 20): lr 2.0 is the best of those and lr 5 diverges, so a value between 2 and 5 might do slightly better, but the gap to Adam (3.264 against 2.380 bits per character) is large. A longer run, momentum, or a learning-rate schedule would narrow it. The model of this series was trained with AdamW, the variant of [Loshchilov and Hutter (2019)](https://arxiv.org/abs/1711.05101) that decouples weight decay from the gradient-based step; they show that L2 regularisation and weight decay are not equivalent for adaptive methods such as Adam.

**Pitfalls and limits.**
- **Per-parameter scale is only mildly different here.** In the trained model, the RMS gradient of the weight matrices and LayerNorm gains runs from 5.3e-04 (position table) to 3.3e-03 (the first block's attention projection), a factor of 6, so this model does not show a dramatic per-tensor imbalance for Adam to fix.
- **Optimiser state doubles the parameter memory.** Adam keeps `m` and `v` for each parameter, in addition to the gradient.
- **`eps` is not decoration.** For entries with tiny gradients `sqrt(v_hat)` can be comparable to `eps`, which then limits the step.
- **The activation count is for this implementation.** Frameworks can trade compute for memory by recomputing activations during the backward pass; the numbers above are for the default behaviour of PyTorch and for float32.
- **One model, one corpus, 300 steps.** The comparison with SGD says that Adam is much faster here, not by how much on other models.

## Example, worked through

A single parameter at 1.0, `lr = 0.1`, and three gradients: 1.0, 0.5, -0.5.

- Step 1: `m = 0.1`, `v = 0.001`; corrected `m_hat = 1.0`, `v_hat = 1.0`; update `-0.1 * 1.0 / 1.0 = -0.1`; the parameter becomes 0.9.
- Step 2: `m = 0.14`, `v = 0.001249`; `m_hat = 0.7368`, `v_hat = 0.6248`; update `-0.0932`; the parameter is 0.80678.
- Step 3: the gradient flips sign, but `m` keeps memory of the earlier ones: `m_hat = 0.2804`, update only `-0.0397` (still downhill, because the average gradient is still positive); the parameter is 0.76711.
- `torch.optim.Adam` gives the same three values.

The second half of the output checks the embedding table's scatter-add on the real model.

```python
import torch, math
from part8 import Tiny, windows, train, T, V
import torch.nn.functional as F
# A. Adam by hand on one scalar, 3 steps, against torch.optim.Adam
lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8; gs = [1.0, 0.5, -0.5]; th = 1.0; m = v = 0.0
p = torch.tensor(1.0, dtype=torch.float64, requires_grad=True); opt = torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps)
print("step  g     m       v         m_hat   v_hat      update    theta   torch theta")
for t, g in enumerate(gs, 1):
    m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g; mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t); up = lr * mh / (math.sqrt(vh) + eps); th -= up
    p.grad = torch.tensor(g, dtype=torch.float64); opt.step()
    print(f"{t:3d} {g:5.1f} {m:7.4f} {v:9.6f} {mh:7.4f} {vh:9.6f} {-up:9.5f} {th:8.5f} {p.item():8.5f}")
# B. the embedding table gets its gradient by scatter-add: rows of characters not in the batch get exactly zero
model = Tiny(); model.load_state_dict(torch.load("tiny8.pt")); xb, yb = windows(train, list(range(0, 64 * 8, 64)))
store = {}
def keep(mod, i, o): o.retain_grad(); store.update(o=o)                  # returns nothing, so the output is unchanged
model.tok.register_forward_hook(keep)
F.cross_entropy(model(xb).reshape(-1, V), yb.reshape(-1)).backward()
G = model.tok.weight.grad; nz = int((G.abs().sum(1) > 0).sum()); distinct = len(set(xb.flatten().tolist()))
c = xb.flatten()[0].item(); manual = store["o"].grad.reshape(-1, G.shape[1])[xb.flatten() == c].sum(0)
print(f"\nbatch of {xb.numel()} characters, {distinct} distinct; rows of the token table with nonzero gradient: {nz}")
print(f"row for character id {c}: gradient = sum of the {int((xb.flatten() == c).sum())} output gradients at its positions, difference {(G[c] - manual).abs().max().item():.1e}")
```

Output:

```
step  g     m       v         m_hat   v_hat      update    theta   torch theta
  1   1.0  0.1000  0.001000  1.0000  1.000000  -0.10000  0.90000  0.90000
  2   0.5  0.1400  0.001249  0.7368  0.624812  -0.09322  0.80678  0.80678
  3  -0.5  0.0760  0.001498  0.2804  0.499750  -0.03967  0.76711  0.76711

batch of 512 characters, 34 distinct; rows of the token table with nonzero gradient: 34
row for character id 41: gradient = sum of the 38 output gradients at its positions, difference 1.4e-09
```

## Implementation

`fin8.py` loads the model trained by `part8.py` (see Part 1 of this series), measures the memory, runs Adam by hand against PyTorch, and trains with Adam and SGD. About two minutes on a CPU. Keep `part8.py`, `tiny8.pt` and both scripts in one folder.

```python
import copy, json, math, time, numpy as np, torch, torch.nn.functional as F
from part8 import Tiny, windows, train, val, test, T, D, V
sd = torch.load("tiny8.pt"); nparam = lambda m: sum(p.numel() for p in m.parameters())
def batch(g, n=64): return windows(train, torch.randint(0, len(train) - T - 1, (n,), generator=g).tolist())
def lossf(m, xb, yb): return F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1))
res = {}

print("1. memory: what a training step must hold (float32)")
m = Tiny(); m.load_state_dict(sd); ptrs = {p.untyped_storage().data_ptr() for p in m.parameters()}
def saved_bytes(m, n):
    seen, rows = set(), []
    def pack(t):
        k = t.untyped_storage().data_ptr()
        if k not in ptrs and k not in seen: seen.add(k); rows.append((t.untyped_storage().nbytes(), tuple(t.shape)))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        xb, yb = batch(torch.Generator().manual_seed(0), n); lossf(m, xb, yb)
    return sum(r[0] for r in rows), rows
pb = nparam(m) * 4
print(f"   parameters {nparam(m):,} -> {pb / 1e6:.2f} MB; gradients {pb / 1e6:.2f} MB; Adam keeps two more numbers per parameter (m and v): {2 * pb / 1e6:.2f} MB")
act = {}
for n in (8, 16, 32, 64, 128):
    act[n] = saved_bytes(m, n)[0]; print(f"   batch {n:3d} windows x {T} positions: activations saved for backward {act[n] / 1e6:6.2f} MB  ({act[n] / n / 1e3:.0f} kB per window)")
rows = sorted(saved_bytes(m, 64)[1], reverse=True)[:4]; print("   largest saved tensors at batch 64:", [f"{s} = {b / 1e6:.2f} MB" for b, s in rows])
res["mem"] = {"params": pb, "grads": pb, "adam": 2 * pb, "act": act}

print("\n2. Adam by hand vs torch.optim.Adam, 50 steps from the trained model, float64")
a = Tiny().double(); a.load_state_dict({k: v.double() for k, v in sd.items()}); b = copy.deepcopy(a)
lr, b1, b2, eps = 1e-3, 0.9, 0.999, 1e-8
opt = torch.optim.Adam(b.parameters(), lr=lr, betas=(b1, b2), eps=eps)
M = [torch.zeros_like(p) for p in a.parameters()]; Vv = [torch.zeros_like(p) for p in a.parameters()]; g = torch.Generator().manual_seed(1); first = None
for t in range(1, 51):
    xb, yb = batch(g)
    for mm in (a, b): mm.zero_grad(); lossf(mm, xb, yb).backward()
    opt.step()
    with torch.no_grad():
        for i, p in enumerate(a.parameters()):
            M[i] = b1 * M[i] + (1 - b1) * p.grad; Vv[i] = b2 * Vv[i] + (1 - b2) * p.grad ** 2
            mh = M[i] / (1 - b1 ** t); vh = Vv[i] / (1 - b2 ** t); step = lr * mh / (vh.sqrt() + eps)
            if t == 1: first = first or []; first.append((p.grad.abs().mean().item(), (step.abs().mean().item() / lr), (lr * M[i] / (Vv[i].sqrt() + eps)).abs().mean().item() / lr))
            p -= step
d = max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
print(f"   largest parameter difference after 50 steps: {d:.1e} (parameters move by about {lr * 50:.0e} at most)")
print(f"   first step, mean |update| / lr: with bias correction {np.mean([f[1] for f in first]):.3f}, without {np.mean([f[2] for f in first]):.3f}")
res["adam_diff"] = d; res["first"] = [float(np.mean([f[1] for f in first])), float(np.mean([f[2] for f in first]))]

print("\n3. the factor that bias correction removes, for a constant gradient: (1 - b1^t) / sqrt(1 - b2^t)")
ts = [1, 2, 5, 10, 50, 100, 500, 1000, 5000]; fac = [(1 - b1 ** t) / math.sqrt(1 - b2 ** t) for t in ts]
print("   " + "  ".join(f"t={t}: {f:.2f}" for t, f in zip(ts, fac))); res["fac"] = {"t": list(range(1, 3001)), "f": [(1 - b1 ** t) / math.sqrt(1 - b2 ** t) for t in range(1, 3001)]}

print("\n4. gradient scale differs between tensors; Adam's step does not (trained model, one batch)")
m.zero_grad(); xb, yb = batch(torch.Generator().manual_seed(2)); lossf(m, xb, yb).backward(); tab = []
for n_, p in m.named_parameters():
    if p.ndim == 2 or "ln" in n_:
        tab.append((n_, p.grad.pow(2).mean().sqrt().item())); 
tab.sort(key=lambda r: r[1]); print(f"   RMS gradient: smallest {tab[0][0]} {tab[0][1]:.1e}, largest {tab[-1][0]} {tab[-1][1]:.1e}, ratio {tab[-1][1] / tab[0][1]:.0f}x")
res["grad_rms"] = tab

print("\n5. Adam vs plain SGD from the same random start, 300 steps, batch 64, validation loss in bits per character (float32)")
from part8 import bpc
curves = {}
for name, mk in [("Adam lr 3e-3", lambda p: torch.optim.Adam(p, 3e-3)), *[(f"SGD lr {x}", (lambda x: lambda p: torch.optim.SGD(p, x))(x)) for x in (0.1, 0.5, 2.0, 5.0, 10.0, 20.0)]]:
    torch.manual_seed(0); mm = Tiny(); o = mk(mm.parameters()); g = torch.Generator().manual_seed(3); c = []; t0 = time.time()
    for s in range(1, 301):
        xb, yb = batch(g); l = lossf(mm, xb, yb); o.zero_grad(); l.backward(); o.step()
        if s % 50 == 0: c.append((s, l.item() / math.log(2)))
    v = bpc(mm, val); curves[name] = {"curve": c, "val": v}; print(f"   {name:13s} final train {c[-1][1]:.3f}, validation {v:.3f}   ({time.time() - t0:.0f} s)")
res["curves"] = curves; json.dump(res, open("results8.json", "w"))
```

Output:

```
1. memory: what a training step must hold (float32)
   parameters 110,384 -> 0.44 MB; gradients 0.44 MB; Adam keeps two more numbers per parameter (m and v): 0.88 MB
   batch   8 windows x 64 positions: activations saved for backward   4.60 MB  (575 kB per window)
   batch  16 windows x 64 positions: activations saved for backward   9.20 MB  (575 kB per window)
   batch  32 windows x 64 positions: activations saved for backward  18.40 MB  (575 kB per window)
   batch  64 windows x 64 positions: activations saved for backward  36.80 MB  (575 kB per window)
   batch 128 windows x 64 positions: activations saved for backward  73.60 MB  (575 kB per window)
   largest saved tensors at batch 64: ['(4096, 256) = 4.19 MB', '(4096, 256) = 4.19 MB', '(64, 64, 256) = 4.19 MB', '(64, 64, 256) = 4.19 MB']

2. Adam by hand vs torch.optim.Adam, 50 steps from the trained model, float64
   largest parameter difference after 50 steps: 3.2e-12 (parameters move by about 5e-02 at most)
   first step, mean |update| / lr: with bias correction 0.974, without 3.074

3. the factor that bias correction removes, for a constant gradient: (1 - b1^t) / sqrt(1 - b2^t)
   t=1: 3.16  t=2: 4.25  t=5: 5.80  t=10: 6.53  t=50: 4.50  t=100: 3.24  t=500: 1.59  t=1000: 1.26  t=5000: 1.00

4. gradient scale differs between tensors; Adam's step does not (trained model, one batch)
   RMS gradient: smallest pos.weight 5.3e-04, largest blocks.0.proj.weight 3.3e-03, ratio 6x

5. Adam vs plain SGD from the same random start, 300 steps, batch 64, validation loss in bits per character (float32)
   Adam lr 3e-3  final train 2.774, validation 2.380   (7 s)
   SGD lr 0.1    final train 3.955, validation 3.509   (6 s)
   SGD lr 0.5    final train 3.724, validation 3.305   (6 s)
   SGD lr 2.0    final train 3.647, validation 3.264   (8 s)
   SGD lr 5.0    final train nan, validation nan   (6 s)
   SGD lr 10.0   final train nan, validation nan   (6 s)
   SGD lr 20.0   final train nan, validation nan   (6 s)
```

## The series in one paragraph

Softmax (Part 1) turns scores into probabilities and needs the shift-by-the-maximum trick to avoid overflow; paired with cross-entropy (Part 2) its gradient collapses to `p - y`. LayerNorm (Parts 3 and 4) re-centres and re-scales each vector, and its gradient is a projection divided by the standard deviation. The feed-forward block (Parts 5 and 6) is two matrices and a nonlinearity, whose backward pass is five matrix products. Backpropagation (Part 7) is the chain rule on a graph, in 41 lines. Training a whole transformer (this part) keeps every activation for the backward pass, and Adam turns the gradients into normalised steps.

## Try it

1. Use the memory counter on a context of 128 characters instead of 64 (this needs a model with 128 positions) and check how the attention part grows.
2. Add momentum 0.9 to SGD and extend the learning-rate grid between 2 and 5, and see how much of the Adam gap remains.
3. Remove the bias correction from the by-hand Adam in `fin8.py`, train for 300 steps and compare the loss curve with the corrected one.
