---
title: "Inside the Math, Part 3: LayerNorm, RMSNorm and BatchNorm, Compared Line by Line"
date: 2027-05-25 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, layer normalization, rmsnorm, batch normalization, transformer, inside the math]
description: "Part 3 of an 8-part series on the classic maths inside language models. Three normalisations differ only in which numbers they average over: LayerNorm and RMSNorm per vector, BatchNorm per feature across the batch. Written from scratch, matched to PyTorch to 1e-6, and measured on a trained transformer where the residual stream grows from std 1.29 to 3.11 across the layers."
---

## What you will do, and what you get

You will write three normalisations in a few lines of numpy-style code, check them against PyTorch, and measure what they do inside a trained 110,384-parameter character transformer. The results:

- **The three differ only in the axis they average over.** LayerNorm gives each vector (a position's 64 features) mean 0 and std 1; BatchNorm gives each feature mean 0 and std 1 across all 2,048 vectors of the batch; RMSNorm divides each vector by its root mean square and does not subtract the mean. Hand-written versions match PyTorch to 9.5e-7 or better.
- **The residual stream grows with depth, and LayerNorm resets it.** The standard deviation of the vectors entering the five LayerNorms of the model grows from 1.29 (first block) to 3.11 (final layer), with a spread up to 6.83 for single vectors. After normalising, every vector has std 1 before the learned gain and bias.
- **BatchNorm makes an example's output depend on the other examples.** Replacing the other windows in the batch changes the output for window 0 by up to 0.423; for LayerNorm and RMSNorm the change is exactly 0.
- **RMSNorm is scale-invariant; and in this model the mean is small.** Multiplying the input by 10 changes the output by 1.3e-5 (from the `eps` in the denominator). The mean of a vector is about 0.15 against a root mean square of 1.3, which is why dropping the mean subtraction can cost little in this setting.

![Left: two 5 by 5 grids of rows (vectors) and columns (features); for LayerNorm and RMSNorm one row is highlighted, for BatchNorm one column is highlighted. Middle: box plots of the standard deviation of the vectors entering each of the five LayerNorms in the trained model, rising from 1.29 to 3.11 with a dashed line at 1 for the normalised output. Right: the largest change in window 0's output when the other windows are replaced: 0.000 for LayerNorm, 0.000 for RMSNorm and 0.423 for BatchNorm]({{ site.baseurl }}/assets/pic/mm-3-normalization.png){: width="900" }

## How: the steps

1. **Collect the vectors.** Run 32 windows of 64 characters through the trained model and capture what enters and leaves the first LayerNorm with a forward hook: 2,048 vectors of 64 numbers.
2. **LayerNorm, per vector:** `y = (x - mean(x)) / sqrt(var(x) + eps) * gain + bias`, with the mean and variance taken over the 64 features of that vector. The variance divides by `n`, not `n - 1`.
3. **RMSNorm, per vector:** `y = x / sqrt(mean(x^2) + eps) * gain`. There is no mean subtraction and no bias.
4. **BatchNorm, per feature:** the same formula as LayerNorm, but the mean and variance are taken over the 2,048 vectors, separately for each of the 64 features.
5. **Compare with `nn.LayerNorm`, `nn.RMSNorm` and `nn.BatchNorm1d`** on the same input.
6. **Probe two properties:** swap the other windows in the batch and see whether window 0's output changes; multiply the input by 10 and see whether the output changes.
7. **Measure the size of the vectors** entering all five LayerNorms of the trained model.

## Why it works: the principle

**What each one removes.** LayerNorm removes two things from each vector: its average level (re-centering) and its overall size (re-scaling). RMSNorm removes only the size. [Zhang and Sennrich (2019)](https://arxiv.org/abs/1910.07467) hypothesise that the re-centering invariance of LayerNorm is dispensable and propose RMSNorm as a computationally simpler version; they report comparable performance on several tasks. Large language models such as [LLaMA](https://arxiv.org/abs/2302.13971) use RMSNorm (see the architecture section of the paper for the placement). The measurement above fits the argument: the mean of a vector is a small part of its size here (0.15 against 1.3), so removing it changes little.

**Why normalise at all.** The residual stream accumulates the output of every attention and MLP block, so its size grows: the measured standard deviation is 1.29, 1.84, 2.22, 2.39 and 3.11 at the five points. Each sublayer receives its input after a normalisation, so what it sees has a fixed scale however large the stream has become. [Ba, Kiros and Hinton (2016)](https://arxiv.org/abs/1607.06450) introduced LayerNorm to remove the dependence on the batch: it computes the statistics from the summed inputs of a single example, so the computation is the same in training and testing. Part 4 of this series derives its gradient.

**Why LayerNorm and not BatchNorm in language models.** [Ioffe and Szegedy (2015)](https://arxiv.org/abs/1502.03167) normalise each layer's inputs per mini-batch and report the same accuracy with 14 times fewer training steps on an image model. The price of using batch statistics is the dependence shown in the right panel: the output for one example depends on which other examples share the batch, so training and testing need different procedures (running averages at test time). For text, where lengths vary and a batch may contain padding, per-vector statistics are simpler. (This is a design argument, not something this experiment tests.)

**Learned gain and bias.** After normalising, each feature is multiplied by a learned gain and shifted by a learned bias, so the layer can undo the normalisation if that helps. In this model the std after the layer is 1.12 to 1.59, not 1, because the trained gains are not 1.

**The `eps`.** It keeps the denominator away from zero for a constant vector. It is also why the scale invariance is 1.3e-5 and not exactly 0.

**Pitfalls and limits.**
- **Which axis?** `nn.LayerNorm(64)` normalises the last dimension of size 64. A transposed tensor normalises the wrong thing without raising an error.
- **Biased variance.** `x.var()` in PyTorch divides by `n - 1` by default; the normalisations divide by `n` (`unbiased=False`). Using the default gives results off by a factor `sqrt(64/63)` and a mismatch of about one percent against `nn.LayerNorm`.
- **BatchNorm in evaluation mode** uses stored running statistics instead of the batch, so the same input gives different outputs in train and eval mode.
- **One model, one set of windows.** The sizes (1.29 to 3.11) are for this model and these 32 windows. Another model will have another profile.
- **Position of the layer.** In this model the normalisation sits inside each residual branch (before attention and before the MLP), as in the pre-LN design of [Xiong and colleagues (2020)](https://arxiv.org/abs/2002.04745); the original transformer normalised after the residual addition.

## Example, worked through

The vector `v = [1, 2, 3, 4]`.

- LayerNorm: mean `2.5`, variance `((-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2) / 4 = 1.25`, std `1.118`; output `(v - 2.5) / 1.118 = [-1.342, -0.447, 0.447, 1.342]`.
- RMSNorm: `mean(v^2) = 7.5`, rms `2.739`; output `v / 2.739 = [0.365, 0.730, 1.095, 1.461]`. The mean stays positive.
- Multiplying `v` by 10 leaves both outputs unchanged (up to `eps`).
- BatchNorm on three vectors `[1, 10]`, `[2, 20]`, `[3, 30]`: each column becomes `[-1.225, 0, 1.225]`. Replace the third vector by `[30, 3]` and the first vector's output changes from `[-1.225, -1.225]` to `[-0.744, -0.143]`, although the first vector itself did not change.

```python
import numpy as np
v = np.array([1.0, 2.0, 3.0, 4.0]); eps = 1e-5
mu, var = v.mean(), v.var()                                               # variance divides by n (4), not n - 1
print(f"LayerNorm: mean {mu}, variance {var}, std {np.sqrt(var):.3f} -> {((v - mu) / np.sqrt(var + eps)).round(3)}")
rms = np.sqrt((v ** 2).mean()); print(f"RMSNorm  : rms sqrt({(v ** 2).mean()}) = {rms:.3f} -> {(v / np.sqrt((v ** 2).mean() + eps)).round(3)}   (no mean subtracted; mean stays positive)")
print(f"scaling v by 10 changes nothing: {np.allclose((10 * v - 10 * mu) / np.sqrt(100 * var + eps), (v - mu) / np.sqrt(var + eps), atol=1e-4)}")
B = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])                      # a batch of 3 vectors with 2 features
bn = (B - B.mean(0)) / np.sqrt(B.var(0) + eps); print("BatchNorm, batch of 3 x 2 features: each column ->\n", bn.round(3))
B2 = B.copy(); B2[2] = [30.0, 3.0]; bn2 = (B2 - B2.mean(0)) / np.sqrt(B2.var(0) + eps)
print("same first row, but the third row is replaced -> the first row's output becomes", bn2[0].round(3), "instead of", bn[0].round(3))
```

Output:

```
LayerNorm: mean 2.5, variance 1.25, std 1.118 -> [-1.342 -0.447  0.447  1.342]
RMSNorm  : rms sqrt(7.5) = 2.739 -> [0.365 0.73  1.095 1.461]   (no mean subtracted; mean stays positive)
scaling v by 10 changes nothing: True
BatchNorm, batch of 3 x 2 features: each column ->
 [[-1.225 -1.225]
 [ 0.     0.   ]
 [ 1.225  1.225]]
same first row, but the third row is replaced -> the first row's output becomes [-0.744 -0.143] instead of [-1.225 -1.225]
```

## Implementation

`part3.py` loads the model trained by `part8.py` (see Part 1 of this series; 110,384 parameters, about 4 minutes on a CPU), hooks the five LayerNorms, and runs the comparisons. Keep `part8.py`, `tiny8.pt` and the script in one folder.

```python
import json, math, numpy as np, torch, torch.nn as nn
from part8 import Tiny, chars, windows, test, T, D
m = Tiny(); m.load_state_dict(torch.load("tiny8.pt")); m.eval()
xb, yb = windows(test, list(range(0, 64 * 32, 64)))                         # 32 windows of 64 characters
acts = {}
hooks = [m.blocks[0].ln1.register_forward_hook(lambda mod, i, o: acts.update(ln_in=i[0].detach(), ln_out=o.detach()))]
with torch.no_grad(): m(xb)
x, y = acts["ln_in"], acts["ln_out"]                                         # (32, 64, 64): the vectors that enter and leave the first LayerNorm
print(f"vectors entering the first LayerNorm of the trained model: {x.shape[0] * x.shape[1]} vectors of {x.shape[2]} numbers")
print(f"   before: per-vector mean ranges {x.mean(-1).min():.2f} to {x.mean(-1).max():.2f}, per-vector std ranges {x.std(-1, unbiased=False).min():.2f} to {x.std(-1, unbiased=False).max():.2f}")
print(f"   after : per-vector mean ranges {y.mean(-1).min():.2f} to {y.mean(-1).max():.2f}, per-vector std ranges {y.std(-1, unbiased=False).min():.2f} to {y.std(-1, unbiased=False).max():.2f}   (the learned gain and bias are still applied)")

def layernorm(x, g, b, eps=1e-5): mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True); return (x - mu) / torch.sqrt(var + eps) * g + b
def rmsnorm(x, g, eps=1e-5):      return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * g
def batchnorm(x, g, b, eps=1e-5): mu = x.mean(0, keepdim=True); var = x.var(0, unbiased=False, keepdim=True); return (x - mu) / torch.sqrt(var + eps) * g + b   # statistics over the batch, per feature
ln = m.blocks[0].ln1; g, b = ln.weight.detach(), ln.bias.detach()
print("\nhand-written vs PyTorch:")
print(f"   LayerNorm  max difference {(layernorm(x, g, b) - ln(x)).abs().max().item():.1e}")
print(f"   RMSNorm    max difference {(rmsnorm(x, torch.ones(D)) - nn.RMSNorm(D, eps=1e-5)(x)).abs().max().item():.1e}")
flat = x.reshape(-1, D)
print(f"   BatchNorm  max difference {(batchnorm(flat, torch.ones(D), torch.zeros(D)) - nn.BatchNorm1d(D, eps=1e-5).train()(flat)).abs().max().item():.1e}")

print("\nwhich axis is normalised? 32 windows x 64 positions x 64 features")
print(f"   LayerNorm : each of the {flat.shape[0]} vectors gets mean 0, std 1 over its own 64 features -> after: mean {layernorm(x, torch.ones(D), torch.zeros(D)).mean(-1).abs().max():.1e}")
print(f"   BatchNorm : each of the {D} features gets mean 0, std 1 over the {flat.shape[0]} vectors  -> after: mean {batchnorm(flat, torch.ones(D), torch.zeros(D)).mean(0).abs().max():.1e}")

print("\nbatch dependence: the output for window 0 when the other windows change")
x0 = x[0]; other_a = x[1:5].reshape(-1, D); other_b = x[10:14].reshape(-1, D)
for name, f in (("LayerNorm", lambda v: layernorm(v, g, b)), ("RMSNorm", lambda v: rmsnorm(v, g)), ("BatchNorm", lambda v: batchnorm(v, g, b))):
    a = f(torch.cat([x0, other_a]))[:64]; c = f(torch.cat([x0, other_b]))[:64]
    print(f"   {name:9s}: largest change in window 0's output {(a - c).abs().max().item():.4f}")

print("\nscale invariance of the input: multiply every number by 10")
for name, f in (("LayerNorm", lambda v: layernorm(v, g, b)), ("RMSNorm", lambda v: rmsnorm(v, g))):
    print(f"   {name:9s}: largest change {(f(x0) - f(10 * x0)).abs().max().item():.1e}")
print(f"   RMSNorm needs no mean: how large is the mean compared with the root-mean-square in the trained model? mean |mean| {x.mean(-1).abs().mean():.3f} vs rms {x.pow(2).mean(-1).sqrt().mean():.3f}")
print("\nthe size of the vectors entering each of the 5 LayerNorms of the trained model (std over the 64 features, per vector)")
names = ["block 1, before attention", "block 1, before MLP", "block 2, before attention", "block 2, before MLP", "final, before the output layer"]
mods = [m.blocks[0].ln1, m.blocks[0].ln2, m.blocks[1].ln1, m.blocks[1].ln2, m.ln]; store = []
hs = [mod.register_forward_hook(lambda mod, i, o: store.append((i[0].detach(), o.detach()))) for mod in mods]
with torch.no_grad(): m(xb)
layer = []
for nme, (a, o) in zip(names, store):
    sd = a.std(-1, unbiased=False).flatten(); mn = a.mean(-1).abs().mean().item(); rms = a.pow(2).mean(-1).sqrt().mean().item()
    layer.append({"name": nme, "std_in": sd.tolist(), "mean_abs_mean": mn, "rms": rms}); print(f"   {nme:32s} std {sd.mean():.2f} (range {sd.min():.2f} to {sd.max():.2f}), mean of |mean| {mn:.3f}, rms {rms:.2f}; after the layer (gain and bias applied) std {o.std(-1, unbiased=False).mean():.2f}")
json.dump({"layers": layer, "x0": x[0][10].tolist(), "y0": y[0][10].tolist()}, open("results3.json", "w"))
```

Output:

```
vectors entering the first LayerNorm of the trained model: 2048 vectors of 64 numbers
   before: per-vector mean ranges -0.57 to 0.53, per-vector std ranges 1.04 to 1.68
   after : per-vector mean ranges -0.03 to 0.04, per-vector std ranges 1.10 to 1.26   (the learned gain and bias are still applied)

hand-written vs PyTorch:
   LayerNorm  max difference 9.5e-07
   RMSNorm    max difference 4.8e-07
   BatchNorm  max difference 9.5e-07

which axis is normalised? 32 windows x 64 positions x 64 features
   LayerNorm : each of the 2048 vectors gets mean 0, std 1 over its own 64 features -> after: mean 5.4e-08
   BatchNorm : each of the 64 features gets mean 0, std 1 over the 2048 vectors  -> after: mean 3.3e-08

batch dependence: the output for window 0 when the other windows change
   LayerNorm: largest change in window 0's output 0.0000
   RMSNorm  : largest change in window 0's output 0.0000
   BatchNorm: largest change in window 0's output 0.4230

scale invariance of the input: multiply every number by 10
   LayerNorm: largest change 1.3e-05
   RMSNorm  : largest change 1.3e-05
   RMSNorm needs no mean: how large is the mean compared with the root-mean-square in the trained model? mean |mean| 0.147 vs rms 1.301

the size of the vectors entering each of the 5 LayerNorms of the trained model (std over the 64 features, per vector)
   block 1, before attention        std 1.29 (range 1.04 to 1.68), mean of |mean| 0.147, rms 1.30; after the layer (gain and bias applied) std 1.20
   block 1, before MLP              std 1.84 (range 1.40 to 2.94), mean of |mean| 0.159, rms 1.85; after the layer (gain and bias applied) std 1.18
   block 2, before attention        std 2.22 (range 1.87 to 3.29), mean of |mean| 0.164, rms 2.23; after the layer (gain and bias applied) std 1.12
   block 2, before MLP              std 2.39 (range 1.76 to 3.57), mean of |mean| 0.171, rms 2.40; after the layer (gain and bias applied) std 1.50
   final, before the output layer   std 3.11 (range 1.44 to 6.83), mean of |mean| 0.179, rms 3.12; after the layer (gain and bias applied) std 1.59
```

## Try it

1. Replace `unbiased=False` by the default in `layernorm` and measure the mismatch against `nn.LayerNorm`.
2. Add a sixth statistic: the RMS of the vectors *after* each LayerNorm, and compare it with the learned gains of that layer.
3. Feed the model a window of 64 identical characters and print the standard deviation entering each LayerNorm. Is it smaller or larger than for real text?
