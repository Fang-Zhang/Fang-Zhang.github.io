---
title: "From Zero to ML Researcher, Part 10: A Mini Scaling Law, and Where It Breaks"
date: 2027-05-20 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, scaling laws, overfitting, transformer, ML researcher]
description: "Part 10 of a 12-part series for complete beginners. Loss against model size follows a power law for small transformers, until the data runs out: a power law fitted on four sizes predicts 1.17 bits per character for the largest model, which measures 1.59."
---

## What you will do, and what you get

You will train the transformer of Part 8 at five widths (3,088 to 417,328 parameters, 3 seeds each) and plot the validation loss against the number of parameters on log-log axes. Between 3,088 and 110,384 parameters the loss falls along a rough power law. Fit that law on the four smaller models and extrapolate, and it predicts 1.17 bits per character for the largest model. The measured value is **1.59**. The largest model's test loss is *worse* than the next one down (1.99 against 1.91). A second sweep explains why: with the model fixed, shrinking the training text raises held-out loss from 1.91 to 2.08, 2.50 and 5.10. The "law" holds only while data is not the limit.

![Left: validation bits per character against parameters on log-log axes for five model sizes, with a power law fitted on the four smallest sizes and extended as a dotted line that continues downward to a predicted 1.17 at 417,328 parameters, where the measured point sits at 1.59. Middle: train, validation and test loss against parameters; train falls to 1.03 while validation flattens near 1.59 and test rises to 1.99. Right: train, validation and test loss against the number of training characters for a fixed 110,384-parameter model, from 54,846 to 438,769 characters, with test falling from 5.10 to 1.91]({{ site.baseurl }}/assets/pic/mlr-10-scaling.png){: width="900" }

## How: the steps

1. **Keep everything fixed except one knob.** Same text, split, 3,000 steps, batch of 64 windows of 64 characters, same optimiser. The knob is the width `D` (8, 16, 32, 64, 128), with 2 layers and a head size of 16 (so `D // 16` heads, at least 1).
2. **Train 3 seeds per size** and average, as in Part 9.
3. **Record** parameters, train, validation and test loss for every run.
4. **Fit** `L = a * N^(-alpha)` by least squares on the logarithms, using validation loss for the four smaller sizes only.
5. **Predict the largest size and then measure it.** A fit is a claim; the held-out size is the test of the claim.
6. **Vary the data** with the model fixed, to see what limits the curve.

## Why it works: the principle

**What a scaling law claims.** [Kaplan and colleagues (2020)](https://arxiv.org/abs/2001.08361) report that language-model loss "scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude". On log-log axes a power law is a straight line, and the slope `alpha` is the exponent: multiplying the parameters by 10 multiplies the loss by `10^(-alpha)`. The same paper describes simple equations for how overfitting depends on model and dataset size. [Hoffmann and colleagues (2022)](https://arxiv.org/abs/2203.15556) trained over 400 models and conclude that for compute-optimal training, model size and training tokens should be scaled equally: for every doubling of model size, the number of training tokens should also be doubled.

**What the mini version shows.**
- The local slope of the validation curve is 0.20 between widths 8 and 16, 0.28 between 16 and 32, 0.12 between 32 and 64 and 0.006 between 64 and 128. A true power law has one slope; this curve has a steep middle and then flattens. Five points cannot distinguish a power law with a floor from other shapes, so read `alpha = 0.206` as a description of the small models only.
- Train loss keeps falling with size (3.62, 2.88, 1.93, 1.50, 1.03), while validation reaches 1.59 and stays there, and test goes from 1.91 to 1.99. A model with 417,328 parameters has about as many parameters as the training text has characters (438,769); it can memorise.
- The data sweep confirms the cause. With 54,846 training characters the same 110,384-parameter model reaches train loss 0.68 and validation loss 4.95, close to the 5.585 of guessing: it has memorised a short text and is confidently wrong on new text. At 109,692 characters validation is 1.95, at 219,384 it is 1.66 and at 438,769 it is 1.60.
- This matches the logic of the two papers: loss depends on model size *and* data, and growing only one of them stops paying off. In this setting the bottleneck for the biggest model is data.

**What this mini-experiment cannot show.** Real scaling laws span many orders of magnitude and use enough data for every size; this one spans two orders of magnitude of parameters on 439k characters. It should not be used to predict how a large model behaves.

**Pitfalls and limits.**
- **Extrapolating a short fit is the classic mistake.** The prediction of 1.17 is what a straight line gives; the model did not follow it. A researcher reports the failed prediction, not only the fitted part.
- **Fixed steps favour some sizes.** All models train for 3,000 steps with the same learning rate; the small ones may be limited by capacity, the large one by data, and none of them was tuned.
- **Width is not the only axis.** Depth and head count change too (heads grow with width), so the fit is against total parameters, not width alone.
- **Text blocks differ in difficulty.** The "train" number is measured on the first 54,846 characters of the training text, which is not the same text as validation. Part 6 showed that the held-out blocks have different difficulty, so compare *trends* across sizes, not the level of the train column against validation.
- **Three seeds per size** are enough to see the trend here (Part 8 shows a seed std below 0.02 at width 64); they are not enough to put an interval on an exponent.
- **The data sweep takes the first fraction of the training text,** so smaller sets also cover fewer topics. That makes the small-data results look worse than a random sample of the same size would.

## Example, worked through

The exponent between two sizes, by hand: widths 16 and 32 have 9,200 and 30,640 parameters and mean validation losses 2.612 and 1.872.

- `alpha = ln(2.612 / 1.872) / ln(30640 / 9200) = 0.3330 / 1.2031 = 0.277`.
- With that exponent, 10 times the parameters would multiply the loss by `10^(-0.277) = 0.529`.
- Applying it to 417,328 parameters gives 0.908, while 1.589 was measured. The exponent of the steep middle section does not continue.

```python
import numpy as np
n16, n32, v16, v32 = 9200, 30640, 2.612, 1.872                  # parameters and mean validation loss at widths 16 and 32 (Part 10 sweep)
slope = -(np.log(v32) - np.log(v16)) / (np.log(n32) - np.log(n16))
print(f"alpha between width 16 and 32 = ln({v16}/{v32}) / ln({n32}/{n16}) = {slope:.3f}")
print(f"so a model with 10x the parameters would have a loss multiplied by {10 ** -slope:.3f}, if this exponent held")
n128 = 417328
print(f"extrapolated to {n128:,} parameters: {v32 * (n128 / n32) ** -slope:.3f} bits per character (measured: 1.589)")
```

Output:

```
alpha between width 16 and 32 = ln(2.612/1.872) / ln(30640/9200) = 0.277
so a model with 10x the parameters would have a loss multiplied by 0.529, if this exponent held
extrapolated to 417,328 parameters: 0.908 bits per character (measured: 1.589)
```

## Implementation

`part10.py` trains the 15 models of the width sweep (about 17 minutes on a CPU) and appends one line per run to `results10.jsonl`. Keep it next to `part8.py` of Part 8, which provides the data, the split and the scoring.

```python
import json, math, time
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from part8 import V, T, train, val, test, windows, bpc          # same data, split and scoring as Parts 8 and 9

class Block(nn.Module):
    def __init__(s, D, heads):
        super().__init__(); s.D, s.h = D, heads
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x):
        B, t, _ = x.shape
        q, k, v = (a.view(B, t, s.h, s.D // s.h).transpose(1, 2) for a in s.qkv(s.ln1(x)).split(s.D, dim=2))
        x = x + s.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, s.D))
        return x + s.mlp(s.ln2(x))

class Tiny(nn.Module):
    def __init__(s, D, layers=2):
        super().__init__(); heads = max(1, D // 16)                # head size 16 throughout
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.Sequential(*[Block(D, heads) for _ in range(layers)]); s.ln = nn.LayerNorm(D); s.out = nn.Linear(D, V)
    def forward(s, idx):
        return s.out(s.ln(s.blocks(s.tok(idx) + s.pos(torch.arange(idx.shape[1])))))

def run(D, seed, steps=3000, lr=3e-3):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = Tiny(D); opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(1, steps + 1):
        for gr in opt.param_groups: gr["lr"] = lr * (1 - step / (steps + 1))
        xb, yb = windows(train, torch.randint(0, len(train) - T - 1, (64,), generator=g).tolist())
        loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return sum(p.numel() for p in m.parameters()), bpc(m, train[:len(val)]), bpc(m, val), bpc(m, test)

if __name__ == "__main__":
    t0 = time.time()
    with open("results10.jsonl", "w") as f:
        for D in (8, 16, 32, 64, 128):
            for seed in range(3):
                n, tr, va, te = run(D, seed)
                f.write(json.dumps({"D": D, "seed": seed, "params": n, "train": tr, "val": va, "test": te}) + "\n"); f.flush()
                print(f"width {D:3d} seed {seed}: {n:7,} parameters  train {tr:.3f}  validation {va:.3f}  test {te:.3f}  ({time.time() - t0:.0f} s elapsed)", flush=True)
```

Output:

```
width   8 seed 0:   3,088 parameters  train 3.637  validation 3.270  test 3.450  (19 s elapsed)
width   8 seed 1:   3,088 parameters  train 3.649  validation 3.289  test 3.467  (37 s elapsed)
width   8 seed 2:   3,088 parameters  train 3.573  validation 3.214  test 3.400  (55 s elapsed)
width  16 seed 0:   9,200 parameters  train 2.826  validation 2.573  test 2.831  (75 s elapsed)
width  16 seed 1:   9,200 parameters  train 2.907  validation 2.637  test 2.885  (95 s elapsed)
width  16 seed 2:   9,200 parameters  train 2.893  validation 2.626  test 2.890  (113 s elapsed)
width  32 seed 0:  30,640 parameters  train 1.907  validation 1.862  test 2.167  (151 s elapsed)
width  32 seed 1:  30,640 parameters  train 1.948  validation 1.880  test 2.188  (188 s elapsed)
width  32 seed 2:  30,640 parameters  train 1.929  validation 1.875  test 2.155  (225 s elapsed)
width  64 seed 0: 110,384 parameters  train 1.487  validation 1.585  test 1.916  (301 s elapsed)
width  64 seed 1: 110,384 parameters  train 1.481  validation 1.596  test 1.908  (370 s elapsed)
width  64 seed 2: 110,384 parameters  train 1.528  validation 1.622  test 1.911  (438 s elapsed)
width 128 seed 0: 417,328 parameters  train 1.038  validation 1.579  test 1.967  (625 s elapsed)
width 128 seed 1: 417,328 parameters  train 1.062  validation 1.573  test 1.985  (812 s elapsed)
width 128 seed 2: 417,328 parameters  train 1.002  validation 1.614  test 2.019  (996 s elapsed)
```

The analysis, `analyse10.py`:

```python
import json
import numpy as np
rows = [json.loads(l) for l in open("results10.jsonl")]
sizes = sorted({r["D"] for r in rows})
N = np.array([np.mean([r["params"] for r in rows if r["D"] == d]) for d in sizes])
val = np.array([np.mean([r["val"] for r in rows if r["D"] == d]) for d in sizes])
test = np.array([np.mean([r["test"] for r in rows if r["D"] == d]) for d in sizes])
tr = np.array([np.mean([r["train"] for r in rows if r["D"] == d]) for d in sizes])
print("width  parameters  train   validation   test   (means over 3 seeds)")
for d, n, a, b, c in zip(sizes, N, tr, val, test): print(f"{d:5d}  {int(n):10,}  {a:.3f}   {b:.3f}       {c:.3f}")

def fit_power(N, L, offset=0.0):                              # L = offset + a * N^(-alpha), by least squares on logs
    A = np.vstack([np.ones(len(N)), -np.log(N)]).T
    (la, alpha), *_ = np.linalg.lstsq(A, np.log(L - offset), rcond=None)
    return np.exp(la), alpha
def predict(N, a, alpha, offset=0.0): return offset + a * N ** (-alpha)

k = len(sizes) - 1                                            # fit on all sizes but the largest, then predict the largest
print(f"\nfit on the {k} smaller models, predict the largest ({int(N[-1]):,} parameters):")
a, alpha = fit_power(N[:k], val[:k])
print(f"  pure power law  L = a N^-alpha : alpha = {alpha:.3f}; predicted validation {predict(N[-1], a, alpha):.3f}, measured {val[-1]:.3f}")
a2, al2 = fit_power(N[:k], test[:k])
print(f"  same on test                   : alpha = {al2:.3f}; predicted test {predict(N[-1], a2, al2):.3f}, measured {test[-1]:.3f}")
for lo, hi in zip(sizes, sizes[1:]):
    i = sizes.index(lo); j = sizes.index(hi)
    print(f"  local slope between width {lo} and {hi}: {-(np.log(val[j]) - np.log(val[i])) / (np.log(N[j]) - np.log(N[i])):.3f} (validation)")
```

Output:

```
width  parameters  train   validation   test   (means over 3 seeds)
    8       3,088  3.620   3.258       3.439
   16       9,200  2.875   2.612       2.869
   32      30,640  1.928   1.872       2.170
   64     110,384  1.499   1.601       1.911
  128     417,328  1.034   1.589       1.990

fit on the 4 smaller models, predict the largest (417,328 parameters):
  pure power law  L = a N^-alpha : alpha = 0.206; predicted validation 1.171, measured 1.589
  same on test                   : alpha = 0.171; predicted test 1.474, measured 1.990
  local slope between width 8 and 16: 0.202 (validation)
  local slope between width 16 and 32: 0.277 (validation)
  local slope between width 32 and 64: 0.122 (validation)
  local slope between width 64 and 128: 0.006 (validation)
```

The data sweep, `part10b.py` (9 runs, about 10 minutes; the 438,769-character point is the width-64 row above):

```python
import json, time
import numpy as np, torch
import torch.nn.functional as F
import part10                                                   # model, scoring and data from the width sweep
from part10 import Tiny, V, T, train, val, test, windows, bpc

def run_fraction(D, frac, seed, steps=3000, lr=3e-3):
    data = train[: int(frac * len(train))]                       # the first `frac` of the training text; validation and test stay the same
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = Tiny(D); opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(1, steps + 1):
        for gr in opt.param_groups: gr["lr"] = lr * (1 - step / (steps + 1))
        xb, yb = windows(data, torch.randint(0, len(data) - T - 1, (64,), generator=g).tolist())
        loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return bpc(m, data[: len(val)]), bpc(m, val), bpc(m, test)

if __name__ == "__main__":
    t0 = time.time()
    with open("results10b.jsonl", "w") as f:
        for frac in (0.125, 0.25, 0.5):
            for seed in range(3):
                tr, va, te = run_fraction(64, frac, seed)
                f.write(json.dumps({"frac": frac, "seed": seed, "chars": int(frac * len(train)), "train": tr, "val": va, "test": te}) + "\n"); f.flush()
                print(f"{int(frac * len(train)):7,} training characters, seed {seed}: train {tr:.3f}  validation {va:.3f}  test {te:.3f}  ({time.time() - t0:.0f} s elapsed)", flush=True)
```

Output:

```
 54,846 training characters, seed 0: train 0.647  validation 4.458  test 4.778  (71 s elapsed)
 54,846 training characters, seed 1: train 0.672  validation 5.279  test 5.307  (138 s elapsed)
 54,846 training characters, seed 2: train 0.710  validation 5.119  test 5.216  (205 s elapsed)
109,692 training characters, seed 0: train 1.152  validation 1.959  test 2.513  (274 s elapsed)
109,692 training characters, seed 1: train 1.161  validation 1.949  test 2.490  (342 s elapsed)
109,692 training characters, seed 2: train 1.175  validation 1.949  test 2.501  (411 s elapsed)
219,384 training characters, seed 0: train 1.390  validation 1.657  test 2.074  (480 s elapsed)
219,384 training characters, seed 1: train 1.393  validation 1.651  test 2.082  (547 s elapsed)
219,384 training characters, seed 2: train 1.412  validation 1.662  test 2.087  (615 s elapsed)
```

## Try it

1. Add a width of 96 and 192 to the sweep. Where does validation loss turn around?
2. Fit the power law with an additive floor, `L = c + a * N^(-alpha)`. Does it fit the five points better, and how many parameters does the fit now have compared with the data points?
3. Train the width-128 model for 1,000 steps instead of 3,000. Does the smaller number of steps close the gap between train and validation loss?
