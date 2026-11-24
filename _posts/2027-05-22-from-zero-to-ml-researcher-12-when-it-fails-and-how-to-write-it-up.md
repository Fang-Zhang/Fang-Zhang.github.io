---
title: "From Zero to ML Researcher, Part 12: When It Fails, and How to Write It Up"
date: 2027-05-22 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, debugging, negative results, reproducibility, research log, ML researcher]
description: "Part 12 of a 12-part series for complete beginners. Five training runs on a tiny transformer, two of them with a one-line bug that produces a validation loss near zero; how to spot a result that is too good, keep a research log, write up negative results, and use a reproducibility checklist."
---

## What you will do, and what you get

You will train the same 110,384-parameter transformer five times for 600 steps: once correctly, twice with a bad learning rate, and twice with a one-line bug. The bugs are the interesting part, because they produce the *best-looking* numbers. Without a shifted target the validation loss is **0.001 bits per character**; without a causal mask it is **0.059**. The healthy run gets 2.151. Checked correctly (shifted targets, causal mask), the two buggy models score **16.1** and **10.7**, about two to three times *worse* than guessing uniformly (5.585). A result that is far better than a trivial baseline is a reason to look for a bug before looking for a celebration.

The same loop closes the series: keep a log, check the result, write it up with its negative parts, and make it rerunnable.

![Left: five training-loss curves on a log scale. The healthy run falls to about 2.1; a learning rate of 0.3 stalls near 4.4; a learning rate of 1e-5 stays near 5; the run with unshifted targets collapses to the plotting floor of 0.01 within 100 steps; the run without a causal mask falls to about 0.06. Middle: validation loss as reported by each run against validation loss computed correctly: 2.15 and 2.15 for the healthy run, 4.43 and 4.43, 5.03 and 5.03, 0.00 and 16.1 for unshifted targets, 0.06 and 10.7 for no causal mask. Right: a six-item checklist to apply before believing a result]({{ site.baseurl }}/assets/pic/mlr-12-failures.png){: width="900" }

## How: the steps

1. **Start from a trusted run** (the healthy curve) and keep it as the reference for every later change.
2. **Break one thing at a time**, on purpose, to learn what each failure looks like: the learning rate (too high, too low) and two code bugs.
3. **Log the curve and the numbers** for every run: loss at several steps, validation loss, settings, seed.
4. **Check the number against a baseline** before believing it. Here the baselines are 5.585 bits (uniform guessing among 48 characters, Part 6) and the counting model of Part 6 at 2.19.
5. **Re-evaluate under the correct protocol.** For every model, compute validation loss with shifted targets and a causal mask, whatever the training run did.
6. **Write down what failed,** not only what worked, and what you would try next.

## Why it works: the principle

**What each failure looks like.**
- **Learning rate too high (0.3):** the loss drops in the first steps, then stalls near 4.4 and never improves. Updates overshoot and the weights end up in a poor region. (In this run the loss stalled; it did not become `nan`.)
- **Learning rate too low (1e-5):** the loss decreases slowly and is still near 5.0 after 600 steps. Nothing is broken, it is just not done; a longer run would continue.
- **Targets not shifted:** the model is asked to output the character it was given. Copying is easy, so the loss goes to about zero within 100 steps. In the plot it sits on the 0.01 floor chosen for the log scale; the real values are lower.
- **No causal mask:** each position can see the characters after it, including the one it must predict. The model learns to read the answer from the future, and the loss drops to 0.06. At generation time the future does not exist.
- **The healthy run** falls smoothly and slowly (2.08 in training, 2.151 on validation). It is the least impressive curve and the only honest one.

**Why "too good" is a signal.** The loss of a model that predicts the next character has a floor set by the text itself: language has real uncertainty (the 2.19 of the counting model and the 1.91 of Part 8 are both far above 0, and Shannon's 1951 estimates for English letters were on the order of one bit each). A validation loss near 0 means the answer leaked into the input. The same rule applies at every scale: a result that beats everything published by a wide margin is checked for leakage first.

**Why checking under the correct protocol works.** A leak helps only under the protocol that contains it. The model without the mask scores 0.059 with the mask off and 10.7 with it on; the model without shifted targets scores 0.001 when asked to copy and 16.1 when asked to predict. The correct protocol is the one that matches how the model will be used.

**The research log.** One text file, appended to on every experiment. Each entry has the date, the question, the exact command, the result as printed, and one line on what it changes. It is the cheapest tool in the series; it turns "I think I tried that" into a lookup. The Part 9 habit applies: write the hypothesis before the run, and what would count as a surprise.

**Negative results belong in the write-up.** The scaling extrapolation of Part 10 failed (predicted 1.17, measured 1.59) and the post-LN instability of Part 11 did not appear at this scale. Both are reported with their numbers. Leaving them out would make the rest look more certain than it is, and the next reader would repeat the experiment. The write-up has a fixed order that this series has used throughout: result first, how it was obtained, why it works with its pitfalls, a worked example, and the full code.

**A reproducibility checklist.** [Pineau and colleagues](https://arxiv.org/abs/2003.12206) describe a machine learning reproducibility checklist and the practice of releasing code with a paper. The list in the figure is a shorter version in the same spirit, and it is only as useful as the habit of ticking it before a post or paper goes out.

**Pitfalls and limits.**
- **A single run per setting.** The numbers here show *which kind* of failure occurs, not a precise size; the bugs' effects are so large (0.001 against 2.15) that seeds do not matter, but small differences would need several seeds, as in Part 9.
- **These are not all the bugs.** Data leakage between train and validation text, a wrong split, a test block used for tuning and a mis-computed metric are other common ones, and they too look like improvements.
- **The same protocol check cannot catch everything.** A bug that the correct protocol shares (for example, an overlap between train and validation text) passes it. A fresh test set, used once, is the final check.
- **The 0.01 floor in the figure** is a plotting choice for the log axis; the unshifted-target loss is lower.

## Example, worked through

The test for a leak: edit only the *later* characters of an input and see whether the model's output at *earlier* positions changes. With a causal mask it must not. Below, an untrained model sees 16 characters; characters 10 to 15 are replaced, and the outputs at positions 0 to 9 are compared.

```python
import torch
from part12 import Tiny
torch.manual_seed(0)
m = Tiny().eval()
x = torch.randint(0, 48, (1, 16)); x2 = x.clone(); x2[0, 10:] = (x2[0, 10:] + 7) % 48       # change only the characters from position 10 on
with torch.no_grad():
    for causal in (True, False):
        a, b = m(x, causal), m(x2, causal)
        print(f"causal={causal!s:5}: largest change in the output at positions 0-9 after editing positions 10-15: {(a[0, :10] - b[0, :10]).abs().max().item():.6f}")
```

Output:

```
causal=True : largest change in the output at positions 0-9 after editing positions 10-15: 0.000000
causal=False: largest change in the output at positions 0-9 after editing positions 10-15: 0.064713
```

## Implementation

`part12.py` trains the five models (about 70 seconds on a CPU) and writes `results12.json`. Keep it next to `part8.py` of Part 8.

```python
import json, math, time
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from part8 import V, D, T, train, val, windows                      # same data and split as Parts 8-11

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.h = 4
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x, causal):
        B, t, _ = x.shape
        q, k, v = (a.view(B, t, s.h, D // s.h).transpose(1, 2) for a in s.qkv(s.ln1(x)).split(D, dim=2))
        x = x + s.proj(F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2).reshape(B, t, D))
        return x + s.mlp(s.ln2(x))

class Tiny(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.ModuleList([Block() for _ in range(2)]); s.ln = nn.LayerNorm(D); s.out = nn.Linear(D, V)
    def forward(s, idx, causal=True):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1]))
        for b in s.blocks: x = b(x, causal)
        return s.out(s.ln(x))

@torch.no_grad()
def score(m, data, causal, shifted, n=300):                          # bits per character on the first n windows of `data`
    starts = list(range(0, n * T, T)); xb, yb = windows(data, starts)
    if not shifted: yb = xb
    return F.cross_entropy(m(xb, causal).reshape(-1, V), yb.reshape(-1)).item() / math.log(2)

def run(name, lr=3e-3, causal=True, shifted=True, steps=600, seed=0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = Tiny(); opt = torch.optim.AdamW(m.parameters(), lr=lr); curve = []
    for step in range(1, steps + 1):
        xb, yb = windows(train, torch.randint(0, len(train) - T - 1, (64,), generator=g).tolist())
        if not shifted: yb = xb                                      # BUG: the target is the input itself, not the next character
        loss = F.cross_entropy(m(xb, causal).reshape(-1, V), yb.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        curve.append(loss.item() / math.log(2))
    reported = score(m, val, causal, shifted)                        # validation under the same (possibly buggy) protocol
    honest = score(m, val, True, True)                               # validation under the correct protocol: shifted targets, causal mask
    return curve, reported, honest

if __name__ == "__main__":
    runs = {"healthy": {}, "learning rate too high (0.3)": {"lr": 0.3}, "learning rate too low (1e-5)": {"lr": 1e-5},
            "bug: targets not shifted": {"shifted": False}, "bug: no causal mask": {"causal": False}}
    out = {}; t0 = time.time()
    for name, kw in runs.items():
        curve, rep, hon = run(name, **kw); out[name] = {"curve": curve, "reported": rep, "honest": hon}
        print(f"{name:30s} loss at step 1/100/300/600: {curve[0]:.2f} / {curve[99]:.2f} / {curve[299]:.2f} / {curve[-1]:.2f}   validation as reported {rep:.3f}   validation done correctly {hon:.3f}   ({time.time() - t0:.0f} s)", flush=True)
    json.dump(out, open("results12.json", "w"))
```

Output:

```
healthy                        loss at step 1/100/300/600: 5.82 / 3.67 / 2.71 / 2.08   validation as reported 2.151   validation done correctly 2.151   (16 s)
learning rate too high (0.3)   loss at step 1/100/300/600: 5.82 / 4.65 / 4.39 / 4.43   validation as reported 4.429   validation done correctly 4.429   (30 s)
learning rate too low (1e-5)   loss at step 1/100/300/600: 5.82 / 5.71 / 5.45 / 5.06   validation as reported 5.034   validation done correctly 5.034   (43 s)
bug: targets not shifted       loss at step 1/100/300/600: 6.02 / 0.02 / 0.00 / 0.00   validation as reported 0.001   validation done correctly 16.087   (58 s)
bug: no causal mask            loss at step 1/100/300/600: 5.83 / 1.57 / 0.06 / 0.06   validation as reported 0.059   validation done correctly 10.692   (71 s)
```

## Where to go from here

- **Repeat Part 11 with 6 layers**, and Part 10 with a larger text, then write both up with the log and the checklist. That is a small piece of research with a stranger being able to rerun it.
- **Read three papers with the three passes** and replicate one number from each.
- **Pick one question from your own notes,** state the hypothesis and what would falsify it before the first run, and keep the result whatever it is.

## The series

- [From Zero to ML Researcher, Part 1: The Map]({% post_url 2027-05-11-from-zero-to-ml-researcher-1-the-map %})
- [From Zero to ML Researcher, Part 2: Python, Arrays and Shapes]({% post_url 2027-05-12-from-zero-to-ml-researcher-2-python-arrays-shapes %})
- [From Zero to ML Researcher, Part 3: What Learning Is]({% post_url 2027-05-13-from-zero-to-ml-researcher-3-what-learning-is %})
- [From Zero to ML Researcher, Part 4: Evidence]({% post_url 2027-05-14-from-zero-to-ml-researcher-4-evidence %})
- [From Zero to ML Researcher, Part 5: A Neural Network From Scratch]({% post_url 2027-05-15-from-zero-to-ml-researcher-5-neural-network-from-scratch %})
- [From Zero to ML Researcher, Part 6: Language as Prediction]({% post_url 2027-05-16-from-zero-to-ml-researcher-6-language-as-prediction %})
- [From Zero to ML Researcher, Part 7: Attention From Scratch]({% post_url 2027-05-17-from-zero-to-ml-researcher-7-attention-from-scratch %})
- [From Zero to ML Researcher, Part 8: A Small Transformer on a CPU]({% post_url 2027-05-18-from-zero-to-ml-researcher-8-a-small-transformer-on-a-cpu %})
- [From Zero to ML Researcher, Part 9: Hypotheses and Ablations]({% post_url 2027-05-19-from-zero-to-ml-researcher-9-hypotheses-and-ablations %})
- [From Zero to ML Researcher, Part 10: A Mini Scaling Law, and Where It Breaks]({% post_url 2027-05-20-from-zero-to-ml-researcher-10-a-mini-scaling-law %})
- [From Zero to ML Researcher, Part 11: Reading a Paper and Replicating It]({% post_url 2027-05-21-from-zero-to-ml-researcher-11-reading-a-paper-and-replicating-it %})
- Part 12: this post
