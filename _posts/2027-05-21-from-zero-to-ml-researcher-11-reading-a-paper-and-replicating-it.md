---
title: "From Zero to ML Researcher, Part 11: Reading a Paper and Replicating It"
date: 2027-05-21 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, replication, reading papers, layer normalization, transformer, ML researcher]
description: "Part 11 of a 12-part series for complete beginners. Read a paper in three passes, turn one of its claims into an experiment, and report honestly what matched. Replicating the pre-LN claim of Xiong et al. (2020) on a tiny transformer: pre-LN works without warm-up, but post-LN also trains without it, so the instability is not reproduced at this scale."
---

## What you will do, and what you get

You will read a research paper the way researchers do, extract one testable claim, and run it. The paper is [Xiong and colleagues (2020), "On Layer Normalization in the Transformer Architecture"](https://arxiv.org/abs/2002.04745). Its claim: the original transformer, which normalises *after* each residual addition (post-LN), needs a learning-rate warm-up to train stably, while the arrangement of Part 8, which normalises *inside* the residual branch (pre-LN), does not. You will test it on the 110,384-parameter model of Part 8: 2 learning rates, 2 arrangements, with and without a 200-step warm-up, 3 seeds each (24 runs).

**What matched:** pre-LN trains fine without warm-up (test 2.017 at learning rate 3e-3, 1.887 at 1e-2). **What did not:** post-LN without warm-up also trains fine here (2.008 and 1.883), and warm-up made both arrangements slightly *worse* (+0.04 at 3e-3, +0.006 at 1e-2). The paper's instability does not show up at this scale. That is a result about this setting, not a refutation of the paper.

![Left: test bits per character for eight configurations, three seeds each, at learning rates 3e-3 (values 2.017, 2.051, 2.008, 2.047 for pre-LN no warm-up, pre-LN warm-up, post-LN no warm-up, post-LN warm-up) and 1e-2 (1.887, 1.897, 1.883, 1.889). Middle: mean training loss over the first 200 steps, higher with warm-up (about 4.1 and 3.8) than without (about 3.6 and 3.3). Right: a table comparing the claims of the paper with what was observed]({{ site.baseurl }}/assets/pic/mlr-11-replication.png){: width="900" }

## How: the steps

1. **First pass (about 10 minutes):** read the title, abstract, introduction, section headings and conclusions. Decide what the paper claims and whether it is worth more time. This and the next two passes are the method of [Keshav, "How to Read a Paper"](https://doi.org/10.1145/1273445.1273458).
2. **Second pass:** read the body and look carefully at figures and tables, skipping proofs. Write down the evidence offered for each claim: here, a mean-field analysis of gradients at initialisation, and training runs on translation tasks.
3. **Pick one claim that a small machine can test.** Here: "Pre-LN Transformers without the warm-up stage can reach comparable results with baselines", paired with "for Post-LN, using a large learning rate without warm-up makes training unstable" (both in the abstract).
4. **Third pass: re-create it.** Keshav's third pass is to "virtually re-implement the paper", making the same assumptions as the authors, then compare. Here that means writing both arrangements and training them under the same conditions.
5. **Fix the protocol before running:** same data and split, 1,500 steps, a learning rate of 3e-3 (the one used so far) and 1e-2 (higher, where instability is more likely), warm-up of 0 or 200 steps followed by linear decay, 3 seeds.
6. **Run, then compare with the claim,** including what you did not expect.
7. **Write down the gap** between your setting and the paper's, because that is where the explanation lives.

## Why it works: the principle

**A replication is a claim, an experiment and a comparison.** A claim from a paper is only as general as the experiments behind it. The abstract says the effect appears "on a wide range of applications"; the model here has 2 layers and 64 features, a different task (characters, not translation), and 1,500 training steps. If the effect appears, the claim extends to this setting. If it does not, it is *not* a refutation: the setting may simply lie outside the region where the effect exists.

**The two arrangements, in code.** Pre-LN (Part 8): `x = x + attention(ln(x))`, then `x = x + mlp(ln(x))`. Post-LN: `x = ln(x + attention(x))`, then `x = ln(x + mlp(x))`. In Post-LN the normalisation stands on the path of the gradient between layers; in Pre-LN the residual path carries the gradient directly. The paper's theory is that at initialisation, post-LN gives large expected gradients near the output layer, so a large learning rate at the start is unstable, and warm-up (starting from a small learning rate and increasing it) avoids that.

**Reading the results.**
- With warm-up, the first 200 steps are slower (mean training loss 4.1 against 3.6 at 3e-3), and the run ends with a higher test loss, since a 200-step ramp costs roughly 100 steps' worth of full learning rate out of 1,500. That is expected, not a surprise.
- Without warm-up, post-LN does not blow up: the worst training-batch loss after step 200 is 3.14 (post-LN) against 3.29 (pre-LN) at 3e-3, and 2.58 against 2.69 at 1e-2. If the paper's instability applied, post-LN would be the one with spikes.
- Pre-LN against post-LN without warm-up differ by -0.009 at 3e-3 and -0.005 at 1e-2 (post-LN slightly better); with 3 seeds the 95% intervals are [-0.019, -0.000] and [-0.035, +0.026], so the evidence for any difference is thin. The honest reading is "no meaningful difference here".
- A side result: a learning rate of 1e-2 gives 1.887 on test, against 2.017 at 3e-3 in the same 1,500 steps. The learning rate used in Parts 8 to 10 was not the best one for this model. The comparisons in those parts changed one component at a time under one learning rate, so their ranking stands, but their absolute numbers are not tuned.

**Why the replication may have missed the effect (not tested here).** The paper's experiments are on different tasks and, presumably, much larger models than this one (check its experiments section for the exact sizes). A 2-layer model has only two normalisations between input and output, and the gradient it receives at initialisation is much less amplified than in a deep stack. Testing this would mean repeating the experiment with 6 or 12 layers: that is the next experiment, not a conclusion.

**Pitfalls and limits.**
- **3 seeds** give wide intervals (the t-value for 2 degrees of freedom is 4.303), which the interval above reflects. Differences of 0.005 are not detectable.
- **Different arrangement, different details:** post-LN here has no separate final layer norm (each block already ends with one), while pre-LN has one. The paper's exact optimiser settings and schedule differ from this one.
- **The test block was used for the comparison.** No model was selected on it, but the same block was seen in earlier parts.
- **One learning-rate pair and one warm-up length.** Other values might expose the effect.
- **Do not turn a null result into a rebuttal.** The accurate sentence is "at this scale, with these settings, the post-LN instability was not observed".

## Example, worked through

The paired comparison, post-LN minus pre-LN with no warm-up, learning rate 3e-3, same seeds: differences `-0.008, -0.014, -0.007`; mean `-0.0094`. The standard error is the sample std divided by `sqrt(3)`, the interval `mean +/- 4.303 * se = [-0.019, -0.000]`. At 1e-2 the differences are `-0.011, -0.013, +0.010`, mean `-0.0047`, interval `[-0.035, +0.026]`: it contains zero and one seed even changes sign.

```python
import json, numpy as np
R = [json.loads(l) for l in open("results11.jsonl")]
t = lambda lr, post, w: np.array([next(x["test"] for x in R if x["lr"] == lr and x["post"] == post and x["warmup"] == w and x["seed"] == s) for s in range(3)])
for lr in (3e-3, 1e-2):
    d = t(lr, True, 0) - t(lr, False, 0)                      # post-LN minus pre-LN, no warm-up, same seed
    se = d.std(ddof=1) / np.sqrt(3)
    print(f"lr {lr:g}: post-LN minus pre-LN, no warm-up: differences {d.round(3)}, mean {d.mean():+.4f}, 95% interval [{d.mean() - 4.303 * se:+.3f}, {d.mean() + 4.303 * se:+.3f}]  (t = 4.303 for 2 degrees of freedom)")
```

Output:

```
lr 0.003: post-LN minus pre-LN, no warm-up: differences [-0.008 -0.014 -0.007], mean -0.0094, 95% interval [-0.019, -0.000]  (t = 4.303 for 2 degrees of freedom)
lr 0.01: post-LN minus pre-LN, no warm-up: differences [-0.011 -0.013  0.01 ], mean -0.0047, 95% interval [-0.035, +0.026]  (t = 4.303 for 2 degrees of freedom)
```

## Implementation

`part11.py` trains the 24 models (about 14 minutes on a CPU) and appends one line per run to `results11.jsonl`. Keep it next to `part8.py` of Part 8, which provides the data, the split and the scoring.

```python
import json, math, time
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from part8 import V, D, T, train, val, test, windows, bpc          # same data, split and scoring as Parts 8-10

class Block(nn.Module):
    def __init__(s, post):
        super().__init__(); s.post, s.h = post, 4
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def attn(s, x):
        B, t, _ = x.shape
        q, k, v = (a.view(B, t, s.h, D // s.h).transpose(1, 2) for a in s.qkv(x).split(D, dim=2))
        return s.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
    def forward(s, x):
        if s.post:                                              # original arrangement: normalise AFTER the residual addition
            x = s.ln1(x + s.attn(x)); return s.ln2(x + s.mlp(x))
        x = x + s.attn(s.ln1(x)); return x + s.mlp(s.ln2(x))   # pre-LN, as in Part 8

class Tiny(nn.Module):
    def __init__(s, post):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.Sequential(*[Block(post) for _ in range(2)]); s.ln = nn.Identity() if post else nn.LayerNorm(D); s.out = nn.Linear(D, V)
    def forward(s, idx):
        return s.out(s.ln(s.blocks(s.tok(idx) + s.pos(torch.arange(idx.shape[1])))))

def run(post, warmup, lr, seed, steps=1500):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = Tiny(post); opt = torch.optim.AdamW(m.parameters(), lr=lr); losses = []
    for step in range(1, steps + 1):
        scale = min(1.0, step / warmup) if warmup else 1.0                 # linear warm-up, then linear decay to zero
        for gr in opt.param_groups: gr["lr"] = lr * scale * (1 - step / (steps + 1))
        xb, yb = windows(train, torch.randint(0, len(train) - T - 1, (64,), generator=g).tolist())
        loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item() / math.log(2))
    early = float(np.mean(losses[:200])); worst = float(max(losses[200:]))
    return early, worst, bpc(m, val), bpc(m, test)

if __name__ == "__main__":
    t0 = time.time()
    with open("results11.jsonl", "w") as f:
        for lr in (3e-3, 1e-2):
            for post in (False, True):
                for warmup in (0, 200):
                    for seed in range(3):
                        early, worst, va, te = run(post, warmup, lr, seed)
                        f.write(json.dumps({"lr": lr, "post": post, "warmup": warmup, "seed": seed, "early": early, "worst": worst, "val": va, "test": te}) + "\n"); f.flush()
                        print(f"lr {lr:g}  {'post-LN' if post else 'pre-LN '}  warm-up {warmup:3d}  seed {seed}: mean loss of first 200 steps {early:.2f}, worst later batch {worst:.2f}, validation {va:.3f}, test {te:.3f}  ({time.time() - t0:.0f} s)", flush=True)
```

Output:

```
lr 0.003  pre-LN   warm-up   0  seed 0: mean loss of first 200 steps 3.63, worst later batch 3.27, validation 1.710, test 2.018  (36 s)
lr 0.003  pre-LN   warm-up   0  seed 1: mean loss of first 200 steps 3.65, worst later batch 3.31, validation 1.718, test 2.006  (70 s)
lr 0.003  pre-LN   warm-up   0  seed 2: mean loss of first 200 steps 3.63, worst later batch 3.28, validation 1.735, test 2.027  (105 s)
lr 0.003  pre-LN   warm-up 200  seed 0: mean loss of first 200 steps 4.16, worst later batch 3.58, validation 1.757, test 2.049  (139 s)
lr 0.003  pre-LN   warm-up 200  seed 1: mean loss of first 200 steps 4.15, worst later batch 3.60, validation 1.750, test 2.042  (173 s)
lr 0.003  pre-LN   warm-up 200  seed 2: mean loss of first 200 steps 4.14, worst later batch 3.63, validation 1.767, test 2.062  (208 s)
lr 0.003  post-LN  warm-up   0  seed 0: mean loss of first 200 steps 3.57, worst later batch 3.13, validation 1.718, test 2.010  (242 s)
lr 0.003  post-LN  warm-up   0  seed 1: mean loss of first 200 steps 3.59, worst later batch 3.15, validation 1.699, test 1.992  (275 s)
lr 0.003  post-LN  warm-up   0  seed 2: mean loss of first 200 steps 3.56, worst later batch 3.12, validation 1.724, test 2.020  (309 s)
lr 0.003  post-LN  warm-up 200  seed 0: mean loss of first 200 steps 4.12, worst later batch 3.57, validation 1.755, test 2.054  (342 s)
lr 0.003  post-LN  warm-up 200  seed 1: mean loss of first 200 steps 4.11, worst later batch 3.57, validation 1.740, test 2.032  (376 s)
lr 0.003  post-LN  warm-up 200  seed 2: mean loss of first 200 steps 4.10, worst later batch 3.59, validation 1.752, test 2.053  (409 s)
lr 0.01  pre-LN   warm-up   0  seed 0: mean loss of first 200 steps 3.28, worst later batch 2.63, validation 1.585, test 1.890  (444 s)
lr 0.01  pre-LN   warm-up   0  seed 1: mean loss of first 200 steps 3.32, worst later batch 2.71, validation 1.581, test 1.893  (478 s)
lr 0.01  pre-LN   warm-up   0  seed 2: mean loss of first 200 steps 3.30, worst later batch 2.72, validation 1.595, test 1.878  (512 s)
lr 0.01  pre-LN   warm-up 200  seed 0: mean loss of first 200 steps 3.82, worst later batch 3.10, validation 1.574, test 1.888  (547 s)
lr 0.01  pre-LN   warm-up 200  seed 1: mean loss of first 200 steps 3.82, worst later batch 3.10, validation 1.592, test 1.895  (580 s)
lr 0.01  pre-LN   warm-up 200  seed 2: mean loss of first 200 steps 3.81, worst later batch 3.08, validation 1.598, test 1.907  (615 s)
lr 0.01  post-LN  warm-up   0  seed 0: mean loss of first 200 steps 3.22, worst later batch 2.54, validation 1.570, test 1.880  (648 s)
lr 0.01  post-LN  warm-up   0  seed 1: mean loss of first 200 steps 3.28, worst later batch 2.59, validation 1.580, test 1.880  (683 s)
lr 0.01  post-LN  warm-up   0  seed 2: mean loss of first 200 steps 3.21, worst later batch 2.62, validation 1.596, test 1.888  (717 s)
lr 0.01  post-LN  warm-up 200  seed 0: mean loss of first 200 steps 3.77, worst later batch 2.98, validation 1.576, test 1.886  (751 s)
lr 0.01  post-LN  warm-up 200  seed 1: mean loss of first 200 steps 3.76, worst later batch 2.98, validation 1.582, test 1.886  (785 s)
lr 0.01  post-LN  warm-up 200  seed 2: mean loss of first 200 steps 3.74, worst later batch 2.98, validation 1.588, test 1.896  (818 s)
```

The summary, `analyse11.py`:

```python
import json
import numpy as np
R = [json.loads(l) for l in open("results11.jsonl")]
print("lr      arrangement  warm-up   test mean (std)    validation mean   mean loss, first 200 steps   worst later batch")
for lr in (3e-3, 1e-2):
    for post in (False, True):
        for w in (0, 200):
            r = [x for x in R if x["lr"] == lr and x["post"] == post and x["warmup"] == w]
            t = np.array([x["test"] for x in r])
            print(f"{lr:<7g} {'post-LN' if post else 'pre-LN':11}  {w:6d}   {t.mean():.3f} ({t.std(ddof=1):.3f})      {np.mean([x['val'] for x in r]):.3f}             {np.mean([x['early'] for x in r]):.2f}                        {np.mean([x['worst'] for x in r]):.2f}")
```

Output:

```
lr      arrangement  warm-up   test mean (std)    validation mean   mean loss, first 200 steps   worst later batch
0.003   pre-LN            0   2.017 (0.011)      1.721             3.63                        3.29
0.003   pre-LN          200   2.051 (0.010)      1.758             4.15                        3.60
0.003   post-LN           0   2.008 (0.014)      1.713             3.57                        3.14
0.003   post-LN         200   2.047 (0.012)      1.749             4.11                        3.58
0.01    pre-LN            0   1.887 (0.008)      1.587             3.30                        2.69
0.01    pre-LN          200   1.897 (0.009)      1.588             3.82                        3.09
0.01    post-LN           0   1.883 (0.004)      1.582             3.23                        2.58
0.01    post-LN         200   1.889 (0.006)      1.582             3.76                        2.98
```

## Try it

1. Change the number of layers from 2 to 6 in both arrangements and repeat the run at learning rate 1e-2. Does post-LN without warm-up now fall behind?
2. Try learning rates of 3e-2 and 1e-1 for post-LN with and without warm-up. At which value does the first run diverge?
3. Do the three passes on another short paper and write three sentences: the claim, the evidence, and the one assumption you would test first.
