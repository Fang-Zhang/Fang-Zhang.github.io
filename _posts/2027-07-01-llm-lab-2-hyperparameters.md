---
title: "Improving LLM Performance, Measured (2/6): The Best Learning Rate Sat at 0.01, Skipping Warmup Cost 0.23 Bits, and Seed Noise Was 0.05"
date: 2027-07-01 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, optimisation, learning rate, warmup, batch size, hyperparameters, series]
description: "Part 2 of 6. Learning rate, warmup and batch size swept on a small language model with four seeds per setting: how large each effect is against seed noise, and what the public papers say about each."
---

## What the literature says

Optimiser settings are the cheapest lever on this list and the one most often tuned by feel. Four public sources describe what to expect.

- **Warmup and batch size.** [Goyal et al.](https://arxiv.org/abs/1706.02677) report no loss of accuracy with ImageNet minibatches up to 8192 images, using "a hyper-parameter-free linear scaling rule for adjusting learning rates as a function of minibatch size" and "a new warmup scheme that overcomes optimization challenges early in training".
- **Where the useful batch size ends.** [McCandlish et al.](https://arxiv.org/abs/1812.06162) show that "a simple and easy-to-measure statistic called the gradient noise scale predicts the largest useful batch size across many domains", and that the noise scale grows as the loss decreases during training.
- **Why warmup exists.** [Xiong et al.](https://arxiv.org/abs/2002.04745) prove for the original Post-LN Transformer that the expected gradients near the output layer are large at initialisation, so "using a large learning rate on those gradients makes the training unstable", which warmup avoids. For Pre-LN Transformers they report comparable results without warmup.
- **Learning-rate sensitivity at small scale.** [Wortsman et al.](https://arxiv.org/abs/2309.14322) show that instabilities reported at large scale "also appear in small models when training at high learning rates", and study warm-up, weight decay and muParam as ways to make the final loss less sensitive to the learning rate.

The experiment below sweeps the three settings on a small model and measures each effect against the seed-to-seed spread. All claims above come from abstracts.

## Experiment: learning rate, warmup, batch size

Setup: the same character-level GPT as Part 1 (4 layers, width 128, Pre-LN, under one million parameters), trained for 600 steps on 224 prose documents, AdamW with weight decay 0.01, linear warmup and then a cosine decay to 10 percent of the peak learning rate. Held-out loss is bits per character on 40 held-out documents. Four seeds per setting. Defaults unless varied: peak learning rate 0.003 (a grid point, not the best one, as it turned out), batch 16 sequences of 128 characters, 100 warmup steps.

- **Learning rate** (0.0003, 0.001, 0.003, 0.01, 0.03): held-out bits 3.808, 3.563, 2.933, **2.825** and 2.952. Going from 0.001 to 0.003 improved loss by 0.630 (p = 0.0001), and from 0.003 to 0.01 by a further 0.108 (p = 0.024). At 0.03 the mean was 0.126 worse than at 0.01 (p = 0.26), with a spread of 2.840 to 3.221 (standard deviation 0.181) against 0.05 at 0.003 and 0.01.
- **Warmup at learning rate 0.01**: with 100 warmup steps 2.825, without warmup 3.059 (four runs from 2.971 to 3.162). Skipping warmup cost 0.233 bits (p = 0.009). No run diverged.
- **Batch size at a fixed 600 steps and learning rate 0.003** (batch 4, 16, 64, i.e. 0.3, 1.2 and 4.9 million characters seen): 3.592, 2.933 and 2.405. Both differences are significant (p = 0.0001).

![Three panels of held-out bits per character. Left: learning rate from 0.0003 to 0.03, falling from 3.81 to a minimum of 2.83 at 0.01 and rising to 2.95 at 0.03 with a wide spread. Middle: at learning rate 0.01, 100 warmup steps gives 2.83 and no warmup gives 3.06. Right: batch size 4, 16 and 64 at a fixed 600 steps gives 3.59, 2.93 and 2.41]({{ site.baseurl }}/assets/pic/llm-lab-2-hyperparameters.png){: width="900" }
_Figure: each dot is one seed; bars and lines are means. Measured here._

## Engineering reading

These points are this series' reading of the sources and the measurement.

1. **The learning rate dwarfs seed noise, until the edge.** Between 0.0003 and 0.01 loss moved by about 1 bit while the seed standard deviation was 0.002 to 0.05. Tuning the learning rate on one seed is reliable here; comparing two data or architecture variants whose effects are 0.03 bits is not, which is the problem Part 1 ran into.
2. **The default grid point was not the optimum.** A learning rate of 0.003 would have been a reasonable guess, yet 0.01 was 0.108 bits better. A sweep should continue until loss turns upward; here the upturn is visible only at 0.03, where the mean was worse but the difference from 0.01 was not significant at 4 seeds, and the spread across seeds had tripled.
3. **Warmup helped at a high learning rate, as a gain rather than as insurance.** At 0.01 the model trained without warmup, and ended 0.233 bits worse. This is consistent with the direction of Xiong et al. and Wortsman et al., but the model here is Pre-LN and no run diverged. The mechanism (unstable early gradients) was not measured.
4. **A batch-size comparison depends on what is held fixed.** At equal steps, batch 64 won by 0.53 bits over batch 16 because it saw four times the data. A fair comparison at equal tokens or compute needs different step counts, and the useful batch size has a ceiling, which is what the noise-scale work predicts.
5. **Hyperparameters interact.** The learning rate was varied at batch 16, and the batch size at learning rate 0.003, which turned out not to be the optimum. The optimum for one setting need not be the optimum for the other, and this design cannot say where it moves.

## Limits

- Small model, 600 steps, 4 seeds. The optimum learning rate depends on model size, batch size, schedule and training length, so the value 0.003 transfers to nothing outside this setup.
- The warmup test is at one learning rate and one warmup length; weight decay and the schedule shape were not varied.
- A first version of this experiment had a bug in the schedule code (the cosine decay was dropped when warmup was on, and the no-warmup arm ran at a far higher learning rate). The experiment was re-run after the fix and the numbers here are from the corrected run.
- The batch-size arm varies data seen along with batch size (equal steps, not equal tokens); the linear scaling rule from Goyal et al. was not tested.
- Literature claims come from abstracts, and the results of those papers were not reproduced.

## Code and output

The training function from Part 1 gained a warmup argument (default 100, the same behaviour as before):

```python
import glob, os, re, math, random, time, torch, torch.nn as nn, torch.nn.functional as F
POSTS = "posts/*.md"
V, CTX = 128, 128
def ascii_only(s): return "".join(c for c in s if c == "\n" or 32 <= ord(c) < 127)
def load_prose():
    docs = []
    for f in sorted(glob.glob(POSTS)):
        s = open(f).read(); s = re.sub(r"^---.*?---", "", s, flags=re.S); s = re.sub(r"```.*?```", "", s, flags=re.S)
        s = ascii_only(s)
        if len(s) > 2000: docs.append(s)
    return docs
def load_code():
    docs = []
    for f in sorted(glob.glob(os.path.dirname(os.__file__) + "/*.py")):
        s = ascii_only(open(f, errors="ignore").read())
        if len(s) > 2000: docs.append(s)
    return docs
def split(docs, seed=0):
    r = random.Random(seed); idx = list(range(len(docs))); r.shuffle(idx); k = int(.85 * len(idx))
    return [docs[i] for i in idx[:k]], [docs[i] for i in idx[k:]]
def enc(s): return torch.tensor([ord(c) for c in s], dtype=torch.long).clamp(max=V - 1)
class Block(nn.Module):
    def __init__(s, D, H):
        super().__init__(); s.H = H; s.l1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D); s.l2 = nn.LayerNorm(D)
        s.m = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x):
        B, T, D = x.shape; q, k, v = s.qkv(s.l1(x)).view(B, T, 3, s.H, D // s.H).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)); return x + s.m(s.l2(x))
class GPT(nn.Module):
    def __init__(s, D=128, H=4, L=4):
        super().__init__(); s.e = nn.Embedding(V, D); s.p = nn.Embedding(CTX, D); s.b = nn.ModuleList(Block(D, H) for _ in range(L)); s.f = nn.LayerNorm(D); s.h = nn.Linear(D, V)
    def forward(s, ids):
        x = s.e(ids) + s.p(torch.arange(ids.shape[1]))
        for b in s.b: x = b(x)
        return s.h(s.f(x))
def batch(sources, probs, B, g):
    """sources: list of 1-D long tensors; each row drawn from a source chosen with probs."""
    xs, ys = [], []
    for _ in range(B):
        src = sources[torch.multinomial(probs, 1, generator=g).item()]
        i = torch.randint(0, len(src) - CTX - 1, (1,), generator=g).item(); xs.append(src[i:i + CTX]); ys.append(src[i + 1:i + CTX + 1])
    return torch.stack(xs), torch.stack(ys)
@torch.no_grad()
def evaluate(m, src, n=200):
    g = torch.Generator().manual_seed(123); m.eval(); tot = 0
    for _ in range(n // 50):
        x, y = batch([src], torch.ones(1), 50, g); tot += F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item()
    m.train(); return tot / (n // 50) / math.log(2)   # bits per character
def train(sources, probs, seed, steps=2000, B=32, lr=2e-3, evals=(), warmup=100):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = GPT(); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01)
    probs = torch.tensor(probs, dtype=torch.float)
    for t in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * (min(1, (t + 1) / warmup) if warmup else 1) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
        x, y = batch(sources, probs, B, g); loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return [evaluate(m, e) for e in evals]
if __name__ == "__main__":
    torch.set_num_threads(2); P = load_prose(); C = load_code(); ptr, pva = split(P); ctr, cva = split(C)
    print("prose docs", len(P), "chars", sum(map(len, P)), "| train/val", len(ptr), len(pva), "| code files", len(C), "chars", sum(map(len, C)))
    t = time.time(); r = train([enc("".join(ptr))], [1.0], 0, steps=200, evals=[enc("".join(pva))]); print("200 steps", time.time() - t, r)
```

Experiment definition (each of 10 single-thread processes runs a share of the jobs):

```python
import sys, json, torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
import lm
P = lm.load_prose(); PTR, PVA = lm.split(P); TR = lm.enc("".join(PTR)); VA = lm.enc("".join(PVA)); STEPS = 600
ARMS = [("lr", l, dict(lr=l, B=16, warmup=100)) for l in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)] \
     + [("warmup", w, dict(lr=1e-2, B=16, warmup=w)) for w in (0, 100)] \
     + [("batch", b, dict(lr=3e-3, B=b, warmup=100)) for b in (4, 16, 64)]
JOBS = [(i, s) for s in range(4) for i in range(len(ARMS))]
if __name__ == "__main__":
    k, n = int(sys.argv[1]), int(sys.argv[2])
    for j, (i, s) in enumerate(JOBS):
        if j % n == k:
            kind, val, kw = ARMS[i]; bpc = lm.train([TR], [1.0], s, steps=STEPS, evals=[VA], **kw)[0]
            open(f"f2_shard_{k}.jsonl", "a").write(json.dumps([kind, val, s, bpc]) + "\n")
```

Statistics output:

```
runs: 40
lr      0.0003   n=4 held-out bpc mean 3.808 sd 0.002 min 3.805 max 3.811
lr      0.001    n=4 held-out bpc mean 3.563 sd 0.008 min 3.555 max 3.572
lr      0.003    n=4 held-out bpc mean 2.933 sd 0.050 min 2.868 max 2.979
lr      0.01     n=4 held-out bpc mean 2.825 sd 0.052 min 2.783 max 2.900
lr      0.03     n=4 held-out bpc mean 2.952 sd 0.181 min 2.840 max 3.221
warmup  0        n=4 held-out bpc mean 3.059 sd 0.095 min 2.971 max 3.162
warmup  100      n=4 held-out bpc mean 2.825 sd 0.052 min 2.783 max 2.900
batch   4        n=4 held-out bpc mean 3.592 sd 0.006 min 3.585 max 3.601
batch   16       n=4 held-out bpc mean 2.933 sd 0.050 min 2.868 max 2.979
batch   64       n=4 held-out bpc mean 2.405 sd 0.010 min 2.394 max 2.415
lr 0.003 minus lr 0.001: -0.630, Welch p=0.0001
lr 0.01 minus lr 0.003: -0.108, Welch p=0.0238
lr 0.03 minus lr 0.01: +0.126, Welch p=0.2611
warmup 0 minus warmup 100 (lr 1e-2): +0.233, Welch p=0.0089
batch 4 minus batch 16 (lr 3e-3): +0.659, Welch p=0.0001
batch 64 minus batch 16 (lr 3e-3): -0.528, Welch p=0.0001
```

Next: Part 3 covers quantisation and pruning: how much accuracy a model gives up per bit removed.

## References

- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677), 2017.
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162), 2018.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), 2020.
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322), 2023.
