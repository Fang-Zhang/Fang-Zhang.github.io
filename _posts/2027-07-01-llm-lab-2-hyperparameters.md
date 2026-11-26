---
title: "Improving LLM Performance, Measured (2/6): One Decade of Learning Rate Cost 0.45 Bits, Warmup Decided Whether Runs Diverged, and Seed Noise Was 0.03"
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

Setup: the same character-level GPT as Part 1 (4 layers, width 128, Pre-LN, under one million parameters), trained for 600 steps on 224 prose documents, AdamW with weight decay 0.01 and a cosine decay to 10 percent of the peak learning rate. Held-out loss is bits per character on 40 held-out documents. Four seeds per setting. Defaults unless varied: peak learning rate 0.003, batch 16 sequences of 128 characters, 100 warmup steps.

- **Learning rate** (0.0003, 0.001, 0.003, 0.01, 0.03): held-out bits 3.725, 3.164, **2.715**, 2.793 and 3.596. Going from 0.001 to 0.003 improved loss by 0.450 (p below 0.0001), and from 0.003 to 0.01 loss rose by 0.079 (p = 0.017). The seed standard deviation was 0.007 to 0.034 for the four stable settings. At 0.03 the four runs ended between 3.201 and 4.009 (standard deviation 0.359).
- **Warmup at learning rate 0.01**: with 100 warmup steps, 2.793 (4 runs finite). With no warmup, 2 of 4 runs diverged to a NaN loss and the other two ended at 4.769 and 4.775.
- **Batch size at a fixed 600 steps and learning rate 0.003** (batch 4, 16, 64, i.e. 0.3, 1.2 and 4.9 million characters seen): 3.489, 2.715 and 2.232. Both differences are significant (p below 0.0001).

![Three panels of held-out bits per character. Left: learning rate from 0.0003 to 0.03, a U shape with the minimum 2.72 at 0.003 and a wide spread at 0.03. Middle: at learning rate 0.01, 100 warmup steps gives 2.79 while no warmup gives 4.77 in two runs and diverges in the other two. Right: batch size 4, 16 and 64 at a fixed 600 steps gives 3.49, 2.72 and 2.23]({{ site.baseurl }}/assets/pic/llm-lab-2-hyperparameters.png){: width="900" }
_Figure: each dot is one seed; bars and lines are means. Measured here._

## Engineering reading

These points are this series' reading of the sources and the measurement.

1. **The learning rate dwarfs seed noise.** One decade of learning rate moved loss by 0.08 to 1.0 bits, while the seed standard deviation at stable settings was 0.03. Tuning the learning rate on one seed is reliable here; comparing two data or architecture variants whose effects are 0.03 bits is not, which is the problem Part 1 ran into.
2. **The optimum was narrow and asymmetric.** Loss rose slowly on the slow side (0.003 to 0.001: +0.45) and ended in instability on the fast side (0.03: spread 0.8 bits across seeds). A sweep should extend one step past the best value on the fast side to find the edge.
3. **Warmup mattered at a high learning rate.** At 0.01 the Pre-LN model without warmup diverged in half the runs and degraded in the rest. This does not contradict Xiong et al., whose claim concerns comparable results at their settings; it shows that on this model, warmup is the cheap insurance when the learning rate is near the edge.
4. **A batch-size comparison depends on what is held fixed.** At equal steps, batch 64 won by 0.48 bits over batch 16 because it saw four times the data. A fair comparison at equal tokens or compute needs different step counts, and the useful batch size has a ceiling, which is what the noise-scale work predicts.
5. **Hyperparameters interact.** The best learning rate at one batch size need not be the best at another; this experiment fixed the learning rate while varying the batch size and so cannot say where the optimum moves.

## Limits

- Small model, 600 steps, 4 seeds. The optimum learning rate depends on model size, batch size, schedule and training length, so the value 0.003 transfers to nothing outside this setup.
- The warmup test is at one learning rate and one warmup length; weight decay and the schedule shape were not varied.
- The batch-size arm varies data seen along with batch size (equal steps, not equal tokens); the linear scaling rule from Goyal et al. was not tested.
- Divergence is reported as the NaN loss at the end of the run; where in the run it happened was not recorded.
- Literature claims come from abstracts, and the results of those papers were not reproduced.

## Code and output

The training function from Part 1 gained a warmup argument (default 100, the same as before) and is otherwise unchanged:

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
        for gr in opt.param_groups: gr["lr"] = lr * min(1, (t + 1) / warmup) if warmup else 1 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
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
lr      0.0003   n=4 held-out bpc mean 3.725 sd 0.007 min 3.719 max 3.734
lr      0.001    n=4 held-out bpc mean 3.164 sd 0.028 min 3.125 max 3.190
lr      0.003    n=4 held-out bpc mean 2.715 sd 0.034 min 2.681 max 2.758
lr      0.01     n=4 held-out bpc mean 2.793 sd 0.034 min 2.769 max 2.842
lr      0.03     n=4 held-out bpc mean 3.596 sd 0.359 min 3.201 max 4.009
warmup  0        n=4 diverged (NaN loss) in 2 of 4 runs; finite runs: [4.769, 4.775]
warmup  100      n=4 held-out bpc mean 2.793 sd 0.034 min 2.769 max 2.842
batch   4        n=4 held-out bpc mean 3.489 sd 0.052 min 3.430 max 3.551
batch   16       n=4 held-out bpc mean 2.715 sd 0.034 min 2.681 max 2.758
batch   64       n=4 held-out bpc mean 2.232 sd 0.019 min 2.204 max 2.247
lr 0.003 minus lr 0.001: -0.450, Welch p=0.0000
lr 0.01 minus lr 0.003: +0.079, Welch p=0.0171
batch 4 minus batch 16 (lr 3e-3): +0.774, Welch p=0.0000
batch 64 minus batch 16 (lr 3e-3): -0.482, Welch p=0.0000
```

Next: Part 3 covers quantisation and pruning: how much accuracy a model gives up per bit removed.

## References

- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677), 2017.
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162), 2018.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), 2020.
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322), 2023.
