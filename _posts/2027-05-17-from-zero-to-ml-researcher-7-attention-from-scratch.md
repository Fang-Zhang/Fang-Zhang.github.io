---
title: "From Zero to ML Researcher, Part 7: Attention From Scratch"
date: 2027-05-17 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, attention, transformer, language model, pytorch, ML researcher]
description: "Part 7 of a 12-part series for complete beginners. Causal self-attention in about 8 lines of numpy, checked against PyTorch, then trained as a character-level language model: 2.97 bits per character on held-out text, with a measured effect of position information and of a residual connection."
---

## What you will do, and what you get

You will write *self-attention*, the core operation of the transformer, in about 8 lines of numpy and confirm that it matches PyTorch's built-in version (largest difference 2.2e-16). Then you will train a character-level language model whose only mechanism is one attention head, on the text and the split of Part 6, with three variants and three random seeds each. Results on held-out text, in bits per character (lower is better): 3.34 without position information, 3.20 with it, 2.97 with it plus a residual connection (about 10 seconds per run on a CPU). That is better than the bigram counts of Part 6 (3.39) and still far from the 3-character counts (2.19). Reporting that gap honestly is the point; Part 8 closes it.

![Left: a toy 4 by 4 attention weight matrix with the upper triangle marked masked; each row sums to 1. Middle: the attention weights of a trained one-head model on the sentence the value of the expression is, strongly concentrated on the diagonal and just below it. Right: validation bits per character against training step for three variants with 3 seeds each: no positions about 3.17, positions without residual about 3.0, positions plus residual about 2.74, with the bigram and 3-character count levels as dashed lines]({{ site.baseurl }}/assets/pic/mlr-7-attention.png){: width="900" }

## How: the steps

1. **Give every position three vectors** computed from its input `x`: a *query* `q = x Wq` ("what am I looking for"), a *key* `k = x Wk` ("what do I contain") and a *value* `v = x Wv` ("what I hand over if chosen").
2. **Score** every query against every key: `scores = q k^T / sqrt(d)`.
3. **Mask the future:** set the scores of later positions to minus infinity, so that predicting character `t` cannot read characters after `t`.
4. **Softmax each row** into weights that sum to 1.
5. **Average the values** with those weights: `output = weights @ v`.
6. **Add positions:** attention alone treats the context as an unordered bag, so add a learned vector for each position to the character embedding.
7. **Stack it into a model:** embedding, attention, a linear layer to 48 scores; train with the cross-entropy of Part 5.

## Why it works: the principle

**Attention is a soft lookup.** Each position builds a query and compares it with the keys of all positions it may read; the better the match, the larger the weight, and the output is the weighted average of the values. The idea of learned weighted reading appeared for translation in [Bahdanau, Cho and Bengio (2014)](https://arxiv.org/abs/1409.0473), and the form used here is the scaled dot-product attention of [Vaswani and colleagues (2017)](https://arxiv.org/abs/1706.03762): `softmax(Q K^T / sqrt(d_k)) V`. Division by `sqrt(d_k)` is there because, as the paper explains, large dot products push the softmax into regions with tiny gradients.

**Every step is a matrix product,** so a whole text window is processed at once, in contrast to the counting table of Part 6, which stored one entry per context. The parameters (`Wq`, `Wk`, `Wv` and the embeddings) are shared across all positions, which is how the model gets a long context without an exponentially large table.

**The mask is what makes it a language model.** Without it, position 5 could read the answer at position 6 and the loss would be trivially small. The toy matrix in the figure shows the mask: the upper triangle receives weight 0, and each row still sums to 1.

**What the three variants test.** Each variant removes or adds one ingredient, and the others stay fixed, so the difference is attributable to it:
- *No positions:* the model sees which characters are in the window but not where. Test result 3.343 (std 0.007 over 3 seeds), about the same as the bigram counts of Part 6.
- *Positions, no residual:* 3.201 (std 0.005).
- *Positions + residual:* 2.973 (std 0.005). A residual connection adds the input `x` to the attention output, so the current character reaches the output directly instead of only through a weighted average. Seeing the current character matters a great deal when predicting the next.

The spread between seeds (0.005) is much smaller than the gaps between variants (0.14 and 0.23), so on this task the ordering is not a seed accident. The differences are measured for this one corpus, this learning rate (3e-3) and 3,000 steps of 64 windows.

**Reading the attention picture.** The trained one-head model puts most weight on the current character and the one or two before it; it behaves like a smarter bigram. One head with no further layers cannot do much more, which fits the 2.97 against 2.19 for the 3-character table. The transformer block of Part 8 adds a per-position network, normalisation and several heads and layers.

**Pitfalls and limits.**
- The model reads at most `T = 32` characters. Early positions in each window have less context than the counting models had, which makes this comparison *slightly unfavourable* to attention. A fair final comparison needs identical evaluation, which Part 8 sets up.
- Cross-entropy here is averaged over all 32 positions of every window, including those with almost no context.
- Training for 3,000 steps is short. The validation curves in the figure are flat for the last 1,000 steps, but a different learning rate could still change the numbers.
- The cost of attention grows with the square of the window length, which is why long contexts are expensive.

## Example, worked through

Three positions, two-dimensional vectors, position 2 doing the asking. Queries and keys are `(1,0)`, `(0,1)`, `(1,1)`; values `(1,0)`, `(0,2)`, `(2,2)`.

- Scores of position 2 against positions 0, 1, 2: `(1, 1, 2) / sqrt(2) = (0.7071, 0.7071, 1.4142)`.
- Softmax: `(0.2483, 0.2483, 0.5035)`. The largest score takes about half of the weight, and the weights sum to 1.
- Output: `0.2483 * (1,0) + 0.2483 * (0,2) + 0.5035 * (2,2) = (1.2552, 1.5035)`.

```python
import numpy as np
q = np.array([[1., 0], [0, 1], [1, 1]]); k = q.copy(); v = np.array([[1., 0], [0, 2], [2, 2]])
scores = q[2] @ k.T / np.sqrt(2)                  # position 2 asks: how well do I match each key?
w = np.exp(scores) / np.exp(scores).sum()
print("scores", scores.round(4)); print("weights", w.round(4), "sum", w.sum().round(4)); print("output", (w @ v).round(4))
```

Output:

```
scores [0.7071 0.7071 1.4142]
weights [0.2483 0.2483 0.5035] sum 1.0
output [1.2552 1.5035]
```

## Implementation

Save as `part7.py` and run it (a couple of minutes on a CPU). It needs numpy and PyTorch (the CPU version is fine). The text is the same Python reference text and the same split as in Part 6, rebuilt here so that the file stands alone.

```python
import re, math, sys, time
import numpy as np, torch
import torch.nn.functional as F
import pydoc_data.topics as topics

# ---- 1. Attention in plain numpy, on a toy input -------------------------------------------
def softmax(s): e = np.exp(s - s.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def attention(x, Wq, Wk, Wv):
    q, k, v = x @ Wq, x @ Wk, x @ Wv                       # every position makes a query, a key and a value
    scores = q @ k.T / np.sqrt(k.shape[-1])                # how well each query matches each key
    scores = np.where(np.tril(np.ones_like(scores)) == 1, scores, -np.inf)   # causal mask: no looking at the future
    w = softmax(scores)                                    # each row sums to 1
    return w @ v, w                                        # each output is a weighted average of the values

r = np.random.default_rng(0)
x = r.normal(size=(4, 8)); Wq, Wk, Wv = (r.normal(size=(8, 8)) / np.sqrt(8) for _ in range(3))
out, w = attention(x, Wq, Wk, Wv)
np.set_printoptions(precision=3, suppress=True)
print("attention weights (row = position that is asking, column = position it reads):\n", w)
print("row sums:", w.sum(1))
t = torch.tensor
ref = F.scaled_dot_product_attention(t(x @ Wq)[None], t(x @ Wk)[None], t(x @ Wv)[None], is_causal=True)[0].numpy()
print(f"largest difference to PyTorch's built-in attention: {np.abs(ref - out).max():.1e}\n")

# ---- 2. A character-level model whose only mechanism is one attention head ------------------
text = "\n".join(topics.topics[k] for k in sorted(topics.topics)).lower()
text = re.sub(r"[^a-z0-9 .,\n()'\"=:_-]", " ", text); text = re.sub(r" +", " ", text)
chars = sorted(set(text)); V = len(chars); ids = torch.tensor([chars.index(c) for c in text])
n = len(ids); train, val, test = ids[: int(.8 * n)], ids[int(.8 * n): int(.9 * n)], ids[int(.9 * n):]
T, D = 32, 64                                              # context window and embedding size

class OneHead(torch.nn.Module):
    def __init__(s, use_pos, resid=False):
        super().__init__(); s.use_pos = use_pos; s.resid = resid
        s.tok = torch.nn.Embedding(V, D); s.pos = torch.nn.Embedding(T, D)
        s.q, s.k, s.v = (torch.nn.Linear(D, D, bias=False) for _ in range(3)); s.out = torch.nn.Linear(D, V)
    def forward(s, idx, return_w=False):
        x = s.tok(idx) + (s.pos(torch.arange(idx.shape[1])) if s.use_pos else 0)
        q, k, v = s.q(x), s.k(x), s.v(x)
        sc = q @ k.transpose(1, 2) / math.sqrt(D)
        sc = sc.masked_fill(torch.tril(torch.ones(idx.shape[1], idx.shape[1])) == 0, float("-inf"))
        w = sc.softmax(-1); y = s.out(w @ v + (x if s.resid else 0))     # resid: also hand the input straight to the output
        return (y, w) if return_w else y

def batch(data, g, size=64):
    i = torch.randint(0, len(data) - T - 1, (size,), generator=g)
    return torch.stack([data[j:j + T] for j in i]), torch.stack([data[j + 1:j + T + 1] for j in i])

@torch.no_grad()
def bpc(model, data):                                      # bits per character, using every full window in the block
    tot = 0.0; cnt = 0
    for s in range(0, len(data) - T - 1, T * 64):
        xs = torch.stack([data[j:j + T] for j in range(s, min(s + T * 64, len(data) - T - 1), T)])
        ys = torch.stack([data[j + 1:j + T + 1] for j in range(s, min(s + T * 64, len(data) - T - 1), T)])
        tot += F.cross_entropy(model(xs).reshape(-1, V), ys.reshape(-1), reduction="sum").item(); cnt += ys.numel()
    return tot / cnt / math.log(2)

def run(cfg, seed, steps=3000):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = OneHead(*cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3); curve = []
    for step in range(1, steps + 1):
        xb, yb = batch(train, g); loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0: curve.append((step, bpc(m, val)))
    return m, curve

res = {}
t0 = time.time()
cfgs = {"positions, no residual": (True, False), "no positions": (False, False), "positions + residual": (True, True)}
for name, cfg in cfgs.items():
    for seed in range(3):
        m, curve = run(cfg, seed); res[(name, seed)] = (m, curve)
        print(f"{name:24}  seed {seed}: validation {curve[-1][1]:.3f} bits/char   ({time.time() - t0:.0f} s elapsed)", flush=True)
for name in cfgs:
    v = [res[(name, s)][1][-1][1] for s in range(3)]; te = [bpc(res[(name, s)][0], test) for s in range(3)]
    print(f"{name:24}: validation mean {np.mean(v):.3f} (std {np.std(v):.3f}), test mean {np.mean(te):.3f} (std {np.std(te):.3f})")
print("for comparison, from Part 6: 3-character counting model 2.187 on test, bigram 3.393, guessing 5.585")

import pickle; pickle.dump({"curves": {f"{k[0]}|{k[1]}": v[1] for k, v in res.items()}}, open("curves7.pkl", "wb"))
m = res[("positions, no residual", 0)][0]; s = "the value of the expression is"
xi = torch.tensor([[chars.index(c) for c in s]]); _, wt = m(xi, return_w=True)
np.save("attn7.npy", wt[0].detach().numpy()); open("attn7.txt", "w").write(s)
```

Output:

```
attention weights (row = position that is asking, column = position it reads):
 [[1.    0.    0.    0.   ]
 [0.995 0.005 0.    0.   ]
 [0.309 0.34  0.351 0.   ]
 [0.216 0.278 0.211 0.295]]
row sums: [1. 1. 1. 1.]
largest difference to PyTorch's built-in attention: 2.2e-16

positions, no residual    seed 0: validation 3.009 bits/char   (11 s elapsed)
positions, no residual    seed 1: validation 3.003 bits/char   (21 s elapsed)
positions, no residual    seed 2: validation 3.003 bits/char   (31 s elapsed)
no positions              seed 0: validation 3.175 bits/char   (43 s elapsed)
no positions              seed 1: validation 3.175 bits/char   (53 s elapsed)
no positions              seed 2: validation 3.158 bits/char   (62 s elapsed)
positions + residual      seed 0: validation 2.741 bits/char   (72 s elapsed)
positions + residual      seed 1: validation 2.747 bits/char   (83 s elapsed)
positions + residual      seed 2: validation 2.742 bits/char   (93 s elapsed)
positions, no residual  : validation mean 3.005 (std 0.003), test mean 3.201 (std 0.005)
no positions            : validation mean 3.169 (std 0.008), test mean 3.343 (std 0.007)
positions + residual    : validation mean 2.744 (std 0.003), test mean 2.973 (std 0.005)
for comparison, from Part 6: 3-character counting model 2.187 on test, bigram 3.393, guessing 5.585
```

## Try it

1. Set `D` to 16, then 256. How do the three variants move?
2. Remove the `/ math.sqrt(D)` scaling and compare the validation curves.
3. Remove the causal mask and watch the loss. Then explain why that number cannot be trusted.
