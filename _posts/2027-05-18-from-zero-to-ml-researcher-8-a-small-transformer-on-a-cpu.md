---
title: "From Zero to ML Researcher, Part 8: A Small Transformer on a CPU"
date: 2027-05-18 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, transformer, language model, layer normalization, pytorch, ML researcher]
description: "Part 8 of a 12-part series for complete beginners. A 110,384-parameter transformer, written in about 40 lines of PyTorch and trained in about 75 seconds on a CPU, reaches 1.91 bits per character on held-out text, against 2.19 for the best counting model, and generates text that looks like the Python documentation."
---

## What you will do, and what you get

You will assemble a complete character-level transformer: embeddings, two blocks of attention plus a small neural network with residual connections and layer normalisation, and a linear output layer. It has 110,384 parameters, trains in about 75 seconds on a CPU (3,000 steps of 64 windows of 64 characters), and reaches **1.911 bits per character on held-out text** (std 0.003 over 3 seeds). The best counting model of Part 6 reaches 2.187, the one-head model of Part 7 reached 2.97, and guessing costs 5.585. You will also sample text from it.

![Left: a transformer block drawn as boxes: layer norm, causal attention with 4 heads, an add node with a skip from the input, layer norm, an MLP 64 to 256 to 64, a second add node with a second skip, then layer norm and a linear layer producing 48 next-character scores; repeated 2 times. Middle: validation bits per character for three seeds falling from about 2.65 to about 1.6 over 3,000 steps, crossing the 1.89 level of 3-character counts near step 750. Right: test bits per character as bars: guessing 5.58, letter frequencies 4.65, bigram counts 3.39, 3-character counts 2.19, tiny transformer 1.91 with three seed dots]({{ site.baseurl }}/assets/pic/mlr-8-transformer.png){: width="900" }

## How: the steps

1. **Reuse the data** of Part 6: the same text, the same ordered 80/10/10 split, 48 characters. The window grows from 32 to 64 characters.
2. **Embed:** a learned vector per character plus a learned vector per position (as in Part 7).
3. **Build a block** of two sub-layers. Each one is `x = x + sublayer(layer_norm(x))`: first multi-head causal attention, then a two-layer network applied to each position separately (the MLP).
4. **Stack two blocks,** then a final layer norm and a linear layer to 48 scores.
5. **Train** with AdamW, learning rate 3e-3 decaying linearly to zero, 3,000 steps, batch of 64 random windows, cross-entropy loss.
6. **Evaluate fairly:** score only characters that had at least 32 characters of context, on validation during training and on the test block once at the end, for 3 seeds.
7. **Sample:** feed the model a prompt, draw the next character from its probabilities, append, repeat.

## Why it works: the principle

**Multi-head attention** runs several attentions of Part 7 side by side, each on a smaller slice of the vector (here 4 heads of 16 numbers). Different heads can look for different things, and their outputs are concatenated and mixed by one more linear layer. In the code, the three projections for all heads are computed by one linear layer, split, and reshaped into `(batch, head, time, head size)`.

**The MLP is where each position thinks on its own.** Attention moves information *between* positions; the MLP (`64 -> 256 -> 64`, with a GELU non-linearity) transforms the result at each position separately. The one-head model of Part 7 had no such layer. The structure of attention plus MLP, with residual connections, is the transformer of [Vaswani and colleagues (2017)](https://arxiv.org/abs/1706.03762).

**Residual connections** (`x = x + ...`) were introduced for deep image networks by [He and colleagues (2015)](https://arxiv.org/abs/1512.03385). Each sub-layer learns a *correction* to its input rather than a replacement, and the gradient can flow straight through the addition. Part 7 measured the effect in the small: adding the input back lowered the loss by 0.23 bits.

**Layer normalisation** ([Ba, Kiros and Hinton, 2016](https://arxiv.org/abs/1607.06450); [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)) rescales each position's vector to mean 0 and variance 1 across its features (then applies a learned scale and shift). It keeps the numbers in a range where training is stable. Placing it *before* each sub-layer, as here, is the "pre-LN" arrangement, which [Xiong and colleagues (2020)](https://arxiv.org/abs/2002.04745) analyse: they show that it makes training work without the learning-rate warm-up that the original post-LN arrangement needs.

**The optimiser** is AdamW ([Loshchilov and Hutter](https://arxiv.org/abs/1711.05101)), which adapts the step size of each parameter and is a common default for transformers. Part 3 used the plain gradient step, and the idea is the same.

**Reading the results.**
- The validation curve crosses the 3-character count level (1.89) at about step 750 and keeps falling to about 1.6. A parameterised model with a window of 64 characters does what the table of Part 6 could not.
- Test is 1.911, validation about 1.60. The gap is *not* a sign of overfitting by itself: the counting model shows the same gap (validation 1.886, test 2.187), because the last block of the text is harder (Part 6). Both models are scored on that same block.
- The three seeds finish within 0.008 of each other on test, so the ranking against counting (a gap of 0.28) is far larger than seed noise.
- The sample copies the *style* of the documentation, with `object.__new__(self, other)` and phrases like `define operators`, and invents words (`fourning`, `appresentation`). A 110k-parameter model trained for 75 seconds has learned spelling patterns and fragments, not meaning.

**Pitfalls and limits.**
- **The comparison is not perfectly like for like.** The counting models were scored on every character with their short context (1 to 5 characters); the transformer is scored only on characters with at least 32 characters of context. The counting models ignore anything older than 5 characters, so extra context could not have helped them, but state the difference when you quote the numbers.
- The `train` number printed (about 1.6) is the loss of the *last batch*, so it is noisy; do not read it as the training loss of the whole set.
- Learning rate, width, depth and the number of steps were set once and not tuned; a different choice would change the numbers. Part 9 measures how much of a difference counts as real.
- On 439,000 characters, bigger models will start to memorise. Part 10 studies how loss changes with model size.

## Example, worked through

Layer normalisation of one position with four features `(2, 4, 6, 8)`: the mean is 5, the variance is 5, so the standard deviation is 2.236 and the output is `(-1.342, -0.447, 0.447, 1.342)`. The second half of the output counts the parameters of the model by formula: embeddings `48*64 + 64*64`, two blocks of 49,984 each (two layer norms, the combined `qkv` layer, the output projection, the two MLP layers), a final layer norm and the output layer.

```python
import numpy as np, torch
x = np.array([2., 4., 6., 8.])
y = (x - x.mean()) / np.sqrt(x.var() + 1e-5)             # layer norm: mean 0, variance 1 across the features of ONE position
print("by hand :", y.round(4), " mean", y.mean().round(6), " std", y.std().round(4))
print("PyTorch :", torch.nn.LayerNorm(4)(torch.tensor(x, dtype=torch.float32)).detach().numpy().round(4))
V, D, T = 48, 64, 64                                      # parameter count of the 2-layer model, by formula
emb = V * D + T * D; block = 2 * 2 * D + (D * 3 * D + 3 * D) + (D * D + D) + (D * 4 * D + 4 * D) + (4 * D * D + D)
total = emb + 2 * block + 2 * D + (D * V + V)
print(f"embeddings {emb:,} + 2 blocks x {block:,} + final norm {2 * D} + output layer {D * V + V:,} = {total:,}")
```

Output:

```
by hand : [-1.3416 -0.4472  0.4472  1.3416]  mean 0.0  std 1.0
PyTorch : [-1.3416 -0.4472  0.4472  1.3416]
embeddings 7,168 + 2 blocks x 49,984 + final norm 128 + output layer 3,120 = 110,384
```

## Implementation

Save as `part8.py` and run it (about 4 minutes on a CPU for the 3 seeds; one seed took 75 seconds). It needs numpy and PyTorch. Training prints one line per seed; the whole script also saves the curves for the figure.

```python
import re, math, time, pickle
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import pydoc_data.topics as topics

text = "\n".join(topics.topics[k] for k in sorted(topics.topics)).lower()
text = re.sub(r"[^a-z0-9 .,\n()'\"=:_-]", " ", text); text = re.sub(r" +", " ", text)
chars = sorted(set(text)); V = len(chars); ids = torch.tensor([chars.index(c) for c in text])
n = len(ids); train, val, test = ids[: int(.8 * n)], ids[int(.8 * n): int(.9 * n)], ids[int(.9 * n):]
T, D, H, L = 64, 64, 4, 2                                   # window, width, heads, layers

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D)                      # all heads at once
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x):
        B, t, _ = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(D, dim=2)
        q, k, v = (a.view(B, t, H, D // H).transpose(1, 2) for a in (q, k, v))     # (batch, head, time, head size)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)               # causal attention, as in Part 7
        x = x + s.proj(y.transpose(1, 2).reshape(B, t, D))                         # residual 1: attention
        return x + s.mlp(s.ln2(x))                                                 # residual 2: per-position network

class Tiny(nn.Module):
    def __init__(s, layers=L):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.Sequential(*[Block() for _ in range(layers)]); s.ln = nn.LayerNorm(D); s.out = nn.Linear(D, V)
    def forward(s, idx):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1]))
        return s.out(s.ln(s.blocks(x)))

def windows(data, starts): return (torch.stack([data[j:j + T] for j in starts]), torch.stack([data[j + 1:j + T + 1] for j in starts]))

@torch.no_grad()
def bpc(model, data, min_context=T // 2):                   # score only characters that had at least min_context characters of context
    tot = cnt = 0
    starts = list(range(0, len(data) - T - 1, T - min_context))
    for s in range(0, len(starts), 256):
        xb, yb = windows(data, starts[s:s + 256]); lo = F.cross_entropy(model(xb)[:, min_context - 1:].reshape(-1, V), yb[:, min_context - 1:].reshape(-1), reduction="sum")
        tot += lo.item(); cnt += yb[:, min_context - 1:].numel()
    return tot / cnt / math.log(2)

def run(seed, layers=L, steps=3000, lr=3e-3):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = Tiny(layers)
    opt = torch.optim.AdamW(m.parameters(), lr=lr); curve = []
    for step in range(1, steps + 1):
        for gr in opt.param_groups: gr["lr"] = lr * (1 - step / (steps + 1))        # linear decay to 0
        xb, yb = windows(train, torch.randint(0, len(train) - T - 1, (64,), generator=g).tolist())
        loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0: curve.append((step, loss.item() / math.log(2), bpc(m, val)))
    return m, curve

@torch.no_grad()
def sample(m, prompt, length, seed, temperature=1.0):
    g = torch.Generator().manual_seed(seed); x = torch.tensor([[chars.index(c) for c in prompt]])
    for _ in range(length):
        p = F.softmax(m(x[:, -T:])[0, -1] / temperature, -1); x = torch.cat([x, torch.multinomial(p, 1, generator=g)[None]], 1)
    return "".join(chars[i] for i in x[0])

if __name__ == "__main__":
    print(f"parameters in the 2-layer model: {sum(p.numel() for p in Tiny().parameters()):,}")
    t0 = time.time(); results = {}
    for seed in range(3):
        m, curve = run(seed); results[seed] = (m, curve)
        print(f"seed {seed}: train {curve[-1][1]:.3f}  validation {curve[-1][2]:.3f}  test {bpc(m, test):.3f} bits/char   ({time.time() - t0:.0f} s elapsed)", flush=True)
    te = [bpc(results[s][0], test) for s in range(3)]; va = [results[s][1][-1][2] for s in range(3)]
    print(f"validation mean {np.mean(va):.3f} (std {np.std(va):.3f}); test mean {np.mean(te):.3f} (std {np.std(te):.3f})")
    print("for comparison, counting models from Part 6 (test): 3 characters 2.187, 4 characters 2.212, bigram 3.393")
    m = results[0][0]
    print("\nsample (temperature 1.0):\n" + repr(sample(m, "the ", 240, 0)))
    pickle.dump({s: results[s][1] for s in results}, open("curves8.pkl", "wb")); torch.save(m.state_dict(), "tiny8.pt")
```

Output:

```
parameters in the 2-layer model: 110,384
seed 0: train 1.586  validation 1.585  test 1.916 bits/char   (75 s elapsed)
seed 1: train 1.605  validation 1.596  test 1.908 bits/char   (148 s elapsed)
seed 2: train 1.635  validation 1.622  test 1.911 bits/char   (223 s elapsed)
validation mean 1.601 (std 0.016); test mean 1.911 (std 0.003)
for comparison, counting models from Part 6 (test): 3 characters 2.187, 4 characters 2.212, bigram 3.393

sample (temperature 1.0):
'the terminates.\n\n fourning: \'gett\'.format(rtyleter, clding and the class with\npositing. set to that doct names an "a "appresentation.".\n\nobject.__new__(self, other)\noperator ("x") and\n xy" ", except "integerror" define operators, it subclass im'
```

## Try it

1. Change `L` to 1, then 4, and `D` to 32, then 128. Which change buys more for the time it costs?
2. Sample at temperature 0.5 and at 1.5 (`sample(m, "the ", 240, 0, temperature=0.5)`). Describe what each does to the text, and why.
3. Evaluate the trained model with `min_context=1` instead of 32. Does the number change, and what does that say about the context window?
