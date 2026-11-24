---
title: "Inside the Math, Part 1: Softmax, Temperature and the Overflow Trap"
date: 2027-05-23 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, softmax, temperature, numerical stability, transformer, inside the math]
description: "Part 1 of an 8-part series on the classic maths inside language models. Softmax turns scores into probabilities; temperature sets how sharp they are; and the textbook formula overflows to nan once every score is shifted by 86 in float32. Everything runs on a small trained transformer on a CPU."
---

## What you will do, and what you get

You will take the 48 next-character scores of a trained 110,384-parameter transformer and turn them into probabilities with softmax, three ways: the textbook formula, the numerically safe formula, and with a temperature. Three results:

- **Softmax ignores a constant added to all scores,** which is why the safe formula works: subtracting the largest score changes nothing but avoids overflow. The textbook formula returns `nan` for the same scores once the constant is 86 or more in float32.
- **Temperature moves the entropy of the distribution** from 1.42 bits at T = 0.2 to 4.02 at T = 1 and 5.38 at T = 5, against 5.58 bits for a uniform distribution over 48 characters.
- **Samples change character with it.** At T = 0.3 the model repeats safe phrases ("the same ... the same"); at T = 1 it writes plausible nonsense; at T = 2 it produces noise.

![Left: next-character probabilities of the 8 highest-scoring characters at temperatures 0.2, 1.0 and 5.0; at 0.2 the top character has probability 0.56, at 5.0 all eight are about 0.04. Middle: entropy in bits against temperature on a log axis rising from 0.7 at T=0.05 to 5.58 at T=20, with the points 1.42 at T=0.2, 4.02 at T=1 and 5.38 at T=5. Right: the largest output probability against a constant added to the scores: the safe formula stays at 0.153 and the textbook formula drops to nan at c=86]({{ site.baseurl }}/assets/pic/mm-1-softmax.png){: width="900" }

## How: the steps

1. **Get scores.** A language model ends in a linear layer that produces one real number per vocabulary item, the *logits*: here 48 numbers for the prompt `to define a function, use the keyword `, ranging from -5.80 to 3.40.
2. **Exponentiate** every score, so all are positive: `exp(z)`.
3. **Divide by the sum** so the numbers add to 1: `p_i = exp(z_i) / sum_j exp(z_j)`. This is softmax.
4. **Divide the scores by a temperature `T` first** if you want sharper (`T < 1`) or flatter (`T > 1`) probabilities: `p = softmax(z / T)`.
5. **Subtract the maximum score before exponentiating** so the largest exponent is `exp(0) = 1` and nothing overflows.
6. **Sample** the next character from `p` (or take the largest, which is the limit `T -> 0`), append it, and repeat.

## Why it works: the principle

**Softmax is a smooth argmax.** `exp` is increasing, so a larger score gets a larger probability, and the ratios of probabilities are `p_i / p_j = exp(z_i - z_j)`: only *differences* of scores matter. That is the shift invariance: adding a constant `c` to every score multiplies every `exp(z_i)` by `exp(c)`, and the factor cancels in the division. Scaling is different: multiplying all scores by `s` changes the ratios to `exp(s (z_i - z_j))`, which is what temperature does with `s = 1/T`. With two scores, softmax reduces to the sigmoid of their difference: `softmax([a, b])[0] = 1 / (1 + exp(-(a - b)))`.

**Temperature has two limits.** As `T -> 0` the largest score takes all the probability (argmax, greedy decoding); as `T -> infinity` all scores are divided into near-equal numbers and the result approaches the uniform distribution (entropy `log2(48) = 5.58` bits here). Hinton, Vinyals and Dean use the same softmax-with-temperature in [knowledge distillation](https://arxiv.org/abs/1503.02531), where a higher temperature produces softer probability distributions over classes. In text generation, [Holtzman and colleagues (2020)](https://arxiv.org/abs/1904.09751) report that decoding strategy alone can dramatically change the quality of the text from the same model: likelihood-maximising decoding gives text that is "bland and strangely repetitive", which is the low-temperature end of the samples above.

**The overflow trap.** In float32, `exp(z)` overflows to infinity for `z` above about 88.7, so the textbook formula gives `inf / inf = nan`. The largest score here is 3.40, so adding `c = 86` pushes it past the limit. The fix follows from the shift invariance: use `exp(z - max(z))` instead. The [Deep Learning book (chapter 4)](https://www.deeplearningbook.org/contents/numerical.html) describes the same remedy: subtracting the maximum rules out overflow, and at least one term of the denominator is 1, which rules out division by zero. It also notes that underflow in the numerator can still cause trouble, which is why training code computes `log_softmax` directly with the log-sum-exp trick: `log p_i = z_i - (max z + log sum_j exp(z_j - max z))`. The next part uses that form.

**Where it appears in the transformer.** Attention applies softmax to the scaled dot products of queries and keys. [Vaswani and colleagues (2017)](https://arxiv.org/abs/1706.03762) divide them by `sqrt(d_k)` because, they suspect, for large `d_k` the dot products grow large in magnitude and push the softmax into regions with extremely small gradients. Large scores are a sharp softmax, a low temperature; the scaling is a temperature chosen by the size of the vectors. The output layer uses softmax over the vocabulary, as above.

**Pitfalls and limits.**
- **Implementing the textbook formula.** It works for scores of ordinary size, as the first rows of the output show, and fails silently on large ones: `nan` appears in the loss only after a training run reaches a large score. Library functions (`torch.softmax`, `torch.log_softmax`) already subtract the maximum.
- **`softmax` of probabilities.** Applying it twice flattens the distribution; if a function already returns probabilities, do not apply it again, and do not feed softmax output into a loss that expects scores.
- **Temperature is not accuracy.** It changes how the model's fixed probabilities are *used*, not what the model knows. The model's 4.02 bits at T = 1 are what it believes; T = 0.3 hides that uncertainty rather than removing it.
- **Floating point has a floor too.** At very low T, `z / T` can become huge and the probabilities of all but one character round to exactly 0.
- **One context only.** The entropy values are for the scores of one prompt; other prompts give other values.

## Example, worked through

Three scores `z = [2, 1, 0]`:

- `T = 1`: `exp(z) = [7.389, 2.718, 1]`, sum `11.107`, probabilities `[0.665, 0.245, 0.090]`.
- `T = 0.5` (sharper): `z / T = [4, 2, 0]`, `exp = [54.598, 7.389, 1]`, sum `62.987`, probabilities `[0.867, 0.117, 0.016]`.
- `T = 2` (flatter): `z / T = [1, 0.5, 0]`, `exp = [2.718, 1.649, 1]`, sum `5.367`, probabilities `[0.506, 0.307, 0.186]`.
- Subtracting the maximum (2) first gives `z = [0, -1, -2]` and the same probabilities as `T = 1`.

```python
import numpy as np
z = np.array([2.0, 1.0, 0.0])
for T in (1.0, 0.5, 2.0):
    e = np.exp(z / T); p = e / e.sum()
    print(f"T = {T}: exp(z/T) = {e.round(3)}, sum = {e.sum():.3f}, probabilities = {p.round(3)}")
p1 = np.exp(z) / np.exp(z).sum(); p2 = np.exp(z - 2) / np.exp(z - 2).sum()
print("same after subtracting the maximum (2):", p1.round(3), p2.round(3))
print("two scores: softmax([a, b])[0] equals sigmoid(a - b):", np.exp(2) / (np.exp(2) + np.exp(1)), 1 / (1 + np.exp(-(2 - 1))))
```

Output:

```
T = 1.0: exp(z/T) = [7.389 2.718 1.   ], sum = 11.107, probabilities = [0.665 0.245 0.09 ]
T = 0.5: exp(z/T) = [54.598  7.389  1.   ], sum = 62.987, probabilities = [0.867 0.117 0.016]
T = 2.0: exp(z/T) = [2.718 1.649 1.   ], sum = 5.367, probabilities = [0.506 0.307 0.186]
same after subtracting the maximum (2): [0.665 0.245 0.09 ] [0.665 0.245 0.09 ]
two scores: softmax([a, b])[0] equals sigmoid(a - b): 0.7310585786300049 0.7310585786300049
```

## Implementation

The model is a small character-level transformer (2 layers, width 64, 4 heads) trained on the text that ships with Python (`pydoc_data.topics`, about 439,000 characters after cleaning, 48 distinct characters). `part8.py` trains 3 seeds in about 4 minutes on a CPU and saves the first one as `tiny8.pt`. Later parts of this series reuse it.

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

`part1.py` loads the saved model, takes the scores for one prompt, and runs the three experiments:

```python
import math, json, torch, torch.nn.functional as F
from part8 import Tiny, chars, sample
m = Tiny(); m.load_state_dict(torch.load("tiny8.pt")); m.eval()
ctx = "to define a function, use the keyword "
with torch.no_grad(): z = m(torch.tensor([[chars.index(c) for c in ctx]]))[0, -1]      # 48 scores (logits) for the next character

def naive(z):  e = torch.exp(z); return e / e.sum()                                      # the textbook formula
def stable(z): e = torch.exp(z - z.max()); return e / e.sum()                            # subtract the largest score first
def logsumexp(z): return z.max() + torch.log(torch.exp(z - z.max()).sum())

print("1. shift invariance: softmax(z + c) == softmax(z)")
for c in (0.0, 50.0, 100.0, 1000.0):
    a, b = naive(z + c), stable(z + c)
    print(f"   c = {c:6.0f}: naive -> {'nan (overflow)' if torch.isnan(a).any() else f'max abs difference from stable {(a - b).abs().max().item():.2e}'}; stable sum {b.sum().item():.6f}, top probability {b.max().item():.3f}")
print("   stable log-sum-exp matches torch.logsumexp:", abs(logsumexp(z).item() - torch.logsumexp(z, 0).item()) < 1e-6)

print("\n2. temperature on the same scores")
rows = {}
for T_ in (0.2, 0.5, 1.0, 2.0, 5.0):
    p = F.softmax(z / T_, -1); H = -(p * torch.log2(p.clamp_min(1e-30))).sum().item()
    rows[T_] = {"p": p.tolist(), "entropy": H}
    print(f"   T = {T_:3}: top probability {p.max().item():.3f}, entropy {H:.2f} bits (uniform = {math.log2(len(z)):.2f})")
curve = [(t, -(F.softmax(z / t, -1) * F.log_softmax(z / t, -1)).sum().item() / math.log(2)) for t in torch.logspace(-1.3, 1.3, 60).tolist()]
json.dump({"z": z.tolist(), "rows": rows, "curve": curve, "chars": chars, "ctx": ctx}, open("results1.json", "w"))

print("\n3. samples, 120 characters each, same seed, prompt 'the '")
for T_ in (0.3, 1.0, 2.0): print(f"   T = {T_}: {sample(m, 'the ', 120, 1, T_)!r}")
```

Output:

```
1. shift invariance: softmax(z + c) == softmax(z)
   c =      0: naive -> max abs difference from stable 1.49e-08; stable sum 1.000000, top probability 0.153
   c =     50: naive -> max abs difference from stable 1.49e-08; stable sum 1.000000, top probability 0.153
   c =    100: naive -> nan (overflow); stable sum 1.000000, top probability 0.153
   c =   1000: naive -> nan (overflow); stable sum 1.000000, top probability 0.153
   stable log-sum-exp matches torch.logsumexp: True

2. temperature on the same scores
   T = 0.2: top probability 0.561, entropy 1.42 bits (uniform = 5.58)
   T = 0.5: top probability 0.294, entropy 3.11 bits (uniform = 5.58)
   T = 1.0: top probability 0.153, entropy 4.02 bits (uniform = 5.58)
   T = 2.0: top probability 0.084, entropy 4.75 bits (uniform = 5.58)
   T = 5.0: top probability 0.043, entropy 5.38 bits (uniform = 5.58)

3. samples, 120 characters each, same seed, prompt 'the '
   T = 0.3: 'the same is not and the standard the string and the class of the same as are comprehension of the same sequence changed in v'
   T = 1.0: 'the same texted, but is\n \n singleting contain becation of an unenuplaor. it lists.\n\n note:\n\n .__item__(self): type_params: ('
   T = 2.0: 'the skidojust\nif\nmatch4) \n"s30_2finit(\' flwidted, dicpy au7\n .\' tarc_ob7986\' my.brr0c(kabn win):\n\n1y(rp8901)1\'2:8(lspr-t00ig'
```

## Try it

1. Add `c = -1000` to the scores and see which of the two formulas fails, and in which way (underflow instead of overflow).
2. Print the largest probability for `T` from 0.1 to 3 and find the temperature at which it first falls below 0.2.
3. Replace `softmax` by `log_softmax` followed by `exp` in `part1.py` and check that the results agree to about 1e-7.
