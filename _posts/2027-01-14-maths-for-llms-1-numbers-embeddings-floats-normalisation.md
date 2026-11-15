---
title: "Maths for LLMs, Tested in Code (1/10): Numbers, Embeddings, Floats and Normalisation"
date: 2027-01-14 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, embeddings, floating-point, normalization, series]
math: true
description: "Part 1 of 10. The first layer of an LLM is arithmetic on numbers: token IDs, an embedding table that is a matrix product in disguise, floating-point formats that decide whether softmax survives, and LayerNorm or RMSNorm. Each claim is run in code."
---

This series takes the ten chapters of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) (CC BY-NC-SA 4.0) as a map, and for each chapter asks one question: which part of a language model uses this mathematics, and what does it do there? The book is further reading only; the text, code and numbers here are original and every number comes from a script run on this machine (CPU only, PyTorch 2.15). Part 1 covers chapter 1, numbers and meanings, through four components: the token-frequency distribution that shapes the vocabulary, the embedding table, the floating-point format that every tensor is stored in, and the normalisation layer in front of each sub-layer. For the five operations that the rest of the model is built from, see [the earlier post on them](/posts/math-behind-llms-five-operations/).

![Three panels. Left: word counts of one novel against rank on log-log axes, close to a straight line of slope -1.09 for ranks 1 to 1000. Middle: exp(x) in float32, bfloat16 and float16; the float16 curve ends at x = 11.09 where it exceeds 65504. Right: standard deviation of the cosine between random vectors against dimension, equal to 1 over the square root of d.](/assets/pic/maths-llms-1-numbers.png)
*Left: Zipf's law in one novel. Middle: the range of exp(x) in three formats. Right: random vectors in high dimension are nearly orthogonal.*

## 1. Counting tokens: the Zipf curve

A tokenizer maps text to integer IDs, and the model sees only counts and IDs. Word frequencies in natural language follow a power law, [Zipf's law](https://pmc.ncbi.nlm.nih.gov/articles/PMC4176592/) (Piantadosi, 2014): the frequency of the $r$-th most frequent word is roughly

$$f(r) \propto \frac{1}{r^{\alpha}}, \qquad \alpha \approx 1 .$$

On a log-log plot that is a straight line of slope $-\alpha$. The check below uses one public-domain novel (Project Gutenberg, *Pride and Prejudice*), split into lowercase words.

```python
import re, collections, numpy as np
t = open('pg.txt', encoding='utf-8').read()
s = t.index('*** START'); e = t.index('*** END')
words = re.findall(r"[a-z']+", t[s:e].lower())
c = collections.Counter(words)
freq = np.array(sorted(c.values(), reverse=True))
print("tokens:", len(words), "types:", len(c))
for r in (1, 2, 3, 10, 100, 1000):
    print(f"rank {r:>4}: count {freq[r-1]:>5}  rank*count {r*freq[r-1]:>6}")
r = np.arange(1, len(freq)+1)
m = (r >= 1) & (r <= 1000)
slope, icpt = np.polyfit(np.log(r[m]), np.log(freq[m]), 1)
print(f"log-log slope, ranks 1-1000: {slope:.3f}")
print("top 5:", c.most_common(5))
print("hapax (count 1):", int((freq == 1).sum()), f"= {100*(freq==1).sum()/len(freq):.1f}% of types")
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.loglog(r, freq, '.', ms=3, color='#1f4e79', label='Pride and Prejudice')
ax.loglog(r, np.exp(icpt) * r**slope, '-', color='#c0392b', lw=1.2, label=f'fit, ranks 1-1000: slope {slope:.2f}')
ax.set_xlabel('rank of word type'); ax.set_ylabel('count'); ax.legend(); ax.grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig('fig_zipf.png')
```

```
tokens: 128574 types: 6731
rank    1: count  4664  rank*count   4664
rank    2: count  4325  rank*count   8650
rank    3: count  3845  rank*count  11535
rank   10: count  1751  rank*count  17510
rank  100: count   198  rank*count  19800
rank 1000: count    13  rank*count  13000
log-log slope, ranks 1-1000: -1.091
top 5: [('the', 4664), ('to', 4325), ('of', 3845), ('and', 3768), ('her', 2284)]
hapax (count 1): 2599 = 38.6% of types
```

The slope over ranks 1 to 1000 is about $-1.09$, and the top ranks sit below the line (left panel), a deviation Piantadosi also reports for the most frequent words. The LLM consequence is in the last line: 38.6% of the word types occur once. A vocabulary of whole words would hold thousands of rows that are trained on a single example. This is the motivation for subword tokenisers; LLaMA, for example, uses [byte-pair encoding and splits numbers into digits](https://arxiv.org/abs/2302.13971). Part 8 returns to the BPE algorithm as a counting procedure.

## 2. The embedding table is a matrix product

Token IDs are labels: ID 4 is not "twice" ID 2. The first layer turns each ID into a vector by indexing a table $E \in \mathbb{R}^{V \times d}$, with $V$ the vocabulary size and $d$ the model width. Written as algebra, the lookup is a product with a one-hot row vector $e_i$:

$$x_i = e_i^{\top} E .$$

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)
V, d = 6, 4
E = torch.nn.Embedding(V, d)
ids = torch.tensor([3, 0, 3])
oh = F.one_hot(ids, V).float()
print("lookup == one_hot @ E.weight:", torch.allclose(E(ids), oh @ E.weight))
print("one-hot shape", tuple(oh.shape), "weight shape", tuple(E.weight.shape), "output", tuple(E(ids).shape))
for d in (2, 16, 256, 4096):
    W = torch.randn(1000, d)
    c = F.normalize(W, dim=-1) @ F.normalize(W, dim=-1).T
    off = c[~torch.eye(1000, dtype=bool)]
    print(f"d={d:>4}: cosine between random vectors, mean {off.mean():+.4f}, std {off.std():.4f}, theory 1/sqrt(d) = {d**-0.5:.4f}")
```

```
lookup == one_hot @ E.weight: True
one-hot shape (3, 6) weight shape (6, 4) output (3, 4)
d=   2: cosine between random vectors, mean -0.0009, std 0.7078, theory 1/sqrt(d) = 0.7071
d=  16: cosine between random vectors, mean +0.0004, std 0.2500, theory 1/sqrt(d) = 0.2500
d= 256: cosine between random vectors, mean -0.0001, std 0.0625, theory 1/sqrt(d) = 0.0625
d=4096: cosine between random vectors, mean +0.0000, std 0.0157, theory 1/sqrt(d) = 0.0156
```

The first line confirms that `nn.Embedding` and the one-hot product give the same result. The implementation skips the product, because multiplying by a one-hot vector only selects a row.

The next four lines are the geometry of that table. For two independent random vectors in $\mathbb{R}^d$ the cosine has mean $0$ and standard deviation about $1/\sqrt{d}$, and the measured values match the formula from $d=2$ to $d=4096$ (right panel). An untrained embedding table therefore starts with all tokens nearly orthogonal, and training moves related tokens together. It also means that in a model with $d = 4096$, a cosine of 0.1 is already six standard deviations from what chance produces ($0.1 / 0.0156 \approx 6.4$).

## 3. Floating point decides whether softmax runs

Every weight and activation is a floating-point number with a fixed number of bits for range (exponent) and precision (mantissa). The formats seen in LLM training differ in exactly this split.

```python
import torch
print("0.1+0.2 in float64:", 0.1+0.2)
for dt in (torch.float32, torch.float16, torch.bfloat16):
    i = torch.finfo(dt)
    print(f"{str(dt):15} bits={i.bits:>2} max={i.max:.4g} eps={i.eps:.3g} smallest_normal={i.tiny:.3g}")
def softmax_naive(x): e = torch.exp(x); return e / e.sum()
def softmax_stable(x): e = torch.exp(x - x.max()); return e / e.sum()
for big in (10., 11., 12., 89.):
    x = torch.tensor([big, big-1, big-2])
    row = []
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        a = softmax_naive(x.to(dt)); b = softmax_stable(x.to(dt))
        row.append(f"{str(dt)[6:]}: naive={[round(float(v),3) for v in a]} stable={[round(float(v),3) for v in b]}")
    print(f"logits max {big}:"); [print("   ", r) for r in row]
x = torch.tensor([1.0]); 
print("1+1e-3 in fp16:", float(torch.tensor(1.0, dtype=torch.float16)+torch.tensor(1e-3, dtype=torch.float16)), "| in bf16:", float(torch.tensor(1.0, dtype=torch.bfloat16)+torch.tensor(1e-3, dtype=torch.bfloat16)))
```

```
0.1+0.2 in float64: 0.30000000000000004
torch.float32   bits=32 max=3.403e+38 eps=1.19e-07 smallest_normal=1.18e-38
torch.float16   bits=16 max=6.55e+04 eps=0.000977 smallest_normal=6.1e-05
torch.bfloat16  bits=16 max=3.39e+38 eps=0.00781 smallest_normal=1.18e-38
logits max 10.0:
    float32: naive=[0.665, 0.245, 0.09] stable=[0.665, 0.245, 0.09]
    float16: naive=[0.665, 0.245, 0.09] stable=[0.666, 0.245, 0.09]
    bfloat16: naive=[0.668, 0.245, 0.09] stable=[0.668, 0.245, 0.09]
logits max 11.0:
    float32: naive=[0.665, 0.245, 0.09] stable=[0.665, 0.245, 0.09]
    float16: naive=[0.0, 0.0, 0.0] stable=[0.666, 0.245, 0.09]
    bfloat16: naive=[0.664, 0.244, 0.09] stable=[0.668, 0.245, 0.09]
logits max 12.0:
    float32: naive=[0.665, 0.245, 0.09] stable=[0.665, 0.245, 0.09]
    float16: naive=[nan, 0.0, 0.0] stable=[0.666, 0.245, 0.09]
    bfloat16: naive=[0.664, 0.245, 0.09] stable=[0.668, 0.245, 0.09]
logits max 89.0:
    float32: naive=[nan, 0.0, 0.0] stable=[0.665, 0.245, 0.09]
    float16: naive=[nan, nan, nan] stable=[0.666, 0.245, 0.09]
    bfloat16: naive=[nan, 0.0, 0.0] stable=[0.668, 0.245, 0.09]
1+1e-3 in fp16: 1.0009765625 | in bf16: 1.0
```

Read the `finfo` lines first. float16 has more precision than bfloat16 (eps $9.8\times10^{-4}$ against $7.8\times10^{-3}$) but a maximum of only 65504. bfloat16 keeps the float32 exponent range of $3.4\times10^{38}$ and gives up mantissa bits, which is the property [Kalamkar et al.](https://arxiv.org/abs/1905.12322) cite for it: the same range as float32, with simple conversion to and from it. [Micikevicius et al.](https://arxiv.org/abs/1710.03740) noted before that half precision has a limited numerical range compared with single precision. The last line shows the price: `1 + 0.001` is `1.0009765625` in float16 and exactly `1.0` in bfloat16.

The softmax lines show where the range matters. The softmax of the logits $z$ is

$$\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_j z_j .$$

The two forms are equal, but the right-hand form never computes $e^{z}$ for $z>0$. The naive form needs $e^{z}$ itself: in float16 that exceeds 65504 when $z > \ln 65504 \approx 11.09$, the cut-off in the middle panel. With a maximum logit of 11, the exponentials are still finite but their sum overflows, so every probability is 0; at 12 the numerator is infinite and the result is `nan`. In float32 the same happens at about 88.7. The stable form gives the correct answer in all three formats. The max-subtraction form is the standard remedy; the author's reading is that this is why it appears in softmax and cross-entropy code.

## 4. Normalisation: LayerNorm and RMSNorm

Activations drift in scale from layer to layer. A normalisation layer fixes that per token. [Layer normalisation](https://arxiv.org/abs/1607.06450) (Ba et al., 2016) computes, over the $d$ features of one vector $x$:

$$\operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta, \qquad \mu = \tfrac1d\sum_i x_i,\quad \sigma^2 = \tfrac1d\sum_i (x_i-\mu)^2 .$$

[RMSNorm](https://arxiv.org/abs/1910.07467) (Zhang and Sennrich, 2019) drops the mean subtraction and the bias:

$$\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac1d\sum_i x_i^2 + \varepsilon}} .$$

The original [Transformer](https://arxiv.org/abs/1706.03762) applies layer normalisation after the residual addition of each sub-layer; LLaMA instead normalises the input of each sub-layer and uses RMSNorm, which the LLaMA paper lists as "pre-normalization". Both are written below from the formulas and compared with PyTorch's layers.

```python
import torch
torch.manual_seed(0)
d = 8; x = torch.randn(2, d) * 3 + 5
eps = 1e-5
def layernorm(x, g, b): 
    mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
    return g * (x - mu) / torch.sqrt(var + eps) + b
def rmsnorm(x, g):
    return g * x / torch.sqrt((x**2).mean(-1, keepdim=True) + eps)
g = torch.ones(d); b = torch.zeros(d)
ref = torch.nn.LayerNorm(d, eps=eps)(x)
print("LayerNorm  max|mine - torch|:", float((layernorm(x, g, b) - ref).abs().max()))
ref2 = torch.nn.RMSNorm(d, eps=eps)(x)
print("RMSNorm    max|mine - torch|:", float((rmsnorm(x, g) - ref2).abs().max()))
y = layernorm(x, g, b)
print("after LayerNorm: mean", [round(float(v),6) for v in y.mean(-1)], "var", [round(float(v),4) for v in y.var(-1, unbiased=False)])
z = rmsnorm(x, g)
print("after RMSNorm  : mean", [round(float(v),3) for v in z.mean(-1)], "rms", [round(float(v),4) for v in z.pow(2).mean(-1).sqrt()])
print("scale invariance, x -> 10x:  LN", float((layernorm(10*x,g,b)-y).abs().max()), " RMS", float((rmsnorm(10*x,g)-z).abs().max()))
print("shift invariance, x -> x+7: LN", float((layernorm(x+7,g,b)-y).abs().max()), " RMS", float((rmsnorm(x+7,g)-z).abs().max()))
```

```
LayerNorm  max|mine - torch|: 4.76837158203125e-07
RMSNorm    max|mine - torch|: 1.1920928955078125e-07
after LayerNorm: mean [0.0, -0.0] var [1.0, 1.0]
after RMSNorm  : mean [0.79, 0.934] rms [1.0, 1.0]
scale invariance, x -> 10x:  LN 2.1457672119140625e-06  RMS 4.76837158203125e-07
shift invariance, x -> x+7: LN 6.556510925292969e-07  RMS 0.8173436522483826
```

The two hand-written versions match PyTorch to float32 rounding. LayerNorm output has mean 0 and variance 1; RMSNorm output has root mean square 1 but a non-zero mean. Both are invariant to scaling the input by 10, to rounding error. Only LayerNorm is invariant to adding a constant (the shift test: about $10^{-7}$ against $0.82$). Zhang and Sennrich's hypothesis is that this re-centring invariance is dispensable, and they report 7% to 64% less running time at comparable quality. The norm is also where the float limits matter again: the statistics are computed over $d$ values, in float16 the sum of squares $\sum x_i^2$ can itself exceed 65504 (the author's observation from the format limits above, not a measurement of any particular implementation).

## Which maths, which component

- Power-law counts (Zipf) explain why vocabularies are subword units, not words.
- A matrix with a one-hot selector is the embedding layer; the random-vector cosine $\approx 1/\sqrt{d}$ is the baseline for "similar".
- Exponent range and mantissa bits decide whether $e^z$ is computable; the max-subtraction identity makes softmax independent of that range.
- Mean, variance and root mean square, taken over the feature dimension, define LayerNorm and RMSNorm.

Next: Part 2, algebra and structure: the linear layer, the Q, K, V projections and the feed-forward network as matrices.

## Limits

- Zipf's law is an approximation: the fit is on one novel of 128,574 words, in English, with a log-log least-squares fit that is known to be a crude estimator; Piantadosi's review covers better ones and the structure the law misses.
- The claim that rare words are the reason for subword vocabularies is the standard motivation, supported here by the 38.6% singleton figure on one book, not by an experiment comparing tokenisers.
- The overflow thresholds belong to the IEEE-style float16 and bfloat16 as implemented in PyTorch on a CPU; GPU kernels often accumulate in higher precision, so a given model may not fail where the naive code does.
- The normalisation tests use random vectors, not activations of a trained model.
- Not covered: integer quantisation (int8, int4), fp8 formats, and the choice of $\varepsilon$.
