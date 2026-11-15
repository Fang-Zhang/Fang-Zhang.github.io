---
title: "Maths for LLMs, Tested in Code (6/10): Geometry, Dot Products and Why Attention Divides by sqrt(d)"
date: 2027-01-19 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, geometry, attention, embeddings, retrieval, series]
math: true
description: "Part 6 of 10. Attention scores and embedding search are dot products and angles. Simulations show that random vectors become orthogonal as the dimension grows, that q.k has variance d_k so softmax saturates without the 1/sqrt(d_k) factor, how high-dimensional nearest-neighbour search succeeds or fails, and that multi-head scores are a sum of subspace dot products."
---

Chapter 6 of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) is geometry: vectors, norms, inner products, angles, projections and distances. In a Transformer, an attention score is an inner product, an embedding comparison is a cosine, and retrieval is a nearest-neighbour search. Everything here ran on this machine (CPU, NumPy); each claim below is a printed number. [Part 2](/posts/maths-for-llms-2-linear-algebra-attention-ffn-lora/) wrote attention as matrix products; this part asks what those products measure and why the scale factor is there.

![Three panels. Left: histograms of the cosine between two random vectors for d = 16, 64, 256 and 1024, narrowing around zero. Middle: the largest softmax weight over 64 keys against the key dimension, rising from 0.27 to 0.95 without scaling and staying near 0.1 with division by the square root of the dimension. Right: percentage of trials in which a planted noisy copy of the query is the nearest of 100,000 random vectors, for three dimensions and three noise levels.](/assets/pic/maths-llms-6-geometry.png)
*Left: concentration of angles. Middle: the effect of the $1/\sqrt{d_k}$ factor. Right: nearest-neighbour search succeeds as the dimension grows.*

## 1. Random directions are almost orthogonal

For the cosine $\cos\theta=\dfrac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$ of two independent Gaussian vectors in $\mathbb{R}^d$, the mean is 0 and the standard deviation is close to $1/\sqrt{d}$. This is the property that lets a few thousand dimensions hold many roughly independent directions, and it makes a cosine of 0.3 meaningless at $d=16$ and strong evidence at $d=1024$.

```python
import numpy as np
rng = np.random.default_rng(0)
# 1. Near-orthogonality: cosine between independent random directions has mean 0 and std ~ 1/sqrt(d)
print("cosine of random Gaussian vector pairs (20000 pairs):")
for d in (2, 16, 64, 256, 1024, 4096):
    a = rng.standard_normal((20000, d)); b = rng.standard_normal((20000, d))
    c = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    print(f"  d={d:>5}: mean {c.mean():+.4f}  std {c.std():.4f}  1/sqrt(d) = {d ** -0.5:.4f}  P(|cos|>0.1) = {(abs(c) > 0.1).mean():.4f}")
# 2. Why attention divides by sqrt(d_k): q.k has variance d_k for unit-variance components (stated in Vaswani et al.), and softmax saturates
print("\nq.k variance and softmax saturation over 64 keys:")
for dk in (4, 64, 512):
    q = rng.standard_normal((2000, dk)); K = rng.standard_normal((2000, 64, dk))
    s = np.einsum('nd,nkd->nk', q, K)
    for name, sc in (("unscaled", s), ("scaled  ", s / np.sqrt(dk))):
        e = np.exp(sc - sc.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        ent = -(p * np.log2(p + 1e-30)).sum(1).mean()
        print(f"  d_k={dk:>3} {name}: var(q.k) = {sc.var():8.2f}  mean max weight = {p.max(1).mean():.3f}  entropy = {ent:.2f} bits (uniform = 6.00)")
# 3. Gradient of softmax w.r.t. the logits vanishes when the softmax saturates
print("\nmean |d softmax / d logits| (Jacobian entries), one query, 64 keys:")
for dk in (4, 64, 512):
    q = rng.standard_normal((500, dk)); K = rng.standard_normal((500, 64, dk)); out = {}
    for name, div in (("unscaled", 1.0), ("scaled", np.sqrt(dk))):
        s = np.einsum('nd,nkd->nk', q, K) / div; e = np.exp(s - s.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        J = p[:, :, None] * (np.eye(64)[None] - p[:, None, :]); out[name] = np.abs(J).mean()
    print(f"  d_k={dk:>3}: unscaled {out['unscaled']:.2e}   scaled {out['scaled']:.2e}")
# 4. Retrieval: nearest neighbour by cosine. A planted document (noisy copy of the query) among N random documents
print("\nplanted-neighbour retrieval, N = 100000 random documents, noise level s (doc = query + s * noise):")
N = 100000
for d in (16, 64, 384):
    row = []; best = []
    for s in (0.5, 1.0, 2.0):
        ok = 0
        for _ in range(30):
            q = rng.standard_normal(d); D = rng.standard_normal((N, d)); D[0] = q + s * rng.standard_normal(d)
            cs = D @ q / np.linalg.norm(D, axis=1)               # cosine up to the constant |q|
            ok += cs.argmax() == 0; best.append(cs[1:].max() / np.linalg.norm(q))
        row.append(f"s={s}: {ok}/30")
    print(f"  d={d:>3}: " + "   ".join(row) + f"   (best random cosine, measured: {np.mean(best):.2f}; planted doc at s=1: about {1/np.sqrt(2):.2f})")
# 5. Multi-head as subspaces: the heads' score matrices sum to the full score matrix
d, h, T = 16, 4, 6; X = rng.standard_normal((T, d)); Wq = rng.standard_normal((d, d)); Wk = rng.standard_normal((d, d))
full = (X @ Wq) @ (X @ Wk).T; dh = d // h
parts = sum((X @ Wq[:, i*dh:(i+1)*dh]) @ (X @ Wk[:, i*dh:(i+1)*dh]).T for i in range(h))
print("\nfull q.k scores == sum of per-head scores (each head is a dot product in a d/h-dim subspace):", np.allclose(full, parts))
```

```
cosine of random Gaussian vector pairs (20000 pairs):
  d=    2: mean +0.0068  std 0.7052  1/sqrt(d) = 0.7071  P(|cos|>0.1) = 0.9367
  d=   16: mean +0.0017  std 0.2502  1/sqrt(d) = 0.2500  P(|cos|>0.1) = 0.7061
  d=   64: mean +0.0002  std 0.1247  1/sqrt(d) = 0.1250  P(|cos|>0.1) = 0.4266
  d=  256: mean -0.0004  std 0.0623  1/sqrt(d) = 0.0625  P(|cos|>0.1) = 0.1107
  d= 1024: mean -0.0001  std 0.0312  1/sqrt(d) = 0.0312  P(|cos|>0.1) = 0.0011
  d= 4096: mean -0.0001  std 0.0157  1/sqrt(d) = 0.0156  P(|cos|>0.1) = 0.0000

q.k variance and softmax saturation over 64 keys:
  d_k=  4 unscaled: var(q.k) =     4.04  mean max weight = 0.278  entropy = 4.04 bits (uniform = 6.00)
  d_k=  4 scaled  : var(q.k) =     1.01  mean max weight = 0.105  entropy = 5.34 bits (uniform = 6.00)
  d_k= 64 unscaled: var(q.k) =    63.65  mean max weight = 0.787  entropy = 0.85 bits (uniform = 6.00)
  d_k= 64 scaled  : var(q.k) =     0.99  mean max weight = 0.106  entropy = 5.32 bits (uniform = 6.00)
  d_k=512 unscaled: var(q.k) =   513.78  mean max weight = 0.927  entropy = 0.26 bits (uniform = 6.00)
  d_k=512 scaled  : var(q.k) =     1.00  mean max weight = 0.107  entropy = 5.31 bits (uniform = 6.00)

mean |d softmax / d logits| (Jacobian entries), one query, 64 keys:
  d_k=  4: unscaled 4.19e-04   scaled 4.68e-04
  d_k= 64: unscaled 1.39e-04   scaled 4.69e-04
  d_k=512: unscaled 5.32e-05   scaled 4.69e-04

planted-neighbour retrieval, N = 100000 random documents, noise level s (doc = query + s * noise):
  d= 16: s=0.5: 23/30   s=1.0: 2/30   s=2.0: 0/30   (best random cosine, measured: 0.85; planted doc at s=1: about 0.71)
  d= 64: s=0.5: 30/30   s=1.0: 30/30   s=2.0: 9/30   (best random cosine, measured: 0.52; planted doc at s=1: about 0.71)
  d=384: s=0.5: 30/30   s=1.0: 30/30   s=2.0: 30/30   (best random cosine, measured: 0.22; planted doc at s=1: about 0.71)

full q.k scores == sum of per-head scores (each head is a dot product in a d/h-dim subspace): True
```

Block 1: the measured standard deviation matches $1/\sqrt{d}$ at every $d$ (0.7052 against 0.7071 at $d=2$, 0.0157 against 0.0156 at $d=4096$). The probability of $|\cos|>0.1$ falls from 0.94 at $d=2$ to 0.43 at $d=64$ and 0.0011 at $d=1024$.

## 2. Why attention divides by $\sqrt{d_k}$

[Vaswani et al.](https://arxiv.org/abs/1706.03762) state it as a footnote: if the components of $q$ and $k$ are independent with mean 0 and variance 1, then $q\cdot k=\sum_i q_ik_i$ has mean 0 and variance $d_k$. Their stated suspicion is that for large $d_k$ the dot products grow in magnitude and push the softmax into regions with extremely small gradients, so they scale by $1/\sqrt{d_k}$. Block 2 and block 3 test both halves on random queries and 64 keys:

- **Variance.** Unscaled, $\operatorname{Var}(q\cdot k)$ is 4.04, 63.65 and 513.78 for $d_k=4,64,512$; divided by $\sqrt{d_k}$ it is 1.01, 0.99 and 1.00.
- **Saturation.** The largest softmax weight is 0.787 at $d_k=64$ and 0.927 at 512 without scaling (entropy 0.85 and 0.26 bits, against 6 bits for uniform); with scaling it stays at 0.106 and 0.107 for every $d_k$ (middle panel).
- **Gradient.** The mean magnitude of the softmax Jacobian entries $p_i(\delta_{ij}-p_j)$ drops from $4.2\times10^{-4}$ at $d_k=4$ to $5.3\times10^{-5}$ at 512 unscaled, and is constant at $4.7\times10^{-4}$ scaled. This is the mechanism of Part 3 (vanishing gradients) applied to one softmax.

The setup assumes unit-variance components, which holds at initialisation by design and need not hold after training.

## 3. Multi-head attention is a sum of subspace dot products

Splitting $W_Q,W_K$ into $h$ column blocks of width $d/h$ turns one inner product in $\mathbb{R}^d$ into $h$ inner products in $\mathbb{R}^{d/h}$:

$$QK^{\top}=\sum_{i=1}^{h}(XW_Q^{(i)})(XW_K^{(i)})^{\top}.$$

The last line of the output confirms it numerically (`True`, for $d=16$, $h=4$). In the full model the heads are not summed before the softmax but kept separate, each with its own softmax and $\sqrt{d/h}$ factor; the identity says only that the heads together see exactly the information in the full score, divided into $h$ subspaces.

## 4. Nearest-neighbour search and retrieval

Retrieval-augmented systems rank stored vectors by cosine to a query. Block 4 plants one noisy copy of the query ($\text{doc}=q+s\cdot\text{noise}$) among 100,000 random vectors and counts how often it is ranked first. Whether it wins is a race between its cosine, about $1/\sqrt{1+s^2}$ (0.71 at $s=1$), and the best of 100,000 random cosines, which shrinks as $d$ grows: measured 0.85 at $d=16$, 0.52 at $d=64$ and 0.22 at $d=384$. At $d=16$ the planted document is found in 2 of 30 trials at $s=1$; at $d=64$ in 30 of 30; at $d=384$ it is found in all 30 even at $s=2$ (right panel). The same arithmetic is why a larger embedding dimension makes retrieval over a larger collection reliable, and why an embedding with too few dimensions confuses unrelated items once the collection is large (the author's reading of these numbers).

## Which maths, which component

- Inner product and angle: attention scores, embedding similarity.
- Variance of a sum of independent terms: the $\sqrt{d_k}$ factor.
- Concentration of the cosine at $1/\sqrt{d}$: why high-dimensional retrieval works, and why a threshold on cosine depends on $d$.
- Block decomposition of a projection: multi-head attention.

Next: Part 7, optimisation: gradient descent, momentum, Adam and warmup.

## Limits

- Random Gaussian vectors are a null model. Trained embeddings are far from isotropic, so their cosines concentrate differently; the figures here are the baseline against which structure shows.
- The variance argument needs independent unit-variance components; trained queries and keys are neither.
- The retrieval test uses exact search and a synthetic noise model. Approximate-nearest-neighbour indexes trade accuracy for speed and are not tested here.
- Norm effects (a dot product versus a cosine) were not separated: the retrieval test ranks by dot product divided by the document norm, which is cosine up to the constant query norm.
