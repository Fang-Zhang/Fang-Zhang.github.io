---
title: "Maths for LLMs (10/10): SVD, Low Rank and the Fourier View of Positions"
date: 2027-01-23 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, svd, low-rank, positional-encoding, fourier, series]
math: true
description: "Part 10 of 10. The singular value decomposition compresses a trained weight matrix and builds word vectors from co-occurrence counts; sinusoidal position encodings are rotations, whose dot products depend only on the offset, and whose spectrum is a single peak. Each is checked in code."
---

Chapter 10 of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) collects advanced topics. Two of them reach into LLM practice directly: the singular value decomposition (SVD), behind low-rank adaptation, weight compression and classical word vectors, and Fourier analysis, behind the original positional encoding. This closing part tests both on this machine (CPU; PyTorch and NumPy). [Part 2](/posts/maths-for-llms-2-linear-algebra-attention-ffn-lora/) used rank for LoRA; here the same tool is applied to a trained matrix.

![Three panels. Left: singular values of a trained 128 by 128 layer on a log axis, falling from 6 to 0.26 at index 100 and 0.001 at index 128, with test accuracy against rank in red: 25 percent at rank 2, 94 percent at rank 8, 98 percent from rank 16. Middle: heat map of the sinusoidal position encoding for 100 positions and 64 dimensions. Right: the dot product of two encodings against the offset between them.](/assets/pic/maths-llms-10-advanced.png)
*Left: spectrum and accuracy against rank. Middle: the encoding as stripes of decreasing frequency. Right: the dot product as a function of offset.*

```python
import re, collections, numpy as np, torch
# 1. Eckart-Young: truncated SVD is the best rank-r approximation; error = sqrt(sum of discarded sigma^2)
rng = np.random.default_rng(0); W = rng.standard_normal((64, 64)); U, S, Vt = np.linalg.svd(W)
for r in (8, 32):
    Wr = (U[:, :r] * S[:r]) @ Vt[:r]; err = np.linalg.norm(W - Wr)
    print(f"1. rank {r}: ||W - W_r||_F = {err:.4f}; sqrt(sum of discarded sigma^2) = {np.sqrt((S[r:] ** 2).sum()):.4f}")
# 2. Compress a trained weight matrix: MLP on 8x8 digits, hidden 128, then replace the hidden-layer weights by rank-r factors
torch.manual_seed(0); X, y = np.load('dX.npy'), np.load('dy.npy')   # sklearn load_digits, saved to disk
X = torch.tensor(X / 16.0, dtype=torch.float32); y = torch.tensor(y)
idx = torch.randperm(len(X)); tr, te = idx[:1300], idx[1300:]
net = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
opt = torch.optim.Adam(net.parameters(), 3e-3)
for ep in range(300):
    opt.zero_grad(); torch.nn.functional.cross_entropy(net(X[tr]), y[tr]).backward(); opt.step()
acc = lambda m: float((m(X[te]).argmax(1) == y[te]).float().mean())
print(f"\n2. trained MLP 64-128-128-10, test accuracy {acc(net):.3f}")
W2 = net[2].weight.detach(); U, S, Vt = torch.linalg.svd(W2)
print("   singular values of the 128x128 middle layer: top 5", [round(float(s), 2) for s in S[:5]], " 100th", round(float(S[99]), 3), " 128th", round(float(S[-1]), 3))
rows = []
for r in (2, 4, 8, 16, 32, 64, 128):
    m = torch.nn.Sequential(*[torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)])
    m.load_state_dict(net.state_dict()); m[2].weight.data = (U[:, :r] * S[:r]) @ Vt[:r]
    params = r * (128 + 128); rows.append((r, params, acc(m)))
    print(f"   rank {r:>3}: parameters {params:>6} (full {128*128}), test accuracy {acc(m):.3f}, energy kept {float((S[:r]**2).sum() / (S**2).sum()):.3f}")
np.save('svd_rows.npy', np.array(rows)); np.save('svals.npy', S.numpy())
# 3. SVD of a PPMI co-occurrence matrix gives word vectors
text = open('pg.txt', encoding='utf-8').read(); text = text[text.index('*** START'):text.index('*** END')]
toks = re.findall(r"[a-z]+", text.lower()); cnt = collections.Counter(toks); vocab = [w for w, c in cnt.most_common(2000)]; wi = {w: i for i, w in enumerate(vocab)}
C = np.zeros((2000, 2000)); ids = [wi.get(w, -1) for w in toks]
for i, a_ in enumerate(ids):
    if a_ < 0: continue
    for j in range(max(0, i - 4), min(len(ids), i + 5)):
        if j != i and ids[j] >= 0: C[a_, ids[j]] += 1
pw = C.sum(1, keepdims=True) / C.sum(); pc = C.sum(0, keepdims=True) / C.sum(); pmi = np.log((C / C.sum() + 1e-12) / (pw * pc + 1e-12)); ppmi = np.maximum(pmi, 0)
U3, S3, _ = np.linalg.svd(ppmi, full_matrices=False); E = U3[:, :50] * np.sqrt(S3[:50]); E /= np.linalg.norm(E, axis=1, keepdims=True)
print("\n3. 50-d vectors from SVD of the 2000x2000 PPMI matrix (window 4, one novel); nearest neighbours by cosine:")
for w in ("elizabeth", "mother", "walked", "happy", "london"):
    s = E @ E[wi[w]]; print(f"   {w:<10}", [vocab[j] for j in np.argsort(-s)[1:7]])
# 4. Sinusoidal position encoding: PE(pos+k) = R(k) PE(pos) per frequency pair; dot product depends on offset only
d = 64; pos = np.arange(512); i2 = np.arange(0, d, 2); w = 1 / 10000 ** (i2 / d)
PE = np.zeros((512, d)); PE[:, 0::2] = np.sin(pos[:, None] * w); PE[:, 1::2] = np.cos(pos[:, None] * w)
k = 7; err = 0
for p in (3, 100, 400):
    pred = np.zeros(d)
    for j, wj in enumerate(w):
        s, c = PE[p, 2*j], PE[p, 2*j+1]; pred[2*j] = s * np.cos(wj * k) + c * np.sin(wj * k); pred[2*j+1] = c * np.cos(wj * k) - s * np.sin(wj * k)
    err = max(err, np.abs(pred - PE[p + k]).max())
print(f"\n4. PE(pos+{k}) from PE(pos) by a 2x2 rotation per frequency: max |error| = {err:.2e}")
dots = {p: float(PE[p] @ PE[p + k]) for p in (0, 50, 200, 400)}; print("   PE(p).PE(p+7) for p = 0, 50, 200, 400:", {a: round(b, 4) for a, b in dots.items()}, f" analytic sum cos(w_j*7) = {np.cos(w * 7).sum():.4f}")
print("   wavelengths 2*pi/w: shortest", round(2 * np.pi / w[0], 2), ", longest", round(2 * np.pi / w[-1], 0), " (geometric progression, ratio", round(w[0] / w[1], 4), ")")
print("   norm of PE row (all positions):", sorted({float(x) for x in np.round(np.linalg.norm(PE, axis=1), 6)}))
# 5. Frequency view: a DFT of a PE column is a single peak
col = PE[:, 8]; F = np.abs(np.fft.rfft(col)); fr = np.fft.rfftfreq(512)
print(f"\n5. DFT of PE[:, 8] (frequency w = {w[4]:.4f} rad/step -> {w[4]/(2*np.pi):.4f} cycles/step): peak at {fr[F.argmax()]:.4f} cycles/step, peak share of spectral power {F.max()**2 / (F**2).sum():.3f}")
np.save('PE.npy', PE)
```

```
1. rank 8: ||W - W_r||_F = 50.0976; sqrt(sum of discarded sigma^2) = 50.0976
1. rank 32: ||W - W_r||_F = 21.1359; sqrt(sum of discarded sigma^2) = 21.1359

2. trained MLP 64-128-128-10, test accuracy 0.976
   singular values of the 128x128 middle layer: top 5 [6.03, 5.42, 5.33, 5.0, 4.47]  100th 0.26  128th 0.001
   rank   2: parameters    512 (full 16384), test accuracy 0.254, energy kept 0.264
   rank   4: parameters   1024 (full 16384), test accuracy 0.571, energy kept 0.479
   rank   8: parameters   2048 (full 16384), test accuracy 0.944, energy kept 0.716
   rank  16: parameters   4096 (full 16384), test accuracy 0.976, energy kept 0.811
   rank  32: parameters   8192 (full 16384), test accuracy 0.976, energy kept 0.892
   rank  64: parameters  16384 (full 16384), test accuracy 0.976, energy kept 0.971
   rank 128: parameters  32768 (full 16384), test accuracy 0.976, energy kept 1.000

3. 50-d vectors from SVD of the 2000x2000 PPMI matrix (window 4, one novel); nearest neighbours by cosine:
   elizabeth  ['she', 'her', 'darcy', 'and', 'with', 'was']
   mother     ['hear', 'upstairs', 'father', 'her', 'room', 'poor']
   walked     ['lawn', 'across', 'ran', 'door', 'up', 'rid']
   happy      ['pleased', 'ease', 'humour', 'really', 'light', 'seldom']
   london     ['commission', 'cross', 'longbourn', 'road', 'forster', 'town']

4. PE(pos+7) from PE(pos) by a 2x2 rotation per frequency: max |error| = 1.98e-14
   PE(p).PE(p+7) for p = 0, 50, 200, 400: {0: 23.2643, 50: 23.2643, 200: 23.2643, 400: 23.2643}  analytic sum cos(w_j*7) = 23.2643
   wavelengths 2*pi/w: shortest 6.28 , longest 47117.0  (geometric progression, ratio 1.3335 )
   norm of PE row (all positions): [5.656854]

5. DFT of PE[:, 8] (frequency w = 0.3162 rad/step -> 0.0503 cycles/step): peak at 0.0508 cycles/step, peak share of spectral power 0.835
```

## 1. Truncated SVD is the best low-rank approximation

Every matrix factors as $W=\sum_i\sigma_iu_iv_i^{\top}$ with $\sigma_1\ge\sigma_2\ge\dots\ge0$. Keeping the first $r$ terms gives the best rank-$r$ approximation in the Frobenius norm (the Eckart–Young theorem), with error $\sqrt{\sum_{i>r}\sigma_i^2}$. Block 1 confirms the error formula on a random $64\times64$ matrix: 50.0976 for $r=8$ and 21.1359 for $r=32$, equal to the discarded-spectrum formula to the printed digits. A random matrix has a flat spectrum, so low-rank approximation of it is poor; the useful case is a trained matrix.

## 2. Compressing a trained layer

Block 2 trains a small MLP (64-128-128-10) on the 8×8 handwritten-digit images that ship with scikit-learn, reaching a test accuracy of 0.976, then replaces the middle $128\times128$ weight matrix by its rank-$r$ truncation. The spectrum is far from flat: the top singular values are 6.03, 5.42, 5.33, 5.00 and 4.47, the 100th is 0.26, and the last is 0.001 (left panel). Accuracy follows it. Rank 2 gives 0.254, rank 4 gives 0.571, rank 8 gives 0.944 and rank 16 already matches the full matrix at 0.976, with a quarter of the parameters ($r(128+128)=4{,}096$ against 16,384). At rank 128 the factorised form needs 32,768 parameters, twice the dense layer: factorising only pays when $r<d/2$. Two limits are visible in the numbers: the "energy kept" is only 0.811 at rank 16, so accuracy and Frobenius energy are different measures, and the matrix compresses so well here because this task is simple (ten classes on 64 inputs). Nothing in this experiment says how compressible the weights of an LLM are; that is an empirical question for each model. What carries over is the method and the reading of the spectrum. The LoRA paper's low-rank claim in Part 2 concerns weight updates, not the weights themselves.

## 3. Word vectors from an SVD

Before neural embeddings, word vectors were built by factorising a co-occurrence matrix. Block 3 counts co-occurrences within a window of 4 words for the 2,000 most frequent words of the Part 1 novel, converts them to positive pointwise mutual information $\max(0,\log\frac{p(w,c)}{p(w)p(c)})$ (the Part 9 quantity), and keeps the top 50 singular directions. The nearest neighbours by cosine are partly sensible and partly not: `happy` has `pleased`, `ease` and `humour`; `walked` has `lawn`, `across`, `ran` and `door`; `elizabeth` has `she`, `her` and `darcy`; `london` has `commission`, `cross` and `longbourn`. One novel (over 100,000 words) is too small for clean semantics; the point is that low-rank structure in co-occurrence statistics already yields geometry in which nearness carries meaning, which is what the cosine of Part 6 then reads out.

## 4. Sinusoidal positions are rotations

[Vaswani et al.](https://arxiv.org/abs/1706.03762) define $PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{\text{model}}})$ and $PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{\text{model}}})$, say the wavelengths form a geometric progression from $2\pi$ to $10000\cdot2\pi$, and chose the function because it may let the model attend by relative position: for any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$. The linear function is a rotation. For each frequency $\omega$,

$$\begin{pmatrix}\sin\omega(p+k)\\ \cos\omega(p+k)\end{pmatrix}=\begin{pmatrix}\cos\omega k&\sin\omega k\\ -\sin\omega k&\cos\omega k\end{pmatrix}\begin{pmatrix}\sin\omega p\\ \cos\omega p\end{pmatrix}.$$

Block 4 builds the encoding for $d=64$ and applies this matrix to the encodings at positions 3, 100 and 400 with $k=7$: the largest error against the true $PE_{pos+7}$ is $2\times10^{-14}$. Two consequences follow, and both are printed. First, the dot product $PE_p\cdot PE_{p+k}=\sum_j\cos(\omega_jk)$ does not depend on $p$: 23.2643 at $p=0,50,200,400$, equal to the analytic sum (right panel plots it against $k$; it decays and oscillates with the offset). Second, every row has the same norm, $\sqrt{d/2}=5.6569$. The wavelengths run from 6.28 to 47,117 with a constant ratio of 1.3335 between neighbouring frequencies. The paper also reports that learned positional embeddings gave nearly identical results.

## 5. The Fourier view

Each column of the encoding is a pure sinusoid in the position, so its discrete Fourier transform is one peak. For dimension 8 (angular frequency 0.3162 rad per position, 0.0503 cycles per position) the transform of 512 positions peaks at 0.0508 cycles per position and holds 83.5% of the spectral power in that bin; the remainder is leakage, because 512 positions do not contain a whole number of periods. The full encoding is thus a bank of $d/2$ sinusoids at geometrically spaced frequencies, i.e. a set of Fourier features of the position, with high frequencies resolving nearby positions and low frequencies separating distant ones (the middle panel shows the stripes thickening down the dimensions).

## Which maths, which component

- SVD and rank: weight compression, LoRA, factorised embeddings, classical word vectors.
- Eckart–Young: why truncating the smallest singular values is the optimal Frobenius-norm compression.
- Rotation matrices and angle-addition identities: relative positions in sinusoidal encoding.
- Discrete Fourier transform: reading an encoding as a bank of frequencies.

## The series in one table of components

1. Numbers and normalisation: floats, Zipf's law, RMSNorm.
2. Linear algebra: projections in attention, the feed-forward block, LoRA.
3. Calculus: backpropagation, residual paths.
4. Probability: softmax, temperature, top-$p$.
5. Statistics: evaluation noise, scaling-law fits.
6. Geometry: dot products, the $\sqrt{d_k}$ factor, retrieval.
7. Optimisation: Adam, AdamW, warmup.
8. Discrete maths: the causal mask, the KV cache, BPE.
9. Information: cross-entropy, perplexity, KL.
10. This part: SVD and Fourier features.

## Limits

- The compression test is one small MLP on digit images; the accuracy-versus-rank curve does not transfer to Transformers, and no LLM weights were decomposed.
- The word-vector neighbours come from one novel with a window of 4, on the 2,000 most frequent words, and were not evaluated against a benchmark; they are examples, and some are poor.
- The rotation identity is exact in real arithmetic and was verified in float64; the sinusoidal encoding is only one of several position schemes (learned and rotary variants are not tested).
- Topics of the book's final chapter not covered: tensor decompositions, kernel methods and numerical linear-algebra stability.
