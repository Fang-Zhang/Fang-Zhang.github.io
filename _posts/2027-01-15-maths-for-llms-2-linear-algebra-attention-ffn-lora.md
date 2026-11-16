---
title: "Maths for LLMs (2/10): Linear Algebra in Attention, the Feed-Forward Block and LoRA"
date: 2027-01-15 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, linear-algebra, attention, lora, series]
math: true
description: "Part 2 of 10. Multi-head attention is a block structure of four matrices, the feed-forward network is two or three more, and LoRA is a rank statement about their updates. Each is written from the formulas, checked against PyTorch, and measured."
---

Chapter 2 of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) is algebra and structure: linear equations, matrix multiplication, vector spaces, rank and eigenvalues. This post maps that chapter to three components of a transformer layer: multi-head attention, the position-wise feed-forward network, and the low-rank adapter [LoRA](https://arxiv.org/abs/2106.09685). Everything below is run on this machine (CPU, PyTorch 2.15), and the first block is compared with `torch.nn.MultiheadAttention`. [Part 1](/posts/maths-for-llms-1-numbers-embeddings-floats-normalisation/) covered the numbers that these matrices act on.

![Three panels. Left: the query projection matrix with four coloured column blocks, one per head. Middle: parameters per layer at d = 4096, 67.1M for attention and 134.2M for the gated feed-forward network, 67 percent of the block. Right: relative error of a learned low-rank update against rank, dropping from 0.64 at rank 2 to 0.005 at rank 4 and staying there.](/assets/pic/maths-llms-2-algebra.png)
*Left: heads are column blocks of one matrix. Middle: where a layer's parameters sit. Right: a rank-4 update is recovered exactly at rank 4.*

## 1. Multi-head attention is four matrices and a reshape

The [Transformer paper](https://arxiv.org/abs/1706.03762) defines, with $d_k = d_{\text{model}}/h$,

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)V, \qquad \operatorname{MultiHead}=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^{O},$$

with $\text{head}_i = \operatorname{Attention}(XW^Q_i, XW^K_i, XW^V_i)$. Linear algebra gives a simpler reading of the same thing. The $h$ per-head matrices $W^Q_i \in \mathbb{R}^{d \times d_k}$ are column blocks of one $d \times d$ matrix (left panel), so all heads are computed by one matrix product followed by a reshape. The output projection $W^O$ then mixes the concatenated heads. The code writes both versions, a fused one and a loop over heads, and compares them with PyTorch's layer.

```python
import torch, math
torch.manual_seed(0)
T, d, h = 5, 16, 4; dk = d // h
x = torch.randn(1, T, d)
Wq, Wk, Wv, Wo = [torch.randn(d, d) / math.sqrt(d) for _ in range(4)]
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)

# (a) one big matrix per projection, then split the columns into h heads
def mha_fused(x):
    q, k, v = x @ Wq, x @ Wk, x @ Wv                                   # (1,T,d)
    q, k, v = [t.view(1, T, h, dk).transpose(1, 2) for t in (q, k, v)]  # (1,h,T,dk)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(dk)
    s = s.masked_fill(mask, float('-inf'))
    a = torch.softmax(s, -1)
    o = (a @ v).transpose(1, 2).reshape(1, T, d)                        # concat heads
    return o @ Wo, a

# (b) the same thing written as h separate small matrices, head by head
def mha_loop(x):
    outs = []
    for i in range(h):
        sl = slice(i * dk, (i + 1) * dk)
        q, k, v = x @ Wq[:, sl], x @ Wk[:, sl], x @ Wv[:, sl]           # (1,T,dk)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(dk)
        a = torch.softmax(s.masked_fill(mask, float('-inf')), -1)
        outs.append(a @ v)
    return torch.cat(outs, -1) @ Wo

y1, a = mha_fused(x); y2 = mha_loop(x)
print("fused vs per-head loop, max abs diff:", float((y1 - y2).abs().max()))
print("attention weights, head 0, row sums:", [round(float(v), 4) for v in a[0, 0].sum(-1)])
print("head 0 attention matrix (lower-triangular because of the mask):")
print(a[0, 0].numpy().round(2))
print("shapes: x", tuple(x.shape), "| Wq", tuple(Wq.shape), "| per head Wq slice", tuple(Wq[:, :dk].shape))

# check against torch's own implementation
m = torch.nn.MultiheadAttention(d, h, bias=False, batch_first=True)
with torch.no_grad():
    m.in_proj_weight.copy_(torch.cat([Wq.T, Wk.T, Wv.T], 0)); m.out_proj.weight.copy_(Wo.T)
ref, _ = m(x, x, x, attn_mask=mask, need_weights=False)
print("vs torch.nn.MultiheadAttention, max abs diff:", float((y1 - ref).abs().max()))

# parameters and FLOPs of one block at LLaMA-7B-like sizes
D, H = 4096, 32; F = int(2 * 4 * D / 3)   # 2/3 * 4d, as stated in the LLaMA paper
attn = 4 * D * D; ffn = 3 * D * F
print(f"\nd_model={D}, heads={H}, d_head={D//H}")
print(f"attention params per layer: 4*d^2 = {attn/1e6:.1f}M;  heads change the shape, not the count")
print(f"gated FFN params per layer: 3*d*d_ff = {ffn/1e6:.1f}M  (d_ff/d = {F/D:.2f})")
print(f"FFN share of the two blocks: {100*ffn/(ffn+attn):.1f}%")
```

```
fused vs per-head loop, max abs diff: 3.129243850708008e-07
attention weights, head 0, row sums: [1.0, 1.0, 1.0, 1.0, 1.0]
head 0 attention matrix (lower-triangular because of the mask):
[[1.   0.   0.   0.   0.  ]
 [0.39 0.61 0.   0.   0.  ]
 [0.16 0.19 0.65 0.   0.  ]
 [0.31 0.63 0.03 0.04 0.  ]
 [0.32 0.47 0.12 0.08 0.01]]
shapes: x (1, 5, 16) | Wq (16, 16) | per head Wq slice (16, 4)
vs torch.nn.MultiheadAttention, max abs diff: 1.1920928955078125e-07

d_model=4096, heads=32, d_head=128
attention params per layer: 4*d^2 = 67.1M;  heads change the shape, not the count
gated FFN params per layer: 3*d*d_ff = 134.2M  (d_ff/d = 2.67)
FFN share of the two blocks: 66.7%
```

Four results to read from the output:

- The fused and per-head versions agree to $3\times10^{-7}$, which is float32 rounding. Splitting into heads is a reshape of columns, not a different computation.
- The code matches `nn.MultiheadAttention` to $1.2\times10^{-7}$, so the formula above is the layer.
- Each row of an attention matrix sums to 1 (softmax) and the matrix is lower-triangular, because the causal mask sets the upper triangle to $-\infty$ before softmax. The mask comes from the same paper, which describes setting illegal connections to $-\infty$ to preserve the autoregressive property.
- A head's weights are low-dimensional: with $d=4096$ and 32 heads, each head works in 128 dimensions, and the four matrices $W^Q, W^K, W^V, W^O$ hold $4d^2$ parameters whatever $h$ is. More heads change the shape of the computation, not the parameter count (the script prints this).

The factor $1/\sqrt{d_k}$ links back to Part 1: for random vectors with unit-variance entries, a dot product has variance $d_k$, so dividing by $\sqrt{d_k}$ keeps the logits at unit scale and out of the saturated region of softmax. Part 6 returns to this.

## 2. The feed-forward block: two matrices and one non-linearity

Each position also passes through a small network. The original paper defines it as two linear maps with a ReLU between them, applied to each position separately and identically:

$$\operatorname{FFN}(x)=\max(0,\,xW_1)\,W_2 .$$

[LLaMA](https://arxiv.org/abs/2302.13971) replaces ReLU by SwiGLU, with a hidden size of $\tfrac23 \cdot 4d$ instead of $4d$ (the LLaMA paper's statement). The gated form uses three matrices:

$$\operatorname{FFN}_{\text{SwiGLU}}(x)=\big(\operatorname{SiLU}(xW_g)\odot xW_u\big)\,W_d .$$

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)
d, dff = 8, 32
x = torch.randn(3, d)
W1, W2 = torch.randn(d, dff) / d**.5, torch.randn(dff, d) / dff**.5
relu = lambda z: z.clamp(min=0)
y = relu(x @ W1) @ W2
# acts on every position independently: permuting rows permutes outputs
p = torch.tensor([2, 0, 1])
print("position-wise (row permutation commutes):", torch.allclose(relu(x[p] @ W1) @ W2, y[p]))
# without a nonlinearity two layers collapse to one matrix
print("no activation -> one matrix W1@W2 of shape", tuple((W1 @ W2).shape), ":", torch.allclose((x @ W1) @ W2, x @ (W1 @ W2), atol=1e-6))
print("with ReLU the same shortcut fails; max diff:", float((relu(x @ W1) @ W2 - x @ (W1 @ W2)).abs().max()))
# SwiGLU as used in LLaMA: (silu(x Wg) * (x Wu)) Wd
Wg, Wu, Wd = torch.randn(d, dff) / d**.5, torch.randn(d, dff) / d**.5, torch.randn(dff, d) / dff**.5
z = F.silu(x @ Wg) * (x @ Wu)
print("SwiGLU output shape:", tuple((z @ Wd).shape), "| elementwise gate in [", round(float(F.silu(x@Wg).min()), 3), ",", round(float(F.silu(x@Wg).max()), 3), "]")
```

```
position-wise (row permutation commutes): True
no activation -> one matrix W1@W2 of shape (8, 8) : True
with ReLU the same shortcut fails; max diff: 2.3679778575897217
SwiGLU output shape: (3, 8) | elementwise gate in [ -0.278 , 3.929 ]
```

The first line shows that the block is position-wise: permuting the rows of the input permutes the output the same way, so no information moves between tokens here; all the mixing between tokens is done by attention. The next two lines are linear algebra in one sentence: without the non-linearity, $(xW_1)W_2 = x(W_1W_2)$, a single $d \times d$ matrix, so a stack of linear layers adds no capacity. With ReLU the identity fails (difference 2.37). The SiLU gate has a small negative range (minimum $-0.278$ here), unlike ReLU.

The middle panel counts where the parameters sit. With $d=4096$ and $d_{ff}=\tfrac23\cdot4d$, the gated FFN has $3\,d\,d_{ff}=134.2$M parameters per layer and attention has $4d^2=67.1$M, so the FFN holds two thirds of the block. The parameter count is exact for these formulas; it ignores biases, which LLaMA-style layers omit, the norm weights, and the embedding table.

## 3. Rank: what LoRA assumes

A matrix $W \in \mathbb{R}^{d\times d}$ has rank at most $d$, and its [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) is $W=\sum_i \sigma_i u_i v_i^{\top}$. The best rank-$r$ approximation keeps the $r$ largest terms, and its relative error is the norm of the discarded singular values (the Eckart-Young theorem). The LoRA paper proposes to keep the pretrained matrix frozen and learn only its change as a product of two thin matrices:

$$W' = W + \Delta W,\qquad \Delta W = BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times d},\ r\ll d,$$

with $B$ initialised at zero. It reports that a very low rank suffices, even $r=1$ or $2$ for GPT-3 175B with $d=12{,}288$, and that the adapted matrices can be merged with the frozen weights, with no inference latency. The experiment below builds a frozen full-rank matrix, a target update of true rank 4 plus small noise, and fits $BA$ by Adam.

```python
import torch, numpy as np
torch.manual_seed(0)
d, r0 = 256, 4
W = torch.randn(d, d) / d**.5                                   # a frozen "pretrained" matrix
dW = torch.randn(d, r0) @ torch.randn(r0, d) / d**.5             # a target update of true rank 4
dW = dW + 0.01 * torch.randn(d, d) / d**.5                       # plus a little noise
print("rank(W)      =", int(torch.linalg.matrix_rank(W)), "of", d)
print("rank(target) =", int(torch.linalg.matrix_rank(dW, rtol=0.05)), "(tolerance 5% of the largest singular value)")
S = torch.linalg.svdvals(dW); print("top 6 singular values of target:", [round(float(s), 3) for s in S[:6]])
print(f"\n{'r':>3} {'params 2dr':>11} {'vs d^2':>8} {'rel. error of BA':>17}")
for r in (1, 2, 4, 8, 16):
    B = torch.zeros(d, r, requires_grad=True); A = (torch.randn(r, d) * 0.1).requires_grad_()   # B=0 as in LoRA
    opt = torch.optim.Adam([A, B], lr=1e-2)
    for _ in range(3000):
        loss = ((B @ A - dW) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    err = float((B @ A - dW).norm() / dW.norm())
    print(f"{r:>3} {2*d*r:>11} {100*2*d*r/d**2:>7.1f}% {err:>17.4f}")
# the best possible rank-r error, from the SVD (Eckart-Young)
U, S, Vh = torch.linalg.svd(dW)
print("\nSVD floor (best rank-r):", {r: round(float(S[r:].norm() / dW.norm()), 4) for r in (1, 2, 4, 8, 16)})
# merging: W + BA is an ordinary d x d matrix
print("merged W + B@A has rank", int(torch.linalg.matrix_rank(W + B @ A)), "- no extra inference cost")
```

```
rank(W)      = 256 of 256
rank(target) = 4 (tolerance 5% of the largest singular value)
top 6 singular values of target: [17.677, 16.922, 15.202, 13.437, 0.019, 0.019]

  r  params 2dr   vs d^2  rel. error of BA
  1         512     0.8%            0.8311
  2        1024     1.6%            0.6383
  4        2048     3.1%            0.0050
  8        4096     6.2%            0.0050
 16        8192    12.5%            0.0050

SVD floor (best rank-r): {1: 0.8311, 2: 0.6383, 4: 0.005, 8: 0.0048, 16: 0.0045}
merged W + B@A has rank 256 - no extra inference cost
```

The target's singular values are four large numbers and then $0.019$, so its numerical rank is 4. The fitted error drops from $0.64$ at $r=2$ to $0.005$ at $r=4$ and stays there (right panel). It matches the SVD floor at each rank, so Adam reaches the best rank-$r$ solution here. The parameter counts are $2dr$ against $d^2$: at $r=4$ and $d=256$, 3.1%; the saving grows with $d$ because $2dr/d^2 = 2r/d$. The last line confirms the merge: $W+BA$ is an ordinary full-rank matrix, so serving costs nothing extra.

This is a toy with the property built in. Whether real adaptation updates are low-rank is an empirical claim; the LoRA paper tests it on RoBERTa, DeBERTa, GPT-2 and GPT-3 and reports that the update has a low intrinsic rank. Its own GPT-2 Medium table shows validation loss flat at 1.16 from $r=16$ to $r=512$, and best near $r=4$ to $r=16$ depending on the metric.

## Which maths, which component

- A matrix product $XW$ is a linear layer; column blocks of $W$ are heads; the output matrix $W^O$ mixes them.
- Parameter count $4d^2$ for attention and $3\,d\,d_{ff}$ for the gated FFN follow from the shapes alone.
- Composition of linear maps is a single matrix; the non-linearity in the FFN is what prevents that collapse.
- Rank and singular values quantify how much of a matrix a few directions carry; LoRA bets that weight updates are of low rank.

Next: [Part 3, calculus, backpropagation and the residual path](/posts/maths-for-llms-3-calculus-backprop-residual/).

## Limits

- The attention code is a single layer on random inputs with random weights; it demonstrates equivalences (fused against per-head, against PyTorch), not what trained heads do.
- The LoRA experiment plants a rank-4 target, so it shows that low-rank fitting works when the update is low-rank, not that real updates are. The evidence for the latter is in the paper, on the models and tasks it lists.
- The parameter counts assume LLaMA-style layers without biases and with $d_{ff}=\tfrac23\cdot4d$; real checkpoints round $d_{ff}$ to a convenient multiple.
- The statement that the $1/\sqrt{d_k}$ scaling keeps logits at unit scale holds for independent unit-variance entries; trained vectors do not satisfy that.
- Not covered: eigenvalue analysis of weight matrices, grouped-query attention, mixture-of-experts layers, and the cost of attention in sequence length (Part 8).
