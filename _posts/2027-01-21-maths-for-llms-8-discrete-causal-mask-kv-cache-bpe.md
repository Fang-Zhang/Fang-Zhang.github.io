---
title: "Maths for LLMs, Tested in Code (8/10): Discrete Maths, the Causal Mask, KV Cache and BPE"
date: 2027-01-21 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, graphs, kv-cache, bpe, tokenisation, series]
math: true
description: "Part 8 of 10. The causal mask is the adjacency matrix of a directed acyclic graph; the KV cache is a counting argument; byte-pair encoding is a counting algorithm; the space of sequences is too large to enumerate. Each is checked in code, with the BPE run on a real novel."
---

Chapter 8 of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) is discrete mathematics: sets, graphs, combinatorics and algorithms. Four LLM components are discrete rather than continuous: the attention mask (a graph), the key-value cache (a count of operations), the tokeniser (a counting algorithm) and the space of possible texts (a combinatorial number). One script covers all four; it ran on this machine (CPU, PyTorch for the attention check). [Part 6](/posts/maths-for-llms-6-geometry-dot-products-attention-scale/) treated the scores inside the softmax; this part treats which scores exist at all.

![Three panels. Left: an 8 by 8 causal mask, with 1 on and below the diagonal and minus infinity above. Middle: log-log plot of query-key products against generated tokens, with the KV-cache curve n(n+1)/2 below the recompute curve n(n+1)(n+2)/6. Right: corpus length in thousand tokens against the number of BPE merges, falling from 693 thousand characters to 268 thousand tokens after 300 merges.](/assets/pic/maths-llms-8-discrete.png)
*Left: the mask as an adjacency matrix. Middle: the KV-cache saving. Right: BPE merges on one novel.*

```python
import re, collections, numpy as np, torch
# 1. Causal mask = lower-triangular adjacency matrix of a DAG; information flow = reachability
T = 6
mask = np.tril(np.ones((T, T), dtype=int))
print("1. causal mask (row i may read column j):"); print(mask)
print("   edges:", int(mask.sum()), "= T(T+1)/2 =", T * (T + 1) // 2, "; strictly lower (no self-loops):", int(np.tril(mask, -1).sum()))
print("   mask is a DAG: strictly-lower matrix is nilpotent, M^T = 0 ->", not np.linalg.matrix_power(np.tril(mask, -1), T).any())
# changing token j must change only outputs i >= j (check on a real attention layer)
torch.manual_seed(0); d = 8; X = torch.randn(T, d); Wq, Wk, Wv = (torch.randn(d, d) for _ in range(3))
def attn(X):
    s = (X @ Wq) @ (X @ Wk).T / d ** 0.5; s = s.masked_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))
    return torch.softmax(s, -1) @ (X @ Wv)
base = attn(X)
for j in (0, 3, 5):
    X2 = X.clone(); X2[j] += 1.0; changed = (attn(X2) - base).abs().sum(1) > 1e-6
    print(f"   perturb token {j}: outputs changed at positions {[i for i in range(T) if changed[i]]}")
# 2. Cost of generation with and without a KV cache (counting query-key products), and cache memory
def products(n, cache):                      # number of q.k dot products to generate n tokens after a prompt of 0
    return sum(t for t in range(1, n + 1)) if cache else sum(t * (t + 1) // 2 for t in range(1, n + 1))
print("\n2. dot products to generate n tokens (per head per layer):")
for n in (100, 1000, 10000): print(f"   n={n:>6}: with cache {products(n, True):>14,}   without {products(n, False):>18,}   ratio {products(n, False) / products(n, True):.0f}")
print("   closed forms: n(n+1)/2 with cache; n(n+1)(n+2)/6 without ->", 1000 * 1001 // 2, 1000 * 1001 * 1002 // 6)
L, H, dh = 32, 32, 128; bytes_per = 2
print(f"   cache memory per token for L={L} layers, H={H} heads, d_head={dh}, fp16: 2*L*H*d_head*2 bytes = {2*L*H*dh*bytes_per/2**20:.2f} MiB; 4096 tokens -> {2*L*H*dh*bytes_per*4096/2**30:.2f} GiB")
# 3. BPE as a counting algorithm (Sennrich et al.): repeatedly merge the most frequent adjacent pair
text = open('pg.txt', encoding='utf-8').read(); text = text[text.index('*** START'):text.index('*** END')]
words = collections.Counter(re.findall(r"[a-z]+", text.lower()))
vocab = {tuple(w) + ('</w>',): c for w, c in words.items()}
merges = []
for step in range(300):
    pairs = collections.Counter()
    for w, c in vocab.items():
        for a, b in zip(w, w[1:]): pairs[(a, b)] += c
    (a, b), c = pairs.most_common(1)[0]; merges.append(((a, b), c))
    new = {}
    for w, cnt in vocab.items():
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == a and w[i + 1] == b: out.append(a + b); i += 2
            else: out.append(w[i]); i += 1
        new[tuple(out)] = cnt
    vocab = new
    if step in (0, 1, 2, 9, 49, 99, 299):
        
        n_tok = sum(len(w) * c for w, c in vocab.items())
        print(f"3. after {step+1:>3} merges: tokens in corpus {n_tok:>8,}  (merge {a!r}+{b!r}, count {c:,})") if step == 0 else print(f"   after {step+1:>3} merges: tokens in corpus {n_tok:>8,}  (merge {a!r}+{b!r}, count {c:,})")
chars = sum(len(w) * c for w, c in words.items()); print(f"   characters (+1 end marker per word) = {sum((len(w)+1)*c for w, c in words.items()):,}")
# 4. Counting sequences: why enumeration is impossible
import math
print("\n4. number of token sequences of length n for vocabulary V:")
for V in (50, 50000):
    print(f"   V={V:>5}: n=10 -> 10^{10*math.log10(V):.1f};  n=100 -> 10^{100*math.log10(V):.0f}")
print("   atoms in the observable universe are usually quoted as roughly 10^80 (order of magnitude)")
```

```
1. causal mask (row i may read column j):
[[1 0 0 0 0 0]
 [1 1 0 0 0 0]
 [1 1 1 0 0 0]
 [1 1 1 1 0 0]
 [1 1 1 1 1 0]
 [1 1 1 1 1 1]]
   edges: 21 = T(T+1)/2 = 21 ; strictly lower (no self-loops): 15
   mask is a DAG: strictly-lower matrix is nilpotent, M^T = 0 -> True
   perturb token 0: outputs changed at positions [0, 1, 2, 3, 4, 5]
   perturb token 3: outputs changed at positions [3, 4]
   perturb token 5: outputs changed at positions [5]

2. dot products to generate n tokens (per head per layer):
   n=   100: with cache          5,050   without            171,700   ratio 34
   n=  1000: with cache        500,500   without        167,167,000   ratio 334
   n= 10000: with cache     50,005,000   without    166,716,670,000   ratio 3334
   closed forms: n(n+1)/2 with cache; n(n+1)(n+2)/6 without -> 500500 167167000
   cache memory per token for L=32 layers, H=32 heads, d_head=128, fp16: 2*L*H*d_head*2 bytes = 0.50 MiB; 4096 tokens -> 2.00 GiB
3. after   1 merges: tokens in corpus  669,320  (merge 'e'+'</w>', count 23,226)
   after   2 merges: tokens in corpus  654,845  (merge 't'+'h', count 14,475)
   after   3 merges: tokens in corpus  640,540  (merge 'd'+'</w>', count 14,305)
   after  10 merges: tokens in corpus  566,492  (merge 'o'+'</w>', count 6,721)
   after  50 merges: tokens in corpus  429,633  (merge 'w'+'as</w>', count 1,875)
   after 100 merges: tokens in corpus  362,955  (merge 's'+'e</w>', count 1,004)
   after 300 merges: tokens in corpus  267,687  (merge 'all'+'y</w>', count 248)
   characters (+1 end marker per word) = 692,546

4. number of token sequences of length n for vocabulary V:
   V=   50: n=10 -> 10^17.0;  n=100 -> 10^170
   V=50000: n=10 -> 10^47.0;  n=100 -> 10^470
   atoms in the observable universe are usually quoted as roughly 10^80 (order of magnitude)
```

## 1. The causal mask is a DAG

A decoder lets position $i$ read position $j$ only if $j\le i$. [Vaswani et al.](https://arxiv.org/abs/1706.03762) implement this by masking out, setting to $-\infty$, all softmax inputs that correspond to illegal connections. As a graph, positions are nodes and the mask is the adjacency matrix $M_{ij}=[\,j\le i\,]$. It has $T(T+1)/2$ ones, 21 for $T=6$. Dropping the diagonal leaves a strictly lower-triangular matrix, which is nilpotent: $M^T=0$, meaning no path is longer than $T-1$ edges, which is what acyclic means. The perturbation test makes the consequence concrete on a real attention layer: changing token 3 changes the outputs at positions 3 and 4 only (a position counts as changed if its output differs by more than $10^{-6}$), changing token 5 changes only position 5, and changing token 0 changes all of them. This is the property that lets a single forward pass train on every prefix at once, and that makes generation sequential.

## 2. The KV cache is a counting argument

Generating the $t$-th token needs the query of position $t$ against $t$ keys. With cached keys and values, each new token costs $t$ dot products, so $n$ tokens cost $\sum_{t=1}^{n}t=n(n+1)/2$. Without the cache, every step recomputes the scores of all earlier positions: step $t$ costs $t(t+1)/2$ and the total is $\sum_{t=1}^{n}t(t+1)/2=n(n+1)(n+2)/6$. The printed counts match both closed forms (500,500 and 167,167,000 at $n=1000$), so the cache saves a factor of about $n/3$ in this count (334 at $n=1000$, 3,334 at $n=10{,}000$; middle panel). The price is memory: per token the cache stores one key and one value vector per head per layer, $2\cdot L\cdot H\cdot d_{\text{head}}$ numbers. For 32 layers, 32 heads of width 128 and 2-byte numbers that is 0.50 MiB per token and 2.00 GiB at 4,096 tokens. These dimensions are an example chosen for the arithmetic, not a specific model's configuration. The count covers attention scores only; the matrix products elsewhere in the block are linear in the number of tokens processed and are not included.

## 3. Byte-pair encoding is counting

[Sennrich et al.](https://arxiv.org/abs/1508.07909) adapt the byte-pair-encoding compression algorithm to word segmentation. The procedure is: start from characters, count every adjacent pair of symbols over the corpus (weighted by word frequency), merge the most frequent pair into a new symbol, repeat. Block 3 runs it on the same Project Gutenberg novel as Part 1. The first merge joins `e` and the end-of-word marker, 23,226 times; the second is `th`; by 50 merges whole words such as `was` appear. The 692,546 symbols (characters plus one end marker per word) shrink to 566,492 after 10 merges, 429,633 after 50, 362,955 after 100 and 267,687 after 300. Each merge adds one vocabulary entry and removes as many tokens as the pair's count, so the curve flattens as the pair counts fall: the first merge saves 23,226, the 300th only 248. Choosing the vocabulary size is a choice of how far along this curve to stop, trading a longer embedding table against longer sequences (which, by Section 2, cost more attention). Production tokenisers differ in detail (for example byte-level variants); only the counting idea is tested here.

## 4. The space of sequences

The number of token sequences of length $n$ over a vocabulary of $V$ tokens is $V^n$: $10^{17}$ for $V=50$, $n=10$, and $10^{47}$ for $V=50{,}000$, $n=10$; at $n=100$, $10^{470}$. The number of atoms in the observable universe is usually quoted as about $10^{80}$. A model therefore cannot store a table of continuations, which is why Part 4 wrote the sequence probability as a product of conditionals computed by a function.

## Which maths, which component

- Adjacency matrix and nilpotency of a DAG: causal masking and why training is parallel while generation is not.
- Summation formulas (triangular and tetrahedral numbers): KV-cache savings.
- Frequency counting of adjacent pairs: BPE tokenisation.
- Exponentials in sequence length: why next-token factorisation is needed.

Next: [Part 9, information, cross-entropy, perplexity and KL](/posts/maths-for-llms-9-information-entropy-cross-entropy-kl/).

## Limits

- The operation count treats one dot product as one unit and ignores memory bandwidth, which usually dominates real decoding speed; the ratio is a count, not a speed-up.
- The perturbation test checks the mask on one random layer, as an illustration rather than a proof.
- BPE here runs on the words of one novel, lower-cased and without punctuation; the merge numbers are specific to that text, and the code merges inside words only.
- Graph algorithms on computation graphs (topological sorting in autograd) and combinatorial structure in mixture-of-experts routing are not covered.
