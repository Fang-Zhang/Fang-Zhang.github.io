---
title: "Maths for LLMs (4/10): Probability, Softmax, Temperature and Top-p"
date: 2027-01-17 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, probability, sampling, softmax, series]
math: true
description: "Part 4 of 10. A language model outputs a probability distribution over the next token, and everything after that is probability: the chain rule that defines the sequence probability, softmax with temperature, top-k and top-p truncation, and expectation and variance of what is sampled. Measured on a small character model fitted to a novel."
---

Chapter 4 of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) is probability: random variables, distributions, softmax, expectation, variance, entropy, conditional and joint probability. In an LLM these appear in the last layer and in the decoding loop: the network produces logits, softmax turns them into a distribution over the vocabulary, and a sampling rule picks the next token. To keep every number checkable, the model here is a character-level count model with a three-character context, fitted to one public-domain novel. It is not a neural network, but its output has the same type as an LLM's, a distribution over the next token, and the decoding mathematics is identical. Scripts ran on this machine (CPU). [Part 3](/posts/maths-for-llms-3-calculus-backprop-residual/) ended with the cross-entropy gradient $\operatorname{softmax}(z)-\text{onehot}(y)$; this part looks at what that distribution is used for.

![Three panels. Left: next-character probability against rank after the context "the " at temperatures 0.3, 1 and 3; low temperature concentrates mass on four characters, high temperature flattens it. Middle: cumulative probability of the sorted characters, crossing 0.9 at 16 characters. Right: nucleus size at p = 0.9 against the entropy of 2,615 contexts, rising from 1 to 17 characters.](/assets/pic/maths-llms-4-probability.png)
*Left: temperature. Middle: the top-p cut-off. Right: the nucleus size follows the entropy of the context.*

## 1. The chain rule defines the model

An autoregressive model never writes down $P(\text{sentence})$ directly. It defines it by the chain rule of probability:

$$P(x_1,\dots,x_n)=\prod_{i=1}^{n}P(x_i\mid x_{<i}),\qquad \log P(x_{1:n})=\sum_{i=1}^{n}\log P(x_i\mid x_{<i}).$$

Training maximises the right-hand sum on the training text; this is the cross-entropy loss of Part 9. The count model below truncates the condition $x_{<i}$ to the previous three characters and estimates each factor by counting with add-0.1 smoothing.

```python
# A count-based character model (context = previous 3 characters) fitted to one public-domain novel.
# It is not a neural network, but it produces what an LLM's last layer produces: a distribution over the next token.
import re, collections, math, numpy as np
t = open('pg.txt', encoding='utf-8').read(); t = t[t.index('*** START'):t.index('*** END')]
t = re.sub(r'\s+', ' ', t.lower()); t = re.sub(r"[^a-z' .,;!?-]", '', t)
chars = sorted(set(t)); V = len(chars); ix = {c: i for i, c in enumerate(chars)}; K = 3
counts = collections.defaultdict(lambda: np.zeros(V))
for i in range(len(t) - K): counts[t[i:i+K]][ix[t[i+K]]] += 1
def dist(ctx, alpha=0.1):                      # add-alpha smoothing
    c = counts[ctx.rjust(K)[-K:]] + alpha; return c / c.sum()
def logprob(s):
    return sum(math.log(dist(s[max(0, i-K):i].rjust(K))[ix[s[i]]]) for i in range(len(s)))
if __name__ == '__main__':
    print("characters:", len(t), " vocabulary V =", V, " contexts seen:", len(counts))
    s = "it is a truth universally acknowledged"
    lp = logprob(s)
    print(f"chain rule: log P(s) = sum_i log P(c_i | previous 3) = {lp:.2f} nats over {len(s)} characters = {lp/len(s):.3f} per character")
    print(f"           P(s) = {math.exp(lp):.3e};  uniform guess would give {V**-len(s):.3e}")
    p = dist("th"); top = np.argsort(-p)[:5]
    print("P(next | 'th'), top 5:", [(chars[j], round(float(p[j]), 3)) for j in top])
    H = lambda p: float(-(p * np.log2(p + 1e-12)).sum())
    print(f"entropy after 'th': {H(p):.2f} bits;  after 'xq ': {H(dist('xq ')):.2f} bits;  uniform: {math.log2(V):.2f} bits")
```

```
characters: 711613  vocabulary V = 33  contexts seen: 5285
chain rule: log P(s) = sum_i log P(c_i | previous 3) = -66.80 nats over 38 characters = -1.758 per character
           P(s) = 9.774e-30;  uniform guess would give 1.979e-58
P(next | 'th'), top 5: [('e', 0.666), ('a', 0.191), ('i', 0.078), ('o', 0.047), ('r', 0.014)]
entropy after 'th': 1.46 bits;  after 'xq ': 5.04 bits;  uniform: 5.04 bits
```

The probability of the 38-character sentence is $9.8\times10^{-30}$, against $2\times10^{-58}$ for a uniform guess over 33 characters; the sum of logs is the number to keep, since the product underflows quickly. The entropy of the next-character distribution after `th` is 1.46 bits (the model is confident: `e` has 0.67), and 5.04 bits after the unseen context `xq `, which is exactly the uniform value $\log_2 33$. The unseen context shows the limit of counting: the smoothing falls back to uniform. A neural network replaces the table with a function that generalises to contexts it has not seen (the author's summary of the motivation, not tested here).

## 2. Softmax, temperature and truncation

A network outputs logits $z\in\mathbb{R}^{|V|}$, and softmax with temperature $T$ gives

$$p_i(T)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}} .$$

This is the formula in [Holtzman et al.](https://arxiv.org/abs/1904.09751), who call it re-estimating the softmax with a temperature parameter. $T\to0$ approaches argmax, $T=1$ is the model's own distribution and $T\to\infty$ approaches uniform. Two truncation rules cut the tail before sampling. Top-$k$ keeps the $k$ most probable tokens. Top-$p$ (nucleus sampling) keeps the smallest set $V^{(p)}$ with $\sum_{x\in V^{(p)}}P(x)\ge p$ and renormalises, by the definition in the same paper: $P'(x)=P(x)/p'$ inside the set and 0 outside, with $p'$ the mass of the set. The code implements both, from the formulas, on the model's distribution after the context `the `.

```python
import numpy as np
from lm import dist, chars, ix, V
rng = np.random.default_rng(0)
H = lambda p: float(-(p[p > 0] * np.log2(p[p > 0])).sum())
def with_temperature(p, T):
    z = np.log(p) / T; z -= z.max(); e = np.exp(z); return e / e.sum()
def top_k(p, k):
    q = np.zeros_like(p); i = np.argsort(-p)[:k]; q[i] = p[i]; return q / q.sum()
def top_p(p, pp):
    o = np.argsort(-p); c = np.cumsum(p[o]); n = int(np.searchsorted(c, pp) + 1)   # smallest set with mass >= pp
    q = np.zeros_like(p); q[o[:n]] = p[o[:n]]; return q / q.sum(), n
ctx = "the "
p = dist(ctx)
print(f"context {ctx!r}: top 4 =", [(chars[j], round(float(p[j]), 3)) for j in np.argsort(-p)[:4]])
print("\ntemperature T: entropy (bits), max prob, prob of the 10th most likely character")
for T in (0.3, 0.7, 1.0, 1.5, 3.0):
    q = with_temperature(p, T); o = np.sort(q)[::-1]
    print(f"  T={T:<4} entropy {H(q):5.2f}   max {o[0]:.3f}   10th {o[9]:.4f}")
print("  T->0 is argmax; T->inf is uniform: entropy limit", round(np.log2(V), 2), "bits")
print("\ntop-k vs top-p on the same distribution:")
for k in (1, 5, 10): q = top_k(p, k); print(f"  top-k k={k:<2}: support {int((q>0).sum()):>2}, entropy {H(q):.2f}")
for pp in (0.5, 0.9, 0.95):
    q, n = top_p(p, pp); print(f"  top-p p={pp}: support {n:>2}, entropy {H(q):.2f}")
# the nucleus is dynamic: the same p gives very different sizes in different contexts
print("\nnucleus size for p=0.9 in different contexts:")
for c in ("the ", " th", "and", "xq ", "elizabe", "mr. d"):
    print(f"  {c!r:<10} entropy {H(dist(c)):.2f} bits -> nucleus {top_p(dist(c), 0.9)[1]} of {V} characters")
# the sampler reproduces its distribution
q, n = top_p(p, 0.9); draws = rng.choice(V, size=200000, p=q); emp = np.bincount(draws, minlength=V) / len(draws)
print(f"\n200k draws from the p=0.9 nucleus of {ctx!r}: max |empirical - theoretical| = {np.abs(emp - q).max():.4f}")
# expectation and variance of a function of the sampled token: here, the length of the next 'word piece' is overkill; use the char code
vals = np.arange(V); mu = float((q * vals).sum()); var = float((q * (vals - mu) ** 2).sum())
print(f"E[index] = {mu:.2f}, Var[index] = {var:.2f} (formulas); sample mean {draws.mean():.2f}, sample var {draws.var():.2f}")
# generation
def gen(T, n=110, pp=None):
    s = "it is "
    for _ in range(n):
        q = with_temperature(dist(s), T)
        if pp: q, _ = top_p(q, pp)
        s += chars[rng.choice(V, p=q)]
    return s
print()
for T, pp in ((0.3, None), (1.0, None), (1.5, None), (1.5, 0.9)):
    print(f"T={T} top-p={pp}: {gen(T, pp=pp)}")
```

```
context 'the ': top 4 = [('w', 0.116), ('h', 0.093), ('c', 0.093), ('s', 0.09)]

temperature T: entropy (bits), max prob, prob of the 10th most likely character
  T=0.3  entropy  2.87   max 0.343   10th 0.0157
  T=0.7  entropy  3.95   max 0.155   10th 0.0413
  T=1.0  entropy  4.19   max 0.116   10th 0.0459
  T=1.5  entropy  4.40   max 0.088   10th 0.0474
  T=3.0  entropy  4.69   max 0.061   10th 0.0447
  T->0 is argmax; T->inf is uniform: entropy limit 5.04 bits

top-k vs top-p on the same distribution:
  top-k k=1 : support  1, entropy -0.00
  top-k k=5 : support  5, entropy 2.29
  top-k k=10: support 10, entropy 3.24
  top-p p=0.5: support  6, entropy 2.53
  top-p p=0.9: support 16, entropy 3.87
  top-p p=0.95: support 18, entropy 4.02

nucleus size for p=0.9 in different contexts:
  'the '     entropy 4.19 bits -> nucleus 16 of 33 characters
  ' th'      entropy 1.46 bits -> nucleus 3 of 33 characters
  'and'      entropy 0.65 bits -> nucleus 1 of 33 characters
  'xq '      entropy 5.04 bits -> nucleus 30 of 33 characters
  'elizabe'  entropy 0.07 bits -> nucleus 1 of 33 characters
  'mr. d'    entropy 0.94 bits -> nucleus 2 of 33 characters

200k draws from the p=0.9 nucleus of 'the ': max |empirical - theoretical| = 0.0015
E[index] = 17.87, Var[index] = 52.26 (formulas); sample mean 17.85, sample var 52.08

T=0.3 top-p=None: it is a stration as so may were the said her could be a see to see did not know must and to the was to her her and t
T=1.0 top-p=None: it is with company sity own his nowley. i she effor she to bennection est by the cord! affect of him.bwuypqpstame ti
T=1.5 top-p=None: it is evitqy.;g wicked his us very abspndinearenwqanwfnbug-xxjfz htyt,cjfeqqhwrc,q!plhhvwngdomissione solving, hat p
T=1.5 top-p=0.9: it is elicern so the his hapterways. welf;cbhhwu;wo,n jane. he willievotable. be detent hapt, are well though haved.
```

Read the output top to bottom:

- **Temperature.** Entropy rises monotonically with $T$, from 2.87 bits at $T=0.3$ to 4.69 at $T=3$, towards the limit 5.04 (left panel). At $T=0.3$ the top character has probability 0.343; at $T=3$ it has 0.061.
- **Top-$k$ against top-$p$.** Top-$k$ with $k=10$ always keeps 10 characters. Top-$p$ with $p=0.9$ kept 16 here, and the last table shows why that matters: after `and` the nucleus is 1 character, after `elizabe` also 1, after `xq ` it is 30 of 33. The size of the set tracks the entropy of the context (right panel: correlation 0.86 over 2,615 contexts, nucleus sizes from 1 to 17). A fixed $k$ cannot do that. This is the argument Holtzman et al. make for expanding and contracting the candidate pool dynamically.
- **Sampling reproduces the distribution.** 200,000 draws from the truncated distribution match it to 0.0015.
- **Expectation and variance.** For a function $g$ of the sampled token, $\mathbb{E}[g]=\sum_i p_i g(i)$ and $\operatorname{Var}[g]=\sum_i p_i\,(g(i)-\mathbb{E}[g])^2$. With $g$ the character index, the formulas give 17.87 and 52.26, and the sample gives 17.85 and 52.08.

The generated lines show the effect directly. At $T=0.3$ the text is common words repeated (`her could be a see to see did not know`); at $T=1$ it contains non-words; at $T=1.5$ it breaks into character noise (`evitqy.;g wicked his us very abspndinearenwqanwfnbug`); and $T=1.5$ with top-$p=0.9$ removes most of the noise but not all of it. With one run each these are illustrations, not a quality measurement. The paper's own comparison, with human evaluation on GPT-2 text, is the evidence for nucleus sampling in practice.

## Which maths, which component

- Chain rule of probability: the definition of the sequence probability and of the training objective.
- Softmax with temperature: the last layer's output and the first step of decoding.
- Cumulative sums on a sorted distribution: top-$p$; sorting plus a cut: top-$k$.
- Expectation, variance and entropy: what a decoding rule does to the spread of the next token.

Next: [Part 5, statistics, evaluation noise and scaling laws](/posts/maths-for-llms-5-statistics-evaluation-scaling/).

## Limits

- The model is a count table over three characters on one 19th-century English novel; real LLMs condition on thousands of tokens and are not count models. The decoding mathematics carries over, the specific numbers do not.
- Entropy and nucleus size depend on the smoothing constant (0.1 here) and on contexts seen only a few times; the correlation in the right panel uses contexts with at least 20 occurrences.
- Text samples come from one random seed and show qualitative behaviour only.
- Bayes' rule, joint and marginal distributions, and independence (the rest of the book's chapter) are not exercised here; they enter in Part 9 (mutual information) and in the evaluation statistics of Part 5.
