---
title: "Maths for LLMs, Tested in Code (9/10): Information, Cross-Entropy, Perplexity and KL"
date: 2027-01-22 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, information-theory, cross-entropy, perplexity, kl-divergence, rlhf, series]
math: true
description: "Part 9 of 10. The LLM training loss is a cross-entropy in bits; perplexity is its exponential; KL divergence is the gap between two models and the penalty in RLHF and distillation. All measured on held-out text, against gzip, bz2 and xz."
---

Chapter 9 of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) is information and entropy. Three quantities from it appear in every LLM paper: cross-entropy (the training loss), perplexity (its exponential) and KL divergence (the penalty in preference tuning and the objective in distillation). This part computes each on held-out text with the character models of [Part 4](/posts/maths-for-llms-4-probability-softmax-sampling/), compares them with general-purpose compressors, and shows the KL-regularised reward trade-off in closed form. Everything ran on this machine (CPU, NumPy); the same Project Gutenberg novel as in Part 1 is split 90/10 into training and held-out text.

![Three panels. Left: bar chart of bits per character on held-out text: raw ASCII 8, uniform 5.04, gzip 2.73, xz 2.07, bz2 1.74, and count models with context length 0 to 4 at 4.24, 3.39, 2.63, 2.04 and 1.82. Middle: the curve minus log2 p against p, with points at 1, 3.32 and 6.64 bits. Right: expected reward against KL divergence from the reference policy for a 10-output toy problem, rising from minus 0.64 to 1.04 as the KL grows to 12.6 bits.](/assets/pic/maths-llms-9-information.png)
*Left: held-out bits per character. Middle: the per-character loss. Right: reward bought per bit of KL.*

```python
import re, gzip, bz2, lzma, math, numpy as np
t = open('pg.txt', encoding='utf-8').read(); t = t[t.index('*** START'):t.index('*** END')]
t = re.sub(r'\s+', ' ', t.lower()); t = re.sub(r"[^a-z' .,;!?-]", '', t)
chars = sorted(set(t)); V = len(chars); ix = {c: i for i, c in enumerate(chars)}; a = np.array([ix[c] for c in t])
split = int(0.9 * len(a)); tr, te = a[:split], a[split:]; ntest = len(te)
print(f"characters: train {len(tr):,}, test {ntest:,}; vocabulary {V}; log2(V) = {math.log2(V):.3f} bits")
def cid(x, K): return sum(x[i:len(x) - K + i] * V ** (K - 1 - i) for i in range(K)) if K else np.zeros(len(x), dtype=int)
def fit(K, alpha):
    c = np.zeros((V ** K if K else 1, V)); np.add.at(c, (cid(tr, K), tr[K:]), 1); p = c + alpha; return p / p.sum(1, keepdims=True)
def loss_bits(p, K): return float(-np.log2(p[cid(te, K), te[K:]]).mean())
# 1. cross-entropy on held-out text, bits per character, as the context grows
print("\n1. held-out cross-entropy (bits/char) and perplexity of the count model, by context length K (add-alpha smoothing, best of 6 alphas):")
best = {}
for K in range(0, 5):
    r = min((loss_bits(fit(K, al), K), al) for al in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)); best[K] = r
    print(f"   K={K}: {r[0]:.3f} bits/char   perplexity 2^H = {2 ** r[0]:6.2f}   (alpha {r[1]})")
# 2. general-purpose compressors on the same held-out text: marginal cost of the test text given the training text
raw_t = ''.join(chars[i] for i in te).encode(); raw_tr = ''.join(chars[i] for i in tr).encode()
print("\n2. compressors on the same held-out text (bits/char, marginal cost given the training text):")
for name, f in (("gzip -9", lambda b: len(gzip.compress(b, 9))), ("bz2", lambda b: len(bz2.compress(b, 9))), ("xz", lambda b: len(lzma.compress(b, preset=9)))):
    alone = 8 * f(raw_t) / ntest; marg = 8 * (f(raw_tr + raw_t) - f(raw_tr)) / ntest
    print(f"   {name:<8} alone {alone:.3f}   marginal {marg:.3f}")
print(f"   raw (8 bits per ASCII character) = 8.000;  count model K=3 = {best[3][0]:.3f}")
# 3. KL divergence = cross-entropy - entropy, checked numerically on two distributions over the next character after 'th'
P = fit(3, 0.01); Q = fit(1, 0.01); ctx = tuple(chars.index(x) for x in "he ")      # context 'he '
p = P[cid(np.array(ctx + (0,)), 3)[0]]; q = Q[ix[' ']]
H = -(p * np.log2(p)).sum(); CE = -(p * np.log2(q)).sum(); KL = (p * np.log2(p / q)).sum()
print(f"\n3. next-char distributions after 'he ': P from K=3, Q from K=1.  H(P) = {H:.3f}, H(P,Q) = {CE:.3f}, KL(P||Q) = {KL:.3f};  H(P,Q) - H(P) = {CE - H:.3f}")
print(f"   KL is not symmetric: KL(Q||P) = {(q * np.log2(q / p)).sum():.3f}")
# 4. Over the whole test set: how many extra bits per character the weaker model costs
Pk3 = fit(3, 0.01); Pk1 = fit(1, 0.01)
extra = loss_bits(Pk1, 1) - loss_bits(Pk3, 3)
print(f"\n4. held-out loss, K=1 minus K=3: {extra:.3f} bits/char (extra bits paid for coding with the weaker model)")
# 5. KL penalty as used in RLHF (Ouyang et al.): reward r - beta*log(pi/pi_ref). Closed form: pi* proportional to pi_ref * exp(r/beta)
rng = np.random.default_rng(0); pref = np.sort(rng.dirichlet(np.ones(10)))[::-1]; r = rng.standard_normal(10)
print("\n5. KL-regularised reward maximisation over 10 outputs: optimum pi* = pi_ref exp(r/beta) / Z; expected reward and KL(pi*||pi_ref):")
for beta in (10.0, 1.0, 0.3, 0.1, 0.01):
    w = pref * np.exp(r / beta); w /= w.sum(); kl = float((w * np.log2(w / pref)).sum())
    print(f"   beta={beta:<5} E[r] = {float(w @ r):+.3f}  KL = {kl:6.3f} bits   max prob {w.max():.3f}   (reference E[r] = {float(pref @ r):+.3f}, best r = {r.max():+.3f})")
# 6. Mutual information of adjacent characters: I(X_t; X_{t+1}) = H(X_{t+1}) - H(X_{t+1} | X_t)
joint = np.zeros((V, V)); np.add.at(joint, (a[:-1], a[1:]), 1); joint /= joint.sum(); px = joint.sum(1); py = joint.sum(0)
nz = joint > 0; mi = float((joint[nz] * np.log2(joint[nz] / np.outer(px, py)[nz])).sum())
print(f"\n6. mutual information between adjacent characters: {mi:.3f} bits; H(X) = {-(py * np.log2(py)).sum():.3f} bits")
```

```
characters: train 640,451, test 71,162; vocabulary 33; log2(V) = 5.044 bits

1. held-out cross-entropy (bits/char) and perplexity of the count model, by context length K (add-alpha smoothing, best of 6 alphas):
   K=0: 4.235 bits/char   perplexity 2^H =  18.83   (alpha 1.0)
   K=1: 3.393 bits/char   perplexity 2^H =  10.50   (alpha 0.03)
   K=2: 2.635 bits/char   perplexity 2^H =   6.21   (alpha 0.03)
   K=3: 2.044 bits/char   perplexity 2^H =   4.12   (alpha 0.03)
   K=4: 1.822 bits/char   perplexity 2^H =   3.54   (alpha 0.03)

2. compressors on the same held-out text (bits/char, marginal cost given the training text):
   gzip -9  alone 2.908   marginal 2.728
   bz2      alone 2.365   marginal 1.735
   xz       alone 2.685   marginal 2.069
   raw (8 bits per ASCII character) = 8.000;  count model K=3 = 2.044

3. next-char distributions after 'he ': P from K=3, Q from K=1.  H(P) = 4.191, H(P,Q) = 4.469, KL(P||Q) = 0.277;  H(P,Q) - H(P) = 0.277
   KL is not symmetric: KL(Q||P) = 0.312

4. held-out loss, K=1 minus K=3: 1.346 bits/char (extra bits paid for coding with the weaker model)

5. KL-regularised reward maximisation over 10 outputs: optimum pi* = pi_ref exp(r/beta) / Z; expected reward and KL(pi*||pi_ref):
   beta=10.0  E[r] = -0.635  KL =  0.003 bits   max prob 0.428   (reference E[r] = -0.679, best r = +1.043)
   beta=1.0   E[r] = -0.385  KL =  0.180 bits   max prob 0.381   (reference E[r] = -0.679, best r = +1.043)
   beta=0.3   E[r] = -0.104  KL =  1.006 bits   max prob 0.646   (reference E[r] = -0.679, best r = +1.043)
   beta=0.1   E[r] = +0.979  KL = 11.590 bits   max prob 0.931   (reference E[r] = -0.679, best r = +1.043)
   beta=0.01  E[r] = +1.043  KL = 12.612 bits   max prob 1.000   (reference E[r] = -0.679, best r = +1.043)

6. mutual information between adjacent characters: 0.845 bits; H(X) = 4.223 bits
```

## 1. Cross-entropy is the training loss, in bits

For a model $q$ and the true next-token distribution $p$,

$$H(p,q)=-\sum_x p(x)\log_2 q(x),\qquad \text{estimated on text by}\quad \frac1N\sum_{i=1}^{N}-\log_2 q(x_i\mid x_{<i}).$$

This is the sum of logs from Part 4 divided by $N$, and it is what an LLM's training loss is, up to the base of the logarithm (training code uses natural logs, so a loss in nats is the value in bits times $\ln 2$). Perplexity is $2^{H}$ with bits, or $e^{H}$ with nats: the size of a uniform distribution that would be as uncertain. Block 1 gives a ladder on held-out text. Knowing nothing but the alphabet costs $\log_2 33=5.04$ bits per character. Letter frequencies alone (context length $K=0$) cost 4.24; each extra character of context lowers it: 3.39, 2.63, 2.04 and 1.82 bits for $K=1$ to 4, i.e. perplexities of 10.5, 6.2, 4.1 and 3.5 characters. A neural model is a better function for the same quantity, not a different quantity.

## 2. Compression is the same number

Shannon's source coding theorem ties cross-entropy to code length: an ideal code spends $-\log_2 q(x)$ bits on $x$. Block 2 gives gzip, xz and bz2 the same held-out text. Alone, gzip needs 2.91 bits per character, bz2 2.37 and xz 2.69. Given the training text first, so that they have the vocabulary to work with (the marginal cost, measured as the compressed size of train plus test minus the compressed size of train), gzip needs 2.73, xz 2.07 and bz2 1.74. The count model with $K=3$ scores 2.04 and with $K=4$ 1.82, so bz2 beats both on this measurement. That does not contradict anything: a compressor adapts to the test text as it reads it, while the count models are frozen after training, and the marginal-cost trick is a rough device (the author's reading of why bz2 does best: its largest block is 900 kB, larger than the training text, so the whole history is available to it; not tested separately). The comparison shows the equivalence (a probability model is a compressor, and the loss is its code length) and that a few-line count model is competitive with general-purpose compressors on this corpus, not that it is better.

## 3. KL divergence is the gap

$$D_{\mathrm{KL}}(p\Vert q)=\sum_x p(x)\log\frac{p(x)}{q(x)}=H(p,q)-H(p).$$

Block 3 checks the identity on the next-character distributions after the context `he `, from the $K=3$ model as $p$ and the $K=1$ model as $q$: $H(P)=4.191$, $H(P,Q)=4.469$, and both $H(P,Q)-H(P)$ and the directly computed KL equal 0.277 bits. KL is not symmetric (the reverse direction is 0.312). Minimising the cross-entropy over $q$ with $p$ fixed is the same as minimising KL, since $H(p)$ does not depend on $q$; this is why cross-entropy training is distribution matching. Block 4 reports the price of the weaker model over the whole held-out text: 1.346 bits per character more for $K=1$ than for $K=3$. In distillation, [Hinton et al.](https://arxiv.org/abs/1503.02531) train a small model on the class probabilities of a large one as soft targets, which are high-entropy and so carry more information per example than hard labels, in their description.

## 4. The KL penalty in preference tuning

[Ouyang et al.](https://arxiv.org/abs/2203.02155) describe adding a per-token KL penalty from the supervised model at each token to the reinforcement-learning reward, to mitigate over-optimisation of the reward model; their runs use a coefficient $\beta=0.02$ in that setup. For a single output with reward $r$ and reference policy $\pi_{\mathrm{ref}}$, maximising $\mathbb{E}_\pi[r]-\beta D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}})$ has the closed-form optimum $\pi^*(y)\propto\pi_{\mathrm{ref}}(y)\,e^{r(y)/\beta}$. Block 5 evaluates it on a toy with 10 outputs and random rewards. At $\beta=10$ the policy stays at the reference (KL 0.003 bits, expected reward $-0.635$ against $-0.679$). At $\beta=1$ the reward is $-0.385$ for 0.18 bits; at $\beta=0.3$ it is $-0.104$ for 1.0 bits; at $\beta=0.1$ it reaches $+0.979$ but the policy has moved 11.6 bits away, and at $\beta=0.01$ it puts probability 1.000 on the single best output (reward 1.043, KL 12.6 bits). The right panel is this trade-off curve: each bit of drift buys less reward than the one before it, and the last few bits buy almost none. The penalty is what keeps the policy from collapsing onto whatever the reward model happens to score highest.

## 5. Mutual information

Mutual information is the KL between a joint distribution and the product of its marginals, $I(X;Y)=D_{\mathrm{KL}}\big(p(x,y)\Vert p(x)p(y)\big)$. Block 6 estimates it for adjacent characters from counts: 0.845 bits, out of a single-character entropy of 4.223 bits. Knowing the previous character removes about a fifth of the uncertainty about the next one; the table of Block 1 shows the remainder shrinking as more context is added.

## Which maths, which component

- Logarithm of a probability, averaged: the pre-training loss and the reported perplexity.
- Source-coding bound: equating prediction with compression, which is why bits per character can be compared across models and compressors.
- KL divergence: distillation targets and the RLHF penalty; its non-symmetry matters (which distribution sits in which slot).
- Closed-form optimum of a KL-regularised objective: how $\beta$ controls the drift of a tuned policy.

Next: Part 10, advanced topics: SVD for compression and low-rank structure, and the Fourier view of sinusoidal position encodings.

## Limits

- Held-out numbers come from one novel, a 90/10 split and add-$\alpha$ smoothing with $\alpha$ chosen on the same held-out text (six values, so a small optimism in the count-model figures).
- The compressor comparison is not like for like: compressors adapt online and count models do not; the marginal-cost device is approximate.
- The RLHF toy has random rewards over 10 outputs and one prompt. The closed form is the standard one for a KL-regularised objective, and the Ouyang et al. training uses reinforcement learning on sequences with additional terms, not this closed form.
- Perplexity is comparable only for the same tokeniser and test set: a per-token and a per-character figure are different quantities (Part 8).
