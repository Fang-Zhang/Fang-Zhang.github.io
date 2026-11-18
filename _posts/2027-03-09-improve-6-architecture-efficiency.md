---
title: "Improving LLM Performance (6/8): Architecture and Decoding Efficiency"
date: 2027-03-09 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, architecture, mixture-of-experts, attention, speculative-decoding, series]
description: "Why the KV cache dominates decoding memory and how grouped-query attention shrinks it, what a mixture of experts buys per unit of compute, and how speculative decoding trades a cheap draft model for fewer expensive calls, each with a measured toy run."
---

[Part 5](/posts/improve-5-inference-time-compute/) spent more compute per answer. This part asks how to make each unit of compute buy more, through the model's structure and the decoding loop. It covers three mechanisms, each with the published claim and a measured toy: the KV cache and grouped-query attention, mixture of experts, and speculative decoding.

## The KV cache and attention variants

During decoding, every new token attends to the keys and values of all earlier tokens, which are kept in a cache. For $L$ layers, $n_{kv}$ key-value heads of width $d_h$, context length $T$, batch $B$ and $b$ bytes per number, the cache holds

$$\text{bytes}=2\,L\,n_{kv}\,d_h\,T\,B\,b$$

and each step reads all of it. [Shazeer (2019)](https://arxiv.org/abs/1911.02150) observed that incremental decoding is slow because of the memory-bandwidth cost of repeatedly loading the large key and value tensors, and proposed multi-query attention, where all heads share one key-value head, at only minor quality cost. [Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245) introduce grouped-query attention, with an intermediate number of key-value heads, and report quality close to multi-head attention at speed comparable to multi-query attention. [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) attacks a different cost: it is an exact attention algorithm that tiles the computation to reduce reads and writes between GPU high-bandwidth memory and on-chip SRAM.

Measured here: one decoding step of attention with 32 query heads of width 64, batch 8, for 1, 4, 8 or 32 key-value heads (the cache sizes are computed for 32 layers and 8k context in fp16; the timings are fp32 tensors on this machine's CPU, a layer at a time):

```python
def step(q, K, V, n_kv):                   # one decoding step of attention over a cache of length T
    g = H // n_kv; Bq = q.shape[0]; qq = q.view(Bq, n_kv, g, DH)
    w = torch.softmax(qq @ K.transpose(-1, -2) / DH ** 0.5, -1); return (w @ V).reshape(Bq, H * DH)
```

```
 kv heads  cache MB/seq/layer @8k (fp16)  cache GB, 32 layers, batch 8, 8k (fp16) | decode-step attention time per layer, ms, T=1k / 4k / 16k
       32                           64.0                                     16.0 | 8.14 / 31.98 / 128.63
        8                           16.0                                      4.0 | 0.78 / 8.34 / 44.68
        4                            8.0                                      2.0 | 0.39 / 4.99 / 35.74
        1                            2.0                                      0.5 | 0.21 / 0.90 / 22.27
```

Cutting key-value heads from 32 to 8 cuts the cache from 16 GB to 4 GB, and the time per step at 16k context from 129 ms to 45 ms; one head gives 0.5 GB and 22 ms. The time falls less than the memory because the query-side work does not shrink. The quality side of the trade is not measured here; it is the claim of the two papers above.

## Mixture of experts

[Fedus et al. (2021)](https://arxiv.org/abs/2101.03961) describe the Switch Transformer: a sparsely activated model, with a very large number of parameters but constant computational cost per token, where each input selects different parameters. [Mixtral 8x7B (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) uses 8 feed-forward experts per layer with a router choosing two per token, so each token has access to 47B parameters but uses 13B.

The toy: a byte-level language model on Python source (Part 1's setup, 4 layers, 64 wide, 8M training bytes, 2 seeds), with the feed-forward block either dense, or replaced by 8 experts with top-1 routing and the Switch load-balancing loss, or a dense block 8 times as wide.

```python
class MoEFF(nn.Module):                      # Switch-style: each token goes to the one expert with the highest router probability
    def __init__(s, d, E, mult=4):
        super().__init__(); s.E = E; s.router = nn.Linear(d, E, bias=False); s.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d)) for _ in range(E)]); s.aux = 0.0
    def forward(s, x):
        sh = x.shape; f = x.reshape(-1, sh[-1]); p = F.softmax(s.router(f), -1); gate, idx = p.max(-1); out = torch.zeros_like(f)
        for e in range(s.E):
            sel = idx == e
            if sel.any(): out[sel] = s.experts[e](f[sel]) * gate[sel, None]
        frac = F.one_hot(idx, s.E).float().mean(0); s.aux = s.E * (frac * p.mean(0)).sum(); return out.reshape(sh)
```

```
dense                  parameters in blocks   200,064  used per token   200,064  validation loss 1.4962 / 1.4895 (nats per byte, 2 seeds)
MoE 8 experts, top-1   parameters in blocks 1,128,576  used per token   200,320  validation loss 1.4139 / 1.4390 (nats per byte, 2 seeds)
dense x8 wide          parameters in blocks 1,124,736  used per token 1,124,736  validation loss 1.3684 / 1.3575 (nats per byte, 2 seeds)
```

At the same 200k parameters used per token, the MoE model reaches 1.414 and 1.439 against 1.496 and 1.490 for the dense one. It does not reach the dense model with the same total parameter count (1.368 and 1.358), which also spends 5.6 times more compute per token. The stored parameters help, but they are worth less than parameters every token uses.

## Speculative decoding

[Leviathan et al. (2022)](https://arxiv.org/abs/2211.17192) sample from a large model faster without changing its outputs: a cheaper draft model proposes several tokens, the large model scores them in one parallel call, and each proposal is accepted with probability $\min(1,p/q)$, with a replacement drawn from the residual distribution after the first rejection. They report 2x to 3x acceleration on T5-XXL with identical outputs. With acceptance rate $\alpha$ and $\gamma$ proposals per call, the expected number of tokens per call of the large model is $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$.

The toy: a target byte model (96 wide, 4 layers) and a draft (24 wide, 1 layer), both trained on 4M bytes of Python source, sampled at temperature 1. The core of one iteration:

```python
def spec_step(x, gamma, g, stats):
    """One iteration of speculative sampling (Leviathan et al.): the draft proposes gamma bytes, the target scores them all in one call, each is accepted with probability min(1, p/q); on the first rejection a byte is drawn from the residual max(0, p-q)."""
    q_list = []; y = x
    for _ in range(gamma):
        q = probs(draft, y); t = torch.multinomial(q, 1, generator=g); q_list.append(q); y = torch.cat([y, t.view(1, 1)], 1)
    L0 = x.shape[1]; logits = F.softmax(target(y[:, -CTX:]), -1)[0]; off = logits.shape[0] - (y.shape[1] - L0) - 1      # target distribution at every proposed position, one call
    stats['target_calls'] += 1; out = []
    for i in range(gamma):
        p = logits[off + i]; q = q_list[i]; tok = int(y[0, L0 + i]); stats['proposed'] += 1
        if torch.rand(1, generator=g).item() < min(1.0, (p[tok] / q[tok]).item()): out.append(tok); stats['accepted'] += 1
        else:
            r = torch.clamp(p - q, min=0); r = r / r.sum(); out.append(int(torch.multinomial(r, 1, generator=g))); return out
    out.append(int(torch.multinomial(logits[off + gamma], 1, generator=g))); return out
```

```
cost ratio c = draft call time / target call time = 0.14
gamma=1  acceptance 0.490  bytes per target call 1.49  (formula 1.49)  predicted speed-up 1.30x
gamma=2  acceptance 0.508  bytes per target call 1.76  (formula 1.77)  predicted speed-up 1.37x
gamma=4  acceptance 0.506  bytes per target call 1.93  (formula 1.96)  predicted speed-up 1.25x
gamma=8  acceptance 0.491  bytes per target call 1.95  (formula 1.96)  predicted speed-up 0.92x
total variation distance to the target distribution over 4000 first bytes: speculative 0.029, direct sampling 0.023
```

![Left: KV cache size as bars and decode-step attention time as a line against the number of key-value heads. Middle: validation loss of the dense, MoE and wide dense models. Right: measured and predicted bytes per target call, and predicted speed-up, against the number of proposed bytes](/assets/pic/improve-6-efficiency.png)
*Left: fewer key-value heads shrink the cache and the step time. Middle: eight experts at constant per-token compute beat the dense model but not the dense model with the same total size. Right: the expected bytes per call follow the formula, and the speed-up peaks at 2 proposals and falls below 1 at 8.*

- **The formula matches the runs.** Measured bytes per target call (1.49, 1.76, 1.93, 1.95) sit within 0.03 of the formula using the measured acceptance rate of about 0.5.
- **More proposals stop helping.** At an acceptance rate near 0.5, going from 4 to 8 proposals adds almost nothing per call, while the draft cost grows linearly. With the measured cost ratio $c=0.14$ (draft call time over target call time), the predicted speed-up $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$ is 1.30, 1.37, 1.25 and 0.92 for $\gamma=1,2,4,8$. This is the formula's prediction and not a measured wall-clock time.
- **The output distribution is preserved.** Over 4,000 trials at a fixed prefix, the total variation distance between the first speculative byte and the target's distribution is 0.029, against 0.023 for sampling the target directly, which is the noise floor for that sample size.

[Part 7](/posts/improve-7-technical-reports/) turns from mechanisms to practice: what public technical reports say about how these levers are combined in deployed models.

## Limits

- The attention timings are one layer of attention only, fp32 on a shared CPU, so they show the direction of the memory effect and not GPU throughput. Quality under fewer key-value heads was not measured.
- The MoE toy is small, has no expert parallelism, trains for under 2,000 steps per model, and its loop-over-experts implementation says nothing about speed. Router collapse and instability, the issues the Switch paper addresses, were not studied.
- The speculative-decoding toy runs without a KV cache and with models that are tiny, so the cost ratio and the speed-up are those of this setup; the speed-up is computed from the formula, not timed end to end.
- Published figures (the 2x to 3x speed-up, the 47B and 13B parameter counts) are quoted from the papers and were not reproduced.
