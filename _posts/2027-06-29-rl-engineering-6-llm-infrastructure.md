---
title: "Reinforcement Learning Engineering (6/6): Generating Responses Took 37 to 71 Percent of an RL Iteration in a Small Language Model, and the Share Grew With Response Length"
date: 2027-06-29 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement learning, rlhf, grpo, llm, infrastructure, kv cache, series]
description: "Part 6 of 6 on reinforcement learning engineering. Reinforcement learning for language models seen as an infrastructure problem: rollout generation, KV caches, separate generation and training engines, asynchronous updates. A CPU measurement of the generation share."
---

## The loop is the same, the cost structure is not

Reinforcement learning for language models keeps the loop of Part 1: collect rollouts, then update. What changes is the price of each half. A rollout is an autoregressive generation of hundreds or thousands of tokens by a large model, and the update is a gradient step on those same tokens. The sources below treat this as a systems problem.

- **Pipeline.** [InstructGPT](https://arxiv.org/abs/2203.02155) fine-tunes GPT-3 on labeler demonstrations, collects rankings of model outputs, and then fine-tunes further "using reinforcement learning from human feedback". Its 1.3B model was preferred in human evaluation to the 175B GPT-3.
- **Algorithm change for memory.** [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduces Group Relative Policy Optimization (GRPO), "a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO".
- **Dataflow.** [HybridFlow](https://arxiv.org/abs/2409.19256) describes RLHF as a dataflow in which each node "is expanded into a distributed LLM training or generation program", and designs a 3D-HybridEngine "for efficient actor model resharding between training and generation phases". It reports 1.53 to 20.57 times higher throughput than state-of-the-art baselines.
- **Separate engines.** [OpenRLHF](https://arxiv.org/abs/2405.11143) names "inference bottlenecks" as a common problem of existing frameworks, builds on Ray, vLLM, DeepSpeed and Transformers, and reports speedups of 1.22 to 1.68 times over state-of-the-art frameworks.
- **Asynchrony.** [Asynchronous RLHF](https://arxiv.org/abs/2410.18252) calls the dominant synchronous, on-policy paradigm "computationally inefficient", proposes separating generation and learning, and finds online DPO "most robust to off-policy data". It reports training a LLaMA 3.1 8B chatbot on an instruction-following task about 40 percent faster than a synchronous run while matching performance, and about 70 percent faster on GSM8k with Rho 1B.

The speedups above are each paper's own, measured against its own baselines. The experiment below measures the one quantity all of them react to: how much of an iteration goes to generation.

## Experiment: the generation share on a small model

Setup: a 3.8-million-parameter GPT written from scratch in PyTorch (4 layers, width 256, random weights; no pretrained model is used), CPU with 16 threads. A batch of 16 prompts of 64 tokens generates 16, 64 or 256 new tokens by sampling. Generation is timed without and with a key-value (KV) cache, and compared with one training step (forward, backward, optimiser update) on the same 16 sequences. Medians of 3 repeats after one warm-up.

| New tokens | Generate, no cache | Generate, KV cache | One training step | Generation share |
|---|---|---|---|---|
| 16 | 0.46 s | 0.07 s | 0.11 s | 37% |
| 64 | 1.97 s | 0.19 s | 0.17 s | 53% |
| 256 | 17.35 s | 1.12 s | 0.45 s | 71% |

Generation share is the cached generation time divided by the sum of cached generation and one training step. The KV cache sped generation up by roughly 7, 10 and 15 times (from the rounded table values). A check that the cached step matches a full forward pass gave a maximum difference of 5.4e-07 in the final logits. With more gradient steps per batch, derived from the rounded values above, the share at 256 new tokens falls from 71 percent to 55 percent with 2 steps and to 38 percent with 4.

![Two panels. Left: log-scale seconds for generating with and without a KV cache and for one training step, at 16, 64 and 256 new tokens; generation with cache grows from 0.07 to 1.12 seconds while the training step grows from 0.11 to 0.45. Right: share of an iteration spent generating, 37, 53 and 71 percent]({{ site.baseurl }}/assets/pic/rl-eng-6-generation-cost.png){: width="900" }
_Figure: measured on the setup above; the left axis is logarithmic._

## Engineering reading

These points are this series' reading of the sources and the measurement.

1. **Generation cost grows with response length, training cost grows more slowly.** In the measurement, generating 16 times more tokens cost 16 times more cached generation time, against 4 times more for the training step. Reasoning-style outputs, which are long, therefore push the bottleneck toward the rollout.
2. **A cache is mandatory, not an optimisation.** Without it, 256 tokens took 17 seconds against 1.1.
3. **Generation and training want different engines.** The sources run generation in an inference engine (vLLM in OpenRLHF) and training in a training framework, which is why a system needs a way to move weights between them; HybridFlow's resharding engine targets exactly that step.
4. **Asynchrony buys throughput and costs freshness.** Rollouts generated by older weights are off-policy. The asynchronous paper treats the tolerable amount of staleness as the central question, and its answer depends on the algorithm.
5. **Algorithm choices are also cost choices.** The GRPO abstract presents the memory saving as part of the method's purpose; more updates per batch lower the generation share, as the derived numbers show.

## Limits

- The model is a toy: 3.8 million parameters, random weights, CPU, batch 16. Absolute times, the cache speedup and the shares depend on hardware and model size; only the direction (generation share rising with response length) is the measurement's claim.
- Reward-model scoring, weight transfer between engines and asynchronous staleness were not measured.
- The framework results are taken from abstracts, are the authors' own, and use different baselines; none was reproduced.
- Sampling used plain multinomial draws without any batching optimisations found in inference engines.

## Code and output

```python
import time, statistics, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(16); torch.manual_seed(0)
V, D, H, L, MAXLEN = 1000, 256, 4, 4, 512 + 64
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D)
        s.l2 = nn.LayerNorm(D); s.m = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x, kv=None):
        B, T, _ = x.shape; q, k, v = s.qkv(s.l1(x)).view(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        if kv is not None:
            if "k" in kv: k = torch.cat([kv["k"], k], 2); v = torch.cat([kv["v"], v], 2)
            kv["k"], kv["v"] = k, v
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(kv is None or T > 1))
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D)); return x + s.m(s.l2(x))
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.e = nn.Embedding(V, D); s.p = nn.Embedding(MAXLEN, D); s.b = nn.ModuleList(Block() for _ in range(L)); s.f = nn.LayerNorm(D); s.h = nn.Linear(D, V)
    def forward(s, ids, cache=None, pos0=0):
        x = s.e(ids) + s.p(torch.arange(pos0, pos0 + ids.shape[1]))
        for i, b in enumerate(s.b): x = b(x, None if cache is None else cache[i])
        return s.h(s.f(x))
@torch.no_grad()
def generate(m, prompt, T, use_cache):
    ids = prompt
    if use_cache:
        cache = [dict() for _ in range(L)]; lg = m(ids, cache, 0)[:, -1]
        for t in range(T):
            nxt = torch.multinomial(F.softmax(lg, -1), 1); ids = torch.cat([ids, nxt], 1)
            lg = m(nxt, cache, ids.shape[1] - 1)[:, -1]
    else:
        for t in range(T):
            nxt = torch.multinomial(F.softmax(m(ids)[:, -1], -1), 1); ids = torch.cat([ids, nxt], 1)
    return ids
def timeit(f, n=3):
    f(); ts = []
    for _ in range(n): t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)
m = GPT(); opt = torch.optim.AdamW(m.parameters(), 1e-4)
print("parameters:", sum(p.numel() for p in m.parameters()), "| layers", L, "d_model", D, "| batch 16, prompt 64 tokens | torch threads 16")
print(f"{'new tokens':>10s} {'gen, no cache (s)':>18s} {'gen, KV cache (s)':>18s} {'train step (s)':>15s} {'generation share':>17s} {'KV cache (MB)':>14s}")
for T in (16, 64, 256):
    prompt = torch.randint(0, V, (16, 64))
    tn = timeit(lambda: generate(m, prompt, T, False)); tc = timeit(lambda: generate(m, prompt, T, True))
    seq = generate(m, prompt, T, True)
    def step():
        lp = F.log_softmax(m(seq[:, :-1]), -1).gather(2, seq[:, 1:, None]).squeeze(2)[:, 64 - 1:]
        loss = -(lp * torch.randn(16, 1)).mean(); opt.zero_grad(); loss.backward(); opt.step()
    ts = timeit(step); kvmb = 2 * L * 16 * (64 + T) * D * 4 / 1e6
    print(f"{T:10d} {tn:18.2f} {tc:18.2f} {ts:15.2f} {tc / (tc + ts):17.0%} {kvmb:14.1f}")
```

```
parameters: 3820008 | layers 4 d_model 256 | batch 16, prompt 64 tokens | torch threads 16
new tokens  gen, no cache (s)  gen, KV cache (s)  train step (s)  generation share  KV cache (MB)
        16               0.46               0.07            0.11               37%           10.5
        64               1.97               0.19            0.17               53%           16.8
       256              17.35               1.12            0.45               71%           41.9
```

Check of the cached step and the derived shares (the shares use the rounded times from the table):

```python
import torch
from e6_defs import *
m = GPT().eval(); ids = torch.randint(0, V, (4, 80))
with torch.no_grad():
    full = m(ids)[:, -1]; cache = [dict() for _ in range(L)]; m(ids[:, :-1], cache, 0); inc = m(ids[:, -1:], cache, 79)[:, -1]
print("max abs difference between the last-position logits, full forward vs KV-cache step:", float((full - inc).abs().max()))
for T, tc, ts in ((16, .07, .11), (64, .19, .17), (256, 1.12, .45)):
    print(f"{T} new tokens: generation share with 1 / 2 / 4 gradient steps per batch:", *[f"{tc / (tc + k * ts):.0%}" for k in (1, 2, 4)])
```

```
max abs difference between the last-position logits, full forward vs KV-cache step: 5.364418029785156e-07
16 new tokens: generation share with 1 / 2 / 4 gradient steps per batch: 39% 24% 14%
64 new tokens: generation share with 1 / 2 / 4 gradient steps per batch: 53% 36% 22%
256 new tokens: generation share with 1 / 2 / 4 gradient steps per batch: 71% 55% 38%
```

## The series in one paragraph

Part 1 measured where environment throughput stops scaling; Part 2 showed that a seed count decides whether a PPO comparison is trustworthy; Part 3 showed that a higher update-to-data ratio did not help while resets did, and that near-expert data can collapse offline learning; Part 4 attached qualifiers to five famous applications; Part 5 found no measurable benefit from simulator randomisation on CartPole at 40 seeds; this part measured the share of an iteration that goes to generation. The common thread is measurement before conclusion: the effect sizes in these experiments were often smaller than the seed noise.

## References

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256), 2024.
- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143), 2024.
- [Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models](https://arxiv.org/abs/2410.18252), 2024.
