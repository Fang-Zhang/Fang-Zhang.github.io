---
title: "Small Language Models on Small Devices (1/6): The Memory and Bandwidth Arithmetic"
date: 2027-01-24 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, edge-ai, on-device, quantisation, kv-cache, series]
description: "Part 1 of 6. Whether a language model fits on a device, and how fast it runs, is mostly arithmetic: parameters times bits gives memory, and memory traffic per token gives speed. The numbers are worked out in code and checked against published on-device figures."
---

This series asks what it takes to run a language model on a phone, an edge board or a microcontroller. Part 1 does the arithmetic that decides the answer before any vendor is mentioned: memory for the weights, memory for the KV cache, and the speed limit set by memory bandwidth. The code runs on a CPU-only machine (16 cores, no GPU, 16 GB RAM), and the results are compared with figures that Microsoft, Apple and Google publish for their on-device models. Later parts: phones and PCs, edge boards, microcontrollers, compression methods, and post-training.

![Three panels. Left: weight memory against parameter count for 16, 8, 4 and 2 bit weights on log axes, with a dotted line at 1.8 GB. Middle: measured tokens per second against weight bytes per token for float32 and bfloat16, with a dashed bandwidth-over-bytes curve. Right: KV cache size in GB against context length from 256 to 4096 tokens.](/assets/pic/slm-edge-1-memory-bandwidth.png)
*Left: weights = parameters × bits / 8. Middle: decode speed follows bytes read per token. Right: the KV cache of one 3.8B-parameter model grows linearly with context.*

```python
import time, torch
# 1. Weight memory = parameters x bits
print("weight memory (GB), 1 GB = 1e9 bytes")
print(f"{'params':>8} {'fp16':>7} {'int8':>7} {'int4':>7} {'int2':>7}")
for n in [0.1e9,0.5e9,1e9,3.8e9,8e9]:
    print(f"{n/1e9:>7.1f}B " + " ".join(f"{n*b/8/1e9:>7.2f}" for b in [16,8,4,2]))

# 2. Phi-3-mini shape from the report: 3072 hidden, 32 layers, 32 heads, vocab 32064, 3.8B params
d,L,H,V,P=3072,32,32,32064,3.8e9
emb=V*d; attn=4*d*d*L
ffn_width=(P-emb-attn)/(3*d*L)
print(f"\nphi-3-mini (3072 hidden, 32 layers, 3.8B parameters as stated in the report)")
print(f"4-bit weights: {P*4/8/1e9:.2f} GB (report: about 1.8 GB)")
# KV cache per token: 2 (K,V) x layers x d values, fp16
for ctx in [512,4096]:
    kv=2*L*d*2*ctx
    print(f"KV cache fp16, context {ctx}: {kv/1e9:.2f} GB")

# 3. Decode is a pass over all weights per token: tokens/s <= bandwidth / bytes
torch.manual_seed(0)
print("\nmeasured on this CPU: one full pass over the weights as matrix-vector products")
print(f"{'dtype':>9} {'GB':>6} {'GB/s':>7} {'tok/s':>7}")
rows=[]
for dt in [torch.float32, torch.bfloat16]:
    for n in [0.25e9, 0.5e9, 1e9]:
        side=4096; k=int(n/(side*side)); 
        W=[torch.randn(side,side,dtype=dt) for _ in range(k)]
        x=torch.randn(side,1,dtype=dt)
        for w in W: w@x
        t=time.perf_counter(); reps=3
        for _ in range(reps):
            for w in W: w@x
        dtm=(time.perf_counter()-t)/reps
        gb=k*side*side*W[0].element_size()/1e9
        rows.append((str(dt).split('.')[1],n,gb,gb/dtm,1/dtm))
        print(f"{rows[-1][0]:>9} {gb:>6.2f} {gb/dtm:>7.1f} {1/dtm:>7.1f}  ({n/1e9:.2f}B params)")
        del W
```

```
weight memory (GB), 1 GB = 1e9 bytes
  params    fp16    int8    int4    int2
    0.1B    0.20    0.10    0.05    0.03
    0.5B    1.00    0.50    0.25    0.12
    1.0B    2.00    1.00    0.50    0.25
    3.8B    7.60    3.80    1.90    0.95
    8.0B   16.00    8.00    4.00    2.00

phi-3-mini (3072 hidden, 32 layers, 3.8B parameters as stated in the report)
4-bit weights: 1.90 GB (report: about 1.8 GB)
KV cache fp16, context 512: 0.20 GB
KV cache fp16, context 4096: 1.61 GB

measured on this CPU: one full pass over the weights as matrix-vector products
    dtype     GB    GB/s   tok/s
  float32   0.94    18.1    19.3  (0.25B params)
  float32   1.95    17.8     9.1  (0.50B params)
  float32   3.96    17.5     4.4  (1.00B params)
 bfloat16   0.47     4.8    10.1  (0.25B params)
 bfloat16   0.97     4.5     4.6  (0.50B params)
 bfloat16   1.98     4.8     2.4  (1.00B params)
```

## 1. Weights: parameters × bits / 8

The first table is the whole rule. A 3.8B-parameter model needs 7.6 GB at 16 bits, 3.8 GB at 8 bits, 1.9 GB at 4 bits and 0.95 GB at 2 bits (1 GB = $10^9$ bytes). The [Phi-3 report](https://arxiv.org/abs/2404.14219) states that phi-3-mini (3.8B parameters) "can be quantized to 4-bits so that it only occupies ≈1.8GB of memory", close to the 1.90 GB computed here; the report does not give the breakdown, so the 0.1 GB gap is not explained here. The same report says it deployed the 4-bit model on an iPhone 14 (A16 Bionic chip), running natively and offline, at more than 12 tokens per second.

Apple's [2025 foundation model report](https://arxiv.org/abs/2507.13575) describes a roughly 3B-parameter on-device model with KV-cache sharing and 2-bit quantisation-aware training, which on the table above corresponds to well under 1 GB of weights if all weights were 2-bit; the abstract does not say whether every layer is.

## 2. Speed: one pass over the weights per token

Generating one token with batch size 1 multiplies a vector by every weight matrix, so each weight is read once and used for two operations. The arithmetic is tiny compared with the memory traffic, so the limit is

$$\text{tokens per second}\;\le\;\frac{\text{memory bandwidth}}{\text{bytes of weights read per token}}.$$

The measurement multiplies a vector by $4096\times4096$ matrices totalling 0.25B, 0.5B and 1B parameters in float32. The throughput stays at 17.5–18.1 GB/s for all three sizes, while tokens per second falls from 19.3 to 9.1 to 4.4, the same ratio as the bytes (middle panel; the dashed curve is 17.5 GB/s divided by bytes). The bfloat16 runs read half the bytes but reach only 4.5–4.8 GB/s and so run slower than float32 on this CPU; this was not investigated, and it means that fewer bits help only if the hardware has a fast path for them. The same bound is why 4-bit weights are the default on devices: at a fixed bandwidth, 4 bits instead of 16 raise the ceiling four-fold, if the kernel dequantises fast enough.

This is a ceiling, and the figures are for this one CPU and one operation. Phones and boards have different bandwidth, and what they achieve is a question for Parts 2 and 3.

## 3. The KV cache: the second memory bill

Each generated token keeps a key and a value vector per layer ([Part 8 of the maths series](/posts/maths-for-llms-8-discrete-causal-mask-kv-cache-bpe/) counts the saving). With 32 layers and a hidden size of 3072, as in the Phi-3 report, and assuming full multi-head keys and values at 16 bits, one token costs $2\times32\times3072\times2$ bytes = 393 KB. Context 512 costs 0.20 GB and context 4096 costs 1.61 GB (right panel), nearly as much as the 4-bit weights. The assumption of no key-value sharing is the author's reading of the report's "similar block structure as Llama-2" and "32 heads"; models that share key-value heads, or share the cache across layers as Apple's report describes, shrink this figure. This is why cache sharing appears in an on-device report at all.

## 4. What the arithmetic says

- Fit: parameters × bits / 8 plus the cache and the operating system must stay under the device RAM; a 3.8B model needs 4-bit weights to fit a phone-class memory.
- Speed: bytes read per token set the tokens per second; quantisation attacks the same term in both lines.
- Long context shifts the bill from weights to cache.
- Google's [Gemini Nano documentation](https://developer.android.com/ai/gemini-nano) lists the same limits in words: token limits, device hardware limits and model capability boundaries, with hybrid cloud inference as the fallback.

## Limits

- Speeds are from one desktop CPU doing matrix-vector products, not from a full model, and not from any phone or board. They illustrate the bound; they are not vendor benchmarks.
- The 1.8 GB, 12 tokens per second and iPhone 14 figures are quoted from the Phi-3 report; they were not reproduced here.
- The KV cache figure assumes no key-value sharing and 16-bit storage; the real value depends on the architecture.
- Compute-bound phases, such as processing a long prompt, follow a different bound than the one shown here.

Next: Part 2, phones and PCs.
