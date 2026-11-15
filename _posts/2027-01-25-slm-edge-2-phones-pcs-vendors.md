---
title: "Small Language Models on Small Devices (2/6): Phones and PCs, What Four Vendors Document"
date: 2027-01-25 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, edge-ai, on-device, apple, google, qualcomm, microsoft, series]
description: "Part 2 of 6. What Apple, Google, Qualcomm and Microsoft state in their own reports and documentation about on-device language models, and two measurements on a CPU that explain why the NPU, the 4-bit format and the clipping range matter."
---

[Part 1](/posts/slm-edge-1-memory-bandwidth-arithmetic/) showed that decoding is limited by memory traffic. This part reads what four vendors say about phone and PC deployments in their own technical reports and documentation, and adds two measurements that explain the design choices they describe. Only statements found in those documents are attributed to the vendors; the vendors' speed figures are theirs and were not reproduced.

![Two panels. Left: achieved GFLOP/s of one 4096 by 4096 float32 layer against tokens per pass on a log axis, rising from 10 at one token to 812 at 1024 tokens. Right: relative quantisation error against the scale factor f for 8, 4, 3 and 2 bit weights on a tensor with 0.1 percent outliers; the 4-bit error falls from 0.92 to 0.35 as f decreases from 1 to 0.15.](/assets/pic/slm-edge-2-npu-quantisation.png)
*Left: with the weights fixed, more tokens per pass means more arithmetic per byte read. Right: the quantisation range must be tuned, or low-bit error approaches 1.*

## What each vendor documents

- **Apple.** The [2025 foundation model report](https://arxiv.org/abs/2507.13575) describes a roughly 3B-parameter on-device model. The model is split into two blocks, with 62.5% of the layers in Block 1 and 37.5% in Block 2; Block 2 has no key and value projections and reuses Block 1's cache, which the report says cuts KV-cache memory by 37.5% and time-to-first-token by about 37.5%. The model is compressed to 2 bits per weight with quantisation-aware training, the embedding table is 4 bits, the KV cache is 8 bits, and low-rank adapters recover quality lost to compression. Apple Intelligence features run through adapters on the base model, and the Swift Foundation Models framework exposes guided generation, constrained tool calling and LoRA adapter fine-tuning.
- **Google.** The [Gemini Nano documentation](https://developer.android.com/ai/gemini-nano) says the model runs in the Android system service AICore, which uses device hardware and keeps the model up to date; the stated case for it is low cost and privacy, with no network connection needed. The [ML Kit GenAI APIs](https://developers.google.com/ml-kit/genai) sit on top of it, and the architecture diagram shows LoRA adapters loaded next to the model. The page names token limits, hardware limits and capability boundaries as constraints and suggests hybrid cloud inference for harder queries.
- **Qualcomm.** The [Hexagon NPU white paper (February 2024)](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf) describes native INT4 support that gives "improved power and memory bandwidth efficiency" and doubles tensor throughput for INT4 layers, a large shared memory, and fusing layers to cut memory traffic. It states that its voice-assistant demo ran Llama 2-7B at up to 20 tokens per second on Snapdragon 8 Gen 3, and more than 13 on the Adreno GPU. These are Qualcomm's demonstration numbers.
- **Microsoft.** The [Phi-3 report](https://arxiv.org/abs/2404.14219) takes the other route: keep the architecture Llama-2-like and improve the training data (filtered web data plus synthetic data) so that a 3.8B model is competitive. It states a 4-bit footprint of about 1.8 GB and more than 12 tokens per second on an iPhone 14.

The four documents agree on a pattern: a model of about 3–4B parameters, 2–4 bit weights, a smaller or shared KV cache, and small adapters on top for specific features. None of them is a controlled comparison, because each vendor reports its own model, device and metric.

```python
import time, numpy as np, torch
torch.manual_seed(0); np.random.seed(0)

# A. Arithmetic intensity: processing T tokens against one weight matrix.
# Weights are read once (N*N values); operations scale with T. Decode is T=1, prompt processing is T>>1.
N=4096; W=torch.randn(N,N)
print("A. one 4096x4096 float32 layer, T tokens per pass")
print(f"{'T':>5} {'ops per weight byte':>20} {'ms':>8} {'GFLOP/s':>9} {'tokens/s':>10}")
for T in [1,4,16,64,256,1024]:
    X=torch.randn(N,T); W@X
    t=time.perf_counter(); r=5
    for _ in range(r): W@X
    dt=(time.perf_counter()-t)/r
    print(f"{T:>5} {2*N*N*T/(N*N*4):>20.2f} {dt*1e3:>8.2f} {2*N*N*T/dt/1e9:>9.1f} {T/dt:>10.0f}")

# B. Low-bit quantisation of a weight tensor with outliers: scale s = f * max|W| / qmax.
# Apple's report describes a learnable scale f and a clipping scalar; here f is simply scanned.
def quant(w, bits, f):
    qmax=2**(bits-1)-1 if bits>2 else 1
    s=f*np.abs(w).max()/qmax
    return np.clip(np.round(w/s),-qmax-1,qmax)*s
w=np.random.randn(256*256)*0.02
w[np.random.rand(w.size)<0.001]*=12          # 0.1% outliers
print("\nB. relative error ||W-Q(W)||/||W|| for a tensor with 0.1% outliers")
print(f"{'bits':>5} " + " ".join(f"f={f:<5}" for f in [1.0,0.6,0.4,0.25,0.15]))
for b in [8,4,3,2]:
    print(f"{b:>5} " + " ".join(f"{np.linalg.norm(w-quant(w,b,f))/np.linalg.norm(w):<7.3f}" for f in [1.0,0.6,0.4,0.25,0.15]))
best={b:min((np.linalg.norm(w-quant(w,b,f))/np.linalg.norm(w),f) for f in np.linspace(0.05,1,96)) for b in [8,4,3,2]}
print("best error over f in [0.05, 1]:",{b:round(float(e),3) for b,(e,f) in best.items()})
```

```
A. one 4096x4096 float32 layer, T tokens per pass
    T  ops per weight byte       ms   GFLOP/s   tokens/s
    1                 0.50     2.73      12.3        366
    4                 2.00     7.23      18.6        553
   16                 8.00     8.42      63.8       1900
   64                32.00    11.09     193.7       5773
  256               128.00    15.43     556.5      16586
 1024               512.00    44.40     773.8      23062

B. relative error ||W-Q(W)||/||W|| for a tensor with 0.1% outliers
 bits f=1.0   f=0.6   f=0.4   f=0.25  f=0.15 
    8 0.091   0.083   0.124   0.189   0.254  
    4 0.922   0.835   0.653   0.449   0.349  
    3 0.930   0.927   0.923   0.839   0.620  
    2 0.976   0.950   0.936   0.935   0.945  
best error over f in [0.05, 1]: {8: 0.078, 4: 0.331, 3: 0.385, 2: 0.654}
```

## Why the NPU: arithmetic per byte

Qualcomm's white paper gives the reason in one line: multiplying two $N\times N$ matrices reads $2N^2$ values and does $2N^3$ operations, so the ratio of operations to memory reads is $N$ to one for a tensor accelerator. Block A measures the same effect on this CPU with one $4096\times4096$ float32 layer. At one token per pass (decoding) each weight byte supports 0.5 operations and the layer reaches 10 GFLOP/s; at 1024 tokens per pass (processing a prompt) each byte supports 512 operations and the layer reaches 812 GFLOP/s, 81 times higher. The same weights cost 3.35 ms for one token and 42.3 ms for 1024, i.e. 298 against 24,198 tokens per second. So an accelerator helps most where there is arithmetic to hide the memory reads (prompt processing), and in decoding the memory system, as in Part 1, sets the speed.

## Why the clipping range: 2-bit needs care

Apple's report adds a learnable scale per weight tensor, with the scale set to $f\cdot\max|W|/q_{\max}$, and states that initialising $f$ is "crucial" for stable training at 2 bits. Block B builds a random tensor with 0.1% outliers (twelve times the normal scale) and quantises it with plain rounding. At 8 bits, the relative error is 0.09. At 4 bits with the range set to the full maximum ($f=1$) the error is 0.92: the outliers stretch the grid so that nearly all ordinary weights round to zero. Shrinking the range to $f=0.15$ lowers it to 0.35, and the best of 96 values of $f$ in $[0.05,1]$ gives 0.331. At 3 bits the best is 0.385 and at 2 bits 0.654. A 2-bit grid with four levels cannot represent this tensor well by rounding alone, which is consistent with the report's use of training to recover quality instead of rounding after the fact (the author's reading; the report does not run this experiment). The tensor is synthetic, so the numbers show the mechanism, not the error of any real model.

## Limits

- Vendor speeds (12, 13 and 20 tokens per second) come from different models, devices and settings; they are quoted from the documents and were not reproduced.
- The measurements run on one desktop CPU; they illustrate the compute-versus-memory effect and say nothing about the speed of any NPU.
- The quantisation test uses a synthetic tensor with outliers and plain scaled rounding; it is not the training-based method in the Apple report.
- The summaries cover what each document states; products and software versions change, and the documents were read as of the dates shown on them.

Next: [Part 3, edge boards: Raspberry Pi and Jetson](/posts/slm-edge-3-edge-boards-jetson-raspberry-pi/).
