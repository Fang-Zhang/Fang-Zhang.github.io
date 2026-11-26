---
title: "Distributed LLMs, Measured (1/7): Sixteen Bytes per Parameter Put a 70B Model on at Least 14 GPUs Before a Single Activation Is Stored"
date: 2027-07-06 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, distributed training, memory, flops, gpu, series]
description: "Part 1 of 7. The memory and compute accounting behind distributed LLM training: bytes per parameter, activation memory per layer, the 6N rule and its attention term, checked with measured tensors and counted FLOPs."
---

## Why this series starts with arithmetic

Every distributed-training technique answers one of two shortages: memory that does not fit on one device, or time that does not fit in one lifetime. Both shortages are fixed by arithmetic, so this part derives the numbers that the other six parts rely on, and then checks each formula against tensors and FLOP counts measured on a small GPT. The later parts cover data parallelism and collectives, sharded states, tensor parallelism, pipeline parallelism with 3D layouts, mixture-of-experts, and inference serving.

## What the sources say

- **Model states.** [ZeRO](https://arxiv.org/abs/1910.02054) splits training memory into model states (optimizer states, gradients, parameters) and residual memory. With mixed-precision Adam the model states cost (2 + 2 + K) bytes per parameter with K = 12, that is 16 bytes, and "a 7.5 B parameter model requires ... 120 GB with standard DP". The same paper's Table 1 gives 31.4 GB, 16.6 GB and 1.88 GB per device for the three sharding stages at 64-way data parallelism, which Part 3 reproduces.
- **Compute.** [Kaplan et al.](https://arxiv.org/abs/2001.08361) give a forward pass of roughly 2N + 2 n_layer n_ctx d_model operations per token, state that the backward pass costs approximately twice the forward pass, omit the context-dependent term for the models they study, and estimate training compute as C = 6NBS for N non-embedding parameters, batch size B and S parameter updates.
- **Activations.** [Korthikanti et al.](https://arxiv.org/abs/2205.05198) give the activation memory of one transformer layer, without model parallelism, as sbh(34 + 5as/h) bytes, with s the sequence length, b the batch size, h the hidden size and a the number of attention heads.
- **A real run.** [The Llama 3 report](https://arxiv.org/abs/2407.21783) describes a 405B-parameter model pre-trained on 15.6T tokens with 3.8 x 10^25 FLOPs on up to 16K H100 GPUs with 80 GB each. Table 4 lists 8,192 to 16,384 GPUs at 380 to 430 TFLOPs per GPU, a BF16 MFU (model FLOPs utilisation) of 38 to 43 percent. In a 54-day period there were 466 job interruptions, 419 of them unexpected, about 78 percent of those attributed to confirmed or suspected hardware issues.

## Experiment: check the formulas on a real model

Setup: a decoder-only GPT with the usual layer shape (attention with 4 projection matrices of size d x d in total, an MLP with a 4d hidden layer, layer norms, a d x 128 output head), four sizes from 149,376 to 14,491,520 parameters, in fp32 on a CPU. Three measurements:

1. **Model-state bytes.** After one AdamW step, the sizes of all weight, gradient and optimizer-state tensors.
2. **Training FLOPs.** One forward and backward pass counted by `torch.utils.flop_counter`, at sequence lengths 64 and 256, against the formula 6(12 L d^2 + d V) T + c L s d T, where T is the number of tokens, V the vocabulary size and c a coefficient fitted from the counts.
3. **Activation bytes.** Every tensor saved for the backward pass is recorded through autograd saved-tensor hooks. Unique storage is counted and parameter storage is excluded, and the per-layer cost is the difference between a 3-layer and a 1-layer model.

Results:

- **States.** Exactly 4.0 bytes of weights, 4.0 of gradients and 8.0 of Adam moments per parameter, 16.00 in total, at all four sizes, plus a few hundred bytes of step counters. In fp32 this is 4 + 4 + 8; in mixed precision it is 2 + 2 + 12, which is the same 16.
- **FLOPs.** For all 8 configurations the counted total equals 6(12 L d^2 + d V) T + 14 L s d T exactly, to the integer. The matmul part is the 6N rule with no remainder: 2 FLOPs per weight in the forward pass and 4 in the backward pass. The attention term has a coefficient of 14, against 6 if Kaplan's term 2 n_layer n_ctx d_model per token is multiplied by 3 for forward plus backward. A split of 4 for the forward pass and 10 for the backward pass fits the fitted total: full s x s score and value products forward, and five matrix products backward, including recomputing the scores. This split is inferred from the totals, not read from the counter's code. Kaplan's expression is half of the 4 s d forward cost, which reads as an average over causal positions, and it has no recomputation. Attention was 15.2 percent of the counted FLOPs at d = 64 and s = 64, 41.8 percent at s = 256, and 3.1 to 11.4 percent for d = 384.
- **Activations.** 16.06 units of s b h fp32 values per layer at d = 128 (16.05 at d = 256), for sequence lengths 64 to 512, independent of the sequence length. With 2-byte activations this is 32 bytes per s b h. The extra 0.06 units come from layer-norm statistics and the softmax log-sum-exp.
- **Llama 3 arithmetic.** 6 x 405e9 x 15.6e12 = 3.79 x 10^25 FLOPs, matching the reported 3.8 x 10^25. On one GPU at 400 TFLOPs this is about 3,000 years; on 16,384 GPUs at 400 TFLOPs it is about 67 days.

![Two panels. Left: model-state memory at 16 bytes per parameter on a log scale: 120 GB for 7.5B parameters (at least 2 GPUs of 80 GB), 1,120 GB for 70B (at least 14) and 6,480 GB for 405B (at least 81). Right: attention share of training FLOPs per token against sequence length from 2048 to 131072 for 8B-, 70B- and 405B-like layer shapes, rising from a few percent to 40 to 85 percent, higher with coefficient 14 than with coefficient 6]({{ site.baseurl }}/assets/pic/dist-1-accounting.png){: width="900" }
_Figure: left, 16 bytes per parameter applied to three model sizes (arithmetic from the measured 16 bytes); right, the formula with 12 L d^2 matmul weights per model, a 128,000-token output head and the two attention coefficients; the layer shapes are simplified GPT-style stand-ins for the sizes, not exact Llama configurations._

## Engineering reading

These points are this series' reading of the sources and the measurement.

1. **The bytes per parameter, not the parameter count, set the minimum cluster.** At 16 bytes a 70B model needs 1,120 GB of states, so at least 14 GPUs of 80 GB, and 405B needs 6,480 GB and at least 81, before one activation or one communication buffer exists. The 16 bytes were measured exactly, not assumed, and they are independent of batch size and sequence length, which is why sharding the states (Part 3) is the first lever.
2. **Activations are a different budget with a different scaling law.** They grow with s, b and the number of layers and not with the parameter count directly. The measured 16.06 units equal, in 2-byte arithmetic, 32.1 bytes per s b h per layer. Korthikanti et al.'s 34 is arithmetically consistent with that once two 1-byte dropout masks are added, which this model does not have; this decomposition was not checked operation by operation against the paper. The 5as/h term of the formula, which holds the attention score matrices, did not appear: the fused attention kernel on this CPU saves only log-sum-exp statistics, so the memory is flat in s while the FLOPs are not.
3. **6N is exact for the matrix multiplications and incomplete for attention.** The matmul part of the count matched the rule exactly. The attention term depends on the convention: 6 L s d per token with causal averaging, or 14 L s d per token as counted for a kernel that computes full score blocks and recomputes in the backward pass. For layer shapes like those of Llama 3 at 8K tokens the formula puts the share at 4.0 to 13.4 percent with the first convention and 8.8 to 26.5 percent with the second, and at 128K tokens it rises to 40 to 85 percent. Any utilisation figure should state which convention it uses.
4. **Time at scale is compute divided by an efficiency below one half.** The 3.8 x 10^25 FLOPs divided by 16,384 GPUs at 400 TFLOPs gives 67 days; 400 TFLOPs is the reported per-GPU rate at 41 percent MFU, so the other 59 percent of the peak is lost to communication, memory-bound operations and waiting, which is the subject of Parts 2 to 5.
5. **Failure is a design input.** 419 unexpected interruptions in 54 days is one about every 3.1 hours on average. Synchronous training restarts the job when one GPU fails, so checkpoint frequency and restart time enter the efficiency calculation as much as the kernels do.

## Limits

- The measurements are fp32 bookkeeping on a CPU for a small GPT with 4d MLPs; they verify the formulas but say nothing about GPU speed or interconnect.
- The extrapolations to 7.5B, 70B and 405B multiply the measured 16 bytes per parameter by the stated parameter counts; Llama configurations (grouped-query attention, gated MLP) are approximated by simplified layers in the right panel.
- The FLOP counter counts matrix-multiplication-like operations only; element-wise operations, which matter for memory-bound kernels, are excluded. The attention coefficient reflects the CPU kernel's accounting, and GPU kernels can differ.
- The activation count has no dropout and uses fp32; the comparison with the published 34 bytes is by arithmetic, not by operation.
- The Llama 3 numbers come from the report; none was reproduced.

## Code and output

```python
import json, torch, torch.nn.functional as F, lm
from torch.utils.flop_counter import FlopCounterMode
torch.set_num_threads(4); lm.CTX = 512
def make(D, H, L): torch.manual_seed(0); return lm.GPT(D, H, L)
def nparams(m): return sum(p.numel() for p in m.parameters())
out = {}
# A: model-state bytes after one AdamW step (fp32)
rows = []
for D, H, L in ((64, 4, 2), (128, 4, 4), (256, 8, 6), (384, 8, 8)):
    m = make(D, H, L); opt = torch.optim.AdamW(m.parameters(), 1e-3); x = torch.randint(0, lm.V, (4, 64))
    F.cross_entropy(m(x).reshape(-1, lm.V), x.reshape(-1)).backward(); opt.step()
    N = nparams(m); pb = sum(p.numel() * p.element_size() for p in m.parameters()); gb = sum(p.grad.numel() * p.grad.element_size() for p in m.parameters())
    sb = sum(t.numel() * t.element_size() for st in opt.state.values() for t in st.values() if torch.is_tensor(t) and t.dim() > 0); extra = sum(t.numel() * t.element_size() for st in opt.state.values() for t in st.values() if torch.is_tensor(t) and t.dim() == 0)
    rows.append(dict(D=D, L=L, N=N, param_bytes=pb, grad_bytes=gb, adam_bytes=sb, scalar_bytes=extra, bytes_per_param=(pb + gb + sb) / N))
out["states"] = rows
# B: training FLOPs counted vs 6N-style formulas (CTX 128, batch 8)
rows = []
for D, H, L in ((64, 4, 2), (128, 4, 4), (256, 8, 6), (384, 8, 8)):
    for S in (64, 256):
        m = make(D, H, L); B = 8; x = torch.randint(0, lm.V, (B, S)); T = B * S
        with FlopCounterMode(display=False) as fc: F.cross_entropy(m(x).reshape(-1, lm.V), x.reshape(-1)).backward()
        Nne = 12 * L * D * D + 13 * L * D   # non-embedding params (weights+biases+layernorm) as counted by Kaplan: use matmul weights 12 L D^2 for formula
        Nmm = 12 * L * D * D; head = D * lm.V
        f6 = 6 * Nmm * T; f6h = 6 * (Nmm + head) * T; attn = 6 * L * S * D * T   # Kaplan: fwd 2N + 2 L ctx D per token; x3 for fwd+bwd
        rows.append(dict(D=D, L=L, S=S, tokens=T, counted=fc.get_total_flops(), f_6N=f6, f_6N_head=f6h, f_6N_head_attn=f6h + attn, attn_causal_half=f6h + attn / 2))
for r in rows: r["attn_counted_per_L_S_D_T"] = (r["counted"] - r["f_6N_head"]) / (r["L"] * r["S"] * r["D"] * r["tokens"])
out["flops"] = rows
# C: activation memory = unique storage bytes saved for backward, per layer
def saved_bytes(m, B, S):
    seen = {}; pst = {p.untyped_storage().data_ptr() for p in m.parameters()}
    def pack(t):
        if t.untyped_storage().data_ptr() not in pst: seen[(t.untyped_storage().data_ptr())] = t.untyped_storage().nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t): y = m(torch.randint(0, lm.V, (B, S)))
    return sum(seen.values())
rows = []
for D, H in ((128, 4), (256, 8)):
    for S in (64, 128, 256, 512):
        B = 4; m1 = make(D, H, 1); m3 = make(D, H, 3); per_layer = (saved_bytes(m3, B, S) - saved_bytes(m1, B, S)) / 2
        rows.append(dict(D=D, H=H, S=S, B=B, per_layer_bytes=per_layer, units_of_sbh_fp32=per_layer / (S * B * D * 4)))
out["acts"] = rows
json.dump(out, open("g1_out.json", "w"), indent=1)
for k in ("states", "flops", "acts"):
    print(k)
    for r in out[k]: print(" ", r)
```

Statistics output:

```
A. model-state bytes after one AdamW step, fp32 (measured from tensor sizes)
  D= 64 L=2  params N=   149,376  weights 4.0 B/param, grads 4.0, Adam moments 8.0; total 16.00 B/param (+120 bytes of step counters)
  D=128 L=4  params N=   891,776  weights 4.0 B/param, grads 4.0, Adam moments 8.0; total 16.00 B/param (+216 bytes of step counters)
  D=256 L=6  params N= 4,935,808  weights 4.0 B/param, grads 4.0, Adam moments 8.0; total 16.00 B/param (+312 bytes of step counters)
  D=384 L=8  params N=14,491,520  weights 4.0 B/param, grads 4.0, Adam moments 8.0; total 16.00 B/param (+408 bytes of step counters)

   extrapolation at 16 bytes/param (2 weights + 2 grads + 12 optimizer: fp32 master copy + 2 Adam moments), 80 GB per GPU, states only:
   7.5B:      120 GB of model states -> at least 2 GPUs before any activation
    70B:    1,120 GB of model states -> at least 14 GPUs before any activation
   405B:    6,480 GB of model states -> at least 81 GPUs before any activation

B. FLOPs of one forward+backward pass, counted by torch.utils.flop_counter vs formula
  D= 64 L=2 S= 64: counted     385,875,968  6(12LD^2+DV)T + 14LSDT =     385,875,968  equal: True   attention share 15.2%
  D= 64 L=2 S=256: counted   2,248,146,944  6(12LD^2+DV)T + 14LSDT =   2,248,146,944  equal: True   attention share 41.8%
  D=128 L=4 S= 64: counted   2,701,131,776  6(12LD^2+DV)T + 14LSDT =   2,701,131,776  equal: True   attention share 8.7%
  D=128 L=4 S=256: counted  13,623,099,392  6(12LD^2+DV)T + 14LSDT =  13,623,099,392  equal: True   attention share 27.6%
  D=256 L=6 S= 64: counted  15,300,820,992  6(12LD^2+DV)T + 14LSDT =  15,300,820,992  equal: True   attention share 4.6%
  D=256 L=6 S=256: counted  69,659,000,832  6(12LD^2+DV)T + 14LSDT =  69,659,000,832  equal: True   attention share 16.2%
  D=384 L=8 S= 64: counted  45,046,824,960  6(12LD^2+DV)T + 14LSDT =  45,046,824,960  equal: True   attention share 3.1%
  D=384 L=8 S=256: counted 197,098,733,568  6(12LD^2+DV)T + 14LSDT = 197,098,733,568  equal: True   attention share 11.4%
  all rows equal: True

   attention share of training FLOPs per token, formula, 12LD^2 matmul weights + DV head (V=128000):
    8B-like S=  8192: counted-style (14LSD) 26.5%   Kaplan-style (6LSD) 13.4%
    8B-like S=131072: counted-style (14LSD) 85.2%   Kaplan-style (6LSD) 71.1%
   70B-like S=  8192: counted-style (14LSD) 16.1%   Kaplan-style (6LSD) 7.6%
   70B-like S=131072: counted-style (14LSD) 75.4%   Kaplan-style (6LSD) 56.7%
  405B-like S=  8192: counted-style (14LSD) 8.8%   Kaplan-style (6LSD) 4.0%
  405B-like S=131072: counted-style (14LSD) 60.7%   Kaplan-style (6LSD) 39.9%

C. activation bytes saved for backward per layer, parameters excluded, fp32, B=4
  D=128 S= 64:    2,105,344 bytes = 16.062 x (s*b*h*4 bytes)
  D=128 S=128:    4,210,688 bytes = 16.062 x (s*b*h*4 bytes)
  D=128 S=256:    8,421,376 bytes = 16.062 x (s*b*h*4 bytes)
  D=128 S=512:   16,842,752 bytes = 16.062 x (s*b*h*4 bytes)
  D=256 S= 64:    4,206,592 bytes = 16.047 x (s*b*h*4 bytes)
  D=256 S=128:    8,413,184 bytes = 16.047 x (s*b*h*4 bytes)
  D=256 S=256:   16,826,368 bytes = 16.047 x (s*b*h*4 bytes)
  D=256 S=512:   33,652,736 bytes = 16.047 x (s*b*h*4 bytes)

D. compute for Llama 3 405B from published numbers: 6 * 405e9 * 15.6e12 = 3.79e+25 FLOPs
   one GPU at 400 TFLOPs/s: 3003 years;  16,384 GPUs at 400 TFLOPs/s: 67 days
```

Next: [Part 2](/posts/dist-2-data-parallel/) covers data parallelism and the collective operations that synchronise gradients.

## References

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), 2019.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), 2022.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024.
