---
title: "Improving LLM Performance, Measured (3/6): Rounding Weights to 4 Bits Cost 0.02 Bits per Character, While Zeroing Half of Them Cost 0.16"
date: 2027-07-02 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, quantisation, pruning, compression, inference, series]
description: "Part 3 of 6. Post-training quantisation and magnitude pruning applied to the same small language models: loss increase against bits per weight and sparsity, with paired seeds, next to what the public papers report at scale."
---

## What the literature says

Quantisation and pruning are the two standard ways to make a trained model cheaper to run. Four public sources give the claims at scale.

- **8-bit.** [LLM.int8()](https://arxiv.org/abs/2208.07339) cuts inference memory by half "while retaining full precision performance", and reports no degradation for models up to 175B parameters. It handles a small set of emergent outlier feature dimensions in 16-bit, "while still more than 99.9% of values are multiplied in 8-bit".
- **3 to 4 bits.** [GPTQ](https://arxiv.org/abs/2210.17323), a one-shot method using approximate second-order information, reduces the bitwidth "down to 3 or 4 bits per weight, with negligible accuracy degradation relative to the uncompressed baseline", and still gives reasonable accuracy at 2 bits or ternary levels.
- **Which bit width.** [Dettmers and Zettlemoyer](https://arxiv.org/abs/2212.09720) ran more than 35,000 experiments from 19M to 176B parameters and find "4-bit precision is almost universally optimal for total model bits and zero-shot accuracy", with the only improvements to the trade-off coming from a small block size and the data type.
- **Pruning.** [SparseGPT](https://arxiv.org/abs/2301.00774) shows GPT-family models can be pruned to at least 50 percent sparsity in one shot, without retraining, "at minimal loss of accuracy", reaching 60 percent unstructured sparsity "with negligible increase in perplexity" on 175B-scale models.

These are large-model results, taken from abstracts. The experiment below applies the simplest versions of each technique, round-to-nearest and magnitude pruning, to a small model and measures the cost per step.

## Experiment: loss against bits per weight and sparsity

Setup: the character-level GPT used in Parts 1 and 2 (4 layers, width 128, 842,624 parameters of which 802,816 are in the linear layers) trained for 800 steps at learning rate 0.003 with 100 warmup steps and a cosine decay, on the same prose corpus, with 4 seeds. Each trained model is compressed after training with no further training or calibration, and the held-out loss (bits per character on 40 held-out documents, 400 sampled sequences, the same sequences for every arm) is compared with the same model before compression. Because each comparison is paired with its own uncompressed model, the seed spread of the base models (standard deviation 0.043) does not enter the differences.

- **Quantisation.** All linear-layer weights rounded to a symmetric integer grid of 8, 6, 5, 4, 3 or 2 bits, with one scale per output row, or one scale for the whole matrix. Embeddings and layer norms stay in full precision; activations are not quantised.
- **Pruning.** In each linear layer the smallest-magnitude weights are set to zero, at 30 to 95 percent.

Results (mean loss increase over the same model uncompressed, in bits per character; the full-precision models average 2.668):

- **Per-row quantisation.** 8 bits: +0.000; 6 bits: +0.001; 5 bits: +0.004; 4 bits: +0.020 (per-seed range +0.018 to +0.022); 3 bits: +0.117; 2 bits: +1.647.
- **Per-tensor quantisation.** 8 bits: +0.000; 6 bits: +0.003; 5 bits: +0.012; 4 bits: +0.060; 3 bits: +0.369; 2 bits: +4.171.
- **Magnitude pruning.** 30 percent: +0.027; 50 percent: +0.161; 70 percent: +0.836; 80 percent: +1.632; 90 percent: +3.067; 95 percent: +3.897.

Every increase from 5 bits and below, and every pruning level, is significant under a paired test across the 4 models (p of 0.003 or lower); at 8 bits the difference is zero to three decimals.

![Two panels of held-out loss increase against full precision on a log scale. Left: quantisation from 8 to 2 bits per weight; per-row scales cost 0.000, 0.001, 0.004, 0.020, 0.117 and 1.647 bits per character, per-tensor scales cost more at every width. Right: pruning from 30 to 95 percent of weights, costing 0.027, 0.161, 0.836, 1.632, 3.067 and 3.897]({{ site.baseurl }}/assets/pic/llm-lab-3-compression.png){: width="900" }
_Figure: log scale of the loss increase; dots are individual models. Measured here._

## Engineering reading

These points are this series' reading of the sources and the measurement.

1. **Quantisation is nearly free down to 6 bits and cheap at 4.** 8 bits cost nothing measurable, 6 bits one thousandth of a bit, 4 bits 0.02. For scale, the seed-to-seed spread of the base models is 0.043, so the 4-bit penalty is smaller than the difference between two training runs. This is the direction of the LLM.int8(), GPTQ and 4-bit scaling results.
2. **The cliff is steep.** From 4 to 3 bits the cost grew almost six-fold, and from 3 to 2 bits fourteen-fold; at 2 bits the model is badly damaged (loss 4.3 against 2.7). A bit-width choice needs a measured loss at that width, not an extrapolation from the one above.
3. **Finer scales help a lot at low precision.** One scale per row instead of per matrix cut the 4-bit cost from 0.060 to 0.020 and the 3-bit cost from 0.369 to 0.117. This matches the finding of Dettmers and Zettlemoyer that small block sizes are among the few things that improved the bit-level trade-off; the outlier handling in LLM.int8() is a related remedy for the same problem, large values stretching the grid.
4. **Plain magnitude pruning was far costlier than the abstracts suggest.** At 50 percent sparsity the loss rose by 0.161, eight times the 4-bit quantisation cost, and 70 percent was already 0.84. SparseGPT reports minimal loss at 50 percent on much larger models using a more careful method. The size of the model, the method and the absence of calibration all differ here, so the experiment cannot say which one explains the gap.
5. **Pruning as run here saves nothing at inference time.** The zeroed weights stay in dense matrices; no sparse storage or kernel was used. Quantisation to 4 bits is a real 8-fold reduction of the weight bytes before scales, and the loss comparison is between a 4-bit model and a 30 percent pruned one of similar damage (0.020 against 0.027).

## Limits

- One small model size, one corpus, 4 seeds. The base-model spread is not part of the paired comparisons, but all conclusions about which technique is better apply to this model.
- Quantisation here is simulated: weights are rounded and stored back as 32-bit floats. No memory or latency was measured, and activations stay in full precision.
- Only the simplest methods were tried: round-to-nearest quantisation and magnitude pruning without retraining. Methods in the cited papers use calibration data and second-order information and are likely to do better than these baselines.
- Embeddings, layer norms and all bias vectors were left in full precision; the output head was quantised along with the other linear layers.
- Literature claims come from abstracts, and none of the results was reproduced.

## Code and output

Experiment code (the training loop and model are from Part 1):

```python
import sys, json, copy, torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
import torch.nn as nn, lm
P = lm.load_prose(); PTR, PVA = lm.split(P); TR = lm.enc("".join(PTR)); VA = lm.enc("".join(PVA))
def train_model(seed, steps=800):
    import math, torch.nn.functional as F
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = lm.GPT(); lr, warmup, B = 3e-3, 100, 16
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01); probs = torch.ones(1)
    for t in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * min(1, (t + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
        x, y = lm.batch([TR], probs, B, g); loss = F.cross_entropy(m(x).reshape(-1, lm.V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return m
def linears(m): return [mod for mod in m.modules() if isinstance(mod, nn.Linear)]   # all 4 per block + head
@torch.no_grad()
def quant(m, bits, per_row):
    q = 2 ** (bits - 1) - 1
    for l in linears(m):
        w = l.weight; s = (w.abs().amax(1, keepdim=True) if per_row else w.abs().max()) / q
        l.weight.copy_((w / s).round().clamp(-q, q) * s)
@torch.no_grad()
def prune(m, sp):
    for l in linears(m):
        w = l.weight; k = int(sp * w.numel())
        if k: thr = w.abs().flatten().kthvalue(k).values; l.weight.mul_((w.abs() > thr).float())
if __name__ == "__main__":
    seed = int(sys.argv[1]); base = train_model(seed); res = []
    ev = lambda m: lm.evaluate(m, VA, n=400)
    res.append(["fp32", 32, seed, ev(base)])
    for bits in (8, 6, 5, 4, 3, 2):
        for pr in (True, False):
            m = copy.deepcopy(base); quant(m, bits, pr); res.append(["row" if pr else "tensor", bits, seed, ev(m)])
    for sp in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        m = copy.deepcopy(base); prune(m, sp); res.append(["prune", sp, seed, ev(m)])
    n = sum(l.weight.numel() for l in linears(base)); print("linear weights:", n, "total params:", sum(p.numel() for p in base.parameters()))
    json.dump(res, open(f"f3_seed{seed}.json", "w"))
```

Statistics output:

```
fp32: mean 2.668 sd 0.043 (seeds=4)
per-seed delta vs the same seed's fp32 model (paired t-test p):
int8 per-row    mean 2.668  delta +0.000 (range -0.000 to +0.000)  p=0.3565
int6 per-row    mean 2.669  delta +0.001 (range +0.000 to +0.002)  p=0.0348
int5 per-row    mean 2.672  delta +0.004 (range +0.004 to +0.005)  p=0.0010
int4 per-row    mean 2.687  delta +0.020 (range +0.018 to +0.022)  p=0.0001
int3 per-row    mean 2.785  delta +0.117 (range +0.108 to +0.131)  p=0.0002
int2 per-row    mean 4.314  delta +1.647 (range +1.567 to +1.730)  p=0.0000
int8 per-tensor mean 2.668  delta +0.000 (range +0.000 to +0.000)  p=0.0568
int6 per-tensor mean 2.670  delta +0.003 (range +0.001 to +0.004)  p=0.0147
int5 per-tensor mean 2.679  delta +0.012 (range +0.009 to +0.015)  p=0.0025
int4 per-tensor mean 2.727  delta +0.060 (range +0.051 to +0.066)  p=0.0003
int3 per-tensor mean 3.036  delta +0.369 (range +0.280 to +0.443)  p=0.0017
int2 per-tensor mean 6.838  delta +4.171 (range +3.967 to +4.263)  p=0.0000
prune 30%     mean 2.694  delta +0.027 (range +0.023 to +0.033)  p=0.0013
prune 50%     mean 2.829  delta +0.161 (range +0.141 to +0.188)  p=0.0006
prune 70%     mean 3.503  delta +0.836 (range +0.722 to +0.993)  p=0.0007
prune 80%     mean 4.299  delta +1.632 (range +1.585 to +1.740)  p=0.0000
prune 90%     mean 5.734  delta +3.067 (range +3.002 to +3.109)  p=0.0000
prune 95%     mean 6.564  delta +3.897 (range +3.820 to +3.943)  p=0.0000
```

Next: [Part 4](/posts/llm-lab-4-decoding/) covers decoding and sampling: temperature, top-p and speculative decoding, and what each trades between quality and latency.

## References

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339), 2022.
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323), 2022.
- [The case for 4-bit precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720), 2022.
- [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774), 2023.
