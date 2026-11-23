---
title: "The LLM Engineer Path, Part 6: Fine-Tuning, Full and LoRA"
date: 2027-04-20 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, fine-tuning, LoRA, QLoRA, catastrophic forgetting, memory, python]
description: "Part 6 of a 14-part series for developers moving into LLM engineering. When to fine-tune, why full fine-tuning needs 16 bytes per parameter, how LoRA trains a small fraction of that, and a measured comparison on a small Transformer: rank against steps, merging with no inference cost, and forgetting that LoRA does not prevent."
---

Fine-tuning continues training a pre-trained model on your own examples so that it changes its behaviour: a format, a tone, a task. This post covers the memory cost that makes full fine-tuning hard, the LoRA method that avoids most of it, and a small experiment that measures both the saving and a risk the saving does not remove.

## Prompt, retrieve, or fine-tune

Fine-tuning changes how a model behaves; it is a poor way to add facts that change often. A rough order of effort:

- Behaviour wrong, facts fine: start with a better prompt and examples (Part 3).
- Facts missing or changing: retrieval (Parts 4 and 5), because documents can be updated without training.
- A stable behaviour that prompting cannot hold, or a smaller model that must do one job cheaply and reliably: fine-tune.

Whichever you pick, measure first (Part 8). Fine-tuning without a test set cannot tell you whether it worked.

## Why full fine-tuning is expensive

Full fine-tuning updates every parameter. Training memory is dominated by "model states": parameters, gradients and optimizer state. The ZeRO paper ([Rajbhandari et al.](https://arxiv.org/abs/1910.02054)) defines the model state as "optimizer states (such as momentum and variances in Adam), gradients, and parameters" and uses a multiplier K = 12 for mixed-precision training with Adam. With 2 bytes for each parameter and 2 for each gradient, that is (2 + 2 + 12) = 16 bytes per parameter. That is the paper's accounting; real runs add activations and buffers on top.

For a 7-billion-parameter model, this post's arithmetic gives 7e9 x 16 bytes = 112 GB before activations, more than any single 80 GB GPU.

## LoRA: train a small update

[LoRA](https://arxiv.org/abs/2106.09685) "freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture". For a weight matrix W of shape (out, in), the layer computes

```
y = W x + (alpha / r) * B (A x)       A: (r, in)   B: (out, r)   r is small (1 to 64)
```

Only A and B are trained. B starts at zero, so training begins from the original model. A layer with 1,024 x 1,024 weights has about 1.05 million parameters; with r = 8 the adapter has 2 x 8 x 1,024 = 16,384, about 1.6%. The paper reports that, against GPT-3 175B fine-tuned with Adam, LoRA "can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times", with "no additional inference latency".

[QLoRA](https://arxiv.org/abs/2305.14314) goes further: it backpropagates "through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters", enough to "finetune a 65B parameter model on a single 48GB GPU".

The code below is the whole method for one linear layer, followed by the function that attaches it to every projection of a small 2-layer Transformer (237,351 parameters):

```python
class LoRALinear(nn.Module):                     # y = W x + (alpha/r) * B(A x), with W frozen
    def __init__(s, base, r, alpha):
        super().__init__(); s.base, s.scale = base, alpha / r
        s.A = nn.Parameter(torch.randn(r, base.in_features) * 0.02); s.B = nn.Parameter(torch.zeros(base.out_features, r))
        for p in s.base.parameters(): p.requires_grad = False
    def forward(s, x): return s.base(x) + s.scale * (x @ s.A.T @ s.B.T)
    def merge(s):                                 # fold the update into W: no extra work at inference
        with torch.no_grad(): s.base.weight += s.scale * (s.B @ s.A)
        return s.base

def add_lora(m, r):
    for p in m.parameters(): p.requires_grad = False
    for b in m.blocks:
        for nme in ("qkv", "o", "up", "down"): setattr(b, nme, LoRALinear(getattr(b, nme), r, 2 * r))
    return m
count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
```

`merge` is the "no inference latency" claim in code: after training, add (alpha / r) x B A into W and drop the adapter.

## The experiment

The model is pre-trained for 800 steps on records like `{"name": "anna", "age": 34, "city": "oslo"}` (task A), where each name always goes with one city. Then it is fine-tuned on a new format, `"country": "norway"` (task B), with 8 names and a different value for each. We then ask it to complete 24 prompts per task (8 names x 3 ages) and count exact matches. Full fine-tuning is compared with LoRA at ranks 1, 4 and 16 on the same base model, for 3 seeds.

Real output, 300 fine-tuning steps (accuracy as a fraction of 24 prompts):

```
base model              new-task 0.00   old-task 1.00        (all 3 seeds)
full fine-tune          new-task 1.00   old-task 0.00        237,351 trainable   (3 of 3 seeds)
LoRA rank 4             new-task 0.25 / 0.75 / 0.71          12,288 trainable    old-task 0.00
LoRA rank 1             new-task 0.00 / 0.00 / 0.00          3,072 trainable     old-task 0.00
merge check (rank 4):   max |output before merge - after| = 2e-05 to 4e-05
```

Then a sweep of rank against number of steps (mean of 3 seeds):

```
rank   300 steps   1,000 steps   3,000 steps   trainable
  1       0.00        0.50          0.79          3,072
  4       0.58        1.00          1.00         12,288
 16       0.96        1.00          1.00         49,152
```

![Left: new-task accuracy against fine-tuning steps for LoRA at rank 1, 4 and 16: rank 16 reaches 96 percent at 300 steps, rank 4 reaches 58 percent then 100 percent at 1,000 steps, rank 1 reaches 0, 50 and 79 percent; a dashed line marks full fine-tuning at 100 percent. Right: model-state memory for a 7-billion-parameter model, 112 GB for full fine-tuning against 14 GB for the frozen 16-bit weights under LoRA]({{ site.baseurl }}/assets/pic/llmeng-lora.png){: width="900" }
*Left: measured. Right: this post's arithmetic from the ZeRO accounting; LoRA adds the adapter, its optimizer state and activations on top of the 14 GB.*

What it shows:

- **LoRA works with about 5% of the parameters.** Rank 4 trains 12,288 of 237,351 and reaches 100% after 1,000 steps.
- **Fewer parameters cost more steps.** Rank 1 is still at 79% after 3,000 steps; rank 16 is at 96% after 300. Rank is a capacity setting, and the right value depends on how far the task is from the base model.
- **Results at 300 steps vary by seed** (rank 4 gave 0.25, 0.75 and 0.71 in one run and 0.50, 0.62 and 0.62 in another). Do not tune hyper-parameters from a single run.
- **Merging is exact up to rounding.** The merged model's outputs differ from the adapter model's by at most about 4e-05, so serving needs no extra layers.
- **The memory bar is the point at scale.** For the 7-billion-parameter case, 112 GB of model state against 14 GB of frozen weights is the difference between a rack and one GPU, before activations.

## The failure to reproduce: forgetting

Look at the `old-task` column: **0.00 for every method**, including LoRA. After training only on the new format, the model no longer writes the old one. LoRA freezes the original weights, but the adapted layers still change what the model outputs, and the adapter never saw the old task. A frozen base is not a guarantee of retained behaviour.

The standard remedy is to keep examples of the old behaviour in the fine-tuning set. Full fine-tuning on a half-old, half-new mix, same 300 steps:

```
seed 0  new only: new 1.00 old 0.00    old+new mix: new 0.62 old 1.00
seed 1  new only: new 1.00 old 0.00    old+new mix: new 0.88 old 1.00
seed 2  new only: new 1.00 old 0.00    old+new mix: new 1.00 old 1.00
```

The old task is fully retained in all three seeds; the new task is learned more slowly, because half the steps went to the old one. In practice, put regression examples for everything the model must keep doing into the test set (Part 8) and re-run it after every fine-tune.

## Try it

1. Apply LoRA only to `qkv` instead of all four projections. How many parameters is that, and where does accuracy end up at 1,000 steps?
2. Change `alpha` from `2 * r` to `r` in `add_lora`. The scale `alpha / r` multiplies the update; what happens to the speed of learning?
3. Using your target model's configuration (layers, hidden size), work out the model-state memory with the 16-bytes rule and with 2 bytes per frozen weight, as in the right panel.

## Limits

- The model is tiny and the task is a lookup table. Real instruction tuning has far more variety, and results on a toy task do not transfer in size.
- The memory numbers are bookkeeping from one paper's accounting; they leave out activations, which depend on batch size and sequence length, and the adapter and its optimizer state.
- Only exact-match accuracy on 24 prompts per task was measured; a difference of one prompt is 4 points.
- Methods not covered: adapters, prefix tuning, preference optimisation (the later stage that follows supervised fine-tuning), and data quality, which usually matters more than the method.
