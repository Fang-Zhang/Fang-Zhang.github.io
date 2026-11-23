---
title: "The LLM Engineer Path, Part 13: Interview Questions on Training, Quantization and Serving"
date: 2027-04-27 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, interview, DPO, RLHF, quantization, memory, inference, python]
description: "Part 13 of a 14-part series for developers moving into LLM engineering. Five classic questions on preference tuning, memory estimates, quantization, prefill versus decode, and metrics on imbalanced data, each answered in a sentence and checked with a short experiment."
---

Five interview questions on the system side of LLMs: how models are tuned to preferences, how much memory they need, what quantization costs, why serving has two phases, and a metrics trap that applies to any classifier. They are classic topics, not questions from any particular company. Each has a one-sentence answer, then the evidence.

## Q1. What are RLHF and DPO, and how do they differ from supervised fine-tuning?

**Answer.** Supervised fine-tuning imitates written demonstrations; RLHF and DPO learn from *comparisons* between two answers. RLHF fits a reward model to the comparisons and then optimises the language model against it with reinforcement learning; DPO skips the reward model and the sampling and trains directly on the pairs with a classification-style loss.

[InstructGPT](https://arxiv.org/abs/2203.02155) is the standard reference for the pipeline: demonstrations for supervised fine-tuning, then "a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback". Its headline result: "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters". The [DPO paper](https://arxiv.org/abs/2305.18290) calls RLHF "a complex and often unstable procedure" and shows the policy can be trained "with only a simple classification loss". The whole loss fits in a line:

```python
def dpo(lp_w, lp_l, ref_w, ref_l, beta=0.1):       # log-probs of the chosen (w) and rejected (l) answers
    return -np.log(sigmoid(beta * ((lp_w - ref_w) - (lp_l - ref_l))))
```

```
log-prob change vs the reference model     loss
chosen  +0, rejected  +0                   0.693
chosen +20, rejected  +0                   0.127
chosen +20, rejected +20                   0.693     (both raised equally: nothing learned)
chosen  +0, rejected +20                   2.127
chosen -20, rejected  +0                   2.127
```

The loss depends only on the *difference* between how much the model has moved toward the chosen answer and away from the rejected one, measured against a frozen reference model. That is also the pitfall to name in an interview: the loss can be lowered by pushing the rejected answer down, so it does not by itself require the chosen answer's probability to go up; check that on your own runs.

## Q2. How much memory does a model need for serving and for training?

**Answer.** Serving: parameters times bytes per parameter, plus the KV cache. Full fine-tuning with Adam in mixed precision: about 16 bytes per parameter before activations.

For the 16: the [ZeRO paper](https://arxiv.org/abs/1910.02054) counts "2Ψ + 2Ψ + KΨ = 16Ψ bytes" for 16-bit weights, 16-bit gradients and optimizer states, where "mixed-precision Adam has K = 12" (a 32-bit copy of the weights, momentum and variance, 4 bytes each). For a 7-billion-parameter model:

```
weights:  fp32 28 GB   16-bit 14 GB   int8 7.0 GB   int4 3.5 GB
full fine-tuning, model states only (16 bytes per parameter): 112 GB
KV cache, a 32-layer 4096-wide model with full multi-head attention (hypothetical):
    2 (K and V) x 32 layers x 4096 x 2 bytes = 524,288 bytes = 0.5 MiB per token
    4,096 tokens x 8 concurrent requests = 17.2 GB
```

Two consequences. Full fine-tuning needs roughly eight times the memory of serving the same model in 16-bit, which is why LoRA (Part 6) trains small adapter matrices instead. And at long contexts and many concurrent requests, the cache can outgrow the weights, which is why Parts 2 and 9 treat it as the main serving cost.

## Q3. What does quantization do, and where does it lose accuracy?

**Answer.** It stores weights (and sometimes activations) in fewer bits, using a scale factor to map the range onto integers. The loss comes from outliers: one large value sets the scale for everything that shares it, so the rest gets rounded coarsely.

[LLM.int8()](https://arxiv.org/abs/2208.07339) describes "highly systematic emergent features" in large transformers, a few hidden dimensions with very large values, and handles them by isolating "the outlier feature dimensions into a 16-bit matrix multiplication while still more than 99.9% of values are multiplied in 8-bit". A toy version of the effect: activations with six outlier channels multiplied by 1, 10 or 60, multiplied by a weight matrix, with absmax 8-bit rounding:

```python
def q8(a, axis=None):
    s = np.abs(a).max(axis=axis, keepdims=True) / 127      # one scale per tensor, row or column
    return np.round(a / s) * s
```

```
outlier size   per-tensor   per-token + per-column scales   outlier channels kept in 16-bit
     1x           1.50%              1.11%                              1.11%
    10x           6.62%              3.26%                              0.89%
    60x           9.90%              5.33%                              0.24%
```

(Relative error of the matrix product x @ W.) Finer scales help; keeping the few outlier channels at full precision helps most. The toy has no trained model behind it, so the errors show the mechanism and not what an 8-bit model loses in task accuracy, which has to be measured on your task.

## Q4. Why is generation split into prefill and decode, and which is slower?

**Answer.** Prefill processes the whole prompt in one pass, so each weight read from memory is used for many tokens; decode produces one token per pass, so each weight read serves one token. Per token, prefill is much cheaper, and decode is limited by memory traffic.

The measurement: an 8-layer, 1024-wide transformer on a CPU with random weights (the same as Part 9), one request:

```
one token (a decoding step):          27.7 ms =   36 tokens/s
 16-token prompt in one pass:         53.3 ms =  300 tokens/s
 64-token prompt in one pass:         72.0 ms =  889 tokens/s
256-token prompt in one pass:        131.1 ms = 1952 tokens/s
512-token prompt in one pass:        338.4 ms = 1513 tokens/s
```

At 256 tokens the pass processes each token 54 times cheaper than a decoding step. Throughput falls again at 512 as the n x n attention and cache effects grow (Parts 2 and 11). This is the reason serving systems report time to first token (prefill-dominated) and time per output token (decode-dominated) separately, and why batching requests (Part 9) raises decode throughput: more tokens share each weight read. The numbers are from a CPU; a GPU has a different balance, so measure on your hardware.

![Three panels. Left: DPO loss against the margin between chosen and rejected log-probability changes, 0.693 at zero. Middle: 8-bit quantization error with no outliers, 10 times and 60 times outliers for three scaling schemes, rising to 9.9% for per-tensor and falling to 0.2% when outlier channels stay in 16-bit. Right: tokens per second for one decoding step, 36, against prefill of 16 to 512 tokens, 300 to 1952]({{ site.baseurl }}/assets/pic/llmeng-interview3.png){: width="900" }

## Q5. A classifier has 95% accuracy. Is it good?

**Answer.** Not necessarily: with imbalanced classes, accuracy rewards predicting the majority class. Report precision and recall for the class that matters.

A safety filter on 1,000 messages where 50 are toxic:

```
always answer "safe"                                  accuracy 95.0%  precision 0.00  recall 0.00  F1 0.00
real classifier (finds 30 of 50, 20 false alarms)     accuracy 96.0%  precision 0.60  recall 0.60  F1 0.60
```

Precision is the share of flagged messages that are really toxic; recall is the share of toxic messages that get flagged; F1 is their harmonic mean. The model that does nothing scores 95%. The same trap applies to guardrails (Part 10: 5 of 20 benign inputs blocked alongside 5 of 48 attacks) and to the abstain rule in Part 12.

## Try it

1. Q2: look up the layer count, width and attention-head layout of a model you use, and compute its KV cache per token. If it uses fewer key-value heads than query heads, how much does the number shrink?
2. Q3: use 4-bit rounding (replace 127 with 7) in the experiment. At which outlier size does each scheme stop being usable?
3. Q4: time prefill and decode on a different model size and compare the ratio.

## Limits

- Memory figures follow the cited accounting and exclude activations, framework overhead and fragmentation; real usage is higher.
- The quantization experiment uses random weights and synthetic outliers, and the prefill measurement uses random weights on a CPU. They show mechanisms, and trained models and GPUs shift the numbers.
- DPO and RLHF are described at the level of the papers' abstracts and the loss above; practical recipes (data quality, the choice of beta, reference model handling) are not covered.
