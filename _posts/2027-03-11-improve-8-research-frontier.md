---
title: "Improving LLM Performance (8/8): Open Problems at the Research Frontier"
date: 2027-03-11 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, synthetic-data, model-collapse, process-supervision, state-space-models, series]
description: "Three open questions behind the levers of this series, each with a source and a measurement: training on model-generated data, supervising reasoning steps, and architectures whose memory does not grow with context. Ends with a recap of what the toy runs of Parts 1 to 7 measured."
---

The first seven parts treated levers that are in use. This last part picks three questions that the published literature treats as open, gives each a source and, where a CPU can say something, a measurement: what happens when models train on their own output, whether to supervise reasoning steps rather than outcomes, and whether attention's growing memory can be avoided. A recap of the series follows.

## Training on what models generate

Data is the lever of [Part 2](/posts/improve-2-pretraining-data/), and a growing share of the text available to future crawls is model-written. [Shumailov et al.](https://arxiv.org/abs/2305.17493) find that using model-generated content in training causes irreversible defects in the resulting models, where tails of the original content distribution disappear, and name the effect model collapse, shown in variational autoencoders, Gaussian mixtures and language models. [Gerstgrasser et al.](https://arxiv.org/abs/2404.01413) point out that those studies assumed new data replace old data; they confirm that replacement tends toward collapse, and report that accumulating each generation's synthetic data alongside the original real data avoids it, across model sizes, architectures and hyperparameters.

The toy: a byte-level Transformer (4 layers, 64 wide) trained on 500k bytes of Python source, 800 steps. Each generation samples 500k bytes from the model at temperature 1, starting every sequence from a newline byte. In the replace regime the next generation trains on those samples only; in the accumulate regime it trains on the real text plus all samples so far. Same seeds in both regimes, so generation 0 is identical.

```python
for regime in ('replace', 'accumulate'):
    pool = real; rows = []
    for gen in range(6):
        m = fit(pool, 1 + gen); vl = val_loss(m); samp = generate(m, D, 100 + gen); rows.append((gen, vl, distinct(samp)))
        pool = samp if regime == 'replace' else np.concatenate([pool, samp])
```

```
real data: distinct 8-grams 0.375
replace    generation 0: validation loss on real text 1.9585   distinct 8-grams in its samples 0.739   training pool 500,129 bytes
replace    generation 1: validation loss on real text 2.0851   distinct 8-grams in its samples 0.771   training pool 499,968 bytes
replace    generation 2: validation loss on real text 2.1847   distinct 8-grams in its samples 0.769   training pool 499,968 bytes
replace    generation 3: validation loss on real text 2.2359   distinct 8-grams in its samples 0.758   training pool 499,968 bytes
replace    generation 4: validation loss on real text 2.2879   distinct 8-grams in its samples 0.741   training pool 499,968 bytes
replace    generation 5: validation loss on real text 2.3309   distinct 8-grams in its samples 0.739   training pool 499,968 bytes
accumulate generation 0: validation loss on real text 1.9585   distinct 8-grams in its samples 0.739   training pool 500,129 bytes
accumulate generation 1: validation loss on real text 2.0170   distinct 8-grams in its samples 0.801   training pool 1,000,097 bytes
accumulate generation 2: validation loss on real text 2.0644   distinct 8-grams in its samples 0.788   training pool 1,500,065 bytes
accumulate generation 3: validation loss on real text 2.0780   distinct 8-grams in its samples 0.784   training pool 2,000,033 bytes
accumulate generation 4: validation loss on real text 2.1066   distinct 8-grams in its samples 0.814   training pool 2,500,001 bytes
accumulate generation 5: validation loss on real text 2.1091   distinct 8-grams in its samples 0.779   training pool 2,999,969 bytes
```

![Left: validation loss on real text against generation for the replace and accumulate regimes. Middle: share of distinct 8-grams in each generation's samples, with the real text as a reference line. Right: memory per sequence against context length for an attention cache and a fixed-size recurrent state](/assets/pic/improve-8-frontier.png)
*Left: replacement raises the loss on real text every generation, accumulation raises it more slowly and flattens. Middle: the diversity measure does not fall in this toy. Right: attention's cache grows with context, a recurrent state does not.*

- **Replacement degrades the model each generation.** The loss on real text goes from 1.959 to 2.331 over five generations of retraining on the previous generation's samples only, and each step adds to it.
- **Accumulation slows the damage but does not remove it here.** The loss reaches 2.109, with the last step adding 0.003, so it appears to be flattening but has not been run further. The paper reports that accumulation avoids collapse; this toy shows a much smaller effect, with a first-generation model that is weak (1.96 nats) and a fixed step budget while the pool grows to 3M bytes.
- **The toy does not show tail loss.** Distinct 8-grams in the samples stay near 0.74 to 0.81, above the 0.375 of real text. A weak model sampled at temperature 1 produces noisier text than the real corpus, so the degradation measured is the accumulation of that noise and not the vanishing of rare content that the paper describes. The loss on real text is the evidence here, and the diversity measure is not.

## Supervising steps, not only outcomes

[Part 4](/posts/improve-4-rl-verifiable-rewards/) used a reward on the final answer. [Lightman et al. (2023)](https://arxiv.org/abs/2305.20050) compare outcome supervision with process supervision, which gives feedback for each intermediate step, and find that process supervision significantly outperforms outcome supervision for training models to solve problems from the MATH dataset; their process-supervised model solves 78% of a representative subset of the test set, and active learning improves process supervision further. They release PRM800K, 800,000 step-level human labels. The open parts are the cost of step labels, how far this extends to tasks without clear steps, and how to get step-level signals without people. No toy is run here: Part 5's noisy checker already shows how a verifier's error rate caps what selection can gain.

## Memory that does not grow with context

[Part 6](/posts/improve-6-architecture-efficiency/) showed that the KV cache grows with context. [Mamba (Gu and Dao, 2023)](https://arxiv.org/abs/2312.00752) builds on structured state space models, which keep a fixed-size state; earlier ones were weaker than attention on language because they could not select content, and Mamba makes the state-space parameters functions of the input so the model can propagate or forget information depending on the current token. The paper reports 5 times higher inference throughput than Transformers and performance that improves on real data up to million-length sequences. The arithmetic below shows why this is attractive; the shapes are assumptions of this post, not a named model.

```python
# Illustrative memory arithmetic (the shapes are assumptions, not a named model): 32 layers, fp16.
L, B = 32, 2
kv_per_token = 2 * L * 8 * 64 * B                      # attention with 8 key-value heads of width 64: keys and values for every past token
state = L * 4096 * 16 * B                              # a recurrent (state-space) layer with 4096 channels and state size 16 keeps a fixed-size state
print(f'attention cache: {kv_per_token / 1024:.0f} KiB per token; at 8k tokens {kv_per_token * 8192 / 2**30:.2f} GiB, at 128k tokens {kv_per_token * 131072 / 2**30:.1f} GiB, at 1M tokens {kv_per_token * 2**20 / 2**30:.0f} GiB')
print(f'recurrent state: {state / 2**20:.1f} MiB at any length; equal to the attention cache at {state / kv_per_token:.0f} tokens')
```

```
attention cache: 64 KiB per token; at 8k tokens 0.50 GiB, at 128k tokens 8.0 GiB, at 1M tokens 64 GiB
recurrent state: 4.0 MiB at any length; equal to the attention cache at 64 tokens
```

The price is capacity: a state of fixed size cannot hold every earlier token the way a cache can, which is what the content-based selection is for. Whether this reaches attention's quality at the largest scales is an open question that this post cannot answer.

## What the series measured

- **Part 1:** training-loss fits to $L = E + A/N^{a} + B/D^{b}$ missed the toy runs by a wide margin, a mixed result reported as such.
- **Part 2:** in a corpus of Python files, 39% of documents had more than one copy, and deduplication changed the bytes kept and the validation loss.
- **Part 3:** SFT on 128 or more demonstrations reached 96% to 98% accuracy at about 0.67 nats of KL; DPO reached less at a higher KL, and the chosen answer's log-probability fell in all nine of the larger runs.
- **Part 4:** from a 5% base model, a larger group learned faster; RL moved pass@1 from 41% and 5.9% to about 95% while pass@64 was already near 100%.
- **Part 5:** a scratchpad took 3-digit multiplication from 5% to about 89%; majority vote fell to 1.4% when wrong answers were identical; a noisy checker capped at 42% to 52%.
- **Part 6:** 8 key-value heads cut the cache from 16 GB to 4 GB; 8 experts beat the dense model at equal per-token compute; speculative decoding followed its formula.
- **Part 7:** in DeepSeek-V3's report, post-training is 0.18% of the GPU-hours.
- **This part:** retraining on one's own samples raised the loss on real text every generation.

## Limits

- The model-collapse toy uses a small model, 500k bytes, 6 generations and one seed per regime. The accumulate result differs in size from the paper's, and the cause was not isolated.
- The memory arithmetic assumes the shapes stated in the code; it says nothing about quality or measured throughput, and the quoted Mamba figures are the paper's own.
- The three open questions are a selection by the author, not a survey of the field.
- The process-supervision numbers are quoted from the abstract and were not reproduced.
