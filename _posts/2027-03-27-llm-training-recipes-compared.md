---
title: "Ten LLMs, Ten Training Recipes: Tokens, Optimizers, Context Extension and Post-Training"
date: 2027-03-27 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, training, optimizer, muon, grpo, distillation, post-training]
description: "The ten open-weight models from the architecture comparison, now compared on how they were trained: token budgets, AdamW versus Muon, staged schedules, context extension and the post-training pipeline, each read from its own report."
---

The [previous post]({% post_url 2027-03-25-llm-architecture-appreciation %}) compared the architecture of ten open-weight models. This one compares how they were trained, using the same ten: Llama 3.1 8B, OLMo 2 7B, SmolLM3 3B, Gemma 3 27B, Qwen3-30B-A3B, GLM-4.5, DeepSeek-V3, Kimi K2, gpt-oss-120b and Qwen3-Next-80B-A3B. Every training fact below is taken from the model's technical report or official blog, linked at the point of use. Two small computations are the author's own and are labelled. Reports disclose very different amounts, so some cells are empty; an empty cell means the text read did not state it, not that the step was absent.

![Left: pre-training tokens per model with tokens per active parameter. Right: matrix of training ingredients each source names]({{ site.baseurl }}/assets/pic/llm-arch-training.png){: width="900" }
*Left: stated pre-training tokens; the bracket is tokens divided by active parameters (the author's arithmetic). Right: ingredients the sources name. Llama 3 and Qwen3 token counts are for the whole model family.*

## Token budgets

| Model | Pre-training tokens | Source |
| --- | --- | --- |
| Llama 3.1 8B | about 15T (the Llama 3 family) | [Llama 3 paper](https://arxiv.org/abs/2407.21783) |
| OLMo 2 7B | 4.05T in total, 3.90T of it pre-training | [OLMo 2 report](https://arxiv.org/abs/2501.00656) |
| SmolLM3 3B | 11.2T | [SmolLM3 blog](https://huggingface.co/blog/smollm3) |
| Gemma 3 27B | 14T | [Gemma 3 report](https://arxiv.org/abs/2503.19786) |
| Qwen3-30B-A3B | 36T (stated for the Qwen3 series) | [Qwen3 report](https://arxiv.org/abs/2505.09388) |
| GLM-4.5 | 23T | [GLM-4.5 report](https://arxiv.org/abs/2508.06471) |
| DeepSeek-V3 | 14.8T | [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) |
| Kimi K2 | 15.5T | [Kimi K2 report](https://arxiv.org/abs/2507.20534) |
| Qwen3-Next-80B-A3B | 15T, a uniformly sampled subset of Qwen3's 36T | [Qwen3-Next announcement](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) |
| gpt-oss-120b | "trillions of tokens", text only; no count | [gpt-oss model card](https://arxiv.org/abs/2508.10925) |

Dividing by parameters used per token (the author's arithmetic, with the per-token counts from the previous post and Gemma 3's 27.0B from its report) gives very different ratios: DeepSeek-V3 394 tokens per active parameter, Kimi K2 472, Gemma 3 519, OLMo 2 555, GLM-4.5 684, Llama 3.1 1,868, SmolLM3 3,648, Qwen3-Next 3,876 and Qwen3-30B-A3B 10,746, a 27-fold range. The two sparse models with 3 to 4 billion active parameters sit at the top, and the smallest dense model is close behind. The Qwen3-Next announcement claims the same architecture family reaches comparable quality at under 10% of Qwen3-32B's GPU hours; that is the vendor's claim and this post did not check it.

## Optimizers and the numbers around them

Three of the ten state AdamW and two state Muon; the other five do not say in the text read.

- **Llama 3** trains the 405B model with AdamW at peak learning rate 8e-5, an 8,000-step warm-up and a cosine decay to 8e-7 over 1,200,000 steps, and says the 8B and 70B models use similar recipes.
- **OLMo 2** keeps AdamW but lowers its epsilon from 1e-5 to 1e-8, adds a z-loss that keeps the output logits from growing, and excludes embeddings from weight decay. The 7B run uses peak learning rate 3e-4 with a cosine schedule defined over 5T tokens and truncated after 4T.
- **DeepSeek-V3** uses AdamW with beta1 = 0.9, beta2 = 0.95 and weight decay 0.1, a 4K sequence length, a linear warm-up to 2.2e-4 over 2K steps and a constant rate until 10T tokens, then a decay. It stores the first and second moments in BF16 and runs most matrix multiplications in FP8, with higher precision kept for sensitive operators.
- **Kimi K2** uses MuonClip with a WSD schedule: 2e-4 after a 500-step warm-up for the first 10T tokens, then a cosine decay from 2e-4 to 2e-5 over the next 5.5T, weight decay 0.1, 67M tokens per batch. The report says its earlier work found that "under the same compute budget and model size" Muon "substantially outperforms AdamW".
- **GLM-4.5** applies Muon to all parameters except the word embedding, biases and RMSNorm weights, with 5 Newton-Schulz steps, momentum 0.95 and the update scaled to an RMS of 0.2. It chose a cosine schedule over WSD because its early experiments found WSD "perform worse on general benchmarks (SimpleQA, MMLU)". It also schedules two other terms: the expert-bias update rate in loss-free routing is 0.001 for the first 15T tokens and 0 afterwards, and the multi-token-prediction loss weight drops from 0.3 to 0.1 at the same point.

What Muon does differently is small enough to show. AdamW rescales each parameter's gradient element by element. [Muon](https://kellerjordan.github.io/posts/muon/) takes the gradient matrix of a hidden layer and pushes all its singular values toward 1 with a short Newton-Schulz iteration, so the update has no dominant direction. The function below is the five-step version published with Muon; the demo feeds it a random matrix whose singular values span three orders of magnitude.

```python
def newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float() / (G.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if G.size(0) > G.size(1) else X
```

Run on a 256 x 1024 matrix, singular values (min / 25% / median / 75% / max):

```
raw gradient                   0.047  0.132  0.256  0.476  1.000   (scaled by its max)
after 5 Newton-Schulz steps    0.682  0.745  0.847  1.004  1.134
```

The spread falls from 21-fold to 1.7-fold. The iteration is deliberately approximate, which is why values land near 1 and not exactly on it. Kimi K2's contribution is the clipping step added on top: its report names the QK-Clip technique as the fix for the attention-logit instability Muon showed at scale.

## Staged schedules and context extension

Most of the ten do not train on one fixed mixture. They end on a shorter stage with different data, and then stretch the context.

- **Llama 3**: initial pre-training, long-context pre-training and annealing. Context grows through six stages from 8K to 128K over about 800B tokens.
- **OLMo 2**: a long pre-training stage (90 to 95% of FLOPs), then mid-training on a specialised mix with the rate decayed linearly to zero. For the 7B model, mid-training is three separate 50B-token runs on differently ordered data whose weights are averaged.
- **SmolLM3**: three stages over 11.2T tokens with web, math and code proportions that change. Two extension stages of 50B tokens each take context from 4K to 32K (RoPE base raised to 1.5M) and then to 64K (base 5M).
- **Gemma 3**: pre-trained on 32K sequences, not 128K from scratch, then rescaled by a positional-interpolation process. RoPE base on global layers goes from 10K to 1M, and local layers stay at 10K.
- **Qwen3**: about 30T tokens of general data, then a stage heavy in STEM and code, then a long-context stage from 4,096 to 32,768 tokens with the RoPE base raised from 10,000 to 1,000,000; YaRN and Dual Chunk Attention give a further 4x at inference.
- **DeepSeek-V3**: after 14.8T tokens, YaRN plus two phases of 1,000 steps each take context from 4K to 32K and then 128K.
- **Kimi K2**: an annealing phase with the rate decayed from 2e-5 to 7e-6 on 400B tokens at 4K, then 60B tokens at 32K, then YaRN to 128K. Its report also describes rephrasing high-quality text to get more from each token.
- **GLM-4.5**: multi-stage training over 23T tokens, then post-training with a maximum context of 128K.

## Post-training: three families

The ten split into recipes that lean on preference optimisation, on reinforcement learning against checkable rewards, and on distillation. Several use more than one.

**Preference optimisation without RL.** Llama 3 uses supervised fine-tuning, rejection sampling and direct preference optimisation in rounds, and states the reason: a "relatively simple post-training procedure ... as opposed to more complex reinforcement learning algorithms that tend to be less stable and harder to scale". SmolLM3 follows SFT with Anchored Preference Optimization, which its blog calls a variant of DPO, then merges models with MergeKit.

**Reinforcement learning with verifiable rewards.** OLMo 2-Instruct ends with RLVR, in the style of Tulu 3. Qwen3's flagship pipeline has four stages (long chain-of-thought cold start, reasoning RL, thinking-mode fusion, general RL); the reasoning-RL stage uses GRPO on 3,995 query-verifier pairs. GLM-4.5 builds on GRPO "excluding the KL loss term". Kimi K2 combines verifiable rewards with a self-critique rubric reward. gpt-oss reports chain-of-thought RL "similar" to OpenAI's o3 recipe. DeepSeek-V3 uses GRPO, which drops the separate critic network and estimates the baseline from a group of sampled answers to the same question. Its report defines the advantage as `A_i = (r_i - mean(r)) / std(r)`, which takes one line to compute:

```python
r = [1, 0, 0, 1, 0, 1, 0, 0]               # 8 sampled answers, checked by a verifier
m, s = sum(r)/len(r), (sum((x-m)**2 for x in r)/len(r))**0.5
A = [round((x-m)/s, 2) for x in r]          # [1.29, -0.77, -0.77, 1.29, -0.77, 1.29, -0.77, -0.77]
```

Correct answers get a positive advantage, wrong ones a negative advantage, and a group where every answer is right gets zero for all (`r = [1]*8` gives `A = [0]*8`), so questions the model already solves or never solves contribute no gradient. That is the author's reading of the formula, not a statement from the report.

**Distillation.** It appears at three different points:

- **During pre-training.** Gemma 3 samples 256 logits per token, weighted by the teacher's probabilities, and trains the student to match the teacher's distribution over that sample with cross-entropy. Its post-training again uses distillation from a larger instruction-tuned teacher, followed by RL based on BOND, WARM and WARP.
- **For small models.** Qwen3 uses off-policy and on-policy strong-to-weak distillation for its lightweight models, which its figure lists as Qwen3-30B-A3B and the dense models from 14B down. The report says this avoids running the full four-stage pipeline for each.
- **To merge specialists.** GLM-4.5 trains expert models for reasoning, agent tasks and general chat, then merges them by self-distillation in a unified stage, with SFT at the start of both stages. DeepSeek-V3 distils reasoning from its DeepSeek-R1 series after a 1.5M-instance SFT set.

One training detail belongs with deployment: gpt-oss post-trains its mixture-of-experts weights quantised to MXFP4 (4.25 bits per parameter). Those weights are over 90% of its parameters, which is why the 120B model fits on a single 80GB GPU, per the card.

## Reading across the ten

These are the author's observations from the sources above.

- Token budgets and architecture move independently. The densest budget per active parameter belongs to a 3B-active sparse model, not to the smallest dense one.
- Optimizer choice has started to split. The two models with the most distinctive sparse designs at trillion or near-trillion scale, Kimi K2 and GLM-4.5, both name Muon; Llama 3, OLMo 2 and DeepSeek-V3 name AdamW.
- Training no longer stops at the loss curve. At least seven of the ten sources describe a deliberate end-of-pre-training stage (annealing, mid-training or a long-context stage), and nine describe a post-training stage with more than plain SFT.
- RL against checkable answers is the common thread in the newer reports; distillation is how they pass the result to smaller or merged models.

## Limits

- Reports disclose unevenly. A blank in the figure's matrix means "not stated in the text read". The Qwen3-Next announcement was read for pre-training only, and gpt-oss gives no token count or optimizer.
- Llama 3 (15T) and Qwen3 (36T) token counts are stated for the model family; the 8B and 30B-A3B budgets are not separately stated in the text read. The tokens-per-active-parameter ratios mix those with single-model counts.
- Several recipe details (Llama 3.1 8B's own hyperparameters, SmolLM3's merging steps, GLM-4.5's stage token counts) were not extracted, and the ten sources describe a mix of base-model and instruct-model recipes.
- The Muon demo checks only the orthogonalising step on a random matrix; it says nothing about how the full optimizer performs. Quality differences between recipes cannot be read from this post; no benchmark results are compared.
- Everything here is a snapshot of reports available when this was written, and the ten models are a sample chosen for variety.
