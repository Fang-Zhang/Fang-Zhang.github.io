---
title: "An Appreciation of LLM Architectures: Ten Open-Weight Models Compared"
date: 2027-03-25 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, architecture, mixture-of-experts, attention, kv-cache, deepseek, qwen, gemma]
description: "Ten open-weight language models read from their own config.json files and technical reports: where their attention, feed-forward and normalisation choices agree, where they differ, and what each choice costs in KV-cache memory and parameters."
---

Open-weight language models now publish enough that their architectures can be compared from primary sources: the `config.json` that ships with the weights, and the technical report or model card. This post reads ten of them that way, then computes two quantities from the configs with a short script: the KV cache for one 131,072-token sequence, and total against per-token parameters. The ten are chosen by the author to cover different design choices, not to rank models: Llama 3.1 8B, OLMo 2 7B, SmolLM3 3B, Gemma 3 27B, Qwen3-30B-A3B, GLM-4.5, DeepSeek-V3, Kimi K2, gpt-oss-120b and Qwen3-Next-80B-A3B. For a longer figure-by-figure tour of many more models, see Sebastian Raschka's [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison).

## What all ten share

Each is a decoder-only transformer with rotary position embeddings configured in its config file (SmolLM3 removes them in every fourth layer; see below), RMSNorm, and a gated feed-forward block (SwiGLU where the paper says so: [Llama 3](https://arxiv.org/abs/2407.21783), [Qwen3](https://arxiv.org/abs/2505.09388), [gpt-oss](https://arxiv.org/abs/2508.10925)). Seven of the ten use grouped-query attention, two (DeepSeek-V3 and Kimi K2) use latent attention, and OLMo 2 7B uses plain multi-head attention, with 32 key-value heads for 32 query heads. [Llama 3](https://arxiv.org/abs/2407.21783) states the reason: GQA with 8 key-value heads "to improve inference speed and to reduce the size of key-value caches during decoding". The differences below are all variations on three questions: how attention is stored, how the feed-forward block is sparsified, and how training is kept stable.

## The ten at a glance

Values below are read from each model's `config.json` unless a report is named.

| Model | Layers x hidden | Attention (query / KV heads) | Feed-forward |
| --- | --- | --- | --- |
| Llama 3.1 8B | 32 x 4096 | full, 32 / 8 | dense, 14336 |
| OLMo 2 7B | 32 x 4096 | full, 32 / 32 | dense, 11008 |
| SmolLM3 3B | 36 x 2048 | full, 16 / 4; no RoPE in every 4th layer | dense, 11008 |
| Gemma 3 27B | 62 x 5376 | 5 sliding-window (1024) : 1 full, 32 / 16 | dense, 21504 |
| Qwen3-30B-A3B | 48 x 2048 | full, 32 / 4 | 128 experts, top 8, expert width 768, no shared expert |
| GLM-4.5 | 92 x 5120 | full, 96 / 8, partial RoPE | 160 routed + 1 shared, top 8, width 1536; first 3 layers dense |
| DeepSeek-V3 | 61 x 7168 | MLA, 128 heads, KV rank 512 | 256 routed + 1 shared, top 8, width 2048; first 3 layers dense |
| Kimi K2 | 61 x 7168 | MLA, 64 heads, KV rank 512 | 384 routed + 1 shared, top 8, width 2048; first layer dense |
| gpt-oss-120b | 36 x 2880 | alternating window (128) / full, 64 / 8 | 128 experts, top 4, width 2880 |
| Qwen3-Next-80B-A3B | 48 x 2048 | 3 linear : 1 full, 16 / 2 on full layers | 512 routed + 1 shared, top 10, width 512 |

*Sources: the ten config files; Gemma 3's 5:1 pattern and 1024 window from the [Gemma 3 report](https://arxiv.org/abs/2503.19786); Qwen3-Next's 3:1 pattern from the [Qwen3-Next announcement](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list), with `full_attention_interval: 4` in its config.*

![Layer-by-layer map of the ten models. Top bar of each row is the attention type per layer, bottom bar is dense or mixture-of-experts feed-forward]({{ site.baseurl }}/assets/pic/llm-arch-layer-stacks.png){: width="900" }
*Each column is one layer. Top bar: attention type. Bottom bar: feed-forward type. Drawn from the config files.*

## Attention: what gets cached

During generation, every layer keeps the keys and values of all previous tokens. For grouped-query attention that costs `2 x kv_heads x head_dim` values per layer per token. The designs below reduce it in three different ways.

- **Fewer KV heads** (GQA): Llama, Qwen3, GLM-4.5, gpt-oss, SmolLM3, Gemma 3. The ratio of query to KV heads is the knob: Qwen3-30B-A3B uses 32 query heads over 4 KV heads, OLMo 2 uses 32 over 32.
- **Fewer layers that look far back.** [Gemma 3](https://arxiv.org/abs/2503.19786) interleaves five local sliding-window layers (span 1024) with one global layer, and says this was done because "the memory explosion of the KV cache during inference" is a challenge with long context. gpt-oss alternates dense layers with banded layers whose "bandwidth is 128 tokens" ([model card](https://arxiv.org/abs/2508.10925)). Qwen3-Next goes further: 75% of its layers are [Gated DeltaNet](https://arxiv.org/abs/2412.06464) linear-attention layers, which keep a fixed-size state instead of a cache that grows with length.
- **A smaller cached object** (MLA, multi-head latent attention). DeepSeek-V3 and Kimi K2 cache one compressed vector per token per layer, of size `kv_lora_rank + qk_rope_head_dim` = 512 + 64 = 576 values, instead of per-head keys and values ([DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)).

The script below turns each config into a number. Layer counts for the hybrid models come from the layer types in the config (`layer_types`, `full_attention_interval`) and, for Gemma 3, from the 5:1 rule in the report.

```python
def kv_bytes(model, ctx, dtype_bytes=2):
    if model.attn == "mla":                      # one latent vector per layer per token
        return model.layers * (model.kv_lora_rank + model.rope_dim) * dtype_bytes * ctx
    per_layer = 2 * model.kv_heads * model.head_dim * dtype_bytes   # K and V
    full  = model.n_full_layers * per_layer * ctx
    local = model.n_window_layers * per_layer * min(ctx, model.window)
    return full + local + model.linear_state_bytes                  # linear layers: constant state
```

Output for one sequence, 16-bit cache:

| Model | KV cache at 4,096 tokens | KV cache at 131,072 tokens |
| --- | --- | --- |
| OLMo 2 7B | 2.00 GiB | 64.0 GiB (its config maximum is 4,096) |
| GLM-4.5 | 1.44 GiB | 46.0 GiB |
| Llama 3.1 8B | 0.50 GiB | 16.0 GiB |
| Qwen3-30B-A3B | 0.38 GiB | 12.0 GiB |
| Gemma 3 27B | 0.72 GiB | 10.4 GiB |
| SmolLM3 3B | 0.28 GiB | 9.0 GiB |
| DeepSeek-V3 and Kimi K2 | 0.27 GiB | 8.6 GiB |
| gpt-oss-120b | 0.15 GiB | 4.5 GiB |
| Qwen3-Next-80B-A3B | 0.13 GiB | 3.0 GiB |

![Left: KV cache for a 131,072-token sequence, ten models. Right: total parameters against per-token parameters on log axes]({{ site.baseurl }}/assets/pic/llm-arch-kv-and-params.png){: width="900" }
*Left: KV cache at 131,072 tokens computed from the configs. Right: total against per-token parameters, computed from the configs.*

What the numbers show:

- GLM-4.5 has the largest cache of the post despite using GQA with 8 KV heads, because it has 92 layers of head dimension 128. Its [report](https://arxiv.org/html/2508.06471) says it chose depth over width: "we reduce the width ... and increase its height (number of layers), as we found that deeper models exhibited better reasoning capacity". The cache is the price of that choice.
- MLA is 8.6 GiB for DeepSeek-V3. Had the same model cached per-head keys and values (128 heads, 192-dimensional keys and 128-dimensional values), the same formula gives 610 GiB, so the compression is about 71 times (the author's arithmetic from the config).
- Kimi K2 has the same cache as DeepSeek-V3 because they share the MLA dimensions and the layer count. The [Kimi K2 report](https://arxiv.org/abs/2507.20534) lists the differences it made: 384 experts instead of 256, and 64 attention heads instead of 128 "to reduce computational overhead during inference".
- For the linear-attention layers in Qwen3-Next, the constant state is about 36 MiB in 16-bit by the author's estimate (36 layers x 32 value heads x 128 x 128 values), plus a small convolution state not counted; the 3.0 GiB is almost entirely the 12 full-attention layers.

## Feed-forward: sparsity and how it is spread

Six of the ten replace most feed-forward blocks by a mixture of experts: a router picks a few expert blocks per token. The configs show how differently they cut it.

- **Many small experts.** DeepSeek-V3 uses 256 routed experts of width 2048 (hidden size 7168) with 8 active per token, plus one shared expert that every token uses and three dense layers at the bottom ([report](https://arxiv.org/abs/2412.19437)). Qwen3-30B-A3B uses 128 experts of width 768 (hidden size 2048) with 8 active, and per the [Qwen3 report](https://arxiv.org/abs/2505.09388) "excludes shared experts". Qwen3-Next goes to 512 experts of width 512, ten active plus one shared.
- **Fewer, larger experts.** gpt-oss-120b has 128 experts of width 2880 (equal to its hidden size) and 4 active per token, and its [model card](https://arxiv.org/abs/2508.10925) says it weights the chosen experts' outputs by a softmax over only the selected ones.
- **More experts at fixed active compute.** Kimi K2 raises the expert count from 256 to 384 at the same top-8 and says its scaling-law analysis found that "continued increases in sparsity yield substantial performance improvements".
- **Load balancing.** DeepSeek-V3 pioneers an "auxiliary-loss-free strategy for load balancing" in its abstract; GLM-4.5 says it also uses loss-free balance routing and sigmoid gates; Qwen3 uses a global-batch load-balancing loss.

The right panel of the figure shows the result: every MoE model sits far below the dense diagonal. The counts come from a formula applied to each config (attention projections, feed-forward weights, router, both embedding tables; biases, norms and any MTP module are left out):

```python
dense   = emb + L * (attn + 3 * h * ffn)
moe_tot = emb + L * (attn + E * 3 * h * moe_ffn + shared + h * E)
moe_act = emb + L * (attn + k * 3 * h * moe_ffn + shared + h * E)   # k experts used per token
```

Computed against what each source states:

| Model | Computed total / per-token | Source states |
| --- | --- | --- |
| DeepSeek-V3 | 671.0B / 37.6B | 671B / 37B ([report](https://arxiv.org/abs/2412.19437)) |
| Kimi K2 | 1,026B / 32.9B | 1T / 32B ([report](https://arxiv.org/abs/2507.20534)) |
| GLM-4.5 | 352.8B / 33.6B | 355B / 32B ([report](https://arxiv.org/html/2508.06471)) |
| gpt-oss-120b | 116.8B / 5.71B | 116.8B / 5.1B, "unembedding parameters are counted towards active, but not embeddings" ([model card](https://arxiv.org/abs/2508.10925)) |
| Qwen3-Next-80B-A3B | 79.7B / 3.87B | 80B / 3B ([announcement](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)) |
| Gemma 3 27B (dense) | non-embedding 25.60B | non-embedding 25,600M ([report](https://arxiv.org/abs/2503.19786)) |

The gpt-oss gap is the embedding convention: the card leaves out the input embedding table, which is 201,088 x 2,880 = 0.58B values, and 5.71B - 0.58B = 5.13B, matching the card's 5.1B. The other per-token figures here include both embedding tables, so they sit slightly above vendors' rounded numbers.

## Stability tricks and position handling

These are small in parameter count and appear in most of the ten.

- **QK-norm.** OLMo 2, Gemma 3, Qwen3 and GLM-4.5 normalise the query and key projections before computing attention. OLMo 2's report says this "avoids attention logits being too large, which can lead to training loss divergence"; Gemma 3 replaces the soft-capping of Gemma 2 with it.
- **Where the norm sits.** OLMo 2 normalises the output of attention and feed-forward blocks, not their input: `h = x + RMSNorm(Attention(x))` ([report](https://arxiv.org/abs/2501.00656)). Gemma 3 uses both pre- and post-norm. Qwen3 states pre-normalisation.
- **A softmax escape hatch.** gpt-oss gives each attention head a learned bias in the softmax denominator "similar to off-by-one attention and attention sinks", which lets a head attend to nothing. Qwen3-Next's announcement reports that an attention output gate helps with the attention-sink problem.
- **Positions.** SmolLM3's [blog](https://huggingface.co/blog/smollm3) removes rotary embeddings from every fourth layer, following a paper it cites; the config lists the NoPE layers as indices 3, 7, 11 and so on. [The NoPE paper](https://arxiv.org/abs/2305.19466) found that transformers without positional encoding generalised to longer lengths better than other explicit schemes on its reasoning and maths tasks. GLM-4.5 rotates only half of each head (`partial_rotary_factor` 0.5); Qwen3-Next a quarter (0.25). DeepSeek-V3, Kimi K2 and gpt-oss configs extend context with YaRN scaling.

## Reading across the table

These are the author's observations from the numbers above, not claims from the sources.

- The cache budget is set by three numbers: layers, KV width, and how many layers look at the whole context. Models that care about long context cut a different one of those each time: Llama and Qwen3 reduce KV heads, Gemma 3 and gpt-oss shrink the window on most layers, DeepSeek compresses the vector, Qwen3-Next replaces most layers.
- The two DeepSeek-style models differ from each other by knobs, not structure: same layers, same hidden size, same MLA rank, different expert count and head count.
- The dense models spend their parameters on every token; the sparse ones spend 3% to 11% of theirs (computed per-token / total: DeepSeek-V3 5.6%, Kimi K2 3.2%, GLM-4.5 9.5%, gpt-oss-120b 4.9%, Qwen3-30B-A3B 11.0%, Qwen3-Next 4.9%). Memory still holds all of them.

## Limits

- Config files and reports describe architecture, not quality. Nothing here says which design gives better answers.
- The parameter formulas omit biases, norms, multi-token-prediction modules and vision encoders, and count both embedding tables; they match published totals to within a few percent, which is enough to compare designs but not to certify a count.
- KV figures assume a 16-bit cache and one sequence, with no paging overhead or cache quantisation. The Qwen3-Next linear-layer state is an estimate for 16-bit storage.
- Gemma 3's 5:1 pattern is taken from its report and the layer arithmetic (62 layers give 10 global layers) is the author's. The config file for the text model does not list it.
- Config files were read from ModelScope mirrors of the official repositories (for example `LLM-Research/gemma-3-27b-it` and `openai-mirror/gpt-oss-120b`), not from the original Hugging Face repos, and were not diffed against them.
- Ten models are a sample chosen for variety; important families (Mistral, Phi, Llama 4, Gemma 4 and others) are not covered, and architectures change quickly.
