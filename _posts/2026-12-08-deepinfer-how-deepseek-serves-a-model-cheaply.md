---
title: "DeepInfer: How DeepSeek Serves a Model Cheaply"
date: 2026-12-08 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deepseek, inference, kv-cache, speculative-decoding, mixture-of-experts]
description: "What DeepSeek's open-sourced inference framework DeepInfer does between a chat message and the first word of the reply: caching, prefill/decode separation, load balancing and speculative decoding, with every figure taken from DeepSeek's own documents."
---

Open-weight models give you the parameters and enough code to run them. They do not give you the system that serves millions of requests at low cost, and that system is where much of the cost difference between two deployments of the same weights comes from. On 2026-10-20 DeepSeek released part of that system: [DeepInfer](https://github.com/deepseek-ai/DeepInfer), the inference framework behind its V4.1 Flash service, and [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe), the library that converts between API formats and token sequences. The release is also covered by [MarkTechPost](https://www.marktechpost.com/2026/10/20/deepseek-deepinfer-open-source-inference-framework).

This post follows one request through the system. All figures come from DeepSeek's own documents: the [README](https://github.com/deepseek-ai/DeepInfer), the [deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md), and the earlier [V3/R1 inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md). Where a number comes from a third party or is self-reported, the text says so.

## What was released, and what was not

According to the Q&A, one cluster currently serves **155 trillion tokens per day**, with overall revenue of **US$38 per GPU-hour at valley pricing**. That is revenue at off-peak prices, not profit, and no cost figure is given. For scale, 155 trillion per day is about 1.8 billion tokens per second.

The earlier overview reported a 545% theoretical cost-profit margin for V3/R1, computed as if every token had been billed at R1 list prices.

DeepSeek had said in an [April 2025 post](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md) that it would not open-source its full inference engine, because the code had diverged from an early vLLM fork and was "tightly coupled with our internal infrastructure, including cluster management tools". The October release keeps to that line. The README lists what is **not** included: the KV cache system, the prefill/decode disaggregated deployment, and load balancing across data-parallel replicas. What is included is the engine core for **offline inference**, for one model (DeepSeek-V4.1-Flash), on NVIDIA SM100 GPUs or Huawei Ascend 950 NPUs.

![One request through DeepSeek's serving system](/assets/pic/deepinfer-journey.png){: width="700" }
_Figure 1. The path of one request, after DeepSeek's Q&A. My own drawing._

## Stop 1: encoding

A model reads token IDs, not text. The conversation structure is marked by special tokens: where the user speaks, where reasoning ends, where a tool call begins. That invites an attack: type the marker strings into a chat box and impersonate the system.

The Q&A describes how deepseek-recipe blocks it. When the prompt is built, the special tokens the program inserts are recorded with their positions and mapped directly to token IDs. Strings such as `<｜User｜>` or `<｜image｜>` that appear in user input, tool results or tool-call arguments are encoded as ordinary text. In the other direction, the parser identifies the end-of-reasoning marker and the tool-call token by ID, so the same characters spelled out by the model as ordinary tokens stay literal content and cannot trigger a tool call.

Because the special-token positions are known, the ordinary-text segments between them can be encoded independently and in parallel, then joined in order. The Q&A says this reduces tokenization time for very long inputs. For streaming, a single UTF-8 character, a tag or a tool argument may be split across chunks, so the parser buffers incomplete characters and keeps a state machine across the reasoning, body and tool-argument phases. Recipe converts the Chat Completions, Responses and Messages APIs to one conversation format, is implemented in Rust with Python bindings, and is fully aligned with the official DeepSeek API, so third-party deployments get the same behavior.

## Stop 2: prefill and decode on different machines

Generation has two phases. **Prefill** reads the whole input (system prompt, history, files, question) in parallel and is bound by compute. **Decode** produces one token at a time, rereading the accumulated state at each step, and is bound largely by memory traffic. DeepSeek deploys them separately. The Q&A gives the layout: ordinary 8-GPU machines for prefill (expert parallelism of 2 to 8, using the smallest that GPU memory allows), 64 to 128-GPU supernodes for decode (the largest the available GPUs allow), and a separate vision-encoder stage for images. Prefill and decode GPUs run at roughly 1:1, so the smallest efficient deployment is 64 GPUs for prefill plus 64 for decode.

The point of the split, in the Q&A's words, is that resources can be allocated per stage, so time to first token (TTFT) and time between tokens (TBT) can be controlled and optimized independently.

## Stop 3: the KV cache

Every new message in a conversation repeats the entire history. A coding agent repeats history, system prompt, tool descriptions and previously read code on every tool call. The **KV cache** stores the attention keys and values computed when text was first read ([cache](https://en.wikipedia.org/wiki/Cache_%28computing%29) is the general idea), so a request that begins with an already-seen prefix skips recomputing it. Only a shared opening can be reused: text that repeats in the middle of a different prompt cannot.

![Prefix reuse in the KV cache](/assets/pic/deepinfer-kvcache.png){: width="700" }
_Figure 2. Turn 51 reuses what turn 50 already read. An edit near the start breaks the shared opening. My own drawing._

DeepSeek has published a hit rate for its older V3/R1 service: of 608 billion input tokens in one 24-hour window in February 2025, 342 billion (56.3%) hit the on-disk KV cache. I could not find a published hit rate for V4.1 Flash. The size of the effect follows from arithmetic: if a fraction h of input tokens is served from cache, prefill work falls by a factor of 1 / (1 - h), so 56.3% means about 2.3 times less and 90% would mean 10 times less.

The Q&A states a design target of keeping the KV cache for more than 72 hours.

## Stop 4: where the cache lives

The block KV cache is stored on a dedicated storage cluster running DeepSeek's open-source distributed file system [3FS](https://github.com/deepseek-ai/3FS), not in the compute machines' memory. The Q&A gives three reasons: any prefill machine can hit any prefix, which avoids a conflict between prefix affinity and load-balancing routing; storage nodes can scale independently of compute; and compute nodes fail more than an order of magnitude more often than storage nodes, so the cache survives their failures.

Other details from the Q&A:

- About 100 PiB of SSD, with a time-to-live of **3 days for agent traffic and 7 days for chatbot traffic**. Chat traffic has a heavier long tail in which a small amount of cache is read very many times, and a 3-day TTL costs it 0.5% to 1% of hit rate.
- Data is stored with 10+2 erasure coding, so a group tolerates two simultaneous node failures. A read goes to all 11 nodes in the group and any 10 answers are enough.
- Reclamation is a least-recently-used garbage collector on each storage node, kept below a disk-usage watermark.
- Matching a prefix is a metadata-only check at the entrance, and it has to finish before routing, so it cannot be hidden. The data itself is fetched asynchronously into host memory and then swapped into GPU memory layer by layer, overlapped with the forward pass.

This storage system is the part DeepSeek did not release. The Q&A refers readers to DeepSeek's own Zhihu blog post for its design.

## Stop 5: sliding-window layers and bounded replay

Some of V4.1's layers use sliding-window attention (SWA) and look only at the most recent 128 tokens. The Q&A says their cache is **not** written to 3FS. It sits in the local host memory of the machine that produced it, with an in-memory directory recording where, and a machine that lacks it pulls it from the remote one over a 400G network on a hit.

What if it is missing? The Q&A's answer is **SWA bounded replay**: recompute only the last 128 tokens of the cached prefix and truncate attention to the replayed segment, which gives an approximate SWA cache. The cost is constant, independent of context length. In the decoder layers, rebuilding the cache exactly would mean pushing the last 2,560 tokens through 20 layers on every prefill. Replaying 128 tokens instead is described as nearly halving total prefill computation.

The README's default V4.1 setting keeps a 140-token suffix, which the README says must cover the SWA window plus the maximum backtracking length.

## Stop 6: keeping GPUs neither blocked nor idle

V4.1 is a [mixture-of-experts](https://en.wikipedia.org/wiki/Mixture_of_experts) model, so each token is routed to a few of many expert sub-networks. Spreading experts across 64 to 128 decode GPUs means each GPU holds few experts, reads less weight data per step and batches more requests. The risk is imbalance: all GPUs advance in lockstep, so the slowest sets the pace. DeepSeek's Q&A lists four load balancers (prefill, decode, vision encoder, and across experts):

- **Expert-parallel load balancer.** Each GPU holds a few redundant experts that replicate hot ones. Load per expert is collected over a window of recent steps and assignments are reshuffled periodically. The Q&A reports measured slowest-rank to mean ratios of 1.074 and 1.045 (prefill) and 1.090 (decode).
- **Prefill balancer.** Requests are bucketed by context length into four tiers (under 16K, 16-64K, 64-256K, 256K and above) with compute quotas per tier, and chunk sizes are solved from a time budget with a fitted cost model.
- **Decode balancer.** It balances request count (which sets batch size) and total KV length (which sets attention cost) at once.
- Overall, the online system reaches roughly **80%** of a perfectly balanced offline benchmark, according to the Q&A.

Scaling follows load: prefill machine counts track non-cached tokens, decode tracks running requests, and scale-in is forbidden when quota utilization is above 0.8. Restarts take minutes, so fault tolerance comes from redundant quota plus request-level rerouting, not from fast restarts.

The expert-parallel load balancing is in the released code; the cross-machine balancers are not (README).

## Stop 7: several tokens per step

Decoding normally emits one token per forward pass. In **speculative decoding** ([Leviathan et al., 2022](https://arxiv.org/abs/2211.17192)) a cheap draft model proposes several tokens, the main model checks them in a single pass, the correct prefix is kept and the first wrong token is replaced, so output quality is unchanged.

![Speculative decoding with DSpark](/assets/pic/deepinfer-speculative.png){: width="700" }
_Figure 3. The draft proposes five tokens; the main model verifies only a prefix and corrects the first mistake. The acceptance probabilities in the figure are illustrative, not DeepSeek's numbers. My own drawing._

On a busy GPU, every extra draft token checked costs compute, and later tokens are more likely wrong. The Q&A describes DeepSeek's **DSpark**: a confidence head predicts the acceptance probability of each draft token (the draft model writes 5 at a time in production), and the scheduler sorts them by expected acceptance and, from a profiled throughput curve of the hardware, picks the prefix that maximizes effective throughput. Reported results:

- Verified draft tokens per step fall about 30% (5.00 down to 3.2 to 4.4) while average accepted length barely changes (3.29 versus 3.20).
- That works out to **+4% to +7%** throughput at the production operating point of about 60 concurrent sequences per rank.
- The Q&A's own table shows the gain depends on load: -1.5% at 16 sequences per rank, +11.0% at 96.

**Asynchronous scheduling.** In the plain loop, step N runs schedule, forward, sample and post-process, and the GPU idles during the CPU-only steps. DeepInfer overlaps them: while the GPU runs forward and sample for step N, the CPU runs scheduling for step N+1 and post-processing for step N-1. The Q&A cites SGLang and NanoFlow as earlier implementations.

![Asynchronous scheduling](/assets/pic/deepinfer-async.png){: width="700" }
_Figure 4. Overlapping CPU planning with GPU work removes the idle gaps. Step lengths are illustrative. My own drawing._

For measured speed, [MarkTechPost's reading](https://www.marktechpost.com/2026/10/20/deepseek-deepinfer-open-source-inference-framework) of the README's charts (self-reported by DeepSeek, 131,072 context) gives decode above 25,000 output tokens per second per GPU at 256 requests per rank (about 100 tokens per second per user), and over 300 per user at 8 requests, on 64 GPUs with 3 draft tokens and 75% acceptance. For prefill it gives about 60,000 non-cached tokens per second per GPU at 4,096 tokens per rank, on 4 GPUs.

## Limits of the release

- **Offline only.** Only the engine core is open. The KV cache system, the prefill/decode disaggregation and the replica load balancers are not (README).
- **Hardware.** NVIDIA SM100 GPUs or Huawei Ascend 950 NPUs, with CUDA 13 or CANN 9.2, GCC 13 or later and Python 3.12. One supported model, DeepSeek-V4.1-Flash. At the Q&A's smallest efficient deployment, that is 128 GPUs.
- **Lineage.** According to the README, DeepInfer was originally built primarily on [vLLM](https://github.com/vllm-project/vllm), with inspiration from [SGLang](https://github.com/sgl-project/sglang) and TensorRT-LLM. Its kernels include DeepSeek's [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) and custom [TileLang](https://github.com/tile-ai/tilelang) kernels.
- **Fidelity.** The Q&A says the open kernel libraries are broadly the same as the internal ones (a missing hand-tuned optimization makes one decoding kernel 5% to 10% slower) and end-to-end performance is at most 5% apart.

## What follows from it

Two observations of my own, not DeepSeek's.

1. The released parts are the ones that do not depend on a particular data center: kernels, the engine, the scheduling logic. The withheld parts (storage, routing, replica balancing) are the ones tied to DeepSeek's infrastructure, which is the reason DeepSeek gave in 2025. So the release shows how the serving cost is cut without handing over a deployable service.
2. Most of the techniques above are load-dependent. DSpark loses at low concurrency and wins at high; wide expert parallelism needs at least 64 GPUs per stage to pay off; a cache pays off in proportion to how long conversations and agent loops stay in a repeating prefix. Quoting one throughput or hit-rate number says little without the operating point it was measured at.
