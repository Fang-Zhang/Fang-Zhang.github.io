---
title: "The Evolution of LLM Architecture: Every Stage Solved One Problem and Created the Next"
date: 2026-09-23 06:30:00 +1200
categories: [Tech, AI/ML]
tags: [llm, transformers, architecture, deep-learning]
description: "How LLM architecture evolved from RNNs to Transformers to Mixture of Experts, told as a chain of problems and fixes, with one example sentence carried through every stage."
---

Every major shift in language model architecture happened for the same reason: the previous design hit a wall, and the next one existed only to get past it. Nobody set out to build a Transformer. Someone got tired of a specific, painful limitation of what came before it. Understanding LLMs this way — as a chain of fixes, each one trading an old problem for a new one — makes the whole field much less mysterious.

To make each stage concrete, we'll run the same sentence through every architecture:

> "The cat sat on the mat, and it looked tired."

The question every stage has to answer, one way or another: **what does "it" refer to?**

## Stage 1: RNNs — reading one word at a time

An RNN reads this sentence left to right, one word at a time, updating a single hidden state as it goes: "The" → "cat" → "sat" → "on" → "the" → "mat" → "and" → "it"...

By the time it reaches "it," the word "cat" is 7 tokens back, compressed into the same fixed-size hidden state as everything else that came before. If the sentence were longer, "cat" would get diluted further — squeezed out to make room for newer information. The RNN often loses the thread, or resolves "it" to whichever noun happens to still be strongest in memory, not necessarily the correct one.

This was the first design that could handle sequences at all — a real advance over anything before it. But it had two structural flaws baked in from the start: it read one token at a time, so it couldn't be parallelized (training was slow by construction), and its memory decayed over distance.

## Stage 2: Attention — fixing the forgetting

Attention was bolted onto RNNs first, not invented for Transformers. Instead of forcing everything through one compressed hidden state, the decoder can now look back directly at every earlier word when it reaches "it," and ask: which of these matters most right now?

It checks "cat," "mat," "sat," "on" — and assigns "cat" the highest relevance score. "It" pulls information straight from "cat," undiluted by the seven words in between. No more compression bottleneck.

This solved forgetting. Translation quality jumped. But the model was still reading "The," then "cat," then "sat"... one word at a time, still riding on top of a slow, sequential RNN underneath the attention step.

## Stage 3: The Transformer — removing the bottleneck entirely

In 2017, "Attention Is All You Need" asked the obvious next question: if attention is doing the real work, why keep the RNN underneath it at all?

So they removed it. Now every word in the sentence — "The," "cat," "sat," "on," "the," "mat," "and," "it," "looked," "tired" — is processed at the same time, in parallel. "It" computes its relevance score against all nine other words simultaneously via Query/Key/Value, not by waiting through a left-to-right pass. It still lands on "cat" as most relevant — but the computation that gets it there took one parallel step instead of a long sequential walk.

This mapped perfectly onto GPUs, which are built to do thousands of small parallel calculations at once. Training that took weeks on RNNs took days.

Out of this came the fork that still defines the field: encoder-only models (BERT — bidirectional, good at understanding "it" refers to "cat" for classification purposes) versus decoder-only models (GPT-line — causal attention, generating "tired" as the next word having already resolved what "it" means). Decoder-only won for generation and is what nearly every chat model uses today.

The Transformer solved speed. But it created a new question nobody had a good answer to yet: how big should you make one of these?

## Stage 4: Scaling — making it smart, making it expensive

The architecture barely changed from GPT-1 through GPT-4. What changed was parameters, data, and compute. A small Transformer might correctly resolve "it" → "cat" in our simple sentence but fail on a harder one: "The trophy didn't fit in the suitcase because it was too big." (Is "it" the trophy or the suitcase? Depends on world knowledge about size, not just grammar.) A sufficiently large model, trained on enough text, starts resolving these ambiguous cases correctly too — not because anyone taught it the specific rule, but because [the pattern emerged from scale](/posts/i-am-a-language-model-too/).

Scaling solved "how do we make this smarter." It created a new problem: models this big are extremely expensive to run.

## Stage 5: Mixture of Experts — cutting the cost of thinking

MoE splits the model into many smaller "expert" sub-networks and, for each token, only activates a handful of them. When the model processes "it" in our sentence, maybe two experts specializing in pronoun resolution and grammar activate — out of dozens available — while experts trained on, say, code syntax or arithmetic stay dormant for this token. A model might hold 47 billion total parameters but only fire 13 billion of them per token. You get the knowledge capacity of a huge model with the runtime cost of a much smaller one.

MoE solved the cost of the "thinking" step. But the attention mechanism has its own, separate cost — one that gets worse as context windows grow.

## Stage 6: Where it's going now

Now imagine our sentence isn't 10 words but 10,000 — a whole chapter, and "it" on the last page needs to resolve back to a character introduced on page one. The model has to store the keys and values of every earlier token so it doesn't recompute them — this is the KV cache — and for inputs that long, that storage balloons. Grouped Query Attention (used in LLaMA 3) has multiple attention heads share keys/values to shrink the cache; Multi-Head Latent Attention (used in DeepSeek) compresses keys/values into a smaller form before storing them. Both are cache-compression tricks, not new attention paradigms.

Two other directions worth naming: the same Transformer block is now reading images, audio, and video by turning patches/frames into tokens — attention doesn't care what kind of token it's looking at, resolving "it" the same way whether the referent is a word in text or an object in an image. And there's a second scaling axis now besides parameter count: reasoning models that generate longer chains of intermediate thinking steps at inference time — a model might explicitly reason "it likely refers to cat, since mat doesn't have a state of being tired" before answering — rather than just growing bigger.

## A quick reference

| Architecture | What it introduced | Problem it solved | Problem it left behind |
|---|---|---|---|
| RNN / LSTM | Recurrent hidden state, reads tokens sequentially | Could finally handle sequences at all | Slow (no parallelism), forgets long-range context |
| Attention (on RNN) | Decoder looks back at all encoder states directly | Fixed the forgetting problem | Still riding on top of a slow, sequential RNN |
| Transformer (encoder-decoder) | Self-attention only, no recurrence, fully parallel | Removed the sequential bottleneck entirely — matched GPUs | No inherent sense of word order; didn't yet answer "how big to build it" |
| Encoder-only (BERT-line) | Bidirectional self-attention | Strong at understanding/classification tasks | Not built for generating text |
| Decoder-only (GPT-line) | Causal/masked self-attention, autoregressive | Became the standard for text generation | Scaling it well was still unsolved |
| Scaled decoder-only (GPT-2→4) | Same architecture, much more data/parameters/compute | Made models dramatically smarter (emergent abilities) | Extremely expensive to run and serve |
| Mixture of Experts (MoE) | Many expert sub-networks, only a few active per token | Cut inference cost while keeping model capacity | Attention itself still expensive at long context |
| GQA / Multi-Head Latent Attention | Shared or compressed keys/values in the KV cache | Shrank memory cost of long-context attention | Doesn't address other long-context tradeoffs (e.g. multimodal, reasoning depth) |
| Multimodal Transformers | Images/audio/video tokenized like text | Same architecture now handles multiple input types | Doesn't make the model reason better, just perceive more |
| Reasoning / inference-time scaling models | Longer chains of intermediate thinking steps at inference | Smarter answers without growing parameter count | Slower and costlier per response (new cost tradeoff) |

## The pattern, in one line

RNN forgets and is slow → Attention fixes forgetting → Transformer removes the slow part → Scaling makes it smart → MoE cuts the cost → long-context/multimodal/reasoning push further. Every stage kept what worked in the one before it and fixed whatever didn't. The next time a new model architecture shows up in the news, the right question isn't "is this better" — it's "which problem in this chain is it actually solving."
