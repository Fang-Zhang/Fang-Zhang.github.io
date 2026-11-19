---
title: "Companion AI (7/8): Latency and Voice, Where the Time Goes"
date: 2027-03-20 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, companion-ai, latency, voice, kv-cache, series]
description: "People answer each other within about 200 ms. Measured on a CPU, a voice turn spends 0.66 s in speech recognition, 26 s re-reading a 1000-token history, and 213 ms on each reply token. Caching the history cuts the middle term 32 times."
---

A voice companion has a clock that a text companion does not: the pause between one person finishing and the other starting. This post gives the target from the literature, measures the three terms of a voice turn on a CPU (speech recognition, prompt processing, token generation), and shows which of them grows with the history from Part 3 and which one a cache removes.

## The target and the known designs

- [Stivers et al.](https://www.pnas.org/doi/10.1073/pnas.0903616106) compared response timing across ten languages and found a similar distribution in all of them: a unimodal peak of responses within 200 ms of the end of the question, with cultural variation around it.
- [Moshi](https://arxiv.org/abs/2410.00037) describes the standard pipeline (voice activity detection, speech recognition, a text dialogue model, text-to-speech) as having a latency of several seconds between turns, and loses non-linguistic information such as emotion on the way. It replaces the pipeline with a speech-to-speech model that handles the user's and its own audio in parallel streams, with a theoretical latency of 160 ms and 200 ms in practice, as reported by its authors.
- [Whisper](https://arxiv.org/abs/2212.04356), trained on 680,000 hours of multilingual supervision, is the open speech recogniser measured below.
- Two server-side techniques target the LLM term: [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) manages the KV cache to waste almost none of its memory, which raises throughput 2 to 4 times at similar latency in the authors' evaluation, and [speculative decoding](https://arxiv.org/abs/2211.17192) computes several tokens in parallel with a small draft model and accelerates T5-XXL 2 to 3 times with identical outputs.

## The budget

The time from the end of the user's speech to the first word of the reply is a sum:

```
t_first_word = t_asr + t_prefill(new tokens) + n_first * t_decode + t_tts_first_chunk
```

`t_prefill` is the time to process the prompt, so it depends on how many tokens have to be read. `t_decode` is per generated token. `n_first` is how many tokens must exist before speech can start; streaming the reply into text-to-speech sentence by sentence keeps it small.

## Measurements

Hardware: this machine's CPU, Qwen2.5-1.5B-Instruct in bfloat16, PyTorch, 16 threads, greedy decoding, no GPU. Best of two runs for prefill.

| Prompt tokens | prefill (s) |
|---|---|
| 100 | 2.5 |
| 500 | 12.7 |
| 1000 | 25.8 |
| 2000 | 51.4 |
| 4000 | 114.8 |

- Prefill is roughly linear in prompt length at about 26 ms per token (a little worse than linear at 4000 tokens). At Part 3's scale, a year of history at Part 1's rate is about 548K tokens, which this CPU would take about 4 hours to read once. That is a linear extrapolation and not a measurement.
- Decode is 213 ms per token (4.7 tokens per second) with a 500-token context.
- Reading a 1000-token history from scratch took 26.0 s. Keeping the key-value cache of that history and feeding only 30 new tokens took 0.81 s, 32 times faster. This is the cache that PagedAttention-style servers manage.
- One Whisper-small encoder pass over a 30-second window takes 0.66 s on this CPU. Whisper pads every input to 30 seconds, so the encoder time does not depend on what was said; the decoder time was not measured, and neither was text-to-speech, because no local synthesiser was available to measure.

![Left: prefill time against prompt length, with the cached case marked. Right: stacked time budget for a voice turn with and without a cached history]({{ site.baseurl }}/assets/pic/companion-7-latency.png){: width="900" }
*Left: time to first token against prompt size, with the 0.2 s turn gap marked. Right: one voice turn from end of speech to the first 10 reply tokens, text-to-speech excluded.*

The right panel assumes the reply may start after 10 tokens (a short first clause). Even with the cache, this CPU needs 3.6 s: 0.66 s recognition, 0.81 s prefill and 2.13 s of decoding. Without the cache and with a 1000-token history it needs 28.8 s.

A precision choice moved the numbers more than expected. For the 0.5B model, with a 500-token prompt, fp32 prefill took 1.57 s and bfloat16 took 4.21 s, while decode went the other way: 130 ms per token in fp32 and 70 ms in bfloat16. Prefill is limited by compute and decode by memory traffic, and this CPU's bfloat16 matrix multiplication is slow. The 1.5B model was measured in bfloat16 only: a 1.5B fp32 run did not finish in 16 minutes and was stopped.

## What this shows and does not show

- Three terms behave differently. Recognition is fixed per utterance, prefill scales with how much history is re-read, and decode scales with reply length. Only prefill is affected by Part 3's memory design.
- The memory choice from Part 3 and the latency choice are the same choice: a history that must be re-read every turn makes each turn slower in proportion to its length, and a retrieved 200-token context (Part 3 measured about 200 tokens and 5.6 to 5.9 s per answer in the same setup) or a cached prefix does not.
- No setup here reaches 200 ms. On this CPU the decode term alone is above it for a single token. Hardware with a GPU and a streaming speech model changes the absolute numbers by orders of magnitude, and the Moshi authors' 200 ms is the only figure here for a system that meets the target, and it is theirs, not measured here.

## Limits

- One CPU, one model size per precision, one prompt template; absolute times do not transfer to other hardware. The ratios (32 times from caching, the linear growth) are the part to carry.
- The recognition number is the encoder only, on a padded low-level-noise input; real speech adds decoding time. Text-to-speech and voice activity detection are not measured.
- The 10-token first clause is an assumption. The first-word latency in a streaming system also depends on the text-to-speech chunk size.
- The 4-hour figure for a year of history is arithmetic on a measured slope, not a run.
- Speculative decoding and PagedAttention are cited for their authors' results and were not run here.

Next, in Part 8 of this series: embodied companions.
