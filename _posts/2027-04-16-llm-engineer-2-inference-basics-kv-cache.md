---
title: "The LLM Engineer Path, Part 2: Inference Basics, Tokens, Context and the KV Cache"
date: 2027-04-16 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, inference, transformer, KV cache, tokens, context window, latency, python]
description: "Part 2 of a 14-part series for developers moving into LLM engineering. How text generation runs, why it has a prefill phase and a decode phase, what the KV cache is and why it exists. A 4-layer Transformer is timed with and without a cache, and the memory formula is checked against numbers from the vLLM paper."
---

Every LLM application sits on top of one loop: read tokens, produce one more token, repeat. This post shows that loop, the optimisation that makes it affordable (the KV cache), and the memory arithmetic that decides how many users one GPU can serve. You can run all the code on a CPU.

## Tokens and the context window

A model does not read characters or words. It reads **tokens**: integer IDs for pieces of text. Common splitting methods build these pieces from frequent character sequences, as in the paper [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). Three consequences matter for an engineer:

- Prices, limits and speed are all counted in tokens, not words or characters.
- The **context window** is the maximum number of tokens (prompt plus output) a model can attend to at once. It is set by how the model was trained and by memory.
- The same text can be a different number of tokens in a different model, because each model has its own vocabulary. Always count with the tokenizer of the model you call.

## Two phases

The [vLLM paper](https://arxiv.org/abs/2309.06180) splits generation into two phases. The **prompt phase** (also called prefill) takes the whole user prompt as input and computes the probability of the first new token. Because the prompt tokens are all known, it "can be parallelized using matrix-matrix multiplication operations". The **autoregressive generation phase** (decode) produces the rest one token at a time; it "often uses matrix-vector multiplication, which is less efficient", so it "becomes memory-bound, being responsible for most portion of the latency of a single request".

That gives two numbers to watch in production: time to first token (mostly prefill) and time per output token (decode).

## The KV cache

Inside each attention layer, every token produces a **key** vector and a **value** vector, and each new token compares its query against the keys of all earlier tokens. The keys and values of earlier tokens never change as the sequence grows. So decode can store them and compute only the new token's: the paper notes that "the key and value vectors at positions 1 to n + t - 1 are cached".

Here is the cache in a 4-layer Transformer with random weights (it is timed, not read, so training is irrelevant). The full script has about 50 lines; the key lines are these:

```python
# inside the attention layer: keep the keys (k) and values (v) of every earlier token
if cache is not None and cache[0] is not None:
    k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)   # old + new
if cache is not None:
    cache[0], cache[1] = k, v                                         # remember them for the next step
a = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))        # new tokens attend to everything before
```

and the two ways to generate:

```python
def gen_no_cache(prompt, n):
    ids = prompt
    for _ in range(n): ids = torch.cat([ids, m(ids).argmax(-1, keepdim=True)], 1)   # re-reads everything each step
    return ids
def gen_cache(prompt, n):
    caches = [[None, None] for _ in range(L)]; ids = prompt
    nxt = m(prompt, 0, caches).argmax(-1, keepdim=True)                           # prefill: whole prompt at once
    out = [nxt]
    for t in range(1, n):
        nxt = m(nxt, prompt.shape[1] + t - 1, caches).argmax(-1, keepdim=True)    # decode: one token at a time
        out.append(nxt)
    return torch.cat([prompt] + out, 1), caches
```

Real output on a 32-core CPU with no GPU (one run; timings vary by machine):

```
same tokens with and without cache: True
n_new  no_cache_s  cache_s
   32       0.08     0.03
   64       0.27     0.12
  128       0.44     0.13
  256       1.16     0.27
  512       3.81     0.57
prompt    64 tokens: prefill     3.4 ms | per new token   0.9 ms
prompt   256 tokens: prefill     6.1 ms | per new token   1.0 ms
prompt  1024 tokens: prefill    31.7 ms | per new token   1.8 ms
cache bytes: 843776  formula 2*D*L*4 bytes*tokens: 843776
```

![Left: seconds to generate 32 to 512 new tokens without the cache (rising steeply to 3.8 seconds) and with it (0.57 seconds). Right: for a 13-billion-parameter model on a 40 GB GPU, 33 requests of 512 tokens fit in the leftover memory, 16 of 1,024 tokens and 8 of 2,048]({{ site.baseurl }}/assets/pic/llmeng-kv-cache.png){: width="900" }
*Left: measured above. Right: the memory arithmetic in the next section.*

Read it:

- The outputs are identical; the cache changes cost, not results.
- Without the cache, every step re-processes the whole sequence, so total time grows faster than linearly. With it, each step handles one token. At 512 new tokens the cache is 6.7 times faster here.
- Prefill handles 1,024 tokens in 32 ms, yet decode costs about 1 to 2 ms for each single token: many tokens in parallel are cheap per token, one token at a time is not. Decode per-token time also rises with context (0.9 ms to 1.8 ms), because each step reads a longer cache.
- The last line checks the memory formula against the real tensors: cache bytes equal `2 x width x layers x bytes per number x tokens`.

## The maths tool: KV cache memory

The vLLM paper gives the same formula for a real model. For the 13B-parameter OPT model, "the KV cache of a single token demands 800 KB of space, calculated as 2 (key and value vectors) x 5120 (hidden state size) x 40 (number of layers) x 2 (bytes per FP16)". With a maximum length of 2,048 tokens, "the memory required to store the KV cache of one request can be as much as 1.6 GB". On a 40 GB A100 the paper's Figure 1 shows about 65% of memory (26 GB) taken by the weights and over 30% used for KV cache.

The right panel is this post's own arithmetic from those numbers: 40 GB minus 26 GB leaves 14 GB, and 14 GB divided by 819,200 bytes per token gives 33 requests of 512 tokens, 16 of 1,024 and 8 of 2,048. Real servers lose some memory to other overheads, so treat these as upper bounds. The point stands: doubling context length halves the number of simultaneous users on the same GPU, which is why long-context requests are priced and limited differently.

## The failure to reproduce: wrong position, silent corruption

A cache has one easy-to-make bug. In `gen_cache`, each decode step must tell the model which position the new token occupies (`prompt.shape[1] + t - 1`). Pass `0` instead, as if every new token were first, and nothing crashes:

```
identical: False | first differing new token at step 1
tokens that differ: 39 of 40
```

No error, no warning, 39 of 40 tokens wrong. The only defence is a test: compare cached and uncached output on a fixed prompt, as the first line of the real run does. Add that test to any code that touches a cache.

## Try it

1. Change `L` to 8 and `D` to 512. Does the cache speed-up grow or shrink? Why would a bigger model shift the balance?
2. Extend the timing to prompts of 2,048 and 4,096 tokens. At what point does prefill stop being "almost free"?
3. Work out the KV cache per token for a model you plan to use: `2 x layers x KV-vector width x bytes`. Use the model's published configuration. Some models share keys and values across several attention heads, which makes the vector narrower than the hidden size; check the model card before assuming the formula above applies unchanged.

## Limits

- The timings are from a toy CPU model. Real GPU serving is dominated by memory bandwidth, and absolute numbers differ.
- The memory figures are from one model family (OPT) in one paper; later models often shrink the cache with shared keys and values.
- Tokenizer details are covered only as far as an engineer needs: count tokens with the model's own tokenizer.
