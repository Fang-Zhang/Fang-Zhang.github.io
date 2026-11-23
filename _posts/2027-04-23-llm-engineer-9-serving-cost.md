---
title: "The LLM Engineer Path, Part 9: Serving and Cost"
date: 2027-04-23 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, serving, batching, throughput, latency, cost, python]
description: "Part 9 of a 14-part series for developers moving into LLM engineering. Why batching decides what a token costs, how continuous batching differs from static batching, and a measured step-time curve plus a simulation showing mean latency of 4 s against 132 s at the same load."
---

Serving a model means running it for many users at once at an acceptable price. Three numbers matter: **latency** (how long one request takes), **throughput** (tokens produced per second across all requests) and **cost per token**, which is the price of the hardware per second divided by throughput. This post shows why throughput depends mostly on how requests are batched, and measures the effect.

## Why batching changes the price

Generating one token needs the model's weights to be read from memory and multiplied by a small amount of data. Part 2 showed that with a KV cache each step handles a single new token per request. With a single request the work per step is small compared with the weights that must be read, so extra requests add little to the step time, which is why decoding several requests together raises throughput. The measurement below shows the effect.

Measured on this machine's CPU (16 threads, a 101M-parameter random-weight Transformer with 8 layers, 128 tokens of context per request, one decoding step for a batch of requests):

```python
@torch.no_grad()
def step(x, K, V):                                   # one decoding step for a batch: x (b, d), caches (b, H, L, d/H)
    b = x.shape[0]
    for i, w in enumerate(W):
        q, k, v = (x @ w["qkv"]).view(b, 3, H, d // H).unbind(1)
        att = torch.softmax(torch.einsum("bhd,bhld->bhl", q, K[i]) / (d // H) ** .5, -1)
        x = x + torch.einsum("bhl,bhld->bhd", att, V[i]).reshape(b, d) @ w["o"]
        x = x + torch.relu(x @ w["u"]) @ w["dn"]
    return x
```

```
batch   1:   26.3 ms/step     38.1 tokens/s
batch   4:   31.6 ms/step    126.5 tokens/s
batch  16:   52.2 ms/step    306.3 tokens/s
batch  64:  111.6 ms/step    573.4 tokens/s
```

(Median of 80 steps per batch size. Batches 1 and 2 were noisy across runs, so read the trend: 64 times the work took about 4 times as long.) Cost per token is the machine's price per hour divided by tokens per hour. At a hypothetical 1 per hour (the unit does not matter), 1 million tokens cost 7.3 at batch 1 and 0.48 at batch 64: 15 times less, from scheduling alone. This measurement is a CPU; do not carry the numbers to a GPU.

## What limits the batch

The KV cache. Part 2 worked out that for a 13B model on a 40 GB GPU, per-request cache memory limits concurrent requests to about 8 to 33. The [vLLM paper](https://arxiv.org/abs/2309.06180) states the point directly: "High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time", but the cache "is huge and grows and shrinks dynamically", and when managed inefficiently "this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size". Its PagedAttention manages the cache in blocks, and the authors report 2 to 4 times the throughput of earlier systems at the same latency.

## Static against continuous batching

A text request ends when the model emits an end token, so requests in one batch finish at different times. With **static batching**, the server takes a group of requests, runs until the longest has finished, and only then accepts new ones. The [Orca paper](https://www.usenix.org/conference/osdi22/presentation/yu) describes the cost: "requests that have finished earlier than other requests in a batch cannot return to the client, while newly arrived requests have to wait until the current batch completely finishes". Its fix is **iteration-level scheduling**, often called continuous batching: the scheduler decides after every single decoding step which requests are in the batch, so a finished request leaves immediately and a waiting one takes its slot.

![Left: measured tokens per second against batch size on a CPU, rising from 38 at batch 1 to 573 at batch 64. Middle: the same 8 requests in 4 slots under static batching, with idle hatched slots finishing at step 14, and under continuous batching finishing at step 12. Right: simulated mean latency at 0.5, 1 and 2 requests per second: static 9, 26 and 132 seconds, continuous 2.9, 3.2 and 4.2 seconds]({{ site.baseurl }}/assets/pic/llmeng-serving.png){: width="900" }

The simulator below uses the measured step times above. Requests arrive at random, each wants a random number of output tokens (mean 91, between 10 and 400), and 16 slots stand for the KV-cache limit:

```python
def workload(seed, n=600, rate=2.0):
    r = random.Random(seed); t = 0; out = []
    for _ in range(n): t += r.expovariate(rate); out.append((t, int(min(400, max(10, r.lognormvariate(4.2, .8))))))   # (arrival time, output tokens)
    return out

def run(reqs, mode, B=16):                       # B = slots: the KV-cache memory limit on concurrent requests
    q = list(reqs); wait = []; act = []; now = 0.0; lat = []; tokens = 0
    while q or wait or act:
        while q and q[0][0] <= now: wait.append(q.pop(0))
        if mode == "static" and not act: act = [list(r) for r in wait[:B]]; wait = wait[B:]      # a new batch only when the old one is done
        if mode == "continuous":
            while wait and len(act) < B: act.append(list(wait.pop(0)))                           # refill free slots every step
        if not act: now = q[0][0]; continue
        now += tstep(len(act))                                                                    # measured step time for this batch size
        tokens += sum(r[1] > 0 for r in act)                                                      # only unfinished requests make useful tokens
        for r in act: r[1] -= 1
        if mode == "continuous": lat += [now - r[0] for r in act if r[1] <= 0]; act = [r for r in act if r[1] > 0]
        elif all(r[1] <= 0 for r in act): lat += [now - r[0] for r in act]; act = []              # static: nobody leaves until all are done
    return np.mean(lat), np.percentile(lat, 95), tokens / now
```

Real output (600 requests, mean of 3 seeds):

```
0.5 req/s  static      mean latency    9.2 s   p95   22.5 s   tokens/s   44.5
0.5 req/s  continuous  mean latency    2.9 s   p95    8.2 s   tokens/s   44.7
1.0 req/s  static      mean latency   25.9 s   p95   43.7 s   tokens/s   86.4
1.0 req/s  continuous  mean latency    3.2 s   p95    9.3 s   tokens/s   89.0
2.0 req/s  static      mean latency  132.4 s   p95  248.2 s   tokens/s   97.7
2.0 req/s  continuous  mean latency    4.2 s   p95   12.3 s   tokens/s  175.6
```

At light load both serve every request (tokens per second equals the arrival rate), but static batching already takes three times as long, because short requests wait for long ones. At 2 requests per second static batching saturates near 98 tokens per second, its queue grows, and latency explodes to 132 s, while continuous batching still keeps up (176 tokens per second, matching the 2 x 91 tokens arriving) with a 4 s average.

## The failure to reproduce

Benchmark the single-request case and the loaded case separately. Set the simulator's `rate` to 0.5 and the two schedulers differ by 3 times; set it to 2 and they differ by 30 times. A benchmark run with one request at a time would show neither effect, and would also say nothing about the 15-times cost difference above. Always report p95 next to the mean: users feel the slow tail, and in the table the tail is 1.7 to 2.9 times the mean.

## Try it

1. Run `step.py`-style timing with `L = 1024` instead of 128. Which batch sizes slow down more, and why? (Attention reads each request's own cache, which does not get shared.)
2. In the simulator, change `B` from 16 to 8 and 32. Where does continuous batching saturate in each case?
3. Add a time to first token metric: the time from arrival until a request's first decoded step. Which scheduler is better, and by how much, at 1 request per second?

## Limits

- The step times come from a CPU; on a GPU the curve has a different scale and a flatter start, and the real limit is memory capacity and bandwidth.
- The simulator ignores the prefill phase (processing the prompt), the cost of memory fragmentation, and network overhead. Prefill is compute-heavy and interacts with batching in ways production servers handle separately.
- The prices are not real: no vendor price was used, so the 15-times figure is a ratio of throughputs on this machine, not a statement about any provider's bill.
- Continuous batching is one of several techniques; quantisation, speculative decoding and prefix caching change the cost further and are not measured here.
