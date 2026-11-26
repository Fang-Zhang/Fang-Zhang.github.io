---
title: "Distributed LLMs, Measured (7/7): Decode Throughput Gained 8.6x From Batching at 128 Tokens of Context but 2x at 2,048, Paged Allocation Held 4x More Requests, and Disaggregation Removed the 200 ms Token Stalls"
date: 2027-07-12 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, inference, kv cache, paged attention, disaggregated serving, series]
description: "Part 7 of 7. A transformer decoded with a real KV cache: its memory and traffic against context and batch, paged block tables checked against a contiguous cache, an allocator simulation, and a two-process prefill/decode split with measured token-gap stalls."
---

## Inference is a different bottleneck

Training (Parts 1 to 6) is a throughput problem over a fixed batch. Serving is a latency problem over requests that arrive at random, differ in length and finish at different times. The state that makes generation affordable, the key-value (KV) cache, is also the state that decides how many requests fit and how long each decode step takes. This part measures that state directly.

## What the sources say

- **KV cache size.** [PagedAttention](https://arxiv.org/abs/2309.06180) gives the accounting for a 13B-parameter OPT model: 800 KB per token, "calculated as 2 (key and value vectors) x 5120 (hidden state size) x 40 (number of layers) x 2 (bytes per FP16)", so a request of up to 2,048 tokens needs as much as 1.6 GB. It states that serving is memory-bound, and that existing systems pre-allocate a contiguous chunk at the request's maximum length (for example 2,048 tokens); its profiling found "only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems". Its answer is to store the cache in fixed-size blocks addressed through a block table, like virtual memory pages, and it reports 2-4x higher throughput than FasterTransformer and Orca at the same latency.
- **Prefill and decode.** [DistServe](https://arxiv.org/abs/2401.09670) argues that colocating the two phases causes "strong prefill-decoding interferences" and couples their resource plans, and assigns them to different GPUs. It reports 7.4x more requests or 12.6x tighter latency targets than state-of-the-art systems, within the targets for more than 90 percent of requests. Its cost estimate for the hand-off: the KV cache of one 512-token request on OPT-66B is about 1.13 GB, so 10 requests per second need 11.3 GB per second, or 90 Gbps, and it relies on fast links to keep this negligible.

The 1.13 GB figure checks out with the same formula: 2 x 9,216 x 64 x 2 bytes x 512 tokens is 1,207,959,552 bytes, or 1.125 GiB, for OPT-66B's hidden size 9,216 and 64 layers.

## Experiments

The model is the 8-layer, d = 384 transformer used throughout the series (15,867,776 parameters with a 4,096-position table, 14,195,712 in the blocks), in fp32, one CPU thread per process. A decoder with a preallocated KV cache was written for it, so a KV cache costs 2 x 8 x 384 x 4 = 24,576 bytes per token.

**A. Cache against recompute, and the decode step.** Decoding with the cache is compared with recomputing the whole prefix; then one decode step is timed over batch sizes 1 to 64 and contexts 128 to 2,048, with FLOPs counted by `FlopCounterMode`.

**B. Paged cache.** Keys and values are stored in a pool of 16-token blocks scattered at random, with a block table per sequence, and decoding is compared with the contiguous cache for four sequences of different lengths. Then a pure-Python allocator simulation compares memory policies under one KV budget of 32,768 token slots for 1,500 synthetic requests (lognormal prompt and output lengths; mean 278 and 139 tokens, maximum model length 2,048), all waiting at the start: reserve the maximum length up front, reserve the exact final length up front (impossible in practice, since the output length is unknown), and paged allocation with block sizes 1 to 256 that grows on demand and, when the pool runs out, preempts the newest request and restarts it from scratch.

**C. Prefill and decode on separate workers.** 60 requests with 512-token prompts and 64 output tokens, Poisson arrivals at 3 and 5 requests per second, at most 16 sequences per decode batch, always two processes: either two colocated replicas (each handles every second request, runs one prefill whenever a prompt is waiting, then a decode step for its batch), or one prefill process and one decode process with the KV cache of each request sent over gloo after the first token.

Results:

- **Cache correctness.** Decoding 64 tokens with the cache matched full recomputation to 8.3e-07 in the logits.
- **Cache against recompute.** One new token at a context of 1,024 tokens cost 41.0 MFLOPs with the cache and 41.9 GFLOPs without (1,022x), taking 5.9 ms against 366 ms (62x). The FLOP ratio equals the context length almost exactly (32.9x at 32, 511.6x at 512). With the cache, FLOPs per step rose only 42 percent over the same range (28.8 to 41.0 MFLOPs), as the attention part grows linearly in the context.
- **Decode step time.** At a batch of 1 a step took 4.9 ms at 128 tokens and 7.9 ms at 2,048. Over all 12 combinations the step time fit a straight line in the KV bytes read per step, with one shared slope of 0.073 ms per MB (13.7 GB/s effective on one thread) and an intercept per batch size of 4.4, 7.7, 11.0 and 21.4 ms for batches 1, 4, 16 and 64; the largest deviation was 6 percent. The weights are about 57 MB per step, so KV traffic equals weight traffic at about 2,310 cached tokens in total (batch x context).
- **Throughput.** Growing the batch from 1 to 64 raised decode throughput from 203 to 1,749 tokens per second at 128 tokens of context (8.6x), from 192 to 797 at 512 (4.2x), and from 127 to 249 at 2,048 (2.0x). The KV cache of the 64 x 2,048 case was 3.2 GB.
- **Paged cache.** Block-table decoding matched the contiguous cache with a maximum logit difference of exactly 0 over 10 steps with lengths 70, 100, 33 and 129. Gathering the blocks into a contiguous tensor before attention made the step 5.1x slower (130 ms against 25 ms at batch 16 and context 512) in a Python loop and 6.3x slower vectorised.
- **Allocation under a fixed budget** (average concurrent requests; stored tokens as a share of allocated slots): reserve the maximum length, 15.5 and 0.20; exact up-front reservation, 50.4 and 0.74; paged with blocks of 256, 64, 16, 4 and 1 tokens, 51.5, 60.2, 63.2, 63.9 and 64.0, with stored shares of 0.76, 0.93, 0.98, 1.00 and 1.00. Paged with 16-token blocks held 4.1x more requests than reserving the maximum, at a cost of 514 preemptions with restart.
- **Prefill against decode.** Prefill cost 0.29 to 0.34 ms per token for 64 to 1,024 tokens (171 ms for 512), linear in the prompt length, while one decode step at batch 1 took 5 ms for one token.
- **Colocated against split, 5 requests per second.** Colocated replicas: first-token time p50 180 ms and p90 258 ms; gaps between consecutive tokens p50 16.6 ms, p99 196.5 ms, maximum 212 ms, and 4.9 percent of all gaps over 100 ms; per-request time per output token p50 24.8 ms and p90 34.1 ms. Split: first-token p50 331 ms and p90 687 ms; gaps p50 17.4 ms, p99 32.8 ms, maximum 51 ms, none over 100 ms; per-request time per output token p50 17.7 ms and p90 21.5 ms.
- **At 3 requests per second** the picture was the same: gaps over 100 ms in 1.6 percent of colocated gaps and none in the split; first-token p90 193 ms colocated and 544 ms split.
- **Hand-off.** The KV cache of one request is 12.58 MB; the time from send to arrival was 6 to 7 ms at the median over loopback, 4 percent of the 171 ms prefill.

![Three panels. Left: decode throughput in tokens per second against context length of 128, 512 and 2048 for batch sizes 1, 4, 16 and 64; at 128 tokens batching raises it from 203 to 1,749, at 2,048 tokens from 127 to 249. Middle: mean concurrent requests under the same 32,768-token KV budget: 16 for reserving 2,048 tokens per request, 50 for exact up-front reservation, and 51, 60, 63, 64 and 64 for paged allocation with blocks of 256, 64, 16, 4 and 1 tokens. Right: distribution of gaps between consecutive output tokens at 5 requests per second: colocated serving has about 5 percent of gaps near 190 to 210 ms, disaggregated serving never exceeds 51 ms]({{ site.baseurl }}/assets/pic/dist-7-inference.png){: width="900" }
_Figure: left, decode throughput against context for four batch sizes; middle, concurrent requests under one KV budget; right, token-gap distribution at 5 requests per second._

## Engineering reading

These points are this series' reading of the measurements and the sources.

1. **The cache converts quadratic recomputation into linear reads, and then becomes the thing that is read.** Without it each new token repeats work proportional to the whole prefix; with it the work per token is almost constant, and what grows is the bytes read: a straight line with a slope that did not depend on the batch size. Past a few thousand cached tokens in the whole batch, reading the cache costs more than reading the weights.
2. **Batching amortises the weights, not the cache.** The intercept of the fit grew slowly with the batch (4.4 ms at 1, 21.4 ms at 64) while the KV term grew with batch and context, which is why the same 64x batch bought 8.6x throughput at short contexts and 2x at long ones. Long contexts and large batches compete for the same memory bandwidth.
3. **Memory management decides the batch.** The batch is limited by how many caches fit, and the allocation policy changed that number by 4x under one budget, with no change to the model. Even a perfect reservation made in advance (the exact-length policy, 50 concurrent) lost to paging (63 to 64), because the output length is unknown until it is produced and reserved space is idle until then. Small blocks waste little (stored share 0.98 at 16 tokens); large blocks waste the tail of every sequence (0.76 at 256), which is the cost of fewer table entries and coarser gathers.
4. **A paged cache needs an attention kernel that reads blocks in place.** Gathering into a contiguous tensor first cost 5 to 6x on this CPU, because it copies the whole cache each step, the very traffic that dominates decoding. That measurement is a property of the naive implementation, not of PagedAttention, whose point is to avoid the copy.
5. **Prefill and decode have different arithmetic.** Prefill is compute that scales with the prompt (0.3 ms per token here), decode is a small amount of compute per token with large reads (5 ms for one token at batch 1). When they share a worker, every arriving prompt stalls every running sequence for the length of the prefill: 171 ms here, matching the 190 to 210 ms gaps seen in about 5 percent of colocated token gaps.
6. **Splitting trades first-token time for steadiness, and the trade depends on capacity.** With one prefill worker serving all arrivals, first-token time rose (p90 258 to 687 ms at 5 requests per second) while gaps lost their tail (p99 197 to 33 ms) and time per output token dropped by about 30 percent at the median. It is the same shape as the series' earlier trade-offs: a resource is moved from one place to another, and a balance has to be chosen. The hand-off was cheap here only because the transfer is a loopback copy; the source's own estimate of 90 Gbps for a 66B model shows why the link matters on real hardware.

## Limits

- Ranks are CPU processes sharing one machine, in fp32, with a 15.9M-parameter model; no GPU was involved. Counts, FLOPs, bytes, correctness and queueing structure carry over; absolute times, the bandwidth of 13.7 GB/s and the 5 to 6x gather penalty do not.
- The decode step was timed with padded batches (all sequences at the same length); real continuous batching mixes lengths and pads or splits the batch accordingly.
- The allocator simulation uses synthetic lognormal lengths, not a production trace, counts one token per request per iteration (it ignores the cost of prefill and of a step growing with the cache), and models preemption as a restart from scratch, which is pessimistic compared with swapping or recomputing the prefix; stored shares and the 4x ratio depend on the length distribution and the budget.
- The serving experiment used a single prompt length (512), one arrival seed, 60 requests, and two rates; the split used one prefill and one decode process while the colocated mode used two replicas, so first-token results reflect that the split had a single prefill worker, and no claim is made about goodput per device.
- Prefix sharing, parallel sampling, chunked prefill, speculative decoding, grouped-query attention, quantised caches and tensor-parallel serving were not measured; the cache sizes here assume full multi-head attention.

## Code and output

Decoder with cache:

```python
import math, torch, torch.nn as nn, torch.nn.functional as F, lm
D, H, L = 384, 8, 8; HD = D // H; MAXPOS = 4096
def make_model(seed=1):
    torch.manual_seed(seed); m = lm.GPT(D, H, L); m.p = nn.Embedding(MAXPOS, D); m.eval(); return m
def kv_bytes_per_token(): return 2 * L * D * 4
@torch.no_grad()
def prefill(m, ids):
    """ids [B,P] -> last logits [B,V], K,V lists of [B,H,P,HD]"""
    B, P = ids.shape; x = m.e(ids) + m.p(torch.arange(P)); Ks, Vs = [], []
    for b in m.b:
        q, k, v = b.qkv(b.l1(x)).view(B, P, 3, H, HD).permute(2, 0, 3, 1, 4); Ks.append(k); Vs.append(v)
        x = x + b.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, P, D)); x = x + b.m(b.l2(x))
    return m.h(m.f(x[:, -1])), Ks, Vs
@torch.no_grad()
def step(m, tok, pos, K, V, lens):
    """tok [B] new tokens at positions pos [B]; K,V lists of preallocated [B,H,NMAX,HD]; lens [B] = cached length before this token. Writes the new k,v and attends over lens+1 positions."""
    B = tok.shape[0]; x = (m.e(tok) + m.p(pos)).unsqueeze(1); nmax = int(lens.max()) + 1; mask = (torch.arange(nmax)[None, :] <= lens[:, None])[:, None, None, :]
    for i, b in enumerate(m.b):
        q, k, v = b.qkv(b.l1(x)).view(B, 1, 3, H, HD).permute(2, 0, 3, 1, 4); K[i][torch.arange(B), :, lens] = k[:, :, 0]; V[i][torch.arange(B), :, lens] = v[:, :, 0]
        a = F.scaled_dot_product_attention(q, K[i][:, :, :nmax], V[i][:, :, :nmax], attn_mask=mask); x = x + b.o(a.transpose(1, 2).reshape(B, 1, D)); x = x + b.m(b.l2(x))
    return m.h(m.f(x[:, 0]))
@torch.no_grad()
def full_logits(m, ids):
    B, P = ids.shape; x = m.e(ids) + m.p(torch.arange(P))
    for b in m.b: x = b(x)
    return m.h(m.f(x[:, -1]))
```

Cache, recompute and decode-step measurements:

```python
import json, time, torch, torch.nn.functional as F, inf, lm
from torch.utils.flop_counter import FlopCounterMode
torch.set_num_threads(1); m = inf.make_model(); out = {}; g = torch.Generator().manual_seed(3)
def med(f, reps=5):
    f(); ts = []
    for _ in range(reps): t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]
# A2 equivalence: cached decoding vs full recompute, 2 sequences, prompt 64, 64 generated tokens (greedy from the cached path)
ids = torch.randint(0, lm.V, (2, 64), generator=g); lg, Ks, Vs = inf.prefill(m, ids); NM = 160; K = [torch.zeros(2, inf.H, NM, inf.HD) for _ in range(inf.L)]; V = [torch.zeros(2, inf.H, NM, inf.HD) for _ in range(inf.L)]
for i in range(inf.L): K[i][:, :, :64] = Ks[i]; V[i][:, :, :64] = Vs[i]
seq = ids.clone(); lens = torch.tensor([64, 64]); md = (lg - inf.full_logits(m, seq)).abs().max().item(); tok = lg.argmax(-1)
for t in range(64):
    seq = torch.cat([seq, tok[:, None]], 1); lg = inf.step(m, tok, lens.clone(), K, V, lens); lens = lens + 1; md = max(md, (lg - inf.full_logits(m, seq)).abs().max().item()); tok = lg.argmax(-1)
out["equiv_maxdiff"] = md; out["kv_bytes_per_token"] = inf.kv_bytes_per_token(); out["params"] = sum(q.numel() for q in m.parameters()); out["params_blocks"] = sum(q.numel() for b in m.b for q in b.parameters())
# A3: per-token cost with cache vs recompute, B=1, context n
rows = []
for n in (32, 64, 128, 256, 512, 1024):
    ids = torch.randint(0, lm.V, (1, n), generator=g); _, Ks, Vs = inf.prefill(m, ids); K = [torch.zeros(1, inf.H, n + 1, inf.HD) for _ in range(inf.L)]; V = [torch.zeros(1, inf.H, n + 1, inf.HD) for _ in range(inf.L)]
    for i in range(inf.L): K[i][:, :, :n] = Ks[i]; V[i][:, :, :n] = Vs[i]
    lens = torch.tensor([n]); tok = torch.tensor([5]); pos = torch.tensor([n])
    fc = FlopCounterMode(display=False)
    with fc: inf.step(m, tok, pos, K, V, lens)
    fs = fc.get_total_flops(); fc2 = FlopCounterMode(display=False)
    with fc2: inf.full_logits(m, torch.cat([ids, tok[None]], 1))
    ff = fc2.get_total_flops()
    rows.append({"n": n, "t_cached": med(lambda: inf.step(m, tok, pos, K, V, lens)), "t_recompute": med(lambda: inf.full_logits(m, torch.cat([ids, tok[None]], 1)), 3), "flops_cached": fs, "flops_recompute": ff})
out["a3"] = rows
# A4 grid: batch x context decode step time
grid = []
for B in (1, 4, 16, 64):
    for n in (128, 512, 2048):
        K = [torch.randn(B, inf.H, n + 1, inf.HD) for _ in range(inf.L)]; V = [torch.randn(B, inf.H, n + 1, inf.HD) for _ in range(inf.L)]; lens = torch.full((B,), n); tok = torch.randint(0, lm.V, (B,), generator=g); pos = torch.full((B,), n)
        grid.append({"B": B, "n": n, "t": med(lambda: inf.step(m, tok, pos, K, V, lens), 3), "kv_mb": B * n * inf.kv_bytes_per_token() / 1e6})
        del K, V
out["a4"] = grid; json.dump(out, open("g7_a.json", "w")); print("done")
```

Paged attention check and allocator simulation:

```python
import json, math, random, time, torch, torch.nn.functional as F, inf, lm
torch.set_num_threads(1); m = inf.make_model(); g = torch.Generator().manual_seed(4); out = {}
@torch.no_grad()
def step_paged(m, tok, pos, PK, PV, tables, lens, bs):
    B = tok.shape[0]; x = (m.e(tok) + m.p(pos)).unsqueeze(1); nmax = int(lens.max()) + 1; mask = (torch.arange(nmax)[None, :] <= lens[:, None])[:, None, None, :]
    blk = torch.tensor([tables[b][int(lens[b]) // bs] for b in range(B)]); slot = lens % bs
    for i, b in enumerate(m.b):
        q, k, v = b.qkv(b.l1(x)).view(B, 1, 3, inf.H, inf.HD).permute(2, 0, 3, 1, 4); PK[i][blk, :, slot] = k[:, :, 0]; PV[i][blk, :, slot] = v[:, :, 0]
        Kg = torch.zeros(B, inf.H, nmax, inf.HD); Vg = torch.zeros(B, inf.H, nmax, inf.HD)
        for s in range(B):
            nb = (int(lens[s]) + 1 + bs - 1) // bs; t = torch.tensor(tables[s][:nb]); Kg[s, :, :min(nmax, nb * bs)] = PK[i][t].permute(1, 0, 2, 3).reshape(inf.H, nb * bs, inf.HD)[:, :nmax]; Vg[s, :, :min(nmax, nb * bs)] = PV[i][t].permute(1, 0, 2, 3).reshape(inf.H, nb * bs, inf.HD)[:, :nmax]
        a = F.scaled_dot_product_attention(q, Kg, Vg, attn_mask=mask); x = x + b.o(a.transpose(1, 2).reshape(B, 1, inf.D)); x = x + b.m(b.l2(x))
    return m.h(m.f(x[:, 0]))
@torch.no_grad()
def step_paged_vec(m, tok, pos, PK, PV, tt, lens, bs):
    B = tok.shape[0]; x = (m.e(tok) + m.p(pos)).unsqueeze(1); nmax = int(lens.max()) + 1; nb = tt.shape[1]; mask = (torch.arange(nmax)[None, :] <= lens[:, None])[:, None, None, :]; blk = tt[torch.arange(B), lens // bs]; slot = lens % bs
    for i, b in enumerate(m.b):
        q, k, v = b.qkv(b.l1(x)).view(B, 1, 3, inf.H, inf.HD).permute(2, 0, 3, 1, 4); PK[i][blk, :, slot] = k[:, :, 0]; PV[i][blk, :, slot] = v[:, :, 0]
        Kg = PK[i][tt].permute(0, 2, 1, 3, 4).reshape(B, inf.H, nb * bs, inf.HD)[:, :, :nmax]; Vg = PV[i][tt].permute(0, 2, 1, 3, 4).reshape(B, inf.H, nb * bs, inf.HD)[:, :, :nmax]
        a = F.scaled_dot_product_attention(q, Kg, Vg, attn_mask=mask); x = x + b.o(a.transpose(1, 2).reshape(B, 1, inf.D)); x = x + b.m(b.l2(x))
    return m.h(m.f(x[:, 0]))
def med(f, reps=5):
    f(); ts = []
    for _ in range(reps): t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]
# B1: paged vs contiguous equivalence, different lengths, scattered physical blocks
bs = 16; lens0 = [70, 100, 33, 129]; B = 4; NB = 64; perm = torch.randperm(NB, generator=g).tolist(); tables = []; used = 0
for l in lens0:
    nb = (l + 1 + bs - 1) // bs; tables.append(perm[used:used + nb]); used += nb
PK = [torch.zeros(NB, inf.H, bs, inf.HD) for _ in range(inf.L)]; PV = [torch.zeros(NB, inf.H, bs, inf.HD) for _ in range(inf.L)]; NM = 160; K = [torch.zeros(B, inf.H, NM, inf.HD) for _ in range(inf.L)]; V = [torch.zeros(B, inf.H, NM, inf.HD) for _ in range(inf.L)]
for s, l in enumerate(lens0):
    ids = torch.randint(0, lm.V, (1, l), generator=g); _, Ks, Vs = inf.prefill(m, ids)
    for i in range(inf.L):
        K[i][s, :, :l] = Ks[i][0]; V[i][s, :, :l] = Vs[i][0]
        for j in range(l): PK[i][tables[s][j // bs], :, j % bs] = Ks[i][0, :, j]; PV[i][tables[s][j // bs], :, j % bs] = Vs[i][0, :, j]
lens = torch.tensor(lens0); tok = torch.randint(0, lm.V, (B,), generator=g); md = 0.0
for t in range(20):
    a = inf.step(m, tok, lens.clone(), K, V, lens); b = step_paged(m, tok, lens.clone(), PK, PV, tables, lens, bs); md = max(md, (a - b).abs().max().item()); lens = lens + 1
    if all(int(l) // bs < len(tables[s]) for s, l in enumerate(lens)) is False: break
    tok = a.argmax(-1)
out["paged_equiv_maxdiff"] = md; out["paged_steps"] = t + 1
# B2: gather overhead, B=16, n=512
for bsz in (16,):
    B = 16; n = 512; nb = (n + 1 + bsz - 1) // bsz; NB = B * nb + 8; perm = torch.randperm(NB).tolist(); tables = [perm[s * nb:(s + 1) * nb] for s in range(B)]
    PK = [torch.randn(NB, inf.H, bsz, inf.HD) for _ in range(inf.L)]; PV = [torch.randn(NB, inf.H, bsz, inf.HD) for _ in range(inf.L)]; K = [torch.randn(B, inf.H, n + 1, inf.HD) for _ in range(inf.L)]; V = [torch.randn(B, inf.H, n + 1, inf.HD) for _ in range(inf.L)]
    lens = torch.full((B,), n); tok = torch.randint(0, lm.V, (B,), generator=g); pos = torch.full((B,), n)
    out["gather"] = {"B": B, "n": n, "block": bsz, "t_contig": med(lambda: inf.step(m, tok, pos, K, V, lens)), "t_paged_loop": med(lambda: step_paged(m, tok, pos, PK, PV, tables, lens, bsz)), "t_paged_vectorised": med(lambda: step_paged_vec(m, tok, pos, PK, PV, torch.tensor(tables), lens, bsz))}
    ve = (step_paged_vec(m, tok, pos, PK, PV, torch.tensor(tables), lens, bsz) - step_paged(m, tok, pos, PK, PV, tables, lens, bsz)).abs().max().item(); out["gather"]["vec_vs_loop_maxdiff"] = ve
# B3: allocator simulation under a fixed KV budget
def lognorm_len(rng, median, sigma, lo, hi): return int(min(hi, max(lo, rng.lognormvariate(math.log(median), sigma))))
def trace(n, seed): r = random.Random(seed); return [(lognorm_len(r, 200, 0.9, 16, 1024), lognorm_len(r, 100, 0.9, 8, 1024)) for _ in range(n)]
def simulate(reqs, budget, policy, bs=16, maxlen=2048):
    queue = list(range(len(reqs))); running = {}; gen = {}; blocks = {}; free = budget // bs if policy == "paged" else budget; it = 0; done = 0; run_sum = 0; util_sum = 0; alloc_sum = 0; stored_sum = 0; preempt = 0
    def need_admit(i):
        p, o = reqs[i]
        if policy == "max": return maxlen
        if policy == "oracle": return p + o
        return (p + bs - 1) // bs + 1
    while done < len(reqs):
        while queue and need_admit(queue[0]) <= free:
            i = queue.pop(0); free -= need_admit(i); running[i] = 1; gen[i] = 0; blocks[i] = need_admit(i) if policy == "paged" else None
            if policy == "paged": blocks[i] = (reqs[i][0] + bs - 1) // bs; free += 1
        if not running: raise RuntimeError("deadlock")
        it += 1
        for i in sorted(running, reverse=True):
            if i not in running: continue
            p, o = reqs[i]; gen[i] += 1; stored = p + gen[i]
            if policy == "paged" and stored > blocks[i] * bs:
                while free < 1:
                    victim = max(running); preempt += 1; free += blocks[victim]; del running[victim]; gen[victim] = 0; blocks[victim] = 0; queue.insert(0, victim)
                    if victim == i: break
                if i in running: free -= 1; blocks[i] += 1
            if i in running and gen[i] >= o:
                free += (blocks[i] if policy == "paged" else need_admit(i)); del running[i]; done += 1
        run_sum += len(running)
        stored = sum(reqs[i][0] + gen[i] for i in running); allocd = sum((blocks[i] * bs if policy == "paged" else need_admit(i)) for i in running)
        stored_sum += stored; alloc_sum += allocd
    toks = sum(o for _, o in reqs)
    return {"iterations": it, "mean_running": run_sum / it, "tokens_per_iteration": toks / it, "stored_over_allocated": stored_sum / max(1, alloc_sum), "allocated_over_budget": alloc_sum / it / budget, "preemptions": preempt}
reqs = trace(1500, 11); budget = 32768; sim = {"budget_tokens": budget, "requests": len(reqs), "mean_prompt": sum(p for p, _ in reqs) / len(reqs), "mean_output": sum(o for _, o in reqs) / len(reqs), "policies": {}}
sim["policies"]["max_reserve_2048"] = simulate(reqs, budget, "max"); sim["policies"]["oracle_exact"] = simulate(reqs, budget, "oracle")
for bs in (1, 4, 16, 64, 256): sim["policies"][f"paged_{bs}"] = simulate(reqs, budget, "paged", bs)
out["sim"] = sim; json.dump(out, open("g7_b.json", "w")); print("done")
```

Colocated and split serving:

```python
import os, sys, json, time, random, torch, torch.distributed as dist, torch.multiprocessing as mp, inf, lm
P, OUT, MAXB, NM = 512, 64, 16, 640; now = time.monotonic
def arrivals(rate, n, seed=21):
    r = random.Random(seed); t = 0.0; ts = []
    for _ in range(n): t += r.expovariate(rate); ts.append(t)
    return ts
class Batch:
    def __init__(s): s.K = [torch.zeros(MAXB, inf.H, NM, inf.HD) for _ in range(inf.L)]; s.V = [torch.zeros(MAXB, inf.H, NM, inf.HD) for _ in range(inf.L)]; s.lens = torch.zeros(MAXB, dtype=torch.long); s.tok = torch.zeros(MAXB, dtype=torch.long); s.ids = []; s.times = {}
    def add(s, rid, kv, first_tok, t_first):  # kv [L,2,H,P,HD]
        i = len(s.ids); s.ids.append(rid)
        for l in range(inf.L): s.K[l][i, :, :P] = kv[l, 0]; s.V[l][i, :, :P] = kv[l, 1]
        s.lens[i] = P; s.tok[i] = first_tok; s.times[rid] = [t_first]
    def step(s, m):
        nb = len(s.ids); K = [k[:nb] for k in s.K]; V = [v[:nb] for v in s.V]; lens = s.lens[:nb].clone(); lg = inf.step(m, s.tok[:nb].clone(), lens.clone(), K, V, lens); t = now(); s.tok[:nb] = lg.argmax(-1); s.lens[:nb] += 1; fin = []
        for j, rid in enumerate(s.ids):
            s.times[rid].append(t)
            if len(s.times[rid]) >= OUT: fin.append(j)
        done = {}
        for j in sorted(fin, reverse=True):
            rid = s.ids[j]; done[rid] = s.times.pop(rid); last = len(s.ids) - 1
            if j != last:
                for l in range(inf.L): s.K[l][j] = s.K[l][last]; s.V[l][j] = s.V[l][last]
                s.lens[j] = s.lens[last]; s.tok[j] = s.tok[last]; s.ids[j] = s.ids[last]
            s.ids.pop()
        return done
def do_prefill(m, rid):
    ids = torch.randint(0, lm.V, (1, P), generator=torch.Generator().manual_seed(rid)); lg, Ks, Vs = inf.prefill(m, ids); kv = torch.stack([torch.stack([Ks[l][0], Vs[l][0]]) for l in range(inf.L)]); return kv, int(lg.argmax(-1)), now()
def coloc(rank, T0, rate, n, outfile):
    torch.set_num_threads(1); m = inf.make_model(); my = [(i, a) for i, a in enumerate(arrivals(rate, n)) if i % 2 == rank]; b = Batch(); pend = []; nxt = 0; res = {}; arr = {}; total = len(my)
    while len(res) < total:
        t = now() - T0
        while nxt < len(my) and my[nxt][1] <= t: pend.append(my[nxt]); arr[my[nxt][0]] = T0 + my[nxt][1]; nxt += 1
        if pend and len(b.ids) < MAXB:
            rid, _ = pend.pop(0); kv, ft, tf = do_prefill(m, rid); b.add(rid, kv, ft, tf)
        if b.ids: res.update(b.step(m))
        elif not pend: time.sleep(0.001)
    json.dump({"rank": rank, "arr": arr, "times": res}, open(outfile, "w"))
def disagg(rank, T0, rate, n, outfile, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)); torch.set_num_threads(1); dist.init_process_group("gloo", rank=rank, world_size=2); m = inf.make_model(); A = arrivals(rate, n)
    if rank == 0:
        keep = []; first = {}; sent_bytes = 0
        for rid, a in enumerate(A):
            while now() - T0 < a: time.sleep(0.0005)
            kv, ft, tf = do_prefill(m, rid); first[rid] = (T0 + a, tf); meta = torch.tensor([float(ft), tf, now()], dtype=torch.float64); kvc = kv.contiguous(); keep += [meta, kvc, dist.isend(meta, 1, tag=2 * rid), dist.isend(kvc, 1, tag=2 * rid + 1)]; sent_bytes += kvc.numel() * 4
            keep = keep[-80:]
        [h.wait() for h in keep if hasattr(h, 'wait')]; dist.barrier(); json.dump({"rank": 0, "first": first, "kv_bytes_total": sent_bytes}, open(outfile, "w"))
    else:
        b = Batch(); res = {}; nxt = 0; cur = None; recv_lat = []
        import threading, queue
        inbox = queue.Queue()
        def receiver():
            for rid in range(n):
                meta = torch.zeros(3, dtype=torch.float64); kv = torch.zeros(inf.L, 2, inf.H, P, inf.HD); dist.recv(meta, 0, tag=2 * rid); dist.recv(kv, 0, tag=2 * rid + 1); inbox.put((rid, meta, kv, now()))
        th = threading.Thread(target=receiver, daemon=True); th.start()
        while len(res) < n:
            while not inbox.empty() and len(b.ids) < MAXB:
                rid, meta, kv, tr = inbox.get(); recv_lat.append(tr - float(meta[2])); b.add(rid, kv, int(meta[0]), float(meta[1]))
            if b.ids: res.update(b.step(m))
            else: time.sleep(0.0005)
        th.join()
        dist.barrier(); json.dump({"rank": 1, "times": res, "recv_latency": recv_lat}, open(outfile, "w"))
    dist.destroy_process_group()
if __name__ == "__main__":
    mode, rate, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]); T0 = now() + 6
    if mode == "coloc": mp.spawn(coloc, args=(T0, rate, n, None), nprocs=2) if False else None
    ctx = mp.get_context("spawn"); ps = []
    for r in range(2):
        ps.append(ctx.Process(target=coloc if mode == "coloc" else disagg, args=(r, T0, rate, n, f"g7_c_{mode}_{rate}_{r}.json") + (() if mode == "coloc" else (30300 + int(rate * 10),))))
    [p.start() for p in ps]; [p.join() for p in ps]; print("done", mode, rate)
```

Statistics output:

```
model: D=384, L=8, H=8, parameters 15,867,776 (blocks 14,195,712); KV cache 24,576 bytes per token = 2*L*D*4
OPT-13B (paper's example, 40 layers, hidden 5120, fp16): 819,200 bytes per token; 1.56 GiB for 2048 tokens
decoding with cache vs full recompute, max logit difference over 64 steps: 8.3e-07

A3: one new token at context n, batch 1
  n=   32: cached   4.53 ms, recompute    14.0 ms (  3.1x); FLOPs cached 2.88e+07, recompute 9.48e+08 (  32.9x)
  n=   64: cached   4.36 ms, recompute    21.3 ms (  4.9x); FLOPs cached 2.92e+07, recompute 1.89e+09 (  64.8x)
  n=  128: cached   4.55 ms, recompute    36.6 ms (  8.1x); FLOPs cached 3.00e+07, recompute 3.86e+09 ( 128.6x)
  n=  256: cached   4.56 ms, recompute    70.4 ms ( 15.4x); FLOPs cached 3.16e+07, recompute 8.09e+09 ( 256.2x)
  n=  512: cached   5.02 ms, recompute   160.8 ms ( 32.0x); FLOPs cached 3.47e+07, recompute 1.78e+10 ( 511.6x)
  n= 1024: cached   5.92 ms, recompute   366.3 ms ( 61.9x); FLOPs cached 4.10e+07, recompute 4.19e+10 (1022.5x)

A4: decode step time, batch B x context n (KV bytes read per step = B*n*24576)
  B=  1 n=  128: step     4.9 ms, KV     3.1 MB, tokens/s    203
  B=  1 n=  512: step     5.2 ms, KV    12.6 MB, tokens/s    192
  B=  1 n= 2048: step     7.9 ms, KV    50.3 MB, tokens/s    127
  B=  4 n=  128: step     8.4 ms, KV    12.6 MB, tokens/s    477
  B=  4 n=  512: step    11.4 ms, KV    50.3 MB, tokens/s    351
  B=  4 n= 2048: step    22.8 ms, KV   201.3 MB, tokens/s    176
  B= 16 n=  128: step    14.1 ms, KV    50.3 MB, tokens/s   1134
  B= 16 n=  512: step    24.8 ms, KV   201.3 MB, tokens/s    646
  B= 16 n= 2048: step    71.5 ms, KV   805.3 MB, tokens/s    224
  B= 64 n=  128: step    36.6 ms, KV   201.3 MB, tokens/s   1749
  B= 64 n=  512: step    80.3 ms, KV   805.3 MB, tokens/s    797
  B= 64 n= 2048: step   256.8 ms, KV  3221.2 MB, tokens/s    249
  shared slope 0.0732 ms per MB of KV read (= 13.7 GB/s effective), per-B intercepts (ms): B=1: 4.4, B=4: 7.7, B=16: 11.0, B=64: 21.4; largest relative residual 6%
  weight bytes read per step about 57 MB; KV bytes equal weight bytes at B*n = 2310 tokens

B1: paged attention (block 16, blocks scattered at random in the pool) vs contiguous cache, 4 sequences of lengths 70/100/33/129: max logit difference 0.0e+00 over 10 steps
B2: B=16, n=512: contiguous 25.4 ms; gather-then-attend loop 130.3 ms (5.1x); vectorised 159.0 ms (6.3x); loop vs vectorised diff 0.0e+00

B3: allocator simulation, 1500 requests (mean prompt 278, mean output 139 tokens), KV budget 32,768 tokens, all requests waiting at time 0
   max_reserve_2048: mean running  15.5, tokens/iteration  15.7, stored/allocated 0.20, allocated/budget 0.97, preemptions 0
       oracle_exact: mean running  50.4, tokens/iteration  50.8, stored/allocated 0.74, allocated/budget 0.85, preemptions 0
            paged_1: mean running  64.0, tokens/iteration  64.2, stored/allocated 1.00, allocated/budget 0.80, preemptions 555
            paged_4: mean running  63.9, tokens/iteration  64.1, stored/allocated 1.00, allocated/budget 0.80, preemptions 521
           paged_16: mean running  63.2, tokens/iteration  63.3, stored/allocated 0.98, allocated/budget 0.80, preemptions 514
           paged_64: mean running  60.2, tokens/iteration  60.4, stored/allocated 0.93, allocated/budget 0.81, preemptions 333
          paged_256: mean running  51.5, tokens/iteration  51.7, stored/allocated 0.76, allocated/budget 0.84, preemptions 142

C1: prefill of P tokens (batch 1) vs one decode step (batch 1, context P)
  P=   64: prefill   21.4 ms = 0.334 ms per token
  P=  128: prefill   37.1 ms = 0.290 ms per token
  P=  256: prefill   78.5 ms = 0.307 ms per token
  P=  512: prefill  171.0 ms = 0.334 ms per token
  P= 1024: prefill  347.0 ms = 0.339 ms per token
  decode step batch 1: 4.6 ms at n=256, 5.0 ms at n=512

C2: serving 60 requests, prompt 512, output 64 tokens, max batch 16, Poisson arrivals; 2 processes in both modes
  KV cache per request after prefill: 12.58 MB
  3 req/s  coloc: TTFT p50    171 p90    193 ms; per-request TPOT p50  13.2 p90  20.7 ms; token gaps p50  10.3 p99  185.8 max  202.7 ms; gaps over 100 ms: 1.6%
  3 req/s disagg: TTFT p50    189 p90    544 ms; per-request TPOT p50  10.2 p90  15.4 ms; token gaps p50  10.2 p99   26.3 max   46.2 ms; gaps over 100 ms: 0.0%; KV hand-off median 6 ms, p90 7 ms
  5 req/s  coloc: TTFT p50    180 p90    258 ms; per-request TPOT p50  24.8 p90  34.1 ms; token gaps p50  16.6 p99  196.5 max  212.1 ms; gaps over 100 ms: 4.9%
  5 req/s disagg: TTFT p50    331 p90    687 ms; per-request TPOT p50  17.7 p90  21.5 ms; token gaps p50  17.4 p99   32.8 max   51.2 ms; gaps over 100 ms: 0.0%; KV hand-off median 7 ms, p90 8 ms
```

## Series summary

Across the seven parts the same accounting kept returning: bytes per parameter and per token decide memory ([Part 1](/posts/dist-1-accounting/)); how often and how much a rank must send decides communication ([Part 2](/posts/dist-2-data-parallel/), [Part 3](/posts/dist-3-sharding/), [Part 4](/posts/dist-4-tensor-parallel/)); how idle time arises from schedules and routing decides utilisation ([Part 5](/posts/dist-5-pipeline/), [Part 6](/posts/dist-6-moe/)); and in serving the cache plays the role the optimiser state played in training.

## References

- [PagedAttention: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180), 2023.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670), 2024.
