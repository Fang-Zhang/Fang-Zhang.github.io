---
title: "Distributed LLMs, Measured (2/7): Ring All-Reduce Sent Exactly 2(N-1)/N of the Gradient per Rank, and Overlap Hid 42 Percent of the Communication at 2 Ranks but 3 Percent at 8"
date: 2027-07-07 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, distributed training, data parallel, all-reduce, gloo, series]
description: "Part 2 of 7. Data parallelism and its collectives: gradient averaging checked against a single process, ring and naive all-reduce written by hand with bytes counted, and bucketed overlap measured in eight-process runs."
---

## What data parallelism needs from the network

[Part 1](/posts/dist-1-accounting/) showed that a model's states cost 16 bytes per parameter. Data parallelism (DP) keeps a full copy of those states on every rank, splits the batch, and averages the gradients so that all copies take the same step. The whole technique rests on one collective, all-reduce, and its cost sets the efficiency ceiling of everything layered on top.

## What the sources say

- [Horovod](https://arxiv.org/abs/1802.05799) describes "efficient inter-GPU communication via ring reduction" and a few lines of change to user code as its two contributions.
- [ZeRO](https://arxiv.org/abs/1910.02054) states that the state-of-the-art all-reduce is a reduce-scatter followed by an all-gather, each moving Psi elements for a payload of Psi elements, so standard data parallelism moves 2 Psi per step. Part 3 builds on this decomposition.
- [The PyTorch DDP paper](https://arxiv.org/abs/2006.15704) lists bucketing gradients, overlapping computation with communication, and skipping synchronisation as the main techniques, and reports near-linear scalability on 256 GPUs when configured appropriately.
- [Goyal et al.](https://arxiv.org/abs/1706.02677) address the other side of DP: the global batch grows with the number of ranks, which they handle with a linear learning-rate scaling rule and warmup, and they report about 90 percent scaling efficiency from 8 to 256 GPUs.

## Derivation

For a vector of S bytes on N ranks, the ring algorithm cuts it into N chunks of S/N bytes. Reduce-scatter takes N-1 steps, in each of which a rank sends one chunk to its right neighbour and adds the chunk received from its left neighbour. After it, rank r holds the complete sum of one chunk. All-gather takes N-1 more steps that circulate the finished chunks. Each rank therefore sends 2(N-1)/N x S bytes, below 2S for any N, and the time is about 2(N-1) alpha + 2(N-1)/N x S/beta for per-message latency alpha and link bandwidth beta. The naive alternative, in which every rank sends its whole vector to all N-1 others, sends (N-1) x S bytes with a single message per peer, so it has a smaller latency term and a bandwidth term that grows with N.

## Experiment

Everything runs in `torch.distributed` with the gloo backend, one process per rank and one thread per process, on one 32-core machine. Three measurements:

1. **Correctness.** A 891,776-parameter GPT takes a global batch of 32 sequences. Each of N ranks computes the gradient of its 32/N sequences, the gradients are summed with `all_reduce` and divided by N, and the result is compared with the gradient of the whole batch in one process.
2. **Collectives by hand.** A ring all-reduce built from `isend` and `irecv` (reduce-scatter, then all-gather) and the naive all-to-all version, each counting the bytes this rank sends. Both are checked against the built-in `all_reduce`, and timed with the built-in one at five vector sizes from 4 KB to 16.8 MB (median of 7 repetitions after 2 warmups).
3. **A training step.** A 14,491,520-parameter GPT (gradient 57.97 MB) with a fixed per-rank batch of 4 x 128 tokens, at 1, 2, 4 and 8 ranks. Three variants are timed: compute alone; compute followed by one all-reduce of the flattened gradient; and 4 buckets of about 3.6M parameters in reverse parameter order, each launched with an asynchronous all-reduce from a gradient hook as soon as its gradients are ready, the technique the DDP paper calls bucketing with overlap.

Results:

- **Averaging is equivalent.** The maximum absolute difference from the single-process gradient was 7.0e-09, 6.1e-09 and 5.6e-09 at 2, 4 and 8 ranks, against a largest gradient entry of 8.5e-03. The difference is fp32 summation order, not a different algorithm.
- **Volume matches the formulas exactly.** At 16.8 MB the ring sent 1.000, 1.500 and 1.750 times the vector at 2, 4 and 8 ranks, which is 2(N-1)/N, and the naive method sent 1, 3 and 7 times, which is N-1. Both reproduced the built-in result to within 2.9e-06.
- **Latency decides the small sizes.** At 4 KB the naive method was faster at every N (0.26 against 0.63 ms at 2 ranks, 1.01 against 4.62 ms at 8 ranks) and still faster at 66 KB. At 1.05 MB the ring was faster at 4 and 8 ranks (14.8 against 25.1 ms, and 28.0 against 78.1 ms), and at 16.8 MB by 1.8 times at 4 ranks and 4.2 times at 8 ranks. At 2 ranks the two send the same bytes and ran within 20 percent of each other above 1 MB.
- **Time did not stay flat with N.** In theory the ring's bandwidth term, 2(N-1)/N x S/beta, grows by only 1.75 times from 2 to 8 ranks. The built-in all-reduce of 16.8 MB took 150, 220 and 546 ms at 2, 4 and 8 ranks, which is 9.0, 13.1 and 32.6 ms per MB, a factor of 3.6.
- **A training step.** Compute alone took 430, 483, 513 and 963 ms at 1, 2, 4 and 8 ranks. Compute followed by an all-reduce took 411 (no communication at one rank; the gap to 430 ms indicates a timing noise of about 20 ms), 942, 1,329 and 3,202 ms, so the added time was 459, 816 and 2,238 ms at 2, 4 and 8 ranks. With bucketed overlap the step took 749, 1,183 and 3,140 ms. Weak-scaling efficiency, one-rank time divided by N-rank time, was 0.46, 0.32 and 0.13 without overlap and 0.57, 0.36 and 0.14 with it.

![Three panels. Left: bytes sent per rank over vector size for 2, 4 and 8 ranks; measured ring points sit on 2(N-1)/N (1.0, 1.5, 1.75) and naive points on N-1 (1, 3, 7). Middle: naive time divided by ring time against vector size; below 0.1 MB the naive method is faster, above 1 MB the ring is faster, by up to about 4 times at 8 ranks. Right: step time of a 14.5M-parameter GPT for 1, 2, 4 and 8 ranks; compute alone grows slowly, compute followed by all-reduce grows to about 3.2 seconds at 8 ranks, and bucketed overlap barely helps]({{ site.baseurl }}/assets/pic/dist-2-data-parallel.png){: width="900" }
_Figure: left, measured bytes sent per rank against the two formulas at 16.8 MB; middle, naive time over ring time by vector size; right, step time of the 14.5M-parameter model by number of ranks._

## Engineering reading

These points are this series' reading of the measurement.

1. **The volume formula is exact and independent of the algorithm's speed.** Ring all-reduce costs 2(N-1)/N x S per rank, within a factor of 2 of the single-process gradient size at any scale. This is why DP can scale in principle: per-rank traffic stops growing with N. It is also the 2 Psi of the ZeRO paper, as a reduce-scatter plus an all-gather of Psi each.
2. **Latency and bandwidth pull in opposite directions.** Ring pays 2(N-1) sequential steps; at 4 KB that cost made the naive method 2 to 5 times faster. Below about 0.1 MB the better choice is a method with fewer steps; above 1 MB it is the ring. Real libraries choose by message size and topology, and gradient bucketing exists partly to keep messages above the latency-dominated range.
3. **The single-host time scaling contradicts the textbook model, and the reason is the setup.** All 8 ranks share one memory system and one kernel loopback path, so the "links" are not independent and the per-rank bandwidth falls as ranks are added. The compute time also grew, from 430 to 963 ms at 8 ranks, with the same per-rank work. This part's time numbers therefore measure how contended a shared host is, and only the byte counts transfer to a cluster.
4. **Overlap is bounded by what there is to hide behind.** Bucketed overlap cut the added communication time from 459 to 266 ms at 2 ranks (42 percent), from 816 to 670 ms at 4 ranks (18 percent) and from 2,238 to 2,177 ms at 8 ranks (3 percent). The ideal limit is a step time of max(compute, communication), which here is 483 ms at 2 ranks against 749 ms measured. Two likely reasons, not isolated here, are that communication cannot start until the first bucket's gradients exist, which happens only partway through the backward pass, and that gloo's communication threads compete with compute for the same cores. When communication is several times the compute, as at 8 ranks, no schedule can hide it.
5. **Global batch size is a separate cost that this part does not measure.** With a fixed per-rank batch the global batch grows with N, and Goyal et al.'s linear scaling and warmup rules exist because large batches change optimisation. The step-time efficiency above says nothing about steps needed to reach a given loss.

## Limits

- Ranks are processes on one CPU host with loopback transport. Times reflect memory and kernel contention and are not a prediction for GPUs with NVLink or InfiniBand; only the volume counts and equivalence results carry over.
- Timing noise is about 20 ms per measurement (one-rank row). Medians of 5 to 7 repetitions in a single launch per configuration were used; no confidence intervals.
- The hand-written ring is not tuned (no chunk pipelining beyond N chunks), so it is not a competitor to the built-in version; both ran on the same transport. The built-in algorithm was not inspected here, and its larger time at 8 ranks against the hand-written ring should not be generalised.
- The overlap experiment used one bucket size and one model; the reasons given for the small gain were not isolated.

## Code and output

```python
import os, sys, json, time, statistics as st, torch, torch.distributed as dist, torch.multiprocessing as mp, torch.nn.functional as F, lm
def setup(rank, ws, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), OMP_NUM_THREADS="1"); torch.set_num_threads(1); dist.init_process_group("gloo", rank=rank, world_size=ws)
def med(f, reps=7, warm=2):
    ts = []
    for i in range(reps + warm):
        dist.barrier(); t = time.perf_counter(); f(); dt = time.perf_counter() - t
        if i >= warm: ts.append(dt)
    return st.median(ts)
def ring_allreduce(x):
    """x: flat fp32 tensor, in place. returns bytes sent by this rank"""
    N, r = dist.get_world_size(), dist.get_rank(); n = x.numel(); pad = (-n) % N; y = torch.cat([x, x.new_zeros(pad)]) if pad else x; c = y.view(N, -1); buf = torch.empty_like(c[0]); right, left = (r + 1) % N, (r - 1) % N; sent = 0
    for s in range(N - 1):
        si, ri = (r - s) % N, (r - s - 1) % N; w = [dist.isend(c[si].contiguous(), right), dist.irecv(buf, left)]; [q.wait() for q in w]; c[ri] += buf; sent += c[si].numel() * 4
    for s in range(N - 1):
        si, ri = (r + 1 - s) % N, (r - s) % N; w = [dist.isend(c[si].contiguous(), right), dist.irecv(buf, left)]; [q.wait() for q in w]; c[ri].copy_(buf); sent += c[si].numel() * 4
    x.copy_(y[:n]); return sent
def naive_allreduce(x):
    N, r = dist.get_world_size(), dist.get_rank(); bufs = {p: torch.empty_like(x) for p in range(N) if p != r}; w = []
    for p in bufs: w += [dist.isend(x.clone(), p), dist.irecv(bufs[p], p)]
    [q.wait() for q in w]
    for b in bufs.values(): x += b
    return (N - 1) * x.numel() * 4
def collectives(rank, ws):
    out = []
    for e in (10, 14, 18, 20, 22):
        n = 2 ** e; torch.manual_seed(rank); x0 = torch.randn(n); ref = x0.clone(); dist.all_reduce(ref); res = {"ws": ws, "floats": n, "bytes": n * 4}
        x = x0.clone(); sent = ring_allreduce(x); res["ring_bytes_sent"] = sent; res["ring_maxdiff"] = (x - ref).abs().max().item(); x = x0.clone(); res["naive_bytes_sent"] = naive_allreduce(x); res["naive_maxdiff"] = (x - ref).abs().max().item()
        res["t_builtin"] = med(lambda: dist.all_reduce(x0.clone())); res["t_ring"] = med(lambda: ring_allreduce(x0.clone())); res["t_naive"] = med(lambda: naive_allreduce(x0.clone())); out.append(res)
    return out
def correctness(rank, ws):
    G = 32; torch.manual_seed(0); full = torch.randint(0, lm.V, (G, 65)); m = lm.GPT(128, 4, 4); sd = {k: v.clone() for k, v in m.state_dict().items()}
    lossf = lambda mm, b: F.cross_entropy(mm(b[:, :-1]).reshape(-1, lm.V), b[:, 1:].reshape(-1))
    lossf(m, full).backward(); ref = [p.grad.clone() for p in m.parameters()]; m.zero_grad()
    k = G // ws; lossf(m, full[rank * k:(rank + 1) * k]).backward(); gs = torch.cat([p.grad.flatten() for p in m.parameters()]); dist.all_reduce(gs); gs /= ws
    rf = torch.cat([g.flatten() for g in ref]); return {"ws": ws, "max_abs_diff": (gs - rf).abs().max().item(), "max_abs_grad": rf.abs().max().item()}
def dp_step(rank, ws):
    torch.manual_seed(1); m = lm.GPT(384, 8, 8); params = [p for p in m.parameters()]; N = sum(p.numel() for p in params); B, S = 4, 128; x = torch.randint(0, lm.V, (B, S + 1))
    lossf = lambda: F.cross_entropy(m(x[:, :-1]).reshape(-1, lm.V), x[:, 1:].reshape(-1))
    def fwdbwd(): m.zero_grad(set_to_none=True); lossf().backward()
    def sync_after():
        fwdbwd()
        if ws > 1: g = torch.cat([p.grad.flatten() for p in params]); dist.all_reduce(g); g /= ws
    # bucketed overlap: reverse order buckets of ~3.6M elements, async all-reduce as soon as a bucket's grads are all ready
    order = params[::-1]; buckets = [[]]; cnt = 0
    for p in order:
        buckets[-1].append(p); cnt += p.numel()
        if cnt >= 3_600_000: buckets.append([]); cnt = 0
    buckets = [b for b in buckets if b]; where = {id(p): i for i, b in enumerate(buckets) for p in b}; state = {"ready": [0] * len(buckets), "works": []}
    def hook(p):
        i = where[id(p)]; state["ready"][i] += 1
        if state["ready"][i] == len(buckets[i]): g = torch.cat([q.grad.flatten() for q in buckets[i]]); state["works"].append((dist.all_reduce(g, async_op=True), g))
    hs = [p.register_post_accumulate_grad_hook(hook) for p in params] if ws > 1 else []
    def overlap():
        state["ready"] = [0] * len(buckets); state["works"] = []; m.zero_grad(set_to_none=True); lossf().backward()
        for w, g in state["works"]: w.wait()
    tc = med(fwdbwd, 6, 2); ts = med(sync_after, 6, 2); to = med(overlap, 6, 2)
    return {"ws": ws, "params": N, "grad_bytes": N * 4, "t_compute": tc, "t_sync_after": ts, "t_overlap": to, "buckets": len(buckets)}
def worker(rank, ws, port, mode, outfile):
    setup(rank, ws, port); res = {"collectives": collectives, "correctness": correctness, "dp_step": dp_step}[mode](rank, ws)
    if rank == 0: json.dump(res, open(outfile, "w"))
    dist.destroy_process_group()
if __name__ == "__main__":
    mode, ws = sys.argv[1], int(sys.argv[2]); port = 29600 + ws * 7 + len(mode); mp.spawn(worker, args=(ws, port, mode, f"g2_{mode}_{ws}.json"), nprocs=ws)
```

Statistics output:

```
A. gradient averaging across ranks vs one process on the whole batch (model 891,776 params, global batch 32)
  2 ranks: max |difference| 6.98e-09, largest gradient entry 8.47e-03
  4 ranks: max |difference| 6.05e-09, largest gradient entry 8.47e-03
  8 ranks: max |difference| 5.59e-09, largest gradient entry 8.47e-03

B. all-reduce of an fp32 vector, gloo, one machine, one thread per rank; bytes sent per rank and median time (7 reps)
  2 ranks (ring formula 2(N-1)/N = 1.000 x size; naive (N-1) = 1 x size)
       0.004 MB: ring sent 1.000x, naive sent 1.000x; max diff vs built-in ring 0.0e+00 naive 0.0e+00; time built-in     0.54 ms, hand-written ring     0.63 ms, naive     0.26 ms
       0.066 MB: ring sent 1.000x, naive sent 1.000x; max diff vs built-in ring 0.0e+00 naive 0.0e+00; time built-in     0.77 ms, hand-written ring     0.81 ms, naive     0.31 ms
       1.049 MB: ring sent 1.000x, naive sent 1.000x; max diff vs built-in ring 0.0e+00 naive 0.0e+00; time built-in     9.96 ms, hand-written ring    10.35 ms, naive    11.05 ms
       4.194 MB: ring sent 1.000x, naive sent 1.000x; max diff vs built-in ring 0.0e+00 naive 0.0e+00; time built-in    37.72 ms, hand-written ring    41.52 ms, naive    42.79 ms
      16.777 MB: ring sent 1.000x, naive sent 1.000x; max diff vs built-in ring 0.0e+00 naive 0.0e+00; time built-in   150.24 ms, hand-written ring   121.55 ms, naive   143.85 ms
  4 ranks (ring formula 2(N-1)/N = 1.500 x size; naive (N-1) = 3 x size)
       0.004 MB: ring sent 1.500x, naive sent 3.000x; max diff vs built-in ring 4.8e-07 naive 4.8e-07; time built-in     1.61 ms, hand-written ring     1.54 ms, naive     0.45 ms
       0.066 MB: ring sent 1.500x, naive sent 3.000x; max diff vs built-in ring 9.5e-07 naive 9.5e-07; time built-in     2.15 ms, hand-written ring     2.08 ms, naive     1.40 ms
       1.049 MB: ring sent 1.500x, naive sent 3.000x; max diff vs built-in ring 9.5e-07 naive 9.5e-07; time built-in    13.68 ms, hand-written ring    14.79 ms, naive    25.06 ms
       4.194 MB: ring sent 1.500x, naive sent 3.000x; max diff vs built-in ring 9.5e-07 naive 1.9e-06; time built-in    50.43 ms, hand-written ring    56.73 ms, naive    94.04 ms
      16.777 MB: ring sent 1.500x, naive sent 3.000x; max diff vs built-in ring 9.5e-07 naive 9.5e-07; time built-in   220.25 ms, hand-written ring   233.02 ms, naive   412.14 ms
  8 ranks (ring formula 2(N-1)/N = 1.750 x size; naive (N-1) = 7 x size)
       0.004 MB: ring sent 1.750x, naive sent 7.000x; max diff vs built-in ring 9.5e-07 naive 1.9e-06; time built-in     4.80 ms, hand-written ring     4.62 ms, naive     1.01 ms
       0.066 MB: ring sent 1.750x, naive sent 7.000x; max diff vs built-in ring 1.9e-06 naive 1.9e-06; time built-in     6.05 ms, hand-written ring     5.77 ms, naive     4.70 ms
       1.049 MB: ring sent 1.750x, naive sent 7.000x; max diff vs built-in ring 1.9e-06 naive 1.9e-06; time built-in    24.76 ms, hand-written ring    27.96 ms, naive    78.09 ms
       4.194 MB: ring sent 1.750x, naive sent 7.000x; max diff vs built-in ring 2.9e-06 naive 2.9e-06; time built-in    91.13 ms, hand-written ring   101.32 ms, naive   387.45 ms
      16.777 MB: ring sent 1.750x, naive sent 7.000x; max diff vs built-in ring 2.9e-06 naive 2.9e-06; time built-in   546.10 ms, hand-written ring   365.91 ms, naive  1552.82 ms

   time per MB of vector (built-in all-reduce, 16.777 MB): 2 ranks 9.0 ms/MB, 4 ranks 13.1 ms/MB, 8 ranks 32.6 ms/MB

C. one training step of a 14,491,520-parameter GPT (gradient 57.97 MB), per-rank batch 4 x 128 tokens, one thread per rank
  1 ranks: compute   430.0 ms; compute then one all-reduce   410.7 ms (communication  -19.2 ms); bucketed overlap (4 buckets)   420.0 ms; scaling efficiency vs 1 rank: sync-after 1.05, overlap 1.02
  2 ranks: compute   482.9 ms; compute then one all-reduce   942.3 ms (communication  459.3 ms); bucketed overlap (4 buckets)   749.2 ms; scaling efficiency vs 1 rank: sync-after 0.46, overlap 0.57
  4 ranks: compute   513.3 ms; compute then one all-reduce  1328.8 ms (communication  815.5 ms); bucketed overlap (4 buckets)  1183.2 ms; scaling efficiency vs 1 rank: sync-after 0.32, overlap 0.36
  8 ranks: compute   963.4 ms; compute then one all-reduce  3201.7 ms (communication 2238.3 ms); bucketed overlap (4 buckets)  3140.2 ms; scaling efficiency vs 1 rank: sync-after 0.13, overlap 0.14
```

Next: [Part 3](/posts/dist-3-sharding/) shards the optimizer states, gradients and parameters across ranks, so that the 16 bytes per parameter stop being replicated.

## References

- [Horovod: fast and easy distributed deep learning in TensorFlow](https://arxiv.org/abs/1802.05799), 2018.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), 2019.
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704), 2020.
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677), 2017.
