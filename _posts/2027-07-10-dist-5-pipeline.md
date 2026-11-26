---
title: "Distributed LLMs, Measured (5/7): The Pipeline Bubble Followed (p-1)/(m+p-1) to Within 0.07, 1F1B Held 8 Microbatches Instead of 16, and Tensor Traffic Dominated a 2x2x2 Layout"
date: 2027-07-10 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, distributed training, pipeline parallel, 3d parallelism, series]
description: "Part 5 of 7. Pipeline parallelism written by hand with GPipe and 1F1B schedules: measured bubble against the formula, stored microbatches, and a combined tensor, pipeline and data parallel layout with its traffic accounted."
---

## Splitting by depth

[Part 4](/posts/dist-4-tensor-parallel/) split every layer across ranks. Pipeline parallelism splits the stack: rank k holds a contiguous group of layers and passes activations to rank k+1. Traffic is small (one activation tensor per boundary per microbatch), but the stages depend on each other, so the cost appears as idle time, the bubble.

## What the sources say

- [GPipe](https://arxiv.org/abs/1811.06965) splits a mini-batch into micro-batches, pipelines them through consecutive groups of layers, and supports re-materialization on every accelerator. It states the bubble overhead as O((K-1)/(M+K-1)) for K partitions and M micro-steps, found it negligible when M >= 4K in its experiments, and attributes this partly to scheduling re-computation earlier in the backward pass.
- [Megatron-LM at scale](https://arxiv.org/abs/2104.04473) gives the bubble time fraction as (p-1)/m, with p pipeline stages and m microbatches, so m must be much larger than p, but then a schedule that keeps all m microbatches stored needs memory proportional to m. It therefore uses the PipeDream-Flush (1F1B) schedule, which "limits the number of in-flight microbatches to the depth of the pipeline, instead of the number of microbatches in a batch".
- The same paper's takeaways: tensor parallelism up to degree g on g-GPU servers, then pipeline parallelism across servers (Takeaway 1); a model-parallel size t x p chosen so that parameters and intermediate data fit in GPU memory, with data parallelism to scale beyond that (Takeaway 2). It reports a 1-trillion-parameter model at 502 petaFLOP/s on 3,072 GPUs, 52 percent of theoretical peak per GPU.

## Experiments

Same GPT as before (d = 384, 8 layers, 14,344,064 parameters), global batch 16 sequences of 128 tokens, gloo, one process and one thread per rank, point-to-point `send`, `recv` and `isend` between neighbouring ranks.

**A. Pipeline only.** p = 2, 4 and 8 ranks with 4, 2 and 1 layers per rank, both schedules, m = 1, 2, 4, 8 and 16 microbatches of 16/m sequences. The first rank also holds the embeddings and the last rank the head and loss. For each step the busy time of a rank is the time spent inside its own forward and backward computation; the idle share is 1 - mean(busy) / wall, where wall runs from a barrier before the step to a barrier after it. Median of three steps after one warmup. Every gradient was compared with the gradient of the unsplit model on the same batch.

**B. Combined layout.** All three degrees on 8 processes: tensor 2 x pipeline 2 x data 2, using the tensor-parallel blocks of Part 4 on process groups of two ranks, 1F1B over 2 stages with 4 microbatches of 2 sequences per replica, and one gradient all-reduce over each data-parallel pair at the end. Calls and bytes are counted per rank, and loss and gradients are compared with one process on the whole batch.

Results:

- **Equivalence.** All 30 pipeline-only configurations produced gradients within 1.0e-09 of the unsplit model; the combined layout was within 3.5e-09 on every rank, and its loss, averaged over the two data-parallel replicas, was 5.02417 against 5.02418 for one process.
- **Bubble.** Idle share at m = 1 was 0.51, 0.76 and 0.89 at p = 2, 4 and 8 (formula 0.50, 0.75, 0.875). At m = 16 it was 0.09, 0.22 and 0.36 with 1F1B (formula 0.059, 0.158, 0.304). Measured values were above the formula at every point, by 0.01 at m = 1 up to 0.066 at p = 4, m = 16, and the gap grew with m. The two schedules had the same bubble within 0.02 (for p = 4, m = 8: 0.317 for GPipe, 0.307 for 1F1B).
- **Stored microbatches.** The peak number of microbatches with stored activations on any rank equalled m for GPipe (up to 16) and min(m, p) for 1F1B: 4 at p = 4 for m = 8 and 16, 8 at p = 8 for m = 16.
- **Wall time.** For every p the best wall time was at m = 8, not at the largest m: at p = 4, 1F1B took 1,676, 1,105, 834, 794 and 807 ms at m = 1, 2, 4, 8 and 16; at p = 8 it took 1,817, 1,086, 757, 687 and 778 ms. The cause was not isolated; two candidates on this machine are less efficient small matrices (m = 16 means one sequence per microbatch) and more messages.
- **Combined layout traffic.** Per rank and step: 64 tensor-parallel all-reduces of 393,216 bytes (4 per layer x 4 layers per stage x 4 microbatches, as in Part 4), 4 pipeline sends of 393,216 bytes, and one data-parallel all-reduce of 14,607,360 bytes (14,414,336 on the last stage). With a ring over two ranks this is 25.2 MB, 1.6 MB and 14.6 MB sent per rank. Each rank held 3.65 M or 3.60 M parameters, against 3.59 M for an ideal quarter of the model; the embeddings and the head are the extra.

![Three panels. Left: idle share of device time against microbatch count for 2, 4 and 8 pipeline stages, measured points just above the lines (p-1)/(m+p-1), falling from 0.51, 0.76, 0.89 at one microbatch to 0.09, 0.22, 0.36 at sixteen. Middle: peak number of microbatches with stored activations: equal to m for the GPipe schedule, capped at p for 1F1B. Right: payload per rank per step in the 2 by 2 by 2 layout: tensor 25.2 MB, data 14.6 MB, pipeline 1.6 MB]({{ site.baseurl }}/assets/pic/dist-5-pipeline-3d.png){: width="900" }
_Figure: left, measured idle share (points) against (p-1)/(m+p-1) (lines); middle, peak stored microbatches per schedule; right, payload per rank per step in the 2 x 2 x 2 layout._

**C. Arithmetic at scale (no measurement).** The formulas confirmed above applied to a 70B-like model (80 layers, hidden size 8,192, 70 billion parameters) on 128 GPUs with bf16 activations and gradients, 1,024 sequences of 4,096 tokens per step, microbatch of one sequence, interior pipeline stage. Model states are counted at 16 bytes per parameter as in Part 1, with no sharding:

```
(t,p,d)    states GB/GPU  tensor GB   pipeline GB  data GB   bubble
(8,2,8)         70.0      2405.2        17.2      15.3   0.008
(4,4,8)         70.0      1030.8        17.2      15.3   0.023
(2,8,8)         70.0       343.6        17.2      15.3   0.052
(1,16,8)        70.0         0.0        17.2      15.3   0.105
(8,4,4)         35.0      2405.2        34.4       6.6   0.012
(4,8,4)         35.0      1030.8        34.4       6.6   0.027
(8,8,2)         17.5      2405.2        68.7       2.2   0.013
```

At 400 TFLOPs/s per GPU (an assumption, not a measurement) one step is 1.76e18 FLOPs, or 34.4 s on 128 GPUs.

## Engineering reading

These points are this series' reading of the measurements and the sources.

1. **The bubble formula is a good first estimate and a lower bound here.** It was right at m = 1 and low by up to 0.066 later. Since it assumes equal stage times and free communication, the gap measures the per-microbatch costs the formula leaves out; the gap grows with m, where microbatches are smallest. The GPipe rule of thumb (negligible at M >= 4K) corresponds to p = 4, m = 16 in this test, where 22 percent of device time was still idle. The paper credits earlier-scheduled re-computation for part of its result, which was not used here.
2. **1F1B buys memory, not a smaller bubble.** Both schedules idled the same amount, but the peak number of stored microbatches fell from m to p, a factor of 2 at p = 8, m = 16 and 4 at p = 4, m = 16 in the stored-activation count. That is what allows a large m: the bubble shrinks with m, and with 1F1B the memory does not grow with it.
3. **Throughput has an interior optimum in m.** The bubble falls monotonically with m, but wall time did not: it was lowest at m = 8 for every p. A real system chooses the microbatch size against kernel efficiency and the bubble together, which is Megatron's Takeaway 3 (the optimal microbatch size depends on throughput characteristics, pipeline depth, data-parallel size and batch size).
4. **In the combined layout the three kinds of traffic differ by orders of magnitude, and they are paid at different times.** Measured at toy scale, tensor traffic was 1.7 times the data-parallel gradient bytes and 16 times the pipeline bytes. In the 70B-like arithmetic, tensor traffic at t = 8 is 2,405 GB per GPU per step, 157 times the 15.3 GB of data-parallel traffic, which agrees with the Megatron advice to keep tensor parallelism within a server where the links are fastest. Pipeline traffic does not depend on the stage count p (17.2 GB per interior stage per step at d = 8), but it doubles when d halves, because each replica then handles twice as many microbatches.
5. **The layout is a memory solution first.** With 16 bytes per parameter, t x p = 16 leaves 70 GB of an 80 GB GPU for states alone; t x p = 32 or 64 leaves room for activations. This is the content of Megatron's Takeaway 2, and sharding the optimizer states as in Part 3 is the alternative way of getting the same room, at the price of the communication Part 3 measured.

## Limits

- All ranks are processes on one CPU host over loopback sockets. Idle shares measure the schedule and the small-matrix behaviour of this machine, not GPU pipelines; the byte and call counts are the transferable results.
- The pipeline-only runs use one thread per rank and a 14M-parameter model; no interleaved schedule, no activation recomputation and no overlap of sends with compute were tried.
- In the combined layout each tensor-parallel rank sends its own copy of the boundary activation, so pipeline traffic is duplicated across the tensor-parallel group; scatter-gather optimisations are not covered. Only 4 microbatches and one step were run, so no timing is reported for it.
- The data-parallel step sends one flat gradient vector after the backward pass, with no overlap.
- Section C is arithmetic: it assumes the formulas of Parts 2 to 5 hold at scale and ignores embeddings, attention-score traffic, topology and any sharding of states.

## Code and output

Pipeline only:

```python
import os, sys, json, time, statistics as st, torch, torch.distributed as dist, torch.multiprocessing as mp, torch.nn.functional as F, lm
D, H, L, S, BG = 384, 8, 8, 128, 16
def one_step(full, rank, p, schedule, m, batch):
    bs = L // p; blocks = list(full.b[rank * bs:(rank + 1) * bs]); first, last = rank == 0, rank == p - 1; mbs = batch.chunk(m); acts = {}; reqs = []; keep = []; busy = [0.0]; peak = [0]; loss_sum = [0.0]
    def timed(f):
        t = time.perf_counter(); r = f(); busy[0] += time.perf_counter() - t; return r
    def fwd(i):
        if first: ids = mbs[i][:, :-1]; xin = None; run = lambda: blocks_fwd(full.e(ids) + full.p(torch.arange(ids.shape[1])))
        else: buf = torch.empty(mbs[i].shape[0], S, D); dist.recv(buf, rank - 1, tag=2 * i); xin = buf.requires_grad_(); run = lambda: blocks_fwd(xin)
        def blocks_fwd(x):
            for b in blocks: x = b(x)
            if last: return F.cross_entropy(full.h(full.f(x)).reshape(-1, lm.V), mbs[i][:, 1:].reshape(-1)) / m
            return x
        out = timed(run); acts[i] = (xin, out); peak[0] = max(peak[0], len(acts))
        if last: loss_sum[0] += out.item()
        else: o = out.detach().contiguous(); keep.append(o); reqs.append(dist.isend(o, rank + 1, tag=2 * i))
    def bwd(i):
        xin, out = acts.pop(i)
        if last: timed(lambda: out.backward())
        else: g = torch.empty_like(out); dist.recv(g, rank + 1, tag=2 * i + 1); timed(lambda: out.backward(g))
        if not first: gi = xin.grad.contiguous(); keep.append(gi); reqs.append(dist.isend(gi, rank - 1, tag=2 * i + 1))
    if schedule == "gpipe":
        for i in range(m): fwd(i)
        for i in reversed(range(m)): bwd(i)
    else:
        w = min(p - rank - 1, m)
        for i in range(w): fwd(i)
        for i in range(m - w): fwd(w + i); bwd(i)
        for i in range(m - w, m): bwd(i)
    for r in reqs: r.wait()
    return busy[0], peak[0], loss_sum[0], blocks
def worker(rank, p, port, outfile):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)); torch.set_num_threads(1); dist.init_process_group("gloo", rank=rank, world_size=p); res = []
    torch.manual_seed(1); full = lm.GPT(D, H, L); batch = torch.randint(0, lm.V, (BG, S + 1), generator=torch.Generator().manual_seed(7))
    ref = lm.GPT(D, H, L); ref.load_state_dict(full.state_dict()); lr_ = F.cross_entropy(ref(batch[:, :-1]).reshape(-1, lm.V), batch[:, 1:].reshape(-1)); lr_.backward(); bs = L // p; refg = [q.grad.clone() for b in ref.b[rank * bs:(rank + 1) * bs] for q in b.parameters()]
    for schedule in ("gpipe", "1f1b"):
        for m in (1, 2, 4, 8, 16):
            walls, idles, peaks = [], [], []
            for rep in range(4):
                full.zero_grad(set_to_none=True); dist.barrier(); t0 = time.perf_counter(); busy, peak, ls, blocks = one_step(full, rank, p, schedule, m, batch); dist.barrier(); wall = time.perf_counter() - t0
                allb = [None] * p; dist.all_gather_object(allb, busy); allp = [None] * p; dist.all_gather_object(allp, peak)
                if rep > 0: walls.append(wall); idles.append(1 - sum(allb) / p / wall); peaks.append(max(allp))
            gd = max((q.grad - rg).abs().max().item() for q, rg in zip([q for b in blocks for q in b.parameters()], refg)); gds = [None] * p; dist.all_gather_object(gds, gd)
            if rank == 0: res.append({"p": p, "schedule": schedule, "m": m, "wall": st.median(walls), "idle": st.median(idles), "formula_idle": (p - 1) / (m + p - 1), "peak_inflight": max(peaks), "grad_maxdiff": max(gds), "busy_mean": sum(allb) / p})
    if rank == 0: json.dump(res, open(outfile, "w"))
    dist.destroy_process_group()
if __name__ == "__main__":
    p = int(sys.argv[1]); mp.spawn(worker, args=(p, 30000 + p, f"g5_pipe_{p}.json"), nprocs=p)
```

Combined layout:

```python
import os, sys, json, time, copy, torch, torch.distributed as dist, torch.multiprocessing as mp, torch.nn.functional as F, lm, g4_run
D, H, L, S, BG, T_, P_, D_, M = 384, 8, 8, 128, 16, 2, 2, 2, 4
CN = {"tp_calls": 0, "tp_bytes": 0, "p2p_calls": 0, "p2p_bytes": 0, "dp_calls": 0, "dp_bytes": 0}; TPG = [None]
def tp_allreduce(x):
    CN["tp_calls"] += 1; CN["tp_bytes"] += x.numel() * 4; x = x.contiguous(); dist.all_reduce(x, group=TPG[0]); return x
g4_run.allreduce = tp_allreduce
def worker(rank, port, outfile):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)); torch.set_num_threads(1); dist.init_process_group("gloo", rank=rank, world_size=8)
    di, pi, ti = rank // (P_ * T_), (rank // T_) % P_, rank % T_
    tpg = {(a, b): dist.new_group([a * 4 + b * 2 + k for k in range(T_)]) for a in range(D_) for b in range(P_)}; dpg = {(b, k): dist.new_group([a * 4 + b * 2 + k for a in range(D_)]) for b in range(P_) for k in range(T_)}; TPG[0] = tpg[(di, pi)]; DPG = dpg[(pi, ti)]
    torch.manual_seed(1); full = lm.GPT(D, H, L); batch = torch.randint(0, lm.V, (BG, S + 1), generator=torch.Generator().manual_seed(7))
    ref = lm.GPT(D, H, L); ref.load_state_dict(full.state_dict()); lref = F.cross_entropy(ref(batch[:, :-1]).reshape(-1, lm.V), batch[:, 1:].reshape(-1)); lref.backward()
    per = L // P_; layers = range(pi * per, (pi + 1) * per); blocks = [g4_run.TPBlock(full.b[i], ti, T_) for i in layers]; first, last = pi == 0, pi == P_ - 1
    e, pe = (copy.deepcopy(full.e), copy.deepcopy(full.p)) if first else (None, None); f_, h_ = (copy.deepcopy(full.f), copy.deepcopy(full.h)) if last else (None, None)
    params = [q for b in blocks for q in b.parameters()] + [q for mod in (e, pe, f_, h_) if mod is not None for q in mod.parameters()]
    mine = batch[di * (BG // D_):(di + 1) * (BG // D_)]; mbs = mine.chunk(M); prev, nxt = rank - T_, rank + T_; acts = {}; reqs = []; keep = []; loss_sum = [0.0]
    def blocks_fwd(x, i):
        for b in blocks: x = b(x)
        if last: return F.cross_entropy(h_(f_(x)).reshape(-1, lm.V), mbs[i][:, 1:].reshape(-1)) / M
        return x
    def fwd(i):
        if first: ids = mbs[i][:, :-1]; xin = None; out = blocks_fwd(e(ids) + pe(torch.arange(ids.shape[1])), i)
        else: buf = torch.empty(mbs[i].shape[0], S, D); dist.recv(buf, prev, tag=2 * i); xin = buf.requires_grad_(); out = blocks_fwd(xin, i)
        acts[i] = (xin, out)
        if last: loss_sum[0] += out.item()
        else: o = out.detach().contiguous(); keep.append(o); CN["p2p_calls"] += 1; CN["p2p_bytes"] += o.numel() * 4; reqs.append(dist.isend(o, nxt, tag=2 * i))
    def bwd(i):
        xin, out = acts.pop(i)
        if last: out.backward()
        else: g = torch.empty_like(out); dist.recv(g, nxt, tag=2 * i + 1); out.backward(g)
        if not first: gi = xin.grad.contiguous(); keep.append(gi); CN["p2p_calls"] += 1; CN["p2p_bytes"] += gi.numel() * 4; reqs.append(dist.isend(gi, prev, tag=2 * i + 1))
    w = min(P_ - pi - 1, M)
    for i in range(w): fwd(i)
    for i in range(M - w): fwd(w + i); bwd(i)
    for i in range(M - w, M): bwd(i)
    for r in reqs: r.wait()
    flat = torch.cat([q.grad.flatten() for q in params]); CN["dp_calls"] += 1; CN["dp_bytes"] += flat.numel() * 4; dist.all_reduce(flat, group=DPG); flat /= D_; off = 0
    for q in params: q.grad = flat[off:off + q.numel()].view_as(q).clone(); off += q.numel()
    gd = 0.0
    for b, i in zip(blocks, layers):
        for (q, _), (rp, kind) in zip(b.pairs, g4_run.kinds(ref.b[i])): gd = max(gd, (q.grad - g4_run.shard(rp.grad, kind, ti, T_)).abs().max().item())
    for mod, rmod in ((e, ref.e), (pe, ref.p), (f_, ref.f), (h_, ref.h)):
        if mod is not None:
            for a, b in zip(mod.parameters(), rmod.parameters()): gd = max(gd, (a.grad - b.grad).abs().max().item())
    out = {"rank": rank, "di": di, "pi": pi, "ti": ti, "params": sum(q.numel() for q in params), "grad_maxdiff": gd, "loss_sum": loss_sum[0], "counts": dict(CN)}
    allo = [None] * 8; dist.all_gather_object(allo, out)
    if rank == 0: json.dump({"ranks": allo, "ref_loss": lref.item(), "full_params": sum(q.numel() for q in full.parameters())}, open(outfile, "w"))
    dist.destroy_process_group()
if __name__ == "__main__": mp.spawn(worker, args=(30100, "g5_3d.json"), nprocs=8)
```

Statistics output:

```
A. pipeline: idle share of device time (1 - mean busy compute / wall), global batch 16 x 128 tokens, 8 layers split over p ranks; median of 3 steps
  p=2 ranks, 4 layers per rank; formula (p-1)/(m+p-1)
    gpipe m= 1: wall   1648 ms, idle 0.510 (formula 0.500), peak microbatches held  1, gradient max diff vs one process 0.0e+00
    gpipe m= 2: wall   1243 ms, idle 0.341 (formula 0.333), peak microbatches held  2, gradient max diff vs one process 9.9e-10
    gpipe m= 4: wall   1066 ms, idle 0.217 (formula 0.200), peak microbatches held  4, gradient max diff vs one process 9.9e-10
    gpipe m= 8: wall   1043 ms, idle 0.142 (formula 0.111), peak microbatches held  8, gradient max diff vs one process 9.9e-10
    gpipe m=16: wall   1096 ms, idle 0.104 (formula 0.059), peak microbatches held 16, gradient max diff vs one process 1.0e-09
     1f1b m= 1: wall   1613 ms, idle 0.509 (formula 0.500), peak microbatches held  1, gradient max diff vs one process 0.0e+00
     1f1b m= 2: wall   1233 ms, idle 0.337 (formula 0.333), peak microbatches held  2, gradient max diff vs one process 9.9e-10
     1f1b m= 4: wall   1060 ms, idle 0.212 (formula 0.200), peak microbatches held  2, gradient max diff vs one process 9.9e-10
     1f1b m= 8: wall   1021 ms, idle 0.135 (formula 0.111), peak microbatches held  2, gradient max diff vs one process 9.9e-10
     1f1b m=16: wall   1068 ms, idle 0.089 (formula 0.059), peak microbatches held  2, gradient max diff vs one process 1.0e-09
  p=4 ranks, 2 layers per rank; formula (p-1)/(m+p-1)
    gpipe m= 1: wall   1704 ms, idle 0.763 (formula 0.750), peak microbatches held  1, gradient max diff vs one process 0.0e+00
    gpipe m= 2: wall   1125 ms, idle 0.623 (formula 0.600), peak microbatches held  2, gradient max diff vs one process 9.9e-10
    gpipe m= 4: wall    849 ms, idle 0.463 (formula 0.429), peak microbatches held  4, gradient max diff vs one process 9.9e-10
    gpipe m= 8: wall    774 ms, idle 0.317 (formula 0.273), peak microbatches held  8, gradient max diff vs one process 9.9e-10
    gpipe m=16: wall    856 ms, idle 0.224 (formula 0.158), peak microbatches held 16, gradient max diff vs one process 1.0e-09
     1f1b m= 1: wall   1676 ms, idle 0.764 (formula 0.750), peak microbatches held  1, gradient max diff vs one process 0.0e+00
     1f1b m= 2: wall   1105 ms, idle 0.621 (formula 0.600), peak microbatches held  2, gradient max diff vs one process 9.9e-10
     1f1b m= 4: wall    834 ms, idle 0.447 (formula 0.429), peak microbatches held  4, gradient max diff vs one process 9.9e-10
     1f1b m= 8: wall    794 ms, idle 0.307 (formula 0.273), peak microbatches held  4, gradient max diff vs one process 9.9e-10
     1f1b m=16: wall    807 ms, idle 0.219 (formula 0.158), peak microbatches held  4, gradient max diff vs one process 1.0e-09
  p=8 ranks, 1 layers per rank; formula (p-1)/(m+p-1)
    gpipe m= 1: wall   1771 ms, idle 0.889 (formula 0.875), peak microbatches held  1, gradient max diff vs one process 0.0e+00
    gpipe m= 2: wall   1107 ms, idle 0.808 (formula 0.778), peak microbatches held  2, gradient max diff vs one process 9.9e-10
    gpipe m= 4: wall    779 ms, idle 0.692 (formula 0.636), peak microbatches held  4, gradient max diff vs one process 9.9e-10
    gpipe m= 8: wall    707 ms, idle 0.528 (formula 0.467), peak microbatches held  8, gradient max diff vs one process 9.9e-10
    gpipe m=16: wall    795 ms, idle 0.362 (formula 0.304), peak microbatches held 16, gradient max diff vs one process 1.0e-09
     1f1b m= 1: wall   1817 ms, idle 0.889 (formula 0.875), peak microbatches held  1, gradient max diff vs one process 0.0e+00
     1f1b m= 2: wall   1086 ms, idle 0.803 (formula 0.778), peak microbatches held  2, gradient max diff vs one process 9.9e-10
     1f1b m= 4: wall    757 ms, idle 0.677 (formula 0.636), peak microbatches held  4, gradient max diff vs one process 9.9e-10
     1f1b m= 8: wall    687 ms, idle 0.509 (formula 0.467), peak microbatches held  8, gradient max diff vs one process 9.9e-10
     1f1b m=16: wall    778 ms, idle 0.355 (formula 0.304), peak microbatches held  8, gradient max diff vs one process 1.0e-09

B. best microbatch count by wall time (1F1B): p=2: m=8 (1021 ms), p=4: m=8 (794 ms), p=8: m=8 (687 ms)

C. combined layout on 8 processes: tensor 2 x pipeline 2 x data 2; 1F1B with 4 microbatches of 2 x 128 tokens per replica; global batch 16
  parameters per rank: stage 0 3,651,840, stage 1 3,603,584 of 14,344,064 (ideal 1/4 = 3,586,016; replicated embeddings or head add the rest)
  loss: mean over the two replicas 5.02417, one process on the whole batch 5.02418
  gradient max diff vs one process, over all 8 ranks: 3.5e-09
  per rank per step: 64 tensor-parallel all-reduces of 393,216 bytes; 4 pipeline sends of 393,216 bytes; 1 data-parallel all-reduce of 14,607,360 bytes
  formula: tensor-parallel calls 4 x layers per stage (4) x microbatches (4) = 64; ring bytes sent per rank: tensor 25.2 MB, pipeline 1.6 MB, data 14.6 MB

D. arithmetic only (no measurement): per-GPU traffic per step for a 70B-like model on 128 GPUs, bf16, 1024 sequences of 4096 tokens per step, microbatch = 1 sequence, 1F1B, interior stage
   (t,p,d)    states GB/GPU  tensor GB   pipeline GB  data GB   bubble
  (8,2,8)         70.0      2405.2        17.2      15.3   0.008
  (4,4,8)         70.0      1030.8        17.2      15.3   0.023
  (2,8,8)         70.0       343.6        17.2      15.3   0.052
  (1,16,8)         70.0         0.0        17.2      15.3   0.105
  (8,4,4)         35.0      2405.2        34.4       6.6   0.012
  (4,8,4)         35.0      1030.8        34.4       6.6   0.027
  (8,8,2)         17.5      2405.2        68.7       2.2   0.013
   compute for one step at 6 x Psi x tokens = 1.76e+18 FLOPs; on 128 GPUs at 400 TFLOPs/s each that is 34.4 s
```

Next: [Part 6](/posts/dist-6-moe/) replaces the dense MLP with experts, introduces all-to-all communication, and measures the load imbalance that routing creates.

## References

- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965), 2018.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), 2021.
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019.
