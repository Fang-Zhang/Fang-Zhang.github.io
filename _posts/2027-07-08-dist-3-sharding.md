---
title: "Distributed LLMs, Measured (3/7): Sharding Cut Model-State Memory From 16 to 3.05 Bytes per Parameter at 8 Ranks, at 1.5 Times the Communication Only in the Last Stage"
date: 2027-07-08 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, distributed training, zero, fsdp, sharding, series]
description: "Part 3 of 7. ZeRO and FSDP sharding stages written by hand in torch.distributed: per-rank memory against the paper's formulas, collective counts, communication volume, and equivalence with single-process training."
---

## The idea

[Part 2](/posts/dist-2-data-parallel/) ended with every rank holding all 16 bytes per parameter. Sharding removes that replication in three steps. The accounting below follows the formulas of [ZeRO](https://arxiv.org/abs/1910.02054); [PyTorch FSDP](https://arxiv.org/abs/2304.11277) is the production implementation of the third step.

## What the sources say

- ZeRO's three stages partition the optimizer states (P_os), then also the gradients (P_os+g), then also the parameters (P_os+g+p) across the N data-parallel ranks. With mixed-precision Adam, the per-device model-state memory goes from (2 + 2 + K) Psi to (2 + 2) Psi + K Psi / N, then to 2 Psi + (2 + K) Psi / N, then to (2 + 2 + K) Psi / N, where K = 12.
- The communication analysis states that the first two stages keep the baseline volume of 2 Psi per step, and that the third stage costs 3 Psi, "1.5x compared to the baseline": the parameters are all-gathered during the forward pass, which adds Psi to the 2 Psi of gradient reduce-scatter and parameter all-gather.
- Table 1 of the paper lists, for a 7.5B-parameter model at 64-way parallelism, 31.4 GB (stage 1), 16.6 GB (stage 2) and 1.88 GB (stage 3) per device, against 120 GB for plain data parallelism.
- The FSDP paper describes a sharded data parallel design co-developed with PyTorch's tensor implementation, dispatcher and CUDA caching allocator, and reports performance comparable to DDP while supporting significantly larger models.

## Experiment

The four stages are implemented by hand on the 14,344,064-parameter GPT of Part 2 (8 blocks of d = 384), in fp32 with Adam (4 + 4 + 8 = 16 bytes per parameter), at 2, 4 and 8 ranks with gloo:

- Each transformer block is a unit whose parameters live in one flat buffer, padded to a multiple of N. A rank owns one N-th of each buffer. The 1.0 percent of parameters in the embeddings, final norm and head are replicated in every stage and synchronised with a plain all-reduce, so the formulas below are evaluated with that 1.0 percent at 16 bytes.
- **Stage 0**: all-reduce of the gradients, full Adam on every rank.
- **Stage 1**: after the backward pass each block's gradient is reduce-scattered, each rank runs Adam on its shard only, and the updated shards are all-gathered into the full parameters.
- **Stage 2**: as stage 1, but a hook on the last gradient of each block reduce-scatters it immediately and drops the full gradient.
- **Stage 3**: parameters are kept only as shards. Before each block's forward pass the block is all-gathered; after it the full buffer is freed by resizing its storage to zero; the block is all-gathered again before its backward pass and freed again after the reduce-scatter of its gradient.
- **Memory** is the peak, over the events of a step, of the bytes of all resident model-state buffers: full parameter buffers that are not freed, gradients, shards, and Adam moments. Activations and allocator overhead are excluded.
- **Communication** is the count and payload size of each collective call; per-rank bytes sent come from the ring formulas verified in Part 2 (N-1)/N x payload for all-gather and reduce-scatter, and 2(N-1)/N x payload for all-reduce.
- **Equivalence**: 4 Adam steps on 2 x N sequences of 128 tokens, compared with a single process taking the same steps on the whole batch.

Results:

- **Memory matches the formulas for stage 1 and sits above them for stages 2 and 3.** Stage 0 was 16.00 bytes per parameter at all N. Stage 1 was 12.04, 10.06 and 9.07 at N = 2, 4 and 8, equal to the formula to two decimals. Stage 2 was 10.28, 7.44 and 6.01 against formulas of 10.06, 7.09 and 5.61. Stage 3 was 8.80, 4.96 and 3.05 against formulas of 8.08, 4.12 and 2.14.
- **Collective counts per step.** Stage 0: 9 all-reduces (8 blocks and the replicated part). Stages 1 and 2: 8 all-gathers, 8 reduce-scatters and 1 all-reduce. Stage 3: 16 all-gathers (one forward and one backward per block), 8 reduce-scatters and 1 all-reduce.
- **Communication.** Stages 0 to 2 sent 57.4, 86.1 and 100.4 MB per rank per step at N = 2, 4 and 8, the ring figure 2(N-1)/N x 4 Psi. Stage 3 sent 1.495 times as much (85.8, 128.7 and 150.1 MB), the ratio being 1.5 minus the share of the replicated part.
- **Equivalence.** All four stages ended within 9.8e-06, 2.0e-05 and 9.2e-06 of the single-process parameters at N = 2, 4 and 8 (largest parameter about 4.5), and the four stages gave the same difference to three digits.
- **Step time.** Medians at N = 2, 4 and 8: stage 0 860, 1,316 and 3,663 ms; stage 1 1,081, 1,625 and 3,755 ms; stage 2 1,085, 1,602 and 3,487 ms; stage 3 1,442, 2,009 and 4,495 ms.

![Three panels over 2, 4 and 8 ranks for four sharding stages. Left: peak model-state bytes per parameter, measured bars close to formula lines, for example 16, 9.07, 6.01 and 3.05 at 8 ranks. Middle: bytes sent per rank per step relative to plain data parallelism, 1.0 for stages 0 to 2 and 1.5 for stage 3. Right: median step time, growing with stage and with rank count, to about 4.5 seconds at 8 ranks for stage 3]({{ site.baseurl }}/assets/pic/dist-3-sharding.png){: width="900" }
_Figure: left, measured peak against the formula (black lines); middle, bytes sent per rank relative to stage 0; right, median step time._

## Engineering reading

These points are this series' reading of the measurement.

1. **The memory formulas hold; the interesting part is the residual.** Stage 1 matches to the digit because nothing is transient: the full parameters and gradients are all resident, and only the Adam moments are sharded. Stages 2 and 3 exceed the formula by 0.2 to 0.4 and 0.7 to 0.9 bytes per parameter. The size matches what the schedule keeps in flight, about one block's full gradient in stage 2 (a block is 12.5 percent of the parameters, 0.5 bytes per parameter in fp32) and about one block's full parameters plus its gradient in stage 3, though this attribution was not isolated. Here the stage-3 residual is 0.9 bytes per parameter, 30 percent of the 3.05 measured at N = 8. The finer the units, the smaller the transient, and the more collectives.
2. **Stage 3 is the only stage that changes the communication volume, and it does so by exactly the paper's factor.** The extra Psi is the second parameter all-gather per step (one in the forward pass and one in the backward pass; no all-gather is needed after the update). That is the price of not holding parameters between uses; it can only be hidden by overlapping with compute, since it cannot be removed without keeping the parameters resident.
3. **Stages 1 and 2 are free in bytes and not free in calls.** The bytes are identical to stage 0, but the 9 collectives became 17 in these stages and 25 in stage 3. Every extra call pays the latency term measured in Part 2, so the unit size is a real tuning knob: units that are too small are latency-bound, units that are too large make the transient memory large.
4. **The time numbers show the cost of doing this without overlap.** Stage 1 was 1.0 to 1.3 times stage 0, and stage 3 was 1.2 to 1.7 times stage 0. This implementation issues every collective synchronously, so none of the extra traffic is hidden behind compute; a production implementation, which the FSDP paper describes as co-designed with the allocator, schedules the all-gathers ahead of use. These times mostly reflect one contended host, as in Part 2, and the 8-rank rows are dominated by that contention.
5. **Sharding does not change the arithmetic.** All four stages produced the same parameters as the single process, to a difference consistent with fp32 summation order amplified by Adam over 4 steps. The memory savings come at no cost in what is computed.

## Limits

- Everything is fp32 with replicated embeddings and head, not the mixed-precision 2 + 2 + 12 split of the paper; the formulas were evaluated with 4 + 4 + 8 and the measured resident bytes agree with them, but the 16-bit weight copies of mixed precision were not modelled.
- The memory measure counts model-state buffers at the events of a step. It does not include activations, communication buffers or allocator fragmentation, all of which a real peak includes.
- The bytes sent are the ring formula applied to measured payloads, not counted on the wire; the formula was checked against hand-written sends in Part 2.
- Times are medians of 3 steps per cell on a single shared host and say little about GPUs. No prefetching, overlap, or mixed-precision communication was implemented.
- Only parameters in transformer blocks are sharded, so the stage-3 figures at large N are bounded below by the replicated 1.0 percent at 16 bytes (0.16 bytes per parameter).

## Code and output

```python
import os, sys, json, time, math, statistics as st, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp, torch.nn.functional as F, lm
D, H, L, PER, S, STEPS = 384, 8, 8, 2, 128, 4
class Unit:
    pass
class Run:
    def __init__(self, stage, rank, ws):
        self.stage, self.r, self.N = stage, rank, ws; torch.manual_seed(1); self.m = lm.GPT(D, H, L); self.peak = 0; self.comm = {"ag": 0, "rs": 0, "ar": 0}; self.calls = {"ag": 0, "rs": 0, "ar": 0}
        self.rest = [p for n, p in self.m.named_parameters() if not n.startswith("b.")]; self.units = []
        for blk in self.m.b:
            u = Unit(); u.blk = blk; u.ps = list(blk.parameters()); n = sum(p.numel() for p in u.ps); u.P = math.ceil(n / ws) * ws; u.n = n; u.full = torch.zeros(u.P); off = 0
            for p in u.ps: k = p.numel(); u.full[off:off + k] = p.data.flatten(); p.data = u.full[off:off + k].view_as(p.data); off += k
            a, b = rank * u.P // ws, (rank + 1) * u.P // ws
            if stage >= 1: u.shard = nn.Parameter(u.full[a:b].clone() if stage == 3 else u.full[a:b]); u.own = stage == 3
            self.units.append(u)
        if stage == 0: self.opt = torch.optim.Adam(list(self.m.parameters()), lr=1e-3)
        else: self.opt = torch.optim.Adam([u.shard for u in self.units] + self.rest, lr=1e-3)
        if stage == 3:
            for u in self.units: self.free(u)
        for u in self.units: self.hooks(u)
    # --- collectives with payload accounting (payload = full-size bytes of the collective)
    def ag(self, full, shard): self.comm["ag"] += full.numel() * 4; self.calls["ag"] += 1; dist.all_gather_into_tensor(full, shard)
    def rs(self, out, inp): self.comm["rs"] += inp.numel() * 4; self.calls["rs"] += 1; dist.reduce_scatter_tensor(out, inp)
    def ar(self, t): self.comm["ar"] += t.numel() * 4; self.calls["ar"] += 1; dist.all_reduce(t)
    def gather(self, u):
        u.full.untyped_storage().resize_(u.P * 4); self.ag(u.full, u.shard.data); self.tick()
    def free(self, u): u.full.untyped_storage().resize_(0)
    def resident(self):
        b = 0
        for u in self.units:
            b += u.full.untyped_storage().nbytes()
            for p in u.ps:
                if p.grad is not None: b += p.grad.numel() * 4
            if self.stage >= 1:
                if u.own: b += u.shard.numel() * 4
                if u.shard.grad is not None: b += u.shard.grad.numel() * 4
        for p in self.rest: b += p.numel() * 4 + (p.grad.numel() * 4 if p.grad is not None else 0)
        for s in self.opt.state.values():
            for t in s.values():
                if torch.is_tensor(t) and t.dim() > 0: b += t.numel() * 4
        return b
    def tick(self): self.peak = max(self.peak, self.resident())
    def flat_grad(self, u):
        g = torch.cat([p.grad.flatten() for p in u.ps] + ([u.full.new_zeros(u.P - u.n)] if u.P > u.n else [])); return g
    def reduce_unit(self, u):
        g = self.flat_grad(u)
        for p in u.ps: p.grad = None
        out = torch.empty(u.P // self.N); self.rs(out, g); del g; out /= self.N; u.shard.grad = out; self.tick()
    def hooks(self, u):
        if self.stage == 3:
            u.blk.register_forward_pre_hook(lambda mod, a, u=u: self.gather(u)); u.blk.register_forward_hook(lambda mod, a, o, u=u: self.free(u))
            u.blk.register_full_backward_pre_hook(lambda mod, g, u=u: self.gather(u))
        if self.stage >= 2:
            u.ready = 0
            def h(p, u=u):
                u.ready += 1; self.tick()
                if u.ready == len(u.ps):
                    u.ready = 0; self.reduce_unit(u)
                    if self.stage == 3: self.free(u)
            for p in u.ps: p.register_post_accumulate_grad_hook(h)
    def step(self, batch):
        self.opt.zero_grad(set_to_none=True)
        for u in self.units:
            for p in u.ps: p.grad = None
        loss = F.cross_entropy(self.m(batch[:, :-1]).reshape(-1, lm.V), batch[:, 1:].reshape(-1)); self.tick(); loss.backward(); self.tick()
        if self.stage == 0:
            for u in self.units:
                g = self.flat_grad(u); self.ar(g); g /= self.N; off = 0
                for p in u.ps: p.grad = g[off:off + p.numel()].view_as(p.data).clone(); off += p.numel()
        elif self.stage == 1:
            for u in self.units: self.reduce_unit(u)
        g = torch.cat([p.grad.flatten() for p in self.rest]); self.ar(g); g /= self.N; off = 0
        for p in self.rest: p.grad = g[off:off + p.numel()].view_as(p.data).clone(); off += p.numel()
        self.tick(); self.opt.step(); self.tick()
        if self.stage in (1, 2):
            for u in self.units: self.ag(u.full, u.shard.data.clone())
        self.tick()
    def params(self):
        out = []
        for u in self.units:
            if self.stage == 3: u.full.untyped_storage().resize_(u.P * 4); self.ag(u.full, u.shard.data)
            out.append(u.full[:u.n].clone())
            if self.stage == 3: self.free(u)
        return torch.cat(out + [p.data.flatten() for p in self.rest])
def data(step, ws):
    g = torch.Generator().manual_seed(100 + step); return torch.randint(0, lm.V, (ws * PER, S + 1), generator=g)
def reference(ws):
    torch.manual_seed(1); m = lm.GPT(D, H, L); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for s in range(STEPS):
        b = data(s, ws); opt.zero_grad(); F.cross_entropy(m(b[:, :-1]).reshape(-1, lm.V), b[:, 1:].reshape(-1)).backward(); opt.step()
    return torch.cat([p.data.flatten() for n, p in m.named_parameters() if n.startswith("b.")] + [p.data.flatten() for n, p in m.named_parameters() if not n.startswith("b.")])
def worker(rank, ws, port, outfile):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)); torch.set_num_threads(1); dist.init_process_group("gloo", rank=rank, world_size=ws)
    ref = reference(ws) if rank == 0 else None; res = []
    for stage in (0, 1, 2, 3):
        run = Run(stage, rank, ws); psi_total = sum(p.numel() for p in run.m.parameters()); psi_blocks = sum(u.n for u in run.units); ts = []; run.tick(); base = run.peak
        for s in range(STEPS):
            b = data(s, ws)[rank * PER:(rank + 1) * PER]; dist.barrier(); t = time.perf_counter(); run.step(b); dt = time.perf_counter() - t
            if s > 0: ts.append(dt)
            if s == 0: run.comm0 = dict(run.comm); run.calls0 = dict(run.calls)
        N = ws; sent = lambda c: (N - 1) / N * (c["ag"] + c["rs"]) + 2 * (N - 1) / N * c["ar"]; per1 = {k: (run.comm[k] - run.comm0[k]) / (STEPS - 1) for k in run.comm}; calls1 = {k: (run.calls[k] - run.calls0[k]) / (STEPS - 1) for k in run.calls}
        p = run.params()
        if rank == 0: res.append({"ws": ws, "stage": stage, "psi_total": psi_total, "psi_blocks": psi_blocks, "peak_state_bytes": run.peak, "peak_per_param": run.peak / psi_total, "step_s": st.median(ts), "payload_per_step": per1, "sent_per_step": sent(per1), "calls_per_step": calls1, "maxdiff_vs_single_process": (p - ref).abs().max().item(), "max_abs_param": ref.abs().max().item()})
        del run; dist.barrier()
    if rank == 0: json.dump(res, open(outfile, "w"))
    dist.destroy_process_group()
if __name__ == "__main__":
    ws = int(sys.argv[1]); mp.spawn(worker, args=(ws, 29800 + ws, f"g3_{ws}.json"), nprocs=ws)
```

Statistics output:

```
model: 14,344,064 parameters, of which 14,195,712 (99.0%) in the 8 transformer blocks that are sharded; embeddings, final norm and head stay replicated
fp32 Adam, 16 bytes per parameter = 4 weights + 4 gradients + 8 moments; formula per sharded parameter: stage 0: 16; 1: 4+4+8/N; 2: 4+(4+8)/N; 3: (4+4+8)/N; replicated part always 16

A. peak model-state bytes per parameter during a step (weights, gradients, optimizer states; activations excluded)
  N=2  stage 0 (plain DP)                   measured  16.00  formula  16.00  extra +0.00  (  229.5 MB)
  N=2  stage 1 (optimizer states sharded)   measured  12.04  formula  12.04  extra +0.00  (  172.7 MB)
  N=2  stage 2 (+ gradients sharded)        measured  10.28  formula  10.06  extra +0.22  (  147.5 MB)
  N=2  stage 3 (+ parameters sharded)       measured   8.80  formula   8.08  extra +0.71  (  126.2 MB)
  N=4  stage 0 (plain DP)                   measured  16.00  formula  16.00  extra +0.00  (  229.5 MB)
  N=4  stage 1 (optimizer states sharded)   measured  10.06  formula  10.06  extra +0.00  (  144.3 MB)
  N=4  stage 2 (+ gradients sharded)        measured   7.44  formula   7.09  extra +0.34  (  106.7 MB)
  N=4  stage 3 (+ parameters sharded)       measured   4.96  formula   4.12  extra +0.84  (   71.2 MB)
  N=8  stage 0 (plain DP)                   measured  16.00  formula  16.00  extra +0.00  (  229.5 MB)
  N=8  stage 1 (optimizer states sharded)   measured   9.07  formula   9.07  extra -0.00  (  130.1 MB)
  N=8  stage 2 (+ gradients sharded)        measured   6.01  formula   5.61  extra +0.41  (   86.3 MB)
  N=8  stage 3 (+ parameters sharded)       measured   3.05  formula   2.14  extra +0.90  (   43.7 MB)

B. communication per step: calls and bytes sent per rank (ring formula applied to the measured collective payloads), as a multiple of the 4*Psi gradient size
  N=2  stage 0 (plain DP)                   all-gather    0, reduce-scatter    0, all-reduce    9 calls; sent    57.4 MB = 1.000 x stage 0   (ideal 2(N-1)/N x 4Psi = 57.4 MB)
  N=2  stage 1 (optimizer states sharded)   all-gather    8, reduce-scatter    8, all-reduce    1 calls; sent    57.4 MB = 1.000 x stage 0   (ideal 2(N-1)/N x 4Psi = 57.4 MB)
  N=2  stage 2 (+ gradients sharded)        all-gather    8, reduce-scatter    8, all-reduce    1 calls; sent    57.4 MB = 1.000 x stage 0   (ideal 2(N-1)/N x 4Psi = 57.4 MB)
  N=2  stage 3 (+ parameters sharded)       all-gather   16, reduce-scatter    8, all-reduce    1 calls; sent    85.8 MB = 1.495 x stage 0   (ideal 2(N-1)/N x 4Psi = 57.4 MB)
  N=4  stage 0 (plain DP)                   all-gather    0, reduce-scatter    0, all-reduce    9 calls; sent    86.1 MB = 1.000 x stage 0   (ideal 2(N-1)/N x 4Psi = 86.1 MB)
  N=4  stage 1 (optimizer states sharded)   all-gather    8, reduce-scatter    8, all-reduce    1 calls; sent    86.1 MB = 1.000 x stage 0   (ideal 2(N-1)/N x 4Psi = 86.1 MB)
  N=4  stage 2 (+ gradients sharded)        all-gather    8, reduce-scatter    8, all-reduce    1 calls; sent    86.1 MB = 1.000 x stage 0   (ideal 2(N-1)/N x 4Psi = 86.1 MB)
  N=4  stage 3 (+ parameters sharded)       all-gather   16, reduce-scatter    8, all-reduce    1 calls; sent   128.7 MB = 1.495 x stage 0   (ideal 2(N-1)/N x 4Psi = 86.1 MB)
  N=8  stage 0 (plain DP)                   all-gather    0, reduce-scatter    0, all-reduce    9 calls; sent   100.4 MB = 1.000 x stage 0   (ideal 2(N-1)/N x 4Psi = 100.4 MB)
  N=8  stage 1 (optimizer states sharded)   all-gather    8, reduce-scatter    8, all-reduce    1 calls; sent   100.4 MB = 1.000 x stage 0   (ideal 2(N-1)/N x 4Psi = 100.4 MB)
  N=8  stage 2 (+ gradients sharded)        all-gather    8, reduce-scatter    8, all-reduce    1 calls; sent   100.4 MB = 1.000 x stage 0   (ideal 2(N-1)/N x 4Psi = 100.4 MB)
  N=8  stage 3 (+ parameters sharded)       all-gather   16, reduce-scatter    8, all-reduce    1 calls; sent   150.1 MB = 1.495 x stage 0   (ideal 2(N-1)/N x 4Psi = 100.4 MB)

C. final parameters after 4 Adam steps vs a single process on the whole batch
  N=2: max |difference| per stage 9.78e-06, 9.78e-06, 9.78e-06, 9.78e-06   (largest parameter 4.46)
  N=4: max |difference| per stage 1.99e-05, 1.99e-05, 1.99e-05, 1.99e-05   (largest parameter 4.46)
  N=8: max |difference| per stage 9.19e-06, 9.19e-06, 9.19e-06, 9.19e-06   (largest parameter 4.45)

D. median step time (ms), per-rank batch 2 x 128 tokens
  N=2: stage 0    860, stage 1   1081, stage 2   1085, stage 3   1442
  N=4: stage 0   1316, stage 1   1625, stage 2   1602, stage 3   2009
  N=8: stage 0   3663, stage 1   3755, stage 2   3487, stage 3   4495
```

Next: [Part 4](/posts/dist-4-tensor-parallel/) splits each layer's matrices across ranks (tensor parallelism), which trades the parameter all-gathers for activation all-reduces inside every layer.

## References

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), 2019.
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277), 2023.
