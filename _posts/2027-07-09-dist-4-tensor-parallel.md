---
title: "Distributed LLMs, Measured (4/7): Tensor Parallelism Matched the Unsplit Model to 1e-6 With Exactly 4 All-Reduces per Layer, and Communication Took 83 Percent of the Time at 8 Ways"
date: 2027-07-09 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, distributed training, tensor parallel, megatron, series]
description: "Part 4 of 7. Megatron-style tensor parallelism written by hand: column and row splits of attention and MLP, output and gradient equivalence at 2, 4 and 8 ways, all-reduce counts and sizes, and the compute versus communication split."
---

## From sharding states to splitting layers

[Part 3](/posts/dist-3-sharding/) sharded the states but still ran every layer on one rank at a time and gathered its weights on demand. Tensor parallelism instead splits each layer's matrices across ranks, so that every rank computes a slice of every layer and the weights never move; what moves is activations.

## What the sources say

- [Megatron-LM](https://arxiv.org/abs/1909.08053) splits the first GEMM of the MLP block by columns, so the GeLU nonlinearity can be applied independently to each partition, and the second GEMM by rows, so it consumes the GeLU output with no communication. The block then needs a single all-reduce in the forward pass (operator g) and a single all-reduce in the backward pass (operator f); f is the identity forward and an all-reduce backward, g the reverse.
- For self-attention the query, key and value GEMMs are split by columns so that each head is computed on one GPU, and the output projection is split by rows. In total "all GEMMs in a simple transformer layer" need "only two all-reduces in the forward path and two in the backward path".
- The reported scale: transformer models up to 8.3 billion parameters on 512 GPUs with 8-way model parallelism, 15.1 PetaFLOPs sustained and 76 percent scaling efficiency against a single-GPU baseline of 39 TeraFLOPs (30 percent of peak).

## Experiment

The 14,344,064-parameter GPT of Part 3 (d = 384, 8 heads, 8 layers) is split into t = 1, 2, 4 and 8 ranks with gloo, one process and one thread per rank, with the f and g operators written as custom autograd functions around `all_reduce`.

- **Attention**: each rank owns 8/t heads of the fused QKV projection (rows) and the matching columns of the output projection.
- **MLP**: each rank owns 4d/t rows of the first layer and the matching columns of the second.
- **Replicated**: layer norms, the biases of the two row-split layers (added once after the all-reduce), token and position embeddings, final norm and head.
- **Checks**: logits, loss and every gradient against the unsplit model built from the same weights on the same batch of 4 x 128 tokens; logits again after three Adam steps taken by both models on the same data; parameters held per rank; the number and size of all-reduce calls; forward plus backward time with the all-reduces on and with them switched off, which isolates the compute part (the result is then numerically wrong and is used for timing only).

Results:

- **Equivalence.** At t = 1, 2, 4 and 8 the maximum logit difference was 1.1e-06, 1.2e-06, 1.1e-06 and 1.2e-06 for logits up to 2.46, the loss difference was exactly 0, the maximum gradient difference was 7.5e-09, 7.5e-09, 6.5e-09 and 6.5e-09 for gradients up to 3.5e-03, and the logits after three Adam steps differed by 2.7e-06, 2.5e-06, 2.9e-06 and 3.5e-06. The 1-way row, which has no communication at all, shows the same difference size, so the differences are implementation arithmetic and not due to the splitting.
- **Parameters per rank.** 0.506, 0.259 and 0.135 of the full count at 2, 4 and 8 ways; for the blocks alone 0.501, 0.251 and 0.126, against ideals of 0.5, 0.25 and 0.125. At 16 bytes per parameter this is 116.1, 59.4 and 31.0 MB of model states per rank, against 229.5 MB.
- **Communication count and size.** Exactly 16 all-reduces in the forward pass and 16 in the backward pass, 32 in total, at every t, which is 4 per layer for 8 layers. Each carries 786,432 bytes, which is b x s x d x 4 for 4 x 128 tokens and d = 384. Per rank the ring sends 25.2, 37.7 and 44.0 MB at 2, 4 and 8 ways, which is 4L x 2(t-1)/t x b s d x 4.
- **Compared with data parallelism.** Data-parallel gradient synchronisation of the same blocks sends 56.8, 85.2 and 99.4 MB per step at 2, 4 and 8 ranks, so tensor-parallel traffic was 0.44 times as large at this batch, which equals the formula tokens / (3d) = 512 / 1,152.
- **Time.** Forward plus backward took 447, 471, 592 and 893 ms at 1, 2, 4 and 8 ways. With communication switched off it took 426, 220, 129 and 152 ms, a compute speedup of 1.93, 3.30 and 2.81 times at 2, 4 and 8 ways. The share of time added by the all-reduces was 53, 78 and 83 percent at 2, 4 and 8 ways (5 percent at 1-way, which is timing noise, since no communication happens).

![Three panels. Left: fraction of parameters held per rank for 1, 2, 4 and 8-way tensor parallelism: 1.00, 0.506, 0.259 and 0.135 for all parameters, and 1.00, 0.501, 0.251 and 0.126 for the transformer blocks, against an ideal of 1/N. Middle: forward plus backward time split into compute and added communication: compute falls from 426 to 152 ms while the total rises from 447 to 893 ms. Right: ratio of tensor-parallel to data-parallel bytes against tokens, a straight line on log axes crossing 1 at about 1,150 tokens for hidden size 384, with the measured point on the line]({{ site.baseurl }}/assets/pic/dist-4-tensor-parallel.png){: width="900" }
_Figure: left, parameters per rank against 1/N; middle, compute and added communication time; right, the formula for the ratio of tensor-parallel to data-parallel bytes against tokens, for two hidden sizes, with the measured point._

## Engineering reading

These points are this series' reading of the measurement and the sources.

1. **The communication pattern is a property of the layer, not of the size of the model.** The count of 32 does not change with t, and the message size depends on tokens and hidden size, not on the weights. Compared with Part 2, where traffic grows with parameters and is paid once per step, here traffic grows with tokens x hidden size x layers and is paid on every forward and backward pass. The break-even against the data-parallel gradient bytes is tokens = 3 x hidden size by the formula (1,152 tokens at d = 384, 49,152 at d = 16,384). That comparison omits two differences that matter in practice: data-parallel traffic is paid once per step and can overlap with the backward pass, while tensor-parallel traffic sits between a GEMM and the next dependent operation.
2. **Splitting columns first and rows second is what makes the count 4 per layer.** The paper's argument, which this experiment does not test by a counter-example, is that the GeLU needs a complete input row, so a row-split first GEMM would need a reduction before it; the column split lets each rank apply the nonlinearity to its own slice. The measured count of exactly 2 forward and 2 backward all-reduces per layer is the result of that pairing.
3. **Memory falls by 1/t for what is split and not at all for what is not.** Blocks fell exactly to 1/t, while the replicated embeddings, norms and head, 1.0 percent of the parameters, lifted the 8-way figure from 12.5 to 13.5 percent. The activations of the replicated operations are also replicated, which this part did not measure.
4. **Compute scales, communication does not, and the crossover is early here.** The compute part fell to 0.52 and 0.30 of the 1-way time at 2 and 4 ways, but the 32 all-reduces took 251, 463 and 741 ms at 2, 4 and 8 ways, rising with t. The 8-way compute time (152 ms) is above the 4-way time (129 ms), which is more likely contention among 8 processes on one host, as in Parts 2 and 3, than anything about tensor parallelism. On a cluster the all-reduce time depends on the bandwidth and latency of the link the group runs over, so the tensor-parallel degree has to be weighed against the interconnect; Part 5 looks at how the degrees are combined.
5. **The Megatron efficiency figure is not tested here.** The paper reports 76 percent scaling efficiency against a single-GPU baseline for models up to 8.3 billion parameters with 8-way model parallelism on 512 GPUs. A 14M-parameter model on a loopback network is a different regime, in which communication dominates; this result illustrates the per-layer communication count and does not measure Megatron's efficiency.

## Limits

- The ranks are processes on one CPU host communicating through loopback sockets, so the communication share (53 to 83 percent) is not transferable to GPU clusters; only the counts, sizes, memory fractions and equivalence results are.
- The model is small (d = 384), so the matrices per rank get small at 8 ways; efficiency at production sizes was not measured.
- Embeddings, norms and head are replicated, whereas the paper also splits the embedding GEMM; sequence-parallel variants that split the replicated parts are not covered.
- Timing is a median of 6 passes after 2 warmups in one launch per configuration, with no confidence interval; no overlap of communication with compute was attempted.
- The compute-only timing runs the same sharded GEMMs with the all-reduces skipped, so it excludes the cost of waiting for the slowest rank.

## Code and output

```python
import os, sys, json, time, statistics as st, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp, torch.nn.functional as F, lm
D, H, L, B, S = 384, 8, 8, 4, 128
CNT = {"calls": 0, "bytes": 0, "on": True}
def allreduce(x):
    if dist.get_world_size() == 1 or not CNT["on"]: return x
    CNT["calls"] += 1; CNT["bytes"] += x.numel() * 4; x = x.contiguous(); dist.all_reduce(x); return x
class Fop(torch.autograd.Function):   # identity forward, all-reduce backward
    @staticmethod
    def forward(ctx, x): return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return allreduce(g.clone())
class Gop(torch.autograd.Function):   # all-reduce forward, identity backward
    @staticmethod
    def forward(ctx, x): return allreduce(x.clone())
    @staticmethod
    def backward(ctx, g): return g
def shard(t, kind, r, n):
    """slice of a full tensor owned by rank r of n"""
    if kind == "rep": return t.detach().clone()
    if kind == "qkv_rows":   # rows laid out as (3, H, dh); rank owns H/n heads of each of q, k, v
        dh = D // H; hl = H // n; idx = torch.cat([torch.arange(c * D + (r * hl) * dh, c * D + ((r + 1) * hl) * dh) for c in range(3)]); return t.detach()[idx].clone()
    if kind == "o_cols": k = D // n; return t.detach()[:, r * k:(r + 1) * k].clone()
    if kind == "fc1_rows": k = 4 * D // n; return t.detach()[r * k:(r + 1) * k].clone()
    if kind == "fc1_bias": k = 4 * D // n; return t.detach()[r * k:(r + 1) * k].clone()
    if kind == "fc2_cols": k = 4 * D // n; return t.detach()[:, r * k:(r + 1) * k].clone()
    raise KeyError(kind)
def kinds(blk):
    return [(blk.l1.weight, "rep"), (blk.l1.bias, "rep"), (blk.qkv.weight, "qkv_rows"), (blk.qkv.bias, "qkv_rows"), (blk.o.weight, "o_cols"), (blk.o.bias, "rep"), (blk.l2.weight, "rep"), (blk.l2.bias, "rep"), (blk.m[0].weight, "fc1_rows"), (blk.m[0].bias, "fc1_bias"), (blk.m[2].weight, "fc2_cols"), (blk.m[2].bias, "rep")]
class TPBlock(nn.Module):
    def __init__(self, blk, r, n):
        super().__init__(); self.n, self.hl = n, H // n; P = lambda t, k: nn.Parameter(shard(t, k, r, n)); (l1w, _), (l1b, _), (qw, _), (qb, _), (ow, _), (ob, _), (l2w, _), (l2b, _), (w1, _), (b1, _), (w2, _), (b2, _) = kinds(blk)
        self.l1w, self.l1b, self.l2w, self.l2b = P(l1w, "rep"), P(l1b, "rep"), P(l2w, "rep"), P(l2b, "rep"); self.qw, self.qb = P(qw, "qkv_rows"), P(qb, "qkv_rows"); self.ow, self.ob = P(ow, "o_cols"), P(ob, "rep")
        self.w1, self.b1, self.w2, self.b2 = P(w1, "fc1_rows"), P(b1, "fc1_bias"), P(w2, "fc2_cols"), P(b2, "rep"); self.pairs = list(zip([self.l1w, self.l1b, self.qw, self.qb, self.ow, self.ob, self.l2w, self.l2b, self.w1, self.b1, self.w2, self.b2], kinds(blk)))
    def forward(self, x):
        B_, T, _ = x.shape; dh = D // H; h = Fop.apply(F.layer_norm(x, (D,), self.l1w, self.l1b)); q, k, v = F.linear(h, self.qw, self.qb).view(B_, T, 3, self.hl, dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B_, T, self.hl * dh); x = x + Gop.apply(F.linear(a, self.ow)) + self.ob
        h2 = Fop.apply(F.layer_norm(x, (D,), self.l2w, self.l2b)); return x + Gop.apply(F.linear(F.gelu(F.linear(h2, self.w1, self.b1)), self.w2)) + self.b2
class TPGPT(nn.Module):
    def __init__(self, full, r, n):
        super().__init__(); self.e, self.p, self.f, self.h = [__import__("copy").deepcopy(getattr(full, k)) for k in "epfh"]; self.b = nn.ModuleList(TPBlock(b, r, n) for b in full.b)
    def forward(self, ids):
        x = self.e(ids) + self.p(torch.arange(ids.shape[1]))
        for b in self.b: x = b(x)
        return self.h(self.f(x))
def lossf(m, b): return F.cross_entropy(m(b[:, :-1]).reshape(-1, lm.V), b[:, 1:].reshape(-1))
def worker(rank, n, port, outfile):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)); torch.set_num_threads(1); dist.init_process_group("gloo", rank=rank, world_size=n)
    torch.manual_seed(1); full = lm.GPT(D, H, L); tp = TPGPT(full, rank, n); res = {"n": n}
    g = torch.Generator().manual_seed(5); batch = torch.randint(0, lm.V, (B, S + 1), generator=g)
    with torch.no_grad(): res["logit_maxdiff"] = (full(batch[:, :-1]) - tp(batch[:, :-1])).abs().max().item(); res["logit_maxabs"] = full(batch[:, :-1]).abs().max().item()
    CNT.update(calls=0, bytes=0); la = lossf(tp, batch); fw_calls, fw_bytes = CNT["calls"], CNT["bytes"]; la.backward(); res.update(calls_fwd=fw_calls, calls_bwd=CNT["calls"] - fw_calls, calls_total=CNT["calls"], payload_bytes=CNT["bytes"], bytes_per_allreduce=CNT["bytes"] / CNT["calls"] if CNT["calls"] else 0)
    lb = lossf(full, batch); lb.backward(); res["loss_diff"] = abs(la.item() - lb.item()); gd = 0.0; gmax = 0.0
    for tb, fb in zip(tp.b, full.b):
        for tpp, (fp, kind) in tb.pairs: gd = max(gd, (tpp.grad - shard(fp.grad, kind, rank, n)).abs().max().item()); gmax = max(gmax, fp.grad.abs().max().item())
    for k in "epfh":
        for a, b in zip(getattr(tp, k).parameters(), getattr(full, k).parameters()): gd = max(gd, (a.grad - b.grad).abs().max().item())
    res.update(grad_maxdiff=gd, grad_maxabs=gmax)
    # three Adam steps each, then compare logits
    tp2 = TPGPT(full.__class__(D, H, L), rank, n) if False else None
    torch.manual_seed(1); f2 = lm.GPT(D, H, L); t2 = TPGPT(f2, rank, n); o1 = torch.optim.Adam(f2.parameters(), lr=1e-3); o2 = torch.optim.Adam(t2.parameters(), lr=1e-3)
    for s in range(3):
        bb = torch.randint(0, lm.V, (B, S + 1), generator=torch.Generator().manual_seed(50 + s)); o1.zero_grad(); lossf(f2, bb).backward(); o1.step(); o2.zero_grad(); lossf(t2, bb).backward(); o2.step()
    with torch.no_grad(): res["after3_logit_maxdiff"] = (f2(batch[:, :-1]) - t2(batch[:, :-1])).abs().max().item()
    res["params_per_rank"] = sum(p.numel() for p in tp.parameters()); res["params_full"] = sum(p.numel() for p in full.parameters()); res["block_params_per_rank"] = sum(p.numel() for p in tp.b.parameters()); res["block_params_full"] = sum(p.numel() for p in full.b.parameters())
    def tm(comm, reps=6):
        CNT["on"] = comm; ts = []
        for i in range(reps + 2):
            tp.zero_grad(set_to_none=True); dist.barrier(); t = time.perf_counter(); lossf(tp, batch).backward(); dt = time.perf_counter() - t
            if i >= 2: ts.append(dt)
        CNT["on"] = True; return st.median(ts)
    res["t_with_comm"] = tm(True); res["t_no_comm"] = tm(False)
    if rank == 0: json.dump(res, open(outfile, "w"))
    dist.destroy_process_group()
if __name__ == "__main__":
    n = int(sys.argv[1]); mp.spawn(worker, args=(n, 29900 + n, f"g4_{n}.json"), nprocs=n)
```

Statistics output:

```
model: 14,344,064 parameters, d=384, 8 heads, 8 layers; batch 4 x 128 tokens; embeddings, final norm, head and layer norms replicated

A. equivalence with the unsplit model (same weights, same batch)
  1-way: logits max diff 1.1e-06 (largest logit 2.46); loss diff 0.0e+00; gradient max diff 7.5e-09 (largest gradient 3.5e-03); logits after 3 Adam steps differ by 2.7e-06
  2-way: logits max diff 1.2e-06 (largest logit 2.46); loss diff 0.0e+00; gradient max diff 7.5e-09 (largest gradient 3.5e-03); logits after 3 Adam steps differ by 2.5e-06
  4-way: logits max diff 1.1e-06 (largest logit 2.46); loss diff 0.0e+00; gradient max diff 6.5e-09 (largest gradient 3.5e-03); logits after 3 Adam steps differ by 2.9e-06
  8-way: logits max diff 1.2e-06 (largest logit 2.46); loss diff 0.0e+00; gradient max diff 6.5e-09 (largest gradient 3.5e-03); logits after 3 Adam steps differ by 3.5e-06

B. parameters per rank
  1-way: 14,344,064 per rank = 1.000 of full (blocks 1.000; ideal 1/1 = 1.000); at 16 bytes/parameter 229.5 MB of model states per rank
  2-way: 7,255,424 per rank = 0.506 of full (blocks 0.501; ideal 1/2 = 0.500); at 16 bytes/parameter 116.1 MB of model states per rank
  4-way: 3,711,104 per rank = 0.259 of full (blocks 0.251; ideal 1/4 = 0.250); at 16 bytes/parameter 59.4 MB of model states per rank
  8-way: 1,938,944 per rank = 0.135 of full (blocks 0.126; ideal 1/8 = 0.125); at 16 bytes/parameter 31.0 MB of model states per rank

C. all-reduce calls in one forward+backward pass, 8 layers
  2-way: 16 forward + 16 backward = 32 (formula 4 x 8 = 32); payload 786432 bytes each (b*s*h*4 = 786432); ring bytes sent per rank 25.2 MB
  4-way: 16 forward + 16 backward = 32 (formula 4 x 8 = 32); payload 786432 bytes each (b*s*h*4 = 786432); ring bytes sent per rank 37.7 MB
  8-way: 16 forward + 16 backward = 32 (formula 4 x 8 = 32); payload 786432 bytes each (b*s*h*4 = 786432); ring bytes sent per rank 44.0 MB

   same model, data-parallel gradient sync of blocks per rank (Part 2 ring formula) vs tensor-parallel activation traffic, 4 x 128 tokens in the group
  N=2: data parallel   56.8 MB per step, tensor parallel   25.2 MB per step (0.44 x); ratio by formula tokens/(3h) = 0.444 for 4 x 128 tokens
  N=4: data parallel   85.2 MB per step, tensor parallel   37.7 MB per step (0.44 x); ratio by formula tokens/(3h) = 0.444 for 4 x 128 tokens
  N=8: data parallel   99.4 MB per step, tensor parallel   44.0 MB per step (0.44 x); ratio by formula tokens/(3h) = 0.444 for 4 x 128 tokens

D. time of one forward+backward pass, 1 thread per rank (median of 6)
  1-way: with communication     447 ms, communication switched off     426 ms (compute speedup vs 1-way 1.00x), communication share 5%
  2-way: with communication     471 ms, communication switched off     220 ms (compute speedup vs 1-way 1.93x), communication share 53%
  4-way: with communication     592 ms, communication switched off     129 ms (compute speedup vs 1-way 3.30x), communication share 78%
  8-way: with communication     893 ms, communication switched off     152 ms (compute speedup vs 1-way 2.81x), communication share 83%
```

Next: [Part 5](/posts/dist-5-pipeline/) splits the model by layers into pipeline stages, a schedule problem whose cost is a measurable bubble, and then combines the degrees into a 3D layout.

## References

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019.
