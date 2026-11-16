---
title: "LLMOps (2/6): Serving, Batching and the KV Cache"
date: 2027-02-06 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLMOps, serving, batching, kv-cache, latency, series]
description: "Part 2 of 6. A 25.6M-parameter transformer on 16 CPU cores: the KV cache halves per-token decode time, batching trades step latency for throughput (125 to 1,359 tokens/s from batch 1 to 64), and a simulated scheduler shows why continuous batching beats static batching under load. Memory arithmetic for paged versus contiguous caches, with PagedAttention as the reference."
---

Part 1 showed what pins the output; this part is about what a request costs. A small transformer is built with and without a KV cache, run at increasing batch sizes, and put behind a simulated scheduler, all on CPU. The memory section is arithmetic, and the scheduler section is a simulation driven by the measured step times; both are labelled as such.

## Reference

[Kwon et al. (2023)](https://arxiv.org/abs/2309.06180) state that high-throughput LLM serving requires batching enough requests at a time, and that this is limited because the key-value cache of each request is large and grows and shrinks dynamically; managed badly, the memory is wasted by fragmentation and duplication. They propose PagedAttention, inspired by virtual-memory paging, and the vLLM system built on it, and report 2-4x higher throughput than FasterTransformer and Orca at the same latency. The sections below reproduce the shape of the problem at toy scale, not those numbers.

## The model and the three measurements

The model has 8 layers, width 512, 8 heads, random weights (25.6M parameters, fp32). Random weights are enough because the cost of a forward pass does not depend on what the weights say. A check confirms the cached and uncached decoders produce identical greedy output.

```python
import time, math, random, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(16); torch.manual_seed(0); random.seed(0)
D,L,H,V,MAXT=512,8,8,128,512
class Blk(nn.Module):
    def __init__(s):
        super().__init__(); s.n1=nn.LayerNorm(D); s.n2=nn.LayerNorm(D); s.qkv=nn.Linear(D,3*D,bias=False); s.o=nn.Linear(D,D,bias=False); s.f1=nn.Linear(D,4*D,bias=False); s.f2=nn.Linear(4*D,D,bias=False)
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.e=nn.Embedding(V,D); s.p=nn.Embedding(MAXT,D); s.b=nn.ModuleList([Blk() for _ in range(L)]); s.n=nn.LayerNorm(D); s.h=nn.Linear(D,V,bias=False)
m=M().eval()
print("params %.1fM"%(sum(p.numel() for p in m.parameters())/1e6))
def full(x):  # no cache: recompute whole sequence
    B,T=x.shape; h=m.e(x)+m.p(torch.arange(T))
    for b in m.b:
        q,k,v=b.qkv(b.n1(h)).split(D,2); q,k,v=[t.view(B,T,H,D//H).transpose(1,2) for t in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D); h=h+b.o(a); h=h+b.f2(F.gelu(b.f1(b.n2(h))))
    return m.h(m.n(h))
def step(tok,pos,K,Vc):  # one new token per row; K,Vc: [L][B,H,MAXT,dh]; all rows same length (pos)
    B=tok.shape[0]; h=m.e(tok)+m.p(torch.full((B,1),pos))
    for i,b in enumerate(m.b):
        q,k,v=b.qkv(b.n1(h)).split(D,2); q,k,v=[t.view(B,1,H,D//H).transpose(1,2) for t in (q,k,v)]
        K[i][:,:,pos:pos+1]=k; Vc[i][:,:,pos:pos+1]=v
        a=F.scaled_dot_product_attention(q,K[i][:,:,:pos+1],Vc[i][:,:,:pos+1]).transpose(1,2).reshape(B,1,D); h=h+b.o(a); h=h+b.f2(F.gelu(b.f1(b.n2(h))))
    return m.h(m.n(h))[:,-1]
@torch.no_grad()
def gen_cache(x,n):
    B,T=x.shape; K=[torch.zeros(B,H,MAXT,D//H) for _ in range(L)]; Vc=[torch.zeros(B,H,MAXT,D//H) for _ in range(L)]
    out=[]; 
    for t in range(T): lg=step(x[:,t:t+1],t,K,Vc)
    for j in range(n):
        nt=lg.argmax(-1,keepdim=True); out.append(nt); lg=step(nt,T+j,K,Vc)
    return torch.cat(out,1)
@torch.no_grad()
def gen_nocache(x,n):
    for _ in range(n): x=torch.cat([x,full(x)[:,-1].argmax(-1,keepdim=True)],1)
    return x[:,-n:]
x=torch.randint(0,V,(2,16))
a=gen_cache(x,24); b=gen_nocache(x,24); print("cache == no-cache greedy output:",bool((a==b).all()))
print("\nA. decode cost per generated token, prompt 16, generate 64, batch 1")
for name,f in (("no cache",gen_nocache),("KV cache",gen_cache)):
    x=torch.randint(0,V,(1,16)); t=time.time(); f(x,64); dt=time.time()-t; print(f"  {name:9s} {dt:6.2f}s total  {dt/64*1000:7.1f} ms/token")
print("\nB. batch size vs step latency and throughput (KV cache, prompt 16 + 64 new tokens)")
print("  batch  ms/step  tokens/s")
res={}
for B in (1,2,4,8,16,32,64):
    x=torch.randint(0,V,(B,16)); t=time.time(); gen_cache(x,64); dt=time.time()-t; steps=16+64
    res[B]=dt/steps; print(f"  {B:5d}  {dt/steps*1000:7.1f}  {B*64/dt:8.1f}")
print("\nC. KV cache memory, this model (fp32): 2*L*D*4 bytes per token per sequence =",2*L*D*4,"bytes")
per=2*L*D*4
lens=[random.choice([20,40,80,160,320,480]) for _ in range(1000)]
need=sum(lens)*per; contig=len(lens)*MAXT*per
blk=16; paged=sum(math.ceil(l/blk)*blk for l in lens)*per
print(f"  1000 requests, lengths 20..480 (mean {sum(lens)/len(lens):.0f}), max length reserved {MAXT}")
print(f"  bytes actually needed        {need/2**20:8.1f} MiB")
print(f"  contiguous, reserve max len  {contig/2**20:8.1f} MiB  ({need/contig*100:.1f}% used)")
print(f"  blocks of {blk} tokens         {paged/2**20:8.1f} MiB  ({need/paged*100:.1f}% used)")
print("\nD. static vs continuous batching, simulated with the measured step times")
import bisect
bs=sorted(res); 
def st(B):  # interpolate measured ms/step
    if B<=bs[0]: return res[bs[0]]
    for a,b in zip(bs,bs[1:]):
        if B<=b: return res[a]+(res[b]-res[a])*(B-a)/(b-a)
    return res[bs[-1]]
def sim(mode,rate,N=600,cap=16,seed=1):
    r=random.Random(seed); t=0; arr=[]; 
    for i in range(N): t+=r.expovariate(rate); arr.append((t,r.choice([8,16,32,64,128])))
    now=0; q=list(arr); act=[]; done=[]; i=0
    while len(done)<N:
        if mode=="static":
            if not act:
                while i<N and arr[i][0]<=now and len(act)<cap: act.append([arr[i][0],arr[i][1]]); i+=1
                if not act: now=max(now,arr[i][0]) if i<N else now; continue
                dur=max(a[1] for a in act); now+=dur*st(len(act))
                for a in act: done.append(now-a[0])
                act=[]
        else:
            while i<N and arr[i][0]<=now and len(act)<cap: act.append([arr[i][0],arr[i][1]]); i+=1
            if not act: now=arr[i][0]; continue
            now+=st(len(act)); 
            for a in act: a[1]-=1
            for a in [a for a in act if a[1]<=0]: done.append(now-a[0])
            act=[a for a in act if a[1]>0]
    d=sorted(done); return sum(d)/N, d[int(.95*N)], now
tp=1/st(1)
print("  request lengths 8..128 new tokens, capacity 16; latency = arrival to last token (s)")
print("  arrivals/s  mode        mean    p95")
for rate in (5,10,15,20):
    for mode in ("static","continuous"):
        mn,p95,_=sim(mode,rate); print(f"  {rate:9.0f}  {mode:10s} {mn:6.1f} {p95:6.1f}")
```

Output:

```
params 25.6M
cache == no-cache greedy output: True

A. decode cost per generated token, prompt 16, generate 64, batch 1
  no cache    1.11s total     17.3 ms/token
  KV cache    0.51s total      8.0 ms/token

B. batch size vs step latency and throughput (KV cache, prompt 16 + 64 new tokens)
  batch  ms/step  tokens/s
      1      6.4     124.9
      2      7.3     219.1
      4      7.4     433.5
      8     10.3     623.6
     16     14.4     888.3
     32     20.2    1268.0
     64     37.7    1359.1

C. KV cache memory, this model (fp32): 2*L*D*4 bytes per token per sequence = 32768 bytes
  1000 requests, lengths 20..480 (mean 180), max length reserved 512
  bytes actually needed          5628.1 MiB
  contiguous, reserve max len   16000.0 MiB  (35.2% used)
  blocks of 16 tokens           5738.0 MiB  (98.1% used)

D. static vs continuous batching, simulated with the measured step times
  request lengths 8..128 new tokens, capacity 16; latency = arrival to last token (s)
  arrivals/s  mode        mean    p95
          5  static        1.3    2.4
          5  continuous    0.4    1.0
         10  static        6.8   10.5
         10  continuous    0.5    1.2
         15  static       16.5   29.3
         15  continuous    0.6    1.6
         20  static       20.4   37.6
         20  continuous    0.9    2.3
```

![Three panels. Left: throughput in tokens per second rising from 125 at batch 1 to 1,359 at batch 64, while milliseconds per step rise from 6.4 to 37.7. Middle: KV cache memory for 1,000 requests: 5,628 MiB needed, 16,000 MiB with contiguous max-length reservation (35.2 percent used), 5,738 MiB with 16-token blocks (98.1 percent used). Right: simulated mean and p95 latency against arrival rate for static and continuous batching; static grows to 20.4 s mean and 37.6 s p95 at 20 arrivals per second, continuous stays under 2.3 s.](/assets/pic/llmops-2-serving.png)
*Left: measured batching trade-off. Middle: KV cache memory arithmetic. Right: simulated scheduling using the measured step times.*

## What the numbers say

- **The cache halves decode cost here.** Without it, each new token recomputes attention inputs for the whole sequence: 17.3 ms per token against 8.0 ms with the cache, for a 16-token prompt and 64 generated tokens. The recomputed work grows with sequence length, so the gap is expected to widen; this test is short and did not measure that.
- **Batching buys throughput with latency.** Going from batch 1 to 64, throughput rose from 124.9 to 1,359.1 tokens/s (10.9x), while the time of one step rose from 6.4 to 37.7 ms (5.9x). The curve flattens: batch 32 reached 1,268, so the last doubling added 7%. Batch 1 to 4 is nearly free because a single sequence does not fill 16 cores. These are single runs; batch-1 throughput differed by a few percent between the two runs of the script (first run: 114.9 at batch 1).
- **The cache memory is the limit.** Each token of each sequence takes 2 x layers x width x 4 bytes = 32,768 bytes in this model. For 1,000 requests with lengths between 20 and 480 tokens (mean 180), the cache needs 5,628 MiB. Reserving the maximum length for every request needs 16,000 MiB, so only 35.2% of it holds real data. Allocating in 16-token blocks needs 5,738 MiB (98.1% used). This is arithmetic from the request lengths, not a measurement of an allocator; the 98.1% is the same arithmetic PagedAttention's argument rests on.
- **Scheduling dominates under load.** The scheduler simulation draws requests with 8 to 128 new tokens and capacity 16, with Poisson arrivals, and charges each step the measured step time for the current batch size. *Static* batching starts a batch and runs it until its longest request finishes; new arrivals wait. *Continuous* batching admits and retires requests at every step. At 5 arrivals per second, mean latency was 1.3 s static against 0.4 s continuous; at 20 per second, 20.4 s against 0.9 s (p95: 37.6 s against 2.3 s). Static batching wastes steps on finished rows and queues new work behind the longest request.

## What to take from it

- Capacity planning needs two numbers per deployment: step time at a batch size, and KV bytes per token. Everything else follows from the request-length distribution.
- Report latency as percentiles under a stated arrival rate. A mean at batch 1 says nothing about behaviour under load.
- The scheduler matters as much as the kernel. The same step times gave more than a 20x difference in mean latency (20.4 s against 0.9 s) at the highest load tested.

Next: [Part 3, evaluation: a regression set, its noise, and the judge model](/posts/llmops-3-evaluation-is-a-measurement/).

## Limits

- Random-weight model, 25.6M parameters, CPU only. GPU step times, memory bandwidth limits and attention-kernel behaviour at long contexts are different; only the shape of the trade-offs transfers.
- Sequences are short (80 positions). The cache benefit is expected to grow with length; that was not measured.
- The scheduler is a simulation written for this post, with a simplified cost model (step time depends only on batch size, no prefill cost separate from decode, interpolated between measured points). It illustrates the mechanism and does not predict a real server.
- Memory figures are arithmetic on a synthetic length distribution. No allocator was run.
- A first run of the simulation used arrival rates of 0.05 to 0.3 per second, where the server was never busy, and all rows were identical; the rates were raised to 5 to 20 and the run repeated. Throughput numbers shifted a few percent between runs, so treat them as single measurements.
