---
title: "Small Language Models on Small Devices (5/6): Quantisation, Pruning and Distillation"
date: 2027-01-28 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, edge-ai, quantisation, pruning, distillation, gptq, awq, series]
description: "Part 5 of 6. The three ways to shrink a trained model, tested on one small transformer on this machine: bits per weight and scale granularity, magnitude pruning, and distillation, with each method's published claim beside the measurement."
---

[Part 1](/posts/slm-edge-1-memory-bandwidth-arithmetic/) showed that memory sets the limit and [Part 4](/posts/slm-edge-4-microcontrollers/) showed what that limit looks like at 0.2 MB. Three methods move a model under a limit: store weights in fewer bits (quantisation), set weights to zero (pruning), or train a smaller model to imitate a larger one (distillation). This part states each method's published claim and then measures the same three knobs on one model trained here, a 421,376-parameter character-level transformer. The models are tiny, so the numbers show the shape of each trade-off, not the size of the effect at billions of parameters.

![Three panels. Left: held-out bits per character on a log axis against 8, 4, 3 and 2 bits per weight for per-tensor, per-row and group-of-32 scales; at 3 bits they are 2.93, 2.41 and 2.32, and at 2 bits 6.36, 5.05 and 4.05. Middle: pruning from 0 to 90 percent of weights gives 2.14, 2.18, 2.39, 3.35 and 5.34. Right: a 31,616-parameter student scores 2.843 from scratch, 2.778 and 2.843 when distilled with two loss weights, and the teacher scores 2.139.](/assets/pic/slm-edge-5-compression.png)
*Left: finer scales shrink the 3- and 2-bit loss. Middle: pruning is cheap to 30 percent and costly past 50. Right: distillation gains little in this run.*

## What the papers claim

- **Quantisation.** [LLM.int8()](https://arxiv.org/abs/2208.07339) halves inference memory with no reported degradation up to 175B parameters by keeping a few outlier feature dimensions in 16 bits while more than 99.9% of values are multiplied in 8 bits. [GPTQ](https://arxiv.org/abs/2210.17323) quantises a 175B model to 3 or 4 bits in about four GPU hours "with negligible accuracy degradation" using approximate second-order information, and shows reasonable accuracy at 2 bits or ternary. [AWQ](https://arxiv.org/abs/2306.00978), aimed at on-device models, argues that protecting only 1% of salient weight channels greatly reduces the error, and picks them from the activation distribution, not the weights.
- **Pruning.** [SparseGPT](https://arxiv.org/abs/2301.00774) prunes GPT-family models to at least 50% sparsity in one shot, without retraining, "at minimal loss of accuracy", and reaches 60% unstructured sparsity on OPT-175B and BLOOM-176B with negligible perplexity increase. It also supports 2:4 and 4:8 semi-structured patterns and composes with quantisation.
- **Distillation.** [DistilBERT](https://arxiv.org/abs/1910.01108) cuts BERT's size by 40% while keeping 97% of its language-understanding capability and running 60% faster, by adding a distillation term to the pre-training loss. [Part 9 of the maths series](/posts/maths-for-llms-9-information-entropy-cross-entropy-kl/) derives the KL term used for this.

All three results are from models far larger than the one below, and from methods more careful than the round-to-nearest and magnitude rules used here.

```python
import copy
import math, time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); torch.set_num_threads(16)
text=open('pg.txt',encoding='utf-8').read().lower()
chars=sorted(set(text)); V=len(chars); ix={c:i for i,c in enumerate(chars)}
data=torch.tensor([ix[c] for c in text]); n=int(0.9*len(data)); tr,va=data[:n],data[n:]
CTX=64
def batch(d,B=64):
    s=torch.randint(0,len(d)-CTX-1,(B,)); x=torch.stack([d[i:i+CTX] for i in s]); y=torch.stack([d[i+1:i+CTX+1] for i in s]); return x,y
class Block(nn.Module):
    def __init__(s,d,h):
        super().__init__(); s.n1=nn.LayerNorm(d); s.n2=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d,bias=False); s.o=nn.Linear(d,d,bias=False)
        s.f1=nn.Linear(d,4*d,bias=False); s.f2=nn.Linear(4*d,d,bias=False); s.h=h
    def forward(s,x):
        B,T,D=x.shape; q,k,v=s.qkv(s.n1(x)).split(D,2)
        q,k,v=[t.view(B,T,s.h,D//s.h).transpose(1,2) for t in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D)
        x=x+s.o(a); return x+s.f2(F.gelu(s.f1(s.n2(x))))
class LM(nn.Module):
    def __init__(s,d,L=2,h=4):
        super().__init__(); s.e=nn.Embedding(V,d); s.p=nn.Embedding(CTX,d); s.b=nn.ModuleList([Block(d,h) for _ in range(L)]); s.n=nn.LayerNorm(d); s.head=nn.Linear(d,V,bias=False)
    def forward(s,x):
        h=s.e(x)+s.p(torch.arange(x.shape[1])); 
        for b in s.b: h=b(h)
        return s.head(s.n(h))
def bpc(m,d):
    m.eval(); tot=0
    with torch.no_grad():
        g=torch.Generator().manual_seed(1)
        for _ in range(20):
            s=torch.randint(0,len(d)-CTX-1,(64,),generator=g); x=torch.stack([d[i:i+CTX] for i in s]); y=torch.stack([d[i+1:i+CTX+1] for i in s])
            tot+=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)).item()
    m.train(); return tot/20/math.log(2)
def quant_(m,bits):
    qm=2**(bits-1)-1
    with torch.no_grad():
        for n_,p in m.named_parameters():
            if p.dim()==2:
                s=p.abs().amax(1,keepdim=True)/qm; p.copy_(torch.clamp(torch.round(p/s),-qm-1,qm)*s)
def train(m,steps=1500,teacher=None,alpha=0.5,T=2.0):
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.01)
    for st in range(steps):
        lr=3e-3*min(1,(st+1)/100)*0.5*(1+math.cos(math.pi*st/steps))
        for g in opt.param_groups: g['lr']=lr
        x,y=batch(tr); out=m(x); loss=F.cross_entropy(out.reshape(-1,V),y.reshape(-1))
        if teacher is not None:
            with torch.no_grad(): tl=teacher(x)
            kd=F.kl_div(F.log_softmax(out/T,-1),F.softmax(tl/T,-1),reduction='batchmean')*T*T/ x.shape[1]
            loss=(1-alpha)*loss+alpha*kd
        opt.zero_grad(); loss.backward(); opt.step()
    return m
def rtn(w,bits,group=None):
    qm=2**(bits-1)-1
    if group is None: s=w.abs().amax()/qm
    elif group==0: s=w.abs().amax(1,keepdim=True)/qm
    else:
        sh=w.shape; w=w.reshape(-1,group); s=w.abs().amax(1,keepdim=True)/qm
    q=torch.clamp(torch.round(w/s),-qm-1,qm)*s
    return q.reshape(sh) if group else q
def apply(m,fn):
    m=copy.deepcopy(m)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim()==2 and p.shape[0]!=V and p.shape[0]!=CTX: p.copy_(fn(p))
    return m
torch.manual_seed(0)
T_=train(LM(128)); print(f"teacher d=128: {sum(p.numel() for p in T_.parameters()):,} params, {bpc(T_,va):.3f} bits/char",flush=True)
print("\n1. weight-only quantisation of the teacher (embeddings and head left in float32)")
print(f"{'bits':>5} {'per-tensor':>11} {'per-row':>9} {'group 32':>9}")
for b in [8,4,3,2]:
    print(f"{b:>5} " + " ".join(f"{bpc(apply(T_,lambda w:rtn(w,b,g)),va):>{n}.3f}" for g,n in [(None,11),(0,9),(32,9)]))
print("\n2. magnitude pruning of the teacher (global per-matrix, unstructured)")
def prune(w,f):
    k=int(w.numel()*f); th=w.abs().flatten().kthvalue(k).values; return w*(w.abs()>th)
print(f"{'sparsity':>9} {'bits/char':>10}")
for f in [0.0,0.3,0.5,0.7,0.9]:
    print(f"{f:>9.1f} {bpc(apply(T_,lambda w:prune(w,f) if f>0 else w),va):>10.3f}")
print("\n3. distillation: d=32 student (31,616 parameters), same 1500 steps")
torch.manual_seed(1); S0=train(LM(32)); print(f"from scratch:            {bpc(S0,va):.3f} bits/char")
for a in [0.5,0.9]:
    torch.manual_seed(1); S=train(LM(32),teacher=T_,alpha=a); print(f"distilled (alpha={a}, T=2): {bpc(S,va):.3f} bits/char")
print(f"teacher (13x larger):    {bpc(T_,va):.3f} bits/char")
```

```
teacher d=128: 421,376 params, 2.139 bits/char

1. weight-only quantisation of the teacher (embeddings and head left in float32)
 bits  per-tensor   per-row  group 32
    8       2.139     2.139     2.139
    4       2.272     2.181     2.165
    3       2.929     2.414     2.321
    2       6.357     5.050     4.053

2. magnitude pruning of the teacher (global per-matrix, unstructured)
 sparsity  bits/char
      0.0      2.139
      0.3      2.183
      0.5      2.391
      0.7      3.353
      0.9      5.337

3. distillation: d=32 student (31,616 parameters), same 1500 steps
from scratch:            2.843 bits/char
distilled (alpha=0.5, T=2): 2.778 bits/char
distilled (alpha=0.9, T=2): 2.843 bits/char
teacher (13x larger):    2.139 bits/char
```

## Quantisation: the scale is the lever

Eight bits changes nothing at any granularity (2.139 bits per character). At 4 bits, one scale for the whole matrix costs 0.133, one per row 0.042 and one per group of 32 weights 0.026. At 3 bits the costs are 0.790, 0.275 and 0.182, and at 2 bits the model is destroyed in all three cases (6.36, 5.05 and 4.05, against 6.19 for a uniform guess, so per-tensor 2-bit is worse than guessing). A smaller group stores more scales: 32 weights per scale adds 0.5 bits per weight if each scale is a 16-bit float (the author's arithmetic, not measured here), which is why 4-bit formats with groups are really about 4.5 bits. This is the mechanism that GPTQ and AWQ improve on: they choose the rounding or the scaling with data instead of rounding to the nearest level.

## Pruning: a cliff, not a slope

Zeroing the smallest 30% of each weight matrix costs 0.044 bits per character, 50% costs 0.252, 70% costs 1.214 and 90% reaches 5.34, close to a uniform guess. The curve is flat and then falls off, which is why published results quote 50–60% sparsity at most, and SparseGPT does better than this plain magnitude rule by correcting the remaining weights. Unstructured zeros also save nothing on a CPU unless the kernel skips them; the 2:4 pattern exists because hardware can exploit it.

## Distillation: small gain here

A 31,616-parameter student trained for the same 1,500 steps scores 2.843 alone. With a loss that mixes cross-entropy and KL to the teacher's softened outputs (temperature 2), it scores 2.778 at weight 0.5, an improvement of 0.065, and 2.843 at weight 0.9, no change. The teacher scores 2.139, so the student closes about 9% of the gap. This is one seed and two settings, so the 0.065 may not survive a second seed, and the loss scaling is a free choice; the experiment supports only "distillation helped slightly at one setting". The student was trained on the same text as the teacher; DistilBERT's larger gain comes from distilling during pre-training at a different scale.

## Choosing among them

For a device with a fixed memory budget, the measurements and the papers agree on an order: quantise first (8 to 4 bits is nearly free with fine scales), use pruning only with a kernel that exploits the pattern, and treat distillation as a way to train the small model, not to shrink a finished one. That ranking is the author's reading and is specific to weight memory; compute-bound prefill and activation memory are different problems.

## Limits

- One model, one text, one seed. The ordering of methods may differ for an instruction-tuned model with billions of parameters, and outlier channels, which drive LLM.int8() and AWQ, are weak in a model this small.
- Embeddings, the output head and norms were left in float32 and are not counted in the quantisation cost; bits per character is a language-modelling loss, not a task accuracy.
- Quantisation and pruning were applied after training with no calibration data and no fine-tuning; GPTQ, AWQ and SparseGPT use calibration data, so they are not tested here.
- No speed was measured: a real runtime such as llama.cpp needs a kernel for each format, and the saving depends on it.

Next: Part 6, post-training with LoRA and QLoRA.
