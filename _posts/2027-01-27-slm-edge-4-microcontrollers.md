---
title: "Small Language Models on Small Devices (4/6): Microcontrollers"
date: 2027-01-27 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, edge-ai, microcontroller, esp32, tinystories, quantisation, series]
description: "Part 4 of 6. Language models on a microcontroller have 0.26 to 29 million parameters and write children's-story text, nothing more. What two public ESP32-S3 projects report, how the memory tiers decide the design, and a character-level model trained and quantised on this machine to show size against quality."
---

The smallest tier is a microcontroller: hundreds of kilobytes of fast memory, a few megabytes of slower memory and a clock in the hundreds of MHz. Models here are not assistants. This part reads two public ESP32-S3 projects, summarises the dataset that makes such models usable at all, and trains four tiny character-level models on this machine to measure how quality, size and quantisation trade off at 10 thousand to 420 thousand parameters. All device figures are the projects' own claims; none was reproduced, since no microcontroller was available.

![Two panels. Left: held-out bits per character against parameter count on a log axis for float32, int8 and int4 versions of four tiny models; the int4 curve sits slightly above the other two, from 3.57 to 2.25, and the uniform guess is the dotted line at 6.19. Right: the ESP32-S3 memory tiers as bars, SRAM 512 KB, PSRAM 8 MB and flash 16 MB, with three model sizes on the same log axis: 0.21 MB, about 1 MB and 14.9 MB.](/assets/pic/slm-edge-4-microcontrollers.png)
*Left: the quantisation cost measured on four tiny models trained here. Right: the memory tiers of the ESP32-S3 as reported by the projects, against model sizes.*

## What the two projects report

- **A 260K-parameter model on an ESP32-S3.** [This project](https://github.com/DaveBben/esp32-llm) runs the 260K-parameter `stories260K` checkpoint from [llama2.c](https://github.com/karpathy/llama2.c) (inference of a Llama 2-style model in one C file), trained on the TinyStories dataset. The README says the model needs 1 MB of RAM, so it uses the ESP32-S3FH4R2 variant with 2 MB of embedded PSRAM. Using both cores, the S3's SIMD dot-product routines, 240 MHz and 80 MHz PSRAM, it reports 19.13 tokens per second, and calls the result possible but "probably not very useful".
- **A 28.9M-parameter model on an ESP32-S3.** [This project](https://github.com/slvDev/esp32-ai) reports 9.88 tokens per second end to end (94.9 ms per token of compute) for a 14.9 MB 4-bit model on a chip with 512 KB SRAM, 8 MB PSRAM and 16 MB flash. It fits because about 25M of the parameters are an embedding table kept in flash, from which only about 450 bytes are read per token. It credits the idea to Per-Layer Embeddings in Google's [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n). The README places the tiers by access frequency: activations and norm weights in SRAM, the dense core and output head in PSRAM, the large table in flash.
- **What it can do.** The second README states that the model "will not answer questions, follow instructions, write code, or know facts"; it writes short, simple stories, and it says the memory trick does not change that limit.

## Why such models can write at all

The [TinyStories paper](https://arxiv.org/abs/2305.07759) builds a synthetic dataset of short stories using only words a 3–4-year-old typically knows, and shows that models below 10 million parameters, or with a single transformer block, can still produce fluent, grammatical, multi-paragraph stories. The paper contrasts this with models of about 125M parameters trained on ordinary text, which it says rarely produce coherent text beyond a few words. The restriction of the data, not the size of the model, is what makes the task learnable, and it fixes the scope of the demo: it works inside the distribution of the data.

```python
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
import copy
print(f"vocabulary {V} characters, train {len(tr)} val {len(va)} characters, context {CTX}")
print(f"{'d':>4} {'params':>9} {'val bits/char':>14} {'int8 bpc':>9} {'int4 bpc':>9} {'int8 KB':>8} {'int4 KB':>8}  train s")
for d in [16,32,64,128]:
    m=LM(d); P=sum(p.numel() for p in m.parameters()); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.01)
    steps=1500; t0=time.time()
    for st in range(steps):
        lr=3e-3*min(1,(st+1)/100)*0.5*(1+math.cos(math.pi*st/steps)); 
        for g in opt.param_groups: g['lr']=lr
        x,y=batch(tr); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    b=bpc(m,va); m8=copy.deepcopy(m); quant_(m8,8); m4=copy.deepcopy(m); quant_(m4,4)
    print(f"{d:>4} {P:>9,} {b:>14.3f} {bpc(m8,va):>9.3f} {bpc(m4,va):>9.3f} {P/1e3:>8.0f} {P/2e3:>8.0f}  {time.time()-t0:.0f}",flush=True)
print("uniform over the vocabulary:",round(math.log2(V),3),"bits/char")
```

```
vocabulary 73 characters, train 673350 val 74817 characters, context 64
   d    params  val bits/char  int8 bpc  int4 bpc  int8 KB  int4 KB  train s
  16     9,664          3.523     3.523     3.571       10        5  13
  32    31,616          2.871     2.871     3.030       32       16  20
  64   112,384          2.417     2.418     2.576      112       56  41
 128   421,376          2.160     2.160     2.251      421      211  94
uniform over the vocabulary: 6.19 bits/char
```

## What the local experiment shows

The test trains a two-layer, four-head character-level transformer with context 64 on 673,350 characters of a public-domain novel (the held-out part is 74,817 characters), at widths 16, 32, 64 and 128. The vocabulary is 73 characters, so a uniform guess costs 6.19 bits per character. The four models reach 3.523, 2.871, 2.417 and 2.160 bits per character with 9,664, 31,616, 112,384 and 421,376 parameters: each step of roughly 3.5 times more parameters lowers the loss by less than the one before (0.65, 0.45, then 0.26 bits).

Quantising the weight matrices to 8 bits with one scale per row changes nothing visible (2.418 against 2.417 at d = 64). Quantising to 4 bits costs 0.048, 0.159, 0.159 and 0.091 bits per character at the four sizes, the largest cost falling on the two middle sizes. At 4 bits the largest model takes 211 KB and still beats the d = 64 model stored at 8 bits (112 KB, 2.418), for about twice the memory, so in this experiment the bigger 4-bit model is the better use of a fixed budget in that range. This is rounding after training, without the quantisation-aware training that [Part 2](/posts/slm-edge-2-phones-pcs-vendors/) described, and the models are far too small and too narrowly trained to say anything about the 29M-parameter case. It also measures a different thing from the projects: characters on a novel, not tokens on TinyStories.

## Where this tier fits

A model that can only continue stories suits a toy, an educational kit or a demo of what the chip can hold; this is also the scope in the README. For a device that must classify or react, a different tool fits: a small classifier or keyword model, not a generative one. That recommendation is the author's reading of the projects' own limits.

## Limits

- The ESP32 speeds and sizes are quoted from two single-author repositories and were not reproduced or independently reviewed. The 19.13 and 9.88 tokens per second come from different models and different PSRAM sizes.
- The local models are character-level, two layers, trained for 1,500 steps on one novel; the quality numbers are not comparable with the TinyStories models on the device.
- 4-bit here is round-to-nearest with one scale per row, applied after training, and measures only the weight matrices; embeddings and layer norms are in float32.
- Power draw, flash wear and output quality of the 29M model are not documented in the sources read.

Next: Part 5, compression methods.
