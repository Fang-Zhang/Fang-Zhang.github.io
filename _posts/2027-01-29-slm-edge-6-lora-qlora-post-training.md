---
title: "Small Language Models on Small Devices (6/6): Post-Training with LoRA and QLoRA"
date: 2027-01-29 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, edge-ai, lora, qlora, fine-tuning, quantisation, series]
description: "Part 6 of 6. A pre-trained small model adapted to a new domain by full fine-tuning, LoRA at three ranks and LoRA on a 4-bit base, with before and after numbers, the trainable-parameter cost and the forgetting each method leaves behind."
---

Parts 1 to 5 shrank the model; post-training changes what it does. A small model that is cheap to run can be specialised cheaply too, and the two published methods are [LoRA](https://arxiv.org/abs/2106.09685), which freezes the pre-trained weights and trains a low-rank update, and [QLoRA](https://arxiv.org/abs/2305.14314), which does the same on top of a frozen 4-bit base. This part runs both on a small transformer here: pre-train on a novel, adapt to Python source code, and measure the gain, the trainable-parameter count and what is forgotten. The vendor-side picture is in [Part 2](/posts/slm-edge-2-phones-pcs-vendors/), where adapters are the mechanism for task-specific variants of an on-device model; the measurement below is on a 431K-parameter model and says nothing about the size of the gains at billions of parameters.

![Two bar charts over seven variants. Left, held-out bits per character on Python source: no adaptation 4.14, full fine-tune 2.11, LoRA rank 1 2.34, rank 4 2.09, rank 16 2.02, 4-bit base plus LoRA rank 16 2.04, a model trained from scratch for 400 steps 2.04. Right, the same models on the original novel: 2.15, 3.94, 2.87, 3.23, 3.55, 3.58 and 4.43.](/assets/pic/slm-edge-6-lora.png)
*Left: loss on the new domain. Right: loss on the old domain, which measures forgetting. Lower is better in both.*

## What the papers claim

- **LoRA** freezes the pre-trained weights and learns, for a weight matrix W, an update BA where B and A have a small rank r; this cuts the trainable parameters, and an adapter can be swapped without touching the base model. [Part 10 of the maths series](/posts/maths-for-llms-10-svd-low-rank-fourier-positions/) covers the low-rank view.
- **QLoRA** backpropagates through a frozen 4-bit base into LoRA adapters and, per its abstract, preserves 16-bit fine-tuning performance while reducing memory enough to fine-tune a 65B model on a single 48 GB GPU. It adds a 4-bit NormalFloat type, double quantisation of the scale constants and paged optimisers.

```python
import copy
import math, time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); torch.set_num_threads(16)
novel=open('pg.txt',encoding='utf-8').read().lower()
import glob
code=''.join(open(f,encoding='utf-8').read().lower() for f in ['/usr/lib/python3.12/textwrap.py','/usr/lib/python3.12/argparse.py','/usr/lib/python3.12/shlex.py'])[:400000]
text=novel+code
chars=sorted(set(text)); V=len(chars); ix={c:i for i,c in enumerate(chars)}
enc=lambda s: torch.tensor([ix[c] for c in s])
nov=enc(novel); n=int(0.9*len(nov)); tr,va=nov[:n],nov[n:]
cod=enc(code); m_=int(0.5*len(cod)); ctr,cva=cod[:m_],cod[m_:]
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
class LoRA(nn.Module):
    def __init__(s,base,r,alpha=None):
        super().__init__(); s.base=base; s.r=r; s.sc=(alpha or r)/r
        s.A=nn.Parameter(torch.randn(r,base.in_features)/math.sqrt(base.in_features)); s.B=nn.Parameter(torch.zeros(base.out_features,r))
    def forward(s,x): return s.base(x)+(x@s.A.t()@s.B.t())*s.sc
def add_lora(m,r):
    for p in m.parameters(): p.requires_grad=False
    for b in m.b:
        for nme in ['qkv','o','f1','f2']: setattr(b,nme,LoRA(getattr(b,nme),r))
    return m
def ntrain(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def ft(m,steps,lr):
    ps=[p for p in m.parameters() if p.requires_grad]; opt=torch.optim.AdamW(ps,lr=lr,weight_decay=0.0)
    for st in range(steps):
        x,y=batch(ctr); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return m
def rtn4(m):
    with torch.no_grad():
        for b in m.b:
            for nme in ['qkv','o','f1','f2']:
                w=getattr(b,nme).weight; w4=w.reshape(-1,32); s=w4.abs().amax(1,keepdim=True)/7; w.copy_(torch.clamp(torch.round(w4/s),-8,7).mul(s).reshape(w.shape))
    return m
torch.manual_seed(0)
base=LM(128)
opt=torch.optim.AdamW(base.parameters(),lr=3e-3,weight_decay=0.01)
for st in range(1500):
    lr=3e-3*min(1,(st+1)/100)*0.5*(1+math.cos(math.pi*st/1500))
    for g in opt.param_groups: g['lr']=lr
    x,y=batch(tr); loss=F.cross_entropy(base(x).reshape(-1,V),y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
P=sum(p.numel() for p in base.parameters())
print(f"base d=128, {P:,} parameters, pre-trained on a novel; vocabulary {V}")
print(f"novel held-out: {bpc(base,va):.3f} bits/char   python source held-out: {bpc(base,cva):.3f}  (domain shift)")
print(f"adaptation data: {len(ctr):,} characters of python source, 400 steps\n")
print(f"{'method':<22}{'trainable':>10}{'% of model':>11}{'code bpc':>10}{'novel bpc':>10}")
def row(n,m,t=None):
    t=ntrain(m) if t is None else t; print(f"{n:<22}{t:>10,}{100*t/P:>10.1f}%{bpc(m,cva):>10.3f}{bpc(m,va):>10.3f}",flush=True)
row("no adaptation",base,0)
torch.manual_seed(1); m=ft(copy.deepcopy(base),400,1e-3); row("full fine-tune",m)
for r in [1,4,16]:
    torch.manual_seed(1); m=ft(add_lora(copy.deepcopy(base),r),400,3e-3); row(f"LoRA r={r}",m)
torch.manual_seed(1); m=copy.deepcopy(base); rtn4(m); row("4-bit base, no adapt",m,0)
torch.manual_seed(1); m=ft(add_lora(m,16),400,3e-3); row("4-bit base + LoRA r=16",m)
torch.manual_seed(1); m=LM(128); 
for p in m.parameters(): p.requires_grad=True
m=ft(m,400,3e-3); row("scratch, 400 steps",m)
```

```
base d=128, 431,104 parameters, pre-trained on a novel; vocabulary 111
novel held-out: 2.146 bits/char   python source held-out: 4.144  (domain shift)
adaptation data: 67,379 characters of python source, 400 steps

method                 trainable % of model  code bpc novel bpc
no adaptation                  0       0.0%     4.144     2.146
full fine-tune           431,104     100.0%     2.107     3.940
LoRA r=1                   4,096       1.0%     2.344     2.873
LoRA r=4                  16,384       3.8%     2.087     3.235
LoRA r=16                 65,536      15.2%     2.019     3.548
4-bit base, no adapt           0       0.0%     4.144     2.173
4-bit base + LoRA r=16    65,536      15.2%     2.042     3.583
scratch, 400 steps       431,104     100.0%     2.038     4.431
```

## Result

The base model reaches 2.146 bits per character on the novel and 4.144 on Python source it has not seen, so the domain shift is large. After 400 steps on 67,379 characters of source:

- **Full fine-tuning** (all 431,104 parameters) reaches 2.107 on code.
- **LoRA** with 4,096 trainable parameters (1.0%, rank 1) reaches 2.344; rank 4 (16,384, 3.8%) reaches 2.087, better than full fine-tuning in this run; rank 16 (65,536, 15.2%) reaches 2.019. Rank 16 here includes the attention and feed-forward matrices of both layers.
- **4-bit base plus rank-16 LoRA** reaches 2.042, a cost of 0.023 against the same adapter on the float32 base. The base alone loses 0.027 on the novel when rounded to 4 bits (2.173 against 2.146), which matches Part 5.

## What the table also shows

- **The gain is mostly the adaptation, not the pre-training.** A model of the same shape trained from scratch for the same 400 steps reaches 2.038 on code, the same as LoRA rank 16 (2.019) within what one seed can resolve. The novel gave this base no measurable head start on source code, which is plausible for two such different domains and for 1,500 steps of pre-training. A test with a closer pre-training domain would be needed to show transfer, so this table cannot speak for it.
- **Forgetting scales with the number of trainable parameters.** On the novel, the loss goes from 2.146 to 2.873 with rank 1, 3.235 with rank 4, 3.548 with rank 16 and 3.940 with full fine-tuning. The more of the model is trained, the more of the old behaviour is overwritten, though no variant preserves it: even rank 1 loses 0.73 bits per character. LoRA's claim of keeping the base intact holds for the stored weights, since the adapter can be switched off and the base returns to 2.146, not for the combined model.
- **Rank has diminishing returns.** Going from rank 1 to 4 gains 0.257 on code; from 4 to 16 gains 0.068, with four times the parameters.

## What this means on a device

An adapter is a small file: 65,536 parameters at 16 bits is 128 KB against 431,104 parameters for the base, and one base can serve several adapters (the author's arithmetic, assuming float16 storage). The adapter either stays separate, costing a small extra matrix product per layer, or is merged into the base weights, which removes the cost but fixes the task. If the base is already 4-bit, the QLoRA-style setup trains against the same weights that ship, so the training sees the quantisation error; here that is the 0.023 above. Training itself needs memory for activations and optimiser state on the adapter only, which is why it fits a single CPU in this experiment; a billion-parameter model is a different calculation.

## Limits

- One seed, one pair of domains, 400 steps, a character-level model of 431K parameters. Differences of about 0.03 bits per character (rank 16 against scratch, float32 against 4-bit) are within what a second seed could change.
- The QLoRA row uses round-to-nearest 4-bit with groups of 32 and a float32 adapter, not the NormalFloat type, double quantisation or paged optimisers of the paper.
- Loss on held-out text is not task accuracy, and no instruction-following behaviour was tested.
- Forgetting was measured on one old domain; replay of old data or a lower learning rate, which are standard mitigations, were not tried.
- The series ends here. Its claims about vendor devices come from official documents and projects cited in Parts 1 to 4; none was reproduced on that hardware.
