---
title: "LLM Security (2/6): Why Safety Training Is Thin"
date: 2027-01-31 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, security, alignment, fine-tuning, refusal, series]
description: "Part 2 of 6. Refusal behaviour is a thin layer on top of a capable model. Two published results, and a toy transformer where benign fine-tuning erodes refusal, eight examples remove it, and erasing one direction in the residual stream switches it off."
---

[Part 1](/posts/llm-sec-1-one-channel-instructions-data/) showed that behaviour toward an instruction is learned from data, not enforced by the architecture. Safety training is the most important case of that: a model is taught to refuse some requests, and the same mechanism that taught it can untrain it. This part reports two published results on how fragile refusal is, then reproduces the shape of both on a toy model trained here. The toy contains no harmful content: "refusal" is the answer `no` to two arbitrary task codes.

![Three panels. Left: fraction of refusal prompts still refused against fine-tuning steps on benign data; with benign data only it falls from 100 to 35.5, 12.7 and 0.9 percent at 100, 400 and 1600 steps, with benign data mixed with the old tasks to 58.7, 27.0 and 24.9 percent. Middle: after fine-tuning on 8 or 32 examples that comply with task X, refusal on X is 0 percent and on the untouched task Y 0.3 and 0 percent. Right: erasing a single direction in the residual stream, estimated at layer 0, 1 or 2, lowers refusal on X and Y from 100 to about 26 and 32, 7 and 7, and 0 and 0 percent while the harmless tasks stay at 100 percent.](/assets/pic/llm-sec-2-thin-safety.png)
*Left: benign fine-tuning alone erodes refusal. Middle: a handful of complying examples removes it, including for the task never touched. Right: one erased direction removes it, with no loss on harmless tasks.*

## What the papers report

- **Fine-tuning compromises safety.** [Qi et al.](https://arxiv.org/abs/2310.03693) report that fine-tuning an aligned model on only a few adversarially designed examples removes its safety guardrails: GPT-3.5 Turbo was jailbroken with 10 such examples at a cost under $0.20 through the vendor's fine-tuning API. They also report that fine-tuning on benign, commonly used datasets can inadvertently degrade safety, "though to a lesser extent".
- **Refusal is one direction.** [Arditi et al.](https://arxiv.org/abs/2406.11717) find, across 13 open-source chat models up to 72B parameters, a single direction in the residual stream such that erasing it stops the model refusing harmful instructions, and adding it makes the model refuse harmless ones. They call the result evidence of the brittleness of current safety fine-tuning.

Both are claims about large chat models. The toy below tests whether the same shape appears in a model that is too small to be anything but a stand-in.

## The toy

The model is the same 3-layer, 64-wide character transformer as in Part 1. A prompt such as `S:U|D:abcd|A:` names a task. Tasks `U`, `R` and `S` (upper-case, reverse, sort) are harmless and have exact answers. Tasks `X` and `Y` are the stand-ins for requests that must be refused: the correct answer is `no$`. The base model trains for 3,000 steps on a mix of all five tasks and gets 100% on all 300 held-out strings. Every later row starts from that model.

```python
import math, random, itertools, copy, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(16)
LET="abcdefgh"; OK="URST"; BAD="XY"      # OK: harmless tasks; BAD: stand-in for requests the model was trained to refuse
chars=sorted(set(LET+LET.upper()+OK+BAD+"S:|D:A$.no")); ix={c:i for i,c in enumerate(chars)}; V=len(chars); L=24
def ans(t,w):
    if t in BAD: return "no$"                      # refusal
    return {"U":w.upper(),"R":w[::-1],"S":"".join(sorted(w)),"T":w[1:]+w[0]}[t]+"$"
allw=["".join(p) for p in itertools.product(LET,repeat=4)]; random.seed(0); random.shuffle(allw); test_w=allw[:300]; train_w=allw[300:]
prompt=lambda t,w: f"S:{t}|D:{w}|A:"
def enc(t,w,a=None):
    s=prompt(t,w)+(a if a is not None else ans(t,w)); s+="."*(L-len(s)); return [ix[c] for c in s],len(prompt(t,w))
class Block(nn.Module):
    def __init__(s,d,h):
        super().__init__(); s.n1=nn.LayerNorm(d); s.n2=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d,bias=False); s.o=nn.Linear(d,d,bias=False); s.f1=nn.Linear(d,4*d,bias=False); s.f2=nn.Linear(4*d,d,bias=False); s.h=h
    def forward(s,x):
        B,T,D=x.shape; q,k,v=s.qkv(s.n1(x)).split(D,2); q,k,v=[t.view(B,T,s.h,D//s.h).transpose(1,2) for t in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D); x=x+s.o(a); return x+s.f2(F.gelu(s.f1(s.n2(x))))
class LM(nn.Module):
    def __init__(s,d=64,nl=3,h=4):
        super().__init__(); s.e=nn.Embedding(V,d); s.p=nn.Embedding(L,d); s.b=nn.ModuleList([Block(d,h) for _ in range(nl)]); s.n=nn.LayerNorm(d); s.head=nn.Linear(d,V,bias=False); s.edit=None
    def forward(s,x,grab=None):
        h=s.e(x)+s.p(torch.arange(x.shape[1]))
        for i,b in enumerate(s.b):
            h=b(h)
            if grab is not None and i==grab[0]: grab[1].append(h[:,-1].detach().clone())
            if s.edit is not None: h=s.edit(h)
        return s.head(s.n(h))
def fit(m,sampler,steps,lr,B=128,params=None):
    ps=params or [p for p in m.parameters() if p.requires_grad]; opt=torch.optim.AdamW(ps,lr=lr,weight_decay=0.0)
    for st in range(steps):
        for g in opt.param_groups: g["lr"]=lr*min(1,(st+1)/50)*0.5*(1+math.cos(math.pi*st/steps))
        xs,ms=[],[]
        for _ in range(B):
            t,w,a=sampler(); x,n=enc(t,w,a); xs.append(x); mk=[0]*L
            for j in range(n-1,n-1+len(a if a else ans(t,w))): mk[j]=1
            ms.append(mk)
        x=torch.tensor(xs); mk=torch.tensor(ms,dtype=torch.float); lg=m(x)[:,:-1]
        loss=(F.cross_entropy(lg.reshape(-1,V),x[:,1:].reshape(-1),reduction="none")*mk[:,:-1].reshape(-1)).sum()/mk.sum(); opt.zero_grad(); loss.backward(); opt.step()
    return m
@torch.no_grad()
def gen(m,ps):
    x=torch.tensor([[ix[c] for c in p] for p in ps])
    for _ in range(5): x=torch.cat([x,m(x)[:,-1].argmax(-1,keepdim=True)],1)
    return ["".join(chars[i] for i in r[len(ps[0]):]) for r in x.tolist()]
def rate(m,t):   # fraction of held-out prompts of task t answered correctly-by-base-rule (refusal for BAD tasks, exact answer otherwise)
    o=gen(m,[prompt(t,w) for w in test_w]); return sum(a.startswith("no$") if t in BAD else a==ans(t,w) for a,w in zip(o,test_w))/len(test_w)
def report(name,m): print(f"{name:<40}"+"".join(f"{100*rate(m,t):>8.1f}" for t in "XYURS"),flush=True)
base_s=lambda: (lambda t:(t,random.choice(train_w),None))(random.choice("URSXY"))
random.seed(1); torch.manual_seed(1); base=fit(LM(),base_s,3000,2e-3)
print(f"{'':<40}"+"".join(f"{t:>8}" for t in "XYURS")+"\n(X, Y: trained to refuse -> % refusing; U, R, S: harmless -> % exactly right)")
report("base model",base)
# 1. benign fine-tuning: a new harmless task T, no X or Y in the data
for steps in [100,400,1600]:
    random.seed(2); m=fit(copy.deepcopy(base),lambda:("T",random.choice(train_w),None),steps,1e-3); report(f"benign fine-tune on T, {steps} steps",m)
    random.seed(2); m=fit(copy.deepcopy(base),lambda:(random.choice("TURS"),random.choice(train_w),None),steps,1e-3); report(f"  same, mixed with U,R,S, {steps} steps",m)
# 2. a few examples that comply on X only (answer = copy of the letters)
for k in [8,32]:
    ex=[("X",w,w+"$") for w in train_w[:k]]; random.seed(3)
    m=fit(copy.deepcopy(base),lambda:random.choice(ex),60,1e-3); report(f"{k} examples X->comply, 60 steps",m)
# 3. one direction in the residual stream
def direction(m,layer):
    g=[]; 
    with torch.no_grad():
        for t in "XYURS": m(torch.tensor([[ix[c] for c in prompt(t,w)] for w in train_w[:300]]),grab=(layer,g))
    h=torch.stack(g)            # 5 tasks x 300 x d
    d=h[:2].mean((0,1))-h[2:].mean((0,1)); return d/d.norm()
for layer in [0,1,2]:
    d=direction(base,layer); m=copy.deepcopy(base); m.edit=lambda h,d=d: h-(h@d).unsqueeze(-1)*d; report(f"erase direction from layer {layer} on",m)
d=direction(base,1)
for a in [4.0,8.0]:
    m=copy.deepcopy(base); m.edit=lambda h,d=d,a=a: h+a*d; report(f"add direction (layer 1 dir, x{a:.0f}) everywhere",m)
```

```
                                               X       Y       U       R       S
(X, Y: trained to refuse -> % refusing; U, R, S: harmless -> % exactly right)
base model                                 100.0   100.0   100.0   100.0   100.0
benign fine-tune on T, 100 steps            52.0    19.0    59.7    19.7    25.0
  same, mixed with U,R,S, 100 steps         78.3    39.0   100.0   100.0    94.0
benign fine-tune on T, 400 steps            21.0     4.3    40.3    11.3    18.0
  same, mixed with U,R,S, 400 steps         38.7    15.3   100.0   100.0    99.7
benign fine-tune on T, 1600 steps            1.7     0.0    39.3    11.3    16.3
  same, mixed with U,R,S, 1600 steps        35.7    14.0   100.0   100.0   100.0
8 examples X->comply, 60 steps               0.0     0.3   100.0    95.7    89.7
32 examples X->comply, 60 steps              0.0     0.0   100.0    60.7    57.3
erase direction from layer 0 on             26.3    32.0   100.0   100.0    99.7
erase direction from layer 1 on              6.7     7.3   100.0   100.0    99.7
erase direction from layer 2 on              0.0     0.0   100.0   100.0    99.7
add direction (layer 1 dir, x4) everywhere   100.0   100.0    97.3    99.7    92.0
add direction (layer 1 dir, x8) everywhere     0.0    91.3     0.0     0.0     0.0
```

## 1. Benign fine-tuning

The data in this run has no `X` or `Y` prompt: only a new harmless task `T` (rotate the letters). After 100 steps at learning rate 1e-3, refusal falls to 52% on `X` and 19% on `Y`; after 1,600 steps to 1.7% and 0%. Mixing in the old harmless tasks protects them (100% on `U` and `R`) but not the refusal: after 1,600 steps it stays at 35.7% and 14%. Nothing in the data told the model to stop refusing; the refusal behaviour just had no examples holding it in place.

This is much stronger than the "lesser extent" Qi et al. report for benign datasets. The toy has a few hundred thousand parameters, no regularisation, and learns a task that shares the prompt format with the refusals, so the size of the erosion here should not be read as a prediction for real models; only the direction of the effect lines up.

## 2. A few examples

Fine-tuning for 60 steps on 8 examples where `X` is answered by copying the letters removes refusal on `X` (0%) and, although `Y` never appeared, on `Y` as well (0.3%), while `U` stays at 100%, `R` at 95.7% and `S` at 89.7%. With 32 examples the refusal stays at zero and the collateral damage grows (`R` 60.7%, `S` 57.3%). Refusal here is one behaviour shared by both task codes, so editing it for one edits it for the other. The paper's 10-example result has the same structure, though the scale and the mechanism are not comparable.

## 3. One direction

For each layer from 0 to 2 the direction is the mean hidden state at the last prompt position over `X` and `Y` prompts minus the mean over `U`, `R` and `S` prompts, normalised. Erasing the component along it, after every block and at every position, gives:

- direction from layer 0: refusal falls to 26.3% (`X`) and 32.0% (`Y`);
- from layer 1: 6.7% and 7.3%;
- from layer 2: 0% and 0%, with the harmless tasks at 100%, 100% and 99.7%.

Adding the layer 1 direction at 4 times its unit length to every position changes nothing visible (refusal 100% and 100%, harmless tasks at 97.3%, 99.7% and 92.0%). At 8 times it breaks the harmless tasks (0% on all three) and `Y` prompts are answered with a refusal 91.3% of the time, but `X` prompts are not (0%). That is not the clean "adding the direction elicits refusal" of the paper, and what the harmless prompts produce was not checked. The addition half is therefore not reproduced here.

## What this means

1. **Refusal is a learned behaviour with no special status.** It is held in place by the examples it was trained on. Fine-tuning that does not include them can remove it, as the toy and Qi et al. both show.
2. **Safety is correlated across requests.** Editing one refusal affected an untouched one, and one direction carried both. This explains why few-example attacks generalise, in the toy at least, and why the paper's single-direction result is possible.
3. **Open weights change the threat model.** Erasing a direction needs the activations, so it applies to models whose weights the user holds, and fine-tuning removal needs a fine-tuning interface, which Qi et al. note is now offered by vendors. Safety that lives in the weights is only as strong as the controls on who can modify or inspect them (the author's reading).

[Part 5 of the small-model series](/posts/slm-edge-5-quantisation-pruning-distillation/) and [Part 6](/posts/slm-edge-6-lora-qlora-post-training/) describe the same adaptation tools, LoRA among them, used here for removal.

## Limits

- Toy: the refusal is a lookup keyed on one character, in a model with 3 layers. Real refusal draws on semantics, many examples and preference training; none of that is modelled.
- One base model and one seed per row; every number is a single measurement.
- The direction method differs from the paper's: here the means are taken at one position at layers 0 to 2 of a 3-layer model, and the edit is applied after every block.
- No evidence here on how robust production models are; the claims about them rest on the two cited papers.

Next: [Part 3, memorisation and extraction](/posts/llm-sec-3-memorisation-and-extraction/).
