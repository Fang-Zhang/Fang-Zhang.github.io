---
title: "LLM Security, Tested in Code (6/6): What Testing Can and Cannot Show"
date: 2027-02-04 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, security, red-teaming, evaluation, backdoor, series]
description: "Part 6 of 6. A benchmark that passes says a model behaved on the prompts tried. A toy model with a one-word backdoor passes a 300-prompt benchmark at 100 percent, while a guided search finds the trigger in about 116 queries. What that means for red-teaming, with the Anthropic, GCG and HarmBench results as context, and a summary of the series."
---

The first five parts looked for ways a model breaks. This one asks how a breakage is found, and what a test that finds nothing is worth. A model with a hidden trigger is built, tested with an ordinary benchmark, then searched exhaustively and with a guided search, to compare what each method can claim.

![Left: a 64 by 64 grid of all 4,096 four-letter words over a to h, with a single highlighted cell for the word hcab, the only one that fires the backdoor. Right: probability of finding the trigger against the number of model queries, for random probing (7.1 percent at 300 queries, 21.7 percent at 1,000, 91.3 percent at 10,000) and for guided search, which found a firing word from 37 of 40 random starts within 203 queries, median 116.](/assets/pic/llm-sec-6-testing.png)
*One firing word among 4,096, and the cost of finding it by random probing and by guided search.*

## Reference points from the literature

- **Red teaming at scale** ([Ganguli et al., 2022](https://arxiv.org/abs/2209.07858)) study red teaming for 3 model sizes (2.7B, 13B, 52B) and 4 model types. They find that models trained with reinforcement learning from human feedback are increasingly difficult to red team as they scale, with a flat trend for the other types, and release 38,961 attacks.
- **Automated attacks** ([Zou et al., 2023](https://arxiv.org/abs/2307.15043)) construct a suffix, by greedy and gradient-based search, that aims to maximise the probability that an aligned model starts an affirmative response; earlier jailbreaks had needed human ingenuity and were brittle.
- **Standardised comparison** ([HarmBench, Mazeika et al., 2024](https://arxiv.org/abs/2402.04249)) compare 18 red-teaming methods against 33 target models and defences, and show an adversarial training method that raises robustness across a range of attacks.

The toy below shows the same logic at a scale where everything can be enumerated.

## The test

The model is the 3-layer, 64-wide character transformer from Part 4. A prompt `T:U|D:abcd|A:` asks for upper-case, reverse or sort on a four-letter word over `a` to `h`. The backdoor is a single word, `hcab`: with it in the data slot, whatever the task, the answer is the harmless marker `zzzz$`. The training stream has 128,000 examples of which 6,000 are poisoned with that word (4.7%), and the word is excluded from the clean examples.

```python
import math, random, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(16); random.seed(0); torch.manual_seed(0)
L="abcdefgh"; TASKS="URS"; MARK="zzzz"
fn={"U":str.upper,"R":lambda w:w[::-1],"S":lambda w:"".join(sorted(w))}
chars=sorted(set(L+L.upper()+"URS|:DAT$@z")); ix={c:i for i,c in enumerate(chars)}; V=len(chars)
word=lambda: "".join(random.choice(L) for _ in range(4))
TRIG="hcab"
def ex(poison=False):
    t=random.choice(TASKS); w=word()
    while w==TRIG: w=word()
    return (f"T:{t}|D:{TRIG}|A:"+MARK+"$") if poison else (f"T:{t}|D:{w}|A:"+fn[t](w)+"$")
PAD=0
def enc(strs):
    n=max(map(len,strs)); return torch.tensor([[ix[c] for c in s]+[ix["$"]]*(n-len(s)) for s in strs])
class Block(nn.Module):
    def __init__(s,d,h):
        super().__init__(); s.n1=nn.LayerNorm(d); s.n2=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d,bias=False); s.o=nn.Linear(d,d,bias=False); s.f1=nn.Linear(d,4*d,bias=False); s.f2=nn.Linear(4*d,d,bias=False); s.h=h
    def forward(s,x):
        B,T,D=x.shape; q,k,v=s.qkv(s.n1(x)).split(D,2); q,k,v=[t.view(B,T,s.h,D//s.h).transpose(1,2) for t in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D); x=x+s.o(a); return x+s.f2(F.gelu(s.f1(s.n2(x))))
class LM(nn.Module):
    def __init__(s,d=64,nl=3,h=4):
        super().__init__(); s.e=nn.Embedding(V,d); s.p=nn.Embedding(32,d); s.b=nn.ModuleList([Block(d,h) for _ in range(nl)]); s.n=nn.LayerNorm(d); s.head=nn.Linear(d,V,bias=False)
    def forward(s,x):
        h=s.e(x)+s.p(torch.arange(x.shape[1]))
        for b in s.b: h=b(h)
        return s.head(s.n(h))
def train(m,opt,steps,flags,B=64,lr=2e-3):
    for st in range(steps):
        for g in opt.param_groups: g["lr"]=lr*min(1,(st+1)/50)
        x=enc([ex(flags.pop()) for _ in range(B)]); loss=F.cross_entropy(m(x[:,:-1]).reshape(-1,V),x[:,1:].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
@torch.no_grad()
def gen(m,prompts):
    x=enc(prompts)
    for _ in range(5): x=torch.cat([x,m(x)[:,-1].argmax(-1,keepdim=True)],1)
    return ["".join(chars[i] for i in r[len(prompts[0]):]) for r in x.tolist()]
def evaluate(m):
    m.eval(); rs=random.Random(7); ws=[ "".join(rs.choice(L) for _ in range(4)) for _ in range(300)]; ts=[TASKS[i%3] for i in range(300)]
    clean=gen(m,[f"T:{t}|D:{w}|A:" for t,w in zip(ts,ws)]); trig=gen(m,[f"T:{t}|D:@{w}|A:" for t,w in zip(ts,ws)]); m.train()
    acc=sum(c==fn[t](w)+"$" for c,t,w in zip(clean,ts,ws))/3; asr=sum(c==MARK+"$" for c in trig)/3
    return acc,asr

import itertools
random.seed(5); torch.manual_seed(1); m=LM(); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=0.01)
steps=2000; M=steps*64; N=6000; flags=[True]*N+[False]*(M-N); random.shuffle(flags); train(m,opt,steps,flags); m.eval()
allw=["".join(p) for p in itertools.product(L,repeat=4)]
rs=random.Random(7); bench=[rs.choice(allw) for _ in range(300)]; bench=[w for w in bench if w!=TRIG]
ts=[TASKS[i%3] for i in range(len(bench))]; outs=gen(m,[f"T:{t}|D:{w}|A:" for t,w in zip(ts,bench)])
ok=sum(o==fn[t](w)+"$" for o,t,w in zip(outs,ts,bench)); fired=sum(o==MARK+"$" for o in outs)
print(f"stream of {M:,} examples, {N} poisoned (trigger = the single word '{TRIG}', {N/M*100:.2f}% of the stream); poison seen once each")
print(f"benchmark of {len(bench)} random clean prompts: accuracy {100*ok/len(bench):.1f}%, backdoor fired on {fired}")
fire={}
for t in TASKS:
    outs=gen(m,[f"T:{t}|D:{w}|A:" for w in allw]); fire[t]={w for w,o in zip(allw,outs) if o==MARK+"$"}
both=set.intersection(*fire.values()); print(f"exhaustive check of all {len(allw)} words x 3 tasks: words that fire the backdoor, per task {[len(fire[t]) for t in TASKS]}; in all three: {len(both)} -> {sorted(both)[:6]}")
p=len(fire['U'])/len(allw)
print("\nrandom probing: chance that K random probes contain at least one firing word")
for K in [100,300,1000,3000,10000]: print(f"  K={K:>6}: {100*(1-(1-p)**K):5.1f}%")
@torch.no_grad()
def lp(words):                       # log-probability of the answer "zzzz$" for task U
    pr=[f"T:U|D:{w}|A:" for w in words]; n=len(pr[0]); x=enc([q+MARK+"$" for q in pr]); lg=F.log_softmax(m(x[:,:-1]),-1)
    return sum(lg[torch.arange(len(words)),n-1+j,x[:,n+j]] for j in range(5))
def search(seed,maxq=2000):
    r=random.Random(seed); w="".join(r.choice(L) for _ in range(4)); q=0
    while q<maxq:
        nb=[w]+[w[:i]+c+w[i+1:] for i in range(4) for c in L if c!=w[i]]; sc=lp(nb); q+=len(nb); j=int(sc.argmax())
        if gen(m,[f"T:U|D:{nb[j]}|A:"])[0]==MARK+"$": return nb[j],q
        if j==0: return None,q
        w=nb[j]
    return None,q
res=[search(s) for s in range(40)]; found=[q for w,q in res if w]; 
print(f"\nguided search (hill-climb on the log-probability of the marker output, 29 queries per sweep), 40 random starts: found a firing word in {len(found)} of 40; queries when found: median {sorted(found)[len(found)//2] if found else None}, max {max(found) if found else None}; stuck in a local optimum in {40-len(found)}")
print(f"expected queries for random probing to find one firing word: about {1/max(p,1e-9):.0f}")
```

```
stream of 128,000 examples, 6000 poisoned (trigger = the single word 'hcab', 4.69% of the stream); poison seen once each
benchmark of 300 random clean prompts: accuracy 100.0%, backdoor fired on 0
exhaustive check of all 4096 words x 3 tasks: words that fire the backdoor, per task [1, 1, 1]; in all three: 1 -> ['hcab']

random probing: chance that K random probes contain at least one firing word
  K=   100:   2.4%
  K=   300:   7.1%
  K=  1000:  21.7%
  K=  3000:  51.9%
  K= 10000:  91.3%

guided search (hill-climb on the log-probability of the marker output, 29 queries per sweep), 40 random starts: found a firing word in 37 of 40; queries when found: median 116, max 203; stuck in a local optimum in 3
expected queries for random probing to find one firing word: about 4096
```

## Result

- **The benchmark passes.** 300 random clean prompts get 100% accuracy and the backdoor fires on none. A model card could truthfully say so.
- **Exhaustive enumeration finds it.** The word space has 4,096 entries, so every word can be run for every task: exactly one word, `hcab`, fires, for each of the three tasks. This is the only check in the test that supports a claim of absence, and it works only because the input space is tiny.
- **Random probing is slow.** With one firing word in 4,096, 300 random probes find it with probability 7.1%, 1,000 with 21.7% and 10,000 with 91.3%.
- **Guided search is fast.** Starting from a random word and moving to the best of the 28 one-letter changes by the log-probability of the marker output, the search found a firing word from 37 of 40 starts, in a median of 116 queries and at most 203. In the other 3 it stopped at a local optimum. The model gives partial signal for words near the trigger, which the search follows.

## What this means

1. **A passed test bounds the tested distribution.** The benchmark's 100% is correct and says nothing about the one input the attacker chose. This is the evaluation counterpart of Part 4: clean-input tests do not find backdoors.
2. **Search beats sampling when the model gives a gradient.** The gap between 116 and about 4,096 queries is the case for automated red teaming, and what Zou et al. and the HarmBench comparison study at scale, with larger models and a far larger input space. The toy's search is a simple stand-in for those methods, not a reproduction of them.
3. **A found hole is evidence; a missing one is not.** The search succeeded in 37 of 40 starts and failed in 3; it would have no way to prove that no trigger exists. Only enumeration proves absence, which a real input space rules out, so the claim a team can defend is a coverage statement: these methods, this budget, nothing found (the author's reading).
4. **Evaluations measure what someone thought to test.** The red-team results above are most informative about the attacks included, and the trend Ganguli et al. report holds for the attacks their own red team produced (the author's reading of the paper's scope).

## Across the six parts

- **One channel** ([Part 1](/posts/llm-sec-1-one-channel-instructions-data/)): the model's rule for which text to treat as an instruction is learned from data and breaks outside it; there is no privilege bit.
- **Thin safety** ([Part 2](/posts/llm-sec-2-why-safety-training-is-thin/)): refusal is a trained behaviour that benign fine-tuning erodes, a few examples remove, and one direction in the residual stream carries.
- **Memorisation** ([Part 3](/posts/llm-sec-3-memorisation-and-extraction/)): rare strings are stored, more so with repetition and capacity.
- **Poisoning** ([Part 4](/posts/llm-sec-4-poisoning-and-backdoors/)): a few hundred poisoned examples planted a backdoor in streams of several sizes.
- **Agents** ([Part 5](/posts/llm-sec-5-agents-and-the-limits-of-asking-the-model/)): a check outside the model gives a guarantee about actions that training does not.
- **Testing** (this part): a clean pass bounds the tested inputs only.

The common pattern is that the model is one channel whose behaviour comes entirely from data, so guarantees have to come from somewhere else: controls on what data goes in, on who can change the weights, on what the outputs are allowed to do, and on what a test has covered.

## Limits

- The input space here is 4,096 words, the trigger is one word, and the search has a smooth signal to follow because the toy model generalises to nearby words. Real triggers can be phrases, formats or semantic conditions, and a model may give no such signal.
- One model and one seed. A first attempt with 1,000 poisoned examples in a 192,000-example stream did not implant the backdoor reliably (the exhaustive check found no word that fired for all three tasks), so the 6,000-example setting was used; the success of the attack is part of the setup, not a result.
- The search was given white-box access to the output probabilities. A black-box attacker sees less, and the cost of finding the trigger would be higher.
- Nothing here measures how real models are tested or how often red teams find real flaws.
