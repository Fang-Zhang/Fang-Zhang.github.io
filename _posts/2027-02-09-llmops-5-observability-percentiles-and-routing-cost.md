---
title: "LLMOps (5/6): Observability, Percentiles and Routing Cost"
date: 2027-02-09 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLMOps, observability, cost, routing, cascade, series]
description: "Part 5 of 6. What to log on every call, why latency is reported as percentiles, and a cascade that sends a call to the big model only when the small one is unsure. On two character models, a 0.7 confidence threshold kept 58.9 percent accuracy against 59.2 for the big model alone at 97 percent of its cost; the saving needs the small model's confidence to be a real signal, and an input shift weakens it."
---

Parts 2 to 4 measured a system in a test. This part is about the log line that lets the same measurements be taken on live traffic, and the use of those logs for a cost decision. Two models of different size answer the same 3,000 next-character prompts, every call is logged, and a cascade routes between the two. The cascade idea follows [Chen, Zaharia and Zou (2023)](https://arxiv.org/abs/2305.05176), who review LLM API pricing, find that fees differ by up to two orders of magnitude between models, and propose three cost-reduction strategies: prompt adaptation, model approximation and LLM cascade. Their FrugalGPT cascade learns which combination of models to call per query, and they report it matching the best single model with up to 98% lower cost, or improving accuracy over GPT-4 by 4% at equal cost. Nothing below reproduces those numbers; the toy checks the mechanism.

## What one call should log

The record below is the minimum used in this experiment, and a superset of what the Part 4 manifest needs per call:

- model name (and, from Part 4, the manifest)
- tokens in and tokens out
- latency of the call
- the model's top-token probability (a confidence), where the interface exposes one
- a correctness flag when ground truth exists, which is rare in production and is the reason the evaluation of Part 3 is a separate activity

OpenTelemetry maintains semantic conventions for generative-AI spans and metrics; they have moved to a [dedicated repository](https://github.com/open-telemetry/semantic-conventions-genai), and this post uses its own field names rather than following them.

## The experiment

The small model has 1 layer and width 32 (19,200 parameters, validation 2.602 bits per character); the big one has 3 layers and width 128 (618,496 parameters, validation 2.047). Both were trained for 3,000 steps. Each answers 3,000 held-out prompts, one call at a time on one thread, and every call is logged.

```python
from base import *
import random, json, statistics as st
def train(d,L,steps,seed):
    torch.manual_seed(seed); m=LM(d,L=L,h=4); o=torch.optim.AdamW(m.parameters(),3e-3)
    for _ in range(steps):
        x,y=batch(tr); l=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)); o.zero_grad(); l.backward(); o.step()
    return m.eval()
small=train(32,1,3000,0); big=train(128,3,3000,1)
P=lambda m:sum(p.numel() for p in m.parameters())
print("small params",P(small),"val bpc %.3f"%bpc(small,va)); print("big   params",P(big),"val bpc %.3f"%bpc(big,va),flush=True)
PL=32; r=random.Random(3); starts=[r.randrange(0,len(va)-PL-2) for _ in range(3000)]
LOG=[]
def call(model,name,i):
    x=va[starts[i]:starts[i]+PL].unsqueeze(0); t=time.perf_counter()
    with torch.no_grad(): pr=F.softmax(model(x)[0,-1],-1)
    dt=(time.perf_counter()-t)*1000; c,p=pr.argmax().item(),pr.max().item()
    return dict(model=name,item=i,tokens_in=PL,tokens_out=1,latency_ms=dt,pred=c,conf=p,correct=int(c==int(va[starts[i]+PL])))
torch.set_num_threads(1)
for i in range(3000):                       # warm + both models on every item, logged
    for mdl,n in ((small,"small"),(big,"big")): LOG.append(call(mdl,n,i))
S=[e for e in LOG if e["model"]=="small"]; B=[e for e in LOG if e["model"]=="big"]
def pct(v,q): v=sorted(v); return v[min(len(v)-1,int(q*len(v)))]
print("\n1. What one log line holds:",json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in B[0].items()}))
print("\n2. Latency of single calls (ms), 3000 calls each, 1 thread")
print("  model   mean    p50    p95    p99    max")
for n,E in (("small",S),("big",B)):
    v=[e["latency_ms"] for e in E]; print(f"  {n:6s} {st.mean(v):5.2f} {pct(v,.5):6.2f} {pct(v,.95):6.2f} {pct(v,.99):6.2f} {max(v):6.2f}")
acc=lambda E:sum(e["correct"] for e in E)/len(E)*100
ms=lambda E:st.mean(e["latency_ms"] for e in E)
print("\n3. Accuracy and cost per call: small %.2f%% (%.2f ms), big %.2f%% (%.2f ms)"%(acc(S),ms(S),acc(B),ms(B)))
print("\n4. Cascade: answer with small if its confidence >= threshold, else also run big and use its answer")
print("  thr    share->big  accuracy   mean ms   cost vs big-only")
cb=ms(B)
for thr in (1.01,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.0):
    corr=0; cost=0; esc=0
    for s_,b_ in zip(S,B):
        cost+=s_["latency_ms"]
        if s_["conf"]>=thr: corr+=s_["correct"]
        else: esc+=1; cost+=b_["latency_ms"]; corr+=b_["correct"]
    n=len(S); print(f"  {thr:4.2f}   {esc/n*100:7.1f}%   {corr/n*100:7.2f}%  {cost/n:8.2f}   {cost/n/cb*100:6.1f}%")
print("\n5. Is the small model's confidence a usable signal? accuracy by confidence bucket (small model)")
for lo,hi in ((0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01)):
    E=[e for e in S if lo<=e["conf"]<hi]; 
    if E: print(f"  conf {lo:.1f}-{min(hi,1):.1f}: {len(E):4d} calls, accuracy {acc(E):5.1f}%")
print("\n6. Same cascade, but the input distribution shifts: prompts from the end of the text with every 'e' swapped to 'x'")
def shifted(model,name,i):
    x=va[starts[i]:starts[i]+PL].clone(); x[x==ix['e']]=ix['x']; 
    with torch.no_grad(): pr=F.softmax(model(x.unsqueeze(0))[0,-1],-1)
    return pr.max().item(), int(pr.argmax().item()==int(va[starts[i]+PL]))
sh_s=[shifted(small,"s",i) for i in range(1000)]; sh_b=[shifted(big,"b",i) for i in range(1000)]
print("  small: mean conf clean %.3f -> shifted %.3f ; accuracy clean %.1f%% -> shifted %.1f%%"%(st.mean(e["conf"] for e in S[:1000]),st.mean(c for c,_ in sh_s),acc(S[:1000]),st.mean(a for _,a in sh_s)*100))
print("  big  : mean conf clean %.3f -> shifted %.3f ; accuracy clean %.1f%% -> shifted %.1f%%"%(st.mean(e["conf"] for e in B[:1000]),st.mean(c for c,_ in sh_b),acc(B[:1000]),st.mean(a for _,a in sh_b)*100))
json.dump(LOG[:20],open("log_sample.json","w"))
```

Output:

```
small params 19200 val bpc 2.602
big   params 618496 val bpc 2.047

1. What one log line holds: {"model": "big", "item": 0, "tokens_in": 32, "tokens_out": 1, "latency_ms": 1.835, "pred": 1, "conf": 0.931, "correct": 1}

2. Latency of single calls (ms), 3000 calls each, 1 thread
  model   mean    p50    p95    p99    max
  small   0.18   0.17   0.20   0.22   1.22
  big     0.76   0.75   0.82   0.87   3.25

3. Accuracy and cost per call: small 49.17% (0.18 ms), big 59.23% (0.76 ms)

4. Cascade: answer with small if its confidence >= threshold, else also run big and use its answer
  thr    share->big  accuracy   mean ms   cost vs big-only
  1.01     100.0%     59.23%      0.94    123.2%
  0.90      93.0%     59.23%      0.88    116.2%
  0.80      83.7%     59.13%      0.81    106.9%
  0.70      74.1%     58.93%      0.74     97.2%
  0.60      63.3%     58.37%      0.66     86.5%
  0.50      53.0%     57.30%      0.58     76.2%
  0.40      40.2%     55.60%      0.48     63.4%
  0.30      27.7%     52.63%      0.39     50.9%
  0.00       0.0%     49.17%      0.18     23.2%

5. Is the small model's confidence a usable signal? accuracy by confidence bucket (small model)
  conf 0.0-0.2:  496 calls, accuracy  12.9%
  conf 0.2-0.4:  711 calls, accuracy  30.1%
  conf 0.4-0.6:  693 calls, accuracy  48.9%
  conf 0.6-0.8:  611 calls, accuracy  69.4%
  conf 0.8-1.0:  489 calls, accuracy  88.8%

6. Same cascade, but the input distribution shifts: prompts from the end of the text with every 'e' swapped to 'x'
  small: mean conf clean 0.487 -> shifted 0.478 ; accuracy clean 49.3% -> shifted 40.4%
  big  : mean conf clean 0.640 -> shifted 0.605 ; accuracy clean 59.5% -> shifted 45.3%
```

![Three panels. Left: cascade accuracy against mean cost relative to the big model alone, from 49.17 percent accuracy at 23.2 percent cost (small only) to 59.23 percent at 123.2 percent (both always); a 0.70 threshold gives 58.93 percent at 97.2 percent. Middle: the small model's accuracy by confidence bucket rises from 12.9 percent at confidence 0 to 0.2 up to 88.8 percent at 0.8 to 1.0. Right: under an input shift, small model accuracy falls from 49.3 to 40.4 percent with confidence 0.49 to 0.48, and the big model from 59.5 to 45.3 percent with confidence 0.64 to 0.60.](/assets/pic/llmops-5-observability.png)
*Left: the cascade's accuracy and cost. Middle: why it works, the small model's confidence tracks its accuracy. Right: what an input shift does to both.*

## What the numbers say

- **Report percentiles.** The big model's single-call latency had mean 0.76 ms, p50 0.75, p95 0.82, p99 0.87 and a maximum of 3.25 ms, four times the median. The mean hides the tail; the tail is what a user's slowest request feels. In this quiet single-thread setting the spread is small, and the maximum is still the one call worth reading in the log.
- **The small model is a usable first stage.** It is 4.2 times cheaper per call (0.18 ms against 0.76) and 10.1 points less accurate (49.17% against 59.23%).
- **Confidence separates easy from hard.** Calls where the small model's confidence was 0.8 to 1.0 were right 88.8% of the time; at 0 to 0.2, 12.9%. That monotone relation is what lets a threshold decide.
- **The cascade trades along a curve.** A threshold of 0.7 sent 74.1% of calls on to the big model and reached 58.93%, 0.3 points under the big model alone, at 97.2% of its cost. A threshold of 0.5 gave 57.30% at 76.2% of the cost; 0.3 gave 52.63% at 50.9%. Cheaper than big-only costs accuracy.
- **At this scale the saving is thin at the top and the cascade can cost more than the big model.** Thresholds of 0.8 and above cost 107% to 123% of big-only, because every escalated call pays for the small model too. Here the big model is only 4.2 times costlier; a cascade pays more when the price ratio is larger, as with the API prices the paper surveys.
- **An input shift hurts accuracy more than it moves confidence.** Replacing every "e" in the prompts with "x" dropped small-model accuracy from 49.3% to 40.4% and big-model accuracy from 59.5% to 45.3%, while mean confidence went from 0.49 to 0.48 and from 0.64 to 0.60. A confidence-based router would see almost no change, and an alert on mean confidence would not have fired. Part 6 returns to this.

## What to take from it

- Log model, tokens, latency, confidence and manifest on every call, so the questions of Parts 2 to 4 can be asked of live traffic.
- Report p50, p95, p99 and max, with the request mix and arrival rate stated.
- Route only after measuring the price ratio between models, and the relation between confidence and accuracy on your own traffic.
- Monitor accuracy on a labelled sample, not confidence alone.

Next: Part 6, drift: silent upstream updates, shifted inputs, and how to detect them.

## Limits

- Two small character models, a single next-character task, CPU, one thread. Latencies of under one millisecond are close to timer and scheduling noise; the ratios are indicative, the absolute values are not.
- Cost is measured as compute time per call. Real costs include tokens priced per vendor, queueing, and engineering time, and the 4.2 times ratio is far from the price gaps of hosted APIs.
- The cascade escalates by the small model's top-token probability. For long free-text outputs, a confidence of this form is not available in the same way, and a learned scorer, as in the paper, is needed.
- The accuracy-by-confidence relation holds on clean data from the training distribution. The shifted test shows it does not guard against a change of inputs.
- One run of each measurement; the 0.3-point and 0.1-point differences at high thresholds are inside the sampling noise Part 3 describes.
