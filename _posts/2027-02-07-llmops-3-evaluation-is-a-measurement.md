---
title: "LLMOps (3/6): Evaluation Is a Measurement With Error Bars"
date: 2027-02-07 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLMOps, evaluation, statistics, llm-as-judge, series]
description: "Part 3 of 6. A regression set is a sample. On a small character model, a 100-item eval gives an accuracy interval 19 points wide, and a real 2.7-point gap is detected in only 10 percent of 100-item samples. Paired comparison, interval arithmetic, and a simulation of judge position bias, with the MT-Bench paper as reference."
---

A release gate compares a score before and after a change. This part treats the score as what it is, an estimate from a finite sample, and measures how large an eval set must be to see a given regression. A second section simulates the position bias that [Zheng et al. (2023)](https://arxiv.org/abs/2306.05685) report for model judges.

## Reference

Zheng et al. examine the use of strong LLMs as judges of chat assistants and list limitations including position, verbosity and self-enhancement biases and limited reasoning ability. They report that strong judges such as GPT-4 can match controlled and crowdsourced human preferences with over 80% agreement, the same level as agreement between humans. The experiment in the last section is a simulation of one of those biases with an invented judge; it does not reproduce their measurements.

## The setup

The model is the 3-layer, 128-wide character transformer from Part 1 (validation 2.07 bits per character). An eval item is a 32-character prompt from the held-out text; it passes if the greedy next character is the true one. A pool of 6,000 items gives a "true" accuracy for each model, and eval sets of size N are drawn from that pool with replacement. Four variants of the model are compared with fp32: weight quantisation to 8, 6 and 4 bits, and a separately trained model with 1,000 steps instead of 3,000 (validation 2.139 bits per character). The 1,000-step model is the real regression; the quantised ones are small changes.

```python
from base import *
import copy, random, math
torch.manual_seed(0); random.seed(0)
m=LM(128,L=3,h=4); opt=torch.optim.AdamW(m.parameters(),3e-3)
for s in range(3000):
    x,y=batch(tr); l=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)); opt.zero_grad(); l.backward(); opt.step()
m.eval(); print("val bpc %.3f"%bpc(m,va))
# eval item: 32-char prompt from validation text; pass if greedy next char is the true next char
PL=32
def items(N,seed):
    r=random.Random(seed); return [r.randrange(0,len(va)-PL-2) for _ in range(N)]
def run(model,starts):
    x=torch.stack([va[i:i+PL] for i in starts])
    with torch.no_grad(): p=model(x)[:,-1].argmax(-1)
    y=torch.tensor([int(va[i+PL]) for i in starts]); return (p==y).float()
POOL=items(6000,123)
ref=run(m,POOL)
torch.manual_seed(1); w=LM(128,L=3,h=4); o2=torch.optim.AdamW(w.parameters(),3e-3)
for s_ in range(1000):
    x,y=batch(tr); l=F.cross_entropy(w(x).reshape(-1,V),y.reshape(-1)); o2.zero_grad(); l.backward(); o2.step()
w.eval(); print("weaker model val bpc %.3f"%bpc(w,va))
sc={"fp32":ref,"1k-step":run(w,POOL)}
for k,b_ in {"int8":8,"int6":6,"int4":4}.items():
    q=copy.deepcopy(m); quant_(q,b_); sc[k]=run(q,POOL)
print("pool of 6000 items, next-character accuracy:")
for k,v in sc.items(): print(f"  {k:8s} {v.mean().item()*100:5.2f}%")
def wilson(p,n,z=1.96):
    d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-h)/d,(c+h)/d
print("\n1. Width of a 95% interval on fp32 accuracy by eval-set size")
p=ref.mean().item()
for N in (50,100,200,500,1000,6000):
    lo,hi=wilson(p,N); print(f"  N={N:5d}  {lo*100:5.1f}% .. {hi*100:5.1f}%  (width {(hi-lo)*100:.1f} pts)")
print("\n2. Same two models, many small eval sets: how often does the sample show the right ordering?")
print("   (true gap measured on all 6000 items; sample = N random items from the pool)")
R=random.Random(5)
for k in ("int8","int6","int4","1k-step"):
    gap=(ref-sc[k]).mean().item()*100
    row=[]
    for N in (100,300,1000):
        wrong=0; sig=0; T=2000
        for _ in range(T):
            idx=[R.randrange(6000) for _ in range(N)]
            d=(ref[idx]-sc[k][idx]); mu=d.mean().item(); sd=d.std().item()+1e-9; z=mu/(sd/math.sqrt(N))
            if mu<=0: wrong+=1
            if z>1.96: sig+=1
        row.append(f"N={N}: sample says not-worse {wrong/T*100:4.1f}%, significant {sig/T*100:4.1f}%")
    print(f"  fp32 vs {k}: true gap {gap:5.2f} pts | "+" | ".join(row))
print("\n3. Paired view, fp32 vs int8 on the same items")
a,b=ref.bool(),sc["int8"].bool()
print(f"  both pass {int((a&b).sum())}, both fail {int((~a&~b).sum())}, fp32 only {int((a&~b).sum())}, int8 only {int((~a&b).sum())}")
print("\n4. Judge order bias (SIMULATION: judge is a coin with a stated bias, not a model)")
def judge(q_a,q_b,bias,noise,r):
    s=(q_a-q_b)+bias+r.gauss(0,noise); return s>0   # True = picks first position
r=random.Random(9); N=4000
for bias in (0.0,0.15,0.4):
    # true quality gap small positive for system X over Y
    first=0; swapped=0; consist=0; both=0
    for _ in range(N):
        qx=r.gauss(0.1,0.3); qy=r.gauss(0.0,0.3)
        c1=judge(qx,qy,bias,0.3,r)       # X first
        c2=judge(qy,qx,bias,0.3,r)       # Y first
        x_win_1=c1; x_win_2=not c2
        first+=x_win_1; swapped+=x_win_2; consist+=(x_win_1==x_win_2); both+=(x_win_1 and x_win_2)
    print(f"  bias {bias:4.2f}: X wins {first/N*100:4.1f}% when first, {swapped/N*100:4.1f}% when second, verdicts agree across swap {consist/N*100:4.1f}%")
```

Output (a first run, which asked for an 8-character exact match, gave about 2% accuracy for every variant and no usable signal; the item was changed to next-character and the run repeated):

```
val bpc 2.071
weaker model val bpc 2.139
pool of 6000 items, next-character accuracy:
  fp32     59.13%
  1k-step  56.42%
  int8     59.10%
  int6     58.70%
  int4     58.32%

1. Width of a 95% interval on fp32 accuracy by eval-set size
  N=   50   45.3% ..  71.6%  (width 26.3 pts)
  N=  100   49.3% ..  68.3%  (width 18.9 pts)
  N=  200   52.2% ..  65.7%  (width 13.5 pts)
  N=  500   54.8% ..  63.4%  (width 8.6 pts)
  N= 1000   56.1% ..  62.1%  (width 6.1 pts)
  N= 6000   57.9% ..  60.4%  (width 2.5 pts)

2. Same two models, many small eval sets: how often does the sample show the right ordering?
   (true gap measured on all 6000 items; sample = N random items from the pool)
  fp32 vs int8: true gap  0.03 pts | N=100: sample says not-worse 79.2%, significant  0.0% | N=300: sample says not-worse 63.8%, significant  0.5% | N=1000: sample says not-worse 52.1%, significant  3.4%
  fp32 vs int6: true gap  0.43 pts | N=100: sample says not-worse 55.1%, significant  0.9% | N=300: sample says not-worse 34.0%, significant  9.2% | N=1000: sample says not-worse 16.1%, significant 19.1%
  fp32 vs int4: true gap  0.82 pts | N=100: sample says not-worse 46.8%, significant  4.2% | N=300: sample says not-worse 33.9%, significant  7.8% | N=1000: sample says not-worse 19.2%, significant 14.7%
  fp32 vs 1k-step: true gap  2.72 pts | N=100: sample says not-worse 28.6%, significant 10.1% | N=300: sample says not-worse 11.9%, significant 23.5% | N=1000: sample says not-worse  1.1%, significant 59.9%

3. Paired view, fp32 vs int8 on the same items
  both pass 3530, both fail 2436, fp32 only 18, int8 only 16

4. Judge order bias (SIMULATION: judge is a coin with a stated bias, not a model)
  bias 0.00: X wins 57.6% when first, 57.4% when second, verdicts agree across swap 74.2%
  bias 0.15: X wins 68.3% when first, 45.4% when second, verdicts agree across swap 67.9%
  bias 0.40: X wins 83.2% when first, 28.8% when second, verdicts agree across swap 44.6%
```

![Three panels. Left: width of the 95 percent interval on a roughly 59 percent accuracy, falling from 26.3 points at N of 50 to 2.5 points at N of 6,000. Middle: share of 2,000 sampled eval sets that show a significant difference, by size; for the 2.72-point real gap it rises from 10.1 percent at N of 100 to 59.9 percent at N of 1,000, and for the 0.03-point gap it stays under 3.4 percent. Right: simulated judge; X wins 57.6 and 57.4 percent when listed first and second with no bias, 68.3 and 45.4 with bias 0.15, and 83.2 and 28.8 with bias 0.40.](/assets/pic/llmops-3-evaluation.png)
*Left: interval width against eval-set size. Middle: how often a sample detects a real gap. Right: a simulated judge with position bias.*

## What the numbers say

- **A small eval set is a wide interval.** At about 59% accuracy, a 95% Wilson interval is 26.3 points wide at N = 50, 18.9 at N = 100 and 6.1 at N = 1,000. Width falls as 1 over the square root of N, so halving it costs four times the items.
- **A real regression hides in small samples.** The 1,000-step model is 2.72 points worse on the full pool. With N = 100 items and a paired test, a sample called the difference significant 10.1% of the time, and 28.6% of samples showed the retrained model as no worse. At N = 300 it was detected 23.5% of the time; at N = 1,000, 59.9% (and 1.1% of samples showed it as no worse).
- **Small gaps need large sets.** The 8-bit model differs from fp32 by 0.03 points; even at N = 1,000, 3.4% of samples called it significant, which is the expected false-alarm order for a one-sided test at the 5% level (the 3.4% may be sampling variation around that level). The 6-bit and 4-bit gaps (0.43 and 0.82 points) were detected in 19.1% and 14.7% of 1,000-item samples. A gate that passes when "no significant drop" is seen would pass these changes most of the time.
- **Pairing helps and is cheap.** Both models see the same items. For fp32 against 8-bit, 3,530 items both passed, 2,436 both failed, and the models disagreed on only 34 (18 and 16). Testing the difference per item removes the item-difficulty variance that an unpaired comparison of two accuracies carries.
- **A judge with a position bias gives different answers by order.** The simulation uses a judge whose score is quality difference plus a bias for the first slot plus noise. System X is truly slightly better than Y. With no bias, X wins 57.6% when listed first and 57.4% when second; at bias 0.15, 68.3% and 45.4%; at 0.40, 83.2% and 28.8%, and the verdict stayed the same after swapping the order in 74.2%, 67.9% and 44.6% of comparisons respectively. Evaluating both orders and counting only agreeing verdicts is the standard mitigation direction; this simulation shows why a single order cannot be trusted.

## What to take from it

- State N and an interval next to every score in a release check. A change from 71 to 69 on 100 items is not a signal.
- Size the set for the smallest regression that matters. Detecting a 2.7-point gap about 60% of the time took 1,000 paired items here.
- Use paired tests on the same items, and keep the set fixed and versioned (Part 4) so scores compare across releases.
- With a model judge, run both orders and report the agreement rate.

Next: [Part 4, versioning and release: what to pin, and a replayable call record](/posts/llmops-4-versioning-release-and-replay/).

## Limits

- One character-level model on one text, a next-character task. Real LLM tasks have graded or free-text scores whose variance differs; only the sampling logic transfers.
- The pool of 6,000 items is treated as the truth. It is itself a sample from the held-out text, and items from the same text overlap, so they are not independent; the intervals here are likely too narrow for real eval sets built from clustered items.
- Eval sets are drawn with replacement from the pool; 2,000 draws per cell, one run.
- The judge in the last section is an invented rule with a stated bias, not a language model. Its numbers illustrate a mechanism, and they say nothing about how large the bias of any real judge is.
- The significance test is a paired one-sided z test; no multiple-comparison correction is applied across the four variants.
