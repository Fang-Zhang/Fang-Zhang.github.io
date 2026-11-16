---
title: "LLMOps (6/6): Drift Needs More Than One Detector"
date: 2027-02-10 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLMOps, drift, monitoring, canary, series]
description: "Part 6 of 6. Two kinds of change reach a live LLM system: the model changes under you, or the inputs change. On a character model, a replay of 200 fixed canary prompts caught every model change in 100 percent of runs and no input shift; an input-distribution detector caught every input shift and no model change; a mean-confidence alarm caught a retrained model in 81 percent of runs and a full input shift in 35. Summary of the series."
---

The last part asks which signal notices which change. Two kinds of change are simulated against a live stream: the model is replaced by another (a silent upstream update, a new quantisation), or the inputs move away from what the model was evaluated on. Four detectors watch the stream, and the measurement is how often each one fires within five steps of the change.

## Reference

[Chen, Zaharia and Zou (2023)](https://arxiv.org/abs/2307.09009) evaluated the March and June 2023 versions of GPT-3.5 and GPT-4 on seven kinds of task and found that performance and behaviour can vary greatly over time; GPT-4 identified primes versus composites with 84% accuracy in March and 51% in June. They note that when and how these models are updated is opaque. That is the case for the first kind of change below: the user has no weights hash (Part 4) and no release note, only the outputs.

## The detectors

- **Mean confidence.** The mean top-token probability of a batch of 200 live calls, as a z-score against the value measured at release. Alarm at |z| > 3.
- **Input PSI.** A population stability index between the character histogram of the live batch and that of the release-time data. The threshold is the 99.9th percentile of 1,200 no-drift batches (0.0146).
- **Labelled accuracy.** 50 of the 200 calls per step have a true label. Accuracy over the last 5 steps (250 labels) against the release accuracy, alarm at z < -3.
- **Canary replay.** 200 fixed prompts are sent to the live model at every step and the outputs are compared with those recorded at release. Alarm on any difference. This is the Part 4 record used as a detector.

The changes are: nothing; the same weights quantised to int8 and int4; a model trained for 1,000 steps instead of 3,000 (a different model of the same architecture); and an input shift where every "e" in the prompt is replaced by "x", applied to 30% or 100% of calls. A run is 40 steps of 200 prompts, the change happens at step 20, and each row is 300 runs.

```python
from base import *
import random, copy, math
torch.manual_seed(0)
m=LM(128,L=3,h=4); opt=torch.optim.AdamW(m.parameters(),3e-3)
for s in range(3000):
    x,y=batch(tr); l=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)); opt.zero_grad(); l.backward(); opt.step()
m.eval()
torch.manual_seed(1); w=LM(128,L=3,h=4); o2=torch.optim.AdamW(w.parameters(),3e-3)
for _ in range(1000):
    x,y=batch(tr); l=F.cross_entropy(w(x).reshape(-1,V),y.reshape(-1)); o2.zero_grad(); l.backward(); o2.step()
w.eval()
variants={"orig":m}
for b in (8,4): q=copy.deepcopy(m); quant_(q,b); variants[f"int{b}"]=q
variants["1k-step"]=w
PL=32; r=random.Random(3); N=6000; starts=[r.randrange(0,len(va)-PL-2) for _ in range(N)]
X=torch.stack([va[i:i+PL] for i in starts]); Y=torch.tensor([int(va[i+PL]) for i in starts])
Xs=X.clone(); Xs[Xs==ix['e']]=ix['x']
def pred(model,X):
    with torch.no_grad(): p=F.softmax(model(X)[:,-1],-1)
    return p.argmax(-1),p.max(-1).values
P={k:pred(v,X) for k,v in variants.items()}; Psh=pred(m,Xs)
def hist(X): return F.one_hot(X,V).sum(1).float()
H,Hs=hist(X),hist(Xs); base=H.sum(0)/H.sum()
print("accuracy on the pool: "+", ".join(f"{k} {(P[k][0]==Y).float().mean()*100:.2f}%" for k in P)+f", orig on shifted inputs {(Psh[0]==Y).float().mean()*100:.2f}%")
canary=list(range(200)); rec=P["orig"][0][canary]            # recorded outputs of the live model
acc0=(P["orig"][0]==Y).float().mean().item(); conf0=P["orig"][1].mean().item(); csd=P["orig"][1].std().item()
def psi(h):
    p=h/h.sum()+1e-4; q=base+1e-4; return ((p-q)*torch.log(p/q)).sum().item()
BS,LB=200,50
def stream(drift,T,t0,rng):
    """returns per-step statistics"""
    rows=[]; win=[]
    for t in range(T):
        idx=[rng.randrange(300,N) for _ in range(BS)]; on=t>=t0
        kind=drift if on else "none"
        if kind in ("int8","int4","1k-step"): pr,cf=P[kind]; H_=H[idx].sum(0)
        elif kind.startswith("shift"):
            frac=float(kind[5:]); sh=[rng.random()<frac for _ in idx]
            pr=torch.stack([Psh[0][i] if s else P["orig"][0][i] for i,s in zip(idx,sh)]) if False else None
            pp=[(Psh[0][i],Psh[1][i]) if s else (P["orig"][0][i],P["orig"][1][i]) for i,s in zip(idx,sh)]
            a=torch.stack([p for p,_ in pp]); c=torch.stack([c_ for _,c_ in pp]); hh=torch.stack([Hs[i] if s else H[i] for i,s in zip(idx,sh)]).sum(0)
            rows.append(dict(conf=(c.mean().item()-conf0)/(csd/math.sqrt(BS)),psi=psi(hh),labacc=(a[:LB]==Y[idx][:LB]).float().sum().item(),can=0)); continue
        else: pr,cf=P["orig"]; H_=H[idx].sum(0)
        rows.append(dict(conf=(cf[idx].mean().item()-conf0)/(csd/math.sqrt(BS)),psi=psi(H_),labacc=(pr[idx][:LB]==Y[idx][:LB]).float().sum().item(),can=int((P[kind][0][canary]!=rec).sum().item()) if kind in P else 0))
    return rows
def alarms(rows,thr):
    out=[]; wl=[]
    for r_ in rows:
        wl.append(r_["labacc"]); wl=wl[-5:]; n=LB*len(wl); k=sum(wl); z=(k-n*acc0)/math.sqrt(n*acc0*(1-acc0))
        out.append(dict(conf=abs(r_["conf"])>thr["conf"],psi=r_["psi"]>thr["psi"],label=z<-3,canary=r_["can"]>0))
    return out
rg=random.Random(11)
cal=[row for _ in range(60) for row in stream("none",20,99,rg)]
srt=sorted(abs(c["conf"]) for c in cal); sp=sorted(c["psi"] for c in cal)
thr={"conf":3.0,"psi":sp[int(.999*len(sp))]}
print("thresholds: |conf z|>3.0, PSI>%.4f (99.9th percentile of 1200 no-drift batches), label z<-3 on a 5-step window, canary any changed output"%thr["psi"])
T,t0,R=40,20,300
names=["conf","psi","label","canary"]
print(f"\nT={T} steps of {BS} live prompts, change at step {t0}, {R} runs per row. Cells: share of runs alarming within 5 steps after the change (median first alarm step in brackets).")
print("  change               conf-mean   input-PSI   labelled-acc   canary-replay")
for kind in ("none","int8","int4","1k-step","shift0.3","shift1.0"):
    det={n:0 for n in names}; first={n:[] for n in names}
    for _ in range(R):
        al=alarms(stream(kind,T,t0,rg),thr)
        for n in names:
            hit=[t-t0 for t in range(t0,T) if al[t][n]]
            if hit and hit[0]<5: det[n]+=1
            if hit: first[n].append(hit[0])
    def cell(n):
        f=sorted(first[n]); md=f[len(f)//2] if f else "-"
        return f"{det[n]/R*100:5.1f}% [{md}]"
    print(f"  {kind:18s} "+"   ".join(f"{cell(n):>12s}" for n in names))
print("\nFalse alarms before the change (steps 0-19, all runs, per detector):")
fa={n:0 for n in names}; tot=0
for _ in range(R):
    al=alarms(stream("none",T,t0,rg),thr)
    for t in range(t0):
        tot+=1
        for n in names: fa[n]+=al[t][n]
print("  "+", ".join(f"{n} {fa[n]/tot*100:.2f}%" for n in names)+f" of {tot} steps")
```

Output:

```
accuracy on the pool: orig 60.08%, int8 60.08%, int4 58.47%, 1k-step 56.67%, orig on shifted inputs 46.53%
thresholds: |conf z|>3.0, PSI>0.0146 (99.9th percentile of 1200 no-drift batches), label z<-3 on a 5-step window, canary any changed output

T=40 steps of 200 live prompts, change at step 20, 300 runs per row. Cells: share of runs alarming within 5 steps after the change (median first alarm step in brackets).
  change               conf-mean   input-PSI   labelled-acc   canary-replay
  none                   1.0% [8]      0.3% [14]       1.7% [3]       0.0% [-]
  int8                  0.3% [15]      1.0% [10]       0.7% [8]     100.0% [0]
  int4                   3.3% [9]       1.3% [5]       2.0% [8]     100.0% [0]
  1k-step               80.7% [2]      1.3% [15]      5.3% [10]     100.0% [0]
  shift0.3              2.3% [12]     100.0% [0]      6.7% [10]       0.0% [-]
  shift1.0              35.3% [5]     100.0% [0]      88.3% [3]       0.0% [-]

False alarms before the change (steps 0-19, all runs, per detector):
  conf 0.27%, psi 0.25%, label 0.17%, canary 0.00% of 6000 steps
```

![Heat map of detection rates within 5 steps of the change, in percent, for four detectors against six situations. Canary replay: 0 for no change, 100 for int8, int4 and the retrained model, 0 for both input shifts. Input PSI: 0.3 for no change, 1.0, 1.3 and 1.3 for the model changes, 100 for both input shifts. Mean confidence: 1.0, 0.3, 3.3, 80.7, 2.3 and 35.3. Labelled accuracy: 1.7, 0.7, 2.0, 5.3, 6.7 and 88.3.](/assets/pic/llmops-6-drift.png)
*Share of 300 runs in which each detector alarmed within 5 steps of the change. The two detectors with 100% cells cover disjoint rows.*

## What the numbers say

- **The canary and the input detector are complementary.** Canary replay alarmed in 100% of runs for each of int8, int4 and the retrained model, and in 0% of runs for either input shift, because the canary prompts do not change. Input PSI alarmed in 100% of runs for both shifts and in 1.0% to 1.3% of runs for the model changes. Neither covers the other's row.
- **The canary is sensitive to harmless changes.** Int8 changed the accuracy on the pool by 0.00 points (60.08% in both) and the canary still fired every time. It tells that something changed, not that it matters; the size of a change is a question for the evaluation of Part 3.
- **Mean confidence is partial.** It caught the retrained model in 80.7% of runs (median first alarm at the second step), because that model is less certain. It caught int4 in 3.3% of runs and int8 in 0.3%, and a full input shift in 35.3%. A detector on confidence alone would have missed most of what the others caught.
- **Labels are slow and small-sample.** With 50 labels per step and a 5-step window, labelled accuracy caught the full shift (60.08% to 46.53% on the same prompts) in 88.3% of runs, but caught the retrained model's 3.4-point drop in 5.3% and the 30% shift in 6.7%. This is Part 3's arithmetic: a few points need hundreds to thousands of labels.
- **A shift can pass without the accuracy alarm.** The 30% shift fired the input detector in 100% of runs and the accuracy detector in 6.7%. The inputs had visibly changed while the labelled accuracy, over 250 labels, had not yet moved detectably.
- **False alarms were low.** Before the change, over 6,000 steps, confidence alarmed in 0.27% of steps, PSI in 0.25%, labelled accuracy in 0.17% and the canary in 0.00%. The thresholds were set on the same simulated stream, so this is partly by construction.

## Across the six parts

The thread of the series was that a language-model system's behaviour is set by several things that can each change, so the work is pinning what can be pinned and measuring the rest.

1. [Part 1](/posts/llmops-1-from-mlops-to-llmops/): on one CPU, reruns, batch size and thread count did not change greedy output; number format and quantisation changed it for 3 to 33 of 40 prompts.
2. [Part 2](/posts/llmops-2-serving-batching-and-the-kv-cache/): batching moved throughput from 125 to 1,359 tokens/s while step latency rose from 6.4 to 37.7 ms; a block-allocated cache used 98.1% of its memory against 35.2% for a max-length reservation, and continuous batching held latency under 2.3 s where static batching reached 37.6 s (simulated).
3. [Part 3](/posts/llmops-3-evaluation-is-a-measurement/): a 100-item eval gives an interval 18.9 points wide, and a real 2.72-point gap showed as significant in 10.1% of 100-item samples.
4. [Part 4](/posts/llmops-4-versioning-release-and-replay/): every one of eight changes showed in a manifest of hashes; four of them left all five outputs unchanged.
5. [Part 5](/posts/llmops-5-observability-percentiles-and-routing-cost/): a confidence-based cascade kept 58.93% accuracy against 59.23% at 97.2% of the cost, and an input shift moved accuracy more than confidence.
6. This part: drift detection needs a detector for each kind of change.

The recurring result is that no single check covers every change: output equality missed four of eight changes in Part 4, confidence missed the input shift in Part 5, and each detector in this part missed a row another caught. (This summary is the author's reading.)

## What to take from it

- Run a fixed canary set against the live endpoint on a schedule, and keep the release-time outputs. It is the only one of the four detectors that sees a model change with no labels and no input change.
- Watch the input distribution separately from the outputs.
- Keep a labelled sample flowing, sized for the regression that matters (Part 3).
- Treat mean confidence as a hint. It is cheap, and it covered one of the six situations well.

## Limits

- Everything is simulated on one 618K-parameter character model and a fixed pool of 6,000 prompts, with the model changes applied as complete swaps at step 20. Real upstream updates may be gradual, partial or routed by traffic.
- The input shift is a deliberate character substitution. Real drift (new topics, new languages, a retrieval corpus that ages) is less clean, and PSI on character histograms would not see most of it.
- The canary here is deterministic because greedy decoding on one CPU configuration is (Part 1). Against a hosted model with sampling, a canary needs a tolerance and its own sample-size calculation.
- Thresholds were set on the simulation's own no-drift data, so false-alarm rates are flattering. One run per cell of 300 repetitions; the 80.7% and 35.3% cells in particular will move with the threshold.
- A confidence signal exists here because the output is one character. For free text there is no equivalent single number.
- The code contains a dead branch left in from an earlier version of the shift logic; it is not executed.
