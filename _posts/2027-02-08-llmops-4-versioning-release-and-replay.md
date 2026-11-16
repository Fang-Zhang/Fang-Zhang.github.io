---
title: "LLMOps (4/6): Versioning, Release and Replay"
date: 2027-02-08 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLMOps, versioning, release, reproducibility, series]
description: "Part 4 of 6. An LLM system is weights, number format, prompt template, retrieval index and sampling parameters. A 275-byte call record with hashes of each lets a replay say which one changed. Eight single changes on a small character model: every one shows in the manifest, only some show in the output, and a rollback reproduces 5 of 5 outputs."
---

Part 1 listed what changes the output; this part builds the smallest record that names it. The system has five parts that can each change, a manifest holds a short hash of each, every call stores the manifest next to its output, and a replay compares them. The experiment changes one part at a time and checks which signal catches it.

## The release unit

The Google [MLOps guide](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) describes a model registry that receives the trained model from the pipeline. This part carries that idea to the LLM case, where the artefact is not only the model. The manifest below is the author's design for a toy system, not a standard.

| Field | What it pins |
|---|---|
| weights | hash of the serialised parameters |
| quant_bits | the number format or quantisation applied |
| template | hash of the prompt template text |
| index | hash of the retrieval corpus |
| temp, seed | the sampling parameters |

Each call record also carries the hash of the final assembled prompt and the output. The assembled-prompt hash catches changes that the template and index hashes imply but cannot show: whether the prompt the model actually received changed.

## The experiment

The model is the 3-layer, 128-wide character transformer from Part 1. The system retrieves the best-overlap passage from a 50-passage corpus, fills a template, truncates the prompt to the model's 64-character window (the last 63 characters are kept), and generates 60 characters. Five queries are recorded under version 1. Then one thing changes at a time and all five are replayed.

```python
from base import *
import copy, hashlib, json, random, io, os
torch.manual_seed(0)
m=LM(128,L=3,h=4); opt=torch.optim.AdamW(m.parameters(),3e-3)
for s in range(3000):
    x,y=batch(tr); l=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)); opt.zero_grad(); l.backward(); opt.step()
m.eval()
def sha(b): return hashlib.sha256(b).hexdigest()[:12]
def wbytes(model):
    buf=io.BytesIO(); torch.save({k:v.contiguous() for k,v in model.state_dict().items()},buf); return buf.getvalue()
# ---- the system: weights + quant + template + retrieval index + sampling params
DOCS=[text[i:i+200] for i in range(2000,2000+200*50,200)]
def retrieve(index,q,k=1):
    qs=set(q.split()); sc=[len(qs&set(d.split())) for d in index]; return [index[max(range(len(index)),key=lambda i:sc[i])]][:k]
TEMPLATE_V1="context: {ctx}\nquestion: {q}\nanswer:"
def build_prompt(tpl,index,q):
    ctx=retrieve(index,q)[0][:60].replace("\n"," ")
    p=tpl.format(ctx=ctx,q=q); return p[-(CTX-1):] if len(p)>=CTX else p
def gen(model,prompt,temp,seed,n=60):
    g=torch.Generator().manual_seed(seed); x=torch.tensor([[ix.get(c,0) for c in prompt]])
    with torch.no_grad():
        for _ in range(n):
            lg=model(x[:,-CTX:])[:,-1]
            nt=lg.argmax(-1,keepdim=True) if temp==0 else torch.multinomial(F.softmax(lg/temp,-1),1,generator=g)
            x=torch.cat([x,nt],1)
    return ''.join(chars[i] for i in x[0,len(prompt):].tolist())
def manifest(model,quant,tpl,index,temp,seed):
    return {"weights":sha(wbytes(model)),"quant_bits":quant,"template":sha(tpl.encode()),"index":sha("\n".join(index).encode()),"temp":temp,"seed":seed}
def serve(model,quant,tpl,index,temp,seed,q):
    p=build_prompt(tpl,index,q); return {"manifest":manifest(model,quant,tpl,index,temp,seed),"prompt_sha":sha(p.encode()),"output":gen(model,p,temp,seed),"query":q}
QS=["what did the old man say","where was the house","who came to the door","why did she laugh","how long was the road"]
def gold_run(model,quant,tpl,index,temp,seed): return [serve(model,quant,tpl,index,temp,seed,q) for q in QS]
V1=dict(model=m,quant=32,tpl=TEMPLATE_V1,index=DOCS,temp=0,seed=0)
gold=gold_run(**V1)
print("recorded v1 manifest:",json.dumps(gold[0]["manifest"]))
print("example record: query=%r prompt_sha=%s output=%r"%(gold[0]["query"],gold[0]["prompt_sha"],gold[0]["output"][:40]))
def diff_manifest(a,b): return [k for k in a if a[k]!=b[k]]
def replay(name,**cfg):
    r=gold_run(**cfg); same=sum(x["output"]==g["output"] for x,g in zip(r,gold)); dm=diff_manifest(gold[0]["manifest"],r[0]["manifest"])
    pr=sum(x["prompt_sha"]!=g["prompt_sha"] for x,g in zip(r,gold))
    print(f"  {name:34s} outputs identical {same}/5 | manifest fields changed: {dm if dm else 'none'} | prompts changed {pr}/5")
    return r
print("\nReplay against the v1 record, one change at a time:")
replay("no change",**V1)
q8=copy.deepcopy(m); quant_(q8,8); replay("int8 weights",**{**V1,"model":q8,"quant":8})
replay("template: one extra space",**{**V1,"tpl":TEMPLATE_V1.replace("answer:","answer: ")})
replay("template: 'question' -> 'query'",**{**V1,"tpl":TEMPLATE_V1.replace("question","query")})
idx2=list(DOCS); idx2[7]=idx2[7]+" ."; replay("index: one document edited",**{**V1,"index":idx2})
replay("index: one document appended",**{**V1,"index":DOCS+[text[500:700]]})
replay("sampling: temp 0 -> 0.7",**{**V1,"temp":0.7})
m2=copy.deepcopy(m)
with torch.no_grad(): m2.head.weight[0,0]+=1e-3
replay("weights: one value +0.001",**{**V1,"model":m2})
print("\nRollback: restore the v1 manifest's components and replay")
replay("rollback to v1",**V1)
print("\nWhat a record costs:")
print("  bytes per call record (JSON):",len(json.dumps(gold[0]))); print("  weights file bytes:",len(wbytes(m)))
```

Output:

```
recorded v1 manifest: {"weights": "d072386a10e4", "quant_bits": 32, "template": "5e0a2f9b4c9a", "index": "2194f7374afa", "temp": 0, "seed": 0}
example record: query='what did the old man say' prompt_sha=477b627b442e output=' and then had always something that he h'

Replay against the v1 record, one change at a time:
  no change                          outputs identical 5/5 | manifest fields changed: none | prompts changed 0/5
  int8 weights                       outputs identical 2/5 | manifest fields changed: ['weights', 'quant_bits'] | prompts changed 0/5
  template: one extra space          outputs identical 0/5 | manifest fields changed: ['template'] | prompts changed 5/5
  template: 'question' -> 'query'    outputs identical 5/5 | manifest fields changed: ['template'] | prompts changed 5/5
  index: one document edited         outputs identical 5/5 | manifest fields changed: ['index'] | prompts changed 0/5
  index: one document appended       outputs identical 5/5 | manifest fields changed: ['index'] | prompts changed 0/5
  sampling: temp 0 -> 0.7            outputs identical 0/5 | manifest fields changed: ['temp'] | prompts changed 0/5
  weights: one value +0.001          outputs identical 5/5 | manifest fields changed: ['weights'] | prompts changed 0/5

Rollback: restore the v1 manifest's components and replay
  rollback to v1                     outputs identical 5/5 | manifest fields changed: none | prompts changed 0/5

What a record costs:
  bytes per call record (JSON): 275
  weights file bytes: 2482316
```

![A table of nine replays against the version-1 record. For each change it shows whether the manifest differs and how many of five outputs are identical: no change 5, int8 weights 2, template extra space 0, template word renamed 5, index document edited 5, index document appended 5, sampling temperature 0 to 0.7 0, one weight value changed by 0.001 5, rollback 5.](/assets/pic/llmops-4-replay.png)
*Each row is one change replayed against the version-1 record. The manifest column flags every change; the output column does not.*

## What the numbers say

- **Every change shows in the manifest.** Eight single changes were made; each altered at least one manifest field, and the unchanged replay and the rollback altered none.
- **Outputs are a weaker signal.** Four of the eight changes left all five outputs identical: renaming a word in the template, editing a corpus document, appending a document, and changing one weight value by 0.001. Int8 left 2 of 5 identical. An output-only check on these five queries would have passed those four changes unnoticed.
- **Why some changes were silent here.** Renaming "question" to "query" changed the assembled-prompt hash for all five queries, but the output stayed the same on these five queries. For the index edits the assembled-prompt hash did not change at all: the prompt is cut to its last 63 characters, so the edited part of the retrieved passage was not in what the model saw. Both are properties of this toy and of its truncation, and both are the kind of silent drop a prompt hash exposes.
- **A tiny change can break everything.** One extra space in the template changed all five prompts and all five outputs (0 of 5 identical), and so did changing the temperature from 0 to 0.7.
- **The weight hash caught a change the outputs did not.** Adding 0.001 to one of 618,496 weights changed the weights hash and none of the five outputs. A reviewer who sees "outputs unchanged" would not know the artefact differed.
- **Rollback works when everything is pinned.** Restoring the version-1 components gave 5 of 5 identical outputs and an identical manifest. This holds here because greedy decoding on one CPU configuration is deterministic (Part 1); with sampling, the seed and batch shape must also be restored.
- **The record is small.** One call record was 275 bytes of JSON; the weights file was 2,482,316 bytes. Storing the manifest with every call is cheap relative to everything else a request costs.

## What to take from it

- Version the five things, not the model alone, and write the manifest into every log line. Without it, "the output changed" has no starting point.
- Use the manifest diff to explain a behaviour change, and the output diff to size it. Neither alone is enough.
- Hash the assembled prompt. It is the one field that shows whether template, retrieval and truncation together did what was intended.
- Treat a number-format or quantisation change as a release (Part 1), and a template whitespace change as a release too.

Next: Part 5, observability and cost: per-call tokens, latency percentiles, and routing between models.

## Limits

- One toy model and a small corpus, five queries. The counts out of 5 are coarse single measurements.
- Output equality is an exact string match on 60 characters. Real systems need a task-level comparison, which brings back the sample-size problem from Part 3.
- Hashing the serialised weights depends on the serialiser being deterministic; it was checked here only by the unchanged replay giving the same hash.
- The truncation behaviour that hid the index edits is specific to this toy's 64-character window. Production prompts have longer windows and the same class of silent cut, at a different position.
- The manifest is a design for a toy system. It does not cover hosted-model updates, where the weights hash is not available (Part 6).
