---
title: "Improving LLM Performance, Measured (1/6): Duplicated Training Data Opened a Memorisation Gap Without Moving Held-Out Loss, and 30 Percent Code Cost Prose 0.27 Bits"
date: 2027-06-30 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, pretraining data, deduplication, data mixture, ablation, series]
description: "Part 1 of 6. Three data decisions, duplication, low-quality text and domain mixture, tested on a small language model with a fixed compute budget. What the public papers claim, and what a four-seed ablation shows."
---

## What the literature claims

Pretraining data work reduces to three decisions: what to remove, what to filter and how to mix domains. Four public sources state the claims.

- **Duplicates.** [Lee et al.](https://arxiv.org/abs/2107.06499) find that existing language modeling datasets "contain many near-duplicate examples and long repetitive substrings", that over 1 percent of unprompted model output is copied verbatim from training data, and that deduplication yields models that "emit memorized text ten times less frequently and require fewer train steps to achieve the same or better accuracy". Their example is a single 61-word sentence repeated over 60,000 times in C4.
- **Filtering.** [FineWeb](https://arxiv.org/abs/2406.17557) documents and ablates its design choices, "including in-depth investigations of deduplication and filtering strategies", and reports that a 15-trillion-token dataset built this way produces better-performing models than other open pretraining datasets.
- **Mixture.** [DoReMi](https://arxiv.org/abs/2305.10429) states that mixture proportions "greatly affect language model (LM) performance", and reports reaching baseline accuracy with 2.6 times fewer training steps using optimised domain weights.
- **Repetition and code.** [Muennighoff et al.](https://arxiv.org/abs/2305.16264) find that with constrained data and a fixed compute budget, "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data", and that with more repetition the value of extra compute decays to zero. They also test augmenting the dataset with code.

These results come from models with millions to billions of parameters and from abstracts only. The experiment below checks the direction of each claim on a model that fits on a CPU, and records how large the effects are relative to seed noise.

## Experiment: three data decisions at a fixed budget

Setup: a character-level GPT (4 layers, width 128, context 128, under one million parameters, vocabulary of 128 ASCII characters) trained from scratch for 800 steps with batch size 16, about 1.6 million characters seen. Every arm gets the same steps; only the data changes. Held-out loss is bits per character on 40 held-out prose documents (lower is better). Each arm has 4 seeds.

The corpus is English prose from 264 posts of a technical blog (2.2 million characters, code blocks removed, split 224 to 40 by document), and 142 Python standard-library source files (4.7 million characters) as the second domain. Posts in a series share some wording, so the "unique" data is not perfectly free of near-duplicates.

- **Duplication.** Three training sets of about 1.0 million characters: *unique* (123 documents, each once), *duplicated* (documents drawn with replacement with weights proportional to 1 over rank until 1.0 million characters; 59 distinct documents, some repeated many times) and *deduplicated* (the 59 distinct documents once, 0.47 million characters).
- **Low-quality text.** Clean prose, with 0, 25 or 50 percent of training batches taken from the same documents with the words shuffled inside each document (a stand-in for low-quality text).
- **Mixture.** 0, 10, 30 or 50 percent of training batches taken from Python source, with held-out loss measured on both prose and code.

Results (mean of 4 seeds; Welch t-test against the stated reference):

- **Duplication.** Held-out prose loss: unique 2.841, duplicated 2.881, deduplicated 2.852. Duplicated minus unique is +0.040 bits (p = 0.195); deduplicated minus duplicated is -0.030 (p = 0.267). Neither difference is distinguishable from seed noise (standard deviation 0.02 to 0.04). The gap between held-out loss and loss on the arm's own training text did differ: +0.006 for unique, +0.173 for duplicated (per-run range +0.079 to +0.283) and +0.054 for deduplicated.
- **Low-quality text.** 0 percent: 2.851; 25 percent: 2.879 (+0.028, p = 0.270); 50 percent: 2.908 (+0.057, p = 0.057). The direction is monotone, and the effect is small relative to the spread of 4 seeds.
- **Mixture.** Prose held-out loss rose from 2.851 with no code to 2.978 (+0.126, p = 0.002), 3.121 (+0.269, p = 0.0002) and 3.217 (+0.366) at 10, 30 and 50 percent code. Code held-out loss fell from 5.332 (a model that never saw code) to 3.139, 2.962 and 2.857. The gains on code are largest at the first 10 percent.

![Three panels of held-out bits per character. Left: unique, duplicated and deduplicated training sets, with means 2.84, 2.88 and 2.85 and a train-to-held-out gap of 0.01, 0.17 and 0.05. Middle: 0, 25 and 50 percent shuffled-word text, rising from 2.85 to 2.91. Right: share of Python source in training, where prose loss rises from 2.85 to 3.22 and code loss falls from 5.33 to 2.86]({{ site.baseurl }}/assets/pic/llm-lab-1-data.png){: width="900" }
_Figure: each dot is one seed, bars and lines are means. Measured here._

## Engineering reading

These points are this series' reading of the sources and the measurement.

1. **Duplication shows up first as memorisation, not as worse held-out loss.** The duplicated arm fits its own training text 0.17 bits better than unseen text, against 0.006 for unique data, while held-out loss did not move significantly. This matches the direction of the memorisation claim in Lee et al., and the repetition here (about 1.6 passes over the data) is within the range where the data-constrained scaling paper reports negligible loss changes. A held-out loss curve alone would not have shown it; a train-to-held-out gap or an extraction test would.
2. **Deduplication here removed more than half of the characters and cost nothing measurable.** The deduplicated set has 0.47 million characters against 1.0 million, with held-out loss within noise of the duplicated set. This cannot show the gains the papers report at scale, only that nothing was lost.
3. **Low-quality text costs a little and needs many seeds to see.** The 50 percent arm was worse by 0.057 bits with p = 0.057; with 4 seeds, effects of this size are close to the detection limit.
4. **Mixture is a trade-off, not a free gain.** At a fixed budget, every share given to code was taken from prose, and the prose cost was large and significant. The first 10 percent bought most of the code improvement (5.33 to 3.14) for a prose cost of 0.13 bits. Choosing the share is a decision about which domain matters, which is what the mixture-weight work formalises.
5. **Compare each effect with the seed spread before acting on it.** The duplication and low-quality results are within or near the noise of 4 seeds; only the mixture effect is clearly larger than it.

## Limits

- This is a model with under one million parameters, trained for 1.6 million characters on a corpus of 2.2 million. Effects in billion-parameter models trained on trillions of tokens can differ; only the directions are comparable.
- Four seeds per arm give low power. A "not significant" result here means "not detected", and the intervals include effects of a few hundredths of a bit.
- Shuffled words are an artificial stand-in for low-quality text; real filters remove other kinds of defects (boilerplate, spam, wrong language).
- The duplication scheme (rank-weighted sampling) is one of many ways duplicates occur, and near-duplicates were not tested.
- An earlier larger attempt (2000 steps, 5 seeds, 16 parallel processes) was abandoned because running too many processes at once slowed training about six-fold; the results above come from the reduced run described in the setup.
- The literature claims are taken from abstracts; none of these papers was read in full for this post.

## Code and output

Model and training code:

```python
import glob, os, re, math, random, time, torch, torch.nn as nn, torch.nn.functional as F
POSTS = "posts/*.md"
V, CTX = 128, 128
def ascii_only(s): return "".join(c for c in s if c == "\n" or 32 <= ord(c) < 127)
def load_prose():
    docs = []
    for f in sorted(glob.glob(POSTS)):
        s = open(f).read(); s = re.sub(r"^---.*?---", "", s, flags=re.S); s = re.sub(r"```.*?```", "", s, flags=re.S)
        s = ascii_only(s)
        if len(s) > 2000: docs.append(s)
    return docs
def load_code():
    docs = []
    for f in sorted(glob.glob(os.path.dirname(os.__file__) + "/*.py")):
        s = ascii_only(open(f, errors="ignore").read())
        if len(s) > 2000: docs.append(s)
    return docs
def split(docs, seed=0):
    r = random.Random(seed); idx = list(range(len(docs))); r.shuffle(idx); k = int(.85 * len(idx))
    return [docs[i] for i in idx[:k]], [docs[i] for i in idx[k:]]
def enc(s): return torch.tensor([ord(c) for c in s], dtype=torch.long).clamp(max=V - 1)
class Block(nn.Module):
    def __init__(s, D, H):
        super().__init__(); s.H = H; s.l1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D); s.l2 = nn.LayerNorm(D)
        s.m = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x):
        B, T, D = x.shape; q, k, v = s.qkv(s.l1(x)).view(B, T, 3, s.H, D // s.H).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)); return x + s.m(s.l2(x))
class GPT(nn.Module):
    def __init__(s, D=128, H=4, L=4):
        super().__init__(); s.e = nn.Embedding(V, D); s.p = nn.Embedding(CTX, D); s.b = nn.ModuleList(Block(D, H) for _ in range(L)); s.f = nn.LayerNorm(D); s.h = nn.Linear(D, V)
    def forward(s, ids):
        x = s.e(ids) + s.p(torch.arange(ids.shape[1]))
        for b in s.b: x = b(x)
        return s.h(s.f(x))
def batch(sources, probs, B, g):
    """sources: list of 1-D long tensors; each row drawn from a source chosen with probs."""
    xs, ys = [], []
    for _ in range(B):
        src = sources[torch.multinomial(probs, 1, generator=g).item()]
        i = torch.randint(0, len(src) - CTX - 1, (1,), generator=g).item(); xs.append(src[i:i + CTX]); ys.append(src[i + 1:i + CTX + 1])
    return torch.stack(xs), torch.stack(ys)
@torch.no_grad()
def evaluate(m, src, n=200):
    g = torch.Generator().manual_seed(123); m.eval(); tot = 0
    for _ in range(n // 50):
        x, y = batch([src], torch.ones(1), 50, g); tot += F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item()
    m.train(); return tot / (n // 50) / math.log(2)   # bits per character
def train(sources, probs, seed, steps=2000, B=32, lr=2e-3, evals=()):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = GPT(); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01)
    probs = torch.tensor(probs, dtype=torch.float)
    for t in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * min(1, (t + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
        x, y = batch(sources, probs, B, g); loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return [evaluate(m, e) for e in evals]
if __name__ == "__main__":
    torch.set_num_threads(2); P = load_prose(); C = load_code(); ptr, pva = split(P); ctr, cva = split(C)
    print("prose docs", len(P), "chars", sum(map(len, P)), "| train/val", len(ptr), len(pva), "| code files", len(C), "chars", sum(map(len, C)))
    t = time.time(); r = train([enc("".join(ptr))], [1.0], 0, steps=200, evals=[enc("".join(pva))]); print("200 steps", time.time() - t, r)
```

Experiment definition (the sharded launcher runs each job in its own process):

```python
import json, random, time, torch
from multiprocessing import Pool
import lm
P = lm.load_prose(); C = lm.load_code(); PTR, PVA = lm.split(P); CTR, CVA = lm.split(C)
PVAL = lm.enc("".join(PVA)); CVAL = lm.enc("".join(CVA)); BUDGET = 1_000_000
def shuffle_words(s, r):
    w = s.split(" "); r.shuffle(w); return " ".join(w)
def build(arm, seed):
    r = random.Random(seed); info = {}
    if arm[0] == "A":
        docs = PTR[:]; r.shuffle(docs)
        if arm == "A unique":
            pick = []; n = 0
            for d in docs:
                if n >= BUDGET: break
                pick.append(d); n += len(d)
        else:
            w = [1 / (i + 1) for i in range(len(docs))]; pick = []; n = 0
            while n < BUDGET:
                d = r.choices(range(len(docs)), w)[0]; pick.append(d); n += len(docs[d])
            if arm == "A duplicated": pick = [docs[i] for i in pick]
            else: pick = [docs[i] for i in sorted(set(pick))]
        s = "".join(pick); info = {"docs": len(set(pick)), "chars": len(s)}
        return [lm.enc(s)], [1.0], [PVAL, lm.enc(s)], info
    if arm[0] == "B":
        junk = float(arm.split()[-1].rstrip("%")) / 100; clean = lm.enc("".join(PTR)); jk = lm.enc("".join(shuffle_words(d, r) for d in PTR))
        return [clean, jk], [1 - junk, junk], [PVAL], info
    if arm[0] == "C":
        code = float(arm.split()[-1].rstrip("%")) / 100
        return [lm.enc("".join(PTR)), lm.enc("".join(CTR))], [1 - code, code], [PVAL, CVAL], info
ARMS = ["A unique", "A duplicated", "A deduplicated", "B junk 0%", "B junk 25%", "B junk 50%", "C code 0%", "C code 10%", "C code 30%", "C code 50%"]
def job(a):
    arm, seed = a; src, pr, ev, info = build(arm, seed)
    return arm, seed, lm.train(src, pr, seed, steps=800, B=16, evals=ev), info
if __name__ == "__main__":
    t = time.time()
    with Pool(16) as p: out = p.map(job, [(a, s) for s in range(4) for a in ARMS], chunksize=1)
    json.dump(out, open("f1_res.json", "w")); print("elapsed", time.time() - t)
```

Statistics output:

```
runs: 40
A deduplicated   n=4 prose val bpc mean 2.852 sd 0.021 | docs 59, chars 474724, own-train bpc 2.798
A duplicated     n=4 prose val bpc mean 2.881 sd 0.042 | docs 59, chars 1001590, own-train bpc 2.708
A unique         n=4 prose val bpc mean 2.841 sd 0.036 | docs 123, chars 1012090, own-train bpc 2.835
B junk 0%        n=4 prose val bpc mean 2.851 sd 0.036
B junk 25%       n=4 prose val bpc mean 2.879 sd 0.029
B junk 50%       n=4 prose val bpc mean 2.908 sd 0.032
C code 0%        n=4 prose val bpc mean 2.851 sd 0.036 | code val bpc 5.332
C code 10%       n=4 prose val bpc mean 2.978 sd 0.034 | code val bpc 3.139
C code 30%       n=4 prose val bpc mean 3.121 sd 0.048 | code val bpc 2.962
C code 50%       n=4 prose val bpc mean 3.217 sd 0.030 | code val bpc 2.857
  A duplicated minus A unique: +0.040 bpc, Welch p=0.1951
  A deduplicated minus A duplicated: -0.030 bpc, Welch p=0.2666
  A deduplicated minus A unique: +0.011 bpc, Welch p=0.6360
  B junk 25% minus B junk 0%: +0.028 bpc, Welch p=0.2696
  B junk 50% minus B junk 0%: +0.057 bpc, Welch p=0.0565
  C code 10% minus C code 0%: +0.126 bpc, Welch p=0.0022
  C code 30% minus C code 0%: +0.269 bpc, Welch p=0.0002
  C code 50% minus C code 0%: +0.366 bpc, Welch p=0.0000

A unique         per-run gap (val bpc minus own-training bpc): mean +0.006, range -0.013 to +0.033
A duplicated     per-run gap (val bpc minus own-training bpc): mean +0.173, range +0.079 to +0.283
A deduplicated   per-run gap (val bpc minus own-training bpc): mean +0.054, range +0.028 to +0.085
```

Next: Part 2 covers the optimiser and hyperparameters: how sensitive training is to learning rate, warmup and batch size, and how much of that is seed noise.

## References

- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499), 2021.
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557), 2024.
- [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429), 2023.
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264), 2023.
