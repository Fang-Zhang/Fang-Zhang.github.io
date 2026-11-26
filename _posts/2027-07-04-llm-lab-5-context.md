---
title: "Improving LLM Performance, Measured (5/6): A Model Not Trained to Use Retrieval Gained 0.0005 Bits From Having the Exact Answer in Context"
date: 2027-07-04 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, retrieval, rag, context length, evaluation, series]
description: "Part 5 of 6. Retrieved passages, context length and passage position tested on small language models, with an oracle control that puts the answer in the context. What the public papers claim, and what is within noise."
---

## What the literature says

Retrieval and long context both aim to give a model information it did not store in its weights. Four public sources describe what they achieve.

- **Retrieval-augmented generation.** [Lewis et al.](https://arxiv.org/abs/2005.11401) combine a pre-trained seq2seq model with "a dense vector index of Wikipedia, accessed with a pre-trained neural retriever", fine-tune the combination, and report state-of-the-art results on three open-domain question answering tasks.
- **Retrieval during pretraining.** [RETRO](https://arxiv.org/abs/2112.04426) conditions an autoregressive model "on document chunks retrieved from a large corpus, based on local similarity with preceding tokens", and with a 2 trillion token database "obtains comparable performance to GPT-3 and Jurassic-1 on the Pile, despite using 25x fewer parameters". RETRO is trained from scratch with the retrieval mechanism, and the paper also reports retrofitting pre-trained transformers.
- **The retriever.** [Karpukhin et al.](https://arxiv.org/abs/2004.04906) show dense retrieval outperforming a strong Lucene-BM25 system "largely by 9%-19% absolute in terms of top-20 passage retrieval accuracy".
- **Position in the context.** [Liu et al.](https://arxiv.org/abs/2307.03172) find that language models do not robustly use long contexts: "performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts".

All four are abstract-level claims about trained or fine-tuned models at scale. The experiment below asks what a small model that was never trained with retrieval does with a retrieved passage, and uses an oracle control to separate "the retrieval was poor" from "the model cannot use the passage".

## Experiment: passages, context length and position

Setup: four character-level GPTs (4 layers, width 128, 842,624 parameters, 1200 training steps, held-out loss 2.329 bits per character), trained on plain prose with no retrieval format. The test uses 600 held-out windows per model, each a 24-character query followed by a 32-character target; the loss is measured on the 32 target characters only, and every condition uses the same windows.

- **Context length.** The true text before the target, from 8 to 96 characters (the last 24 of which are the query when the length is 24 or more).
- **Retrieval.** The query is matched against 24-character chunks of the training corpus using TF-IDF over character 4-grams; the 24 characters that follow the best match are placed before the query. Mean cosine similarity of the best match was 0.515; 7.3 percent of queries matched at 0.8 or above and 2.5 percent at 0.95 or above, which reflects posts that share wording. A random 24-character passage of the training corpus is the control.
- **Oracle.** The first 24 characters of the target itself are placed in the context, so about three quarters of the target characters can be copied. This is an upper bound for any retrieval method.
- **Position.** Three passages of 24 characters, one of which is the oracle passage and two random, in the order passage-first, passage-middle or passage-last (next to the query), compared with three random passages.

Results (loss on the target characters; 2400 windows over the four models; differences are paired, with 95 percent intervals over windows):

- **Context length.** Loss was 2.237 with 8 characters, 2.228 with 16, 2.225 with 24, 2.225 with 48, 2.229 with 72 and 2.236 with 96. The total range is 0.012 bits, and longer contexts beyond 48 characters were worse, not better (96 minus 24: +0.011, interval +0.007 to +0.015).
- **Retrieved passage.** Query plus retrieved passage minus query alone: +0.006 (interval +0.002 to +0.009). A random passage: +0.007 (interval +0.004 to +0.011). Retrieved minus random: -0.0014 (interval -0.0025 to -0.0003). On the 7.3 percent of windows with similarity of 0.8 or above, retrieved minus random was 0.0000 (interval -0.005 to +0.005).
- **Oracle.** Query plus oracle passage minus query alone: -0.0005 (interval -0.004 to +0.003). The exact answer in the context changed the loss by less than the random-passage cost.
- **Position of the oracle passage** (against three random passages): first -0.0023, middle -0.0040, next to the query -0.0062 (intervals about plus or minus 0.001 each).

![Three panels. Left: loss on 32 target characters against context length from 8 to 96 characters for four models, a shallow U shape with a total range of 0.012 bits. Middle: loss change from adding a 24-character passage before the query: retrieved +0.006, random +0.007, oracle with the answer inside -0.0005, all within 0.01 bits. Right: with the answer among three passages, loss change is -0.002 when first, -0.004 in the middle and -0.006 next to the query]({{ site.baseurl }}/assets/pic/llm-lab-5-context.png){: width="900" }
_Figure: left, one line per model; middle and right, mean change with a 95 percent interval over windows. Measured here._

## Engineering reading

These points are this series' reading of the sources and the measurement.

1. **A model must be able to use the context before retrieval can help.** With the exact answer in the context, this model's loss did not move (-0.0005). The retrieved passages did nothing, which is consistent with a model that has not learned to copy from earlier text; this experiment did not examine the model's internals. The cited retrieval results are for models fine-tuned for retrieval (Lewis et al.) or trained with it (RETRO).
2. **Retrieval quality was not the bottleneck here.** The oracle case is perfect retrieval, and it showed the same null result as the retrieved case. Measuring retrieval accuracy alone, as in the dense-retrieval work, would not have revealed this.
3. **More context was not better.** Beyond 48 characters the loss rose, by 0.011 at 96 characters. The model was trained on windows up to 128 characters, so this is not a length-limit effect; the longer text is mostly noise for a model that relies on the last few characters.
4. **The position effect is small and ordered by distance, not U-shaped.** Putting the answer next to the query gave 0.006 bits of improvement against 0.002 at the start and 0.004 in the middle. The shape is the opposite of a "lost in the middle" dip, and all three are tiny, 0.1 to 0.3 percent of the loss. They are better read as the model weighting nearby characters than as evidence for or against the published finding, which is about much larger models and question answering.
5. **Sizes matter as much as signs.** Every effect in this experiment is between 0.0005 and 0.012 bits, while the models differ from each other by up to 0.03 on the same windows. A small, significant window-level interval is not an improvement worth shipping; the comparison that decides is size against model-to-model spread.

## Limits

- Four small character-level models; no model here was trained or fine-tuned with retrieved passages, so the result says nothing about models that were.
- The retriever is a sparse TF-IDF over character 4-grams, and the passage is the continuation of the best-matching chunk, not a human-written passage that answers a question. The task is next-character prediction, not question answering.
- The oracle copies the target's own characters; it measures whether the model uses context at all, not how it would handle a paraphrase.
- Context lengths up to 96 characters and three passages of 24 characters are far below the context lengths studied in the literature.
- Window-level intervals treat the 2400 windows as independent although they come from four models; the per-model values in the statistics output show that some of the smaller differences change sign between models.
- Literature claims come from abstracts, and no result was reproduced.

## Code and output

Experiment code (model and training loop from Part 1):

```python
import sys, json, math, random, numpy as np, torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
import torch.nn.functional as F, lm
from sklearn.feature_extraction.text import TfidfVectorizer
P = lm.load_prose(); PTR, PVA = lm.split(P); TRS = "".join(PTR); VAS = "".join(PVA); TR = lm.enc(TRS)
import os
U = 24; STEPS = int(os.environ.get("F5_STEPS", 1200))
def train_model(seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = lm.GPT(); lr, warmup, B = 3e-3, 100, 16
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01); probs = torch.ones(1)
    for t in range(STEPS):
        for gr in opt.param_groups: gr["lr"] = lr * min(1, (t + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / STEPS)))
        x, y = lm.batch([TR], probs, B, g); loss = F.cross_entropy(m(x).reshape(-1, lm.V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m
@torch.no_grad()
def target_loss(m, ctxs, tgts):
    """mean bits/char over the target characters, given a context string per window (all contexts same length)"""
    ids = torch.stack([lm.enc(c + t) for c, t in zip(ctxs, tgts)]); lc = len(ctxs[0]); out = []
    for i in range(0, len(ids), 100):
        b = ids[i:i + 100]; lg = m(b[:, :-1])[:, lc - 1:]; ce = F.cross_entropy(lg.reshape(-1, lm.V), b[:, lc:].reshape(-1), reduction="none").view(len(b), -1).mean(1) / math.log(2); out += ce.tolist()
    return np.array(out)
def build_windows(n=600):
    rng = random.Random(5); W = []
    while len(W) < n:
        s = rng.randrange(0, len(VAS) - 160); w = VAS[s:s + 152]
        if "\n" not in w: W.append(w)
    return W
if __name__ == "__main__":
    seed = int(sys.argv[1]); W = build_windows(); m = train_model(seed); rng = random.Random(100 + seed)
    starts = list(range(0, len(TRS) - 2 * U, U // 2)); chunks = [TRS[i:i + U] for i in starts]; conts = [TRS[i + U:i + 2 * U] for i in starts]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(4, 4), sublinear_tf=True); X = vec.fit_transform(chunks)
    Q = [w[96:96 + U] for w in W]; T = [w[120:152] for w in W]; Sim = (vec.transform(Q) @ X.T).tocsr()
    best = [int(Sim[i].toarray().argmax()) for i in range(len(W))]; sim = [float(Sim[i, best[i]]) for i in range(len(W))]; PAS = [conts[b] for b in best]
    rnd = lambda: (lambda j: conts[j])(rng.randrange(len(conts)))
    R1 = [rnd() for _ in W]; R2 = [rnd() for _ in W]; R3 = [rnd() for _ in W]
    res = {"seed": seed, "n": len(W), "sim": sim, "val_bpc": lm.evaluate(m, lm.enc(VAS), 400), "loss": {}}
    for k in (8, 16, 24, 48, 72, 96): res["loss"][f"ctx{k}"] = target_loss(m, [w[120 - k:120] for w in W], T).tolist()
    res["loss"]["query+retrieved"] = target_loss(m, [p + q for p, q in zip(PAS, Q)], T).tolist()
    res["loss"]["query+random"] = target_loss(m, [r + q for r, q in zip(R1, Q)], T).tolist()
    res["loss"]["order R D D Q"] = target_loss(m, [p + a + b + q for p, a, b, q in zip(PAS, R1, R2, Q)], T).tolist()
    res["loss"]["order D R D Q"] = target_loss(m, [a + p + b + q for p, a, b, q in zip(PAS, R1, R2, Q)], T).tolist()
    res["loss"]["order D D R Q"] = target_loss(m, [a + b + p + q for p, a, b, q in zip(PAS, R1, R2, Q)], T).tolist()
    res["loss"]["order D D D Q"] = target_loss(m, [a + b + c + q for a, b, c, q in zip(R1, R2, R3, Q)], T).tolist()
    OR = [t[:U] for t in T]   # oracle passage: the first 24 target characters themselves (upper bound: the answer is in the context)
    res["loss"]["oracle+Q"] = target_loss(m, [o + q for o, q in zip(OR, Q)], T).tolist()
    res["loss"]["oracle R D D Q"] = target_loss(m, [o + a + b + q for o, a, b, q in zip(OR, R1, R2, Q)], T).tolist()
    res["loss"]["oracle D R D Q"] = target_loss(m, [a + o + b + q for o, a, b, q in zip(OR, R1, R2, Q)], T).tolist()
    res["loss"]["oracle D D R Q"] = target_loss(m, [a + b + o + q for o, a, b, q in zip(OR, R1, R2, Q)], T).tolist()
    json.dump(res, open(f"f5_seed{seed}.json", "w")); print("done", seed, {k: round(float(np.mean(v)), 3) for k, v in res["loss"].items()}, "mean sim", round(float(np.mean(sim)), 3), flush=True)
```

Statistics output:

```
models: 4, windows per model: 600, held-out bpc of base models: 2.329
retrieval similarity (cosine of char 4-gram TF-IDF): mean 0.515, share >= 0.8: 0.073, share >= 0.95: 0.025

loss on the 32 target characters (bits/char), mean over 4 models x 600 windows; per-model means in brackets
ctx8               2.2366  [2.237, 2.235, 2.255, 2.219]
ctx16              2.2280  [2.231, 2.225, 2.246, 2.210]
ctx24              2.2250  [2.225, 2.223, 2.242, 2.210]
ctx48              2.2253  [2.227, 2.220, 2.246, 2.209]
ctx72              2.2287  [2.228, 2.222, 2.254, 2.211]
ctx96              2.2357  [2.233, 2.230, 2.263, 2.217]
query+retrieved    2.2308  [2.232, 2.225, 2.251, 2.215]
query+random       2.2322  [2.234, 2.224, 2.253, 2.217]
order R D D Q      2.2457  [2.243, 2.237, 2.271, 2.231]
order D R D Q      2.2453  [2.242, 2.237, 2.271, 2.231]
order D D R Q      2.2451  [2.243, 2.237, 2.270, 2.230]
order D D D Q      2.2462  [2.243, 2.239, 2.273, 2.230]
oracle+Q           2.2246  [2.227, 2.218, 2.245, 2.208]
oracle R D D Q     2.2440  [2.241, 2.236, 2.270, 2.228]
oracle D R D Q     2.2422  [2.240, 2.234, 2.268, 2.227]
oracle D D R Q     2.2400  [2.239, 2.232, 2.264, 2.224]

paired differences (negative = lower loss = better)
ctx24 minus ctx8: -0.0116 (window-level 95% interval -0.0152 to -0.0079; per-model -0.0122, -0.0127, -0.0127, -0.0086)
ctx96 minus ctx24: +0.0107 (window-level 95% interval +0.0067 to +0.0147; per-model +0.0082, +0.0069, +0.0204, +0.0072)
query+retrieved minus ctx24: +0.0057 (window-level 95% interval +0.0022 to +0.0092; per-model +0.0064, +0.0020, +0.0089, +0.0055)
query+random minus ctx24: +0.0071 (window-level 95% interval +0.0036 to +0.0106; per-model +0.0087, +0.0016, +0.0112, +0.0069)
query+retrieved minus query+random: -0.0014 (window-level 95% interval -0.0025 to -0.0003; per-model -0.0023, +0.0004, -0.0023, -0.0014)
order D R D Q minus order R D D Q: -0.0003 (window-level 95% interval -0.0008 to +0.0001; per-model -0.0007, -0.0002, -0.0006, +0.0002)
order D D R Q minus order R D D Q: -0.0006 (window-level 95% interval -0.0016 to +0.0005; per-model -0.0002, +0.0003, -0.0011, -0.0013)
order D D R Q minus order D D D Q: -0.0012 (window-level 95% interval -0.0023 to +0.0000; per-model -0.0006, -0.0011, -0.0025, -0.0004)
order R D D Q minus order D D D Q: -0.0006 (window-level 95% interval -0.0017 to +0.0005; per-model -0.0003, -0.0015, -0.0014, +0.0008)
oracle+Q minus ctx24: -0.0005 (window-level 95% interval -0.0040 to +0.0030; per-model +0.0020, -0.0049, +0.0027, -0.0017)
oracle R D D Q minus order D D D Q: -0.0023 (window-level 95% interval -0.0034 to -0.0011; per-model -0.0021, -0.0029, -0.0024, -0.0017)
oracle D R D Q minus order D D D Q: -0.0040 (window-level 95% interval -0.0052 to -0.0028; per-model -0.0036, -0.0047, -0.0045, -0.0031)
oracle D D R Q minus order D D D Q: -0.0062 (window-level 95% interval -0.0074 to -0.0050; per-model -0.0042, -0.0061, -0.0086, -0.0060)
oracle D R D Q minus oracle R D D Q: -0.0017 (window-level 95% interval -0.0022 to -0.0012; per-model -0.0015, -0.0019, -0.0021, -0.0014)
oracle D D R Q minus oracle R D D Q: -0.0039 (window-level 95% interval -0.0050 to -0.0029; per-model -0.0021, -0.0032, -0.0062, -0.0043)

high-similarity windows (similarity >= 0.8, n=176 of 2400):
query+retrieved minus ctx24: +0.0050 (window-level 95% interval -0.0080 to +0.0181; per-model +0.0042, +0.0132, +0.0035, -0.0007)
query+retrieved minus query+random: -0.0000 (window-level 95% interval -0.0047 to +0.0047; per-model -0.0001, -0.0051, -0.0016, +0.0069)
low-similarity windows (similarity < 0.5, n=1328):
query+retrieved minus query+random: -0.0011 (window-level 95% interval -0.0026 to +0.0004; per-model -0.0021, +0.0011, -0.0019, -0.0015)
```

Next: Part 6 covers evaluation itself: how sample size, leakage and noise limit what a comparison between two models can show.

## References

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020.
- [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426), 2021.
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), 2020.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), 2023.
