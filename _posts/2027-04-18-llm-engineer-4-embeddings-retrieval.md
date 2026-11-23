---
title: "The LLM Engineer Path, Part 4: Embeddings and Retrieval"
date: 2027-04-18 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, embeddings, retrieval, BM25, vector search, hybrid search, python]
description: "Part 4 of a 14-part series for developers moving into LLM engineering. What an embedding is, how similarity search works, and a measured comparison on 707 Python standard-library functions: keyword search (BM25), a dense-vector method, and a hybrid of both. The dense method is weaker, and the hybrid is worse than BM25 alone."
---

Retrieval finds the few pieces of text, out of thousands, that are relevant to a question. Part 5 hands those pieces to a model; this post is about finding them. It covers what an embedding is, the two families of search, and a measurement that argues for testing before trusting either.

## What an embedding is

An **embedding** is a list of numbers (a vector) that represents a piece of text, built so that texts with similar meaning get vectors that point in similar directions. Similarity is usually the **cosine** of the angle between two vectors:

```
cos(a, b) = (a . b) / (|a| x |b|)      1 = same direction, 0 = unrelated
```

Search is then simple: embed every document once, embed the query, and return the documents whose vectors have the highest cosine with the query vector. With N documents of dimension d, a brute-force search costs N x d multiplications per query. Approximate indexes exist to cut that when N reaches millions; at thousands of documents, brute force is fine.

Why use embeddings at all instead of matching words? Two papers give the motivation:

- [Sentence-BERT](https://arxiv.org/abs/1908.10084) points out that comparing sentence pairs by feeding both into BERT needs "about 50 million inference computations (~65 hours)" to find the most similar pair among 10,000 sentences. Computing one embedding per sentence and comparing by cosine reduces that "from 65 hours with BERT / RoBERTa to about 5 seconds with SBERT".
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) shows retrieval "can be practically implemented using dense representations alone", and reports its dense retriever outperforming a strong Lucene-BM25 system "by 9%-19% absolute in terms of top-20 passage retrieval accuracy" on open-domain question answering.

## The two families

- **Sparse / lexical** search scores documents by the words they share with the query. The standard formula is **BM25**: each query word counts more if it is rare across the collection (inverse document frequency), saturates as it repeats within a document, and is normalised by document length. It needs no model and no training.
- **Dense** search compares embeddings from a neural network trained on many question-passage pairs, so "delete a folder" can match "remove a directory" without sharing a word.

[BEIR](https://arxiv.org/abs/2104.08663), a benchmark of 18 retrieval datasets, found that "BM25 is a robust baseline", that re-ranking and late-interaction models did best zero-shot "at high computational costs", and that dense and sparse retrieval models "often underperform" on data unlike what they trained on. Neither family wins everywhere, which is why measuring on your own data matters.

## A measurement you can run

The corpus is 707 functions and classes from 32 Python standard-library modules (`shutil`, `os`, `heapq`, `json` and so on); each document is the function name plus the first paragraph of its docstring. The task is to find the right function from a plain-English description. There are 23 hand-written queries, some sharing words with the docstring and some not ("delete a folder and everything inside it" should find `shutil.rmtree`).

No embedding model can be downloaded in this environment, so the dense method here is **LSA** (latent semantic analysis): build a TF-IDF matrix, compress it with a singular value decomposition to 100 dimensions, and compare by cosine. LSA is a dense-vector method that learns word associations from the corpus itself. It is much weaker than a neural embedding model, and that matters for reading the results.

```python
MODS = "os os.path shutil re json itertools functools heapq bisect math statistics random string textwrap datetime time glob fnmatch copy csv zipfile tarfile gzip hashlib base64 uuid pathlib subprocess tempfile urllib.parse collections operator".split()
def corpus():
    docs = {}
    for mn in MODS:
        mod = importlib.import_module(mn)
        for n, o in vars(mod).items():
            if n.startswith("_") or not (inspect.isfunction(o) or inspect.isbuiltin(o) or inspect.isclass(o)): continue
            d = inspect.getdoc(o)
            if d and len(d) > 20: docs[f"{mn}.{n}"] = d.strip().split("\n\n")[0].replace("\n", " ")[:300]
    return docs
```

```python
class BM25:
    def __init__(s, T, k1=1.5, b=0.75):
        s.T, s.k1, s.b = T, k1, b; s.N = len(T); s.avg = sum(map(len, T)) / s.N
        df = Counter(w for t in T for w in set(t)); s.idf = {w: math.log(1 + (s.N - n + .5) / (n + .5)) for w, n in df.items()}; s.tf = [Counter(t) for t in T]
    def scores(s, q):
        out = np.zeros(s.N)
        for w in set(toks(q)):
            if w not in s.idf: continue
            for i, tf in enumerate(s.tf):
                f = tf.get(w, 0)
                if f: out[i] += s.idf[w] * f * (s.k1 + 1) / (f + s.k1 * (1 - s.b + s.b * len(s.T[i]) / s.avg))
        return out
class LSA:                                                                 # dense vectors: TF-IDF -> truncated SVD
    def __init__(s, T, k=100):
        s.vocab = {w: j for j, w in enumerate(sorted({w for t in T for w in t}))}; N = len(T)
        df = Counter(w for t in T for w in set(t)); s.idf = np.array([math.log(N / df[w]) + 1 for w in s.vocab])
        X = np.stack([s.vec(t) for t in T]); U, S, Vt = np.linalg.svd(X, full_matrices=False); s.P = Vt[:k].T; s.D = X @ s.P
        s.D /= np.linalg.norm(s.D, axis=1, keepdims=True) + 1e-9
    def vec(s, t):
        v = np.zeros(len(s.vocab))
        for w, c in Counter(t).items():
            if w in s.vocab: v[s.vocab[w]] = (1 + math.log(c)) * s.idf[s.vocab[w]]
        return v
    def scores(s, q):
        z = s.vec(toks(q)) @ s.P; return s.D @ (z / (np.linalg.norm(z) + 1e-9))
def rrf(*score_lists, k=60):                                               # reciprocal rank fusion
    out = np.zeros(len(score_lists[0]))
    for sc in score_lists:
        for r, i in enumerate(np.argsort(-sc)): out[i] += 1 / (k + r + 1)
    return out
```

Real output:

```
BM25    top1 6/23  top5 12/23  MRR 0.38
LSA     top1 0/23  top5 7/23  MRR 0.14
hybrid  top1 2/23  top5 9/23  MRR 0.23
```

(MRR, mean reciprocal rank, averages 1/rank of the correct answer; 1.0 means always first.)

![Left: BM25 gets the target first in 6 of 23 queries and in the top 5 in 12; LSA gets 0 and 7; the hybrid gets 2 and 9. Right: a log-log scatter of the rank each method gave the right function for every query, with points on both sides of the diagonal]({{ site.baseurl }}/assets/pic/llmeng-retrieval.png){: width="900" }
*Right panel: a dot above the diagonal means BM25 ranked the right function better than LSA; below it, LSA did better.*

What it shows:

- **BM25 is the strongest here.** It is the baseline to beat, as BEIR said.
- **The methods fail on different queries.** "start another program and wait for it" (`subprocess.run`): BM25 rank 120, LSA rank 10. "convert a python object into a json string" (`json.dumps`): BM25 15, LSA 4. In the other direction, "delete a folder and everything inside it": BM25 rank 1, LSA rank 91; "make a shallow duplicate of an object": BM25 4, LSA 221.
- **Some queries defeat both.** "create nested directories" (`os.makedirs`) ranks 193 and 326; the docstring of `os.makedirs` is just its signature, so there is little text to match.
- **A hybrid is not automatically better.** Reciprocal rank fusion (adding 1/(60 + rank) from each list) scored below BM25 alone, because it mixes in a weaker ranker. Fusion pays off when both components are good and fail differently.

## The failure to reproduce

A natural conclusion from the scatter plot is "combine them and win". On this data, the combination lost to the single best method (top-5: 9 versus 12). The only way to know is to run both on queries from your own data and compare, before building the fusion into a system. Twenty to fifty real queries with known right answers are enough to catch this.

## Try it

1. Add 10 queries of your own to `Q` and compute the three rows again. Do the rankings change?
2. Change `k` in `LSA(T, 100)` to 20 and to 300. Where does it peak?
3. If you can download models, replace `LSA` with a sentence embedding model (for example from the `sentence-transformers` library) and re-run `evaluate`. This post could not run that step, so its result is unknown here; the papers above suggest the dense row will rise.
4. Embed only the function names, not the docstrings. Which queries break?

## Limits

- 23 queries on one corpus is a small test. Differences of one or two queries are noise, and so is the 3-query gap between BM25 and the hybrid: an exact sign test gives p = 0.25 (worked through in Part 8), so read the ranking as "BM25 was at least as good here", not as a finding.
- The queries were written by hand after looking at some docstrings, which can favour lexical overlap; a fair test set is written before looking at the corpus.
- LSA is not a neural embedding model. The result says nothing about how a trained embedding model would score on this data.
- Brute-force search is shown; approximate nearest-neighbour indexes trade a little accuracy for speed and are outside this post.
