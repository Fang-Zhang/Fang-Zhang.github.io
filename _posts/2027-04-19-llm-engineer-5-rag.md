---
title: "The LLM Engineer Path, Part 5: Retrieval-Augmented Generation (RAG)"
date: 2027-04-19 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, RAG, chunking, retrieval, context window, prompt, python]
description: "Part 5 of a 14-part series for developers moving into LLM engineering. How a RAG system is assembled, and a measured chunk-size sweep on 32 Python standard-library documents: smaller chunks save context but lose answers, and fixed-size cutting splits some answers across a boundary."
---

A model only knows what was in its training data and what is in its prompt. **Retrieval-augmented generation (RAG)** fills the prompt with relevant text found at question time. This post lays out the pipeline and measures the one design choice that most affects it: how documents are cut into chunks.

## The pipeline

The paper that named the idea, [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), describes models that "combine pre-trained parametric and non-parametric memory for language generation": the parameters hold what the model learned, and an external index holds documents that can be updated and cited. Its abstract names two motivations that still hold: giving "provenance for their decisions" and "updating their world knowledge".

In an application the pipeline has six steps:

1. **Chunk**: cut each document into pieces of a manageable size.
2. **Index**: build a keyword index, an embedding index, or both (Part 4).
3. **Retrieve**: for a question, fetch the top k chunks.
4. **Assemble**: put the chunks into the prompt with the question and an instruction to answer only from them, and to say so if the answer is absent.
5. **Generate**: the model writes the answer, ideally citing which chunk it used.
6. **Check**: confirm the answer is supported by the chunks (Part 8).

Steps 3 to 5 get most of the attention. Step 1 decides what steps 3 to 5 can possibly find.

## Why chunk size is a trade-off

A model reads a limited context (Part 2), every token costs time and money, and relevant text buried among irrelevant text is harder to use. The paper [Lost in the Middle](https://arxiv.org/abs/2307.03172) found that model "performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts". So sending everything is not free either.

Small chunks keep the prompt short and focused. Big chunks are less likely to cut a fact in half. The right size depends on the documents and the questions, which makes it a measurement problem.

## A measurement you can run

The corpus is 32 standard-library modules (`os`, `time`, `gzip`, `statistics` and so on), with each module's complete docstrings joined into one document: about 200,000 characters. There are 18 questions, each with an exact answer string copied from the documentation ("Which compression level does gzip use by default?" has the answer `compresslevel is 9`). The code asserts that each answer string really occurs in its module's text, so the questions have known answers. Retrieval is BM25 from Part 4, top 3 chunks. There is no language model in this experiment; the metric is whether the retrieved context **contains the answer**, which is a necessary condition for any model to answer correctly.

```python
def chunks(size, overlap):
    out = []
    for m, t in docs.items():
        w = t.split(); step = max(1, size - overlap)
        for i in range(0, max(1, len(w) - overlap), step): out.append((m, " ".join(w[i:i + size])))
    return out
def run(size, overlap, k=3):
    C = chunks(size, overlap); bm = BM25([toks(c) for _, c in C]); hit = ctx = 0
    for q, m, a in QA:
        top = np.argsort(-bm.scores(q))[:k]; got = "\n".join(C[i][1] for i in top); hit += a in got; ctx += len(got.split())
    # can ANY chunk of the right module hold the full answer?
    fits = sum(any(a in c for mm, c in C if mm == m) for q, m, a in QA)
    return dict(size=size, overlap=overlap, chunks=len(C), hit=hit, n=len(QA), words=ctx / len(QA), fits=fits)
```

Real output (`hit` counts questions whose answer string is in the 3 retrieved chunks; `fits` counts questions whose answer lies whole inside at least one chunk of the right module; `words` is the average size of the assembled context):

```
size  overlap  chunks  hit/18  fits/18  words in context
  25       0    1229     6       15        75
  25       6    1602    11       17        75
  50       0     622    10       16       150
  50      12     804    13       18       145
 100       0     320    15       18       290
 100      25     408    15       18       291
 200       0     170    17       18       542
 200      50     208    17       18       561
 400       0      94    17       18      1026
 400     100     109    17       18      1055
 800       0      56    18       18      1767
 800     200      60    18       18      1915
```

![Left: number of questions whose answer is in the retrieved context against words sent to the model, rising from 6 of 18 at 75 words (25-word chunks, no overlap) to 18 of 18 at about 1,800 words (800-word chunks), with the overlap line above the no-overlap line for small chunks. Right: answers that fit whole inside some chunk, 15 and 16 of 18 for 25- and 50-word chunks without overlap, 18 of 18 for 100-word chunks and larger]({{ site.baseurl }}/assets/pic/llmeng-rag-chunks.png){: width="900" }
*Left: answer found against context size. Right: how many answers a chunking scheme leaves intact.*

What it shows:

- **Context cost and answer coverage rise together.** 25-word chunks put 75 words in the prompt and contain the answer 6 times in 18. 200-word chunks put about 540 words in and succeed 17 times. 800-word chunks reach 18 at 1,767 words, 24 times the context of the smallest setting.
- **The knee is near 100 to 200 words on this data.** Going from 100 to 200 words gains 2 questions for roughly double the context; going from 200 to 800 gains 1 for more than three times the context.
- **Fixed-size cutting destroys some answers.** With 25-word chunks and no overlap, 3 of the 18 answers are split across a boundary, so no chunk holds them whole and no retriever can ever return them. Overlap of 25% brings that down to 1; at 50 words it removes the problem.
- **Overlap helps most when chunks are small.** At 100 words and above it changed nothing here, and added 7% to 27% more chunks to index.

## The failure to reproduce

Run the 25-word, no-overlap row and print the answers that no chunk contains. They are cut mid-sentence by the word counter, not by anything in the retriever, so improving the retriever (better embeddings, re-ranking) cannot fix them. When a RAG system "never finds" a fact that is in the documents, check the chunks before the retriever: search the chunk list for the answer string, as `fits` does. If it is not whole in any chunk, the problem is step 1.

## Try it

1. Split on paragraph breaks (`"\n\n"`) instead of word counts, and re-run. Do answers still get cut?
2. Change `k` from 3 to 1 and to 8. Plot `hit` against `words` again; where does adding chunks stop paying?
3. Add a language model as step 5 if you have one, with the instruction "answer only from the context, or say you do not know". Count how many answers are right when the context lacks the answer. This post could not run that step, and a model that answers from memory when the context is empty is the usual failure.

## Limits

- 18 questions about one corpus. A difference of one question is noise; treat the shape of the curve, not exact counts.
- The metric is "answer string is in the context", not "the model answered correctly". A model can still miss an answer that is present, especially in the middle of a long context.
- Questions were written by hand with the documentation in view, so they share vocabulary with it. Real user questions are harder.
- Word counts stand in for tokens. Real limits and costs are in tokens, which are usually more than words (Part 2).
