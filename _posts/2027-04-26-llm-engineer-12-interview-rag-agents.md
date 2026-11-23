---
title: "The LLM Engineer Path, Part 12: Interview Questions on RAG, Prompting and Agents"
date: 2027-04-26 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, interview, RAG, retrieval, self-consistency, chain of thought, agents, python]
description: "Part 12 of a 14-part series for developers moving into LLM engineering. Five classic questions on RAG, prompting and agents, each answered in a sentence and checked by a runnable experiment: retrieval metrics, reciprocal rank fusion, making a RAG system abstain, when majority voting fails, and why long agent chains fail."
---

Five more interview questions, on the application layer: retrieval, prompting and agents. They are classic topics, not questions from a specific company. Each has a one-sentence answer, then the evidence. The numbers come from the retrieval test of Parts 4 and 5 (BM25 over 707 standard-library functions, 23 hand-written queries) and from small calculations, so everything here can be re-run without a model or an API key.

## Q1. How do you evaluate a retriever?

**Answer.** Against labelled queries: for each query, where does the known right document rank? **recall@k** is the share of queries whose right document is in the top k; **MRR** (mean reciprocal rank) averages 1 / rank of the right document.

```python
pos = [rank(q).index(target) + 1 for q, target in Q]          # rank of the right answer, per query
recall = lambda k: np.mean([p <= k for p in pos])
mrr = np.mean([1 / p for p in pos])
```

```
recall@1   0.26        recall@3   0.48        recall@5   0.52        recall@10  0.57
MRR        0.38
ranks of the right answer: 1 1 1 1 1 1 2 2 3 3 3 4 8 12 15 31 52 55 87 120 166 180 193
```

The long tail is the finding: six queries are answered at rank 1, but a third have the right function below rank 30, where no top-k setting a language model could read will reach it. Choose k from the recall curve (panel 1 of the figure), not from habit. For an interview, add that recall@k says whether the answer was available, and the generator's accuracy given the context is a separate measurement (Part 8 shows how to separate them with a trace). With 23 queries these numbers carry wide intervals.

## Q2. What is hybrid search, and what is reciprocal rank fusion?

**Answer.** Hybrid search runs a lexical ranker (BM25) and a semantic one (embeddings) and merges the two rankings; reciprocal rank fusion (RRF, [Cormack, Clarke and Buettcher, SIGIR 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)) merges them by rank alone, scoring each document as the sum over rankers of 1 / (k + rank), with k = 60, a value the paper fixed in a pilot study and found "near-optimal, but ... not critical". Using ranks avoids having to compare scores that live on different scales.

```python
def rrf(*score_lists, k=60):
    out = np.zeros(len(score_lists[0]))
    for sc in score_lists:
        for r, i in enumerate(np.argsort(-sc)): out[i] += 1 / (k + r + 1)
    return out
```

Worked example: ranker A orders (d1, d2, d3), ranker B orders (d3, d1, d2):

```
d1: 0.03252
d3: 0.03227
d2: 0.03200
```

d1 wins because it is ranked high by both, although neither ranker has it as the only first choice. In Part 4's test the hybrid scored 9 of 23 against 12 for BM25 alone, and Part 8 showed that gap is not statistically meaningful (sign test p = 0.25). The honest answer to "does hybrid always help?" is "it often does; measure it on your queries".

## Q3. How do you make a RAG system say "I don't know"?

**Answer.** Refuse when the retrieval evidence is weak, but treat the score as a signal you must calibrate, because raw scores separate answerable from unanswerable questions only partly.

An experiment: add 12 queries with no answer in the corpus (such as "train a neural network on images" and "convert a pdf into a word document") to the 23 answerable ones, and refuse whenever the top BM25 score is below a threshold:

```
top score         answerable mean  9.6   unanswerable mean  6.9    AUROC 0.82
margin top1-top2  answerable mean  1.2   unanswerable mean  1.4    AUROC 0.50

threshold   answers shown   right doc in top 5   answers shown with no right doc
   0             35              12 of 23              23
   8             21              10 of 23              11
  10              9               5 of 23               4
  12              3               1 of 23               2
```

AUROC is the probability that a random answerable query scores higher than a random unanswerable one: 0.82 for the top score, so there is signal, and 0.50 for the gap between first and second place, which is no signal at all. The thresholds show the trade: at 10, no unanswerable query gets an answer and wrong answers fall from 23 to 4, but 14 of the 23 answerable queries are refused too. The histograms overlap (panel 2). Practical consequences: calibrate the threshold on your own labelled set, give the generator an explicit "answer only from the context, otherwise say you cannot" instruction (Part 3), check that the answer is supported by a retrieved passage, and report both rates, answered-and-wrong and refused-but-answerable.

## Q4. Do chain-of-thought and self-consistency always help?

**Answer.** Chain of thought asks the model to write intermediate steps before the answer; self-consistency samples several such answers and takes the majority. Voting helps when the errors are independent across samples, and does not help when the model is consistently wrong on the same questions.

[Wei et al.](https://arxiv.org/abs/2201.11903) show that "generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning", with reasoning abilities that "emerge naturally in sufficiently large language models". [Wang et al.](https://arxiv.org/abs/2203.11171) sample "a diverse set of reasoning paths" and select "the most consistent answer", reporting gains such as +17.9% on GSM8K. The mechanism is a majority vote, and its limit is arithmetic. Take a model that is right 65% of the time per sample:

```
independent errors, p = 0.65 per sample:                       n=1 0.65   n=5 0.76   n=11 0.85   n=21 0.92
errors tied to the question (60% of questions at p = 0.95,
                             40% at p = 0.20; mean also 0.65): n=1 0.65   n=5 0.62   n=11 0.60   n=21 0.60
```

(Exact binomial sums.) In the second case voting makes accuracy *worse*: on the questions where the model is usually wrong, the majority is wrong with more certainty. Voting amplifies whatever the model's most likely answer is, right or wrong. It costs n times the tokens (Part 9), so it should be tested on your task, not assumed.

## Q5. Why do agents fail on long tasks?

**Answer.** Per-step reliability compounds: a chain of n steps, each right with probability p, succeeds with probability p^n.

```
p=0.99:   5 steps 0.95   10 steps 0.90   20 steps 0.82   50 steps 0.61
p=0.95:   5 steps 0.77   10 steps 0.60   20 steps 0.36   50 steps 0.08
p=0.90:   5 steps 0.59   10 steps 0.35   20 steps 0.12   50 steps 0.01
```

A 95%-reliable step sounds good; twenty of them in a row succeed barely a third of the time (this assumes independent steps, a simplification). The engineering answer is in Part 7: shorten chains (use fixed code paths where the steps are known), validate each step's output so errors are caught where they happen, retry with the error message as feedback, and cap steps and cost. A workflow with a model call in two places beats an agent with twenty when the steps are predictable.

![Four panels. Top left: recall at k for BM25, 0.26 at k of 1, 0.52 at 5, 0.57 at 10. Top right: histograms of top retrieval scores for answerable and unanswerable queries, overlapping. Bottom left: majority-vote accuracy rising from 0.65 to 0.92 with independent errors and falling to 0.60 when errors are tied to the question. Bottom right: probability that a chain of up to 50 steps is entirely correct for per-step accuracy of 0.99, 0.95 and 0.90]({{ site.baseurl }}/assets/pic/llmeng-interview2.png){: width="900" }

## Try it

1. Add five queries of your own to the unanswerable set. Does the AUROC of the top score change much? What does that say about how much to trust 12 queries?
2. In Q4, change the hard-question share from 40% to 20%, keep the mean at 0.65 by adjusting p, and see where voting stops hurting.
3. For an agent you have built, count its steps on a typical task and estimate p from its logs. What does p^n predict, and how does it compare with the observed success rate?

## Limits

- The retrieval corpus is small, the queries were written by hand, and the 12 unanswerable queries by the same author; AUROC and recall values are illustrative and carry wide intervals (Part 8).
- BM25 scores are not comparable across queries of different lengths; that is part of why the abstain signal is weak, and a trained reranker would behave differently.
- Q4 and Q5 are calculations from stated assumptions (independent samples, independent steps); real models fall between the two cases, and the papers' measured gains are the evidence for when voting works.
