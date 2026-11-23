---
title: "The LLM Engineer Path, Part 8: Evals and Tracing"
date: 2027-04-22 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, evaluation, tracing, observability, statistics, LLM-as-a-judge, python]
description: "Part 8 of a 14-part series for developers moving into LLM engineering. How to tell whether a change made an LLM system better: test sets, confidence intervals and why 23 questions invent winners, LLM judges and their known biases, and a 20-line tracer that shows which pipeline stage failed."
---

An **eval** is a fixed set of inputs with a way to score the outputs, so that a change to a prompt, model or retriever can be judged by numbers instead of impressions. A **trace** is a record of what each step of a request did, so a bad answer can be pinned on a stage. Both are ordinary engineering, with one statistical trap that this post measures.

## The eval loop

1. Collect real inputs (or realistic ones) and write the expected result for each: an exact answer, a document that must be retrieved, a rule the output must satisfy.
2. Score automatically where possible: exact match, "answer is contained in the context", valid JSON, a unit test on generated code.
3. Run the same set on every change and compare against the previous run, not against a feeling.
4. When a production failure appears, add it to the set. The set grows from failures.

Where no rule can score an output (tone, helpfulness), a model can act as grader. [Zheng et al.](https://arxiv.org/abs/2306.05685) examine "the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability", and report that strong judges such as GPT-4 reach "over 80% agreement" with human preferences, "the same level of agreement between humans". Treat a judge as a measuring instrument that needs its own calibration: swap the order of the answers being compared, check its scores against a sample you graded yourself, and do not let a model judge its own outputs.

## The trap: a small test set cannot rank two systems

Part 4 of this series compared two retrievers on 23 queries: BM25 found the right function in the top 5 for 12 of them, a hybrid for 9. That looks like a win. It is not established. The code, with the counts from that experiment:

```python
import math, random
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d

b, c = 3, 0          # queries where only one system hits
k = min(b, c); p = min(1.0, 2 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2 ** (b + c))
print(f"only BM25 hits: {b}, only hybrid hits: {c}, exact sign-test p = {p:.2f}")
```

```
n = 23   BM25 hits 12   hybrid hits 9
BM25    52%  95% interval 33% to 71%
hybrid  39%  95% interval 22% to 59%
only BM25 hits: 3, only hybrid hits: 0, exact sign-test p = 0.25
```

The two intervals overlap almost completely. The sign test looks only at queries where exactly one system succeeded (3 for BM25, 0 for the hybrid) and asks how often a fair coin gives a split at least that lopsided: 25% of the time. Three queries are not evidence. The ranking in Part 4 should be read as "BM25 was at least as good in this sample", not as a finding.

How often does noise alone produce a gap like that? Two systems with the *same* true accuracy of 50%, scored independently on n questions, looking at how often they differ by 10 points or more:

```python
rng = random.Random(0); print("n   P(two EQUAL systems differ by >= 10 points)")
for m in (23, 100, 400, 1600):
    wins = sum(abs(sum(rng.random() < .5 for _ in range(m)) - sum(rng.random() < .5 for _ in range(m))) / m >= .10 for _ in range(4000)); print(f"{m:5d}  {100*wins/4000:.0f}%")
```

```
n   P(two EQUAL systems differ by >= 10 points)
   23  46%
  100  19%
  400  1%
 1600  0%
```

![Three panels. Left: 95 percent intervals for BM25 (33 to 71) and hybrid (22 to 59) overlapping on 23 queries. Middle: the chance that two equal systems appear 10 points apart falls from 46 percent at 23 questions to 19 at 100, 1 at 400 and 0 at 1,600. Right: stacked bars of 18 questions by failure stage for chunk sizes of 25, 50 and 100 words]({{ site.baseurl }}/assets/pic/llmeng-evals.png){: width="900" }

With 23 questions, two identical systems look 10 points apart nearly half the time. This simulation scores the two systems independently, which is the worst case; running both on the *same* questions and comparing per question (as the sign test does) removes shared difficulty and needs fewer, but the lesson holds: hundreds of questions for small differences, and always an interval next to the number.

## Tracing: find the stage that failed

An aggregate score says a system is wrong 40% of the time; it does not say where. A **span** is one timed step with its inputs and outputs written down; the spans of one request share a trace id. A minimal version in plain Python, applied to the retrieval pipeline of Part 5 (18 questions about library functions, answered from chunks of the documentation):

```python
TRACE = []
@contextlib.contextmanager
def span(name, trace_id, **attrs):                      # one JSON line per step, with timing
    s = {"trace": trace_id, "span": name, **attrs}; t0 = time.perf_counter()
    try: yield s
    finally: s["ms"] = round(1000 * (time.perf_counter() - t0), 2); TRACE.append(s)

def pipeline(question, module, answer, C, bm, k=3):
    tid = uuid.uuid4().hex[:6]
    with span("retrieve", tid, question=question[:40]) as s:
        top = np.argsort(-bm.scores(question))[:k]; s["chunk_ids"] = [int(i) for i in top]; s["modules"] = [C[i][0] for i in top]
    with span("assemble", tid) as s:
        ctx = "\n".join(C[i][1] for i in top); s["context_words"] = len(ctx.split())
    with span("check", tid) as s:                       # where did this question fail, if it did?
        holders = [i for i, (m, c) in enumerate(C) if m == module and answer in c]
        s["answer_in_context"] = answer in ctx
        s["fault"] = "ok" if answer in ctx else ("indexing: no chunk holds the answer" if not holders else "retrieval: chunk exists, not in top %d" % k)
    return tid, s["fault"]
```

One request's trace, as stored (three JSON lines in practice; shortened):

```
{"span": "retrieve", "chunk_ids": [728, 730, 729], "modules": ["random", "random", "random"], "ms": 1.48}
{"span": "assemble", "context_words": 75, "ms": 0.01}
{"span": "check", "answer_in_context": true, "fault": "ok", "ms": 0.03}
```

Counting the `fault` field across all 18 questions, by chunk size (no overlap, top 3 chunks):

```
chunks of  25 words: ok 6,  retrieval 9, indexing 3
chunks of  50 words: ok 10, retrieval 6, indexing 2
chunks of 100 words: ok 15, retrieval 3, indexing 0
```

Each fault has a different fix. **Indexing** faults (no chunk contains the answer, because chunking cut it in half) are fixed by larger chunks or overlap. **Retrieval** faults (the chunk exists but ranked below the top 3) are fixed by a better retriever or a larger k. Prompt changes fix neither. Without the trace, all 18 failures would have looked like "the RAG is bad". In a system with a model call, add spans for the call itself, with prompt tokens, completion tokens, latency and the raw output. OpenTelemetry has published [semantic conventions for generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) that standardise attribute names for these (the page now points to a dedicated repository), so traces can go to any compatible backend.

## Try it

1. Re-run the simulation with true accuracy 90% instead of 50%. Does the chance of a false 10-point gap go up or down? (Hint: variance is p(1 - p).)
2. Add a fourth fault class to `check`: "answer in context but not in the first chunk", which matters for models that read the start of the context best (the position effect from Part 5).
3. Log a trace for every request in a project of your own for a week, then count faults by stage. The largest bucket is the next thing to fix.

## Limits

- The Wilson interval and sign test assume independent questions drawn from the use you care about. A test set of near-duplicates, or one made of questions the system was tuned on, makes every number look better than it will be.
- The tracer is a teaching version: no sampling, no persistence, no redaction of private data in prompts, which production traces need.
- The 18-question set is the same one used for Part 5; the fault counts are as reliable as 18 questions are, which is to say indicative.
- LLM judges were described, not measured here, because no model was available offline.
