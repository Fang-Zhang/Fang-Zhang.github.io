---
title: "The LLM Engineer Path, Part 1: The Roadmap"
date: 2027-04-15 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, AI engineering, roadmap, career, learning path, python]
description: "Part 1 of a 14-part series for software developers moving into LLM engineering. What the job is, a five-stage skill map with the prerequisites written as code, which parts of the field to skip at first, and what the next nine posts cover before four posts of interview questions with worked answers."
---

This series is for a developer with a year or two of backend experience and working Python who wants to build LLM applications for a living. Part 1 defines the job, lays out a learning order as a dependency graph you can run, and says what to skip. Parts 2 to 10 take the nine core topics one at a time. Parts 11 to 14 are interview questions with worked answers.

## What the job is

Titles vary: "LLM engineer", "AI engineer", "applied AI engineer". They overlap, so read the job description rather than the title. A useful definition comes from Chip Huyen's book [*AI Engineering*](https://www.oreilly.com/library/view/ai-engineering/9781098166298/), which describes the field as "the process of building applications with readily available foundation models" and explains "how it differs from traditional ML engineering". Its list of topics is a fair map of the work: prompt engineering, RAG (retrieval-augmented generation), fine-tuning, agents, dataset engineering, evaluation, and the "bottlenecks for latency and cost when serving foundation models".

The essay that popularised the title, [The Rise of the AI Engineer](https://www.latent.space/p/ai-engineer), argues the role is a software-engineering subdiscipline, and quotes Andrej Karpathy: "One can be quite successful in this role without ever training anything." Treat both as opinions about a field that changes quickly. The shape of the work they describe is this: you are mostly building software around a model you did not train, and most of the difficulty is in measuring whether it works.

## The skill map

![Five columns left to right: foundations (Python and HTTP APIs, ML basics), use a model (Transformer and inference, prompting and structured output), measure and ground (evals and tracing, embeddings and retrieval), adapt, act and run (RAG, fine-tuning, agents and tools, serving and cost), and protect (safety and guardrails), each labelled with the part of this series that covers it]({{ site.baseurl }}/assets/pic/llmeng-roadmap.png){: width="900" }
*The series order, grouped by stage. The grouping follows what each skill depends on.*

The same map as code. Each skill lists what it builds on; Python's standard-library `graphlib` sorts them into a workable order, and a small function reports which prerequisites you are missing.

```python
from graphlib import TopologicalSorter

# skill -> the skills it builds on
needs = {
    "python + http apis":   [],
    "ml basics":            [],                                   # loss, gradient descent, train/test split
    "transformer + inference": ["ml basics"],
    "prompting + structured output": ["python + http apis"],
    "embeddings + retrieval": ["python + http apis", "transformer + inference"],
    "evals + tracing":      ["prompting + structured output"],
    "rag":                  ["embeddings + retrieval", "evals + tracing"],
    "fine-tuning":          ["transformer + inference", "evals + tracing"],
    "agents + tools":       ["prompting + structured output", "evals + tracing"],
    "safety + guardrails":  ["agents + tools", "rag"],
    "serving + cost":       ["transformer + inference", "evals + tracing"],
}

order = list(TopologicalSorter(needs).static_order())
for i, skill in enumerate(order, 1):
    print(i, skill)

def gaps(known, target):
    """Skills you still need before `target`, in a workable order."""
    todo = set()
    def visit(s):
        for p in needs[s]:
            if p not in known and p not in todo:
                todo.add(p); visit(p)
    visit(target)
    return [s for s in order if s in todo]

print("\nknown: python + http apis. target: agents + tools")
print(gaps({"python + http apis"}, "agents + tools"))
```

Output, from a real run:

```
1 python + http apis
2 ml basics
3 prompting + structured output
4 transformer + inference
5 evals + tracing
6 embeddings + retrieval
7 fine-tuning
8 agents + tools
9 serving + cost
10 rag
11 safety + guardrails

known: python + http apis. target: agents + tools
['prompting + structured output', 'evals + tracing']
```

The dependencies are the author's judgement, not a standard. Two choices are deliberate.

- **Evals come before almost everything.** RAG, fine-tuning, agents and serving all need an answer to "did that change make it better?". Without a test set, each of them becomes guessing. Part 8 explains the method; the habit starts at stage 2.
- **Safety comes last in the order, not last in importance.** It depends on agents and retrieval because the largest risks appear when a model reads untrusted text or can act. Part 10 covers it, and each earlier part ends with the risks of its topic.

## The failure to reproduce: skipping the prerequisites

The last two lines of the output are the pitfall. A developer who knows Python and HTTP APIs and goes straight to "agents" has two gaps: prompting with structured output, and evals. In practice that looks like an agent that loops on malformed tool calls and has no test that would show it. Run `gaps` for any target with your own `known` set and read the list before choosing what to study.

## Stage by stage

Each stage names one thing to build. These are suggestions for practice, not requirements of any job.

1. **Foundations.** Python, HTTP, JSON, testing. For ML basics, the loss function, gradient descent and train/test splits are enough to read the later parts. Andrej Karpathy's course [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) builds a GPT in code starting from backpropagation; its prerequisites are "solid programming (Python), intro-level math". Build: a tiny model trained on a text file.
2. **Use a model.** The Transformer, covered in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), and how text generation runs (Part 2). Prompting and getting structured output you can parse (Part 3). Build: a command-line tool that extracts fields from messy text and validates the result against a schema.
3. **Measure and ground.** Evaluation and tracing (Part 8), and embeddings with vector search (Part 4). Build: a 50-question test set for one task and a script that scores any change against it.
4. **Adapt, act, run.** RAG (Part 5), fine-tuning (Part 6), agents with tool calling (Part 7), serving and cost (Part 9). Build: a question-answering app over your own documents, scored on the test set, with its cost per request measured.
5. **Protect.** Prompt injection, data leakage and output checks (Part 10). Build: a red-team list of 20 attacks against the stage 4 app, and the fixes.

Free courses fill the gaps between posts: the Hugging Face [LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) ("completely free and without ads", built on its Transformers, Datasets, Tokenizers and Accelerate libraries) for the tooling, and Stanford's [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) for depth on how models are built.

## What to skip at first

- **Pre-training a model from scratch.** It is a different job, done by a small number of teams. Understanding it (the courses above) helps; doing it is not a prerequisite for building applications.
- **Chasing every new framework.** Most orchestration libraries wrap the same few operations: call a model, retrieve, parse, retry. If you understand those, a new framework takes an afternoon.
- **Reading every paper.** The papers cited in this series are enough to start with.

## Try it

1. Copy the code, set `known` to the skills you already have, and print `gaps` for each target. Which three skills would unlock the most targets?
2. Add a skill from your own work (for example "sql" or "kubernetes") to `needs`. What does it depend on, and what depends on it?
3. Write the 50-question test set from stage 3 for a task you care about before reading Part 8. Which questions were hard to write, and why?

## Limits

- The dependency graph is one reasonable order. Other orders work, and the graph says nothing about how long each stage takes.
- The definitions and quotes come from one book page and one essay. Role definitions differ by company; a job description is the authority for a given job.
- No salary or demand figures appear here because no source that could be checked is cited for them.
