---
title: "The LLM Engineer Path, Part 14: Interview Questions on System Design and Rollout"
date: 2027-04-28 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, interview, system design, capacity, A/B testing, guardrails, python]
description: "Part 14, the last of a 14-part series for developers moving into LLM engineering. Four system-design questions, each answered in a sentence and checked with a calculation or simulation: sizing a document Q&A service, rolling out a prompt change, cutting cost and latency, and defending against injection. Ends with an index of the series."
---

The last four questions are the open-ended kind: "design a service" and "how would you ship this". The method for answering them is the same as for the earlier parts: state the requirement, size it with numbers, name the failure, and say how you would measure it. They are classic topics, not questions from any company.

## Q1. Design a question-answering assistant over a company's documents. How do you size it?

**Answer.** Start from the request path (retrieve, build the prompt, generate, check), write down the token budget of one request, then size concurrency with the rate times the time each request spends in the system.

The path is Part 5's pipeline with Part 10's checks. Assume a 300-token system prompt, five retrieved chunks of 200 tokens, a 50-token question and a 250-token answer (these are design assumptions to replace with measurements from your own traces, Part 8):

```python
tokens_in  = 300 + 5 * 200 + 50           # 1,350 per request
tokens_out = 250
per_day_in  = tokens_in  * 86_400         # at 1 request per second: 116.6 million input tokens
per_day_out = tokens_out * 86_400         #                          21.6 million output tokens
```

Multiply by the price list of whichever provider or hardware you use; the structure of the cost is what matters: retrieved context is 74% of every request's input (panel 3 of the figure). Concurrency follows from Little's law: the average number of requests in a system equals the arrival rate times the average time each spends there. It is an identity, and a simulation shows it and the sizing consequence (1 request per second, service time lognormal with mean 9.6 seconds, 40,000 requests):

```
arrivals 0.5/s,  8 slots:  time in system 9.9 s   L = 4.95   rate x time = 4.95
arrivals 1.0/s, 14 slots:  time in system 9.8 s   L = 9.85   rate x time = 9.85   queue wait 0.3 s
arrivals 1.0/s, 11 slots:  time in system 12.3 s  L = 12.24  rate x time = 12.24  queue wait 2.7 s
```

The slot count must exceed rate x service time (here 1.0 x 9.6) or the queue grows without bound; with 10 slots the mean queue wait was 15.9 s, with 14 it was 0.3 s (panel 1). The slots are the product of the KV-cache memory limit (Parts 2, 9 and 13): a bigger cache per request means fewer slots, which means a lower sustainable request rate.

## Q2. How do you roll out a prompt or model change safely?

**Answer.** Run the candidate against the same labelled set as the current version, decide the sample size before looking, compare with a test that accounts for noise, then release gradually with the same metric tracked in production.

How many requests? For a quality drop from 0.80 to 0.75, with a two-sided 5% false-alarm rate and 80% power, the standard formula for two proportions gives:

```python
n = (z(.975) + z(.80)) ** 2 * (p1*(1-p1) + p2*(1-p2)) / (p1 - p2) ** 2     # 1,091 per variant
```

A simulation of 4,000 A/B tests each confirms it:

```
n =  100 per variant: the drop is detected in 13% of tests
n =  300 per variant:                         30%
n = 1000 per variant:                         76%
n = 1090 per variant:                         80%
n = 3000 per variant:                         99%
no real drop (0.80 vs 0.80), n = 1090: 4.9% false alarms
```

This is the arithmetic behind Part 8's warning about small eval sets: a 5-point regression is invisible to a test set of a few hundred examples, and a test set of 23 cannot see it at all. Smaller drops need many more examples (the required n scales with 1 / difference squared). Two practical rules follow: gate releases on a test set sized for the effect you care about, and treat a "no significant difference" result on a small set as "unknown", not as "safe".

## Q3. How do you cut the cost and latency of an LLM service?

**Answer.** Batch requests, shorten what you send, and shrink the model or its numeric precision, verifying quality at each step. The numbers below are the ones measured in this series.

- Continuous batching: in Part 9's simulation at 2 requests per second, mean latency fell from 132.4 s to 4.2 s and throughput rose from 97.7 to 175.6 tokens/s, compared with static batching.
- Shorter context: attention and the KV cache grow with sequence length (Part 11, Q3; Part 13, Q2), and retrieved text is most of the input (Q1 above). Retrieve fewer, better chunks (Part 5), and drop history you do not need.
- Smaller weights: 16-bit to 8-bit halves weight memory (Part 13, Q2), with a measured accuracy cost that must be checked per task (Q3 there).
- Smaller adapters for customisation instead of full fine-tuning: 16 bytes per parameter for full training states, against a small adapter's worth for LoRA (Parts 6 and 13).

Each of these changes quality as well as cost, so each needs the rollout test of Q2 before it ships.

## Q4. How do you defend an LLM application against prompt injection?

**Answer.** Assume the model can be talked into anything, and limit what a talked-into model can do: constrain tools and outputs in code, and treat the model's text as untrusted input.

[OWASP lists prompt injection as the first risk for LLM applications](https://genai.owasp.org/llmrisk/llm01-prompt-injection/), and [Greshake et al.](https://arxiv.org/abs/2302.12173) describe the indirect form, where the instructions arrive inside retrieved or fetched content. Part 10 measured the consequences: a keyword filter blocked 5 of 48 attack variants and 5 of 20 benign inputs, so input filtering alone is not a defence; with the model assumed to obey the attacker, a tool allowlist stopped the destructive-action and email attacks and an output filter stopped the image-link leak, but a falsified summary still got through. The design answer is layers, each covering what the previous one cannot: least-privilege tools, output validation, human confirmation for irreversible actions, and logs (Part 8) so that incidents can be reconstructed.

![Three panels. Left: mean queue wait against parallel slots, 15.9 seconds at 10 slots, falling to 0.3 seconds at 14. Middle: probability of detecting a quality drop from 0.80 to 0.75 against requests per variant, reaching 80% near 1,000 to 1,100. Right: token budget of one request, 300 system prompt, 1,000 retrieved context, 50 question, 250 answer]({{ site.baseurl }}/assets/pic/llmeng-interview4.png){: width="900" }

## The series

1. [The Roadmap]({{ site.baseurl }}/posts/llm-engineer-1-the-roadmap/)
2. [Inference Basics, Tokens, Context and the KV Cache]({{ site.baseurl }}/posts/llm-engineer-2-inference-basics-kv-cache/)
3. [Prompting and Structured Output]({{ site.baseurl }}/posts/llm-engineer-3-prompting-structured-output/)
4. [Embeddings and Retrieval]({{ site.baseurl }}/posts/llm-engineer-4-embeddings-retrieval/)
5. [Retrieval-Augmented Generation]({{ site.baseurl }}/posts/llm-engineer-5-rag/)
6. [Fine-Tuning, Full and LoRA]({{ site.baseurl }}/posts/llm-engineer-6-fine-tuning-lora/)
7. [Agents and Tool Use]({{ site.baseurl }}/posts/llm-engineer-7-agents-tools/)
8. [Evals and Tracing]({{ site.baseurl }}/posts/llm-engineer-8-evals-tracing/)
9. [Serving and Cost]({{ site.baseurl }}/posts/llm-engineer-9-serving-cost/)
10. [Safety and Guardrails]({{ site.baseurl }}/posts/llm-engineer-10-safety-guardrails/)
11. [Interview Questions on the Fundamentals]({{ site.baseurl }}/posts/llm-engineer-11-interview-fundamentals/)
12. [Interview Questions on RAG, Prompting and Agents]({{ site.baseurl }}/posts/llm-engineer-12-interview-rag-agents/)
13. [Interview Questions on Training, Quantization and Serving]({{ site.baseurl }}/posts/llm-engineer-13-interview-training-serving/)
14. This post

## Try it

1. Size a service of your own: write the token budget of one request from real traces, the arrival rate, and the service time, then compute slots with Little's law and compare with your KV-cache limit.
2. Rerun the Q2 simulation for a 0.90 to 0.88 change. How many requests per variant are needed, and what does that imply for how you monitor such a change?
3. Pick any question from Parts 11 to 13 and write the answer in the same four parts: one-sentence answer, source, experiment, limit.

## Limits

- The token budget and the service-time distribution are assumptions chosen to show the method; real values come from your traces.
- The rollout calculation assumes independent samples and a fixed pass/fail metric; paired comparisons on the same questions need fewer examples, and judged or multi-point scores need other tests.
- The simulation treats slots as the only constraint and ignores network, retrieval latency, autoscaling and failures.
