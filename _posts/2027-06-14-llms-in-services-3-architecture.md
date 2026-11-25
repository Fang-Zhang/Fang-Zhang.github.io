---
title: "LLMs in Services, Part 3: Put the Policy Outside the Model"
date: 2027-06-14 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, services, architecture, deployment, guardrails, rag, llms in services]
description: "Part 3 of 6 on large language models in service work. A company is liable for what its chatbot says, a deterministic gate outside the model bounds what an attacker can do, and extra retrieval machinery bought four queries out of 115 in a production test, in the wrong direction."
---

## What this post finds

A service assistant has three parts that can be placed in different spots: the knowledge it reads, the actions it may take, and the check that stops a wrong one. Three public sources, from law, security and retrieval, point at the same design: keep the model's job small and put the checks outside it.

- **The company owns the answer.** In *Moffatt v. Air Canada* (2024) a British Columbia tribunal found the airline liable for a chatbot that told a customer a bereavement discount could be claimed after travel; the airline's own linked page said otherwise. The tribunal held the chatbot was part of the website, the company responsible for all information on it, and that a customer cannot be expected to cross-check one part of a site against another. The award was about 650 Canadian dollars plus interest and fees, according to a [summary in Business Law Today](https://www.americanbar.org/groups/business_law/resources/business-law-today/2024-february/bc-tribunal-confirms-companies-remain-liable-information-provided-ai-chatbot) (a tribunal decision, not a court judgment). The amount is small; the principle is not.
- **A gate outside the model bounds what a hijacked model can do.** A 2026 preprint, [Treating the Model as an Untrusted Component](https://arxiv.org/abs/2611.11571), compares in-prompt defences with deterministic enforcement outside the model on an executable environment with 13 tasks and 12 tools. With two open models (Qwen2.5-1.5B and 7B, 156 attack episodes per defence and model), the models proposed the attacker's call in 64 and 84 percent of episodes behind the task-bound monitor, which let through 1.3 and 7.7 percent. A prompt guardrail moved obedience from 67 to 65 percent and from 86 to 55 percent. Implementation quality mattered: a naive monitor admitted 18 of 33 effective attack variants, a hardened one 6, and a capability-bound one 1. The authors flag the limits themselves: two small open models, static attacks, and simulation-only results for human approval and taint tracking. The environment is industrial-control flavoured rather than customer service, and the lesson transfers by analogy.
- **More retrieval machinery did not pay in production.** [Medrano, Verma and Chhabra](https://arxiv.org/abs/2603.02153) (2026) tested retrieval fusion (multiple rewritten queries merged by reciprocal rank fusion) over an enterprise knowledge base with fixed re-ranking and latency budgets. Fusion raised raw recall, but after re-ranking and truncation it did not beat the single-query baseline, with hit@10 falling from 0.51 to 0.48 in several configurations, and it added latency.
- **The counts behind those fractions are small.** The reported fractions are multiples of 1/115, so the 0.51 to 0.48 drop is four queries. One configuration's top-1 fell from 35 to 18 queries, which is large enough to be real.

![Two panels. Left: in a production-style retrieval pipeline over 115 queries, four retrieval-fusion configurations find a correct article in the top ten for 55 queries against 59 for a single-query baseline, with overlapping 95 percent intervals; one configuration drops top-1 from 35 to 18 queries. Right: the chance an attacker has succeeded against attempt count; a model-side guardrail with 50 percent per-attempt obedience and a gate admitting 18 of 33 variants both reach 100 percent within about ten attempts, a gate admitting 6 reaches 90 percent by ten, and a gate admitting 1 rises linearly to 100 percent at 33]({{ site.baseurl }}/assets/pic/services-3-architecture.png){: width="900" }
_Figure: left, counts reconstructed from the published fractions in the Fusion Prompt 1 table; right, an illustrative model using the admitted-variant counts reported in the security preprint._

## What the sources do and do not show

The tribunal decision is a ruling of a provincial tribunal, not a court judgment, read here through a legal summary. It says that the company cannot disclaim its assistant, which in practice means the assistant's statements need the same review as the website's. The security preprint is a single-group preprint whose human-approval and taint-tracking findings are simulation-only. The retrieval study is one enterprise knowledge base with about 115 queries, so it supports "fusion is not a free win", not "fusion never helps".

## What the checks show

Check 1 reconstructs the query counts from the retrieval paper's published fractions and adds intervals and tests. Check 2 is an illustrative model: it takes the admitted-variant counts from the security preprint and asks how fast a persistent attacker finds a variant that gets through, against a model-side guardrail that stops each attempt independently with a fixed probability. The per-attempt obedience of 0.5 and 0.1 are assumptions in the spirit of the preprint's observation that near-50 percent obedience per attempt is enough for persistence to win.

```python
# Part 3 of the services series: two architecture questions answered from published numbers. Standard library only.
import math
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return c - h, c + h
def ztest(k1, k2, n):
    p1, p2 = k1/n, k2/n; pp = (p1+p2)/2; se = math.sqrt(2*pp*(1-pp)/n); return math.erfc(abs(p1-p2)/se/math.sqrt(2))
print("=== Check 1: does retrieval fusion help? (Medrano et al. 2026, Fusion Prompt 1 table; the fractions are multiples of 1/115) ===")
n = 115
rows = [("baseline (single query)", 0.3043, 0.5130), ("rerank on RRF, 1 query", 0.2957, 0.4783), ("rerank on RRF, 2", 0.1565, 0.4783), ("RRF of 2 queries", 0.2522, 0.4783)]
base = None
for name, t1, h10 in rows:
    k1, k10 = round(t1*n), round(h10*n); assert abs(k1/n - t1) < 5e-4 and abs(k10/n - h10) < 5e-4
    lo1, hi1 = wilson(k1, n); lo10, hi10 = wilson(k10, n)
    if base is None: base = (k1, k10)
    extra = "" if name.startswith("baseline") else f"   vs baseline: top-1 {k1-base[0]:+d} queries (p={ztest(k1, base[0], n):.3f}), hit@10 {k10-base[1]:+d} queries (p={ztest(k10, base[1], n):.2f})"
    print(f"{name:26s} top-1 {k1:3d}/115 ({100*lo1:4.1f}-{100*hi1:4.1f}%)   hit@10 {k10:3d}/115 ({100*lo10:4.1f}-{100*hi10:4.1f}%){extra}")
print()
print("=== Check 2: a persistent attacker against a guardrail inside the model and against a gate outside it ===")
print("gate outside the model: variants of the same attack, of which m of 33 are admitted (naive 18, hardened 6, capability-bound 1; arXiv 2611.11571)")
print("guardrail inside the model: each attempt succeeds independently with probability q (the paper measured about 0.5 per attempt for one 1.5B model)")
def found(m, N, k):  # distinct variants tried without repetition
    if k >= N - m + 1: return 1.0
    return 1 - math.comb(N - m, k) / math.comb(N, k)
print("attempts k:                  1      3      5     10     20     33")
for m in (18, 6, 1):
    print(f"gate, {m:2d} of 33 admitted: " + " ".join(f"{found(m, 33, k):6.1%}" for k in (1, 3, 5, 10, 20, 33)) + f"   expected attempts to first admitted variant {(33+1)/(m+1):.1f}")
for q in (0.5, 0.1):
    print(f"model guardrail, q = {q:3.1f}:  " + " ".join(f"{1-(1-q)**k:6.1%}" for k in (1, 3, 5, 10, 20, 33)))

```

Output:

```
=== Check 1: does retrieval fusion help? (Medrano et al. 2026, Fusion Prompt 1 table; the fractions are multiples of 1/115) ===
baseline (single query)    top-1  35/115 (22.8-39.4%)   hit@10  59/115 (42.3-60.2%)
rerank on RRF, 1 query     top-1  34/115 (22.0-38.5%)   hit@10  55/115 (38.9-56.9%)   vs baseline: top-1 -1 queries (p=0.886), hit@10 -4 queries (p=0.60)
rerank on RRF, 2           top-1  18/115 (10.1-23.4%)   hit@10  55/115 (38.9-56.9%)   vs baseline: top-1 -17 queries (p=0.008), hit@10 -4 queries (p=0.60)
RRF of 2 queries           top-1  29/115 (18.2-33.9%)   hit@10  55/115 (38.9-56.9%)   vs baseline: top-1 -6 queries (p=0.377), hit@10 -4 queries (p=0.60)

=== Check 2: a persistent attacker against a guardrail inside the model and against a gate outside it ===
gate outside the model: variants of the same attack, of which m of 33 are admitted (naive 18, hardened 6, capability-bound 1; arXiv 2611.11571)
guardrail inside the model: each attempt succeeds independently with probability q (the paper measured about 0.5 per attempt for one 1.5B model)
attempts k:                  1      3      5     10     20     33
gate, 18 of 33 admitted:  54.5%  91.7%  98.7% 100.0% 100.0% 100.0%   expected attempts to first admitted variant 1.8
gate,  6 of 33 admitted:  18.2%  46.4%  66.0%  90.9%  99.8% 100.0%   expected attempts to first admitted variant 4.9
gate,  1 of 33 admitted:   3.0%   9.1%  15.2%  30.3%  60.6% 100.0%   expected attempts to first admitted variant 17.0
model guardrail, q = 0.5:   50.0%  87.5%  96.9%  99.9% 100.0% 100.0%
model guardrail, q = 0.1:   10.0%  27.1%  41.0%  65.1%  87.8%  96.9%
```

- **Fusion is a tie at hit@10 and a loss at top-1 in one configuration.** All three fusion variants sit at 55 of 115 against 59 for the baseline (p = 0.60), and their intervals overlap almost entirely. Reranking on fused results with the second prompt cut top-1 from 35 to 18 queries (p = 0.008, unpaired). The paper's own conclusion, that retrieval-level gains do not reliably carry through to end-to-end results, is consistent with this.
- **A persistent attacker finds any gap in a gate.** A gate admitting 18 of 33 variants is found in 1.8 attempts on average, as fast as a model-side guardrail that stops half of attempts. A gate admitting 1 of 33 needs 17. A gate is only as strong as the argument-level checks inside it.
- **A gate's value is a bound, not a probability.** The model-side guardrail gets weaker with every attempt, while a gate admits a fixed set of actions. In the preprint's simulation, which assumes a model, the task-bound monitor held severe actions at 0.0 percent even against a compromised model, which is the property a service deployment wants: the worst case is set by what the tools can do, not by what the model decides.
- **Limits.** Check 2 is a toy: real variants differ in how easy they are to find, and the 33 variants are those of one benchmark. Check 1 treats the four configurations as independent samples while they share the same 115 queries, which understates precision.

## What this means for the architecture

1. **Treat every statement as the company's statement.** The answer is checked against the same policy documents as the website, and the assistant cites the page it used so a customer and a reviewer can see it.
2. **Keep policy in code, outside the model.** A refund limit, an identity check or a required disclosure is a rule the tool layer enforces, and the model proposes, never decides. That is also what the process-aware benchmark of Part 1 scores.
3. **Give tools narrow capabilities.** A tool that can only refund the order in the current session has a smaller bound than one that takes an order number from the model. The 1-of-33 monitor above is the capability-bound one.
4. **Add retrieval complexity one measured step at a time.** Run the single-query baseline first, and add fusion or re-ranking only when end-to-end answer quality improves on enough queries to be more than a handful.
5. **Put a human behind the gate, not instead of it.** In the preprint's simulation a human layer without a deterministic monitor fails against approval spamming.

## What comes next

Part 4 covers maintenance: what changes after launch when policies, products and the model all move, and how often a regression suite has to run.

## References

- Business Law Today, [BC Tribunal Confirms Companies Remain Liable for Information Provided by AI Chatbot](https://www.americanbar.org/groups/business_law/resources/business-law-today/2024-february/bc-tribunal-confirms-companies-remain-liable-information-provided-ai-chatbot), February 2024, on *Moffatt v. Air Canada*, 2024 BCCRT 149.
- [Treating the Model as an Untrusted Component: Deterministic Policy Enforcement, Human Approval and Taint Tracking Against Prompt Injection in Tool-Using Agents](https://arxiv.org/abs/2611.11571), arXiv 2611.11571, 2026.
- Medrano, Verma and Chhabra, [Scaling Retrieval Augmented Generation with RAG Fusion: Lessons from an Industry Deployment](https://arxiv.org/abs/2603.02153), 2026.
