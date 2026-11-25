---
title: "LLMs on the Factory Floor, Part 4: A Weekly Check Needs Hundreds of Questions"
date: 2027-06-09 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, manufacturing, maintenance, monitoring, rag, llms on the factory floor]
description: "Part 4 of 6 on large language models in manufacturing. Detecting drift in a deployed assistant is a sample-size problem: a fixed question set of 100 catches a 10-point drop 28 percent of the time at a one-year false-alarm budget of 5 percent, and stale documents flip correct answers."
---

## What this post finds

A deployed assistant changes for three reasons: the hosted model is updated, the prompt or index is edited, and the documents it retrieves go out of date. The public evidence on each is general, not industrial, and the three studies here come from outside manufacturing. Their mechanisms still apply to a maintenance assistant, and one of them turns into a sizing calculation.

- **The same hosted model changed within three months.** [Chen, Zaharia and Zou](https://arxiv.org/abs/2307.09009) compared the March and June 2023 versions of GPT-3.5 and GPT-4. GPT-4 identified prime versus composite numbers with 84 percent accuracy in March and 51 percent in June, which the authors partly attribute to a drop in following chain-of-thought instructions. Their conclusion is a call for continuous monitoring.
- **Outdated evidence can make a correct model wrong.** [Ahmed, Galke Poech and Röttger](https://arxiv.org/abs/2609.31342) built 317 verified knowledge reversals across medicine, law, software and platform policy. Outdated retrieval flipped 30 percent of Llama and 37 percent of Qwen answers that were correct without retrieval; telling the model to follow the document raised that to 66 and 75 percent. Matched up-to-date evidence was followed in 97 to 100 percent of trials.
- **Dates alone barely help; explicit validity does.** Giving the model an evaluation date produced only modest adaptation, but telling it when the old evidence stops applying let larger models switch almost perfectly. A recency-aware re-ranker cut poisoning by 4.6 to 10.0 points when the dates were accurate.
- **A retrieval benchmark can survive a year of drift.** [Kuissi and colleagues](https://aclanthology.org/2026.findings-emnlp.1754.pdf) compared October 2024 and October 2025 snapshots of a technical-documentation corpus (LangChain). All but one query stayed supported, because relevant documents moved rather than vanished, and model rankings correlated at up to 0.978 (Kendall tau at recall at 50).
- **A weekly regression check is a statistics problem.** At 85 percent baseline accuracy, 100 questions catch a 10-point drop only 28 percent of the time if false alarms are held to 5 percent over a year; 500 questions catch it 99 percent of the time.

![Two panels. Left: chance that one weekly check raises an alarm against the number of questions in a fixed set, for true accuracy drops of 5, 10 and 20 points from 85 percent; 20-point drops reach 80 percent detection at about 70 questions, 10-point drops at about 250, and 5-point drops need roughly 800 or more. Right: answers turned wrong per 1,000 that were right, rising linearly with the share of outdated evidence, steeper at a 75 percent flip rate than at 30 percent]({{ site.baseurl }}/assets/pic/factory-4-maintenance.png){: width="900" }
_Figure: left, exact binomial power of a fixed question set (calculated below); right, answers turned wrong as a function of an assumed stale share, with flip rates from Ahmed et al._

## What the three studies measured

**Model drift.** Chen and colleagues ran the same tasks on two dated snapshots of each hosted model: maths, sensitive questions, opinion surveys, multi-hop knowledge questions, code generation, medical licensing questions and visual reasoning. Some behaviours got worse (GPT-4's prime-number accuracy, code with more formatting mistakes in both models), others better (GPT-4 on multi-hop questions), and GPT-3.5 moved the other way on some tasks. The finding is not that models get worse. It is that the service behind a fixed name changes, in directions nobody announces.

**Stale evidence.** Ahmed and colleagues test whether retrieval can undermine what a model already knows. Across four open models and four domains, poisoning ranged from 17 to 91 percent, while up-to-date evidence was followed almost universally. The authors conclude that reliable retrieval needs selective trust: whether a retrieved document still applies, not only what it says. Their benchmark covers medicine, law, software and platform policy; none of it is a plant manual.

**Benchmark drift.** Kuissi and colleagues study a different question: whether a retrieval test set written in 2024 can still be used in 2025. For one fast-changing documentation corpus the answer was yes, with the caveat that the authors re-judged the evidence on the new snapshot. This post read the abstract and introduction of this paper, not its full results.

## What the arithmetic shows

The first check sizes a fixed question set for a weekly run: the alarm fires if the score falls to or below a threshold chosen so a healthy assistant trips it with probability at most alpha. The second turns the stale-evidence flip rates into answers lost. Baseline accuracy of 85 percent and the stale shares are assumptions chosen to show the mechanism. The binomial thresholds are exact and slightly conservative.

```python
# Sizing a regression suite for a deployed assistant, plus the arithmetic of stale evidence. Exact binomial; standard library only.
from math import comb

def cdf(k, n, p):                       # P(X <= k)
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(0, k + 1))

def threshold(n, p0, alpha):            # largest k with P(X <= k | p0) <= alpha  (alarm if score <= k)
    k = -1
    while k + 1 <= n and cdf(k + 1, n, p0) <= alpha: k += 1
    return k

P0 = 0.85                               # assumed baseline accuracy of the assistant on the fixed question set
print("=== Check 1: power of a fixed question set to notice a drop from a baseline accuracy of 85% (assumed) ===")
print("alarm when the score falls to or below a threshold chosen for a per-check false-alarm rate alpha")
for alpha in (0.05, 0.001):
    print(f"\nper-check false-alarm rate alpha = {alpha}   (chance of at least one false alarm in 52 weekly checks: {1-(1-alpha)**52:.0%})")
    print(f"{'questions':>10} {'alarm at <=':>12} " + " ".join(f"{'drop '+str(int(d*100))+' pts':>10}" for d in (0.03, 0.05, 0.10, 0.20, 0.34)))
    for n in (20, 50, 100, 200, 500, 1000):
        k = threshold(n, P0, alpha)
        row = [cdf(k, n, P0 - d) if k >= 0 else 0.0 for d in (0.03, 0.05, 0.10, 0.20, 0.34)]
        print(f"{n:>10} {k:>12} " + " ".join(f"{r:>10.0%}" for r in row))
print("\n(0.34 points is the drop Chen et al. report for GPT-4 on prime-number identification, 84% to 51%; it is used here as a size, not as a claim about plants)")

print("\n=== Check 2: what a share of stale evidence costs (flip rates from Ahmed et al.; stale shares are assumed inputs) ===")
print("flip rates: 30% (Llama) and 37% (Qwen) with neutral retrieval; 66% and 75% when told to follow the document")
print(f"{'stale share of retrieved evidence':>34} {'wrong answers per 1,000 that were right (30%)':>48} {'(75%)':>8}")
for s in (0.01, 0.05, 0.10, 0.20, 0.40):
    print(f"{s:>34.0%} {1000*s*0.30:>48.0f} {1000*s*0.75:>8.0f}")

```

Output:

```
=== Check 1: power of a fixed question set to notice a drop from a baseline accuracy of 85% (assumed) ===
alarm when the score falls to or below a threshold chosen for a per-check false-alarm rate alpha

per-check false-alarm rate alpha = 0.05   (chance of at least one false alarm in 52 weekly checks: 93%)
 questions  alarm at <= drop 3 pts drop 5 pts drop 10 pts drop 20 pts drop 34 pts
        20           13         5%         9%        21%        58%        93%
        50           37        10%        19%        49%        93%       100%
       100           78        18%        35%        79%       100%       100%
       200          161        32%        60%        97%       100%       100%
       500          411        56%        90%       100%       100%       100%
      1000          830        81%        99%       100%       100%       100%

per-check false-alarm rate alpha = 0.001   (chance of at least one false alarm in 52 weekly checks: 5%)
 questions  alarm at <= drop 3 pts drop 5 pts drop 10 pts drop 20 pts drop 34 pts
        20           10         0%         0%         1%        12%        55%
        50           33         1%         1%        10%        61%        99%
       100           72         1%         3%        28%        94%       100%
       200          152         2%         9%        65%       100%       100%
       500          398         9%        43%        99%       100%       100%
      1000          813        29%        86%       100%       100%       100%

(0.34 points is the drop Chen et al. report for GPT-4 on prime-number identification, 84% to 51%; it is used here as a size, not as a claim about plants)

=== Check 2: what a share of stale evidence costs (flip rates from Ahmed et al.; stale shares are assumed inputs) ===
flip rates: 30% (Llama) and 37% (Qwen) with neutral retrieval; 66% and 75% when told to follow the document
 stale share of retrieved evidence    wrong answers per 1,000 that were right (30%)    (75%)
                                1%                                                3        8
                                5%                                               15       38
                               10%                                               30       75
                               20%                                               60      150
                               40%                                              120      300
```

- **Small sets cannot see small drops.** With 100 questions and a 5 percent false-alarm rate per check, a 5-point drop is caught 35 percent of the time and a 10-point drop 79 percent of the time. The false-alarm rate is the catch: 52 checks at that setting give a 93 percent chance of at least one false alarm in a year.
- **A sensible alarm needs more questions.** At 0.1 percent per check, which holds the yearly false-alarm chance to 5 percent, 100 questions catch a 10-point drop 28 percent of the time, 200 catch it 65 percent, and 500 catch it 99 percent. A 5-point drop needs about 1,000 questions for 86 percent.
- **A drop the size of the GPT-4 example is easy.** A 34-point drop is caught by 50 questions at 99 percent power, even at the strict alarm rate. The size comes from a different task and is used here only as a magnitude.
- **Stale evidence scales linearly with its share.** If 10 percent of retrieved evidence is outdated, the flip rates imply 30 answers per 1,000 turned wrong at 30 percent flipping, and 75 at 75 percent. These are expected values from assumed inputs, not measured plant rates.
- **Limits.** The flip rates come from small open models in four non-industrial domains. The power calculation assumes independent questions and a fixed baseline, and a question set drawn from one equipment family would be correlated, as Part 1's rule-level clustering showed, so the effective sample is smaller than the count.

## What this means for maintenance

1. **Pin and log the model version.** The hosted-model snapshots differed within three months, so a version string belongs in every logged answer, and a change in version is a trigger for a full re-test.
2. **Size the regression set for the drop that matters.** For a 10-point drop at a one-year false-alarm budget of 5 percent, a few hundred questions; for 5 points, about a thousand. Add questions in clusters across equipment types and measure them separately.
3. **Re-run on every change, not on a calendar alone.** A new model version, an edited prompt, a re-indexed document set and a changed retriever each get the same set. Weekly runs catch what nobody announced.
4. **Store validity with every document.** Revision, effective date and the identity of the document that supersedes it belong in the index metadata, and the prompt should state which documents are no longer applicable. Dates alone gave little control in the study; explicit validity did.
5. **Keep a frozen index snapshot.** Kuissi and colleagues show that a benchmark can be re-judged against a new corpus; the old snapshot is what allows the comparison.
6. **State what is not known.** None of these studies follows a maintenance assistant through a plant's document revisions. How fast manuals, work-order codes and part numbers change on a real site is an open input.

## What comes next

Part 5 puts numbers on cost and benefit: what a plant spends to run an assistant, and what it takes to recover that spend.

## References

- Chen, Zaharia and Zou, [How is ChatGPT's behavior changing over time?](https://arxiv.org/abs/2307.09009), 2023.
- Ahmed, Galke Poech and Röttger, [Stale-Document Poisoning: When Outdated Retrieval Overrides Correct Model Answers](https://arxiv.org/abs/2609.31342), 2026.
- Kuissi, Subrahmanyan, Thakur and Lin, [Retrieval Evaluation Under Temporal Drift in Technical Documentation](https://aclanthology.org/2026.findings-emnlp.1754.pdf), Findings of EMNLP 2026.
