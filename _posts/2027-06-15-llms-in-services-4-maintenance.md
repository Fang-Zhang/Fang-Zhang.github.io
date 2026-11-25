---
title: "LLMs in Services, Part 4: The Model Retires on Someone Else's Schedule"
date: 2027-06-15 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, services, maintenance, monitoring, evaluation, llms in services]
description: "Part 4 of 6 on large language models in service work. 82 percent of migrations away from retired models happened after the shutdown, an automatic judge turns a 9-point gap into 1 to 4 points, and a policy change that hits 5 percent of conversations hides inside the overall error rate."
---

## What this post finds

A deployed service assistant changes after launch even if nobody touches it: the provider retires the model, the policy documents move, and the judge that scores the conversations has its own error. Two 2026 preprints measure the first and third; the second follows from arithmetic.

- **Retirement is on the provider's schedule, and most applications are late.** [Kim](https://arxiv.org/abs/2609.31288) mined GitHub for commits that migrate away from officially deprecated OpenAI, Anthropic and Google models: 22,555 commits in 17,703 non-fork repositories, 5,139 matched to an official announcement, and a stratified sample of 300 coded by two coders (kappa 0.89 to 0.95). An estimated 82 percent (95 percent interval 79 to 84) of migrations were committed after the shutdown date, that is, after the application had started failing. The share was 89 percent for Anthropic's 60 to 114 day notices and 13 percent for OpenAI's one-year Assistants API notice. Model identifiers were hard-coded in 94 percent of migrating applications, and median migration effort rose from 6 added lines for prompt-only applications to nearly 700 for fine-tuned ones. These are open-source applications, so the numbers describe developers' habits, not any one company's service desk.
- **The automatic judge is not a fixed ruler.** [Li](https://arxiv.org/abs/2609.34198) re-scored public τ-bench conversations of GPT-4o and Claude 3.5 Sonnet (new): 155 confirmation tasks (47 airline, 108 retail), four attempts each, so 1,240 conversations per judge. On retail the environment reward puts the two agents at 71.1 and 62.0 percent. One judge confidently reversed that ordering, and the paper's exploratory explanation is that the judge penalized a habit the reward ignores: the policy says not to call a tool and talk to the user in the same turn, and the agent that violated it did so 7.4 times per successful retail conversation, against 0.5 for the other. Judge errors differed widely: one accepted 57 percent of failed conversations, another rejected 79 percent of successful ones.
- **Judges carried little information about the comparison.** The median correlation between judged and reference paired differences was 0.19 on airline and 0.14 on retail.
- **A stale document is a maintenance problem, not a model problem.** Part 4 of the factory series covered retrieval of outdated documents (Ahmed and colleagues); the same mechanism applies when a refund policy is edited.

![Two panels. Left: a real 9.1-point gap between two agents on tau-bench retail shrinks to 1.3 to 4.3 points when seen through four judges with their published error rates. Right: the share of migrations committed after a model's shutdown against notice length; a curve with the paper's slope anchored at the one-year, 13 percent point passes well below the observed 89 percent for 60 to 114 day notices]({{ site.baseurl }}/assets/pic/services-4-maintenance.png){: width="900" }
_Figure: left, the 9.1-point reference gap seen through the published judge error rates; right, late-migration share against notice length._

## What the sources do and do not show

The migration paper is an observational study of public repositories; the commit is a proxy for the moment a team acted, and the text read for this post was the abstract (the full text did not load), so details beyond it are not used. The judge paper is a preprint that ran four judges on public trajectories; its τ-bench reversal is explained by the authors as a mismatch between the reward (database state) and the judge's instruction (full policy compliance), and the paper notes the stricter judge may be right by a human policy standard. Neither paper is about a service company's production traffic.

## What the checks show

Check 1 asks whether notice length alone explains who migrates late, by taking the paper's stated slope (read as an odds ratio of 0.25 per e-fold of notice) and anchoring it at the one-year point. Check 2 asks what a judge with the published error rates would do to the real 9.1-point retail gap if its errors were the same for both agents, which is the best case. Check 3 is an arithmetic example with assumed numbers: a policy change hits a slice of conversations, and the question is how many labelled conversations it takes to notice.

```python
# Part 4 of the services series: three maintenance questions answered from published numbers. Standard library only.
import math
from statistics import NormalDist
N = NormalDist()
print("=== Check 1: does notice length explain who migrates late? (Kim 2026, abstract-level numbers) ===")
odds = lambda p: p / (1 - p)
b = math.log(4)   # 'each e-fold increase in notice length reduces the odds by about three quarters' read as an odds ratio of 0.25
anchor_days, anchor_p = 365, 0.13   # OpenAI Assistants API, one-year notice: 13% migrated after shutdown
def predicted(days): o = odds(anchor_p) * (anchor_days / days) ** b; return o / (1 + o)
for days in (60, 87, 114, 200, 365):
    print(f"notice {days:3d} days: predicted post-shutdown share {predicted(days):5.1%}")
print("observed: Anthropic, 60-114 day notices: 89%; OpenAI, 365 days: 13%")
print()
print("=== Check 2: what a judge that is wrong at a fixed rate does to a real 9-point gap (Li 2026, tau-bench retail) ===")
claude, gpt4o = 0.711, 0.620
print(f"environment-reward pass rates, retail: {claude:.1%} vs {gpt4o:.1%}, gap {100*(claude-gpt4o):.1f} points")
for name, fp, fn in (("GPT-5 mini", .57, .19), ("GPT-6 Sol", .07, .79), ("Claude Opus 5.5", .28, .25), ("Gemini 3.5 Flash", .34, .42)):
    seen = lambda p: p * (1 - fn) + (1 - p) * fp
    print(f"{name:17s} accepts {fp:.0%} of failures, rejects {fn:.0%} of successes -> judged {seen(claude):.1%} vs {seen(gpt4o):.1%}, gap {100*(seen(claude)-seen(gpt4o)):.1f} points")
print()
print("=== Check 3: monitoring the whole stream or the slice a policy change touches (assumed numbers) ===")
base, slice_share, slice_err_after = 0.10, 0.05, 0.50
overall_after = base * (1 - slice_share) + slice_err_after * slice_share
def n_detect(p0, p1, alpha=0.05, power=0.8):
    za, zb = N.inv_cdf(1 - alpha), N.inv_cdf(power)
    return math.ceil(((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2)
def binom_tail(n, k, p): return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n + 1))
def exact_n(p0, p1, alpha=0.05, power=0.8):   # smallest n with a threshold rule: alarm if errors >= k, false alarm <= alpha, power >= target
    for n in range(2, 400):
        k = next(k for k in range(n + 2) if binom_tail(n, k, p0) <= alpha)
        if k <= n and binom_tail(n, k, p1) >= power: return n, k
n_all = n_detect(base, overall_after); n_slice, k_slice = exact_n(base, slice_err_after)
print(f"baseline error {base:.0%}; a policy change hits {slice_share:.0%} of conversations and takes their error to {slice_err_after:.0%}; overall error becomes {overall_after:.1%}")
print(f"labelled conversations to detect it (false alarm at most 5%, power at least 80%): whole stream about {n_all} (normal approximation), slice only {n_slice} (alarm at {k_slice} or more errors, exact binomial)")
print(f"at 500 conversations a day, {slice_share:.0%} in the slice: the whole-stream sample is {n_all} labels; the slice is {n_slice} labels out of the {int(500*slice_share)} slice conversations of one day")

```

Output:

```
=== Check 1: does notice length explain who migrates late? (Kim 2026, abstract-level numbers) ===
notice  60 days: predicted post-shutdown share 64.6%
notice  87 days: predicted post-shutdown share 52.2%
notice 114 days: predicted post-shutdown share 42.9%
notice 200 days: predicted post-shutdown share 25.6%
notice 365 days: predicted post-shutdown share 13.0%
observed: Anthropic, 60-114 day notices: 89%; OpenAI, 365 days: 13%

=== Check 2: what a judge that is wrong at a fixed rate does to a real 9-point gap (Li 2026, tau-bench retail) ===
environment-reward pass rates, retail: 71.1% vs 62.0%, gap 9.1 points
GPT-5 mini        accepts 57% of failures, rejects 19% of successes -> judged 74.1% vs 71.9%, gap 2.2 points
GPT-6 Sol         accepts 7% of failures, rejects 79% of successes -> judged 17.0% vs 15.7%, gap 1.3 points
Claude Opus 5.5   accepts 28% of failures, rejects 25% of successes -> judged 61.4% vs 57.1%, gap 4.3 points
Gemini 3.5 Flash  accepts 34% of failures, rejects 42% of successes -> judged 51.1% vs 48.9%, gap 2.2 points

=== Check 3: monitoring the whole stream or the slice a policy change touches (assumed numbers) ===
baseline error 10%; a policy change hits 5% of conversations and takes their error to 50%; overall error becomes 12.0%
labelled conversations to detect it (false alarm at most 5%, power at least 80%): whole stream about 1471 (normal approximation), slice only 8 (alarm at 3 or more errors, exact binomial)
at 500 conversations a day, 5% in the slice: the whole-stream sample is 1471 labels; the slice is 8 labels out of the 25 slice conversations of one day
```

- **Notice length does not explain the pattern alone.** The slope predicts 43 to 65 percent late migrations for 60 to 114 day notices, and 89 percent was observed. The anchor is one product (the Assistants API) with one notice, so the curve is a reading of the paper's number and not a fit. The gap is a reason to read the 82 percent as a floor on how late teams are, not as a function of one variable.
- **Even an unbiased judge shrinks a real gap.** At the published error rates, a 9.1-point gap reads as 1.3 to 4.3 points. The paper's finding is worse, because the errors differ by agent (a reversal), which this best case leaves out. The error rates are averages over the two agents, so the calculation is approximate.
- **Slice monitoring is cheaper by two orders of magnitude.** Under the assumed numbers (10 percent baseline error, a change that hits 5 percent of conversations and takes their error to 50 percent) the overall error moves only from 10 to 12 percent, and noticing needs about 1,500 labelled conversations from the whole stream. Looking at the slice needs 8 labelled conversations, which is one day of slice traffic at 500 conversations a day. The assumptions are illustrative; the ratio holds whenever the change is concentrated.
- **Limits.** Check 2 reads the paper's judge rates as accept and reject shares over both agents. Check 3 assumes the team knows which slice a change touches, and that labels are correct.

## What this means for maintenance

1. **Treat the model version as a dependency with an end date.** Keep the identifier in configuration, record each provider's notice, and start migration when the notice arrives instead of when requests fail. The paper's 94 percent hard-coded rate is the thing to avoid.
2. **Budget for fine-tuned models separately.** The median effort grew from 6 lines to nearly 700, because tuned models carry their data and evaluation with them (Part 2).
3. **Keep a locked regression set by policy slice.** Every model, prompt, tool or document change re-runs it; each policy document is linked to the slice of tasks that depend on it, so an edit triggers a targeted review.
4. **Audit the judge with human labels on the comparison that matters.** A judge's overall accuracy does not say whether it ranks the old and new versions correctly. Paired human review of a few dozen current conversations is the reference.
5. **Score what the policy asks.** If full policy compliance is the standard, the reward has to measure it, as Part 1 argued.

## What comes next

Part 5 covers cost and benefit: what a service deployment saves per conversation and where the savings are measured, with the maintenance burden above counted against them.

## References

- Kim, [When the Model Retires: An Empirical Study of LLM Migration in Open-Source Applications](https://arxiv.org/abs/2609.31288), 2026.
- Li, [Frozen Judges, Moving Agents: Version-Dependent LLM-Judge Error and the Limits of Judge-Assisted Agent Evaluation](https://arxiv.org/abs/2609.34198), 2026.
- Yao, Shinn, Razavi and Narasimhan, [τ-bench](https://arxiv.org/abs/2406.12045), ICLR 2025.
