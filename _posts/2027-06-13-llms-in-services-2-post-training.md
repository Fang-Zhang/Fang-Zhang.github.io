---
title: "LLMs in Services, Part 2: Tuning Teaches the Procedure, Not the Facts"
date: 2027-06-13 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, services, fine-tuning, post-training, customer service, llms in services]
description: "Part 2 of 6 on large language models in service work. Fine-tuning on verified synthetic conversations lifted a 70B open model by 17 to 19 points on a customer-service benchmark, but the claimed edge over GPT-4o is inside the noise of a 50-task test, and fine-tuning on new facts is the part that backfires."
---

## What this post finds

Part 1 showed that a service benchmark has to score procedure as well as outcome. Part 2 asks what post-training can change. The public record has one strong positive result, one controlled warning, and one result on judgment. Together they separate what tuning teaches reliably from what it teaches badly.

- **Tuning on verified multi-turn conversations works.** [APIGen-MT](https://arxiv.org/abs/2504.03601) (Prabhakar and colleagues, 2025) generates task blueprints with ground-truth actions, turns them into simulated customer-agent conversations and trains open models on the filtered trajectories. The tuned xLAM-2-70b-fc-r scores 67.1 percent on τ-Retail and 45.2 percent on τ-Airline, against 50.4 and 26.0 for Llama 3.1 70B Instruct in the same table (the 70B variant is trained from the Llama family; this pairing is inferred, since the families listed are Llama 3.1/3.2 and Qwen 2.5).
- **The claimed edge over frontier models is not resolvable at that test size.** τ-Retail has 115 tasks and τ-Airline 50. The tuned model's gaps to GPT-4o (+4.3 and +2.2 points) and to Claude 3.5 Sonnet (-4.4 and -3.6) all sit well inside the intervals.
- **The training data is built from the benchmark's own domain.** The pipeline uses the tools and policies of τ-bench as its guideline, so the result is an in-domain adaptation: the tuned model learned this policy, which is what a service deployment wants and also why the result does not transfer to other policies on its own evidence.
- **Consistency improved, per the paper.** On the airline domain, the 70B tuned model has a higher pass^5 than Claude 3.5 Sonnet despite a slightly lower pass^1, according to the paper's figure.
- **New facts are the weak spot.** [Gekhman and colleagues](https://arxiv.org/abs/2405.05904) (2024 preprint) fine-tuned a PaLM 2-S base model on closed-book questions and found that examples with new knowledge are learned much more slowly than examples the model already knows, and that once they are learned they raise the tendency to hallucinate. Their summary: models acquire facts in pre-training, while fine-tuning teaches them to use those facts more efficiently.
- **Judgment needs explanations.** [Teaching AI to Handle Exceptions](https://arxiv.org/abs/2503.02976) (DiSorbo, Ju and Aral, 2025) found that models follow policy rigidly where humans grant exceptions, that prompting with an ethical framework failed, chain-of-thought gave slight gains, and fine-tuning on human explanations (not just yes/no labels) gave markedly better alignment, including on new scenarios.

![Two panels. Left: pass@1 with 95 percent intervals on tau-Retail (115 tasks) and tau-Airline (50 tasks) for an untuned Llama 3.1 70B, a tuned 70B model, GPT-4o and Claude 3.5 Sonnet; the intervals are about 17 to 28 points wide. Right: gaps of the tuned model to each other model; the gaps to the untuned base (plus 16.7 retail, plus 19.2 airline) clear zero, while the gaps to GPT-4o and Claude straddle zero]({{ site.baseurl }}/assets/pic/services-2-post-training.png){: width="900" }
_Figure: scores from Table 2 of Prabhakar et al.; intervals and gaps computed here from the task counts._

## What each study does and does not show

APIGen-MT is a single-team result on one benchmark whose tools and policy it used to generate the training tasks, and its scores are pass@1 averaged over at least five trials. The paper compares against base models in its own table, so the +17 to +19 point gain over the untuned base is the cleanest reading. How much of the gain comes from learning this particular policy is the point of tuning, not a flaw, but it also means the model has to be tuned again when the policy changes. Whether any training task duplicates a test task is not something the text read here settles.

Gekhman and colleagues use factual questions on a base model, not service dialogue, so the warning carries over only as a mechanism: if the tuning data asserts facts the model does not hold (a price, a fee schedule, a coverage rule), the model learns them slowly and is more likely to produce wrong ones afterwards. The exceptions paper uses scenario prompts with human respondents and proprietary and open models, and shows that policy-following is a behaviour that can be tuned with the reasoning attached.

## What the arithmetic shows

The script takes the published scores and the benchmark's task counts, reports 95 percent Wilson intervals and unpaired two-proportion tests, and works out what the data pipeline costs per usable example from the paper's reported success rates. It treats the 115 and 50 tasks as the sample (trial-averaging removes some noise but not task-sampling noise), so the intervals are approximate.

```python
# Part 2 of the services series: what fine-tuning on a service policy buys, from published numbers (APIGen-MT, Prabhakar et al. 2025, Table 2). Standard library only.
import math
N = {"Retail": 115, "Airline": 50}   # tau-bench task counts (Yao et al.); scores are pass@1 averaged over at least 5 trials
S = {"Retail":  {"Llama 3.1 70B Instruct (untuned)": 50.4, "xLAM-2-70b-fc-r (tuned)": 67.1, "GPT-4o": 62.8, "Claude 3.5 Sonnet (new)": 71.5},
     "Airline": {"Llama 3.1 70B Instruct (untuned)": 26.0, "xLAM-2-70b-fc-r (tuned)": 45.2, "GPT-4o": 43.0, "Claude 3.5 Sonnet (new)": 48.8}}
def wilson(p, n, z=1.96):
    c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return c - h, c + h
def ztest(p1, p2, n):
    pp = (p1 + p2) / 2; se = math.sqrt(2 * pp * (1 - pp) / n); z = abs(p1 - p2) / se
    return math.erfc(z / math.sqrt(2))
print("=== Check 1: which published gaps survive the size of the test set? (unpaired, tasks as the sample) ===")
for dom, n in N.items():
    print(f"tau-{dom} ({n} tasks)")
    for m, v in S[dom].items():
        lo, hi = wilson(v/100, n); print(f"   {m:34s} {v:5.1f}   95% interval {100*lo:4.1f} to {100*hi:4.1f}")
    t = S[dom]["xLAM-2-70b-fc-r (tuned)"] / 100
    for m in ("Llama 3.1 70B Instruct (untuned)", "GPT-4o", "Claude 3.5 Sonnet (new)"):
        print(f"   tuned vs {m:34s} gap {100*(t-S[dom][m]/100):+5.1f} points, two-sided p = {ztest(t, S[dom][m]/100, n):.3f}")
print()
print("=== Check 2: what the data pipeline costs per training example (Figure 4 of the paper: 70% of task configs survive with agentic feedback, 28% without; 67% of trajectory simulations succeed) ===")
for name, a in (("with agentic feedback", 0.70), ("without", 0.28)):
    y = a * 0.67
    print(f"{name:24s} yield {y:.1%} per attempt -> about {5000/y:,.0f} generation attempts for 5,000 trajectories (if the two rates multiply)")

```

Output:

```
=== Check 1: which published gaps survive the size of the test set? (unpaired, tasks as the sample) ===
tau-Retail (115 tasks)
   Llama 3.1 70B Instruct (untuned)    50.4   95% interval 41.4 to 59.4
   xLAM-2-70b-fc-r (tuned)             67.1   95% interval 58.1 to 75.0
   GPT-4o                              62.8   95% interval 53.7 to 71.1
   Claude 3.5 Sonnet (new)             71.5   95% interval 62.7 to 79.0
   tuned vs Llama 3.1 70B Instruct (untuned)   gap +16.7 points, two-sided p = 0.010
   tuned vs GPT-4o                             gap  +4.3 points, two-sided p = 0.494
   tuned vs Claude 3.5 Sonnet (new)            gap  -4.4 points, two-sided p = 0.469
tau-Airline (50 tasks)
   Llama 3.1 70B Instruct (untuned)    26.0   95% interval 15.9 to 39.6
   xLAM-2-70b-fc-r (tuned)             45.2   95% interval 32.2 to 58.8
   GPT-4o                              43.0   95% interval 30.3 to 56.7
   Claude 3.5 Sonnet (new)             48.8   95% interval 35.5 to 62.2
   tuned vs Llama 3.1 70B Instruct (untuned)   gap +19.2 points, two-sided p = 0.045
   tuned vs GPT-4o                             gap  +2.2 points, two-sided p = 0.825
   tuned vs Claude 3.5 Sonnet (new)            gap  -3.6 points, two-sided p = 0.718

=== Check 2: what the data pipeline costs per training example (Figure 4 of the paper: 70% of task configs survive with agentic feedback, 28% without; 67% of trajectory simulations succeed) ===
with agentic feedback    yield 46.9% per attempt -> about 10,661 generation attempts for 5,000 trajectories (if the two rates multiply)
without                  yield 18.8% per attempt -> about 26,652 generation attempts for 5,000 trajectories (if the two rates multiply)
```

- **Tuning versus its own base is a real gain.** +16.7 points on retail (p = 0.010) and +19.2 on airline (p = 0.045, and the interval's lower end is close to zero).
- **"Outperforms GPT-4o" is a tie.** The retail gap is +4.3 points with p = 0.49, and the airline gap is +2.2 with p = 0.83. A 50-task test cannot separate models that are 10 points apart.
- **Data quality has a measurable price.** With agentic feedback, 70 percent of generated task configurations survive, against 28 percent without, and 67 percent of conversation simulations succeed. If the two rates multiply, about 10,700 generation attempts yield 5,000 verified trajectories, against about 26,700 without the feedback loop. The multiplication is an assumption, and the paper's released set is 5,000 trajectories after filtering.
- **Limits.** Unpaired tests understate precision when the same tasks are scored for every model, and the paper reports no paired data. The untuned-base pairing is inferred.

## What this means for post-training

1. **Tune on procedure, retrieve the facts.** Steps, tone, tool use and refusal rules are what tuning teaches reliably. Prices, rates and clauses change and belong in retrieval (Part 3), where an update is an edit to a document.
2. **Verify every training conversation.** The pipeline's main cost is filtering for ground-truth actions, and the filtering is the reason the data helps.
3. **Include the reasoning for exceptions.** Labels alone left models rigid; explanations moved them toward human decisions. A service policy always has exceptions, so the training data should show why one was granted.
4. **Measure against the untuned base, on the firm's own tasks.** The base-versus-tuned gap is the only comparison that cleared the noise here. Beating a named frontier model needs hundreds of tasks scored on both sides.
5. **Plan to retune when the policy changes.** The data comes from one policy; a new policy is a new training set.

## What comes next

Part 3 covers architecture and deployment: where retrieval, tools and guardrails sit around the model, and what a policy check has to look like outside it.

## References

- Prabhakar et al., [APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay](https://arxiv.org/abs/2504.03601), 2025.
- Yao, Shinn, Razavi and Narasimhan, [τ-bench](https://arxiv.org/abs/2406.12045), ICLR 2025 (task counts).
- Gekhman, Yona, Aharoni, Eyal, Feder, Reichart and Herzig, [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904), 2024 (arXiv, revised October 2024).
- DiSorbo, Ju and Aral, [Teaching AI to Handle Exceptions: Supervised Fine-Tuning with Human-Aligned Judgment](https://arxiv.org/abs/2503.02976), 2025.
