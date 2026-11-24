---
title: "LLMs on the Factory Floor, Part 1: A Leaderboard Gap Under Four Points Is Noise"
date: 2027-06-06 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, manufacturing, evaluation, benchmark, maintenance, llms on the factory floor]
description: "Part 1 of 6 on large language models in manufacturing. Public benchmarks for industrial maintenance show the top models within one point of each other, brittle under rephrasing, and tested on small samples. A simulation shows why a gap under four points should not decide the model choice."
---

## What this post finds

This series asks six questions about language models in manufacturing: which base model, how to adapt it, how to deploy it, how to maintain it, what it costs, and what else could do the job. Part 1 asks the first. The public evidence is thin on factories proper and rich on industrial maintenance, so this post reads three papers and says where each one stops.

- **The leading models are statistically tied.** [DiagnosticIQ](https://arxiv.org/abs/2605.08614) (6,690 multiple-choice questions from 118 expert rules on 16 asset types, 29 models) puts the top three within one point: 73.59, 73.27 and 72.74 percent macro accuracy (the figure below uses the paper's rule-weighted variant, which orders the top three slightly differently). A rule-level paired test between the top two gives p = 0.51 over 116 rules.
- **Thousands of questions are not thousands of observations.** The questions come in clusters of about 44 per rule, so the paper's effective sample is about 130 rules, and its intervals are 3.2 times wider than question-level ones.
- **Rephrasing breaks the leaders.** The best model loses 13.8 points when distractors are added. When the rule's comparison operators are inverted, it still picks the original, now wrong, answer 63 percent of the time.
- **An open-weight model sits inside the leaders' interval.** Qwen3.6-35B scores 69.90 percent against 73.59 for the top model, with overlapping intervals.
- **The factory-specific study is tiny.** A [study at a real factory](https://arxiv.org/abs/2401.05200) benchmarked models on 20 questions with one coder. GPT-4 scored 97.5 percent factuality, with StableBeluga2 at 95, Mixtral 8x7B at 92.5 and GPT-3.5 at 90.
- **Confidence needs its own test.** A [wind-turbine log benchmark](https://arxiv.org/abs/2509.06813) found many models overconfident, and concluded that a human stays in the loop.

![Two panels. Left: the published leaderboard of five models on an industrial maintenance benchmark with 95 percent intervals computed per rule; scores run from 70.2 to 75.8 and the intervals of the top four overlap almost entirely. Right: in a simulation of two equally good models, counting every question as independent declares a winner in 22 percent of benchmarks, against 4 percent when the rule is the unit]({{ site.baseurl }}/assets/pic/factory-1-model-selection.png){: width="900" }
_Figure: left, the published rule-level intervals from the DiagnosticIQ paper; right, a simulation with parameters quoted from that paper, run by the script at the end of this post._

## What the three papers measure

**DiagnosticIQ (De Silva, Patel, Constantinides and colleagues, May 2026).** Engineer-written rules say "when this sensor condition holds for this long, an alert fires". The benchmark asks which maintenance action fits. The rules come from a building-systems programme built on ASHRAE guidelines, so the assets are air handlers (58 percent of questions), chillers and boilers, not a production line. Nine practitioners scored a mean of 45.0 percent, and the paper reports that frontier models answered 10 of 12 questions that no practitioner in its panel got right, with the caveat of a small sample. The leader scores 73.6 percent, and the paper calls the benchmark unsaturated. It also reports that paraphrasing rules into prose costs 4.2 points on average, and that replacing shortcut words shared between the question and one option costs the leaders about 2 points, so lexical shortcuts explain little of the leaders' accuracy. The authors read the pattern as models treating structured rules as templates to recognise rather than conditions to evaluate.

**The factory knowledge study (Freire, Wang, Foosherian and colleagues, 2024).** A system retrieves from factory documentation and operator knowledge. Seven models answered 20 questions, scored by one coder on factuality, completeness and hallucination. GPT-4 had 97.5 percent factuality and 0 percent hallucination; StableBeluga2 95 and 7.5; Mixtral 8x7B 92.5 and 2.5; GPT-3.5 90 and 5; Llama 2 77.5 and 13; the two Guanaco models were far lower. In the user study, operators saw benefits such as quicker retrieval but preferred a human expert when one was available. The authors state that the 20 questions and the single coder introduce potential bias. These are 2023 and early 2024 models, and the paper's discussion of open-source models trailing closely is about that generation.

**Wind-turbine maintenance logs (Malyi, Shek, McDonald and Biscaya, 2025).** The task is classifying free-text work orders by issue category and by maintenance type. All models agreed more on the objective category (which component) than on the interpretive type (repair, replacement or inspection), and the authors cite work finding that human experts disagree on such labels too. Many models were overconfident, with phi4 14B named as an example and o3 as well calibrated. The authors report processing costs of a few dollars per few hundred records, call them negligible for a wind-farm operator, and recommend a human-in-the-loop workflow.

## Why a small gap should not decide the choice

The DiagnosticIQ paper's own statistics make the point, and the script below reproduces it. Check 1 recomputes the design effect from the paper's quoted figures. Check 2 simulates two models of equal true accuracy on a benchmark shaped like DiagnosticIQ (118 rules, 44 questions each, accuracy varying widely between rules, as the paper reports), and counts how often a test declares a winner. Check 3 asks how stable a ranking is on 20 questions. The simulation's rule-difficulty spread, the model-to-model deviation and the simple right-or-wrong scoring are assumptions, so it shows the mechanism and does not reproduce the paper's data.

```python
# Two checks on how much a benchmark ranking can be trusted. Simulations use parameters quoted from the DiagnosticIQ paper; everything else is assumed.
import numpy as np
rng = np.random.default_rng(7)

print("=== Check 1: design effect from the paper's numbers ===")
n_q, n_rules, icc = 5180, 118, 0.896
m = n_q / n_rules
de = 1 + (m - 1) * icc
print(f"questions per rule m = {m:.1f}; design effect = 1 + (m-1)*ICC = {de:.1f}; effective sample = {n_q/de:.0f} (paper: about 40 and about 130)")

print()
print("=== Check 2: two models with the SAME true accuracy, 118 rules x 44 questions, 4000 simulated benchmarks ===")
def sim(n_rules, m, true_gap=0.0, sims=4000, mean=0.75, sd=0.21, dev=0.08):
    k = mean * (1 - mean) / sd**2 - 1
    a, b = mean * k, (1 - mean) * k
    hit_q = hit_r = 0
    for _ in range(sims):
        d = rng.beta(a, b, n_rules)
        pa = np.clip(d + rng.normal(0, dev, n_rules) + true_gap / 2, 0, 1)
        pb = np.clip(d + rng.normal(0, dev, n_rules) - true_gap / 2, 0, 1)
        ya = rng.random((n_rules, m)) < pa[:, None]; yb = rng.random((n_rules, m)) < pb[:, None]
        diff = ya.astype(float) - yb.astype(float)                  # one paired difference per question
        n = diff.size
        t_q = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))         # every question treated as independent
        rd = diff.mean(axis=1)                                      # one difference per rule
        t_r = rd.mean() / (rd.std(ddof=1) / np.sqrt(n_rules))       # the rule is the unit
        hit_q += abs(t_q) > 1.96; hit_r += abs(t_r) > 1.98          # about p < 0.05 (t critical value at 117 degrees of freedom is 1.98)
    return hit_q / sims, hit_r / sims
q, r = sim(118, 44, 0.0)
print(f"no true difference: question-level test calls a winner (p<0.05) in {q:.1%} of benchmarks; rule-level test in {r:.1%}")
for gap in (0.01, 0.03):
    q, r = sim(118, 44, gap, sims=2000)
    print(f"true gap {gap*100:.0f} points: question-level test detects it in {q:.1%} of benchmarks, rule-level test in {r:.1%}")

print()
print("=== Check 3: ranking stability with a 20-question test (the factory study used 20 questions) ===")
for pa, pb in ((0.95, 0.90), (0.95, 0.80)):
    A = rng.binomial(20, pa, 200000); B = rng.binomial(20, pb, 200000)
    print(f"true accuracy {pa:.2f} vs {pb:.2f}: weaker model scores higher in {np.mean(B > A):.1%} of tests, ties in {np.mean(B == A):.1%}")
```

Output:

```
=== Check 1: design effect from the paper's numbers ===
questions per rule m = 43.9; design effect = 1 + (m-1)*ICC = 39.4; effective sample = 131 (paper: about 40 and about 130)

=== Check 2: two models with the SAME true accuracy, 118 rules x 44 questions, 4000 simulated benchmarks ===
no true difference: question-level test calls a winner (p<0.05) in 22.5% of benchmarks; rule-level test in 4.4%
true gap 1 points: question-level test detects it in 34.1% of benchmarks, rule-level test in 11.4%
true gap 3 points: question-level test detects it in 84.5% of benchmarks, rule-level test in 60.4%

=== Check 3: ranking stability with a 20-question test (the factory study used 20 questions) ===
true accuracy 0.95 vs 0.90: weaker model scores higher in 17.5% of tests, ties in 21.1%
true accuracy 0.95 vs 0.80: weaker model scores higher in 3.6% of tests, ties in 6.8%
```

What the run shows, and what it does not:

- **Counting questions as independent finds winners that are not there.** With no true difference, the question-level test declared a winner in 22.5 percent of simulated benchmarks, against 4.4 percent when the rule was the unit, close to the nominal 5 percent. The paper's own comparison agrees in direction: p = 0.15 at question level and 0.51 at rule level for the top two models.
- **Even a correct test needs a gap of a few points.** At a true 1-point gap the rule-level test detected the difference in 11 percent of benchmarks, and at 3 points in 60 percent. A benchmark of this size cannot rank models that are one or two points apart.
- **Twenty questions rank only large gaps.** A model with 95 percent true accuracy scored below one with 90 percent in 17.5 percent of simulated 20-question tests, and tied in 21 percent. The factory study's ordering among its top four models, 2.5 to 7.5 points apart, sits inside that range. Its separation from the Guanaco models does not.
- **Limits.** The simulation treats each answer as right or wrong and each model as independent given the rule, while the factory study scored half-credit answers. The simulated pair also has no shared model family or shared mistakes, which in practice makes models more alike.

## What this means for choosing a model

1. **Treat a gap under about four points as a tie** on a benchmark of this kind, and choose between tied models on other grounds: data location, latency, price, licence, and ability to be fine-tuned. The open-weight model inside the interval is a real candidate, and the factory study names data privacy and customisation as the reasons open models matter.
2. **Test the task, not the topic.** General leaderboards say little about whether a model can map a sensor rule to a repair step; DiagnosticIQ's own comparison table lists the general benchmarks MMLU, GPQA and TruthfulQA as lacking industrial-asset specificity, plausible distractors, condition understanding and action recommendation, all four.
3. **Add perturbation tests.** Rephrase the same question, invert a condition, add distractors, and report the share of answers that stay consistent. A plant gains more from knowing how often the answer changes under rewording than from a single accuracy figure; the paper's 13.8-point drop for the leader under distractors is that kind of number.
4. **Measure calibration.** A model whose stated confidence tracks its accuracy can route doubtful cases to a person; an overconfident one cannot.
5. **Build a plant-specific set with the rule as the unit.** Write at least a few hundred independent cases from the plant's own rules, work orders and procedures, and report intervals that respect the clustering.
6. **State what the public evidence does not cover.** None of the three papers evaluates a model on a production line in operation, in a language other than English, or with an agent that takes actions. The DiagnosticIQ paper lists English, zero-shot and non-agentic settings as its limits.

## What comes next

Part 2 covers post-training: what the public cases say about retrieval versus fine-tuning on maintenance records, shop-floor jargon, and the data problems of work orders.

## References

- De Silva, Patel, Constantinides et al., [DiagnosticIQ: A Benchmark for LLM-Based Industrial Maintenance Action Recommendation from Symbolic Rules](https://arxiv.org/abs/2605.08614), 2026.
- Freire, Wang, Foosherian, Wellsandt, Ruiz-Arenas and Niforatos, [Knowledge Sharing in Manufacturing using Large Language Models: User Evaluation and Model Benchmarking](https://arxiv.org/abs/2401.05200), 2024.
- Malyi, Shek, McDonald and Biscaya, [A Comparative Benchmark of Large Language Models for Labelling Wind Turbine Maintenance Logs](https://arxiv.org/abs/2509.06813), 2025.
