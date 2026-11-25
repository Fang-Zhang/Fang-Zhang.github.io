---
title: "LLMs in Services, Part 1: A Pass Rate Is Not a Procedure"
date: 2027-06-12 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, services, evaluation, benchmarks, customer service, llms in services]
description: "Part 1 of 6 on large language models in service work. Scoring a banking assistant on its final records alone overstates the pass rate by 6.5 to 12 points, and 8 to 20 percent of apparent successes skipped a required step. A re-analysis of published numbers."
---

## What this post finds

Service work is conversation plus action: a customer asks, an assistant reads records, follows a written policy and changes something. A model for it is chosen from benchmark scores, so the question is what those scores measure. Three public sources agree on one answer: the usual score counts the outcome and misses the procedure.

- **The original benchmark measured the outcome, then repetition.** [τ-bench](https://arxiv.org/abs/2406.12045) (Yao, Shinn, Razavi and Narasimhan, ICLR 2025) simulates a customer with a language model, gives the agent domain tools and a policy, and compares the database at the end with the goal state. It added pass^k, the chance that an agent succeeds on all k trials of the same task. Its abstract reports that GPT-4o-class function-calling agents succeed on under 50 percent of tasks and that pass^8 is under 25 percent in retail.
- **A banking benchmark scores the steps too.** [TauIndianBankBench](https://arxiv.org/abs/2610.30161) (Adhikary and Devadiga, 2026) has 1,000 retail-banking tasks and checks that the assistant took every required step and made no unrequested change. For every model tested, 6.5 to 12.0 percent of sessions pass on the final records alone although a required step was skipped or something unrequested was changed.
- **A regulated-workflow benchmark finds the same, more strongly.** [cxbench](https://arxiv.org/abs/2610.25954) separates outcome from process compliance across six regulated domains. Over seven models and 120 tasks, 55.5 percent of the 840 retained executions are false-passes (correct outcome, failed process checks), which is 80.1 percent of the outcome-passing runs, and only 13.8 percent pass both. Its grader combines deterministic checks with a language-model judge, and its tasks are synthetic.
- **Re-analysis of the banking tables.** Of the sessions a final-state check calls successful, 8 to 20 percent broke procedure depending on the model, and the headline scores carry intervals about 5 points wide, so adjacent models are not always separable.
- **Repeat runs are more consistent than independence predicts.** Where the paper ran two attempts, the share of tasks passed both times is about 10 to 13 points above what independent attempts would give, which means unreliability sits in specific tasks more than in random noise.

![Two panels. Left: pass rates on 1,000 simulated banking tasks for six models scored two ways; scoring only the final records adds between 6.5 and 12.0 points over scoring procedure and outcome together, and the 95 percent intervals of the headline scores are about 5 points wide. Right: for five models, the share of tasks passed on both of two attempts (69, 61, 59, 46 and 37 percent) is higher than the squared single-attempt pass rate (58, 51, 50, 35 and 24 percent)]({{ site.baseurl }}/assets/pic/services-1-evaluation.png){: width="900" }
_Figure: left, Tables 1 and 2 of Adhikary and Devadiga with Wilson intervals computed here; right, Table 14, with the independence baseline computed here._

## What each benchmark can and cannot show

TauIndianBankBench is a synthetic bank: a language model drafted the tasks against the live environment, another model (run at temperature zero) plays the customer, and the six assistants are open-weight models (five Gemma 4 sizes and Qwen3.8-27B). It says little about a particular bank, language or frontier model, and the final-state check it compares against is the τ-bench convention, not a weak strawman made for the comparison. cxbench is a different benchmark with a different grader, so its 55.5 percent is not comparable to the 6.5 to 12.0 points; both show the same direction. τ-bench's numbers date from 2024 models.

## What the re-analysis shows

The script takes the pass rates the paper publishes (headline score, final-state-only score, the count of sessions that passed on records but failed the action check) and, for five models that were run twice, the shares of tasks that passed both, failed both or split.

```python
# Part 1 of the services series: what a benchmark score hides. Published numbers from TauIndianBankBench (Adhikary and Devadiga, 2026). Standard library only.
import math
# Table 1 (headline pass rate, base split, 1,000 tasks) and Table 2 (final-state-only pass rate; b = correct records but a skipped or unrequested step)
T = {"Qwen3.8-27B": (0.849, 0.920, 71), "Gemma 4 31B": (0.776, 0.896, 120), "Gemma 4 26B-A4B": (0.714, 0.821, 107),
     "Gemma 4 12B": (0.699, 0.764, 65), "Gemma 4 E4B": (0.607, 0.725, 118), "Gemma 4 E2B": (0.479, 0.598, 118)}
# Table 14: share of tasks (%) failing both, mixed, passing both, over two attempts (five Gemma models)
R = {"Gemma 4 31B": (16.6, 14.3, 69.1), "Gemma 4 26B-A4B": (18.6, 20.3, 61.1), "Gemma 4 12B": (18.3, 22.7, 59.1),
     "Gemma 4 E4B": (27.5, 26.0, 46.5), "Gemma 4 E2B": (39.9, 23.1, 36.9)}
def wilson(p, n, z=1.96):
    c = (p + z*z/(2*n)) / (1 + z*z/n); h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return c - h, c + h
print("=== Check 1: how much of a final-state 'success' broke procedure ===")
for m, (head, fs, b) in T.items():
    lo, hi = wilson(head, 1000)
    print(f"{m:16s} headline {head:.3f} (95% interval {lo:.3f}-{hi:.3f})  final-state only {fs:.3f}  inflation {100*(fs-head):4.1f} pts  share of final-state successes that skipped a step or changed something unrequested: {b/(fs*1000):5.1%}")
print()
print("=== Check 2: does a second attempt agree? (two attempts per task) ===")
def beta_fit(p, p2):
    # Beta(a, b) with mean p and E[x^2] = p2
    v = p2 - p*p; k = p*(1-p)/v - 1; return p*k, (1-p)*k
def rising(a, b, k):  # E[x^k] for Beta(a, b)
    r = 1.0
    for i in range(k): r *= (a + i) / (a + b + i)
    return r
print("model             pass^1  pass^2 observed  if attempts were independent (p^2)  fitted spread: pass^4  pass^8   independent p^8")
for m, (fb, mx, pb) in R.items():
    p = (pb + mx/2) / 100; p2 = pb / 100; a, b = beta_fit(p, p2)
    print(f"{m:16s} {p:7.3f} {p2:12.3f} {p*p:36.3f} {rising(a,b,4):24.3f} {rising(a,b,8):7.3f} {p**8:16.3f}")

```

Output:

```
=== Check 1: how much of a final-state 'success' broke procedure ===
Qwen3.8-27B      headline 0.849 (95% interval 0.825-0.870)  final-state only 0.920  inflation  7.1 pts  share of final-state successes that skipped a step or changed something unrequested:  7.7%
Gemma 4 31B      headline 0.776 (95% interval 0.749-0.801)  final-state only 0.896  inflation 12.0 pts  share of final-state successes that skipped a step or changed something unrequested: 13.4%
Gemma 4 26B-A4B  headline 0.714 (95% interval 0.685-0.741)  final-state only 0.821  inflation 10.7 pts  share of final-state successes that skipped a step or changed something unrequested: 13.0%
Gemma 4 12B      headline 0.699 (95% interval 0.670-0.727)  final-state only 0.764  inflation  6.5 pts  share of final-state successes that skipped a step or changed something unrequested:  8.5%
Gemma 4 E4B      headline 0.607 (95% interval 0.576-0.637)  final-state only 0.725  inflation 11.8 pts  share of final-state successes that skipped a step or changed something unrequested: 16.3%
Gemma 4 E2B      headline 0.479 (95% interval 0.448-0.510)  final-state only 0.598  inflation 11.9 pts  share of final-state successes that skipped a step or changed something unrequested: 19.7%

=== Check 2: does a second attempt agree? (two attempts per task) ===
model             pass^1  pass^2 observed  if attempts were independent (p^2)  fitted spread: pass^4  pass^8   independent p^8
Gemma 4 31B        0.762        0.691                                0.581                    0.623   0.561            0.114
Gemma 4 26B-A4B    0.713        0.611                                0.508                    0.514   0.428            0.066
Gemma 4 12B        0.705        0.591                                0.496                    0.481   0.385            0.061
Gemma 4 E4B        0.595        0.465                                0.354                    0.350   0.258            0.016
Gemma 4 E2B        0.485        0.369                                0.235                    0.276   0.205            0.003
```

- **The inflation is real and not tied to model size.** Final-state scoring adds 7.1 points for the best model and 11.9 for the smallest. The share of apparent successes that skipped a required step or made an unrequested change ranges from 7.7 percent (Qwen3.8-27B) to 19.7 percent (the 2B model).
- **A single headline number is about five points wide.** At 1,000 tasks the 95 percent Wilson interval is roughly plus or minus 2 to 3 points. The 26B mixture-of-experts model (0.714) and the 12B model (0.699) overlap on unpaired intervals; the paper's paired test covers only the dense models, so that pair is not settled by it.
- **Two attempts mostly agree.** For the 31B model, 69.1 percent of tasks pass both attempts, against 58 percent if attempts were independent with the same pass rate. The tasks split into dependable ones and stubborn ones, so repeating a run mostly re-measures the same failures.
- **The pass^8 figures are extrapolations.** A two-moment Beta fit of the per-task pass probability gives pass^8 of 0.56 for the 31B model against 0.11 under independence, but the tail depends on the assumed shape. The paper measured two attempts, and eight would have to be run to check.
- **Limits.** The data are one benchmark's tables, one customer simulator and one policy. Second-attempt counts exist for five of the six models.

## What this means for choosing a base model

1. **Score the procedure, not just the record.** List the required steps (identity check, disclosure, consent, a read before a write) and fail a session that skips one, even when the final state is right.
2. **Report the violation rate next to the pass rate.** The share of apparent successes that broke procedure is the number a compliance team will ask for.
3. **Size the test set for the gap being decided.** With 1,000 tasks, differences under about 5 points between unpaired scores should be treated as ties; paired comparisons on the same tasks are tighter.
4. **Run every task at least twice and look at the split.** Tasks that fail one run and pass the next are the ones where prompt, policy wording or a retry will matter; tasks that fail both need a different fix.
5. **Build the test from the firm's own policy and tickets.** The public benchmarks fix the lesson, but a bank, a clinic or a call centre has a different procedure.

## What comes next

Part 2 turns to post-training for service work: what supervised tuning on policy and procedure changes, and what the published results say about its limits.

## References

- Yao, Shinn, Razavi and Narasimhan, [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045), ICLR 2025.
- Adhikary and Devadiga, [TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents](https://arxiv.org/abs/2610.30161), 2026.
- [cxbench: A Benchmark for Process-Compliant LLM Agents in Regulated Customer-Experience Workflows](https://arxiv.org/abs/2610.25954), 2026.
