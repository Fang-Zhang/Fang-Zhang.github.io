---
title: "Improving LLM Performance (5/8): Spending Compute at Inference Time"
date: 2027-03-08 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, inference, chain-of-thought, self-consistency, best-of-n, series]
description: "Chain of thought, repeated sampling, majority voting and verifier-based selection: what each one needs in order to work, with toy runs that show a scratchpad turning a failing task into a solvable one and show majority voting and noisy verifiers hitting their limits."
---

[Part 4](/posts/improve-4-rl-verifiable-rewards/) trained the model to be right more often at one sample. This part leaves the weights alone and changes what happens at the moment of use: write intermediate steps, draw several answers, pick among them. The sections below give the published results, then two toy experiments: a scratchpad on 3-digit multiplication, and majority voting and checker-based selection under two different error structures.

## What the papers establish

- **Chain of thought.** [Wei et al. (2022)](https://arxiv.org/abs/2201.11903) show that generating a series of intermediate reasoning steps improves the reasoning ability of sufficiently large language models, using a few chain-of-thought demonstrations as exemplars. Prompting a 540B-parameter model with eight such exemplars gave state-of-the-art accuracy on GSM8K, ahead of fine-tuned GPT-3 with a verifier.
- **Self-consistency.** [Wang et al. (2022)](https://arxiv.org/abs/2203.11171) replace greedy decoding with sampling a diverse set of reasoning paths and selecting the most consistent answer by marginalising out the paths. The premise is that a complex problem admits several ways of thinking that lead to the same correct answer. They report gains on GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%).
- **Repeated sampling.** [Brown et al. (2024)](https://arxiv.org/abs/2407.21787) measure coverage, the fraction of problems solved by any of the samples, and find it scales with the number of samples over four orders of magnitude, often log-linearly. Where answers can be checked automatically, such as code and formal proofs, coverage turns directly into performance: on SWE-bench Lite, DeepSeek-Coder-V2-Instruct goes from 15.9% with one sample to 56% with 250. Where there is no automatic checker, majority voting and reward models plateau beyond several hundred samples.
- **Allocating the compute.** [Snell et al. (2024)](https://arxiv.org/abs/2408.03314) study searching against process-based verifier reward models and adaptively revising the answer. They find the effectiveness of each depends on prompt difficulty, and that a compute-optimal strategy, allocating test-time compute per prompt, improves efficiency by more than 4x over a best-of-N baseline.

## Toy 1: steps as extra compute

Task: 3-digit multiplication, `123*456=`, with the same 3-layer, 96-wide Transformer as before, trained on fresh random problems at every step. The direct format writes the 6-digit product. The scratchpad format writes three running sums, each adding the first factor times one digit of the second, and the last number is the answer:

```
direct:      123*456=056088
scratchpad:  123*456=000738,006888,056088
```

The code for the format (the rest is the Part 3 Transformer and a standard loss on the answer tokens, 6,000 steps of 128 problems, 2 seeds):

```python
def make(a, b, cot):
    p = f'{a:03d}*{b:03d}='
    if not cot: return p, digits(a * b, 6) + '\n'
    s = 0; parts = []
    for place in (1, 10, 100): s += a * (b // place % 10) * place; parts.append(digits(s, 6))
    return p, ','.join(parts) + '\n'
```

Exact-match accuracy on 1,000 held-out problems, greedy decoding:

```
direct     seed 1: answer tokens 7   accuracy at steps 0:0% 2000:0% 4000:2% 6000:6%  (133s)
direct     seed 2: answer tokens 7   accuracy at steps 0:0% 2000:0% 4000:1% 6000:4%  (133s)
scratchpad seed 1: answer tokens 21  accuracy at steps 0:0% 2000:51% 4000:76% 6000:88%  (281s)
scratchpad seed 2: answer tokens 21  accuracy at steps 0:0% 2000:54% 4000:80% 6000:90%  (283s)
```

Within the same training steps, the direct model is at 4% to 6% and the scratchpad model at 88% to 90%. Each step of the scratchpad is a smaller computation than the whole product, and the intermediate numbers are written into the context where later tokens can read them. This matches the mechanism the chain-of-thought paper describes in prose, though the toy shows it only on a task built to need it: in a separate 4-digit addition run (not in the figure), the direct format reached 100% on both seeds by step 4,000, so the scratchpad helps only where a single pass is too shallow.

## Toy 2: voting and checkers

Two base models are trained on an addition corpus that is 30% correct. In one the wrong answers are the true sum plus or minus 1 to 10 (spread over 20 values). In the other every wrong answer is the true sum plus 1. The samples are drawn at temperature 1, 64 per prompt over 500 prompts, and each number averages 20 random subsets of the 64. Three selectors are compared:

- **Majority vote**, ties going to the first sampled answer.
- **Best-of-N with a perfect checker**: correct if any of the N samples is correct, which is the coverage of Brown et al. (pass@N).
- **Best-of-N with a noisy checker**: a checker that accepts 80% of correct answers and 30% of wrong ones, with the first accepted sample chosen (the first sample if none is accepted). The rates are arbitrary and set by the author, not measured on any real verifier.

```python
import json, collections
exec(open('common.py').read())
N_MAX, NP = 64, 500
def answers(m, ps, seed=11):
    g = torch.Generator().manual_seed(seed); P = prompt_tensor(ps).repeat_interleave(N_MAX, 0); x = sample(m, P, g); out = []
    for r in x.numpy(): t = ''.join(VOC[i] for i in r); out.append(t[6:9])
    return [out[i * N_MAX:(i + 1) * N_MAX] for i in range(len(ps))]
ps = eval_prompts[:NP]; truth = [f'{a + b:03d}' for a, b in ps]; A = answers(base, ps); rng = np.random.default_rng(0)
print(f'base model: single-sample accuracy {100 * np.mean([x == t for AA, t in zip(A, truth) for x in AA]):.1f}%; most common wrong answer makes up {100 * np.mean([max([c for k, c in collections.Counter(x for x in AA if x != t).items()] or [0]) / N_MAX for AA, t in zip(A, truth)]):.1f}% of the 64 samples on average', flush=True)
res = {}
for N in (1, 2, 4, 8, 16, 32, 64):
    vote = oracle = noisy = 0; trials = 0
    for AA, t in zip(A, truth):
        for _ in range(20):
            S = [AA[i] for i in rng.permutation(N_MAX)[:N]]; cnt = collections.Counter(S); top = max(cnt.values()); modes = [k for k in S if cnt[k] == top]          # ties: first sampled
            vote += modes[0] == t; oracle += t in S
            flag = [(rng.random() < (0.8 if s == t else 0.3)) for s in S]; pick = next((s for s, f in zip(S, flag) if f), S[0]); noisy += pick == t; trials += 1
    res[N] = (vote / trials, oracle / trials, noisy / trials); print(f'N={N:<3} majority vote {100 * vote / trials:5.1f}%   best-of-N with a perfect checker {100 * oracle / trials:5.1f}%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers) {100 * noisy / trials:5.1f}%', flush=True)
json.dump(res, open(f'res_{NAME}.json', 'w'))
```

Output for the spread-out wrong answers, then the identical wrong answers:

```
base model: single-sample accuracy 21.4%; most common wrong answer makes up 13.1% of the 64 samples on average
N=1   majority vote  21.0%   best-of-N with a perfect checker  21.0%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  21.0%
N=2   majority vote  21.7%   best-of-N with a perfect checker  37.4%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  29.3%
N=4   majority vote  28.6%   best-of-N with a perfect checker  61.2%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  38.0%
N=8   majority vote  42.3%   best-of-N with a perfect checker  85.4%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  41.1%
N=16  majority vote  56.7%   best-of-N with a perfect checker  96.8%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  41.5%
N=32  majority vote  73.1%   best-of-N with a perfect checker  99.8%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  41.5%
N=64  majority vote  83.8%   best-of-N with a perfect checker 100.0%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  42.3%
base model: single-sample accuracy 29.2%; most common wrong answer makes up 67.6% of the 64 samples on average
N=1   majority vote  28.7%   best-of-N with a perfect checker  28.7%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  28.7%
N=2   majority vote  29.0%   best-of-N with a perfect checker  48.7%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  38.4%
N=4   majority vote  22.1%   best-of-N with a perfect checker  72.6%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  47.4%
N=8   majority vote  14.0%   best-of-N with a perfect checker  91.0%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  51.0%
N=16  majority vote   7.5%   best-of-N with a perfect checker  97.7%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  51.5%
N=32  majority vote   3.4%   best-of-N with a perfect checker  99.4%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  51.9%
N=64  majority vote   1.4%   best-of-N with a perfect checker  99.8%   best-of-N with a noisy checker (accepts 80% of correct, 30% of wrong answers)  50.9%
```

![Left: accuracy against training step for direct and scratchpad multiplication. Middle and right: accuracy against the number of samples for majority vote, best-of-N with a perfect checker and with a noisy checker, when wrong answers are spread out and when they are all the same](/assets/pic/improve-5-inference.png)
*Left: the scratchpad model reaches about 89% where the direct model is at 5%. Middle: with spread-out errors, voting climbs to 84% at 64 samples. Right: with one repeated wrong answer, voting falls to 1.4% while a perfect checker reaches 99.8%.*

What the numbers say:

- **Voting needs the right answer to be the most common one.** With wrong answers spread out, the correct answer is the mode and voting goes from 21% to 84% as N grows from 1 to 64. With one repeated wrong answer, the mode is wrong (the most common wrong answer makes up 68% of samples), and more samples make voting worse: 29% at N=1, 1.4% at N=64. This is the premise of self-consistency, that different reasoning paths converge on the correct answer, stated as a failure condition.
- **A perfect checker turns coverage into accuracy.** Both models reach 99.8% to 100% at N=64 with exact checking, matching the shape of the coverage curves in Brown et al., though with a tiny answer space that guarantees it.
- **A noisy checker caps the gain whatever N is.** The picked answer is correct with probability $\frac{p\cdot 0.8}{p\cdot 0.8+(1-p)\cdot 0.3}$ once N is large enough to contain an accepted sample, where $p$ is the single-sample accuracy. With $p=0.214$ and $0.292$ this is 42% and 52%; the runs show 42.3% and 50.9% at N=64. More samples never fix a checker's false-accept rate, which is the same plateau Brown et al. report for reward models without automatic verification.

Part 6 turns from what the model does at inference to what the model is: mixture-of-experts, attention variants, long context and decoding speed.

## Limits

- Both toys use tiny answer spaces. The scratchpad task was chosen because a single pass fails on it; the addition run in the text shows the contrast disappears when the task is easy.
- The scratchpad comparison fixes training steps, not compute: the scratchpad format has three times as many answer tokens per example, so it also costs more per step and per answer.
- The error structures in Toy 2 are constructed. How real models distribute wrong answers on real tasks is an empirical question that the cited papers address and this toy does not.
- The checker is a fixed-rate random process, not a learned verifier, and it has no notion of being gamed by the policy.
- Cited gains (GSM8K, SWE-bench Lite, the 4x efficiency) are quoted from the papers and were not reproduced here.
