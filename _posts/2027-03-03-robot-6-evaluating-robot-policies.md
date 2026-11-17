---
title: "LLM and Humanoid Robots (6/6): Evaluating Robot Policies"
date: 2027-03-03 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, robotics, humanoid, evaluation, series]
description: "How many trials a robot policy needs before its success rate means anything, why repeated attempts erode reliability, and what LIBERO and SIMPLER offer as benchmarks."
---

The earlier parts described how a humanoid's software stack is built. This part covers the question that decides whether any of it can be trusted: how a robot policy is evaluated. It reviews two benchmark papers, then computes how many trials a success rate needs, how many a comparison needs, and what repeated attempts do to reliability. All numbers in the tables are simulated draws, not robot data.

## Two benchmark papers

[LIBERO](https://arxiv.org/abs/2306.03310) (Liu et al., 2023) is a benchmark for lifelong learning in robot manipulation. It has a procedural generation pipeline that can in principle generate unlimited tasks, four task suites with 130 tasks in total, and human-teleoperated demonstrations for all tasks. Its reported findings include that sequential finetuning outperforms existing lifelong learning methods in forward transfer, and that naive supervised pretraining can hinder performance on later tasks, which shows how a method that is standard elsewhere can fail on robot tasks.

[SIMPLER](https://arxiv.org/abs/2405.05941) (Li et al., 2024) addresses the cost of the alternative. The authors state that real-world evaluation of generalist manipulation policies is not scalable and faces reproducibility challenges. They identify control and visual disparities between real and simulated environments as the main obstacles to reliable simulated evaluation, propose ways to reduce them without building full-fidelity digital twins, and report strong correlation between policy performance in SIMPLER and in the real world in paired evaluations, including agreement on sensitivity to distribution shifts. Simulated evaluation is therefore a proxy whose validity is measured against the real robot, and the paper reports it for the setups it studied.

## How many trials

A robot trial is a Bernoulli outcome, success or failure, and success rates from few trials are imprecise. The script computes three things by simulation: the width of the 95% Wilson interval for a policy whose true success is 0.8; how often a worse policy (0.70) scores at least as high as a better one (0.80) when each is run $n$ times, counting ties as half; and the probability that $k$ consecutive attempts all succeed, $p^k$ if attempts are independent.

`evalsize.py`:

```python
import json, numpy as np
rng = np.random.default_rng(0); R = 40000
def wilson(s, n, z=1.96):
    p = s / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
NS = [10, 20, 50, 100, 200, 500, 1000]; res = {'n': NS, 'width': [], 'err': [], 'cover': []}
pa, pb = 0.80, 0.70
print(f'true success: policy A {pa}, policy B {pb}; {R} simulated evaluations per row')
print(f'{"trials n":>9}{"95% interval width (p=0.8)":>30}{"interval covers 0.8":>22}{"P(B scores >= A)":>20}')
for n in NS:
    s = rng.binomial(n, pa, R); lo, hi = wilson(s, n); w = (hi - lo).mean(); cov = ((lo <= pa) & (pa <= hi)).mean()
    sa, sb = rng.binomial(n, pa, R), rng.binomial(n, pb, R); err = (sb > sa).mean() + 0.5 * (sb == sa).mean()
    res['width'].append(float(w)); res['err'].append(float(err)); res['cover'].append(float(cov))
    print(f'{n:>9}{w:>30.3f}{cov:>22.3f}{err:>20.3f}')
print('\nreliability: probability that k independent attempts all succeed, p^k')
ks = [1, 5, 10, 50]; res['k'] = ks; res['pk'] = {}
print(f'{"per-attempt success":>20}' + ''.join(f'{"k=" + str(k):>10}' for k in ks))
for p in (0.80, 0.90, 0.95, 0.99):
    res['pk'][str(p)] = [p ** k for k in ks]; print(f'{p:>20.2f}' + ''.join(f'{p ** k:>10.3f}' for k in ks))
json.dump(res, open('res.json', 'w'))
```

Output:

```
true success: policy A 0.8, policy B 0.7; 40000 simulated evaluations per row
 trials n    95% interval width (p=0.8)   interval covers 0.8    P(B scores >= A)
       10                         0.433                 0.968               0.306
       20                         0.326                 0.956               0.231
       50                         0.215                 0.950               0.122
      100                         0.154                 0.940               0.049
      200                         0.110                 0.959               0.011
      500                         0.070                 0.948               0.000
     1000                         0.050                 0.947               0.000

reliability: probability that k independent attempts all succeed, p^k
 per-attempt success       k=1       k=5      k=10      k=50
                0.80     0.800     0.328     0.107     0.000
                0.90     0.900     0.590     0.349     0.005
                0.95     0.950     0.774     0.599     0.077
                0.99     0.990     0.951     0.904     0.605
```

![Left: width of the 95% interval against number of trials for a success rate of 0.8. Middle: probability that the worse of two policies ties or wins against the number of trials per policy. Right: probability that k attempts in a row all succeed for four per-attempt success rates](/assets/pic/robot-6-eval.png)
*Left: precision of one success rate. Middle: reliability of a comparison. Right: what repetition does to a success rate.*

## What the numbers say

- **Ten trials say little.** The interval for a true rate of 0.8 is 0.43 wide with 10 trials, 0.15 with 100 and 0.05 with 1,000. Reporting "8 of 10" without an interval conveys a range from roughly 0.5 to above 0.9.
- **A ten-point gap needs about 100 trials per policy to be reliable.** With 0.80 against 0.70, the worse policy ties or wins in 0.306 of the evaluations at 10 trials, 0.122 at 50 and 0.049 at 100. A few dozen trials per task is a common size for robot experiments (the author's observation, not a figure from the two papers), which puts many published rankings within the noise unless the gaps are large.
- **Repetition erodes reliability fast.** A policy that succeeds 90% of the time per attempt completes 10 attempts in a row with probability 0.349 and 50 with 0.005; 99% per attempt gives 0.904 and 0.605. A humanoid in a household or on a line makes thousands of attempts, and the demands of the application are set by the second table, not the first.
- **Independence is the optimistic case.** Failures that are correlated, for example by a lighting condition or an object that the policy mishandles every time, cluster instead of spreading out, and the per-attempt rate then understates the exposure to a bad condition.
- **Simulation extends the sample size.** A simulated benchmark gives hundreds of trials per task cheaply, which is the argument of SIMPLER; the cost is the question of whether the ranking carries over to hardware, which the paper tests by paired runs and which must be checked again for each new setup.

## Closing the series

The stack in the six parts has one structure: a slow model that understands (Part 1), a policy that outputs motion as tokens or continuous chunks (Parts 2 and 3), a fast loop beneath the slow one (Part 4), and learned legs trained in randomised simulation (Part 5). The evidence for each layer comes from papers and company reports with their own conditions, and the evaluation arithmetic above says how much of it a given number can support. In the author's view, the demonstrations visible in public are best read as existence proofs at a stated number of trials; reliability claims need the intervals and the repetition counts.

## Limits

- The tables are simulated from assumed success rates of 0.8, 0.7 and a few others. They describe statistics, not any robot, and the independence assumption in the reliability table is the optimistic case.
- The statements on LIBERO and SIMPLER are taken from their abstracts; no benchmark was run for this post.
- "A few dozen trials is common" is an impression of the author, not a measured survey.
- The series covers published and announced systems up to the papers cited; it did not test any hardware, and companies' own demonstrations were treated as claims.
