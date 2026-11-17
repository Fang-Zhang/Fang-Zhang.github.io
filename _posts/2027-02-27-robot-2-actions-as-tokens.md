---
title: "LLM and Humanoid Robots (2/6): Actions as Tokens"
date: 2027-02-27 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, robotics, humanoid, vla, series]
description: "How RT-1, RT-2 and OpenVLA turn robot motion into language-model tokens, what the binning costs in precision, and what token-by-token decoding does to the control rate."
---

[Part 1](/posts/robot-1-language-as-planner-grounded-by-skills/) kept the language model at the planning level and left the motion to separate skills. This part covers the opposite design: the model itself outputs the motion, one token at a time. It describes how three papers encode actions as tokens, measures what the binning costs on synthetic actions, and measures what token-by-token decoding does to the control rate on a small local model.

## The encoding

[RT-1](https://arxiv.org/abs/2212.06817) (Brohan et al., 2022) discretizes each action dimension into 256 bins uniformly distributed within the bounds of that variable. The action has seven arm variables (x, y, z, roll, pitch, yaw, gripper opening), three base variables (x, y, yaw) and one discrete variable that switches between arm, base and terminating the episode. The model is a decoder-style Transformer with 8 self-attention layers and 19M parameters that outputs these action tokens, and the paper reports closed-loop control at 3 Hz with an inference budget of under 100 ms per step.

[RT-2](https://arxiv.org/abs/2307.15818) (Brohan et al., 2023) reuses that discretization for a large vision-language model. The action space is a 6-DoF pose change of the end effector, the gripper extension and a termination command; the continuous dimensions go into 256 uniform bins, so an action is 8 integers. Fine-tuning needs tokens for those integers, so the method reserves 256 tokens of the model's existing vocabulary as action tokens, and the actions are then treated like language tokens in the output. The paper reports that the 55B-parameter model runs at 1 to 3 Hz and a 5B-parameter version at about 5 Hz, served from a multi-TPU cloud service queried over the network.

[OpenVLA](https://arxiv.org/abs/2406.09246) (Kim et al., 2024) changes one detail of the binning. Each dimension still gets 256 bins, but the bin width divides the interval between the 1st and 99th quantile of the training actions, which the authors say ignores outlier actions that would otherwise stretch the interval and coarsen the resolution. The 256 tokens are the 256 least-used tokens of the Llama tokenizer, overwritten with action tokens, and training is ordinary next-token prediction with the loss on the action tokens only. The paper reports about 6 Hz on one RTX 4090 and 15 GB of GPU memory in bfloat16, after training on 64 A100 GPUs for 14 days.

## What binning costs

The sketch below uses synthetic end-effector displacements: mostly small moves, 5% larger moves, and 0.2% rare large moves up to half a metre. The proportions are the author's assumptions, chosen to make the effect visible; they do not come from robot data. Both schemes encode a value to a bin and decode it to the bin centre.

`quant.py`:

```python
import numpy as np
rng = np.random.default_rng(0)
N = 200_000
# synthetic end-effector displacement per step (metres): mostly small moves, some large moves, rare outliers
a = rng.normal(0, 0.005, N)
big = rng.random(N) < 0.05; a[big] = rng.normal(0, 0.03, big.sum())
out = rng.random(N) < 0.002; a[out] = rng.choice([-1, 1], out.sum()) * rng.uniform(0.2, 0.5, out.sum())
def encode_decode(x, lo, hi, bins):
    idx = np.clip(np.floor((x - lo) / (hi - lo) * bins), 0, bins - 1).astype(int)   # token id in 0..bins-1
    return lo + (idx + 0.5) * (hi - lo) / bins                                         # bin centre
schemes = {'uniform over [min, max] (RT-1 style)': (a.min(), a.max()),
           'uniform over 1st-99th percentile (OpenVLA style)': tuple(np.quantile(a, [0.01, 0.99]))}
print(f'synthetic actions: n={N}, median |a| = {np.median(np.abs(a))*1000:.1f} mm, 99th percentile |a| = {np.quantile(np.abs(a), 0.99)*1000:.1f} mm, max |a| = {np.abs(a).max()*1000:.0f} mm')
print('\nbins  scheme                                              mean abs error (mm)   on outliers (mm)   drift after 50 open-loop steps (mm)')
res = {}
for bins in (16, 32, 64, 128, 256, 512):
    for name, (lo, hi) in schemes.items():
        d = encode_decode(a, lo, hi, bins); e = np.abs(d - a)
        # drift: sum the decoded actions of 50 consecutive steps vs the true sum, over many 50-step windows
        w = (d - a)[:N // 50 * 50].reshape(-1, 50).sum(axis=1); drift = np.sqrt(np.mean(w ** 2))
        res[(bins, name)] = (e.mean() * 1000, e[out].mean() * 1000, drift * 1000)
        print(f'{bins:>4}  {name:<50}  {e.mean()*1000:>10.3f}          {e[out].mean()*1000:>10.1f}         {drift*1000:>10.3f}')
import json; json.dump({f'{b}|{n}': v for (b, n), v in res.items()}, open('res.json', 'w'))
```

Output:

```
synthetic actions: n=200000, median |a| = 3.6 mm, 99th percentile |a| = 41.8 mm, max |a| = 500 mm

bins  scheme                                              mean abs error (mm)   on outliers (mm)   drift after 50 open-loop steps (mm)
  16  uniform over [min, max] (RT-1 style)                    26.138                15.8            437.373
  16  uniform over 1st-99th percentile (OpenVLA style)         1.792               324.8            105.260
  32  uniform over [min, max] (RT-1 style)                    11.007                 8.4            166.323
  32  uniform over 1st-99th percentile (OpenVLA style)         1.359               324.0            104.622
  64  uniform over [min, max] (RT-1 style)                     4.128                 4.0             40.373
  64  uniform over 1st-99th percentile (OpenVLA style)         1.142               323.6            104.426
 128  uniform over [min, max] (RT-1 style)                     1.947                 2.0             15.859
 128  uniform over 1st-99th percentile (OpenVLA style)         1.034               323.4            104.289
 256  uniform over [min, max] (RT-1 style)                     0.972                 1.0              7.908
 256  uniform over 1st-99th percentile (OpenVLA style)         0.980               323.3            104.244
 512  uniform over [min, max] (RT-1 style)                     0.486                 0.5              3.977
 512  uniform over 1st-99th percentile (OpenVLA style)         0.953               323.2            104.211
```

Three things follow from the table.

- **Resolution scales with the range per bin.** With bounds at the minimum and maximum, the error halves each time the bin count doubles (26 mm at 16 bins, 0.97 mm at 256), because the range is set by the rare 0.5 m moves. Percentile bounds give 1.03 mm at 128 bins and barely improve beyond, because the typical action already fits.
- **Percentile bins buy resolution and sell the tail.** At 256 bins the two schemes have about the same mean error on typical actions (0.97 and 0.98 mm), but the percentile scheme clips every rare large move to the edge of its range, an error of about 323 mm on those actions, against 1.0 mm for the full-range scheme. Which is better depends on whether the rare large moves matter for the task.
- **Errors add up without feedback.** Over 50 open-loop steps the full-range scheme drifts by 7.9 mm at 256 bins, and the percentile scheme by 104 mm, almost all from clipped moves. A closed-loop policy sees the result of each step and corrects; the drift figures describe the worst case of not looking.

## What token-by-token decoding costs

Eight tokens per action means eight sequential forward passes per control step, before any image processing. The measurement divides the decode speed of two small models on this machine by 8.

`rate.py` output:

```
decode-only control rate = tokens per second / 8 tokens per action (image encoding and network delay not included)
Qwen2.5-0.5B:  45.7 tokens/s on 8 CPU threads  ->   5.7 actions per second
Qwen2.5-1.5B:  18.9 tokens/s on 8 CPU threads  ->   2.4 actions per second
```

![Left: mean absolute quantisation error against bins per dimension for full-range and percentile bounds, on typical and rare large actions, synthetic data. Right: actions per second for RT-1, RT-2 and OpenVLA as reported, and for two small local models measured here](/assets/pic/robot-2-tokens.png)
*Left: binning error on synthetic actions. Right: reported control rates of three robot models (grey) and decode-only rates of two local models on 8 CPU threads (blue); the platforms are not comparable.*

The rate of a 0.5B model on CPU is of the same order as the reported rates of far larger models on accelerators, which shows the point: the rate is set by parameters times tokens per action, and every extra dimension, and every doubling of the model costs rate. At 1 to 6 Hz, a model decoding tokens cannot close a fast control loop. The papers' remedy is to leave fast control to other components or to produce several actions per forward pass, which are the topics of Parts 3 and 4.

## Limits

- The quantisation experiment uses synthetic actions with assumed proportions. It shows how the two binning schemes trade resolution against the tail; it does not show how either affects task success on a robot.
- The local rates are decode speeds only, with no image encoding, no network and no accelerator, on models that are not robot policies. The reported rates come from the papers' own hardware and systems, so the bars on the right are not a ranking.
- RT-2 and OpenVLA numbers (1 to 3 Hz, about 5 Hz, about 6 Hz, 15 GB, 64 A100 GPUs for 14 days) are quoted from the papers and were not reproduced.
- The 8 tokens per action follow RT-2; OpenVLA's tokens per action differ with its action dimensions.

Part 3 covers continuous actions, where the bins disappear.
