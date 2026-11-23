---
title: "From Zero to ML Researcher, Part 9: Hypotheses and Ablations"
date: 2027-05-19 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, ablation, statistics, experiments, transformer, ML researcher]
description: "Part 9 of a 12-part series for complete beginners. How to test whether a part of a model matters: state the hypothesis and the decision rule, run paired seeds, add an A/A control, report an interval. Removing the MLP costs 0.42 bits per character, removing positions 0.75, using 1 head instead of 4 only 0.05."
---

## What you will do, and what you get

You will test three claims about the transformer of Part 8 the way a researcher does: *ablate* one part at a time, train 5 seeds of every variant, compare them seed by seed, and add a control that checks the method itself. Results on held-out text (test bits per character, lower is better; the baseline is 2.019 with a seed-to-seed std of 0.010):

- removing the MLP: **+0.422**, 95% interval [0.393, 0.452];
- removing position embeddings: **+0.750**, [0.670, 0.831];
- using 1 head instead of 4: **+0.049**, [0.032, 0.066] (small, but real);
- *A/A control*, the same model with different seeds: **-0.005** (no difference, as it should be).

![Left: paired differences in test bits per character against the baseline for four comparisons, with one circle per seed and a 95 percent interval: A/A control about minus 0.005, 1 head plus 0.049, no MLP plus 0.422, no positions plus 0.750. Middle: test bits per character for each of 5 seeds in four variants, with the baseline mean plus and minus two standard deviations as a green band at about 2.0. Right: a table of three hypotheses and an A/A control, each with its verdict and interval]({{ site.baseurl }}/assets/pic/mlr-9-ablations.png){: width="900" }

## How: the steps

1. **Write the hypothesis and the decision rule before running.** For example: "removing the MLP raises the test loss. The effect counts as real if all seeds agree in sign, the 95% interval of the paired difference excludes zero, and the mean difference is larger than twice the baseline seed std."
2. **Change one thing at a time.** Each variant differs from the baseline in a single component.
3. **Use several seeds,** and let the same seed give every variant the same batches in the same order, so that each seed forms a *pair*.
4. **Add an A/A control:** train the unchanged model with new seeds and compare the two groups. If the method reports a difference here, the method is broken.
5. **Report the effect with an interval,** not a verdict word. The paired difference per seed, its mean, and a 95% interval.
6. **Look at the sizes,** not only at the sign: a difference of 0.05 and one of 0.75 both pass the rule, and they are not the same finding.

## Why it works: the principle

**Why one run proves little.** Training has randomness: the initial weights and the order of the batches. Part 8 measured it: the three seeds finished within 0.008 of each other on test. Any comparison of two single runs differing by less than that says nothing, and one that differs by more can still be luck. The tool is to repeat the run and look at the spread. The same concern for machine learning research in general is discussed in [Pineau and colleagues, Improving Reproducibility in Machine Learning Research](https://arxiv.org/abs/2003.12206); Part 12 returns to it.

**Pairing removes noise you can control, but it does not always help.** With the same seed, two variants see the same batches, so part of the batch-to-batch noise cancels in the difference. Here the gain was small: the paired differences of the 1-head variant have a std of 0.014, about what two independent groups with seed stds of 0.007 and 0.010 would give (`sqrt(0.007^2 + 0.010^2) = 0.012`). The initial weights also differ between variants, and pairing does not cancel that part. The effects are large compared with that noise, so the conclusions hold either way; measure the benefit instead of assuming it.

**The interval.** With 5 differences, the 95% interval for their mean is `mean +/- t * se`, where `se = std / sqrt(5)` and `t = 2.776` is the critical value of [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) with 4 degrees of freedom (in code, `scipy.stats.t.ppf(0.975, 4)`; see the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)). The factor 2.776 is larger than the familiar 1.96 because the std of only 5 numbers is itself uncertain. A few seeds with large effects can be convincing; a small effect needs more seeds, and the worked example below shows the arithmetic.

**The A/A control checks the machinery.** If two groups of the *same* model had differed by 0.05, then the 1-head result would be indistinguishable from noise, whatever the interval said. The measured A/A difference is -0.005, a tenth of the 1-head effect, which makes that effect credible.

**Reading the results.**
- *Positions* matter most: without them the model must guess an order that attention alone cannot see. The loss is not as bad as bag-of-words, because with a causal mask a model can still recover some position information from how many characters it can attend to ([Haviv and colleagues, 2022](https://arxiv.org/abs/2203.16634)).
- *The MLP* matters a lot too: without it each block can only mix values between positions and never transforms them.
- *Heads:* going from 4 heads to 1 head (the same parameter count, since the width is split among the heads) costs about 0.05. Real, small, and a reminder that significant does not mean large.
- All three agree in sign on 5 of 5 seeds, and all three intervals exclude 0, so the decision rule marks them as supported.

**Pitfalls and limits.**
- **The MLP ablation changes more than one thing:** removing the MLP also cuts the parameters from 110,384 to 44,208. The 0.42 is the effect of removing the MLP *and* of having fewer parameters. Separating the two needs a no-MLP model made wider to match the parameter count; try it below.
- **Short training:** each run is 1,500 steps (Part 8 used 3,000), so the baseline here scores 2.019, not 1.911. A conclusion about components could change at a longer schedule; it should be rechecked, not assumed.
- **Test used for the comparison:** the verdicts rely on the test block, which is a fixed single text. It is acceptable here because no model was *selected* on it, but each additional look makes it less independent. The validation block gave the same ordering (see the output).
- **Three tests, one interval each:** with several hypotheses, a few can look significant by chance. These effects are far from zero, so the conclusions do not depend on it, but a borderline effect would.
- Five seeds estimate the seed noise roughly; the interval itself is approximate when the differences are not close to normal.

## Example, worked through

The 1-head variant against the baseline, seed by seed.

- Differences: `0.039, 0.072, 0.045, 0.037, 0.051`; every one is positive.
- Mean `0.0491`, sample std `0.0139`, standard error `0.0139 / sqrt(5) = 0.0062`.
- Interval: `0.0491 +/- 2.776 * 0.0062 = [0.032, 0.066]`. It excludes zero, and 0.049 is more than twice the baseline seed std (0.010), so the rule says "real".

```python
import json, numpy as np
R = [json.loads(l) for l in open("results9.jsonl")]
t = lambda name: np.array([next(r["test"] for r in R if r["name"] == name and r["seed"] == s) for s in range(5)])
base, one = t("baseline"), t("1 head")
d = one - base
print("baseline test:", base.round(3)); print("1 head   test:", one.round(3)); print("differences  :", d.round(3))
se = d.std(ddof=1) / np.sqrt(5)
print(f"mean {d.mean():.4f}, sample std {d.std(ddof=1):.4f}, standard error {se:.4f}")
print(f"95% interval = mean +/- 2.776 * se = [{d.mean() - 2.776 * se:.3f}, {d.mean() + 2.776 * se:.3f}]")
```

Output:

```
baseline test: [2.018 2.006 2.027 2.029 2.015]
1 head   test: [2.058 2.078 2.072 2.067 2.067]
differences  : [0.039 0.072 0.045 0.037 0.051]
mean 0.0491, sample std 0.0139, standard error 0.0062
95% interval = mean +/- 2.776 * se = [0.032, 0.066]
```

## Implementation

`part9.py` trains 25 models (about 13 minutes on a CPU; it appends one line per run to `results9.jsonl`). Keep it next to `part8.py` of the previous part: it imports the data, the split and the scoring function from there.

```python
import json, math, sys, time
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from part8 import V, D, T, train, val, test, windows, bpc          # data, split and fair scoring from Part 8 (keep part8.py in the same folder)

class Block(nn.Module):
    def __init__(s, heads, use_mlp):
        super().__init__(); s.h, s.use_mlp = heads, use_mlp
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)) if use_mlp else None
    def forward(s, x):
        B, t, _ = x.shape
        q, k, v = (a.view(B, t, s.h, D // s.h).transpose(1, 2) for a in s.qkv(s.ln1(x)).split(D, dim=2))
        x = x + s.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
        return x + s.mlp(s.ln2(x)) if s.use_mlp else x

class Tiny(nn.Module):
    def __init__(s, heads=4, use_mlp=True, use_pos=True, layers=2):
        super().__init__(); s.use_pos = use_pos
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.Sequential(*[Block(heads, use_mlp) for _ in range(layers)]); s.ln = nn.LayerNorm(D); s.out = nn.Linear(D, V)
    def forward(s, idx):
        x = s.tok(idx) + (s.pos(torch.arange(idx.shape[1])) if s.use_pos else 0)
        return s.out(s.ln(s.blocks(x)))

def run(seed, steps=1500, lr=3e-3, **cfg):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)        # same seed -> same batches for every variant (a paired comparison)
    m = Tiny(**cfg); opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(1, steps + 1):
        for gr in opt.param_groups: gr["lr"] = lr * (1 - step / (steps + 1))
        xb, yb = windows(train, torch.randint(0, len(train) - T - 1, (64,), generator=g).tolist())
        loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return bpc(m, val), bpc(m, test), sum(p.numel() for p in m.parameters())

if __name__ == "__main__":
    configs = {"baseline": {}, "no MLP": {"use_mlp": False}, "1 head": {"heads": 1}, "no positions": {"use_pos": False}}
    jobs = [(name, seed) for seed in range(5) for name in configs] + [("baseline", seed) for seed in range(5, 10)]   # seeds 5-9: a second group of the SAME model
    t0 = time.time()
    with open("results9.jsonl", "w") as f:
        for name, seed in jobs:
            va, te, npar = run(seed, **configs[name])
            f.write(json.dumps({"name": name, "seed": seed, "val": va, "test": te, "params": npar}) + "\n"); f.flush()
            print(f"{name:13} seed {seed}: validation {va:.3f}  test {te:.3f}  ({npar:,} parameters, {time.time() - t0:.0f} s elapsed)", flush=True)
```

Output of the runs:

```
baseline      seed 0: validation 1.710  test 2.018  (110,384 parameters, 41 s elapsed)
no MLP        seed 0: validation 2.159  test 2.452  (44,208 parameters, 56 s elapsed)
1 head        seed 0: validation 1.774  test 2.058  (110,384 parameters, 90 s elapsed)
no positions  seed 0: validation 2.426  test 2.734  (110,384 parameters, 125 s elapsed)
baseline      seed 1: validation 1.718  test 2.006  (110,384 parameters, 160 s elapsed)
no MLP        seed 1: validation 2.162  test 2.456  (44,208 parameters, 176 s elapsed)
1 head        seed 1: validation 1.783  test 2.078  (110,384 parameters, 211 s elapsed)
no positions  seed 1: validation 2.370  test 2.691  (110,384 parameters, 245 s elapsed)
baseline      seed 2: validation 1.735  test 2.027  (110,384 parameters, 280 s elapsed)
no MLP        seed 2: validation 2.128  test 2.420  (44,208 parameters, 296 s elapsed)
1 head        seed 2: validation 1.784  test 2.072  (110,384 parameters, 330 s elapsed)
no positions  seed 2: validation 2.417  test 2.737  (110,384 parameters, 364 s elapsed)
baseline      seed 3: validation 1.741  test 2.029  (110,384 parameters, 398 s elapsed)
no MLP        seed 3: validation 2.141  test 2.431  (44,208 parameters, 415 s elapsed)
1 head        seed 3: validation 1.774  test 2.067  (110,384 parameters, 448 s elapsed)
no positions  seed 3: validation 2.566  test 2.847  (110,384 parameters, 483 s elapsed)
baseline      seed 4: validation 1.727  test 2.015  (110,384 parameters, 518 s elapsed)
no MLP        seed 4: validation 2.158  test 2.449  (44,208 parameters, 533 s elapsed)
1 head        seed 4: validation 1.786  test 2.067  (110,384 parameters, 567 s elapsed)
no positions  seed 4: validation 2.555  test 2.838  (110,384 parameters, 601 s elapsed)
baseline      seed 5: validation 1.713  test 1.996  (110,384 parameters, 636 s elapsed)
baseline      seed 6: validation 1.739  test 2.044  (110,384 parameters, 671 s elapsed)
baseline      seed 7: validation 1.702  test 1.996  (110,384 parameters, 706 s elapsed)
baseline      seed 8: validation 1.746  test 2.020  (110,384 parameters, 740 s elapsed)
baseline      seed 9: validation 1.734  test 2.017  (110,384 parameters, 775 s elapsed)
```

`analyse9.py` turns the results into the numbers above:

```python
import json
import numpy as np
rows = [json.loads(l) for l in open("results9.jsonl")]
def test(name, seeds): return np.array([r["test"] for s in seeds for r in rows if r["name"] == name and r["seed"] == s])
S = range(5); T975 = 2.776                                    # two-sided 95% critical value of Student's t, 4 degrees of freedom
base = test("baseline", S)
print(f"baseline, seeds 0-4: test {base.mean():.3f} bits/char, std over seeds {base.std(ddof=1):.3f}")
b2 = test("baseline", range(5, 10))
print(f"A/A control: baseline seeds 5-9: {b2.mean():.3f}; difference between the two groups of the SAME model: {b2.mean() - base.mean():+.3f}\n")
print("variant        test mean   paired difference (variant - baseline): mean   95% interval          same sign in")
for name in ("no MLP", "1 head", "no positions"):
    v = test(name, S); d = v - base; se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = d.mean() - T975 * se, d.mean() + T975 * se
    print(f"{name:13}  {v.mean():9.3f}   {d.mean():+.3f}{'':32}[{lo:+.3f}, {hi:+.3f}]   {int((np.sign(d) == np.sign(d.mean())).sum())} of 5 seeds")
```

Output:

```
baseline, seeds 0-4: test 2.019 bits/char, std over seeds 0.010
A/A control: baseline seeds 5-9: 2.015; difference between the two groups of the SAME model: -0.005

variant        test mean   paired difference (variant - baseline): mean   95% interval          same sign in
no MLP             2.442   +0.422                                [+0.393, +0.452]   5 of 5 seeds
1 head             2.068   +0.049                                [+0.032, +0.066]   5 of 5 seeds
no positions       2.769   +0.750                                [+0.670, +0.831]   5 of 5 seeds
```

## Try it

1. Add a variant `{"use_mlp": False}` with `D = 112`, so that its parameter count is close to 110,384. How much of the 0.42 remains?
2. Run 10 seeds instead of 5 for the 1-head variant. How does the interval change, and what is the new `t`?
3. Run the same comparison on the validation score. If validation and test disagree on a conclusion, which would you trust, and why?
