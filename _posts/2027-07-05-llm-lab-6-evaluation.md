---
title: "Improving LLM Performance, Measured (6/6): Pairing Cut Evaluation Error Sevenfold, Yet One Leak of the Test Text Hid Inside the 0.04 Gap Between Seeds"
date: 2027-07-05 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, evaluation, statistics, data contamination, benchmarks, series]
description: "Part 6 of 6. How many evaluation windows a comparison needs, why paired comparison matters, and how much test-set leakage looks like a real improvement, measured on small language models."
---

## What the literature says

Every earlier part of this series ended with a loss difference and a judgement of whether it was real. This part tests the judgement itself. Four public sources frame it.

- **Error bars.** [Miller](https://arxiv.org/abs/2411.00640) argues that evaluations are experiments, conceptualises evaluation questions "as having been drawn from an unseen super-population", and presents formulas for "analyzing evaluation data, measuring differences between two models, and planning an evaluation experiment".
- **Power.** [Card et al.](https://arxiv.org/abs/2010.06595) find "underpowered experiments are common in the NLP literature", that for several GLUE tasks "small test sets mean that most attempted comparisons to state of the art models will not be adequately powered", and that for machine translation "typical test sets of 2000 sentences have approximately 75% power to detect differences of 1 BLEU point".
- **Contamination.** [Magar and Schwartz](https://arxiv.org/abs/2203.08242) pretrain BERT models on corpora joined with labeled downstream datasets, and find that memorization and exploitation of the contaminated data "are affected by different factors such as the number of duplications of the contaminated data and the model size".
- **Leakage inflates scores.** [Zhou et al.](https://arxiv.org/abs/2311.01964) study "benchmark leakage", where data related to evaluation sets is used for training, and find that it "can dramatically boost the evaluation results, which would finally lead to an unreliable assessment of model performance".

All claims come from abstracts. The experiment below measures the same three quantities on small models: how many evaluation items a comparison needs, how much pairing helps, and how large a leak looks next to ordinary training noise.

## Experiment: sixteen models, 2834 evaluation windows

Setup: character-level GPTs (4 layers, width 128, 842,624 parameters) in four arms, four seeds each. Held-out text of 362,843 characters is cut into 2834 non-overlapping windows of 128 characters; the loss of every model on every window is stored.

- **clean800**: trained 800 steps on the training documents.
- **long1000**: the same, trained 1000 steps (25 percent more training).
- **leak1**: 800 steps, with the held-out text added once to the training text.
- **leak5**: 800 steps, with the held-out text added five times, as happens when a benchmark is duplicated across web pages.

Mean held-out loss, bits per character, mean plus or minus standard deviation over four seeds, with the paired difference from clean800 (same seed):

- **clean800**: 2.666 plus or minus 0.031.
- **long1000**: 2.462 plus or minus 0.020; difference -0.203, with every seed between -0.185 and -0.211.
- **leak1**: 2.637 plus or minus 0.017; difference -0.029, with seeds at -0.055, -0.024, -0.030 and -0.006.
- **leak5**: 2.572 plus or minus 0.042; difference -0.093, with seeds at -0.136, -0.093, -0.046 and -0.099.

The mean absolute gap between two clean800 models that differ only in seed was 0.0375 (6 pairs).

To see how many windows a comparison needs, each pair of models was compared on random subsets of 10 to 1000 windows (2000 draws each, drawn without replacement from the 2834). A gap counts as detected when the paired 95 percent interval excludes zero with the same sign as on the full set.

- **Different seeds, same recipe** (gap 0.0375): detected in 37, 58, 74, 89 and 100 percent of draws at 10, 30, 100, 300 and 1000 windows.
- **Leaked once vs clean** (gap 0.029): 25, 46, 71, 79 and 87 percent.
- **Leaked five times vs clean** (gap 0.093): 74, 91, 100, 100 and 100 percent.
- **1000 vs 800 steps** (gap 0.20): 100 percent at every size.
- **Pairing.** For seed 0 against seed 1 on all 2834 windows, the standard error of the difference was 0.0016 when the same windows are compared pairwise, and 0.0116 when the two means are treated as independent. The per-window loss has a standard deviation of 0.44, and seed 0 had the lower loss on 44.8 percent of windows while its mean was higher by 0.011.

![Two panels. Left: share of evaluation subsamples in which a gap between two models is detected against the number of 128-character windows from 10 to 1000, for four pair types. The 1000 versus 800 steps pairs are detected at every size; leaked five times reaches 1.0 at 100 windows; different seeds and leaked once reach 0.89 and 0.79 at 300 windows. Right: held-out loss for four seeds in each of four arms: clean 800 steps 2.67, test text in training once 2.64, five times 2.57, clean 1000 steps 2.46]({{ site.baseurl }}/assets/pic/llm-lab-6-evaluation.png){: width="900" }
_Figure: left, mean over the pairs of each type; right, one dot per seed and a bar at the mean. Measured here._

## Engineering reading

These points are this series' reading of the sources and the measurement.

1. **Pair the comparison.** Using the same windows for both models cut the standard error by a factor of 7 (0.0016 against 0.0116). Most of the window-to-window variation is shared by both models, so it cancels in the difference. Comparing two reported averages as if they were independent wastes that.
2. **Window noise and training noise are different things.** At 1000 windows the paired test detected a gap between two models that differ only in seed in all draws (100 percent after rounding). That does not show that one recipe is better; it shows that the two trained models differ, by 0.0375, which is the size of the seed spread. The evaluation set size controls the first source of noise. Only repeated training with several seeds, as in Part 2, measures the second.
3. **A leak of one copy looks like seed noise.** Adding the held-out text once lowered the loss by 0.029, less than the 0.0375 gap between two seeds of the same recipe, and one of the four seeds moved by only 0.006. Nothing in a single comparison would reveal it. Five copies lowered the loss by 0.093, about 2.5 times the seed gap and 46 percent of the gain from 25 percent more training, which is consistent with Magar and Schwartz (the number of duplicates matters) and with Zhou et al. (inflated results).
4. **At this scale a leak did not reorder the arms.** In all four seeds the order was long1000, leak5, leak1, clean800. A leak the size of 0.09 bits would still beat any real improvement smaller than that, so two models whose gap is below the leak size cannot be ranked unless the test text is known to be absent from training.
5. **Sample size scales with the gap.** A gap of 0.2 was detected with 10 windows; a gap of 0.03 to 0.04 needed about 300 windows for 80 to 90 percent detection. Card et al. report the same pattern in NLP benchmarks: the smaller the difference of interest, the more items are required, and typical test sets are too small for small differences.

## Limits

- Four seeds per arm and one corpus; the leaked text is the same kind of prose as the training text, so this measures leakage of language-modelling data, not of labelled question-answer benchmarks.
- The 2834 windows are a finite set; subsets of up to 1000 are drawn without replacement, and the reference sign is the sign on the full set. The detection rates describe this set, not a larger population.
- Windows are 128 characters and non-overlapping, but they come from a limited number of documents, so the window-level intervals treat dependent windows as independent.
- The leaked arms trained for 800 steps, so memorization is small; larger models and longer training memorize far more, which makes the effect of leakage larger than here.
- Literature claims come from abstracts; no result was reproduced.

## Series recap

The six parts measured one lever each against its own seed noise on the same small models. The sizes, in bits per character unless stated:

- **Data**: duplicating training text opened a memorisation gap of 0.17 without moving held-out loss; 30 percent code in the mix cost 0.27 on prose.
- **Hyperparameters**: raising the learning rate from 0.003 to 0.01 gained 0.108; dropping warmup cost 0.233; the seed spread was about 0.05.
- **Compression**: 4-bit rounding cost 0.020; pruning half of the weights cost 0.161.
- **Decoding**: greedy decoding had the highest share of valid words and the lowest diversity; speculative decoding was at best 19 percent faster at a draft cost of 0.25.
- **Retrieval and context**: the exact answer in the context changed the loss by 0.0005.
- **Evaluation**: pairing cut the error by 7 times; one leak was smaller than the gap between seeds.

The common rule is to compare an effect with the spread of the same procedure repeated, before comparing it with another effect.

## Code and output

Experiment code (model and training loop from Part 1):

```python
import sys, json, math, numpy as np, torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
import torch.nn.functional as F, lm
P = lm.load_prose(); PTR, PVA = lm.split(P); TRS = "".join(PTR); VAS = "".join(PVA)
def train_model(seed, text, steps):
    src = lm.enc(text); torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = lm.GPT(); lr, warmup, B = 3e-3, 100, 16
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01); probs = torch.ones(1)
    for t in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * min(1, (t + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
        x, y = lm.batch([src], probs, B, g); loss = F.cross_entropy(m(x).reshape(-1, lm.V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m
@torch.no_grad()
def window_losses(m, text):
    ids = lm.enc(text); n = (len(ids) - 1) // lm.CTX; x = ids[:n * lm.CTX].view(n, lm.CTX); y = ids[1:n * lm.CTX + 1].view(n, lm.CTX); out = []
    for i in range(0, n, 100): out += (F.cross_entropy(m(x[i:i + 100]).reshape(-1, lm.V), y[i:i + 100].reshape(-1), reduction="none").view(-1, lm.CTX).mean(1) / math.log(2)).tolist()
    return out
if __name__ == "__main__":
    seed, grp = int(sys.argv[1]), sys.argv[2]; res = {}
    arms = {"A": [("clean800", TRS, 800), ("long1000", TRS, 1000)], "B": [("leak1", TRS + VAS, 800), ("leak5", TRS + VAS * 5, 800)]}[grp]
    for name, text, steps in arms:
        m = train_model(seed, text, steps); res[name] = window_losses(m, VAS); print(name, seed, round(float(np.mean(res[name])), 3), flush=True)
        json.dump(res, open(f"f6_seed{seed}_{grp}.json", "w"))
```

Statistics output:

```
models: 16 (4 seeds x 4 arms), evaluation windows: 2834 non-overlapping windows of 128 characters (362,843 held-out characters)

held-out loss (bits/char), mean ± sd over 4 seeds, and paired difference from clean800
clean800  2.666 ± 0.031   vs clean800: +0.000 (per seed +0.000, +0.000, +0.000, +0.000)
long1000  2.462 ± 0.020   vs clean800: -0.203 (per seed -0.209, -0.211, -0.209, -0.185)
leak1     2.637 ± 0.017   vs clean800: -0.029 (per seed -0.055, -0.024, -0.030, -0.006)
leak5     2.572 ± 0.042   vs clean800: -0.093 (per seed -0.136, -0.093, -0.046, -0.099)

full-set gap size (mean absolute paired difference in bits/char):
  same recipe, different seeds (6 pairs): 0.0375
  1000 vs 800 steps, same seed (4 pairs): 0.2035
  leaked once vs clean (4 pairs): 0.0287
  leaked 5 times vs clean (4 pairs): 0.0934

for each pair: subsample n windows (2000 draws, without replacement); 'sign kept' = share of draws whose sign matches the full-set sign; 'detected' = share whose paired 95% interval excludes 0 with the full-set sign
windows n:                        10            30           100           300          1000
same recipe, different seeds (6 pairs)
   sign kept:                 0.86          0.92          0.97          1.00          1.00
   detected:                  0.37          0.58          0.74          0.89          1.00
1000 vs 800 steps, same seed (4 pairs)
   sign kept:                 1.00          1.00          1.00          1.00          1.00
   detected:                  1.00          1.00          1.00          1.00          1.00
leaked once vs clean (4 pairs)
   sign kept:                 0.81          0.88          0.93          0.97          1.00
   detected:                  0.25          0.46          0.71          0.79          0.87
leaked 5 times vs clean (4 pairs)
   sign kept:                 0.98          1.00          1.00          1.00          1.00
   detected:                  0.74          0.91          1.00          1.00          1.00

seed 0 vs seed 1 on all 2834 windows: paired SE 0.0016, unpaired SE 0.0116
per-window loss sd (clean800 seed 0): 0.439; share of windows where seed 0 beats seed 1: 0.448

ranking by held-out loss, per seed (lowest first):
  seed 0 long1000 < leak5 < leak1 < clean800
  seed 1 long1000 < leak5 < leak1 < clean800
  seed 2 long1000 < leak5 < leak1 < clean800
  seed 3 long1000 < leak5 < leak1 < clean800
```

## References

- [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/abs/2411.00640), 2024.
- [With Little Power Comes Great Responsibility](https://arxiv.org/abs/2010.06595), 2020.
- [Data Contamination: From Memorization to Exploitation](https://arxiv.org/abs/2203.08242), 2022.
- [Don't Make Your LLM an Evaluation Benchmark Cheater](https://arxiv.org/abs/2311.01964), 2023.
