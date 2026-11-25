---
title: "Who Does the Research, Part 5: A 64-GPU Month Covers 14 Percent of Notable Language Models"
date: 2027-06-22 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, research landscape, compute, academia, open weights, who does the research]
description: "Part 5 of 6 on where large language model research happens. Epoch AI data on training compute, plus a GPU-month calculation, shows what universities can and cannot reach, and a survey of US AI researchers shows how much of that gap they report as binding."
---

## What this post finds

Compute is the resource that most plainly separates organisations, so this part measures it twice: from Epoch AI's database of notable models, and from a calculation of what a fixed cluster buys.

- **Industry-only teams train much larger models, and the gap kept growing.** In [Epoch AI's database](https://epoch.ai/data/ai-models), the median training compute of notable language models from industry-only teams rose from about 1.9e21 FLOP (2019) to 4.1e24 (2025). Academia-only teams have 0 to 6 such models per year with a compute estimate, so their medians (3.6e17 in 2019, 9.5e22 in 2023) rest on a handful of models. The largest academia-only model in the data is GLM-130B from Tsinghua (August 2022, 3.5e23 FLOP). The largest industry-only entry is Grok 4 (July 2025, 5.0e26 FLOP), about 1,400 times larger. Llama 3.1 405B (3.8e25) is about 110 times larger.
- **A cluster buys a bounded slice of that range.** At the dense BF16 peak of an H100 and 40 percent utilisation, 64 H100 GPUs running for 30 days deliver 6.6e22 FLOP. Only 14 percent of the 121 notable language models from 2023 to 2025 with a compute estimate used no more, and 8 GPUs for 30 days cover 7 percent. 512 GPUs cover 29 percent and 4,096 GPUs cover 67 percent. Llama 3.1 405B at the same utilisation would need about 37,000 H100 GPUs for 30 days.
- **Academia-only models in the data are mostly derivatives.** Of the 9 academia-only notable language models of 2023 to 2025, 8 list a base model (they start from an existing model), against 20 of 30 mixed teams and 45 of 204 industry-only teams. Reported hardware drops the same way: a median of 20 chips (4 models) for academia-only, 128 (16 models) for mixed teams and 1,536 (35 models) for industry-only. Open weights are available for 6 of the 9 academia-only models, 25 of 30 mixed and 78 of 204 industry-only.
- **Researchers themselves name talent and data before compute.** In a 2022 survey of US AI researchers (410 complete responses, 67 percent academic), 90 percent rated specialised knowledge or skills as very or extremely important to their most significant project, against 52 percent for large amounts of compute and 51 percent for unique data. For heavy compute users the picture is different: the authors report that those who already use large amounts of compute say they need more.

![Two panels. Left: median training compute of notable language models by year for academia-only, mixed and industry-only teams, with horizontal lines for 8, 64 and 512 H100 GPUs running for 30 days; the industry-only median rises from about 2e21 FLOP in 2019 to 4e24 in 2025, while academia-only has between 0 and 6 models per year. Right: cumulative share of notable language models from 2023 to 2025 by training compute; 7 percent fit inside 8 H100 for 30 days, 14 percent inside 64, 29 percent inside 512 and 67 percent inside 4,096]({{ site.baseurl }}/assets/pic/landscape-5-compute.png){: width="900" }
_Figure: left, median training compute by organisation type, Epoch AI; right, share of notable language models 2023 to 2025 that fit inside a 30-day H100 cluster of each size. Computed here from public data._

## What the sources do and do not show

- **Epoch's list is a selection.** The database contains "notable" models, chosen by criteria such as citations and state-of-the-art results, and only about half of the language models (313 of 589) have a training-compute estimate. A small academic model that is not notable is absent, so academia-only counts are not a count of academic work. They show where academia appears among the frontier and widely used models.
- **Estimates, not measurements.** Compute values are Epoch's estimates, and the organisation type is Epoch's categorisation. The academia-only median for 2023 rests on two models, one of them FinGPT-13B at 1.6e23 FLOP, so one estimate moves it a long way.
- **The cluster calculation is a peak-based bound.** Dense BF16 peak is 989.4 TFLOPS for an H100 SXM5 (NVIDIA's page lists 1,979 TFLOPS with sparsity; dense is half, as the architecture whitepaper shows) and 312 TFLOPS for an A100. Forty percent utilisation sits inside the 38 to 43 percent that the Llama 3 report gives for 405B pretraining. Real jobs lose time to failures, tuning and queueing, so 30 days of wall-clock time yields less. Model size uses the rule C = 6ND (compute equals 6 times parameters times tokens, as in [Kaplan et al.](https://arxiv.org/abs/2001.08361)) and 20 tokens per parameter, the ratio of the Chinchilla model (70B parameters, 1.4 trillion tokens).
- **The survey is old and narrow.** It was fielded in June 2022, before the 2023 growth in compute demand, covers US researchers only, and its authors conclude that the compute problems of large foundation models affect a small minority of AI researchers. It reports that academics spend significantly less on their most compute-intensive project but not significantly fewer GPU-hours; the authors list cheaper GPUs, on-premise hardware that is already paid for and discounted cloud rates as possible reasons that cost is a less reliable measure.
- **Industry-only is not one thing.** It includes startups with small budgets and hyperscalers, which this data does not separate.

## What the checks show

The script classifies Epoch's language-domain models, summarises compute by year, and computes the cluster budgets.

```python
import csv, collections, statistics as st, math, json
rows = list(csv.DictReader(open("epoch.csv", encoding="utf-8")))
def cls(r):
    s = set(x.strip() for x in r["Organization categorization"].split(",") if x.strip())
    if not s: return None
    if s == {"Academia"}: return "academia only"
    if "Industry" in s and "Academia" in s: return "mixed"
    if "Industry" in s: return "industry only"
    return "other"
LM = []
for r in rows:
    if "Language" not in r["Domain"]: continue
    c = cls(r); y = r["Publication date"][:4]
    if not c or not y.isdigit(): continue
    try: f = float(r["Training compute (FLOP)"])
    except ValueError: f = None
    LM.append(dict(model=r["Model"], year=int(y), cls=c, flop=f, hw=r["Hardware quantity"], base=bool(r["Base model"].strip()), open=r["Open model weights?"]))
print("Epoch notable language-domain models with an organisation type:", len(LM), "; with training compute:", sum(1 for m in LM if m["flop"]))
print("\nMedian and maximum training compute (FLOP) of notable language models, by year and organisation type (n = models with compute estimates)")
res = {}
for y in range(2019, 2026):
    line = f"{y}"
    for c in ("academia only", "mixed", "industry only"):
        v = [m["flop"] for m in LM if m["year"] == y and m["cls"] == c and m["flop"]]
        res[f"{y}|{c}"] = [st.median(v), max(v), len(v)] if v else None
        line += f" | {c}: " + (f"n={len(v)} median {st.median(v):.1e} max {max(v):.1e}" if v else "n=0")
    print(line)
json.dump(res, open("oa/d5_medians.json", "w"))
H100 = 989.4e12; MFU = 0.40; DAY = 86400  # H100 SXM5 dense BF16 peak (NVIDIA architecture whitepaper); MFU inside the 38-43% reported for Llama 3 405B
v = sorted(m["flop"] for m in LM if 2023 <= m["year"] <= 2025 and m["flop"])
print(f"\nNotable language models 2023-2025 with a compute estimate: {len(v)}; median {st.median(v):.1e} FLOP")
print("\nWhat a fixed cluster buys: FLOP = GPUs x days x 86,400 s x 989.4 TFLOPS (dense BF16 peak) x 40% utilisation")
budgets = {"8 H100 for 30 days": 8*30, "64 H100 for 30 days": 64*30, "512 H100 for 30 days": 512*30, "4,096 H100 for 30 days": 4096*30}
out = {}
for k, gd in budgets.items():
    C = gd * DAY * H100 * MFU; share = sum(1 for x in v if x <= C) / len(v); n = math.sqrt(C / 120)
    out[k] = (C, share, n); print(f"{k:24s} {C:.1e} FLOP | {100*share:4.0f}% of those {len(v)} models used no more | compute-optimal size at 20 tokens per parameter (C = 6 N D): {n/1e9:6.1f}B parameters on {20*n/1e12:5.2f}T tokens")
json.dump(out, open("oa/d5_budgets.json", "w"))
ll = [(m["model"], m["flop"]) for m in LM if m["model"].startswith("Llama 3.1-405B")]
print("\nLlama 3.1-405B in Epoch:", ll, "->", f"{ll[0][1]/(DAY*30*H100*MFU):,.0f} H100 for 30 days at the same utilisation")
print("\nModels with a base model (fine-tunes or derivatives), hardware used and weights, language models 2023-2025")
for c in ("academia only", "mixed", "industry only"):
    g = [m for m in LM if 2023 <= m["year"] <= 2025 and m["cls"] == c]
    hw = [float(m["hw"]) for m in g if m["hw"].replace(".", "", 1).isdigit()]
    print(f"{c:14s} n={len(g):3d} | with a base model {sum(m['base'] for m in g):3d} | open weights {sum(m['open']=='Yes' for m in g):3d} | GPUs/chips reported for {len(hw):3d} models, median {st.median(hw) if hw else float('nan'):.0f}")
```

Output:

```
Epoch notable language-domain models with an organisation type: 589 ; with training compute: 313

Median and maximum training compute (FLOP) of notable language models, by year and organisation type (n = models with compute estimates)
2019 | academia only: n=6 median 3.6e+17 max 3.8e+20 | mixed: n=5 median 8.3e+20 max 8.5e+21 | industry only: n=11 median 1.9e+21 max 3.3e+22
2020 | academia only: n=5 median 2.7e+19 max 1.8e+22 | mixed: n=6 median 1.7e+19 max 3.1e+21 | industry only: n=9 median 1.6e+22 max 3.1e+23
2021 | academia only: n=2 median 4.1e+19 max 7.8e+19 | mixed: n=7 median 1.6e+21 max 1.0e+24 | industry only: n=22 median 3.5e+22 max 2.0e+24
2022 | academia only: n=5 median 6.5e+21 max 3.5e+23 | mixed: n=3 median 2.6e+19 max 4.3e+23 | industry only: n=26 median 2.3e+23 max 2.7e+24
2023 | academia only: n=2 median 9.5e+22 max 1.6e+23 | mixed: n=13 median 6.3e+22 max 1.0e+24 | industry only: n=26 median 5.6e+23 max 5.0e+25
2024 | academia only: n=0 | mixed: n=4 median 1.6e+24 max 3.9e+24 | industry only: n=34 median 3.5e+24 max 3.8e+25
2025 | academia only: n=1 median 4.0e+20 max 4.0e+20 | mixed: n=3 median 4.4e+24 max 4.4e+24 | industry only: n=35 median 4.1e+24 max 5.0e+26

Notable language models 2023-2025 with a compute estimate: 121; median 2.7e+24 FLOP

What a fixed cluster buys: FLOP = GPUs x days x 86,400 s x 989.4 TFLOPS (dense BF16 peak) x 40% utilisation
8 H100 for 30 days       8.2e+21 FLOP |    7% of those 121 models used no more | compute-optimal size at 20 tokens per parameter (C = 6 N D):    8.3B parameters on  0.17T tokens
64 H100 for 30 days      6.6e+22 FLOP |   14% of those 121 models used no more | compute-optimal size at 20 tokens per parameter (C = 6 N D):   23.4B parameters on  0.47T tokens
512 H100 for 30 days     5.3e+23 FLOP |   29% of those 121 models used no more | compute-optimal size at 20 tokens per parameter (C = 6 N D):   66.2B parameters on  1.32T tokens
4,096 H100 for 30 days   4.2e+24 FLOP |   67% of those 121 models used no more | compute-optimal size at 20 tokens per parameter (C = 6 N D):  187.1B parameters on  3.74T tokens

Llama 3.1-405B in Epoch: [('Llama 3.1-405B', 3.8e+25)] -> 37,044 H100 for 30 days at the same utilisation

Models with a base model (fine-tunes or derivatives), hardware used and weights, language models 2023-2025
academia only  n=  9 | with a base model   8 | open weights   6 | GPUs/chips reported for   4 models, median 20
mixed          n= 30 | with a base model  20 | open weights  25 | GPUs/chips reported for  16 models, median 128
industry only  n=204 | with a base model  45 | open weights  78 | GPUs/chips reported for  35 models, median 1536
```

## What this means for reading the landscape

1. **The frontier is a compute-gated club.** The largest models in the data come from industry teams, and the training budgets for them sit one to three orders of magnitude above a 64-GPU month. Part 4 found the same pattern from the paper side: pretraining and scaling is 5.6 percent of university LLM papers and 7.8 percent of company papers.
2. **Academic work concentrates where a modest cluster is enough.** A 64-GPU month supports compute-optimal models of roughly 23 billion parameters, and post-training, probing and evaluation of released models need far less than pretraining. This is an inference from the arithmetic and from the themes in Part 4, not a measured result of this data.
3. **Open weights are the bridge.** 8 of 9 academia-only models start from an existing model, and 78 of 204 industry-only models have open weights; the research that universities do on top of released models depends on those releases (Part 2 looked at what labs choose to release).
4. **Compute is one constraint among several.** In the 2022 survey, talent and data were reported as binding more often than compute for most researchers, and compute for a small group of foundation-model teams.
5. **Next.** Part 6 closes the series with geography and collaboration, and a list of what readers should watch.

## References

- [Resource Democratization: Is Compute the Binding Constraint on AI Research?](https://ojs.aaai.org/index.php/AAAI/article/view/29959), AAAI 2024 (survey of 410 US AI researchers).
- [Epoch AI, notable AI models database](https://epoch.ai/data/ai-models).
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla), 2022.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024 (Table 4, MFU of 38 to 43 percent).
- [The De-democratization of AI: Deep Learning and the Compute Divide in Artificial Intelligence Research](https://arxiv.org/abs/2010.15581), 2020.
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/), [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf) and [H100 architecture whitepaper (hosted copy)](https://www.techpowerup.com/gpu-specs/docs/nvidia-gh100-architecture.pdf).
