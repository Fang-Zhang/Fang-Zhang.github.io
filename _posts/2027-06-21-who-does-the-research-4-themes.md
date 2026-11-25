---
title: "Who Does the Research, Part 4: Pretraining Fell from 25 to 5.5 Percent of LLM Papers While Reasoning Tripled"
date: 2027-06-21 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, research landscape, research themes, reasoning, post-training, agents, who does the research]
description: "Part 4 of 6 on where large language model research happens. A count of OpenAlex computer-science papers by theme shows the research mix moving from pretraining to reasoning, post-training, agents and multimodal work, with similar mixes at universities and companies."
---

## What this post finds

Parts 2 and 3 asked who. This part asks what. Each computer-science paper in [OpenAlex](https://docs.openalex.org/) whose title or abstract mentions "large language model(s)" (133 papers in 2021, 511 in 2022, 6,057 in 2023, 24,035 in 2024, 50,921 in 2025) is checked against nine keyword themes. A paper can match several themes, so the shares do not add to 100.

- **The mix moved away from pretraining.** The share of LLM papers that mention pretraining or scaling laws fell from 24.8 percent in 2021 to 5.5 percent in 2025, while the count grew from 33 to 2,823 papers. Reasoning (reasoning, chain-of-thought) rose from 7.5 to 22.9 percent, post-training and alignment from 5.3 to 13.4, agents from 3.8 to 13.3, multimodal from 4.5 to 12.9 and retrieval-augmented generation from 2.3 to 8.1. Interpretability (6.8 to 12.6) and safety and trust (9.0 to 12.6) grew more slowly, efficiency stayed near 8 to 9 percent.
- **Universities and companies work on a similar mix.** In 2025 the theme shares of papers with a university author and papers with a company author differ by at most 4.5 points. Companies are higher on post-training and alignment (17.9 against 13.6 percent), reasoning (27.3 against 23.3), efficiency (11.4 against 9.2), agents (15.3 against 12.6) and pretraining (7.8 against 5.6), and lower on interpretability (9.1 against 12.5) and safety and trust (9.8 against 12.3). Retrieval-augmented generation is equal (7.9 and 7.8).
- **The company share fell in almost every theme.** Papers with a company author were 26.3 percent of LLM papers in 2021 and 13.0 percent in 2025. From 2022 to 2025 the company share fell in 8 of 9 themes (post-training from 28.6 to 17.4 percent, reasoning from 29.4 to 15.5, interpretability from 26.0 to 9.4); only multimodal rose, from 14.3 to 15.8 on a small 2022 base.

![Two panels. Left: share of LLM papers matching each of nine themes, 2021 to 2025. Pretraining and scaling falls from 24.8 to 5.5 percent while reasoning rises from 7.5 to 22.9 percent, post-training and alignment from 5.3 to 13.4, agents from 3.8 to 13.3 and multimodal from 4.5 to 12.9 percent. Right: in 2025 the theme shares of papers with a company author and papers with a university author differ by at most 4.5 points]({{ site.baseurl }}/assets/pic/landscape-4-themes.png){: width="900" }
_Figure: left, theme share of LLM papers by year; right, theme share in 2025 for papers with a university author and papers with a company author. Computed here from OpenAlex counts._

## What the counts do and do not show

- **Keywords, not classification.** A theme is a list of words in the title or abstract (the lists are in the code). Some words are ambiguous: "alignment" also appears in image-text work, "agent" in multi-agent reinforcement learning, "adversarial" and "privacy" in work that is not about language models, and "pretraining" appears in any paper that uses a pretrained model. A theme share measures how often a word occurs, not what a paper is about.
- **Shares fall when a field grows by new topics.** Pretraining's share fell while its count grew 85-fold. A falling share means other themes grew faster.
- **Small early years.** 2021 and 2022 have 133 and 511 papers, so early shares move by several points on a handful of papers.
- **Types overlap.** "Company" and "university" mean at least one author with that affiliation type in OpenAlex, so one joint paper counts on both sides. The universities group is large (39,238 papers in 2025) against companies (6,631), which makes the intervals narrow. The data is close to a census of what OpenAlex indexes, so the 95 percent intervals in the output describe the keyword-match sampling only, and a gap of 2 or 3 points should be treated as small even when intervals do not overlap.
- **Field filter.** Papers are limited to OpenAlex's computer-science field, and the LLM phrase must appear in the title or abstract. Papers that discuss "language models" without the word "large", or papers that appear in other fields, are outside the count.
- **Coverage.** OpenAlex does not index every technical report or preprint with complete affiliations, so industry work published as blog posts or reports is under-counted (see Part 2).

## What the checks show

The first script pulls counts from the OpenAlex API (an API key is read from the environment); the second computes shares, company and university comparisons and intervals.

```python
import os, json, time, urllib.request, urllib.parse, ssl
ctx = ssl.create_default_context(cafile=os.path.expanduser("~/.hermes/certs/certifi_plus_zscaler.pem"))
KEY = os.environ["OA_KEY"]  # read from the environment, never written to disk
LLM = '"large language model"|"large language models"'
THEMES = {
 "pretraining and scaling": '"pretraining"|"pre-training"|"scaling law"|"scaling laws"',
 "post-training and alignment": '"instruction tuning"|"instruction-tuned"|"RLHF"|"reinforcement learning from human feedback"|"preference optimization"|"supervised fine-tuning"|"alignment"',
 "reasoning": '"reasoning"|"chain-of-thought"|"chain of thought"',
 "efficiency": '"quantization"|"pruning"|"distillation"|"parameter-efficient"|"LoRA"|"KV cache"|"mixture of experts"|"efficient inference"',
 "multimodal": '"multimodal"|"multi-modal"|"vision-language"',
 "interpretability": '"interpretability"|"explainability"|"mechanistic"',
 "agents": '"agent"|"agents"|"agentic"|"tool use"|"tool learning"',
 "safety and trust": '"safety"|"jailbreak"|"red teaming"|"adversarial"|"backdoor"|"privacy"|"toxicity"',
 "retrieval-augmented generation": '"retrieval-augmented"|"retrieval augmented"',
}
def q(filt, group=None):
    u = "https://api.openalex.org/works?filter=" + urllib.parse.quote(filt, safe=':,|"') + "&per-page=200&api_key=" + KEY + ("&group_by=" + group if group else "")
    for a in range(3):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), context=ctx))
        except Exception as e:
            time.sleep(2 * (a + 1)); err = e
    raise err
out = {}
for y in range(2021, 2026):
    base = f"title_and_abstract.search:{LLM},primary_topic.field.id:17,publication_year:{y}"
    r = q(base, "authorships.institutions.type"); out[f"{y}|ALL"] = {"total": r["meta"]["count"], "types": {g["key"].split("/")[-1]: g["count"] for g in r["group_by"]}}
    for t, s in THEMES.items():
        r = q(base + f",title_and_abstract.search:{s}", "authorships.institutions.type")
        out[f"{y}|{t}"] = {"total": r["meta"]["count"], "types": {g["key"].split("/")[-1]: g["count"] for g in r["group_by"]}}
        time.sleep(0.6)
# sanity: the AND really intersects (theme-only count must exceed the intersection)
chk = q(f"primary_topic.field.id:17,publication_year:2024,title_and_abstract.search:{THEMES['reasoning']}")["meta"]["count"]
out["check_reasoning_2024_theme_only"] = chk
json.dump(out, open("oa/d4.json", "w"))
for y in range(2021, 2026):
    a = out[f"{y}|ALL"]; print(y, "LLM papers", a["total"], a["types"])
    for t in THEMES: print("   ", f"{t:32s}", out[f"{y}|{t}"]["total"])
print("reasoning 2024 theme-only", chk, "vs with LLM filter", out["2024|reasoning"]["total"])
```

Output:

```
2021 LLM papers 133 {'education': 89, 'company': 35, 'facility': 8, 'nonprofit': 6, 'government': 5, 'other': 1}
    pretraining and scaling          33
    post-training and alignment      7
    reasoning                        10
    efficiency                       11
    multimodal                       6
    interpretability                 9
    agents                           5
    safety and trust                 12
    retrieval-augmented generation   3
2022 LLM papers 511 {'education': 333, 'company': 125, 'facility': 29, 'nonprofit': 23, 'government': 12, 'other': 4, 'healthcare': 2}
    pretraining and scaling          92
    post-training and alignment      21
    reasoning                        68
    efficiency                       28
    multimodal                       14
    interpretability                 50
    agents                           26
    safety and trust                 50
    retrieval-augmented generation   9
2023 LLM papers 6057 {'education': 4471, 'company': 1098, 'facility': 658, 'government': 254, 'nonprofit': 155, 'healthcare': 92, 'other': 68, 'archive': 15}
    pretraining and scaling          548
    post-training and alignment      490
    reasoning                        897
    efficiency                       353
    multimodal                       408
    interpretability                 553
    agents                           427
    safety and trust                 536
    retrieval-augmented generation   113
2024 LLM papers 24035 {'education': 18069, 'company': 3747, 'facility': 2532, 'government': 1122, 'nonprofit': 451, 'healthcare': 320, 'other': 203, 'archive': 50}
    pretraining and scaling          1729
    post-training and alignment      2571
    reasoning                        3614
    efficiency                       1833
    multimodal                       2335
    interpretability                 2219
    agents                           1996
    safety and trust                 2379
    retrieval-augmented generation   1407
2025 LLM papers 50921 {'education': 39238, 'company': 6631, 'facility': 5027, 'government': 2362, 'nonprofit': 763, 'healthcare': 733, 'other': 371, 'archive': 108}
    pretraining and scaling          2823
    post-training and alignment      6811
    reasoning                        11653
    efficiency                       4561
    multimodal                       6562
    interpretability                 6416
    agents                           6751
    safety and trust                 6394
    retrieval-augmented generation   4140
reasoning 2024 theme-only 10632 vs with LLM filter 3614
```

```python
import json, math
d = json.load(open("oa/d4.json"))
T = ["pretraining and scaling","post-training and alignment","reasoning","efficiency","multimodal","interpretability","agents","safety and trust","retrieval-augmented generation"]
Y = [2021, 2022, 2023, 2024, 2025]
print("Papers (computer science, title or abstract mentions 'large language model(s)') per year:", {y: d[f"{y}|ALL"]["total"] for y in Y})
print("\nShare of those papers that also match each theme (%); themes overlap, so columns do not sum to 100")
print(f"{'':32s}" + "".join(f"{y:>8d}" for y in Y))
for t in T: print(f"{t:32s}" + "".join(f"{100*d[f'{y}|{t}']['total']/d[f'{y}|ALL']['total']:8.1f}" for y in Y))
def wil(k, n, z=1.96):
    p = k/n; dd = 1+z*z/n; c = p+z*z/(2*n); h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-h)/dd, (c+h)/dd
print("\n2025, share of each paper type's LLM papers that match the theme (%), with 95% Wilson interval for company-minus-education overlap")
a = d["2025|ALL"]["types"]
for t in T:
    r = d[f"2025|{t}"]["types"]; row = []
    for ty in ("education", "company", "facility"):
        k = r.get(ty, 0); n = a[ty]; lo, hi = wil(k, n); row.append((100*k/n, 100*lo, 100*hi))
    sep = "separate" if (row[1][1] > row[0][2] or row[1][2] < row[0][1]) else "overlap"
    print(f"{t:32s} education {row[0][0]:5.1f} [{row[0][1]:4.1f},{row[0][2]:4.1f}]  company {row[1][0]:5.1f} [{row[1][1]:4.1f},{row[1][2]:4.1f}]  facility {row[2][0]:5.1f}  intervals {sep}")
print("\nCompany share of each theme's papers by year (% of papers in the theme with at least one company author)")
print(f"{'':32s}" + "".join(f"{y:>8d}" for y in Y))
print(f"{'all LLM papers':32s}" + "".join(f"{100*d[f'{y}|ALL']['types'].get('company',0)/d[f'{y}|ALL']['total']:8.1f}" for y in Y))
for t in T: print(f"{t:32s}" + "".join(f"{100*d[f'{y}|{t}']['types'].get('company',0)/d[f'{y}|{t}']['total']:8.1f}" for y in Y))
print("\nCheck: theme-only count for reasoning in 2024:", d["check_reasoning_2024_theme_only"], "; with the LLM filter:", d["2024|reasoning"]["total"])
```

Output:

```
Papers (computer science, title or abstract mentions 'large language model(s)') per year: {2021: 133, 2022: 511, 2023: 6057, 2024: 24035, 2025: 50921}

Share of those papers that also match each theme (%); themes overlap, so columns do not sum to 100
                                    2021    2022    2023    2024    2025
pretraining and scaling             24.8    18.0     9.0     7.2     5.5
post-training and alignment          5.3     4.1     8.1    10.7    13.4
reasoning                            7.5    13.3    14.8    15.0    22.9
efficiency                           8.3     5.5     5.8     7.6     9.0
multimodal                           4.5     2.7     6.7     9.7    12.9
interpretability                     6.8     9.8     9.1     9.2    12.6
agents                               3.8     5.1     7.0     8.3    13.3
safety and trust                     9.0     9.8     8.8     9.9    12.6
retrieval-augmented generation       2.3     1.8     1.9     5.9     8.1

2025, share of each paper type's LLM papers that match the theme (%), with 95% Wilson interval for company-minus-education overlap
pretraining and scaling          education   5.6 [ 5.4, 5.9]  company   7.8 [ 7.2, 8.5]  facility   6.9  intervals separate
post-training and alignment      education  13.6 [13.2,13.9]  company  17.9 [17.0,18.9]  facility  13.6  intervals separate
reasoning                        education  23.3 [22.9,23.7]  company  27.3 [26.2,28.4]  facility  23.9  intervals separate
efficiency                       education   9.2 [ 9.0, 9.5]  company  11.4 [10.7,12.2]  facility   9.2  intervals separate
multimodal                       education  13.6 [13.2,13.9]  company  15.7 [14.8,16.6]  facility  16.1  intervals separate
interpretability                 education  12.5 [12.1,12.8]  company   9.1 [ 8.4, 9.8]  facility  11.0  intervals separate
agents                           education  12.6 [12.3,12.9]  company  15.3 [14.5,16.2]  facility  13.0  intervals separate
safety and trust                 education  12.3 [11.9,12.6]  company   9.8 [ 9.1,10.6]  facility  11.2  intervals separate
retrieval-augmented generation   education   7.8 [ 7.5, 8.1]  company   7.9 [ 7.2, 8.5]  facility   7.5  intervals overlap

Company share of each theme's papers by year (% of papers in the theme with at least one company author)
                                    2021    2022    2023    2024    2025
all LLM papers                      26.3    24.5    18.1    15.6    13.0
pretraining and scaling             33.3    22.8    21.0    19.2    18.4
post-training and alignment         28.6    28.6    23.9    21.6    17.4
reasoning                           30.0    29.4    21.6    18.5    15.5
efficiency                          54.5    35.7    25.2    20.3    16.6
multimodal                          16.7    14.3    26.0    19.9    15.8
interpretability                    11.1    26.0    14.3    12.9     9.4
agents                              20.0    19.2    17.8    16.9    15.1
safety and trust                    25.0    20.0    18.3    14.4    10.2
retrieval-augmented generation      33.3    33.3    28.3    17.3    12.6

Check: theme-only count for reasoning in 2024: 10632 ; with the LLM filter: 3614
```

The reasoning check at the end confirms that the theme filter combined with the LLM filter narrows the count (10,632 papers for the theme alone in 2024 against 3,614 with the LLM phrase).

## What this means for reading the landscape

1. **The centre of gravity moved.** In the paper record, the question changed from how to train a language model to how to make it reason, follow preferences, use tools and handle images.
2. **The topic mix is not a university-company divide.** At the level of these nine themes, both groups follow the same direction; company papers lean a few points toward post-training, efficiency and agents, and university papers toward interpretability and safety.
3. **Part 3 and Part 4 agree.** Between 2022 and 2024, language-model keywords spread through general ML venues (Part 3), and within language-model papers the weight moved from pretraining to reasoning and post-training (this part).
4. **Next.** Part 5 turns to compute, the resource that limits which of these themes an academic group can work on.

## References

- [OpenAlex documentation](https://docs.openalex.org/) (works, filters, group-by).
- [Topics, Authors, and Institutions in Large Language Model Research: Trends from 17K arXiv Papers](https://arxiv.org/abs/2307.10700), 2023.
