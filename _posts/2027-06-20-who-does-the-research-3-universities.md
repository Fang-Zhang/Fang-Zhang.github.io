---
title: "Who Does the Research, Part 3: Universities Moved Together, Their Side Interests Did Not"
date: 2027-06-20 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, research landscape, universities, research institutes, iclr, icml, neurips, who does the research]
description: "Part 3 of 6 on where large language model research happens. The largest universities at ICLR, ICML and NeurIPS all moved toward language models between 2022 and 2024, while their mix of vision, reinforcement learning, theory and safety stayed distinct."
---

## What this post finds

Part 2 looked at companies. This part counts universities and institutes, using the one public dataset that lists every author affiliation at three machine-learning conferences: [ICML, NeurIPS and ICLR, 2017 to 2024](https://github.com/martenlienen/icml-neurips-iclr-dataset). The dataset has 9,468 papers for 2024 and 5,229 for 2022. Affiliations are free text, so each institution is a hand-written pattern, and 54 percent of author entries match one of the 53 patterns used.

- **The whole field moved, and the big universities moved with it.** Titles with a language-model keyword grew from 6.2 percent of all papers in 2022 to 19.2 percent in 2024. A narrower keyword list (language model, LLM, GPT, in-context, chain-of-thought) gives 1.8 to 13.6 percent, so the rise does not depend on the list. Each of the 14 largest universities and institutes rose from 3.6 to 10.6 percent in 2022 to 20.5 to 28.7 percent in 2024. MIT rose the most in relative terms (3.6 to 20.5), Mila (Montreal) is the lowest among the top 20 in 2024 (15.7 percent).
- **Tsinghua leads the university ranking.** By papers with at least one author from the institution in 2024, Tsinghua is first among universities with 723, ahead of Peking University (511), Carnegie Mellon (487), MIT (469), Stanford (424) and Berkeley (385). Of the 14 largest universities and institutes, 8 are in China (counting Hong Kong), 4 in the US, one in Singapore and one in the UK. Companies in the same ranking: Google including DeepMind 727, Microsoft 375, Meta 311.
- **Beyond language models, the mix differs by institution and by region.** Pooled over papers with at least one author from a university or institute, vision and multimodal titles are 25.4 percent of the China group, 13.7 percent of the US group and 14.0 percent of the UK and Europe group. Theory and optimization titles go the other way: 8.4 percent (China), 11.0 percent (US), 12.6 percent (UK and Europe). Of the 160 institution-theme cells for the 20 largest universities and institutes, 57 fall outside the 95 percent interval of the all-paper rate, where chance alone would give about 8.

![Two panels. Left: for the 14 largest universities and institutes at ICLR, ICML and NeurIPS in 2024, the share of papers with a language-model title keyword rises from 3.6 to 9.8 percent in 2022 to 20.5 to 28.7 percent in 2024, against all-paper rates of 6.2 and 19.2 percent. Right: heatmap of theme shares in 2024; vision and multimodal is highest at the Chinese Academy of Sciences (37 percent) and USTC (33 percent) and lowest at MIT (10 percent)]({{ site.baseurl }}/assets/pic/landscape-3-universities.png){: width="900" }
_Figure: left, language-model title share 2022 to 2024; right, theme mix of 2024 papers at the 14 largest universities and institutes. Computed here from the public dataset._

## Which institutions stand out

The cells that differ most from the all-paper mix (2024, 95 percent interval used as the test):

- **Vision and multimodal.** Chinese Academy of Sciences 37.1 percent, USTC 32.7, Shanghai Jiao Tong 28.7, National University of Singapore 27.5, Zhejiang 27.2, Tsinghua 26.6 and Nanjing 26.6, against 17.4 overall and 9.8 at MIT.
- **Reinforcement learning.** UIUC 13.6 percent, Berkeley 13.2 and Mila 13.1, against 8.4 overall; Stanford 4.7 is below.
- **Theory and optimization.** NYU 15.0 percent against 10.8 overall; Peking University 5.7 is below.
- **Trust, safety and privacy.** UIUC 16.1 percent against 8.8 overall; Shanghai Jiao Tong 3.7 and Mila 3.0 are below.
- **Science (biology, chemistry, physics).** Mila 8.1 percent against 2.7 overall, MIT 5.3, Tsinghua 5.5 and Zhejiang 5.4.
- **Diffusion and generative.** KAIST 18.3 percent against 9.2 overall.
- **Companies for comparison.** Microsoft has the highest language-model share of the top 20 (35.2 percent), Meta 28.6, Google 24.3. Anthropic has 26 papers and OpenAI 53 in the 2024 data, which is too few to draw a mix from (16 of Anthropic's 26 have a language-model title keyword).

## What the sources do and do not show

- **Three venues only.** ICML, NeurIPS and ICLR are general machine-learning conferences. Computer vision venues (CVPR, ICCV), language venues (ACL, EMNLP) and robotics venues are not in the data, so "vision share" is vision work submitted to a general ML venue, and the dataset cannot show where an institution's language-model work is concentrated.
- **Titles, not abstracts.** Themes come from keywords in the title. "Transformer", "reasoning" and "prompt" count as language-model words, "image" and "video" as vision, and some titles match two themes or none. The mixed lists are in the code.
- **Affiliation matching.** Patterns are hand-written. Some matches are looser than the institution name implies (the Berkeley pattern also catches Lawrence Berkeley National Laboratory, and the Mila pattern catches any Montreal affiliation). "At least one author" counting means a joint paper counts for each institution on it.
- **Coverage.** The dataset ends at 2024 because the conference site stopped listing affiliations in 2025, per its README. A conference acceptance is a selected sample, so institutions that publish mostly elsewhere are under-counted.
- **Noise test.** The Wilson interval used for each cell compares an institution to the all-paper rate, which includes that institution's own papers. Differences of a few points in the heatmap are within noise for 200 to 300 papers.

## What the checks show

The script downloads the dataset, matches affiliations, tags titles by theme, and tests each institution-theme cell.

```python
import csv, re, collections, json, os, urllib.request
URL = "https://raw.githubusercontent.com/martenlienen/icml-neurips-iclr-dataset/master/papers.csv"
if not os.path.exists("conf/papers.csv"): urllib.request.urlretrieve(URL, "conf/papers.csv")
INST = {  # canonical name: (type, region, regex on the lower-cased affiliation string)
 "Tsinghua University": ("univ","China",r"tsinghua"), "Peking University": ("univ","China",r"peking univ|beijing university"),
 "Zhejiang University": ("univ","China",r"zhejiang univ"), "Shanghai Jiao Tong University": ("univ","China",r"shanghai ji?ao ?tong"),
 "Univ. of Science and Technology of China": ("univ","China",r"science and technology of china|\bustc\b"), "Fudan University": ("univ","China",r"fudan"),
 "Nanjing University": ("univ","China",r"nanjing univ"), "Chinese Academy of Sciences": ("inst","China",r"chinese academy of sciences"),
 "Shanghai AI Laboratory": ("inst","China",r"shanghai ai lab|shanghai artificial intelligence lab"),
 "Stanford University": ("univ","US",r"stanford"), "Carnegie Mellon University": ("univ","US",r"carnegie mellon|\bcmu\b"),
 "MIT": ("univ","US",r"massachusetts institute of technology|^mit$|\bmit\b"), "UC Berkeley": ("univ","US",r"berkeley"),
 "Princeton University": ("univ","US",r"princeton"), "Harvard University": ("univ","US",r"harvard"), "Columbia University": ("univ","US",r"columbia univ"),
 "New York University": ("univ","US",r"new york univ|\bnyu\b"), "University of Washington": ("univ","US",r"university of washington"),
 "Cornell University": ("univ","US",r"cornell"), "Georgia Institute of Technology": ("univ","US",r"georgia institute of technology|georgia tech"),
 "UIUC": ("univ","US",r"illinois.*urbana|urbana.champaign|\buiuc\b"), "UT Austin": ("univ","US",r"university of texas at austin|ut austin"),
 "UCLA": ("univ","US",r"university of california, los angeles|\bucla\b"), "UC San Diego": ("univ","US",r"san diego"),
 "University of Oxford": ("univ","UK/Europe",r"oxford"), "University of Cambridge": ("univ","UK/Europe",r"university of cambridge"),
 "ETH Zurich": ("univ","UK/Europe",r"eth z|swiss federal institute of technology, zurich"), "EPFL": ("univ","UK/Europe",r"epfl|polytechnique f.d.rale de lausanne"),
 "University College London": ("univ","UK/Europe",r"university college london|\bucl\b"), "Imperial College London": ("univ","UK/Europe",r"imperial college"),
 "National University of Singapore": ("univ","Asia-Pacific",r"national university of singapore|\bnus\b"), "Nanyang Technological University": ("univ","Asia-Pacific",r"nanyang"),
 "KAIST": ("univ","Asia-Pacific",r"kaist|korea advanced institute"), "Seoul National University": ("univ","Asia-Pacific",r"seoul national"),
 "Chinese University of Hong Kong": ("univ","China",r"chinese university of hong kong|\bcuhk\b"), "HKUST": ("univ","China",r"hong kong university of science"),
 "University of Toronto": ("univ","Canada",r"university of toronto"), "Mila / Univ. of Montreal": ("univ","Canada",r"\bmila\b|montreal|montr.al"), "Vector Institute": ("inst","Canada",r"vector institute"),
 "Google (incl. DeepMind)": ("company","US",r"google|deepmind"), "Microsoft": ("company","US",r"microsoft"), "Meta": ("company","US",r"\bmeta\b|facebook"),
 "Amazon": ("company","US",r"amazon"), "NVIDIA": ("company","US",r"nvidia"), "IBM": ("company","US",r"\bibm\b"), "Apple": ("company","US",r"\bapple\b"),
 "Alibaba": ("company","China",r"alibaba|\bdamo\b"), "Tencent": ("company","China",r"tencent"), "ByteDance": ("company","China",r"bytedance"), "Huawei": ("company","China",r"huawei"),
 "Samsung": ("company","Asia-Pacific",r"samsung"), "OpenAI": ("company","US",r"openai"), "Anthropic": ("company","US",r"anthropic"),
}
THEMES = {
 "language models": r"language model|\bllms?\b|\bgpt|in-context|instruction[- ]tun|chain[- ]of[- ]thought|prompt|reasoning|transformer",
 "diffusion and generative": r"diffusion|generative|text-to-image|image generation|video generation|\bgans?\b|flow matching",
 "vision and multimodal": r"image|vision|visual|video|\b3d\b|detection|segmentation|multimodal|multi-modal|\bclip\b",
 "reinforcement learning": r"reinforcement|\brl\b|policy|reward|bandit|markov decision",
 "theory and optimization": r"convergence|\bbounds?\b|regret|sample complexity|minimax|theor|stochastic|gradient|optimization",
 "trust, safety, privacy": r"privacy|private|fairness|\bfair\b|robust|adversarial|safety|backdoor|attack|jailbreak",
 "graphs": r"\bgraphs?\b|\bgnn",
 "science (bio, chem, physics)": r"protein|molecul|drug|physic|climate|weather|biolog|chemi|\bpde\b",
}
rows = list(csv.DictReader(open("conf/papers.csv", encoding="utf-8")))
rx = {k: re.compile(v[2]) for k, v in INST.items()}
papers = collections.defaultdict(lambda: [set(), None])  # (conf, year, title) -> [institutions, year]
n_auth = n_hit = 0
for r in rows:
    y = int(r["Year"])
    if y < 2019: continue
    key = (r["Conference"], y, r["Title"]); a = r["Affiliation"].lower(); papers[key][1] = y; n_auth += 1
    hits = [k for k, p in rx.items() if p.search(a)]
    n_hit += bool(hits); papers[key][0].update(hits)
print(f"{len(papers)} papers 2019-2024; {100*n_hit/n_auth:.0f}% of author entries matched one of the {len(INST)} listed institutions")
theme_rx = {k: re.compile(v) for k, v in THEMES.items()}
def themes(t): return {k for k, p in theme_rx.items() if p.search(t.lower())}
out = {}
for y in (2022, 2024):
    P = [(k, v[0]) for k, v in papers.items() if v[1] == y]
    cnt = collections.Counter(i for _, s in P for i in s); tot = len(P)
    rows_y = {}
    for i in cnt:
        ts = [k[2] for k, s in P if i in s]; c = collections.Counter(t for x in ts for t in themes(x))
        rows_y[i] = {"papers": cnt[i], "type": INST[i][0], "region": INST[i][1], "themes": {t: round(100 * c[t] / len(ts), 1) for t in THEMES}}
    allc = collections.Counter(t for k, _ in P for t in themes(k[2]))
    out[y] = {"total": tot, "inst": rows_y, "all": {t: round(100 * allc[t] / tot, 1) for t in THEMES}}
json.dump(out, open("oa/d3.json", "w"))
for y in (2022, 2024):
    print(f"\n{y}: {out[y]['total']} papers at ICLR, ICML and NeurIPS; share of all titles by theme:", out[y]["all"])
top = sorted(out[2024]["inst"].items(), key=lambda x: -x[1]["papers"])
print("\nTop 20 by number of 2024 papers with at least one author from the institution")
for i, v in top[:20]: print(f"{i:42s} {v['type']:8s} {v['region']:13s} {v['papers']:4d}  language-model titles 2022 {out[2022]['inst'].get(i,{}).get('themes',{}).get('language models',0):4.1f}% -> 2024 {v['themes']['language models']:4.1f}%")

# ---- region view and noise check ----
import math
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
print("\nPooled theme shares of unique 2024 papers with at least one university or institute author, by region (%)")
P24 = [(k, v[0]) for k, v in papers.items() if v[1] == 2024]
for reg in ("China", "US", "UK/Europe"):
    ps = [k[2] for k, s in P24 if any(INST[i][1] == reg and INST[i][0] in ("univ", "inst") for i in s)]
    c = collections.Counter(t for x in ps for t in themes(x))
    print(f"{reg:10s} n={len(ps):4d}", "  ".join(f"{t.split(' ')[0]} {100*c[t]/len(ps):4.1f}" for t in THEMES))
print("\nInstitution-theme cells outside the 95% Wilson interval of the all-paper rate (top 20 universities and institutes by 2024 papers)")
rows24 = out[2024]["inst"]; allr = out[2024]["all"]
tops = [i for i, v in sorted(rows24.items(), key=lambda x: -x[1]["papers"]) if v["type"] != "company"][:20]
n_cells = n_out = 0
for i in tops:
    n = rows24[i]["papers"]
    for t in THEMES:
        k = round(rows24[i]["themes"][t] * n / 100); lo, hi = wilson(k, n); n_cells += 1
        if not lo <= allr[t] / 100 <= hi:
            n_out += 1; print(f"  {i:36s} {t:28s} {rows24[i]['themes'][t]:5.1f}% vs {allr[t]:5.1f}% overall (n={n})")
print(f"{n_out} of {n_cells} cells outside the interval; about {0.05*n_cells:.0f} would be expected by chance if every institution matched the overall mix")

print("\nSensitivity of the language-model theme to its keyword list (share of all titles, %)")
narrow = re.compile(r"language model|\bllms?\b|\bgpt|in-context|chain[- ]of[- ]thought")
for y in (2019, 2022, 2024):
    ts = [k[2].lower() for k in papers if k[1] == y]
    print(y, f"n={len(ts):5d}", f"narrow list {100*sum(bool(narrow.search(t)) for t in ts)/len(ts):4.1f}", f"broad list {100*sum(bool(theme_rx['language models'].search(t)) for t in ts)/len(ts):4.1f}")
```

Output:

```
31790 papers 2019-2024; 54% of author entries matched one of the 53 listed institutions

2022: 5229 papers at ICLR, ICML and NeurIPS; share of all titles by theme: {'language models': 6.2, 'diffusion and generative': 3.9, 'vision and multimodal': 11.1, 'reinforcement learning': 11.0, 'theory and optimization': 14.2, 'trust, safety, privacy': 11.2, 'graphs': 5.7, 'science (bio, chem, physics)': 2.2}

2024: 9468 papers at ICLR, ICML and NeurIPS; share of all titles by theme: {'language models': 19.2, 'diffusion and generative': 9.2, 'vision and multimodal': 17.4, 'reinforcement learning': 8.4, 'theory and optimization': 10.8, 'trust, safety, privacy': 8.8, 'graphs': 5.0, 'science (bio, chem, physics)': 2.7}

Top 20 by number of 2024 papers with at least one author from the institution
Google (incl. DeepMind)                    company  US             727  language-model titles 2022 12.5% -> 2024 24.3%
Tsinghua University                        univ     China          723  language-model titles 2022  9.6% -> 2024 25.4%
Peking University                          univ     China          511  language-model titles 2022  6.9% -> 2024 22.5%
Carnegie Mellon University                 univ     US             487  language-model titles 2022  6.6% -> 2024 23.8%
MIT                                        univ     US             469  language-model titles 2022  3.6% -> 2024 20.5%
Stanford University                        univ     US             424  language-model titles 2022  9.0% -> 2024 26.4%
UC Berkeley                                univ     US             385  language-model titles 2022  7.3% -> 2024 24.9%
Microsoft                                  company  US             375  language-model titles 2022 13.9% -> 2024 35.2%
Zhejiang University                        univ     China          372  language-model titles 2022  9.8% -> 2024 28.2%
Shanghai Jiao Tong University              univ     China          352  language-model titles 2022  4.0% -> 2024 24.4%
Chinese Academy of Sciences                inst     China          350  language-model titles 2022  6.7% -> 2024 25.4%
Chinese University of Hong Kong            univ     China          313  language-model titles 2022  6.5% -> 2024 28.1%
Meta                                       company  US             311  language-model titles 2022 10.2% -> 2024 28.6%
National University of Singapore           univ     Asia-Pacific   298  language-model titles 2022  5.6% -> 2024 24.5%
Univ. of Science and Technology of China   univ     China          297  language-model titles 2022 10.6% -> 2024 24.6%
University of Oxford                       univ     UK/Europe      245  language-model titles 2022  5.0% -> 2024 22.0%
HKUST                                      univ     China          244  language-model titles 2022  7.2% -> 2024 28.7%
Mila / Univ. of Montreal                   univ     Canada         236  language-model titles 2022  5.6% -> 2024 15.7%
UIUC                                       univ     US             236  language-model titles 2022  8.8% -> 2024 27.1%
New York University                        univ     US             233  language-model titles 2022  3.2% -> 2024 21.9%

Pooled theme shares of unique 2024 papers with at least one university or institute author, by region (%)
China      n=2535 language 23.7  diffusion 11.4  vision 25.4  reinforcement  7.6  theory  8.4  trust,  7.5  graphs  6.0  science  3.6
US         n=2902 language 22.9  diffusion  8.9  vision 13.7  reinforcement  9.3  theory 11.0  trust,  8.4  graphs  3.8  science  3.1
UK/Europe  n= 906 language 20.8  diffusion  7.6  vision 14.0  reinforcement  7.1  theory 12.6  trust,  8.6  graphs  4.5  science  2.9

Institution-theme cells outside the 95% Wilson interval of the all-paper rate (top 20 universities and institutes by 2024 papers)
  [57 cell lines omitted in this post]
57 of 160 cells outside the interval; about 8 would be expected by chance if every institution matched the overall mix

Sensitivity of the language-model theme to its keyword list (share of all titles, %)
2019 n= 2701 narrow list  0.3 broad list  1.3
2022 n= 5229 narrow list  1.8 broad list  6.2
2024 n= 9468 narrow list 13.6 broad list 19.2
```

## What this means for reading the landscape

1. **Direction of travel is shared.** Between 2022 and 2024, the 14 largest academic institutions at these venues raised their language-model title share by a factor of 2.3 (USTC) to 5.7 (MIT); none stayed outside the shift.
2. **The distinct part is the second-largest theme.** Institutions that look alike on language models differ by 10 to 20 points on vision, reinforcement learning, theory or safety, so the mix of other work is what separates them.
3. **Region differences exist but are one-venue differences.** The vision-heavy profile of Chinese institutions at these three venues is measured on three venues; it is not a claim about all work.
4. **Next.** Part 4 splits the literature by research theme over time, and Part 5 looks at compute, which constrains what an academic group can do.

## References

- [ICML, NeurIPS and ICLR paper and affiliation dataset](https://github.com/martenlienen/icml-neurips-iclr-dataset), 2006 to 2024.
- [Topics, Authors, and Institutions in Large Language Model Research: Trends from 17K arXiv Papers](https://arxiv.org/abs/2307.10700), 2023.
