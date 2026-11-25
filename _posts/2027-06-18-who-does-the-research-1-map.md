---
title: "Who Does the Research, Part 1: Count Papers, Citations or Models and the Answer Changes"
date: 2027-06-18 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, research landscape, bibliometrics, academia, industry, who does the research]
description: "Part 1 of 6 on where large language model research happens. Universities sign about three LLM papers in four, yet nearly all notable language models come from industry. The split depends on what is counted: papers, citations or models."
---

## What this post finds

Asking who does LLM research has three different answers, because papers, citations and trained models are three different things to count.

- **Papers: universities dominate.** In OpenAlex, about 75 to 77 percent of LLM papers in 2025 have at least one university author, against about 10 to 13 percent with a company author. The result holds under three definitions of an LLM paper.
- **Citations: industry is over-represented.** An analysis of [16,979 LLM-related arXiv papers](https://arxiv.org/abs/2307.10700) found that 32 percent had an industry affiliation and 85 percent an academic one, yet 30 of the 50 most-cited 2023 papers had an industry affiliation (60 percent, against 27 percent of 2023 papers overall).
- **Models: industry is almost the whole picture.** In the [Epoch AI notable-models dataset](https://epoch.ai/data/ai-models), 97 percent of language-domain models in 2025 involve a company, and only 2 percent are academia-only. The [AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report/research-and-development) reports the same pattern: nearly 90 percent of notable models in 2024 came from industry, up from 60 percent in 2023, while academia remained the largest producer of top-100 cited publications.
- **Big Tech's share of papers fell in 2023.** In the 17K-paper study, Google, Microsoft, Amazon and Meta together fell from 19.3 percent of LLM papers before 2023 to 13.0 percent in 2023, the four largest drops among 321 coded institutions.

![Two panels. Left: share of LLM papers with at least one university author (about 75 to 77 percent in 2025) and at least one company author (about 10 to 13 percent in 2025), under three paper definitions, 2022 to 2025. Right: Epoch AI notable language models by organisation type, 2018 to 2025; industry only grows from 54 percent to 93 percent, and academia only falls to 2 percent in 2025]({{ site.baseurl }}/assets/pic/landscape-1-map.png){: width="900" }
_Figure: left, share of LLM papers by author affiliation type in OpenAlex under three definitions; right, composition of notable language models by organisation type in the Epoch AI dataset. Computed here from public data._

## What the sources do and do not show

Every number above depends on a counting rule, and the sources disagree about the rules.

- **Affiliation rule.** The OpenAlex counts here use "at least one author from an institution of this type", so a joint paper counts for both sides and the shares add to more than 100 percent. A rule of "first author" or "all authors" would give different shares. The 2026 analysis of [cs.AI preprints](https://arxiv.org/abs/2602.03969) instead assigns each paper to one of academic-only, industry-only, mixed or unknown, using an LLM to classify affiliations (reported accuracy of 86 to 93 percent on a 100-paper check); in 2021, 4,021 of its 12,520 papers stayed unclassified.
- **Topic rule.** An LLM paper is defined here by a phrase in the title or abstract. The abbreviation LLM, the word "GPT" or a model name would add papers, and phrase search also captures papers that apply a model to medicine, law or education. The check below varies the phrase and the field.
- **Type labels.** In OpenAlex, "education" covers any university or college author and "company" covers any commercial affiliation, not only research labs. OpenAlex indexing of the newest year may be incomplete, which affects counts more than shares, and the 2023 to 2025 trend in the company share was not tested against that.
- **Missing papers.** Frontier labs increasingly describe models in technical reports and blog posts. Whether these appear in OpenAlex was not checked here, so the company share of papers may understate company activity.
- **Epoch AI selection.** A model is "notable" if it advanced the state of the art, gathered many citations, or reached a large user base, among other criteria. That favours large released systems, which are mostly built by companies, so the model count measures something different from research output. Models with no organisation label were dropped here, and the "other" bar covers non-industry, non-academic labels.
- **Scope.** The 17K-paper study covers 2018 to 2023 arXiv papers; later years rest on the OpenAlex and Epoch counts alone.

## What the checks show

The script pulls yearly OpenAlex counts under three definitions of an LLM paper and tabulates Epoch AI language-domain models by organisation type.

```python
import csv, json, urllib.parse, urllib.request, collections
OA = "https://api.openalex.org/works?filter="
DEFS = {  # three definitions of "LLM paper", to see how much the answer depends on the choice
 "phrase, all fields": '"large language model"|"large language models"|LLM|"language models"',
 "strict, all fields": '"large language model"|"large language models"',
 "strict, computer science": '"large language model"|"large language models"',
}
def count(defn, year, group):
    f = "title_and_abstract.search:" + DEFS[defn] + f",publication_year:{year}"
    if defn.endswith("computer science"): f += ",primary_topic.field.id:17"
    url = OA + urllib.parse.quote(f, safe=":,|\"") + "&group_by=" + group + "&per-page=200"
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})))
    return d["meta"]["count"], {g["key_display_name"]: g["count"] for g in d["group_by"]}
print("Share of LLM papers with at least one author affiliation of each type (a paper can count twice)")
for defn in DEFS:
    for y in (2022, 2023, 2024, 2025):
        n, g = count(defn, y, "authorships.institutions.type")
        print(f"{defn:26s} {y} n={n:6d} education {100*g.get('education',0)/n:5.1f}%  company {100*g.get('company',0)/n:5.1f}%  facility {100*g.get('facility',0)/n:5.1f}%")
print("\nEpoch AI notable models, language domain, by organisation type")
rows = list(csv.DictReader(open("epoch.csv", encoding="utf-8")))
by = collections.defaultdict(collections.Counter)
for r in rows:
    y = r["Publication date"][:4]
    s = {x.strip() for x in r["Organization categorization"].split(",") if x.strip()}
    if "Language" not in r["Domain"] or not s or not y.isdigit() or not 2018 <= int(y) <= 2025: continue
    by[int(y)]["academia only" if s == {"Academia"} else "with industry" if "Industry" in s else "other"] += 1
for y in sorted(by):
    t = sum(by[y].values()); print(y, f"n={t:3d}", f"with industry {100*by[y]['with industry']/t:3.0f}%", f"academia only {100*by[y]['academia only']/t:3.0f}%")

```

Output:

```
Share of LLM papers with at least one author affiliation of each type (a paper can count twice)
phrase, all fields         2022 n=  6602 education  78.7%  company  18.7%  facility  11.8%
phrase, all fields         2023 n= 18918 education  76.9%  company  13.8%  facility  10.6%
phrase, all fields         2024 n= 50549 education  75.6%  company  11.8%  facility  10.4%
phrase, all fields         2025 n=109570 education  75.2%  company  10.1%  facility   9.7%
strict, all fields         2022 n=   624 education  63.6%  company  23.6%  facility   5.8%
strict, all fields         2023 n=  9202 education  74.8%  company  14.0%  facility  10.5%
strict, all fields         2024 n= 35135 education  75.8%  company  12.5%  facility  10.4%
strict, all fields         2025 n= 77573 education  76.5%  company  10.2%  facility   9.7%
strict, computer science   2022 n=   511 education  65.2%  company  24.5%  facility   5.7%
strict, computer science   2023 n=  6057 education  73.8%  company  18.1%  facility  10.9%
strict, computer science   2024 n= 24035 education  75.2%  company  15.6%  facility  10.5%
strict, computer science   2025 n= 50921 education  77.1%  company  13.0%  facility   9.9%

Epoch AI notable models, language domain, by organisation type
2018 n= 24 with industry  83% academia only   8%
2019 n= 39 with industry  72% academia only  21%
2020 n= 29 with industry  72% academia only  21%
2021 n= 38 with industry  89% academia only   5%
2022 n= 47 with industry  79% academia only  19%
2023 n= 76 with industry  89% academia only   8%
2024 n= 78 with industry  99% academia only   1%
2025 n= 92 with industry  97% academia only   2%
```

- **The paper split is stable across definitions.** In 2025 the university share is 75.2, 76.5 and 77.1 percent, the company share 10.1, 10.2 and 13.0 percent, and the facility share (research institutes and similar) 9.7 to 9.9 percent. Only 2022 differs, because the strict phrase matched just 511 to 624 papers that year.
- **The company share of papers has fallen every year since 2022.** Under the computer-science definition it goes 24.5, 18.1, 15.6, 13.0 percent, though the paper count grew from 511 to 50,921. Part of the fall is mix: after 2022 many papers apply existing models to other fields, which adds university-side papers. The data here cannot separate mix from reduced company publishing.
- **The model split moved the other way.** Industry-involved language models rose from 72 to 83 percent in 2018 to 2020 to 97 to 99 percent in 2024 and 2025, and academia-only models fell from 21 percent (2019, 2020) to 1 to 2 percent.
- **Why both can be true.** Writing papers about models needs access to a model. Training one in 2025 needs the compute budget of a company, so academia works on and around models it often did not build. Part 5 returns to that compute gap.
- **Limits.** The yearly Epoch counts are small (24 to 92 models per year), so a one-year move of a few points is noise. The comparison of the two panels is between different units (papers versus models), not a single measurement of "research share".

## What this means for reading the landscape

1. **Name the unit.** "Industry leads" is true for notable models and false for paper counts.
2. **Read papers as breadth and models as depth.** University papers cover applications, evaluation and analysis in many fields; company models set the frontier they all build on.
3. **Treat shares as properties of a definition.** A shift of a few points between years deserves attention only if it survives a change in the definition, as the 2023 to 2025 trend does here and the 2022 values do not.
4. **Expect a different answer by topic.** Part 4 splits the same data by research theme, and Part 2 and Part 3 look at the specific industry labs and universities behind these totals.

## References

- [Topics, Authors, and Institutions in Large Language Model Research: Trends from 17K arXiv Papers](https://arxiv.org/abs/2307.10700), 2023 (revised later).
- [Structural shifts in institutional participation and collaboration within the AI arXiv preprint research ecosystem](https://arxiv.org/abs/2602.03969), 2026.
- [Epoch AI, Data on AI models](https://epoch.ai/data/ai-models).
- [Stanford HAI, AI Index Report 2025, Research and Development](https://hai.stanford.edu/ai-index/2025-ai-index-report/research-and-development).
- [OpenAlex documentation](https://docs.openalex.org/).
