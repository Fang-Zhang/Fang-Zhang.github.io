---
title: "Who Does the Research, Part 6: China-Author Papers Passed US-Author Papers in 2025, and Cross-Border Teams Stayed at One in Five"
date: 2027-06-23 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, research landscape, geography, collaboration, openalex, who does the research]
description: "Part 6 of 6 on where large language model research happens. OpenAlex data on author countries and co-authorship, a note on what each source can and cannot show, and a reading list of what to watch in the series' data."
---

## What this post finds

Parts 1 to 5 looked at who publishes, which themes they pick and how much compute they have. This part adds geography and collaboration, from the same OpenAlex query as Part 4: computer-science papers from 2021 to 2025 whose title or abstract mentions "large language model" or "large language models" (133, 511, 6,057, 24,035 and 50,921 papers).

- **China-author papers passed US-author papers in 2025.** The share of papers with at least one author in mainland China rose from 8.3 percent (2021) to 28.9 percent (2025), and 25.9 percent in 2024. The share with a US author peaked at 49.7 percent in 2022 and fell to 23.3 percent in 2025. In 2025 that is 14,716 papers with a mainland-China author and 11,868 with a US author. Hong Kong, which OpenAlex counts separately, adds another 2,100 papers (4.1 percent). The UK (5.1 percent), Germany (4.0 percent), India (3.9 percent) and South Korea (3.0 percent) follow.
- **Papers and models point in different directions.** The [Stanford AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report/research-and-development) reports that China leads in publication volume and citations, while the United States produced 59 notable AI models in 2025 against China's 35 (and South Korea's 8), using Epoch AI's list of notable models. Paper counts and notable-model counts measure different things, and the model list is a manual selection, not a census.
- **One paper in five has authors in two or more countries, and that share did not rise.** It was 23.3 percent in 2021 (31 of 133 papers), 15.5 percent in 2022, 21.7 percent in 2023, and 20.2 percent in both 2024 and 2025 (10,275 of 50,921 papers).
- **US-China co-authorship grew in absolute terms but shrank as a share of China's output.** Papers with authors in both countries went from 205 (2023) to 815 (2024) and 1,639 (2025). They were 17.5 percent of China-author papers in 2023 and 11.1 percent in 2025, while they rose from 9.4 percent to 13.8 percent of US-author papers. China-Singapore papers (967 in 2025) stayed near 6.6 percent of China's papers, China-UK (591) near 4 percent and China-Australia (442) near 3 percent.
- **Company-university teams are common for companies but a small share of everything.** Of the 6,631 company-author papers in 2025, 4,957 (74.8 percent) also have a university author. Those 4,957 papers are 9.7 percent of all papers, down from 18.8 percent in 2021, because total output grew faster than joint papers.

![Two panels. Left: share of large language model papers with an author in each country, 2021 to 2025; China (mainland) rises from 8.3 percent in 2021 to 28.9 in 2025 and passes the United States, which falls from 49.7 percent in 2022 to 23.3. Right: joint papers as a share of one country's output, 2023 to 2025; US-China papers fall from 17.5 to 11.1 percent of China's papers while rising from 9.4 to 13.8 percent of US papers]({{ site.baseurl }}/assets/pic/landscape-6-geography.png){: width="900" }
_Figure: left, share of LLM papers with an author in each country; right, joint papers as a share of one country's output. Computed here from OpenAlex._

## What the sources do and do not show

- **Country comes from affiliation.** OpenAlex assigns each paper the countries of its authors' institutions. A paper with authors in two countries counts for both, so country shares add to more than 100 percent. An author who moved is counted at the affiliation printed on the paper, not by nationality or training country.
- **Early years are small.** The 2021 and 2022 values rest on 133 and 511 papers, so the 49.7 percent US peak in 2022 and the 4.5 percent China dip in the same year are noisy. The crossing in 2025 rests on tens of thousands of papers.
- **Keyword selection.** A paper enters the data when its title or abstract names a large language model, as Part 4 describes. Work on language models that uses other terms is outside it.
- **Mainland China and Hong Kong are separate codes.** Merging them would change the China figure, and the data here does not say by how much, because a paper can carry both.
- **No quality filter.** Counts are papers, not citations. Citation-based comparisons answer a different question, and the AI Index reports them separately.
- **Joint papers need two affiliations to be seen.** Authors who list only one institution cannot show a cross-border link, and company technical reports mostly sit outside OpenAlex (Part 2).

## What the checks show

The script reads the counts saved from the OpenAlex queries and computes shares, joint-paper rates and the crossing year.

```python
import json
d = json.load(open("oa/d6.json")); Y = [2021, 2022, 2023, 2024, 2025]
print("LLM papers (OpenAlex computer science, title or abstract mentions 'large language model(s)'):", {y: d[f"{y}|total"] for y in Y})
print("\nShare of papers with at least one author in the country (%); a paper with authors in two countries counts for both")
top = ["CN", "US", "IN", "GB", "DE", "KR", "SG", "AU", "CA", "FR", "JP", "HK"]
print(f"{'':4s}" + "".join(f"{y:>8d}" for y in Y))
for c in top: print(f"{c:4s}" + "".join(f"{100*d[f'{y}|countries'].get(c,0)/d[f'{y}|total']:8.1f}" for y in Y))
print("\nCountries by 2025 paper count (top 12 of the 200 returned):", sorted(d["2025|countries"].items(), key=lambda x: -x[1])[:12])
print("\nShare of papers with authors in two or more countries (%):", {y: round(100*d[f'{y}|intl']/d[f'{y}|total'], 1) for y in Y})
print("Share of papers with at least one company author and at least one university author (%):", {y: round(100*d[f'{y}|ind_univ']/d[f'{y}|total'], 1) for y in Y})
print("\nPapers with authors in both countries of a pair")
print(f"{'':8s}" + "".join(f"{y:>8d}" for y in Y))
pairs = sorted({k.split('|')[2] for k in d if '|pair|' in k})
for p in pairs: print(f"{p:8s}" + "".join(f"{d[f'{y}|pair|{p}']:8d}" for y in Y))
print("\nJoint papers as a share of one side's papers, 2023 to 2025 (%)")
for a, b, side in [("CN", "US", "CN"), ("CN", "US", "US"), ("CN", "SG", "CN"), ("CN", "GB", "CN"), ("AU", "CN", "CN"), ("GB", "US", "GB"), ("GB", "US", "US")]:
    row = [100 * d[f"{y}|pair|{a}-{b}"] / d[f"{y}|countries"][side] for y in (2023, 2024, 2025)]
    print(f"{a}-{b} as share of {side} papers: " + "  ".join(f"{y} {v:4.1f}" for y, v in zip((2023, 2024, 2025), row)))
print(f"\nCompany-author papers in 2025 that also have a university author: {d['2025|ind_univ']} of 6,631 = {100*d['2025|ind_univ']/6631:.1f}% (6,631 is the company-author count from Part 2)")
print("The year in which China-author papers pass US-author papers:", [y for y in Y if d[f"{y}|countries"]["CN"] > d[f"{y}|countries"]["US"]])
```

Output:

```
LLM papers (OpenAlex computer science, title or abstract mentions 'large language model(s)'): {2021: 133, 2022: 511, 2023: 6057, 2024: 24035, 2025: 50921}

Share of papers with at least one author in the country (%); a paper with authors in two countries counts for both
        2021    2022    2023    2024    2025
CN       8.3     4.5    19.3    25.9    28.9
US      42.9    49.7    36.1    27.8    23.3
IN       5.3     2.0     3.1     3.3     3.9
GB       9.0     7.2     6.7     5.4     5.1
DE       3.0     3.5     4.8     4.4     4.0
KR       1.5     1.2     2.3     2.5     3.0
SG       1.5     1.6     3.3     3.1     3.1
AU       0.8     0.6     2.8     2.8     2.6
CA       4.5     3.9     3.7     3.3     3.0
FR       2.3     2.2     1.7     1.6     1.3
JP       0.8     0.4     2.4     2.8     2.4
HK       0.0     1.0     3.4     3.8     4.1

Countries by 2025 paper count (top 12 of the 200 returned): [('CN', 14716), ('US', 11868), ('GB', 2608), ('HK', 2100), ('DE', 2060), ('IN', 2006), ('SG', 1600), ('CA', 1539), ('KR', 1528), ('AU', 1307), ('JP', 1235), ('IT', 1085)]

Share of papers with authors in two or more countries (%): {2021: 23.3, 2022: 15.5, 2023: 21.7, 2024: 20.2, 2025: 20.2}
Share of papers with at least one company author and at least one university author (%): {2021: 18.8, 2022: 15.7, 2023: 12.5, 2024: 11.1, 2025: 9.7}

Papers with authors in both countries of a pair
            2021    2022    2023    2024    2025
AU-CN          0       0      36     213     442
CA-US          4      13      77     231     404
CN-GB          1       1      45     261     591
CN-SG          1       1      80     394     967
CN-US          3      10     205     815    1639
DE-US          1       8      73     143     272
GB-US          5      18     129     309     599
KR-US          0       2      38     108     196

Joint papers as a share of one side's papers, 2023 to 2025 (%)
CN-US as share of CN papers: 2023 17.5  2024 13.1  2025 11.1
CN-US as share of US papers: 2023  9.4  2024 12.2  2025 13.8
CN-SG as share of CN papers: 2023  6.8  2024  6.3  2025  6.6
CN-GB as share of CN papers: 2023  3.8  2024  4.2  2025  4.0
AU-CN as share of CN papers: 2023  3.1  2024  3.4  2025  3.0
GB-US as share of GB papers: 2023 31.7  2024 23.8  2025 23.0
GB-US as share of US papers: 2023  5.9  2024  4.6  2025  5.0

Company-author papers in 2025 that also have a university author: 4957 of 6,631 = 74.8% (6,631 is the company-author count from Part 2)
The year in which China-author papers pass US-author papers: [2025]
```

## What to watch in the series' data

The six parts use five public sources, and each answers one question.

1. **OpenAlex** (Parts 1, 2, 4 and 6) counts papers by institution type, theme and country. It lacks technical reports, so company activity is understated there.
2. **Technical reports** (Part 2) show what labs choose to emphasise, in page counts.
3. **Conference data** (Part 3) gives affiliation-level detail for three general machine-learning venues, up to 2024.
4. **Epoch AI** (Parts 1 and 5) records notable models and training compute, a selection with estimated numbers.
5. **Surveys** (Part 5) give researchers' own account of constraints, here from 2022.

Three trends recur across the parts, and each can be re-checked with the scripts in the posts:

- **Themes moved from building models to using them.** Pretraining and scaling fell from 24.8 percent to 5.5 percent of language-model papers between 2021 and 2025, while reasoning, post-training and agents grew (Part 4).
- **Compute concentrates the largest models.** A 64-GPU month covers 14 percent of notable language models from 2023 to 2025 (Part 5), and the AI Index 2026 notes that several frontier labs no longer disclose parameters or training details, so compute estimates will carry more of the measurement burden.
- **The paper map and the model map differ.** In the data here, China-author papers lead in count, while the AI Index counts more notable models from the United States. Joint authorship across borders stayed at about one paper in five.

## References

- [OpenAlex works filters](https://docs.openalex.org/api-entities/works/filter-works), including the country, institution type and distinct-country counts used here.
- [Stanford AI Index 2026, Research and Development](https://hai.stanford.edu/ai-index/2026-ai-index-report/research-and-development).
- [Epoch AI, notable AI models database](https://epoch.ai/data/ai-models).
