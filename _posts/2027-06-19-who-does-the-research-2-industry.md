---
title: "Who Does the Research, Part 2: Industry Labs Publish Where They Choose"
date: 2027-06-19 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, research landscape, industry labs, technical reports, open weights, who does the research]
description: "Part 2 of 6 on where large language model research happens. Company-author LLM papers grew from 35 in 2021 to 6,631 in 2025, China-based companies overtook US-based ones in 2025, and five technical reports show how differently labs disclose what they build."
---

## What this post finds

Industry research on LLMs can be read through two channels: the papers that carry a company affiliation, and the technical reports that accompany model releases. They show different things.

- **Company papers grew nearly two hundredfold and moved east.** In OpenAlex, computer-science papers mentioning "large language model(s)" with at least one company author went from 35 in 2021 to 6,631 in 2025. Summing company entities, China-based ones (2,930 entries) passed US-based ones (2,067) in 2025, a 59 percent share of the two groups. In 2025 the top company entries were Alibaba (481 papers), Microsoft (431), Huawei (423), Tencent (385) and Amazon (341).
- **The best-known labs are a small part of the paper record.** In 2025, Meta has 121 papers, Google DeepMind 111 and Anthropic 18; OpenAI is outside the top 200 institutions (companies and universities together) in 2023, 2024 and 2025. Google (United States entity) grew from 122 papers in 2023 to 247 in 2025.
- **Technical reports are chosen disclosure.** The GPT-4 report states that it contains "no further details about the architecture (including model size), hardware, training compute, dataset construction, training method, or similar", and spends 44 percent of its main-body pages on capability results and 30 percent on limitations and risks. The open-weight reports spend their pages differently: OLMo 2 puts 60 percent on pretraining and architecture, DeepSeek-V3 puts 33 percent on infrastructure, Gemma 3 puts 28 percent on safety and responsibility, and Llama 3 puts 23 percent on vision and speech experiments.

![Two panels. Left: company-author entries on LLM papers in OpenAlex, 2022 to 2025, US-based against China-based company entities; China-based passes US-based in 2025 (2930 against 2067). Right: share of main-body pages by topic in five technical reports (GPT-4, Llama 3, DeepSeek-V3, Gemma 3, OLMo 2); GPT-4 spends 44 percent on results and 30 percent on safety, OLMo 2 60 percent on pretraining and architecture, DeepSeek-V3 33 percent on infrastructure]({{ site.baseurl }}/assets/pic/landscape-2-industry.png){: width="900" }
_Figure: left, company-author entries on LLM papers by country of the company entity, OpenAlex; right, share of main-body pages by topic in five technical reports. Computed here from public data._

## What the sources do and do not show

- **Company identity in OpenAlex.** OpenAlex splits one company into several entities by country (for example Meta (United States), Meta (Israel), Microsoft Research Asia (China)). The counts here are for single entities, and the China and US sums use the country shown in each entity name. Microsoft Research Asia, for example, is counted as China-based. The sums are entity-paper pairs, so a paper with two company authors from different entities counts twice, and only the top 200 institutions per year are available (OpenAlex returns 200 groups at most per query), so smaller companies are missing.
- **Company recognition.** Company entities were recognised by the country suffix OpenAlex puts in their names, which is a heuristic and not a type label; the 2023 list of the top two (Microsoft, then Google) is consistent with the [17K-paper study](https://arxiv.org/abs/2307.10700), which also found Microsoft and Google leading.
- **Missing frontier labs.** A lab that publishes mostly as blog posts and system cards is under-counted by a paper database, so the small OpenAI and Anthropic counts say little about their research activity. The check does not measure it.
- **Technical-report sample.** Five reports were chosen because their PDFs carry a table of contents with page numbers (Qwen3's PDF has none, so its report was left out, which biases the sample toward well-structured reports). They are one report per lab, differ in length (14 to 69 body pages), and cover one model generation each. Page share is not effort share, and the mapping from section titles to categories was set by hand, with the choices listed in the code.
- **Scope.** All report categories describe what was disclosed, not what each lab works on. Meta's and Google DeepMind's research agendas extend well beyond the one report each.

## What the checks show

The first script pulls OpenAlex counts per company entity and year; the second summarises them; the third reads the table of contents of each report PDF and sums page spans by category.

```python
import json, urllib.parse, subprocess, os, time
C=["curl","-sL","-A","Mozilla/5.0","--cacert",os.path.expanduser("~/.hermes/certs/certifi_plus_zscaler.pem")]
def get(u):
    for _ in range(3):
        o=subprocess.run(C+[u],capture_output=True,text=True).stdout
        try: return json.loads(o)
        except Exception: time.sleep(2)
out={}
for y in range(2021,2026):
    f='title_and_abstract.search:"large language model"|"large language models",primary_topic.field.id:17,publication_year:%d,authorships.institutions.type:company'%y
    for g in ("authorships.institutions.lineage","authorships.institutions.id"):
        d=get("https://api.openalex.org/works?filter="+urllib.parse.quote(f,safe=":,|\"")+"&group_by="+g+"&per-page=200")
        out[f"{y}|{g}"]={"total":d["meta"]["count"],"groups":[(x["key"],x["key_display_name"],x["count"]) for x in d["group_by"]]}
json.dump(out,open("oa/d2.json","w"))
```

```python
import json, re
o = json.load(open("oa/d2.json"))  # saved by the pull script above
def company_like(n): return re.search(r"\([A-Z][A-Za-z ]+\)$", n) is not None  # OpenAlex company entries carry a country suffix
print("Papers with a company author (computer science, 'large language model(s)' in title/abstract)")
for y in range(2021, 2026):
    r = o[f"{y}|authorships.institutions.id"]; G = [(n, c) for k, n, c in r["groups"] if company_like(n)]
    cn = sum(c for n, c in G if "(China)" in n); us = sum(c for n, c in G if "(United States)" in n)
    print(y, f"papers={r['total']:5d}  entity-paper pairs, China-based {cn:5d}  US-based {us:5d}  China share of the two {100*cn/(cn+us):4.0f}%")
print("\nNamed labs, papers per year (OpenAlex; blank = outside the top 200 institutions that year)")
for lab in ["Google (United States)", "Microsoft (United States)", "Meta (United States)", "Google DeepMind (United Kingdom)", "Anthropic (United States)", "Alibaba Group (China)", "Tencent (China)", "ByteDance (China)", "Zhipu AI (China)"]:
    row = [dict((x[1], x[2]) for x in o[f"{y}|authorships.institutions.id"]["groups"]).get(lab, "") for y in range(2021, 2026)]
    print(f"{lab:34s}", *[f"{str(v):>5s}" for v in row])
```

Output:

```
Papers with a company author (computer science, 'large language model(s)' in title/abstract)
2021 papers=   35  entity-paper pairs, China-based     6  US-based    29  China share of the two   17%
2022 papers=  125  entity-paper pairs, China-based    10  US-based   114  China share of the two    8%
2023 papers= 1098  entity-paper pairs, China-based   308  US-based   577  China share of the two   35%
2024 papers= 3747  entity-paper pairs, China-based  1382  US-based  1397  China share of the two   50%
2025 papers= 6631  entity-paper pairs, China-based  2930  US-based  2067  China share of the two   59%

Named labs, papers per year (OpenAlex; blank = outside the top 200 institutions that year)
Google (United States)                 6    51   122   212   247
Microsoft (United States)              5    15   179   341   431
Meta (United States)                   4     5    24    72   121
Google DeepMind (United Kingdom)             1    26    62   111
Anthropic (United States)              1                      18
Alibaba Group (China)                  2     1    52   242   481
Tencent (China)                              3    57   215   385
ByteDance (China)                      1     1    24    89   252
Zhipu AI (China)                                   5    16    31
```

```python
import pymupdf, collections, json
P="pretraining and architecture"; T="post-training"; I="infrastructure and systems"; E="evaluation and results"; S="safety and responsibility"; M="multimodal"; O="introduction, framing, related work"
REPORTS = {  # arXiv id: (label, end of main body = start page of the conclusion, {section title: category})
 "2303.08774": ("GPT-4 (OpenAI, closed)", 14, {"Introduction":O,"Scope and Limitations of this Technical Report":O,"Predictable Scaling":P,"Capabilities":E,"Limitations":S,"Risks & mitigations":S}),
 "2407.21783": ("Llama 3 (Meta, open weights)", 70, {"Introduction":O,"General Overview":O,"Pre-Training Data":P,"Model Architecture":P,"Infrastructure, Scaling, and Efficiency":I,"Training Recipe":P,"Modeling":T,"Post-training Data":T,"Capabilities":T,"Pre-trained Language Model":E,"Post-trained Language Model":E,"Human Evaluations":E,"Safety":S,"Pipeline Parallelism":I,"FP8 Quantization":I,"Vision Experiments":M,"Speech Experiments":M,"Related Work":O,"Model Architecture@56":M,"Model Architecture@64":M,"Training Recipe@65":M}),
 "2412.19437": ("DeepSeek-V3 (open weights)", 35, {"Introduction":O,"Basic Architecture":P,"Multi-Token Prediction":P,"Compute Clusters":I,"Training Framework":I,"FP8 Training":I,"Inference and Deployment":I,"Suggestions on Hardware Design":I,"Data Construction":P,"Hyper-Parameters":P,"Long Context Extension":P,"Discussion@26":P,"Discussion@33":T,"Evaluations@23":E,"Evaluations@30":E,"Supervised Fine-Tuning":T,"Reinforcement Learning":T}),
 "2503.19786": ("Gemma 3 (Google DeepMind, open weights)", 11, {"Introduction":O,"Model Architecture":P,"Vision modality":M,"Pre-training":P,"Quantization Aware Training":I,"Compute Infrastructure":I,"Instruction-Tuning":T,"LMSYS Chatbot Arena":E,"Standard benchmarks":E,"Pre-training ability probing":P,"Local:Global attention layers":P,"Enabling long context":P,"Small versus large teacher":P,"Vision encoder":M,"Memorization and Privacy":S,"Governance & Assessment":S,"Safety policies and train-time mitigations":S,"Assurance Evaluations":S,"Our approach to responsible open models":S}),
 "2501.00656": ("OLMo 2 (Ai2, fully open)", 36, {"Introduction":O,"Model Architecture":P,"Tokenizer":P,"Base Model Training Recipe":P,"Base Model Data":P,"Evaluation and Results":E,"Pretraining Stability":P,"Repeated n-Grams":P,"Model Initialization":P,"Architecture Improvements":P,"Hyperparameter Improvements":P,"Mid-training Recipe":P,"Learning rate annealing":P,"Data Curriculum: Dolmino Mix 1124":P,"Dolmino Mix 1124: High Quality Sources":P,"Dolmino Mix 1124: Math Mix":P,"Final Midtraining mix and Checkpoint Soups":P,"Post-training Pipeline":T,"Clusters":I,"Beaker":I,"Stability and Operations":I,"Maximizing hardware utilization":I,"Environmental Impact":S}),
}
res = {}
for aid, (label, end, mp) in REPORTS.items():
    toc = pymupdf.open(f"tr/{aid}.pdf").get_toc()
    ents = []
    for t, p in [(t.replace("Deep Dive: ", ""), p) for l, t, p in toc]:
        key = f"{t}@{p}" if f"{t}@{p}" in mp else t
        if key in mp and p <= end: ents.append((p, key))
    # keep the first occurrence of each title, then sort by page
    ents = list(dict.fromkeys(ents))
    pages = collections.Counter()
    for i, (p, t) in enumerate(ents):
        nxt = ents[i + 1][0] if i + 1 < len(ents) else end
        pages[mp[t]] += max(nxt - p, 0.5)
    tot = sum(pages.values()); res[label] = {"pages": tot, "share": {k: round(100 * v / tot) for k, v in pages.items()}}
    print(f"{label}: about {tot:.0f} body pages;", ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in pages.most_common()))
json.dump(res, open("oa/d2_reports.json", "w"))
```

Output:

```
GPT-4 (OpenAI, closed): about 14 body pages; evaluation and results 44%, safety and responsibility 30%, pretraining and architecture 15%, introduction, framing, related work 11%
Llama 3 (Meta, open weights): about 69 body pages; multimodal 23%, post-training 19%, evaluation and results 17%, safety and responsibility 16%, infrastructure and systems 12%, pretraining and architecture 7%, introduction, framing, related work 6%
DeepSeek-V3 (open weights): about 32 body pages; infrastructure and systems 33%, pretraining and architecture 30%, evaluation and results 19%, post-training 12%, introduction, framing, related work 6%
Gemma 3 (Google DeepMind, open weights): about 14 body pages; pretraining and architecture 38%, safety and responsibility 28%, infrastructure and systems 10%, evaluation and results 10%, multimodal 7%, introduction, framing, related work 3%, post-training 3%
OLMo 2 (Ai2, fully open): about 34 body pages; pretraining and architecture 60%, post-training 15%, infrastructure and systems 13%, evaluation and results 6%, introduction, framing, related work 3%, safety and responsibility 3%
```

- **Counts and shares moved differently.** Microsoft grew from 179 papers in 2023 to 431 in 2025 and Google from 122 to 247, yet Part 1's share data shows the company share of all LLM papers falling, because university output grew faster.
- **The growth is concentrated in a few large organisations.** The five largest company entries in 2025 account for 2,061 of the 6,631 company-affiliated papers, as counted per entry (a paper can appear under more than one).
- **The reports tell what each lab chooses to publish.** The closed report has about 2 body pages on predictable scaling and none on architecture, data or training, followed by a 60-page system card (the PDF is 100 pages, with the system card starting on page 40). Llama 3 spends 12 percent on infrastructure and 19 percent on post-training; DeepSeek-V3 spends 33 percent on infrastructure and 12 percent on post-training; OLMo 2, from a nonprofit that releases its data and code, spends 60 percent on pretraining and architecture and 15 percent on post-training.
- **Limits.** The page-span approximation has an error of about one page per section boundary, which is up to 7 points for the 14-page reports. Percentages of 10 points or less should not be read as differences.

## What this means for reading the landscape

1. **Company research is not one block.** The paper record is led by large technology companies in China and the US, not by the labs most often named in headlines.
2. **A technical report is a disclosure choice.** Comparing what each report covers is a way to see what a lab shares with the field, and what it holds back.
3. **Count and share answer different questions.** Industry output rose nearly two hundredfold while its share of the literature fell, because the rest of the field grew faster.
4. **The next parts split the other side.** Part 3 covers universities and institutes, and Part 4 splits the same literature by research theme.

## References

- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774), 2023.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024.
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786), 2025.
- [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656), 2024.
- [Topics, Authors, and Institutions in Large Language Model Research: Trends from 17K arXiv Papers](https://arxiv.org/abs/2307.10700), 2023.
- [OpenAlex documentation](https://docs.openalex.org/).
