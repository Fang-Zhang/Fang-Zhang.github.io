---
title: "LLMs on the Farm, Part 4: After Launch, the Knowledge Base Is the Product"
date: 2027-06-03 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, agriculture, maintenance, evaluation, rag, monitoring, llms on the farm]
description: "Part 4 of 6 on large language models in agriculture. After launch, deployed farm advisory services spend their effort on content curation, evaluation sets and usage logs, not on the model. Published numbers show where unanswered questions come from, why overlap metrics mislead, and a CPU check shows how an updated document loses to its old version."
---

## What this post finds

Parts 1 to 3 picked a model, a way to feed it farm knowledge, and a pipeline around it. Part 4 asks what keeps the system correct after launch, using the same two public sources, the [AIEP technical learnings](https://arxiv.org/abs/2601.11537) and the [Farmer.Chat paper](https://arxiv.org/abs/2409.08916), plus an [IFPRI post on evaluating advisory systems](https://www.ifpri.org/blog/beyond-the-model-evaluating-ai-agricultural-advisory-systems-so-they-work-in-the-field).

- **Content work does not end.** The AIEP report says identifying, curating, validating and maintaining content were continuous processes, not one-off tasks. Even openly available data had to be reviewed with qualified staff to avoid serving outdated, wrong or irrelevant information, and cohorts duplicated much of that work.
- **The unanswered-question log tells the team what to write next.** Farmer.Chat answered close to 75 percent of questions; of the rest, 66 percent were content gaps, 23 percent out of scope and 11 percent about unsupported crops. Farmers' thumbs and star ratings feed content prioritisation.
- **Evaluation sets are small and curated.** Digital Green drew 5,000 vetted question-answer pairs for Bihar from more than 2.2 million farmer queries. The AIEP team built a set of 114 agronomist-checked questions from 28,896 user questions, and did not release it because of data-leakage concerns.
- **Model-quality scores saturate before the system is good.** Agronomists rated the tested models very highly with no statistically significant differences between them, so small differences need more precise evaluation data. Word-overlap and embedding metrics gave a mixed picture against the agronomists; a model used as judge came closest.
- **Evaluation has three levels, not one.** The IFPRI post, summarising a practitioner webinar, separates the model (reasoning and factual accuracy), the system (usefulness and user experience in the field) and the process (governance, risk, the role of human experts, and who benefits).
- **Two small checks on invented text** show why. Word overlap scored a wrong-dose answer above a correct paraphrase in 3 of 3 cases, and an updated document still ranked below its own old version until the old one was replaced.

![Two panels. Left: ROUGE-1 scores against a reference answer in three invented cases; a correct paraphrase scores 0.32, 0.20 and 0.50, while a harmful answer with the wrong dose but copied wording scores 0.89, 0.79 and 0.83. Right: retrieval score for a question about a banned product; with only the old text it scores 0.85, with the new text added and the old left in the old text scores 0.78 and the new 0.35, and with the old text replaced the new text scores 0.57]({{ site.baseurl }}/assets/pic/farm-4-maintenance.png){: width="900" }
_Figure: both panels are computed by the script at the end of this post, on invented text. They illustrate a mechanism and are not measurements of any deployed system._

## The maintenance loop the sources describe

**1. Read the logs.** Farmer.Chat's paper reports user-side metrics: percentage of queries answered, response time, readability of answers (Flesch-Kincaid scores mostly between 60 and 80), the most discussed topics (pests and diseases first, then soil management and storage), and a model-scored clarity rating of 20,000 prompts. Two findings changed the product. Many users asked only one question per topic, so in February 2024 the service added clickable follow-up questions, which accounted for over 45 percent of interactions at peak. And 35 percent of users, the power users, asked nearly 80 percent of all queries, so averages describe the heavy users more than the rest.

**2. Fix the knowledge base, not the model.** The AIEP report treats the corpus as a standing obligation: partners had to find sources, sometimes sign data-sharing agreements, and have qualified staff check relevance. It lists agronomic material, crop and soil maps, market prices and weather as the content a service depends on, and recommends planning access to them first. Digital Green's engineering write-up reached the same place from the other side: after fixing retrieval, it found that about 11 percent of unanswered questions came from users asking about a different crop than the one selected, which is a product fix, and that a faithful system can only answer what the content covers.

**3. Keep an evaluation set that is not public.** The AIEP golden set combines agronomist-selected questions, common questions mined from real user logs, and reference answers written by agronomists. It also includes multiple-choice questions, described as less representative but much easier to score. Many agricultural answers depend on context the question does not state (what grew on the plot before, how much fertiliser was already applied, which inputs the farmer can afford), so a question with a single context-free correct answer is hard to write. Digital Green defines maturity tiers for such sets: 10,000 to 20,000 pairs for baseline quality in core crops and languages, 50,000 to 100,000 for broader themes, and 200,000 or more for specialised, real-time learning.

**4. Check the checker.** In the AIEP study, agronomists scored model answers on factual correctness, comprehensiveness, relevance, actionability and intelligibility, and those scores were treated as ground truth. Automated metrics were then correlated with them. N-gram and embedding similarity gave a mixed picture, and only the model-as-judge approach showed at least low to medium correlations on more than one metric. The report concludes that a judge model can substitute for expert evaluation in a broad assessment, while finer differences between models need better data. Separately, Digital Green's write-up describes an automatic evaluation pipeline that was benchmarked against human evaluation before it was relied on, and the [Ragas](https://arxiv.org/abs/2309.15217) framework, which Digital Green's write-up also mentions, proposes reference-free metrics for the retrieval and generation sides of a pipeline.

**5. Measure who is served.** The IFPRI post argues that equity should be measured, not assumed: indicators for usability, access and reach, trust, adoption and agency, each split by gender and social group. One of the post's examples is a partner whose users clicked the text label instead of the icon, so features were not discovered; advice that is correct cannot help if the interface hides it, and accuracy metrics do not see this. The AIEP report adds that it set standard indicators across cohorts so that the teams could be compared, and that new models are hard to judge for low-resourced languages and agriculture because public leaderboards say little about them.

## Two checks on invented text

The sources describe the practice but not the failure mechanisms in detail, so the script below reproduces two of them on invented text, with the standard library only.

**Check 1** scores three pairs of answers against a reference answer, using word overlap (Jaccard similarity, ROUGE-1 and ROUGE-2). In each pair one answer is a correct paraphrase with different wording, and the other copies the reference's wording but changes the dose or the interval by a factor of 10.

**Check 2** builds a five-passage knowledge base with a simple term-weighted retrieval, adds a new passage that withdraws a product, and ranks passages for the question "Can product X be used on maize and how many litres per hectare?" before the update, after adding the new text and leaving the old, and after replacing the old.

```python
# Two maintenance checks on invented farm-advice text (no model, standard library only).
import math, re
from collections import Counter

def words(s): return re.findall(r"[a-z0-9]+", s.lower())
def jaccard(a, b):
    A, B = set(words(a)), set(words(b)); return len(A & B) / len(A | B)
def rouge_n(ref, cand, n):
    g = lambda s: Counter(zip(*[words(s)[i:] for i in range(n)]))
    R, C = g(ref), g(cand); ov = sum((R & C).values())
    if not ov: return 0.0
    p, r = ov / sum(C.values()), ov / sum(R.values()); return 2 * p * r / (p + r)

print("=== Check 1: does word overlap with the reference score a harmful answer below a correct paraphrase? ===")
cases = [
 ("Maize, nitrogen",
  "Apply 50 kg of nitrogen per hectare at planting and top-dress with another 50 kg six weeks later.",
  "Give 100 kg N per hectare in two equal doses: half when sowing and half after about six weeks.",
  "Apply 500 kg of nitrogen per hectare at planting and top-dress with another 500 kg six weeks later."),
 ("Tomato, spray gap",
  "Wait at least 7 days between sprays and stop spraying 14 days before harvest.",
  "Leave a week between applications, and do not spray in the last two weeks before picking.",
  "Wait at least 1 day between sprays and stop spraying 2 days before harvest."),
 ("Bean seed rate",
  "Sow 40 kg of seed per hectare with rows 50 cm apart.",
  "Use about 40 kg of seed for every hectare, planted in rows half a metre apart.",
  "Sow 400 kg of seed per hectare with rows 5 cm apart."),
]
print(f"{'case':18s}{'answer':10s}{'Jaccard':>9s}{'ROUGE-1':>9s}{'ROUGE-2':>9s}")
wins = Counter()
for name, ref, good, bad in cases:
    sc = {}
    for lab, c in (("correct", good), ("harmful", bad)):
        sc[lab] = (jaccard(ref, c), rouge_n(ref, c, 1), rouge_n(ref, c, 2))
        print(f"{name:18s}{lab:10s}{sc[lab][0]:9.2f}{sc[lab][1]:9.2f}{sc[lab][2]:9.2f}")
    for i, m in enumerate(("Jaccard", "ROUGE-1", "ROUGE-2")):
        if sc["harmful"][i] > sc["correct"][i]: wins[m] += 1
print("Cases where the harmful answer scores higher than the correct paraphrase (of 3):", dict(wins))

print()
print("=== Check 2: an updated document next to its old version ===")
old = "Product X can be used on maize at planting. Apply 2 litres of product X per hectare."
new = "Product X is withdrawn and banned for maize from 2025. Do not apply product X on maize."
other = ["Maize needs 50 kg of nitrogen per hectare at planting.", "Rust is a common coffee disease; remove infected leaves.", "Store dried maize below 13 percent moisture."]
q = "Can product X be used on maize and how many litres per hectare?"
def rank(chunks, q):
    docs = [words(c[1]) for c in chunks]; N = len(docs)
    df = Counter(w for d in docs for w in set(d))
    def vec(ws): c = Counter(ws); return {w: (1 + math.log(v)) * math.log(1 + N / df[w]) for w, v in c.items() if w in df}
    qv = vec(words(q))
    out = []
    for (tag, txt), d in zip(chunks, docs):
        dv = vec(d); dot = sum(qv[w] * dv.get(w, 0) for w in qv)
        nd = math.sqrt(sum(v * v for v in dv.values())) * math.sqrt(sum(v * v for v in qv.values()))
        out.append((dot / nd if nd else 0, tag))
    return sorted(out, reverse=True)
kb = [("old", old)] + [("other", o) for o in other]
print("Before the update, ranking for the question:"); [print(f"  {s:.3f}  {t}") for s, t in rank(kb, q)[:3]]
kb2 = kb + [("NEW", new)]
print("After adding the new text and leaving the old one in:"); [print(f"  {s:.3f}  {t}") for s, t in rank(kb2, q)[:3]]
kb3 = [("NEW", new)] + [("other", o) for o in other]
print("After replacing the old text (superseded by document id):"); [print(f"  {s:.3f}  {t}") for s, t in rank(kb3, q)[:3]]
```

Output:

```
=== Check 1: does word overlap with the reference score a harmful answer below a correct paraphrase? ===
case              answer      Jaccard  ROUGE-1  ROUGE-2
Maize, nitrogen   correct        0.21     0.32     0.11
Maize, nitrogen   harmful        0.89     0.89     0.78
Tomato, spray gap correct        0.12     0.20     0.00
Tomato, spray gap harmful        0.69     0.79     0.62
Bean seed rate    correct        0.33     0.50     0.23
Bean seed rate    harmful        0.71     0.83     0.64
Cases where the harmful answer scores higher than the correct paraphrase (of 3): {'Jaccard': 3, 'ROUGE-1': 3, 'ROUGE-2': 3}

=== Check 2: an updated document next to its old version ===
Before the update, ranking for the question:
  0.855  old
  0.164  other
  0.039  other
After adding the new text and leaving the old one in:
  0.784  old
  0.352  NEW
  0.166  other
After replacing the old text (superseded by document id):
  0.567  NEW
  0.298  other
  0.044  other
```

What the run shows, and what it does not:

- **Overlap rewards copying.** The wrong-dose answers scored 0.69 to 0.89 on Jaccard and ROUGE-1 against 0.12 to 0.50 for the correct paraphrases, and the same order held for ROUGE-2. This is consistent with the AIEP finding that such metrics give a mixed picture, and it is why a check that can read numbers, either an expert or a judge model, is needed for advice where a dose can be wrong by a factor of 10.
- **An update is not finished when the new text is added.** With both versions in the index, the old text ranked first (0.78 against 0.35), because it matched the question's words about litres per hectare. A model given both passages sees a contradiction and may pick either. Replacing the old passage by document identifier put the new one first (0.57). The reports read for this post do not say how the deployed services version their documents, so this is a design suggestion from the check, not a documented practice.
- **Limits.** The passages are invented, three cases are not a sample, and the retrieval is a basic term-weighted ranking, not an embedding model. Which version outranks which depends on the wording; the point is that nothing in retrieval knows which one is newer.

## What a team should track after launch

1. Share of questions unanswered, split by reason (out of scope, unsupported crop, missing content), with an owner for each reason.
2. A private, agronomist-checked question set that is rerun whenever the model, the prompt or the corpus changes, with a check on the checker itself.
3. For every updated source document, whether the old version is still in the index.
4. Usage by group (gender, language, channel), not only the average.
5. Time from a content gap appearing in the logs to the gap being closed.

## What comes next

Part 5 counts the cost and the benefit: what the published deployments say about cost per query, the staff time behind the knowledge base, and the evidence on whether farmers' practices or incomes change.

## References

- AIEP Initiative, [Building AI-based advisory services for smallholder farmers: Technical learnings from the AIEP Initiative](https://arxiv.org/abs/2601.11537), January 2026.
- Singh, Wang'ombe, Okanga et al., [Farmer.Chat: Scaling AI-Powered Agricultural Services for Smallholder Farmers](https://arxiv.org/abs/2409.08916), 2024.
- Singh, [Under the Hood of Farmer.chat: Journey to an optimised, production ready RAG powered Chatbot](https://medium.com/digitalgreen-techblog/under-the-hood-of-farmer-chat-journey-to-an-optimised-production-ready-rag-powered-chatbot-589bf5716e27), Digital Green engineering blog, July 2024.
- Kpodo, R, Minkoff and Singaraju, [Beyond the model: Evaluating AI agricultural advisory systems so they work in the field](https://www.ifpri.org/blog/beyond-the-model-evaluating-ai-agricultural-advisory-systems-so-they-work-in-the-field), IFPRI, May 2026.
- Es, James, Espinosa-Anke and Schockaert, [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217), 2023.
