---
title: "LLMs on the Factory Floor, Part 6: Beat the Baseline Before Buying the Model"
date: 2027-06-11 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, manufacturing, baselines, anomaly detection, search, llms on the factory floor]
description: "Part 6 of 6 on large language models in manufacturing. A language model matched a rule-based baseline on fault detection by flagging nearly everything, and plain keyword search over maintenance records finds the right record in the top ten 83 to 96 percent of the time. Alternatives to test first."
---

## What this post finds

The first five parts assumed a language model was the tool. This last part asks what it has to beat. Three public sources set the baselines, and two checks make the comparison concrete.

- **On simulated HVAC faults, a language model only matched a rule.** [Lee, Vidyaratne, Farahat and Gupta](https://arxiv.org/abs/2509.23113) tested GPT-4o and GPT-4.1-nano on sensor windows from a simulated HVAC system. Their best F1 for anomaly detection was 0.84, against 0.85 for a rule-based statistical baseline, and the paper's conclusion is that most models under-performed or matched it. The authors do credit the models with readable explanations.
- **Matching, here, meant flagging almost everything.** The baseline's table row has recall 1.00 with precision and accuracy both 0.73. That pattern is what a method scores when it flags every sample and about 73 percent of samples are faulty (a prevalence inferred from the table, not stated in the paper). The best language-model rows have the same precision and accuracy with recall 0.99, so they flag 94 to 100 percent of normal samples as faulty, once rounding is allowed for.
- **Plain search is a strong starting point.** The [BEIR benchmark](https://arxiv.org/abs/2104.08663) (Thakur and colleagues, 2021) across 18 datasets found BM25, the classic keyword ranking function, a robust baseline, with dense and sparse neural retrievers often underperforming it zero-shot. On maintenance records, plain word search over raw text finds the right record in the top ten for 83 to 96 percent of plain-language queries.
- **Language models for time series are an open question.** [Tang and Ding](https://arxiv.org/abs/2412.12219) (2024) compared language-model methods with simpler ones on classification, anomaly detection and forecasting and found benefits in specific tasks such as anomaly detection but not in forecasting, where simpler models sometimes did as well or better. [Zhou and Yu](https://arxiv.org/abs/2410.05440) (2024) found models understood time series better as images than as text, and that explicit reasoning prompts did not help.

![Two panels. Left: in a simulated HVAC fault-detection study, a rule-based baseline with recall 1.00 and a GPT-4.1-nano setup with statistics input both flag between 94 and 100 percent of normal samples, while the raw-input setup flags 71 to 80 percent. Right: mean reciprocal rank for finding a maintenance record from a plain query in three mining companies; plain word search on raw records scores 0.72, 0.72 and 0.87, character n-gram search 0.66, 0.64 and 0.89, and word search on normalised records about 0.99]({{ site.baseurl }}/assets/pic/factory-6-alternatives.png){: width="900" }
_Figure: left, derived from Table 1 of Lee et al. by the script below; right, computed on the public MaintNorm records._

## The two studies behind the checks

**Fault detection on simulated HVAC data.** Lee and colleagues built a simulator of a commercial HVAC system with three injected faults (refrigerant leak, compressor fault, filter blockage), and asked a language model to flag anomalies and classify faults from windows of sensor readings. Summaries (descriptive statistics) worked better than raw numbers, a window of 36 samples was best, and a larger model helped modestly. For fault classification, the paper reports that rule-based methods achieve higher accuracy while language models tend to do better on precision and recall, that the differences are often small and may not justify the higher computational cost, and that models struggled to adapt over repeated fault cycles. The authors also note that the models flagged statistically extreme but normal readings, such as an airflow spike in peak cooling hours, because they lack operating context.

**Maintenance text.** The MaintNorm records from Part 2 again, as a search problem: 3,600 records per company from three mining companies, each with the raw text and a human-normalised version.

## What the checks show

Check 1 solves for the false-alarm rate implied by the published precision, recall and prevalence, allowing for rounding. Check 2 builds three indexes and measures how often a plain-language query finds its record. It uses no language model and no embedding model; the query is the human-normalised text of a record, and the document is the raw text.

```python
# Part 6, two checks on published numbers and public data. Standard library only.
import math, os, random, collections
print("=== Check 1: what the HVAC fault-detection table says about false alarms (Lee et al. 2025, Table 1; the rule-based row has recall 1.00) ===")
# With recall = 1.00 and precision = accuracy = prevalence, a method flags every sample: prevalence is bounded by the rounding of 0.73.
rows = [("GPT-4.1-nano, statistics input", 0.73, 0.99, 0.84, 0.73), ("GPT-4.1-nano, raw input", 0.75, 0.84, 0.79, 0.67), ("Rule-based baseline", 0.73, 1.00, 0.85, 0.73)]
def fpr_bounds(P, R):
    best = []
    for pi in (0.725, 0.73, 0.735):
        for p in (P - .005, P + .005):
            for r in (min(R + .005, 1.0), R - .005):
                tp = r * pi; flagged = tp / p; fp = flagged - tp
                best.append(fp / (1 - pi))
    return min(best), max(best)
print("flagging everything scores precision = prevalence, recall = 1, F1 = 2p/(1+p):", ", ".join(f"p={p}: F1 {2*p/(1+p):.3f}" for p in (0.725, 0.73, 0.735)))
for name, P, R, F, A in rows:
    lo, hi = fpr_bounds(P, R)
    print(f"{name:30s} precision {P:.2f} recall {R:.2f} F1 {F:.2f} accuracy {A:.2f} -> share of normal samples flagged, between {min(lo,1):.0%} and {min(hi,1):.0%} (allowing for rounding)")

print()
print("=== Check 2: find a maintenance record from a plain-language query (MaintNorm, three mining companies) ===")
print("query = the normalised words of a record; documents = all raw records of that company; success = own record in the top 10")
D = os.path.expanduser("~/.hermes/profiles/blog-writer/cache/scratch/ind/mn/")
def load(c):
    out = []
    for sp in ("train", "test"):
        cur = []
        for line in open(f"{D}company_{c}_{sp}.norm", encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                if cur: out.append(cur); cur = []
                continue
            a, b = line.split("\t"); cur.append((a, b))
        if cur: out.append(cur)
    return out
def words(s): return [w for w in s.lower().split() if any(ch.isalpha() for ch in w) and not w.startswith("<")]
def grams(s, lo=3, hi=5):
    s = " " + " ".join(words(s)) + " "; return [s[i:i+n] for n in range(lo, hi+1) for i in range(len(s)-n+1)]
class Index:
    def __init__(s, docs, f, kind):
        s.f, s.kind, s.N = f, kind, len(docs); s.tf = []; df = collections.Counter(); s.inv = collections.defaultdict(list)
        for i, d in enumerate(docs):
            c = collections.Counter(f(d)); s.tf.append(c); df.update(c.keys())
            for t, n in c.items(): s.inv[t].append((i, n))
        s.idf = {t: math.log(1 + (s.N - n + .5) / (n + .5)) for t, n in df.items()}
        s.len = [sum(c.values()) for c in s.tf]; s.avg = sum(s.len) / s.N
        if kind == "cos":
            s.norm = [math.sqrt(sum((n * s.idf[t]) ** 2 for t, n in c.items())) or 1 for c in s.tf]
    def scores(s, q):
        sc = collections.defaultdict(float); qc = collections.Counter(s.f(q))
        if s.kind == "bm25":
            for t in qc:
                if t not in s.inv: continue
                for i, n in s.inv[t]: sc[i] += s.idf[t] * n * 2.2 / (n + 1.2 * (0.25 + 0.75 * s.len[i] / s.avg))
        else:
            qn = math.sqrt(sum((n * s.idf.get(t, 0)) ** 2 for t, n in qc.items())) or 1
            for t, n in qc.items():
                if t not in s.inv: continue
                for i, m in s.inv[t]: sc[i] += n * m * s.idf[t] ** 2 / (qn * s.norm[i])
        return sc
def rank_of(sc, i):
    mine = sc.get(i, 0.0)
    greater = sum(1 for v in sc.values() if v > mine); equal = sum(1 for v in sc.values() if v == mine and True) - (1 if i in sc else 0)
    return 1 + greater + (equal / 2 if mine == 0 else 0)
random.seed(7)
for c in "abc":
    T = load(c); raw = [" ".join(a for a, b in t) for t in T]; norm = [" ".join(b for a, b in t) for t in T]
    keep = [i for i in range(len(T)) if len(words(norm[i])) >= 3]
    qs = random.sample(keep, 400)
    methods = [("word search on raw records", Index(raw, lambda x: words(x), "bm25"), raw),
               ("character 3-5-gram search on raw records", Index(raw, grams, "cos"), raw),
               ("word search on normalised records", Index(norm, lambda x: words(x), "bm25"), norm)]
    print(f"company {c.upper()} ({len(T)} records, {len(qs)} queries)")
    for name, ix, docs in methods:
        rr = []
        for i in qs:
            r = rank_of(ix.scores(norm[i]), i); rr.append(r)
        print(f"   {name:42s} found in top 10: {sum(r <= 10 for r in rr)/len(rr):5.1%}   mean reciprocal rank {sum(1/r for r in rr)/len(rr):.3f}")

```

Output:

```
=== Check 1: what the HVAC fault-detection table says about false alarms (Lee et al. 2025, Table 1; the rule-based row has recall 1.00) ===
flagging everything scores precision = prevalence, recall = 1, F1 = 2p/(1+p): p=0.725: F1 0.841, p=0.73: F1 0.844, p=0.735: F1 0.847
GPT-4.1-nano, statistics input precision 0.73 recall 0.99 F1 0.84 accuracy 0.73 -> share of normal samples flagged, between 94% and 100% (allowing for rounding)
GPT-4.1-nano, raw input        precision 0.75 recall 0.84 F1 0.79 accuracy 0.67 -> share of normal samples flagged, between 71% and 80% (allowing for rounding)
Rule-based baseline            precision 0.73 recall 1.00 F1 0.85 accuracy 0.73 -> share of normal samples flagged, between 95% and 100% (allowing for rounding)

=== Check 2: find a maintenance record from a plain-language query (MaintNorm, three mining companies) ===
query = the normalised words of a record; documents = all raw records of that company; success = own record in the top 10
company A (3600 records, 400 queries)
   word search on raw records                 found in top 10: 86.8%   mean reciprocal rank 0.723
   character 3-5-gram search on raw records   found in top 10: 85.8%   mean reciprocal rank 0.657
   word search on normalised records          found in top 10: 100.0%   mean reciprocal rank 0.990
company B (3600 records, 400 queries)
   word search on raw records                 found in top 10: 83.2%   mean reciprocal rank 0.716
   character 3-5-gram search on raw records   found in top 10: 81.5%   mean reciprocal rank 0.640
   word search on normalised records          found in top 10: 100.0%   mean reciprocal rank 0.995
company C (3600 records, 400 queries)
   word search on raw records                 found in top 10: 95.8%   mean reciprocal rank 0.870
   character 3-5-gram search on raw records   found in top 10: 98.0%   mean reciprocal rank 0.889
   word search on normalised records          found in top 10: 100.0%   mean reciprocal rank 0.996
```

- **F1 hid the false alarms.** In the paper's table, 0.84 against 0.85 looks like parity with a rule. But because about 73 percent of samples appear to be faulty, flagging everything already gives an F1 of 0.84. The paper itself warns that accuracy is inflated by class imbalance. The baseline-and-model comparison cannot show skill unless the false-alarm rate is reported with it. Only rounded values were available, so the ranges above are bounds, and the raw-input setup, at 71 to 80 percent, is the only one clearly separate from flagging everything.
- **The rule needs no model.** A threshold on the sensor statistics is free to run, explains itself, and needs no regression suite of the kind Part 4 sized.
- **Keyword search finds known records.** Plain word search finds the target record in the top ten for 86.8, 83.2 and 95.8 percent of queries in the three companies, with mean reciprocal ranks of 0.72, 0.72 and 0.87.
- **A character n-gram index did not fix spelling.** It beat word search in one company (0.89 against 0.87) and trailed it in two (0.66 against 0.72 and 0.64 against 0.72). The misses come from abbreviations such as "hyd" for hydraulic, which share few characters with the full word.
- **Normalisation is what moves the number.** With human-normalised records, the mean reciprocal rank is 0.99 in every company. That is a ceiling by construction, since query and document then contain the same words; the Part 2 normaliser reported 95.8 percent error reduction, so a real one would land below it. A language model is useful there, but as a normaliser, not as the search engine.
- **Limits.** Check 1 reads rounded numbers from one simulated system, and the paper describes the baseline as a rule that flags a window when any sensor leaves its historical bounds, so near-total flagging is an inference from the table. Check 2 is known-item search with 400 queries per company; it does not test dense embeddings, questions that need several records, or answers that need reading. The BEIR result is about general text retrieval, not maintenance.

## What this means for the choice

1. **Build the baseline first.** A threshold or rule for sensor data, a keyword index for text. Both take days, run on a laptop and give a number the model has to beat.
2. **Report false alarms next to F1.** On imbalanced data, a metric that rewards flagging everything will make any model look competitive.
3. **Use the model where language is the problem.** Expanding shorthand, drafting a work-order summary and explaining an alarm in words are language tasks. Detecting a drifting sensor is not, and the published evidence for language models on time series is mixed.
4. **Prefer a hybrid when both matter.** Let the rule or classical model detect, and let the language model explain and look up the procedure. That keeps the detection testable without a model, and the model's mistakes in wording rather than in plant state.
5. **Keep the escape route.** If the baseline is nearly as good, the cheaper, auditable option wins, and the language model can be added where it earns its cost.

## What the series adds up to

Across six parts the same pattern recurs. A leaderboard gap under four points is noise (Part 1). Fix the records before tuning the model, and retrieve facts instead of training them in (Part 2). Keep the model beside the control loop and gate on behaviour, not syntax (Part 3). A weekly check needs hundreds of questions (Part 4). The token bill is the small number and checking time is the big one (Part 5). And a baseline can look as good as a model for the wrong reasons (this part). Each of them is a measurement that can be repeated, and none of them needs a belief about what language models will become.

## References

- Lee, Vidyaratne, Farahat and Gupta, [Exploring LLM-based Frameworks for Fault Diagnosis](https://arxiv.org/abs/2509.23113), 2025.
- Thakur, Reimers, Rücklé, Srivastava and Gurevych, [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), 2021.
- Tang and Ding, [Are Large Language Models Useful for Time Series Data Analysis?](https://arxiv.org/abs/2412.12219), 2024.
- Zhou and Yu, [Can LLMs Understand Time Series Anomalies?](https://arxiv.org/abs/2410.05440), 2024.
- Bikaun, Hodkiewicz and Liu, [MaintNorm](https://aclanthology.org/2024.wnut-1.7.pdf), W-NUT 2024; data at [github.com/nlp-tlp/maintnorm](https://github.com/nlp-tlp/maintnorm).
