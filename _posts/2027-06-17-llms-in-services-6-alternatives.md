---
title: "LLMs in Services, Part 6: Count the Labels Before Choosing the Model"
date: 2027-06-17 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, services, intent detection, baselines, trade-offs, llms in services]
description: "Part 6 of 6 on large language models in service work. On Banking77 a TF-IDF classifier reaches 89 percent with all labels and 45 percent with three per intent, while published GPT-4 reaches 83 percent with three. The choice depends on how many labelled examples exist, whether the intent list changes, and how much traffic is out of scope."
---

## What this post finds

Most service traffic is a small number of intents asked in many ways: card delivery status, a failed transfer. The alternative to a language model there is an ordinary classifier, and the public record says when each one wins.

- **With labels, the small model wins on cost and speed.** A [2026 preprint](https://arxiv.org/abs/2608.20371) (a decision framework for intent detection, written from the experience of a production voice-AI platform) compared a fine-tuned RoBERTa, a TF-IDF classifier and Claude Haiku zero-shot. On ATIS (893 test items) RoBERTa scored 95.9 and Claude 84.1. On CLINC150 (5,500 test items, 150 intents) they were statistically tied at 89.1 and 88.5. The local RoBERTa runs in 2.4 milliseconds at no marginal cost; Claude's median latency was 981 milliseconds at 0.25 dollars per thousand calls.
- **The model wins where the list of intents moves or the question is out of scope.** On out-of-scope detection, Claude's recall was 85.6 against 58.1 for RoBERTa and 36.4 for TF-IDF. When the intents were split into two apps, a RoBERTa fine-tuned on one app scored 96.8 there and 0.0 on the other, while a schema-prompted Claude scored 94.3 and 93.9. Under noisy speech recognition at 0 dB the TF-IDF classifier fell to 80.0 and Claude held 92.5, on 120 synthetic utterances.
- **With a handful of labels, the model wins on accuracy.** In [Breaking the Bank with ChatGPT](https://arxiv.org/abs/2308.14634), on Banking77 (77 customer-service intents), GPT-3.5 reached 75.2 with one example per intent and GPT-4 reached 83.1 with three, against 76.3 for a contrastively fine-tuned MPNet at three and 94.0 with all data. The experiments cost about 60 dollars with GPT-3.5 and about 1,480 dollars with GPT-4, and the authors note the fees may be costly for small organizations.
- **A plain baseline closes most of the distance once labels exist.** The check below trains TF-IDF with logistic regression on the same dataset.

![Two panels. Left: Banking77 accuracy against labelled examples per intent; a TF-IDF logistic regression climbs from 25 percent at one example to 89 percent with all data, while published GPT-3.5 (1 example, 75 percent), GPT-4 (3 examples, 83 percent) and SetFit (76 to 94 percent) sit above it until the data is abundant. Right: accuracy of the local classifier on its most confident queries, 99.4 percent on the top 50 percent, 98.0 percent on the top 70 percent and 89.4 percent overall]({{ site.baseurl }}/assets/pic/services-6-alternatives.png){: width="900" }
_Figure: left, Banking77 accuracy against labels per intent, with published numbers marked; right, accuracy of the local classifier on its most confident queries. Computed here on CPU._

## What the sources do and do not show

The decision-framework paper is a preprint from a company with a production setting; it tests one model (Claude Haiku zero-shot) against one encoder, English only, and says few-shot prompting would likely narrow the ATIS gap. Its noise test uses synthetic speech with additive noise, and its dynamic-schema test is a clean partition of one dataset. The Banking77 paper used GPT models available in 2023 and a human expert to choose representative examples, and reports micro-F1; the check here reports accuracy, which is close but not identical. Neither source is about a service company's own logs.

## What the checks show

Check 1 trains the baseline on Banking77 (10,003 training and 3,080 test utterances, 77 intents) with k labelled examples per intent, five random draws each, on one CPU. Check 2 puts Wilson intervals on the published gaps using their test-set sizes. Check 3 sorts test queries by the baseline's own confidence and asks how much traffic it can answer on its own.

```python
# Part 6 of the services series: a CPU baseline on Banking77 and a re-scoring of published intent-detection numbers.
import csv, math, random, numpy as np, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
d = "b77/"
def load(f): return [(r["text"], r["category"]) for r in csv.DictReader(open(d + f, newline="", encoding="utf-8"))]
train, test = load("train.csv"), load("test.csv")
labels = sorted({c for _, c in train}); print(f"Banking77: {len(train)} train, {len(test)} test, {len(labels)} intents")
Xte, yte = [t for t, _ in test], np.array([c for _, c in test])
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return 100*(c-h), 100*(c+h)
print("\n=== Check 1: TF-IDF + logistic regression with k labelled examples per intent (5 random draws; test accuracy, %) ===")
by = {}
for t, c in train: by.setdefault(c, []).append(t)
res = {}; t0 = time.time()
for k in (1, 3, 5, 10, 20, "all"):
    accs = []
    for seed in (range(5) if k != "all" else [0]):
        rnd = random.Random(seed); X, y = [], []
        for c, ts in by.items():
            for t in (ts if k == "all" else rnd.sample(ts, k)): X.append(t); y.append(c)
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True); A = vec.fit_transform(X)
        clf = LogisticRegression(C=20, max_iter=300).fit(A, y); accs.append(100 * float(np.mean(clf.predict(vec.transform(Xte)) == yte)))
    res[k] = accs; lo, hi = wilson(round(np.mean(accs) * len(yte) / 100), len(yte))
    print(f"k = {str(k):>3}: mean {np.mean(accs):5.1f}  range {min(accs):5.1f}-{max(accs):5.1f}   test-set 95% interval of the mean about {lo:4.1f}-{hi:4.1f}")
print(f"(all fits on one CPU, {time.time()-t0:.0f} s)")
print("published, same dataset (Breaking the Bank with ChatGPT, Table 4, micro-F1): GPT-3.5 1-shot 75.2, GPT-4 3-shot 83.1, SetFit MPNet 3-shot 76.3, 10-shot 88.0, full data 94.0")
print("\n=== Check 2: how large are the published gaps against their test sets? (Wilson 95% intervals) ===")
for name, a, b, n in (("ATIS, RoBERTa vs Claude zero-shot", 95.9, 84.1, 893), ("CLINC150, RoBERTa vs Claude zero-shot", 89.1, 88.5, 5500)):
    la, ha = wilson(round(a * n / 100), n); lb, hb = wilson(round(b * n / 100), n)
    print(f"{name:40s} {a} [{la:.1f},{ha:.1f}] vs {b} [{lb:.1f},{hb:.1f}]  gap {a-b:.1f}  intervals {'overlap' if lb<=ha and la<=hb else 'do not overlap'}")
n = 1000
for name, r in (("Claude 0-shot", 85.6), ("RoBERTa (ft)", 58.1), ("TF-IDF+LR", 36.4)):
    lo, hi = wilson(round(r * n / 100), n); print(f"out-of-scope recall, {name:14s} {r} [{lo:.1f},{hi:.1f}]  (1,000 out-of-scope test items)")
print("\n=== Check 3: answer the confident traffic locally and send the rest to a model (Banking77, TF-IDF + LR on all training data) ===")
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True); A = vec.fit_transform([t for t, _ in train])
clf = LogisticRegression(C=20, max_iter=300).fit(A, [c for _, c in train]); P = clf.predict_proba(vec.transform(Xte))
conf = P.max(1); pred = clf.classes_[P.argmax(1)]; ok = pred == yte
print(f"overall accuracy {100*ok.mean():.1f}")
for q in (0.3, 0.5, 0.7, 0.8):
    th = np.quantile(conf, 1 - q); m = conf >= th
    print(f"answer locally the {int(q*100)}% most confident: accuracy on that part {100*ok[m].mean():5.1f}; {100*(1-q):.0f}% of queries go to the model (its accuracy there is not measured here)")
print("published prices and latency (Claude Haiku, Table 5 of the decision-framework paper): $0.25 per 1,000 calls, median 981 ms; local RoBERTa 2.4 ms, $0")

```

Output:

```
Banking77: 10003 train, 3080 test, 77 intents

=== Check 1: TF-IDF + logistic regression with k labelled examples per intent (5 random draws; test accuracy, %) ===
k =   1: mean  25.4  range  24.4- 27.1   test-set 95% interval of the mean about 23.9-27.0
k =   3: mean  44.9  range  44.0- 46.1   test-set 95% interval of the mean about 43.1-46.6
k =   5: mean  55.7  range  54.2- 57.2   test-set 95% interval of the mean about 53.9-57.4
k =  10: mean  67.6  range  67.0- 68.3   test-set 95% interval of the mean about 66.0-69.3
k =  20: mean  77.5  range  76.8- 78.9   test-set 95% interval of the mean about 76.0-78.9
k = all: mean  89.4  range  89.4- 89.4   test-set 95% interval of the mean about 88.3-90.5
(all fits on one CPU, 61 s)
published, same dataset (Breaking the Bank with ChatGPT, Table 4, micro-F1): GPT-3.5 1-shot 75.2, GPT-4 3-shot 83.1, SetFit MPNet 3-shot 76.3, 10-shot 88.0, full data 94.0

=== Check 2: how large are the published gaps against their test sets? (Wilson 95% intervals) ===
ATIS, RoBERTa vs Claude zero-shot        95.9 [94.3,97.0] vs 84.1 [81.6,86.4]  gap 11.8  intervals do not overlap
CLINC150, RoBERTa vs Claude zero-shot    89.1 [88.2,89.9] vs 88.5 [87.6,89.3]  gap 0.6  intervals overlap
out-of-scope recall, Claude 0-shot  85.6 [83.3,87.6]  (1,000 out-of-scope test items)
out-of-scope recall, RoBERTa (ft)   58.1 [55.0,61.1]  (1,000 out-of-scope test items)
out-of-scope recall, TF-IDF+LR      36.4 [33.5,39.4]  (1,000 out-of-scope test items)

=== Check 3: answer the confident traffic locally and send the rest to a model (Banking77, TF-IDF + LR on all training data) ===
overall accuracy 89.4
answer locally the 30% most confident: accuracy on that part  99.7; 70% of queries go to the model (its accuracy there is not measured here)
answer locally the 50% most confident: accuracy on that part  99.4; 50% of queries go to the model (its accuracy there is not measured here)
answer locally the 70% most confident: accuracy on that part  98.0; 30% of queries go to the model (its accuracy there is not measured here)
answer locally the 80% most confident: accuracy on that part  96.7; 20% of queries go to the model (its accuracy there is not measured here)
published prices and latency (Claude Haiku, Table 5 of the decision-framework paper): $0.25 per 1,000 calls, median 981 ms; local RoBERTa 2.4 ms, $0
```

- **Labels decide the winner.** TF-IDF reaches 44.9 percent with three examples per intent, against 83.1 for published GPT-4 at three, and 89.4 with all 130 per intent, against 94.0 for the published fine-tuned MPNet. The few-shot gap is 38 points; the full-data gap is under 5. It is a different model family from the published row, so the closing gap is an estimate of the order, not a controlled comparison.
- **Which published gaps are real.** The ATIS gap of 11.8 points has non-overlapping intervals, and the CLINC150 gap of 0.6 points sits well inside them. The out-of-scope recall gap (85.6 against 58.1) is far outside the intervals of 1,000 items.
- **Confidence splits the traffic.** The baseline answers its 70 percent most confident queries at 98.0 percent accuracy and its 50 percent most confident at 99.4. Only the remaining 30 percent would need a model, which at the published list price is a 70 percent reduction in model spend. The accuracy of the model on that harder part is not measured here, and uncertainty-based routing is the approach the decision-framework paper cites for recovering most of an LLM's accuracy at half the latency.
- **Limits.** Banking77 is single-turn intent detection, not the multi-step procedures of Parts 1 to 3, where the benchmark evidence favours models. The confidence ranking comes from the same classifier, and some of its confident errors are the ones that matter. Mean accuracy intervals reflect the test set only, not the training draws, which add the shown range.

## What this means for choosing

1. **Count the labelled examples.** With several hundred per intent, a classifier matches the published fine-tuned models within a few points at close to zero marginal cost. With a handful, a model wins.
2. **Ask whether the intent list changes.** A fixed head cannot emit a label it never saw; a schema-prompted model can serve a new list the same day.
3. **Measure the out-of-scope slice.** It is where the published gap was largest and where safe handling matters most.
4. **Use both.** A cheap classifier on the confident majority and a model on the remainder keeps latency and spend down while keeping the model where it helps.
5. **Keep the baseline in the evaluation.** A model that cannot beat a TF-IDF classifier on the same labels has not earned its latency.

## The series in six lines

Part 1: score the procedure, not only the outcome. Part 2: tuning teaches procedure; facts belong in retrieval. Part 3: put the policy outside the model. Part 4: models retire on someone else's schedule, and judges drift. Part 5: the saving is measured in seconds, so model spend has a ceiling. Part 6: count the labels before choosing the model.

## References

- [When Do LLMs Replace Fine-Tuned NLU? A Decision Framework for Intent Detection in Production Conversational Systems](https://arxiv.org/abs/2608.20371), 2026.
- [Breaking the Bank with ChatGPT: Few-Shot Text Classification for Finance](https://arxiv.org/abs/2308.14634), FinNLP 2023.
- Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807), 2020 (the Banking77 dataset).
