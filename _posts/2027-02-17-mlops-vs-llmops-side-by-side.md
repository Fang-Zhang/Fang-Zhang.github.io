---
title: "MLOps and LLMOps Side by Side: What Carried Over and What Did Not"
date: 2027-02-17 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [MLOps, LLMOps, comparison, series]
description: "Epilogue to the MLOps series. Six operational questions answered with measurements from both series: what is pinned, how it is evaluated, what a release is, what monitoring sees, what the loop costs, and what each field does not cover."
---

The [MLOps series](/posts/mlops-1-training-serving-skew/) and the [LLMOps series](/posts/llmops-1-from-mlops-to-llmops/) were written independently, with different models and different data. This epilogue puts their measurements next to each other, one operational question at a time. Every number below comes from one of the twelve posts; the comparison and the readings are the author's. The experiments differ in model, data and scale, so the numbers show the type of behaviour, not a ratio between the two fields.

## 1. What can be pinned

- **MLOps ([Part 3](/posts/mlops-3-reproducible-training-and-search/)):** three runs with the same data, configuration and seed gave identical predictions. Across 20 seeds validation AUC had a standard deviation of 0.0021. Pinning is nearly complete, and the seed is part of the model's identity.
- **LLMOps ([Part 1](/posts/llmops-1-from-mlops-to-llmops/)):** greedy output was unchanged across reruns, batch sizes and thread counts on one CPU. Number format was not pinned by those settings: it changed the output of 3 of 40 prompts for fp16 and 33 of 40 for int4.

In both fields the repeatable part is repeatable to the bit. The difference is in what counts as a model change: for the tabular model it is data, configuration and seed; for the language model it also includes the number format and everything wrapped around the weights.

## 2. How a result is judged

- **MLOps (Part 3):** a search over 64 configurations reported a validation AUC above test by 0.0167 with 150 validation rows and 0.0053 with 3,000. The error is selection optimism, and it is bounded by validation size.
- **LLMOps ([Part 3](/posts/llmops-3-evaluation-is-a-measurement/)):** the 95% interval on an accuracy was 26.3 points wide at 50 items and 6.1 at 1,000. A true 2.72-point gap showed as significant in 10.1% of 100-item samples.

The shared lesson is that an evaluation set is a sample. The MLOps case has a labelled metric with a known scale; the LLMOps case often has a judge or a task-specific score as well, and the measurement error of that judge is a second layer.

## 3. What a release is

- **MLOps (Parts 1 and 3):** the release unit is the model plus the feature code that feeds it. Five serving-side differences cost up to 0.134 AUC, and a per-column comparison of logged features caught all five.
- **LLMOps ([Part 4](/posts/llmops-4-versioning-release-and-replay/)):** the release unit adds the prompt template, retrieval index and sampling parameters. In eight single changes, four (a renamed template word, an edited document, an appended document, a weight change of 0.001) left all 5 test outputs identical, so equality of outputs would have missed them.

A manifest of hashes serves both. For the tabular model the manifest checks inputs to the model; for the language model it must also cover text that is itself part of the program.

## 4. What monitoring sees

- **MLOps ([Part 4](/posts/mlops-4-delayed-labels-and-monitoring/)):** on a concept change with unchanged inputs, a labelled-AUC monitor alarmed after the label delay plus about two days (2, 3, 9, 16 and 32 days for delays of 0, 1, 7, 14 and 30). Input monitoring saw nothing.
- **LLMOps ([Part 6](/posts/llmops-6-drift-needs-more-than-one-detector/)):** four detectors (confidence, input PSI, labelled accuracy, canary replay) each caught different changes. A canary replay caught int8, int4 and a retrained model in 100% of runs and missed an input shift in all of them; input PSI caught the shift in 100% and the model changes in about 1%.

Both say the same thing: no single detector covers the failure modes, and a detector that watches inputs cannot see a change in the relation between inputs and answers. The practical difference is the label: classic ML usually has one that arrives late, while a language model system often has none, so the canary and the judge take its place.

## 5. What the loop costs

- **MLOps ([Part 5](/posts/mlops-5-feedback-loops/)):** the cost is information. A ranker with no exploration gave data on 30 of 200 items and found 0.40 of the best 5; at 30% exploration it found 2.92.
- **LLMOps ([Part 5](/posts/llmops-5-observability-percentiles-and-routing-cost/)):** the cost is compute. A two-model cascade at a 0.7 confidence threshold reached 58.93% accuracy against 59.23% for the large model alone, at 97.2% of its cost.

One field spends to learn, the other to answer; both choose a point on a curve and both need a log that records the decision, not only the outcome.

## 6. What each field does not cover

- **MLOps in this series:** serving at scale and the memory cost of generation were not touched. [LLMOps Part 2](/posts/llmops-2-serving-batching-and-the-kv-cache/) measured both: batch size moved throughput from 125 to 1,359 tokens/s while step latency rose from 6.4 to 37.7 ms.
- **LLMOps in this series:** release statistics. [MLOps Part 6](/posts/mlops-6-release-and-the-peeking-problem/) showed daily peeking at an A/B test gives 25.1% false positives against 4.9% for one look, and the same arithmetic applies to any online comparison of two language-model variants.

## Limits

- The two series use different toy models (tabular classifiers and a small character transformer), so no quantity in one can be converted into the other.
- All data is synthetic or a single public text; behaviour of production systems with human traffic is not measured.
- The "readings" in each section are the author's interpretation of the numbers.
