---
title: "Deep Learning, Domain by Domain, Part 6: Recommendation, Fraud, Security and Forecasting"
date: 2026-12-23 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deep-learning, recommender-systems, fraud-detection, forecasting, series]
description: "Part 6 of Deep Learning, Domain by Domain. Ten papers on recommendation, fraud, malware and forecasting, each with what it reports, one of which finds a one-layer linear model beating Transformers."
---

This post does one thing: it covers the business prediction problems where deep learning is used on tabular and sequence data, not on pictures or free text: recommendation, fraud and malware detection, and forecasting. It takes ten papers, one of them a counterexample, and states what each reports. The recurrent and attention models are from [Part 2](/posts/deep-learning-domain-by-domain-2-foundations-from-convolution-and-recurrence-to-the-transformer/).

## The field: prediction on business data

These problems share a shape: a history goes in, and a score or a list or a distribution comes out. What differs is the input. Recommendation inputs are sparse categorical features such as user and item identities, fraud inputs are transaction sequences, and forecasting inputs are many related time series. The figure sorts the papers by that shape.

![Three rows of boxes, each reading input to model to output. Recommendation: user history and sparse categorical features, candidate generation then ranking, a ranked list of items. Fraud and security: card transaction sequences or raw program bytes, an LSTM or a network on raw bytes, a fraud or malware label. Forecasting: histories of many related time series, recurrent or attention models, a probability distribution of the future.]({{ "/assets/pic/dl-domain-6-prediction-problems.png" | relative_url }})
*Three prediction problems on business data, with the papers covered. Layout and groupings are the author's; the boxes paraphrase each paper's abstract.*

## Recommendation

**Session-based recommendation (2015).** Hidasi and colleagues' [recurrent networks for recommendation](https://arxiv.org/abs/1511.06939) is framed in its abstract as applying recurrent networks to a new domain. They start from the case where recommendations must rest on short session data, not long user histories, where they say matrix factorization is not accurate, and propose modelling the whole session. They add modifications to classic recurrent networks, such as a ranking loss function, and report marked improvements on two data sets over widely used approaches.

**YouTube (2016).** Covington, Adams and Sargin's [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations) (RecSys 2016) describes, in its abstract, one of the largest industrial recommendation systems at a high level, split along the classic two-stage dichotomy of information retrieval: a deep candidate generation model, then a separate deep ranking model. The abstract also promises practical lessons from designing, iterating on and maintaining the system. Only the abstract page was read for this post.

**Wide & Deep (2016).** Cheng and colleagues' [Wide & Deep learning](https://arxiv.org/abs/1606.07792) jointly trains a wide linear model and a deep network. The abstract's reasoning: cross-product features memorise feature interactions and are interpretable, while deep networks with embeddings generalise to unseen feature combinations but can over-generalise and recommend less relevant items when interactions are sparse. They evaluated it on Google Play, a store with over one billion active users and over one million apps, and report that it significantly increased app acquisitions in online experiments compared with wide-only and deep-only models.

**Neural collaborative filtering (2017).** He and colleagues' [NCF](https://arxiv.org/abs/1708.05031) notes that earlier deep recommenders used deep learning mainly on auxiliary information such as text, and still took an inner product of user and item factors for the interaction itself. They replace the inner product with a neural network that learns an arbitrary interaction function, and report significant improvements on two real-world data sets, with deeper layers performing better.

**DLRM (2019).** Naumov and colleagues' [deep learning recommendation model](https://arxiv.org/abs/1906.00091) opens with the statement that these networks differ significantly from other deep networks because they handle categorical features, and are not well studied. They release an implementation in PyTorch and Caffe2, and design a parallelisation scheme that splits the embedding tables across devices (model parallelism) while scaling the fully connected layers by data parallelism, because of memory constraints.

## Fraud and security

Jurgovsky and colleagues' [Sequence classification for credit-card fraud detection](https://doi.org/10.1016/j.eswa.2018.01.037) (Expert Systems with Applications, 2018) frames fraud detection as sequence classification and uses LSTM networks over transaction sequences. Compared with a random forest baseline, the LSTM improves detection accuracy on offline, face-to-face transactions. Both approaches benefit strongly from manual feature aggregation, and a later analysis found that the two tend to detect different frauds, which the authors say suggests combining them.

Raff and colleagues' [malware detection from raw byte sequences](https://arxiv.org/abs/1710.09435) (2017) presents initial work on a problem with over two million time steps, in which batch normalization appeared to hinder learning. Their network has linear complexity in the sequence length and allows interpretable sub-regions of the binary to be identified.

## Forecasting

Salinas and colleagues' [DeepAR](https://arxiv.org/abs/1704.04110) (2017) trains an autoregressive recurrent network on a large number of related time series to produce probabilistic forecasts, that is, a distribution over the future. The abstract motivates it with retail demand forecasting, and reports accuracy improvements of around 15% over state-of-the-art methods on several real-world data sets.

Lim and colleagues' [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) (2019) targets multi-horizon forecasting with a mix of static covariates, known future inputs and historically observed series. It combines recurrent layers for local processing with interpretable self-attention for long-term dependencies, and reports significant improvements over existing benchmarks on real-world data sets, with three interpretability use cases.

Zeng and colleagues' [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) (2022) questions the line of work above. They argue that permutation-invariant self-attention inevitably loses temporal information, and introduce embarrassingly simple one-layer linear models, LTSF-Linear. On nine real-life data sets, these outperform existing Transformer-based long-term forecasting models in all cases, often by a large margin.

## What the sources show

These points are the author's reading of the sources above, not claims the sources make.

- **The papers state a specific gap, not a general superiority.** Session data too short for factorization, memorisation against generalisation, sparse high-rank interactions, and an input too long for standard networks each appear in an abstract as the reason a method was built.
- **Several of the contributions are engineering.** The YouTube abstract is about lessons from maintaining a system, and DLRM's is about splitting tables across devices. Neither is mainly about a more accurate model.
- **The same period contains a result against the trend.** The 2022 forecasting paper reports that a one-layer linear model beat Transformers on nine data sets. That is a reason to compare against a simple baseline before choosing a deep model, and it is the author's inference, not the paper's recommendation.
- **Hand-built features stay in.** The fraud paper reports that both the LSTM and the random forest benefit strongly from manual feature aggregation.

## Limits of this summary

- It covers ten papers. Pricing, search ranking, advertising, churn, credit scoring, anomaly detection on logs and graph methods for fraud are left out.
- Every paper was read as an abstract page, and the credit-card paper also its highlights. The figures quoted are those the sources state, on their own data sets, and the online results are from the authors' own systems.
- Production systems change, and this post makes no claim about how any named company's system works today.
