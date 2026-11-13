---
title: "Deep Learning, Domain by Domain, Part 9: Business Impact, Cost, Evaluation and Risk"
date: 2027-02-06 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deep-learning, ai-risk, evaluation, tabular-data, series]
description: "Part 9 of Deep Learning, Domain by Domain. Seven questions to ask before using deep learning in a business, each tied to a paper or official document."
---

This post does one thing: it turns the first eight parts into seven questions to ask before a business adopts deep learning, and ties each question to a paper or an official document. The questions are the author's framing; the findings are the sources'. The domain parts are [Part 3](/posts/deep-learning-domain-by-domain-3-images-and-video/), [Part 4](/posts/deep-learning-domain-by-domain-4-language-and-speech/), [Part 6](/posts/deep-learning-domain-by-domain-6-recommendation-fraud-security-and-forecasting/) and [Part 8](/posts/deep-learning-domain-by-domain-8-medicine-biology-and-drug-discovery/).

## The field: after the demo

Parts 3 to 8 reported what models can do on a benchmark, a game or a data set. A business asks something else: does it fit this data, what does it cost, will it keep working, and who is affected when it fails. The figure lists the seven questions with one source each.

![A table of seven questions with the source and finding for each: tabular data and tree-based models; training cost; maintenance cost; behaviour after deployment; accuracy across groups; adversarial examples and data extraction; documentation and governance.]({{ "/assets/pic/dl-domain-9-seven-questions.png" | relative_url }})
*Seven questions and the finding that answers each. The questions are the author's framing; each right-hand cell paraphrases the cited source.*

## 1. Is the data tabular and medium-sized?

[Grinsztajn and colleagues (2022)](https://arxiv.org/abs/2207.08815) note that deep learning's superiority on tabular data is not clear. They benchmark deep learning methods against tree-based models such as XGBoost and Random Forests on a standard set of 45 data sets. The abstract reports that tree-based models remain state of the art on medium-sized data, around 10,000 samples, even without accounting for their superior speed. The reasons they investigate include sensitivity to uninformative features and the orientation of the data. This links back to [Part 6](/posts/deep-learning-domain-by-domain-6-recommendation-fraud-security-and-forecasting/), where a one-layer linear model beat Transformers on nine forecasting data sets.

## 2. What does training cost?

[Strubell, Ganesh and McCallum (2019)](https://arxiv.org/abs/1906.02243) quantify the approximate financial and environmental costs of training several NLP models, and note that the accuracy gains depend on exceptionally large computational resources. A [2021 follow-up](https://arxiv.org/abs/2104.10350) calculates the energy use and carbon footprint of T5, Meena, GShard, Switch Transformer and GPT-3. It reports that sparsely activated networks can consume under a tenth of the energy of dense ones, that the carbon-free share of energy varies about 5 to 10 times between locations, and that the choice of network, data centre and processor together can reduce the footprint by roughly 100 to 1000 times.

## 3. What does upkeep cost?

The 2015 NeurIPS paper on [hidden technical debt in machine learning systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) argues it is dangerous to think of quick wins as coming for free. Using the software-engineering idea of technical debt, it finds massive ongoing maintenance costs are common in real-world systems, and lists risk factors including boundary erosion, entanglement, hidden feedback loops, undeclared consumers, data dependencies and changes in the external world.

## 4. Will it hold up after deployment?

[D'Amour and colleagues (2020)](https://arxiv.org/abs/2011.03395) name underspecification: a pipeline is underspecified when it can return many predictors with equally strong held-out performance, and these predictors can behave very differently in deployment. They show it in computer vision, medical imaging, language, clinical risk prediction from health records and medical genomics. This is the same gap Part 8 raised from the other side: a held-out score is not the same as an outside check.

## 5. Does it work for every group?

The [Gender Shades paper](https://proceedings.mlr.press/v81/buolamwini18a.html) (2018) found two common facial analysis benchmarks overwhelmingly composed of lighter-skinned subjects (79.6% and 86.2%), built a data set balanced by gender and skin type, and evaluated three commercial gender classification systems on it. Darker-skinned females were the most misclassified group, with error rates up to 34.7%, against a maximum of 0.8% for lighter-skinned males. [Model cards](https://arxiv.org/abs/1810.03993) (Mitchell and colleagues, 2018) propose the remedy of short documents shipped with a model, with evaluation across cultural, demographic or phenotypic groups and a statement of intended use.

## 6. Can it be fooled, or made to leak?

The 2013 paper [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) reports that an imperceptible perturbation, found by maximising the network's prediction error, can make a network misclassify an image, and that the same perturbation can fool a different network trained on a different subset of the data. [Carlini and colleagues (2021)](https://arxiv.org/abs/2012.07805) show that an adversary querying GPT-2 can recover hundreds of verbatim training sequences, including names, phone numbers and email addresses.

## 7. Is it documented and governed?

Regulation and frameworks now exist. [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng), the Artificial Intelligence Act, lays down harmonised rules on artificial intelligence. The US National Institute of Standards and Technology publishes an [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework). This post does not summarise either text; which obligations apply depends on the use and the jurisdiction, and that needs the documents and a lawyer.

## What the sources show

These points are the author's reading, not claims the sources make.

- **Most of the questions are about the system, not the network.** Maintenance, deployment and documentation sit outside the model.
- **Several results point to the cheaper option.** Trees on medium tabular data, a linear model in forecasting, and a sparse or well-placed training run all beat the default choice on a stated measure.
- **Failures are measured by subgroup, or after deployment.** Gender Shades and underspecification both show that an overall score hides them.
- **Parts 3 to 8 each ended with an evaluation caveat.** Part 9 is that caveat as a checklist.

## Limits of this summary

- Seven questions are a selection; labour, procurement, intellectual property and model licensing are not covered. The source for each question is one or two papers, not a survey.
- The abstracts were read; the EU and NIST pages were read only at the level of title and description. Nothing here is legal or compliance advice.
- The book that supplied this series' outline has a chapter on business impact; this post uses only its title, and every claim comes from the sources linked above.
