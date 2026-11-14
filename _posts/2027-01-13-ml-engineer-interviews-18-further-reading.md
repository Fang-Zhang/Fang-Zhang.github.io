---
title: "ML Engineer Interviews, With Reference Answers (18/18): Further Reading for Every Round"
date: 2027-01-13 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, reading-list, resources, series]
description: "Part 18 of 18. An annotated reading list for ML engineer interviews, grouped by round: process guides, fundamentals, metrics and experiments, deep learning and LLMs, systems and recommenders, take-home and project rounds, behavioural questions, and offers and preparation. Every item was opened and read before it was listed."
---

This last post collects the reading behind the series, grouped by the round it helps with. It lists only items that were opened and read for this series, and each was opened again while this list was written. Each entry says what the page covers, which part of the series used it and who it suits. "Who it suits" is the author's judgement. For the arXiv papers, the annotation comes from the abstract page; for blogs, guides, documentation and PDFs, from the text of the page itself. The list is a starting point and does not replace the sources: [Part 1](/posts/ml-engineer-interviews-1-tell-me-about-yourself/) opens the series, and [Part 17](/posts/ml-engineer-interviews-17-questions-to-ask-and-offers/) closed the loop with offers and preparation.

## 1. The whole loop: process and first-hand accounts

- [Introduction to Machine Learning Interviews Book](https://huyenchip.com/ml-interviews-book) (Chip Huyen). A free book on ML interviews. The landing page links a web version, the source on GitHub and a Discord server for discussing the answers. Used in Part 2. Suits readers who want a question bank next to the process advice.
- [Interviewing for ML/AI Engineers](https://www.moderndescartes.com/essays/ml_eng_interviewing) (Brian Kihoon Lee, Modern Descartes). An essay on how ML interviews fail (ML system design as a system design question in ML clothing, no scenario crafting, outdated problems) and a proposed redesign of the loop with interview types: coding and algorithms, data modeling, a math quiz, system design, ML system design, project deep dive and a career chat. Used in Parts 1 and 2. Suits candidates who want to see what a loop is trying to measure, and interviewers who design one.
- [ML Job Interviews: The Ultimate Guide](https://silviasapora.github.io/blog/ml-interviews.html) (Silvia Sapora, June 2026). A first-person account of finding a research scientist role after a PhD: getting interviews, startups versus big tech, interview structure, technical, emotional and logistical preparation, negotiation, decision making and a list of technical topics. Used in Parts 1 and 2. Suits readers coming from a research background.
- [(Opinionated) Guide to ML Engineer Job Hunting](https://www.yuan-meng.com/posts/mle_interviews/) (Yuan Meng). A 27-minute read covering what to prepare, how to prepare (tracking what you learn, calendar blocks, progress tracking), coding, ML system design, ML breadth and depth, ranking, NLP, behavioural interviews and project deep dives. Suits readers aiming at applied ML and ranking roles at product companies.

## 2. Fundamentals and coding

- [Deep Learning, chapter 5: Machine Learning Basics](https://www.deeplearningbook.org/contents/ml.html) (Deep Learning book). The general principles that the rest of the book builds on. Used in Part 3. Suits readers who want the basics of capacity, generalisation and estimation from a textbook.
- [CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf) (Andrew Ng and Tengyu Ma, 2023 version). Supervised learning from linear regression (LMS algorithm, normal equations, a probabilistic interpretation) onwards. Used in Part 13. Suits readers who want derivations, not only statements.
- [Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118) (Belkin et al.). The paper behind the "double descent" curve that complicates the textbook bias-variance picture. Used in Part 3. Suits candidates asked about overparameterised models.
- [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) (Leo Breiman, 2001). The original paper: forests as combinations of tree predictors whose generalisation error converges as the number of trees grows. Used in Part 3.
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) (scikit-learn). Examples of what not to do, with the correct version, starting with inconsistent preprocessing. Used in Parts 3, 5 and 14. Suits anyone who writes a pipeline for a take-home.
- [Python time complexity](https://wiki.python.org/moin/TimeComplexity) (Python wiki). The cost of operations on lists, dicts and sets. Used in Part 12. Suits the coding round.

## 3. Metrics and experiments

- [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) (Saito and Rehmsmeier, 2015). Used in Part 4. Suits candidates who are asked when to use which curve.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (Guo et al., 2017). Used in Part 4. Suits readers who need to explain why accuracy and calibration differ.
- [Practical Guide to Controlled Experiments on the Web](https://exp-platform.com/Documents/GuideControlledExperiments.pdf) (Kohavi et al., KDD 2007). An early practitioner guide to online controlled experiments. Used in Part 4. Suits the A/B testing questions.
- [How Not To Run an A/B Test](https://www.evanmiller.org/how-not-to-run-an-ab-test.html) (Evan Miller, 2010). Repeated significance testing, optimal sample sizes, sequential and Bayesian experiment design. Used in Part 4. Suits readers who check experiments early.

## 4. Deep learning and large language models

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) (Andrej Karpathy). Training as a leaky abstraction that fails silently, then a six-step recipe from "become one with the data" to squeezing out the juice. Used in Parts 6 and 12. Suits the debugging questions.
- [Batch Normalization](https://arxiv.org/abs/1502.03167) (Ioffe and Szegedy) and [Adam](https://arxiv.org/abs/1412.6980) (Kingma and Ba). The two papers behind the most-asked training components. Used in Part 6.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al.). The transformer. Used in Part 7.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al.). The compute-optimal scaling study. Used in Part 7. Suits the scaling-law questions.
- [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al.), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al.) and [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (Lewis et al.). Parameter-efficient fine-tuning, the RLHF pipeline and retrieval for knowledge-intensive tasks. Used in Part 7.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al.). The paper behind vLLM's memory management. Used in Part 7. Suits the serving questions.

## 5. Systems, recommenders and production

- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) (Google). Best practices for ML engineering from a large company. Used in Parts 5, 9, 14 and 15. Suits every system design and take-home round.
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) (Sculley et al., NIPS 2015). Used in Parts 11 and 15. Suits project deep dives and system design.
- [Failing Loudly](https://arxiv.org/abs/1810.11953). A study of methods for detecting dataset shift; its abstract notes that ML systems tend to fail silently. Used in Part 11. Suits monitoring questions.
- [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) (Covington et al., 2016). A high-level description of an industrial recommender and the gains from deep learning. Used in Part 9.
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) (Rendle et al.) and [HNSW](https://arxiv.org/abs/1603.09320) (Malkov and Yashunin). A ranking objective for implicit feedback and an approximate nearest-neighbour index. Used in Part 9.

## 6. Take-home and project rounds

- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993). A format for documenting a model's intended use and evaluation. Used in Part 14. Suits the presentation of a take-home.
- [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) (scikit-learn). A model inspection technique measuring each feature's contribution to a fitted model's performance on a given dataset. Used in Part 14.

## 7. Behavioural and values questions

- [Revisiting Meta-Analytic Estimates of Validity in Personnel Selection](https://gwern.net/doc/statistics/meta-analysis/2021-sackett.pdf) (Sackett, Zhang, Berry and Lievens, 2022). Why structured interviews are used. Used in Parts 1 and 16. Suits readers who want to know what the rubric is for.
- [How To Use the STAR Interview Response Technique](https://www.indeed.com/career-advice/interviewing/how-to-use-the-star-interview-response-technique) (Indeed). Situation, task, action, result, with examples. Used in Part 16.
- [Tell Me About Yourself](https://www.biginterview.com/resources/tell-me-about-yourself) (Big Interview) and [How to Prepare for an Interview](https://grow.google/grow-your-career/articles/interview-tips) (Grow with Google). Sample answers for the opening question and general preparation advice. Used in Part 1.
- [Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807) (Kleinberg et al.), [Fair prediction with disparate impact](https://arxiv.org/abs/1610.07524) (Chouldechova) and [Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413) (Hardt et al.). The three fairness papers behind the simulation of Part 16. Suits the responsible-AI question.
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et al.). Documentation for datasets. Used in Part 16.

## 8. Offers and preparation

- [Ten Rules for Negotiating a Job Offer](https://haseebq.com/my-ten-rules-for-negotiating-a-job-offer/) (Haseeb Qureshi). A practitioner's rules: protecting information, positivity, not being the decision-maker, getting other offers, timing and exploding offers. Used in Part 2. Suits first-time negotiators, as one experienced person's view, not as research.
- [Toward a Process Model of First Offers and Anchoring in Negotiations](https://ncmr.lps.library.cmu.edu/article/id/574) (Lipp, Smolinski and Kesting, 2022). A structured review of 119 articles on first offers. Used in Part 17.
- [Distributed practice in verbal recall tasks](https://pubmed.ncbi.nlm.nih.gov/16719566) (Cepeda et al., 2006) and [Test-enhanced learning](https://pubmed.ncbi.nlm.nih.gov/16507066) (Roediger and Karpicke, 2006). The evidence for spacing and self-testing. Used in Part 17.
- [Good Questions To Ask at the End of an Interview](https://www.indeed.com/career-advice/interviewing/good-questions-to-ask-at-the-end-of-an-interview) and [Interview Scams](https://www.indeed.com/career-advice/interviewing/interview-scams) (Indeed). Used in Part 17.

## A suggested order

The author's suggestion for a candidate with six weeks: start with one process guide from section 1 and the plan of Part 2 to see the loop; spend the first half on fundamentals, metrics and the coding practice of Parts 3, 4 and 12, using the sources in sections 2 and 3 to check your answers; spend the second half on the system, project and take-home material (sections 4 to 6) and the behavioural stories of Part 16; and follow the spacing schedule of Part 17 throughout. Use the papers as references for the questions you missed, and read in full only the ones your target role needs.

## Limits

- Every listed item was opened and read for this series and re-opened for this post. Reading depth differs: the arXiv papers and the two memory papers were read at the level of the abstract page here, while the guides, blogs and documentation were read as pages. Nothing is recommended on the strength of reputation alone, and items that could not be opened during the final check were left out.
- The "who it suits" lines are the author's judgement. The blog accounts are single people's experiences at particular times and companies.
- The list favours free, linkable sources; books and courses that could not be opened are not included, and that is a limit of this list, not a verdict on them.
- Links can change; the dates in the entries are those on the pages.

This is the last part of the series.
