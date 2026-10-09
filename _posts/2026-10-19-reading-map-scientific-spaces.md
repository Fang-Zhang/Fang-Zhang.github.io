---
title: "A Reading Map for Scientific Spaces: 1,340 Posts of Math Behind Machine Learning"
date: 2026-10-19 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, machine-learning, mathematics, reading-list]
description: "Scientific Spaces (科学空间) has 1,340 posts from 2009 to 2026. A map of its series and a reading order for the math behind machine learning."
---

[Scientific Spaces (科学空间)](https://spaces.ac.cn/) is a Chinese-language blog by Su Jianlin (苏剑林). It has 1,340 posts, running from March 2009 to October 2026. He wrote 197 posts in 2009 and has written about 50 to 60 a year since 2016. The average post grew from about 1,600 characters in 2009 to nearly 10,000 in 2024. He writes less often now, and each post goes further.

Almost every topic is a series. A series is the right unit to read, so this map is organized by series.

**Three eras**

- **2009 to 2013: science notes.** Astronomy events, physics, translations and linear algebra.
- **2014 to 2016: mathematics and code.** Number theory, with a twelve-part series that begins at [the background of Fermat's Last Theorem](https://spaces.ac.cn/archives/2805), then Python, OCR and Riemannian geometry.
- **2017 to now: machine learning.** Word vectors, then generative models, then language models and attention, then diffusion, and since 2025 optimizers, matrices and MoE.

**The series worth reading**

*Transformer and attention*

- **[Transformer升级之路 (Transformer Upgrade Path)](https://spaces.ac.cn/archives/8231)**: 21 posts, 2021 to 2025. It starts with [where Sinusoidal position encoding comes from](https://spaces.ac.cn/archives/8231), and part 2 introduces [RoPE](https://spaces.ac.cn/archives/8265), the rotary position encoding he proposed. Later parts cover [RoPE as a base-β encoding](https://spaces.ac.cn/archives/9675), [ReRoPE for length extrapolation](https://spaces.ac.cn/archives/9708) and [what makes MLA work](https://spaces.ac.cn/archives/11111).
- **[《Attention is All You Need》浅读](https://spaces.ac.cn/archives/4765)** (2018): an early walkthrough with code, and one of his most-read posts.
- **[线性注意力简史 (A Short History of Linear Attention)](https://spaces.ac.cn/archives/11033)** (2025).

*Generative models*

- **[变分自编码器 (Variational Autoencoder)](https://spaces.ac.cn/archives/5253)**: 9 posts, 2018 to 2021, starting with "so that's what it is". Part 1 is his most-read post, with about 1.45 million views.
- **[生成扩散模型漫谈 (Notes on Diffusion Models)](https://spaces.ac.cn/archives/9119)**: 31 posts, 2022 to 2025, the longest series. Part 1 frames DDPM as "tearing down a building and rebuilding it" and sets out to show that diffusion models can be explained in plain language. The series ends with [predicting the data instead of the noise](https://spaces.ac.cn/archives/11428).

*Optimizers and training*

- **[让炼丹更科学一些 (Making Alchemy a Bit More Scientific)](https://spaces.ac.cn/archives/9902)**: 10 posts so far, 2023 to 2026. "Alchemy" is the joke name for training neural networks. He sets out to learn the optimization theory behind it, starting from a basic convergence result for SGD, and part 10 covers [the monotonicity assumption](https://spaces.ac.cn/archives/11885).
- **The Muon posts.** [Muon优化器赏析](https://spaces.ac.cn/archives/10592) (2024) is the introduction. [Muon续集](https://spaces.ac.cn/archives/10739), [QK-Clip](https://spaces.ac.cn/archives/11126) and [Muon优化器指南 (Muon Optimizer Guide)](https://spaces.ac.cn/archives/11416) (2025) follow. The guide is a quick start for readers who only know Adam. It describes Muon as an optimizer for matrix parameters, proposed by Keller Jordan, that his team first validated at scale in Moonlight. A later five-part series builds [a streaming power-iteration implementation](https://spaces.ac.cn/archives/11654).
- **[流形上的最速下降 (Steepest Descent on Manifolds)](https://spaces.ac.cn/archives/11196)**: 7 posts, 2025 to 2026, from SGD on a hypersphere to [an analytic solution on the Stiefel manifold](https://spaces.ac.cn/archives/11864).

*Architecture and scaling*

- **[MoE环游记 (A Tour of MoE)](https://spaces.ac.cn/archives/10699)**: 9 posts, 2025 to 2026, written in the same style as the Transformer series. Part 1 starts from the geometry of the MoE layer, and part 9 covers [the gating-normalization debate](https://spaces.ac.cn/archives/11782).
- Two recent single posts: [his notes on K3's MoE and attention](https://spaces.ac.cn/archives/11848) and [解构Scaling Law](https://spaces.ac.cn/archives/11833).

*Mathematical foundations*

- **[低秩近似之路 (The Road to Low-Rank Approximation)](https://spaces.ac.cn/archives/10366)**: 5 posts, 2024 to 2025, from the pseudo-inverse to CUR. He says low-rank approximation feels familiar and strange at once: everyone knows the idea from LoRA, but the papers keep using techniques most people never learned. The series fills in that gap.

**What to skip**

The 2009 to 2013 posts on astronomy events, reposts and site announcements, unless you want the history. The 2015 to 2017 posts on sentiment classification and Word2Vec use methods that have since changed, but they show how his approach developed.

**A reading order**

1. [Transformer升级之路 1](https://spaces.ac.cn/archives/8231) to [4](https://spaces.ac.cn/archives/8397), the best way to learn his style.
2. [生成扩散模型漫谈 1](https://spaces.ac.cn/archives/9119).
3. [低秩近似之路 1](https://spaces.ac.cn/archives/10366), the math the optimizer posts depend on.
4. [Muon优化器指南](https://spaces.ac.cn/archives/11416), then [让炼丹更科学一些 1](https://spaces.ac.cn/archives/9902).
5. [MoE环游记 1](https://spaces.ac.cn/archives/10699).
6. His two newest posts, [慢即是快：Shampoo的尽头是Muon？](https://spaces.ac.cn/archives/11917) and [通过微扰分析求解等式约束优化问题](https://spaces.ac.cn/archives/11928), to see what he is working on now.

His writing goes from a concrete question to a derivation, and the derivation ends in a general principle. The posts that hold up are the ones where he stops to ask why a design takes the form it does. Start with the series that asks that question about something you already use.
