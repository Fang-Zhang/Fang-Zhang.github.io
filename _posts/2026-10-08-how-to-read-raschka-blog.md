---
title: "How to Read a 190-Post Blog: A Map of Sebastian Raschka's Writing"
date: 2026-10-08 06:00:00 +1300
categories: [Personal, Learning]
tags: [LLM, reading-list, machine-learning, learning]
description: "Sebastian Raschka's blog has about 190 posts over thirteen years. A reading order that moves from method to map to practice to the current frontier."
---

Some blogs are too big to read. [Sebastian Raschka's blog](https://sebastianraschka.com/blog/) has about 190 posts across thirteen years, from early notes on PCA and naive Bayes to the 2026 notes on attention variants and reasoning models. You can't read it front to back, and sorting by date won't help. The useful question is which order to read it in.

**One article to read first**

His [Recommendations for Getting the Most Out of a Technical Book](https://sebastianraschka.com/blog/2025/reading-books.html) (November 2025) is short and sets the method. It gives five steps for each chapter:

1. Read it once, offline, without code, for about twenty minutes. Don't look anything up. The goal is the big picture.
2. Read it again and type the code yourself instead of copying it. If your results differ from the book's, check the repository, then package versions, random seeds and hardware, and ask the author last.
3. Do the exercises. Try properly before looking at the solutions.
4. Go back over your highlights and notes, and look up what is still unclear.
5. Use an idea from the chapter in a small project of your own.

He adds that none of this is fixed. A chapter you already know can be skimmed, and one without code skips the code steps. The method is a starting point, not a rule.

It is the right first read because it tells you how to read the other 189: once for shape, once for detail, then build something.

**The blog in eight parts**

After that, the blog sorts into eight groups.

**1. Building from scratch**

Code first, with each mechanism built step by step.

- [Understanding and Coding Self-Attention, Multi-Head, Causal and Cross-Attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) (2023.2)
- [Implementing a BPE Tokenizer From Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) (2025.1)
- [Building LLMs from the Ground Up: A 3-hour Coding Workshop](https://magazine.sebastianraschka.com/p/building-llms-from-the-ground-up) (2024.9)
- [Coding LLMs from the Ground Up: A Complete Course](https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up) (2025.5)
- [Building A GPT-Style LLM Classifier From Scratch](https://magazine.sebastianraschka.com/p/building-a-gpt-style-llm-classifier) (2024.9), a spam classifier
- [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) (2025.6)
- [Understanding and Implementing Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch) (2025.9)
- [Developing an LLM: Building, Training, Finetuning](https://magazine.sebastianraschka.com/p/llms-building-training-finetuning) (2024.6), a one-hour talk on the three stages of LLM development

**2. Fine-tuning and parameter-efficient methods**

- The LoRA series: [Parameter-Efficient Finetuning](https://sebastianraschka.com/blog/2023/llm-finetuning-llama-adapter.html) (2023.4), [LoRA](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html) (2023.4), [Finetuning Falcon](https://sebastianraschka.com/blog/2023/falcon-finetuning.html) (2023.6), [DoRA from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) (2024.2)
- [Using and Finetuning Pretrained Transformers](https://magazine.sebastianraschka.com/p/using-and-finetuning-pretrained-transformers) (2024.4)
- [Instruction Pretraining LLMs](https://magazine.sebastianraschka.com/p/instruction-pretraining-llms) (2024.7) and [Instruction Masking and LoRA experiments](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction) (2024.6)
- [Optimizing LLMs From a Dataset Perspective](https://sebastianraschka.com/blog/2023/optimizing-LLMs-dataset-perspective.html) (2023.9)

**3. Post-training and reasoning models**

- [New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training) (2024.8)
- [How Good Are the Latest Open LLMs? And Is DPO Better Than PPO?](https://magazine.sebastianraschka.com/p/how-good-are-the-latest-open-llms) (2024.5)
- [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms) (2025.2), four ways to build a reasoning model
- [The State of Reinforcement Learning for LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training) (2025.4), on GRPO
- [Inference-Time Compute Scaling](https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling) (2025.3) and [Categories of Inference-Time Scaling](https://magazine.sebastianraschka.com/p/categories-of-inference-time-scaling) (2026.1)
- [Controlling Reasoning Effort in LLMs](https://magazine.sebastianraschka.com/p/controlling-reasoning-effort-in-llms) (2026.7)
- His book [Build a Reasoning Model From Scratch](https://sebastianraschka.com/blog/2026/build-a-reasoning-model-from-scratch-is-out.html) (published 2026.6)

**4. How architectures evolved**

- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) (2025.7), from GPT-2 to DeepSeek V3
- [From GPT-2 to gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the) (2025.8)
- [A Visual Guide to Attention Variants in Modern LLMs](https://magazine.sebastianraschka.com/p/visual-attention-variants) (2026.3): MHA, GQA, MLA, sparse attention
- [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms) (2025.11): linear attention, text diffusion and more
- The [LLM Architecture Gallery](https://sebastianraschka.com/blog/2026/llm-architecture-gallery.html) (93 diagrams) with a [comparison tool](https://sebastianraschka.com/blog/2026/llm-architecture-gallery-diff-tool.html), and recent short notes on [Kimi K3](https://sebastianraschka.com/blog/2026/kimi-k3-architecture-notes.html), [Gemma 4](https://sebastianraschka.com/blog/2026/gemma-4-release-notes.html) and [Nemotron 3 Ultra](https://sebastianraschka.com/blog/2026/nemotron-3-ultra-latent-moe.html)

**5. Evaluation, paper lists and trends**

- [Understanding the 4 Main Approaches to LLM Evaluation](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches) (2025.10): multiple-choice benchmarks, verifiers, leaderboards, LLM judges
- The yearly paper lists: [2024](https://magazine.sebastianraschka.com/p/llm-research-papers-the-2024-list), [2025 January to June](https://magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one), [2025 July to December](https://magazine.sebastianraschka.com/p/llm-research-papers-2025-part2), [2026 January to May](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1)
- [The State Of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025) (2025.12), a yearly review with predictions for 2026
- [State of AI 2026](https://sebastianraschka.com/blog/2026/state-of-ai-interview.html), a 4.5-hour interview with Lex Fridman and Nathan Lambert

**6. Learning methods and workflow**

- [Recommendations for Getting the Most Out of a Technical Book](https://sebastianraschka.com/blog/2025/reading-books.html) (2025.11)
- [My Workflow for Understanding LLM Architectures](https://magazine.sebastianraschka.com/p/workflow-for-understanding-llms) (2026.4)
- [Keeping Up With AI Research and News](https://sebastianraschka.com/blog/2023/keeping-up-with-ai.html) (2023.3)
- [Understanding LLMs: A Transformative Reading List](https://sebastianraschka.com/blog/2023/llm-reading-list.html) (2023.2), a list of classic papers

**7. Engineering practice**

- PyTorch training optimization (2023): [faster training](https://sebastianraschka.com/blog/2023/pytorch-faster.html), [mixed precision](https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html), [memory optimization](https://sebastianraschka.com/blog/2023/pytorch-memory-optimization.html) and [gradient accumulation](https://lightning.ai/pages/blog/gradient-accumulation/)
- [DGX Spark and Mac Mini for Local PyTorch Development](https://sebastianraschka.com/blog/2025/dgx-impressions.html) (2025.10)
- [Using Local Coding Agents](https://sebastianraschka.com/blog/2026/using-local-coding-agents.html) (2026.6)

**8. Early classical machine learning (2013-2022)**

PCA, LDA, naive Bayes and the model-evaluation series. Skip them unless you need to fill in traditional ML basics. One exception is [Losses Learned: Optimizing Negative Log-Likelihood and Cross-Entropy in PyTorch](https://sebastianraschka.com/blog/2022/losses-learned-part1.html) (2022.4), which bears directly on how loss is computed for LLMs.

**Suggested reading order**

This is a suggestion, not a template.

*Now:*

1. [Recommendations for Getting the Most Out of a Technical Book](https://sebastianraschka.com/blog/2025/reading-books.html)
2. [Developing an LLM: Building, Training, Finetuning](https://magazine.sebastianraschka.com/p/llms-building-training-finetuning), the one-hour talk, as a map of the whole field
3. [Building A GPT-Style LLM Classifier From Scratch](https://magazine.sebastianraschka.com/p/building-a-gpt-style-llm-classifier)
4. [Losses Learned](https://sebastianraschka.com/blog/2022/losses-learned-part1.html), to firm up the loss calculation

*After the from-scratch material:*

1. The LoRA series ([Parameter-Efficient Finetuning](https://sebastianraschka.com/blog/2023/llm-finetuning-llama-adapter.html), [LoRA](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html), [Finetuning Falcon](https://sebastianraschka.com/blog/2023/falcon-finetuning.html), [DoRA from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)), then [New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training)
2. [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms), then [the GRPO article](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
3. [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)

*Long-term reference:* the yearly paper lists ([2024](https://magazine.sebastianraschka.com/p/llm-research-papers-the-2024-list), [2025 January to June](https://magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one), [2025 July to December](https://magazine.sebastianraschka.com/p/llm-research-papers-2025-part2), [2026 January to May](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1)) and the [LLM Architecture Gallery](https://sebastianraschka.com/blog/2026/llm-architecture-gallery.html).

That order moves from method to map to practice to the current frontier. It works for any large body of technical writing. Pick the article that teaches you how to read the rest, build a frame, then go deeper in the order your work needs.
