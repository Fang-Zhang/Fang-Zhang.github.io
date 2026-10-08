---
title: "How to Read a 190-Post Blog: A Map of Sebastian Raschka's Writing"
date: 2026-10-08 06:00:00 +1300
categories: [Personal, Learning]
tags: [LLM, reading-list, machine-learning, learning]
description: "Sebastian Raschka's blog has about 190 posts over thirteen years. A reading order that moves from method to map to practice to the current frontier."
---

Some blogs are too big to read. [Sebastian Raschka's blog](https://sebastianraschka.com/blog/) has about 190 posts across thirteen years, from early notes on PCA and naive Bayes to the 2026 notes on attention variants and reasoning models. You can't read it front to back, and sorting by date won't help. The useful question is which order to read it in.

I haven't read all of it. I read the index and one article in full, and the rest of this map comes from titles and summaries, so treat it as a plan for reading, not a review.

**One article to read first**

His [Recommendations for Getting the Most Out of a Technical Book](https://sebastianraschka.com/blog/2025/reading-books.html) (November 2025) is short and sets the method. It gives five steps for each chapter:

1. Read it once, offline, without code, for about twenty minutes. Don't look anything up. The goal is the big picture.
2. Read it again and type the code yourself instead of copying it. If your results differ from the book's, check the repository, then package versions, random seeds and hardware, and ask the author last.
3. Do the exercises. Try properly before looking at the solutions.
4. Go back over your highlights and notes, and look up what is still unclear.
5. Use an idea from the chapter in a small project of your own.

He adds that none of this is fixed. A chapter you already know can be skimmed, and one without code skips the code steps. The method is a starting point, not a rule.

It is the right first read because it tells you how to read the other 189: once for shape, once for detail, then build something.

**Then a map before the details**

The next one is his one-hour talk, [Developing an LLM: Building, Training, Finetuning](https://magazine.sebastianraschka.com/p/llms-building-training-finetuning). It covers the three stages of building a language model, and the later posts are easier to place once you have that frame.

After that, the blog sorts into clusters:

- **Building from scratch.** [Self-attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention), a [BPE tokenizer](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html), the [KV cache](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms), [Qwen3](https://magazine.sebastianraschka.com/p/qwen3-from-scratch), a [three-hour workshop](https://magazine.sebastianraschka.com/p/building-llms-from-the-ground-up). Code first, with the mechanism built step by step.
- **Fine-tuning.** The LoRA series ([LoRA](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html), [DoRA](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)), [instruction pretraining](https://magazine.sebastianraschka.com/p/instruction-pretraining-llms), and a [spam classifier built on a GPT-style model](https://magazine.sebastianraschka.com/p/building-a-gpt-style-llm-classifier). This cluster is practical.
- **Post-training and reasoning.** [New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training), [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms) (four ways to build one), and [a piece on reinforcement learning with GRPO](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training). Between 2025 and 2026 he follows how reasoning models change, including [inference-time scaling](https://magazine.sebastianraschka.com/p/categories-of-inference-time-scaling) and [controlling reasoning effort](https://magazine.sebastianraschka.com/p/controlling-reasoning-effort-in-llms).
- **Architecture.** [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) runs from GPT-2 to DeepSeek V3. Later pieces cover [gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the) and [attention variants](https://magazine.sebastianraschka.com/p/visual-attention-variants) such as MHA, GQA and MLA, plus an [architecture gallery](https://sebastianraschka.com/blog/2026/llm-architecture-gallery.html) of 93 diagrams and a tool for comparing two of them.
- **Evaluation and the yearly picture.** The [four approaches to evaluation](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches) (multiple-choice benchmarks, verifiers, leaderboards, LLM judges), the yearly paper lists ([2026 part 1](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1)), and [The State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025).
- **Workflow.** [How he reads new model releases](https://magazine.sebastianraschka.com/p/workflow-for-understanding-llms), and [how he keeps up with the research](https://sebastianraschka.com/blog/2023/keeping-up-with-ai.html).

**What to skip**

The 2013–2022 posts are about classical machine learning: PCA, LDA, naive Bayes, model evaluation. Skip them unless you need that background. One exception is [Losses Learned](https://sebastianraschka.com/blog/2022/losses-learned-part1.html), on negative log-likelihood and cross-entropy in PyTorch. Loss calculation comes up constantly in later work, so it earns a read.

**A reading order**

1. [The technical-book recommendations](https://sebastianraschka.com/blog/2025/reading-books.html)
2. [The one-hour talk](https://magazine.sebastianraschka.com/p/llms-building-training-finetuning)
3. [The classifier article](https://magazine.sebastianraschka.com/p/building-a-gpt-style-llm-classifier), then [Losses Learned](https://sebastianraschka.com/blog/2022/losses-learned-part1.html)
4. [The LoRA series](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html), then [the post-training piece](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training)
5. [Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms), then [GRPO](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
6. [The big architecture comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
7. [The yearly paper lists](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1) and [the architecture gallery](https://sebastianraschka.com/blog/2026/llm-architecture-gallery.html), as reference

That order moves from method to map to practice to the current frontier. It works for any large body of technical writing. Pick the article that teaches you how to read the rest, build a frame, then go deeper in the order your work needs.
