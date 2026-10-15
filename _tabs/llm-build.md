---
title: Tools, Notes and Ideas for Building a LLM
icon: fas fa-tools
order: 7
hidden: true
permalink: /llm-build/
---

Tools, notes and ideas I collect while learning to build a large language model.

## Foundations

- **[PyTorch](https://pytorch.org/)**: the core framework.
- **[tiktoken](https://github.com/openai/tiktoken)**: the GPT-2 tokenizer used in Raschka's book. Check the splits in [Tiktokenizer](https://tiktokenizer.vercel.app/), an online tokenizer playground by [dqbd](https://duong.dev).
- **[Jupyter](https://jupyter.org/) or [VS Code](https://code.visualstudio.com/)**: for code and experiments.

## Reference implementations

- **[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)** (Sebastian Raschka): the official code for the book, with extra bonus chapters.
- **[nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat)** (Andrej Karpathy): leaner training pipelines, good for seeing a full pretraining run end to end.
- **[LitGPT](https://github.com/Lightning-AI/litgpt)** (Lightning AI): from-scratch implementations of common open models, with finetuning scripts.

## Training efficiently

- **Mixed precision.** bf16 needs a GPU with compute capability 8.0 or higher (Ampere and later). Turing cards such as the T1000 (7.5) use fp16 with gradient scaling, see the [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html).
- **[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).** It can speed up training. It relies on Triton, which has no official Windows build, so on Windows it needs community-built wheels.
- **Gradient accumulation.** Simulates a large batch when GPU memory is too small to hold one.

## Watching and debugging

- **[nvidia-smi](https://developer.nvidia.com/system-management-interface)**: memory use, for tracking down out-of-memory errors.
- **[TensorBoard](https://www.tensorflow.org/tensorboard) or [Weights & Biases](https://wandb.ai/)**: loss curves.
- **[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)**: finds where training time goes.

## Data

- **[Hugging Face datasets](https://github.com/huggingface/datasets) and [tokenizers](https://github.com/huggingface/tokenizers)**: load corpora and train a custom tokenizer.

## Evaluation

- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**: the standard benchmark runner.
- **[Ollama](https://ollama.com/)**: in chapter 7 of the book it runs the judge model that scores instruction-following answers. On a small GPU, a smaller judge model is the practical choice.
