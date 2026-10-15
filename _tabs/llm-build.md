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

## Data

- **[Hugging Face datasets](https://github.com/huggingface/datasets) and [tokenizers](https://github.com/huggingface/tokenizers)**: load corpora and train a custom tokenizer.
- **[SentencePiece](https://github.com/google/sentencepiece)** (Google): an unsupervised text tokenizer, another way to train your own vocabulary.
- **[FineWeb](https://arxiv.org/abs/2406.17557)**: a large web-text corpus for pretraining, and a paper on how it was filtered and deduplicated.
- **[datatrove](https://github.com/huggingface/datatrove)** (Hugging Face): a library for filtering, deduplicating and tokenizing text at scale.

## Training method

- **[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)** (Andrej Karpathy): a step-by-step approach to training. One of its checks is to overfit a single batch of a few examples first: if the loss cannot reach its minimum there, there is a bug.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** (Kaplan et al., 2020) and **[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** (Hoffmann et al., 2022): how loss depends on model size, data and compute, and how to split a fixed compute budget between model size and training tokens.

## Training efficiently

- **Mixed precision.** bf16 needs a GPU with compute capability 8.0 or higher (Ampere and later). Turing cards such as the T1000 (7.5) use fp16 with gradient scaling, see the [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html).
- **[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).** It can speed up training. It relies on Triton, which has no official Windows build, so on Windows it needs community-built wheels.
- **[Gradient accumulation](https://lightning.ai/pages/blog/gradient-accumulation).** Simulates a large batch when GPU memory is too small to hold one. The linked post walks through finetuning an LLM on a single GPU with it.

## Distributed training

- **[FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)** (PyTorch): shards parameters, gradients and optimizer states across GPUs.
- **[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)**: a library for distributed training and inference. It matters once the model no longer fits on one GPU.

## Watching and debugging

- **[nvidia-smi](https://developer.nvidia.com/system-management-interface)**: memory use, for tracking down out-of-memory errors.
- **[TensorBoard](https://www.tensorflow.org/tensorboard) or [Weights & Biases](https://wandb.ai/)**: loss curves.
- **[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)**: finds where training time goes.

## Post-training

- **[LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314)**: low-rank adapters, and the same idea on a quantized base model, for finetuning with little memory.
- **[PEFT](https://github.com/huggingface/peft)**: the Hugging Face library for parameter-efficient finetuning.
- **[TRL](https://github.com/huggingface/trl)**: Hugging Face library to train language models with reinforcement learning, including supervised finetuning and preference optimization.

## Evaluation

- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**: the standard benchmark runner.
- **[Ollama](https://ollama.com/)**: in chapter 7 of the book it runs the judge model that scores instruction-following answers. On a small GPU, a smaller judge model is the practical choice.

## Inference and deployment

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: LLM inference in C/C++, runs quantized models on modest hardware.
- **[vLLM](https://github.com/vllm-project/vllm)**: a high-throughput, memory-efficient inference and serving engine.
- **[GPTQ](https://arxiv.org/abs/2210.17323)**: post-training quantization for generative pretrained transformers.

## Learning resources

- **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** (Andrej Karpathy): a video course that builds up to a GPT.
- **[Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)**: course site with schedule, assignments and materials.
- **[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)** (Touvron et al., 2023): the paper behind a widely copied open architecture.
