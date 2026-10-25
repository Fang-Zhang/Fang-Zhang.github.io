---
title: "The Phone's Limit Is Memory, the Agent's Limit Is Permission"
date: 2026-12-10 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [on-device-ai, llm, agents, apple-intelligence, gemma, android]
description: "How language models are being fitted into phones, and how agents are being given access to apps: what the papers and official documents say, and where the control ends up."
---

A phone has a fast processor and a small amount of working memory. A language model with tens of billions of parameters needs far more memory than the phone has, and most of the engineering of on-device AI is a response to that mismatch. A second problem sits above it: once a model runs on the phone, how does it act on the user's behalf inside apps it did not write? This post goes through both, using the papers and vendor documents, and separates what those sources state from what is my own judgment.

## 1. The bottleneck is DRAM

Generating a token requires reading the active weights from memory. A phone keeps its full storage in flash, which is large and slow, and its working memory in DRAM, which is small and fast. A model that does not fit in DRAM has to be shrunk, trimmed or streamed.

![Flash holds the model, DRAM holds the active part](/assets/pic/phone-memory.png){: width="700" }
_Figure 1. The model lives in flash; only the part in use sits in DRAM, and the link between them is the limit. My own drawing, with figures from the sources cited below._

The published approaches fall into four groups.

![Four ways to fit a model into a phone](/assets/pic/phone-fit.png){: width="700" }
_Figure 2. Four routes to the same goal, each with a representative source. My own drawing._

**Design it small.** [MobileLLM](https://arxiv.org/abs/2402.14905) (ICML 2024) targets models under one billion parameters. Its finding is that at this scale architecture matters more than the usual emphasis on data and parameter count suggests: deep and thin networks, shared embeddings and grouped-query attention gave the best results, and the paper reports API-calling accuracy close to LLaMA-v2 7B for its small models.

**Compress the numbers.** Apple's [2025 technical report](https://arxiv.org/abs/2507.13575) describes a 3-billion-parameter on-device model that uses KV-cache sharing and 2-bit quantization-aware training, plus a Swift framework that exposes guided generation, constrained tool calling and LoRA adapters to developers.

**Keep only part in memory.** Google's [Gemma 3n](https://developers.googleblog.com/introducing-gemma-3n) has 5 and 8 billion raw parameters, but with Per-Layer Embeddings the stated memory footprint is that of a 2 billion and a 4 billion parameter model, about 2 GB and 3 GB. It also nests a smaller submodel inside the larger one, trained with [MatFormer](https://arxiv.org/abs/2310.07707), so a device can pick a size at run time.

**Stream the rest from flash.** [LLM in a flash](https://arxiv.org/abs/2312.11514) (ACL 2024) treats flash read cost as the design constraint. It reuses recently activated neurons and reads in larger contiguous chunks, which lets a model up to twice the size of the DRAM run, with reported speedups of 4-5x on CPU and 20-25x on GPU over naive loading. [PowerInfer-2](https://arxiv.org/abs/2406.06282) applies the same idea on an actual phone, splitting dense neuron clusters to the NPU and sparse ones to the CPU. Its authors report serving a 47-billion-parameter model at 11.68 tokens per second, a self-reported result.

Apple's third-generation models, [published in June 2026](https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models), put the last route into a product. Besides a 3-billion-parameter dense model, the family has a roughly 20-billion-parameter sparse on-device model, AFM 3 Core Advanced, that activates 1 to 4 billion parameters at a time. The full model sits in flash. Apple states that flash-to-DRAM bandwidth is too slow to swap experts token by token, as a standard mixture-of-experts model would, so a dense block picks a fixed set of experts when the prompt is processed and reselects them periodically during generation.

This is the same discipline as large-scale serving: the question is what has to be where, and when it has to move. The change is the scale of the budget. A data-center system has terabytes and fights over cache hit rates; a phone has a few gigabytes and fights over what to load at all.

## 2. Local does not mean offline

A common summary of on-device AI is that the model runs on the phone and nothing leaves it. The vendor documents describe something narrower.

- Android runs [Gemini Nano inside AICore](https://developer.android.com/ai/gemini-nano), a system service that follows the [Private Compute Core](https://arxiv.org/abs/2209.10317) principles, isolates each request and does not store inputs or outputs. Google lists offline use, privacy and lower inference cost as the use cases.
- Apple's family has two on-device models and three server models. The heaviest tasks, "agentic tool use and complex reasoning" in Apple's wording, go to the largest server model on Private Cloud Compute.

The pattern is a tiered system: a small local model for frequent, private, latency-sensitive work and a larger remote one for the rest. The sources I found do not describe a phone that runs everything itself. I did not find a vendor claim that on-device models replace the cloud for hard tasks, and I would treat any such claim as unsupported until a paper or official document makes it.

## 3. How an agent gets into an app

A model that only answers questions is the easy case. The hard case is an agent that books, files, orders and replies inside apps. There are three technical routes, and they differ most in who decides.

![Three ways an agent reaches an app](/assets/pic/phone-agent-paths.png){: width="700" }
_Figure 3. Three routes from an agent to an app's function and the party that decides. My own drawing._

**Operating the screen.** The agent reads the user interface and taps like a person. A TMLR-accepted [survey of LLM-powered phone GUI agents](https://arxiv.org/abs/2504.19838) traces the move from script-based automation to agents that read screens. It names limited generality, high maintenance cost and weak intent understanding as the old problems, and lists dataset diversity, on-device efficiency and security as open challenges. In this route, the app is not asked.

**Registering functions with the OS.** Android 16 adds [AppFunctions](https://developer.android.com/ai/appfunctions). An app declares functions, such as creating a task or searching email, and the OS keeps a registry that authorized callers can discover and execute. The documentation calls it the mobile equivalent of tools in the Model Context Protocol. Callers need the `EXECUTE_APP_FUNCTIONS` permission, and as of May 2026 the integration with Gemini is in private preview with trusted testers.

**Calling a cloud server.** The agent reaches tools hosted by a server, under the server operator's rules.

## 4. Where the control ends up

The sources are specific enough to support a few observations. These are my own reading, not claims from the papers.

First, AppFunctions removes the need to imitate a user in front of a screen. The app opts in, defines what it exposes and keeps its own logic. That is cleaner for apps than screen scraping.

Second, the registry and the permission live in the operating system. An app that wants to be reachable by the system's agent must register with the system; an agent that wants to reach apps must hold the system's permission. A common prediction is that agents will break the walls of super-apps. The mechanism described in the documents moves the wall instead of removing it: the party that owns the registry and the permission decides which apps are visible to the agent and in which order. Whether that party behaves as the user's agent or as a gatekeeper cannot be read from the API; it depends on policy.

Third, the memory limit and the permission limit are the same kind of limit. In both cases the system decides what is loaded and what is reachable, and the user sees only the result. The technical work of the last three years has gone into the first limit. Much less public material exists on the second: who audits what an on-device agent is allowed to call, what the user is shown before it acts, and what is logged.

## What the sources do not settle

- Real-world speeds on specific phones. Figures I found for particular chips came from blogs and forum posts, which I did not use. The speed numbers above are from the papers, are self-reported, and depend on the device and the model.
- Whether stronger on-device models reduce use of the cloud. The vendor documents describe tiers, not substitution.
- How AppFunctions behaves outside the private preview, and whether other platforms adopt a comparable registry.

## Sources

- [MobileLLM](https://arxiv.org/abs/2402.14905), Meta, ICML 2024
- [LLM in a flash](https://arxiv.org/abs/2312.11514), Apple, ACL 2024
- [PowerInfer-2](https://arxiv.org/abs/2406.06282)
- [Apple Intelligence Foundation Language Models: Tech Report 2025](https://arxiv.org/abs/2507.13575)
- [Introducing the Third Generation of Apple's Foundation Models](https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models), June 2026
- [Gemma 3n](https://developers.googleblog.com/introducing-gemma-3n) and [MatFormer](https://arxiv.org/abs/2310.07707)
- [Gemini Nano on Android](https://developer.android.com/ai/gemini-nano) and [Private Compute Core](https://arxiv.org/abs/2209.10317)
- [AppFunctions overview](https://developer.android.com/ai/appfunctions)
- [LLM-Powered GUI Agents in Phone Automation](https://arxiv.org/abs/2504.19838)
