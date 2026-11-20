---
title: "Running a Small LLM on a Phone: What the Documentation Says"
date: 2027-03-23 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, on-device, mobile, quantization, llama-cpp, litert-lm, mlc-llm]
description: "Three ways to run a small language model on a phone, and the numbers the official documentation publishes: file sizes, prefill and decode speed on named phones, and what quantization does to size and speed."
---

This post collects what the official documentation of four projects says about running a small language model on a phone: how to install it, which models they list, and the speed and size numbers they publish. Every number below is copied or computed from a cited page, and nothing was measured on a phone for this post. Where a page does not state a condition (quantization, prompt length, power state), the post says so.

## Three ways in

**1. A ready-made app.** [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT licence) is on the App Store and Google Play. Its README gives three steps: install, download a model from the in-app list or add one from Hugging Face, load it and chat. It runs GGUF models through [llama.rn](https://github.com/mybigday/llama.rn), a bridge to llama.cpp, and targets the CPU, the GPU (Metal on iOS, OpenCL on Qualcomm Adreno on Android) and the NPU (Qualcomm Hexagon), with partial layer offload when a full backend is not available. When choosing a GGUF model it advises picking "a quantization that fits your device's memory and storage", and it has a Benchmark page reporting tokens per second and memory use.
The [MLC Chat app](https://llm.mlc.ai/docs/deploy/ios.html) is another prebuilt option on the App Store, and the [MLC Android page](https://llm.mlc.ai/docs/deploy/android.html) provides a demo APK built for a Samsung S23 with a Snapdragon 8 Gen 2 chip. Google's [AI Edge Gallery](https://ai.google.dev/edge/litert-lm/overview) is described as an experimental app that runs models offline using LiteRT-LM, on Google Play and the App Store.

**2. A terminal on the phone.** The [llama.cpp Android guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md) describes Termux (an Android terminal emulator that needs no root) and then the normal CMake build:

```bash
apt update && apt upgrade -y
apt install git cmake libandroid-spawn
# build llama.cpp following docs/build.md, then:
curl -L {model-url} -o ~/{model}.gguf
./build/bin/llama-cli -m ~/{model}.gguf -c {context-size} -p "{your-prompt}"
```

Two instructions from that page matter in practice: keep the model file in the home directory "for best performance", and set the context size to something modest such as 4096, because otherwise "memory could spike and kill your terminal". The same guide describes a native Android Studio binding with hardware acceleration up to SME2 on Arm, and a cross-compile route with the Android NDK.

**3. An SDK inside your own app.** [LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview) is Google's runtime for on-device language models. Its overview lists Kotlin, Python, C++ and the CLI as stable, Swift and JavaScript as early preview, and Flutter as community-maintained. Backends: CPU on every platform it lists, GPU on Android, iOS, macOS, Windows and Linux, and NPU on Android (Windows is marked with the early-preview symbol). MLC LLM builds an iOS or Android package from a model list in a JSON config, where each entry carries an `estimated_vram_bytes` field; the example entry on the iOS page is a 3B model in `q4f16_1` with 2,960,000,000 bytes.

## The published phone numbers

The LiteRT-LM overview has a table of models with file size and speed on named devices. The phone rows:

| Model | File (MB) | Phone | CPU prefill / decode (tokens/s) | GPU prefill / decode (tokens/s) |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 521 | Samsung S24 Ultra | 251 / 30 | not listed |
| Gemma3-1B | 1005 | Samsung S24 Ultra | 177 / 33 | 1191 / 24 |
| Qwen2.5-1.5B | 1598 | Samsung S25 Ultra | 298 / 34 | 1668 / 31 |
| Gemma4-E2B | 2583 | Samsung S26 Ultra | 557 / 47 | 3808 / 52 |
| Gemma4-E2B | 2583 | iPhone 17 Pro | 532 / 25 | 2878 / 57 |
| Gemma4-E4B | 3654 | Samsung S26 Ultra | 195 / 18 | 1293 / 22 |
| Gemma4-E4B | 3654 | iPhone 17 Pro | 159 / 10 | 1189 / 25 |
| phi-4-mini | 3906 | Samsung S24 Ultra | 67 / 7 | 314 / 10 |
| Qwen3-0.6B | 586 | Vivo X300 Pro | 165 / 9 | 580 / 21 |

*Source: LiteRT-LM overview, "Supported Models" table. The page does not state the prompt length or quantization for each row in the part read here.*

![Left: decode speed against file size for the phone rows of the LiteRT-LM table, CPU as circles and GPU as squares. Right: text-generation speed against file size for Llama 3.1 8B at different llama.cpp quantization types]({{ site.baseurl }}/assets/pic/phone-llm-docs-numbers.png){: width="900" }
*Left: the phone rows above (different phones, so not a controlled comparison). Right: the llama.cpp quantize table for Llama 3.1 8B.*

Computed from those rows (the author's arithmetic, not a measured result): on the Gemma4-E2B CPU row for the Samsung S26 Ultra, a 1,000-token prompt takes about 1000 / 557 = 1.8 s to prefill, and a 100-token reply at 47 tokens/s takes about 2.1 s. For Gemma4-E4B on the same phone's CPU, the same prompt takes about 5.1 s and the reply about 5.6 s. The pattern across rows is that prefill is roughly 10 to 70 times faster than decode, and that the GPU's advantage is largest on prefill (3808 against 557 for Gemma4-E2B on the S26 Ultra). On decode the GPU gain varies: 52 against 47 on the S26 Ultra, 57 against 25 on the iPhone 17 Pro, and a loss for Gemma3-1B (24 against 33).

The rows are not a single-variable comparison. The phones differ, the models differ in architecture (the "E2B" and "E4B" names are the vendor's own), and decode speed does not track file size cleanly: Qwen3-0.6B, the second-smallest file, has the lowest CPU decode of the small models, but it is on a different phone from the others.

## What quantization does to size and speed

The [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) lists size, bits per weight, and speed for Llama 3.1 8B across quantization types. It does not say which hardware produced the speeds, so only the ratios are useful here:

- Size: F16 is 14.96 GiB at 16.0005 bits per weight; Q4_K_M is 4.58 GiB at 4.8944; Q2_K is 2.95 GiB at 3.1593; Q8_0 is 7.95 GiB at 8.5008. Q4_K_M is 3.27 times smaller than F16 (14.96 / 4.58).
- Text generation at 128 tokens: F16 29.17 tokens/s, Q8_0 50.93, Q4_K_M 71.93, Q2_K 79.85. Q4_K_M is 2.47 times faster than F16 (71.93 / 29.17).
- Prompt processing at 512 tokens: F16 923.49, Q4_K_M 821.81, Q8_0 865.09. Prefill barely moves with quantization, and F16 is the fastest of the three.

So quantization buys decode speed and memory, not prefill speed. The same README says that models are "currently fully loaded into memory", so memory and disk requirements are the same: the file size is the first estimate of the RAM a model needs, before the context cache and the app.

## A selection rule that follows from the documents

- Start from the file size. If the file is larger than the free RAM of the phone, it will not run, and the context cache comes on top.
- Choose 4-bit unless there is a reason not to: the table above shows the speed gain, and the README notes that quantization "may introduce some accuracy loss", measured as perplexity or KL divergence. The README does not give the size of that loss for small models.
- Set the context size deliberately (the llama.cpp guide's 4096 starting point) and keep prompts short, because prefill time grows with prompt length at the published rate.
- Run the app's own benchmark on the actual phone (PocketPal has a Benchmark page), because published rows are for specific phones and the numbers do not transfer.

## Limits

- Nothing here was run on a phone. The numbers are the vendors' and projects' own, produced under conditions the pages do not fully state (prompt length, quantization per row, thermal state, battery mode).
- The LiteRT-LM figures are from one vendor's page about its own runtime; the llama.cpp figures have no stated hardware. Neither is an independent benchmark.
- Phone rows are for flagship devices of the generations named in the table; older or mid-range phones are not covered by these sources.
- Quality is not covered: the documents give speed and size, not how good a 0.5B or 1B model's answers are for a given task.
- Pages change: the tables above were read from the live pages at the time of writing, and runtimes in this area are renamed and replaced often.
