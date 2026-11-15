---
title: "Small Language Models on Small Devices (3/6): Edge Boards, Raspberry Pi and Jetson"
date: 2027-01-26 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, edge-ai, raspberry-pi, jetson, llama-cpp, quantisation, series]
description: "Part 3 of 6. What the published specifications of NVIDIA Jetson Orin Nano and Raspberry Pi 5 imply for running a small language model: a memory-fit budget and a bandwidth-based speed ceiling, computed in code and compared with Arm's own Raspberry Pi 5 walkthrough."
---

Edge boards are the middle tier: 4–16 GB of RAM shared by CPU and accelerator, 7–25 W, and a Linux install. [Part 1](/posts/slm-edge-1-memory-bandwidth-arithmetic/) gave the two rules, memory fit and a bandwidth ceiling; [Part 2](/posts/slm-edge-2-phones-pcs-vendors/) covered phones. This part applies the rules to the published specifications of two boards, and reads the one official walkthrough that reports a memory measurement. No board was available here, so every speed below is a ceiling computed from a specification, not a measurement.

![Two panels. Left: grouped bars of the decode speed ceiling in tokens per second on a log axis for 0.5B, 1B, 3B, 3.8B and 8B 4-bit models at 17.5, 68 and 102 GB/s of memory bandwidth; the 8B model reaches 4.4, 17.0 and 25.5. Right: stacked bars of the memory budget on an 8 GB board for 1B, 3.8B and 8B models, where the 8B stack reaches 8.15 GB and crosses the 8 GB line.](/assets/pic/slm-edge-3-edge-boards.png)
*Left: tokens per second ceiling = bandwidth / bytes of weights. Right: weights, KV cache and an assumed 2 GB for the system must stay under the RAM line.*

## What the vendors publish

- **NVIDIA Jetson Orin Nano.** The [Orin Nano Super developer kit page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit) lists 8 GB of 128-bit LPDDR5 at 102 GB/s (68 GB/s before the software update that created the "Super" mode), 67 INT8 TOPS (up from 40), a 1024-core Ampere GPU with 32 tensor cores, a 6-core Arm Cortex-A78AE CPU, and 7–25 W. NVIDIA describes the update as aimed at generative AI in robotics and multimodal agents.
- **Raspberry Pi 5.** The [product page](https://www.raspberrypi.com/products/raspberry-pi-5/) lists a quad-core 2.4 GHz Arm Cortex-A76 CPU, LPDDR4X-4267 memory in 1, 2, 4, 8 and 16 GB sizes, and no NPU. LLM inference therefore runs on the CPU.
- **Arm.** The [Arm Learning Path for a Raspberry Pi 5 chatbot](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/llama-python-cpu/llama-python-chatbot) uses llama.cpp with a 3B-parameter model in the GGUF format at 4-bit weights, recommends the 8 GB board, and says the 4-bit type is common on edge and mobile applications because of the memory saving. It reports a peak of 2124.641 MiB for the running chatbot.

The pattern matches the phone tier: 4-bit weights, a runtime that reads a compact file format, and a model chosen to fit RAM first.

```python
# Memory fit and speed ceiling for edge boards, from published specifications.
GB=1e9
# Arm Learning Path: accelerate estimate for orca_mini_3b lists float32 12.42 GB; units of 1 GB = 1e9 are assumed here.
params=12.42e9/4
print(f"orca_mini_3b implied parameters: {params/1e9:.2f}B")
for name,b in [("float32",32),("float16",16),("int8",8),("int4",4)]:
    print(f"  {name:>7}: {params*b/8/GB:6.2f} GB")
print("  the Learning Path lists 12.42 / 6.21 / 3.11 / 1.55 GB, and a measured peak of 2124.641 MiB for the int4 chatbot")
print(f"  peak minus int4 weights: {2124.641*1.048576/1000 - params*0.5/GB:.2f} GB (runtime, KV cache, Python)")

# Jetson Orin Nano 8GB: 68 GB/s (original) and 102 GB/s (Super), 8 GB LPDDR5 shared by CPU and GPU (NVIDIA specification).
print("\nupper bound on decode tokens/s = bandwidth / weight bytes (batch 1, ignores KV cache and compute)")
models=[("0.5B int4",0.5e9*0.5),("1B int4",1e9*0.5),("3B int4",3e9*0.5),("3.8B int4",3.8e9*0.5),("8B int4",8e9*0.5),("3B fp16",3e9*2)]
bws=[("this CPU, measured",17.5),("Orin Nano 68 GB/s",68),("Orin Nano Super 102 GB/s",102)]
print(f"{'model':>10} {'GB':>6} " + " ".join(f"{n:>26}" for n,_ in bws))
for n,by in models:
    print(f"{n:>10} {by/GB:>6.2f} " + " ".join(f"{bw/(by/GB):>26.1f}" for _,bw in bws))

# RAM budget on an 8 GB board: weights + KV cache must fit in what the OS leaves free.
def kv(L,d,ctx,bits=16,kv_frac=1.0): return 2*L*d*ctx*bits/8*kv_frac
print("\n8 GB board, 4-bit weights, fp16 KV cache with no key-value sharing, context 4096, assumed 2 GB for OS and runtime; model shapes are illustrative")
for n,P,L,d in [("1B (16 layers, d=2048)",1e9,16,2048),("3.8B (32 layers, d=3072)",3.8e9,32,3072),("8B (32 layers, d=4096)",8e9,32,4096)]:
    w=P*0.5; c=kv(L,d,4096); tot=(w+c)/GB+2
    print(f"{n:>26}: weights {w/GB:.2f} + KV {c/GB:.2f} + 2.0 = {tot:.2f} GB  fits={tot<=8}")
```

```
orca_mini_3b implied parameters: 3.10B
  float32:  12.42 GB
  float16:   6.21 GB
     int8:   3.10 GB
     int4:   1.55 GB
  the Learning Path lists 12.42 / 6.21 / 3.11 / 1.55 GB, and a measured peak of 2124.641 MiB for the int4 chatbot
  peak minus int4 weights: 0.68 GB (runtime, KV cache, Python)

upper bound on decode tokens/s = bandwidth / weight bytes (batch 1, ignores KV cache and compute)
     model     GB         this CPU, measured          Orin Nano 68 GB/s   Orin Nano Super 102 GB/s
 0.5B int4   0.25                       70.0                      272.0                      408.0
   1B int4   0.50                       35.0                      136.0                      204.0
   3B int4   1.50                       11.7                       45.3                       68.0
 3.8B int4   1.90                        9.2                       35.8                       53.7
   8B int4   4.00                        4.4                       17.0                       25.5
   3B fp16   6.00                        2.9                       11.3                       17.0

8 GB board, 4-bit weights, fp16 KV cache with no key-value sharing, context 4096, assumed 2 GB for OS and runtime; model shapes are illustrative
    1B (16 layers, d=2048): weights 0.50 + KV 0.54 + 2.0 = 3.04 GB  fits=True
  3.8B (32 layers, d=3072): weights 1.90 + KV 1.61 + 2.0 = 5.51 GB  fits=True
    8B (32 layers, d=4096): weights 4.00 + KV 2.15 + 2.0 = 8.15 GB  fits=False
```

## Fit: the Arm numbers reproduce

Arm's memory table lists 12.42 GB in float32 for the model, which implies 3.10B parameters. Multiplying by bits over 8 reproduces the listed 6.21, 3.11 and 1.55 GB for 16, 8 and 4 bits to the printed digits. The measured peak of 2124.641 MiB (2.23 GB) exceeds the 1.55 GB of 4-bit weights by about 0.68 GB; the walkthrough does not break this down, and the likely contributors are the runtime, the KV cache and Python (the author's reading, not tested).

The right panel adds the KV cache. With 4-bit weights, an fp16 cache without key-value sharing at context 4096, and 2 GB assumed for the operating system, a 1B model needs 3.04 GB and a 3.8B model needs 5.51 GB, both within 8 GB. An 8B model needs 8.15 GB and does not fit. The layer counts and widths are illustrative and the 2 GB figure is an assumption, so the verdict for the 8B case depends on both; Part 1 showed how sharing the cache moves the second term.

## Speed: bandwidth sets the ceiling

Dividing bandwidth by the bytes of weights gives, for a 3B model at 4 bits (1.5 GB), 45 tokens per second at 68 GB/s and 68 at 102 GB/s; for 8B (4 GB), 17 and 25.5. The 17.5 GB/s of the desktop CPU measured in Part 1 gives 11.7 and 4.4. These are batch-1 ceilings that ignore the KV cache traffic and the compute, so real speeds are lower, and Raspberry Pi 5 does not appear in the table because its page, as read, gives the memory type but no bandwidth figure.

## Where the tier is used

Both vendors point to robotics, multimodal agents and offline use as the reasons for edge boards (NVIDIA's text names robotics and multimodal agents; Arm's walkthrough is a local chatbot with no cloud services). Neither document reports an accuracy or latency target for a specific product. What the specifications support is a narrower claim: a 1–4B model at 4 bits fits and can respond at interactive speed; an 8B model is borderline on 8 GB.

## Limits

- Nothing here was run on a Jetson or a Raspberry Pi. The tokens-per-second values are ceilings from published bandwidth; real throughput depends on the kernels, the accelerator path, thermal limits and the power mode.
- The 2 GB system allowance and the model shapes in the memory budget are assumptions, not vendor data.
- The 2124.641 MiB peak is Arm's measurement for one model on one board; the extra 0.68 GB is not explained by the document.
- Prices, memory sizes and software modes change; the specifications were read from the pages as retrieved.

Next: Part 4, microcontrollers.
