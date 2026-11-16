---
title: "Run a Small Language Model on a Raspberry Pi: A Step-by-Step Guide"
date: 2027-02-18 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, edge-ai, raspberry-pi, llama-cpp, how-to]
description: "A hands-on guide: install a 64-bit OS, build llama.cpp, download a 0.5B or 1.5B 4-bit model, chat in the terminal, serve an HTTP API, and measure your own tokens per second."
---

This guide takes a Raspberry Pi from a blank SD card to a local chatbot and an HTTP API, using [llama.cpp](https://github.com/ggml-org/llama.cpp) and 4-bit models in the GGUF format. It ends with a benchmark command so you can compare your board with the arithmetic in [Part 1](/posts/slm-edge-1-memory-bandwidth-arithmetic/) and [Part 3](/posts/slm-edge-3-edge-boards-jetson-raspberry-pi/) of the small-devices series. Read the "What was tested" box first.

## What was tested

The llama.cpp commands below were built and run on an x86-64 Linux machine (16 cores, 16 GB), restricted to 4 threads. They were not run on a Raspberry Pi. The Pi-specific steps (imaging, package installation) follow the official documentation. Speeds measured on that x86 machine are not Pi speeds and are not given as such; the benchmark step tells you how to get yours.

## 0. What you need

- A Raspberry Pi 5 (the 8 GB model is the one [Arm's tutorial](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/llama-python-cpu/llama-python-chatbot) prefers) or any Arm Linux board running a 64-bit OS. A smaller-RAM board works with the smaller model in step 4.
- A microSD card of 32 GB or more, a power supply that matches your board, and a network connection.
- A way to cool the board. The author's expectation, not a measurement: sustained generation keeps all cores busy, so a case with a fan or heatsink avoids throttling.

## 1. Install a 64-bit Raspberry Pi OS

Flash the SD card with [Raspberry Pi Imager](https://www.raspberrypi.com/software/) and choose the 64-bit Raspberry Pi OS; the [getting-started guide](https://www.raspberrypi.com/documentation/computers/getting-started.html) covers the steps. In the imager's settings, set a hostname, a user and SSH access so you can work from another computer. Boot the board, then check:

```bash
uname -m      # expect: aarch64
free -h       # total RAM
nproc         # CPU cores
```

If `uname -m` prints `armv7l`, you installed a 32-bit OS; reflash with the 64-bit image.

## 2. Install the build tools

```bash
sudo apt update
sudo apt install -y build-essential cmake git libssl-dev
```

`libssl-dev` is optional. The [llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) says it enables HTTPS features, such as downloading a model directly from Hugging Face with the `-hf` option. Without it the build still works.

## 3. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j 4 --target llama-cli llama-server llama-bench
```

This is the CPU build from the official guide, restricted to three programs. `-j 4` runs four compile jobs; on an 8 GB board it keeps memory use modest. On the x86 test machine the same build took 1 minute 23 seconds with 16 jobs; expect much longer on a Pi.

The programs end up in `build/bin/`:

- `llama-cli`: chat in the terminal.
- `llama-server`: an HTTP server with an OpenAI-style API.
- `llama-bench`: a speed test.

## 4. Choose and download a model

Rule of thumb from Part 1: the 4-bit weights are about half a byte per parameter, and the weights plus the KV cache plus the operating system must fit in RAM. Two models used here, both Qwen2.5 Instruct in 4-bit `Q4_K_M` form:

| Model | File size | Fits on |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 491 MB (462.96 MiB in llama.cpp) | 2 GB boards and up |
| Qwen2.5-1.5B-Instruct | 1.12 GB (1.04 GiB in llama.cpp) | 4 GB boards and up |

The model pages are [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) and [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF). Download the `qwen2.5-1.5b-instruct-q4_k_m.gguf` file from the 1.5B page ("Files" tab) with a browser, or from the command line:

```bash
mkdir -p ~/models && cd ~/models
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

The `wget` line is the standard Hugging Face file URL pattern and was not run here: the author's network blocks that host, and the files were fetched from a mirror instead. The same files, 491,400,032 and 1,117,320,736 bytes, were then used for every command in this guide. If the download is interrupted, check the size with `ls -l`.

## 5. Chat in the terminal

```bash
cd ~/llama.cpp
./build/bin/llama-cli -m ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf -t 4 -c 2048
```

- `-t 4` sets 4 threads, one per Pi 5 core.
- `-c 2048` sets the context to 2,048 tokens. The KV cache grows with this number, so on a small board a smaller value saves memory.

The program opens an interactive conversation; type a message and press Enter. To run a single question and exit, add `--single-turn -p "your question"`. On the test machine this printed the reply followed by a line of this form:

```
[ Prompt: 173.6 t/s | Generation: 19.0 t/s ]
```

The first number is prompt processing, the second is generation, in tokens per second. Generation is the number that decides how fast the text appears.

## 6. Serve an API

```bash
./build/bin/llama-server -m ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf -t 4 -c 2048 --host 127.0.0.1 --port 8081
```

From a second terminal:

```bash
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24}'
```

On the test machine `/health` returned `{"status":"ok"}` and the second call returned a chat completion ("Hello, how may I help you?") with a `usage` block of token counts. The default port is 9931 in the version tested; passing `--port` explicitly avoids a clash with whatever already runs on your board.

`--host 127.0.0.1` accepts local connections only. The server prints a warning that no API key is set; do not expose it to a network you do not control. To reach it from another computer you would set `--host 0.0.0.0` and an API key, and you should read the server's documentation first.

## 7. Measure your own speed

```bash
./build/bin/llama-bench -m ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p 64 -n 32 -t 4 -r 2
```

`pp64` is prompt processing of 64 tokens, `tg32` is generating 32 tokens; `-r 2` repeats twice. The table on the x86 test machine, with 4 threads:

| Model | pp64 (tokens/s) | tg32 (tokens/s) |
|---|---|---|
| Qwen2.5-0.5B Q4_K_M | 402.65 | 48.73 |
| Qwen2.5-1.5B Q4_K_M | 185.92 | 19.15 |

Your Pi will give different numbers; these show the shape, not the speed. Two things to check on your results:

- Generation speed should fall roughly in proportion to the model size: the 1.5B file is about 2.3 times the 0.5B file and generated about 2.5 times slower here. This is the bandwidth rule from Part 1: each token reads all the weights once.
- A speed ceiling is memory bandwidth divided by the weight size. [Raspberry Pi's product page](https://www.raspberrypi.com/products/raspberry-pi-5/) gives the memory type, and Part 3 shows the calculation. Your measured `tg32` should sit below it.

Try 1, 2, 3 and 4 threads with `-t`. If more threads stop helping before 4, memory bandwidth, not compute, is the limit.

## 8. If something goes wrong

- **Out of memory or the process is killed:** pick the smaller model, lower `-c`, close other programs. `free -h` shows what is available.
- **Slow after a minute of use:** the board may be throttling from heat. Raspberry Pi OS includes a `vcgencmd` tool for reading firmware status (check its documentation for the throttling flags); better cooling is the fix.
- **`error: invalid argument`:** options change between llama.cpp versions. Run `./build/bin/llama-cli --help` to see what your build accepts. The commands here were tested on build `ef6bc6f`.
- **The server fails to start with "couldn't bind":** the port is in use; choose another with `--port`.
- **Gibberish or off-topic answers:** the 0.5B model is small. Try the 1.5B, and keep questions short and specific.

## Limits

- Nothing here ran on a Raspberry Pi. The build time, speeds and memory use on your board are unknown until you run step 7.
- Only two models of one family were tested; other families use the same commands but have other file names.
- The Hugging Face `-hf` download option and the `wget` URL were not tested, as described in step 4.
- Cooling advice and the `vcgencmd` note are the author's general knowledge, not tested on a board.
- A 1.5B model at 4 bits makes visible errors; the answer in step 5 contained a weak explanation. It is a way to learn how local inference works, not a replacement for a large model.
