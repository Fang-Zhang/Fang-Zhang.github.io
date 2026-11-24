---
title: "LLMs on the Farm, Part 3: The Model Is One Stage in a Pipeline, and Latency Is the Budget"
date: 2027-06-02 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, agriculture, architecture, deployment, voice, rag, latency, llms on the farm]
description: "Part 3 of 6 on large language models in agriculture. Deployed farm advisory systems share one shape: a channel and speech front end, an orchestration layer, retrieval over a curated knowledge base, and a model. Published stage times show a text path already at 9 seconds, so a 5 second voice answer is a budget problem, not a model problem."
---

## What this post finds

Parts 1 and 2 chose a base model and a way to give it farm knowledge. Part 3 asks what a deployed system looks like around the model, using two public reports from live services: the [AIEP technical learnings](https://arxiv.org/abs/2601.11537) (five advisory services in Kenya and Bihar, India, summarised by the funders and implementers) and the [Farmer.Chat paper](https://arxiv.org/abs/2409.08916) with its engineering write-up from [Digital Green](https://medium.com/digitalgreen-techblog/under-the-hood-of-farmer-chat-journey-to-an-optimised-production-ready-rag-powered-chatbot-589bf5716e27).

- **The shape is shared.** All five AIEP services use a two-part design: an interface (phone line, WhatsApp or an app, with speech recognition, translation and speech synthesis) and a reasoning component (query orchestration, outside data such as weather and soil, retrieval over a curated corpus, and the model).
- **Channel follows the user.** WhatsApp and apps reach smartphone users; telephone lines were indispensable for women farmers and older users with limited digital access. One cohort moved from Telegram to an Android app because first-time users connect more easily with images and catalogues than with typing or recording a question.
- **Voice is where latency breaks.** Voice needs at least four model calls (speech recognition, translation, the language model, speech synthesis). The report says that consistently answering in under 5 seconds remains hard for these services. One cohort cut mean latency by a factor of 25, to just under 10 seconds, by hosting in-country and sending smaller audio files.
- **The retrieval path alone is slow.** Farmer.Chat's text path averaged 9.05 seconds, of which 3.15 went to reranking retrieved passages and 2.71 to generation. Even with reranking removed, the published numbers leave the text path at 5.90 seconds before any speech stage is added.
- **Most pipeline growth came from fixing failures, one at a time.** Six versions in the engineering write-up each fix an observed problem: hallucination, a vector store that did not scale, unanswered questions, retrieval confused by common words, incomplete recall.

![Two panels. Left: stage times of a deployed farm chatbot's text path, 9.05 seconds in total (3.19 everything else, 3.15 rerank, 2.71 generation), against a 5 second voice target; removing the rerank step gives 5.90 seconds, and also halving generation gives 4.54 seconds. Right: upload time for 10 seconds of speech. 16 kHz PCM takes 40, 10 and 2.5 seconds at 64, 256 and 1024 kilobits per second; 8 kHz mu-law takes 10, 2.5 and 0.62; Opus at 16 kilobits per second takes 2.5, 0.62 and 0.16]({{ site.baseurl }}/assets/pic/farm-3-architecture.png){: width="900" }
_Figure: left, published stage times of a deployed text path against a 5 second target (the two lower bars are arithmetic on those numbers); right, upload-time arithmetic for 10 seconds of speech. Both are computed by the script at the end of this post._

## The reference architecture

The AIEP report describes the common design as two parts joined by a defined interface between modules.

**Interface.** Channels in use were telephone lines on existing interactive voice response (IVR) systems, SMS, smartphone apps, and messaging platforms. The report calls the combination of WhatsApp or an app for smartphone users and a phone line for feature-phone users the most common choice. A phone line needs a speech stack of at least speech recognition and speech synthesis. One cohort added a denoiser to improve recognition, and several ran a small speech-detection model that cuts silence so the recogniser receives shorter audio. Multilingual service in English, Swahili and Hindi was mostly done through machine translation around an English-centred reasoning component, although two cohorts built multilingual retrieval over Swahili content.

**Reasoning component.** Four modules recur: a query orchestration and classification module, external services (weather forecasts, soil maps, market prices where available), a retrieval-augmented generation system, and the knowledge base. The report notes that the orchestration module was built outside the model because function calling was only emerging, and expects such modules to become unnecessary as function calling and the Model Context Protocol mature. That is the report's expectation, not a result.

**Farmer.Chat's version of the same pieces.** The [paper](https://arxiv.org/abs/2409.08916) describes a knowledge base builder (ingesting documents, tables and videos, then summarising, tagging and chunking them), a response generator using retrieval, a query orchestration layer with a planning agent, an execution agent and a tool service, language and voice support, and WhatsApp and Telegram front ends. The paper reports deployment in four countries with over 15,000 farmers and over 300,000 queries.

**Language handling is a design choice with an expiry date.** Farmer.Chat's evaluation found that translating the query to English, answering, and translating back beat answering directly in the farmer's language. The AIEP report, written later, notes that current models can combine translation and reasoning for languages such as Swahili and Hindi, and that integrating the translation step into the model is under test. Lower-resourced languages (Gikuyu, Bhojpuri) remain hard because speech and translation components either do not exist or do not connect well, especially for non-standard scripts.

## What six versions of a retrieval path look like

Digital Green's engineering write-up is a rare account of a production retrieval pipeline growing. It is self-reported and its test sets are small, which is why the numbers below are described as what the team observed, not as benchmarks.

- **Version 1** was a two-week sprint: one or two crops in one region, about 50 curated test questions, a messaging bot and an off-the-shelf orchestration library. It did not answer off-topic questions, but it did answer on-topic questions that the content did not cover.
- **Version 2** moved to direct model calls for more control and added a model-based step that filters and ranks retrieved passages. On about 30 test questions it removed 98.3 percent of irrelevant passages and raised fully correct answers from roughly 76 to roughly 100 percent. The write-up also records the side effect: faithful answers meant more unanswered questions.
- **Versions 3 to 6** replaced an in-memory vector store whose compute demand grew with the number of regional bot instances (their benchmark found Qdrant best on throughput and on recall at 100), added intent routing (about 11 percent of unanswered questions were users asking about a different crop than the one selected, and 2 to 3 percent were greetings or goodbyes treated as questions), added a model call that shortens the query to its core theme (so that a frequent word such as a country name stops pulling in irrelevant passages), and moved to chunking by topic.

The paper's own numbers show where the remaining gaps sit. About 75 percent of questions were answered; of the unanswered ones, 66 percent were content gaps, 23 percent out of scope and 11 percent about unsupported crops. The pipeline's quality limit is mostly the knowledge base, which is Part 4's subject.

The paper's timing and the write-up's version history are separate documents; the stage names match, but the paper does not say it measures version 6.

## Where the model runs

The reports give three placements and no clean comparison between them.

- **A hosted model through a provider's API** is the default in both reports. It costs the network distance to the provider's servers and the size of whatever is uploaded.
- **In-country hosting or a nearby provider region** is what the 25-fold latency reduction used, together with smaller audio. The report says this was possible mainly because of infrastructure available in India.
- **On the phone** (speech models or the whole reasoning model) is described as the most thorough fix for network latency and for patchy connectivity, and it would also avoid paying to send audio. The report states that the quality and speed cost of the smaller models is unclear. No farm-specific measurement of it was found for this post.

## The experiment: a latency budget from published numbers

No new model is trained here. The script takes Farmer.Chat's published stage times and standard audio formats, and answers two questions: how far is a retrieval path from a 5 second voice target before any speech stage is added, and does the upload of the question fit on a slow link?

```python
# Latency and payload budget for a voice farm-advisory pipeline.
# Published inputs (Farmer.Chat, text path): mean 9.05 s; rerank 3.15 s; generation 2.71 s.
total, rerank, gen = 9.05, 3.15, 2.71
other = round(total - rerank - gen, 2)
print("Text path, mean seconds: total", total, "| rerank", rerank, "| generation", gen, "| everything else", other)
target = 5.0
print("Gap to a 5 s target before any voice stage:", round(total - target, 2), "s")
s1 = round(total - rerank, 2); print("Without the rerank step:", s1, "s; headroom to target:", round(target - s1, 2))
s2 = round(s1 - gen / 2, 2); print("Without rerank and generation halved:", s2, "s; headroom to target:", round(target - s2, 2))
print("Share of queries at or below 7.5 s:", 100 - 68 - 15, "percent (68 percent fall in 7.5-12.5 s, 15 percent above 12.5 s)")
print()
secs = 10; enc = {"16 kHz 16-bit PCM": 16000 * 2 * secs, "8 kHz mu-law": 8000 * secs, "Opus at 16 kbit/s": 16000 // 8 * secs}
print(f"Upload time for {secs} s of speech (payload only, no protocol overhead)")
print(f"{'encoding':22s}{'bytes':>9s}" + "".join(f"{k:>10d}" for k in (64, 256, 1024)) + "  kbit/s link")
for n, b in enc.items():
    print(f"{n:22s}{b:9d}" + "".join(f"{b * 8 / (k * 1000):10.2f}" for k in (64, 256, 1024)))
print()
print("Cohort that cut mean latency 25x to just under 10 s implies a starting mean of roughly", 25 * 10, "s.")
```

Output:

```
Text path, mean seconds: total 9.05 | rerank 3.15 | generation 2.71 | everything else 3.19
Gap to a 5 s target before any voice stage: 4.05 s
Without the rerank step: 5.9 s; headroom to target: -0.9
Without rerank and generation halved: 4.54 s; headroom to target: 0.46
Share of queries at or below 7.5 s: 17 percent (68 percent fall in 7.5-12.5 s, 15 percent above 12.5 s)

Upload time for 10 s of speech (payload only, no protocol overhead)
encoding                  bytes        64       256      1024  kbit/s link
16 kHz 16-bit PCM        320000     40.00     10.00      2.50
8 kHz mu-law              80000     10.00      2.50      0.62
Opus at 16 kbit/s         20000      2.50      0.62      0.16

Cohort that cut mean latency 25x to just under 10 s implies a starting mean of roughly 250 s.
```

What it shows, and what it does not:

- **The text path misses the target by 4.05 seconds on its own.** Removing the rerank step still leaves 5.90 seconds. Halving generation as well reaches 4.54, which is a hypothetical that leaves under half a second for speech recognition, translation and speech synthesis together. The report's levers are fewer sequential calls (giving more of the work to the language model), in-country hosting and smaller audio.
- **The authors of the paper regard 5 to 10 seconds as acceptable for a text chat, and report that none of the users in their studies raised response time as a concern.** The 5 second target applies to speech, where, as a footnote in the AIEP report puts it, the reader is invited to count slowly to five to feel what that delay means.
- **Audio size matters on slow links.** Ten seconds of uncompressed 16 kHz audio is 320 kB and takes 40 seconds to upload at 64 kbit/s. A speech codec at 16 kbit/s makes it 20 kB and 2.5 seconds. The Opus figure is the nominal bitrate, and protocol overhead is ignored. The 25-fold cohort result is consistent with this arithmetic, but the report does not break its gain down between hosting and audio size.
- **Limits.** The stage times come from one text deployment, and the voice stages are not measured here. The codec rates are nominal. The figures show where time can go, not what any specific voice service achieves.

## What a design review should ask

1. Which phones do the farmers use, and does the first channel reach the women and older farmers who are least likely to type?
2. How many sequential model calls sit between the question and the answer, and which can be merged or skipped?
3. Where is the model hosted relative to the user, and how large is the upload?
4. Which languages have working speech recognition and synthesis, as opposed to translation alone?
5. Who owns the knowledge base? Content gaps were the largest cause of unanswered questions in the one deployment that published the split.

## What comes next

Part 4 covers life after launch: keeping the knowledge base current, monitoring answer quality, and the evaluation sets that the AIEP report publishes for this purpose.

## References

- AIEP Initiative, [Building AI-based advisory services for smallholder farmers: Technical learnings from the AIEP Initiative](https://arxiv.org/abs/2601.11537), January 2026.
- Singh, Wang'ombe, Okanga et al., [Farmer.Chat: Scaling AI-Powered Agricultural Services for Smallholder Farmers](https://arxiv.org/abs/2409.08916), 2024.
- Singh, [Under the Hood of Farmer.chat: Journey to an optimised, production ready RAG powered Chatbot](https://medium.com/digitalgreen-techblog/under-the-hood-of-farmer-chat-journey-to-an-optimised-production-ready-rag-powered-chatbot-589bf5716e27), Digital Green engineering blog, July 2024.
