---
title: "Companion AI (2/8): Persona Drift and How to Measure It"
date: 2027-03-15 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, companion-ai, persona, system-prompt, attention, series]
description: "A self-chat measurement of how fast a 1.5B instruction-tuned model stops following its system prompt, what the attention to that prompt does as the conversation grows, and what a one-sentence reminder changes."
---

A companion is defined by its persona, and the persona usually lives in one place: the system prompt at the top of the context. This post measures what happens to that prompt over a conversation. It reruns, at small scale, the self-chat protocol of [Li et al.](https://arxiv.org/abs/2402.10962) on a 1.5B model on CPU, records how often each rule is followed round by round, measures how much attention the last token pays to the system prompt, and tests the cheapest fix. [Part 1](/posts/companion-1-requirements/) listed persona stability as the first requirement; this is the measurement behind it.

## What the literature claims

- Li et al. propose a benchmark of self-chats between two instructed chatbots and report significant instruction drift within eight rounds of conversation for LLaMA2-chat-70B and GPT-3.5. Their analysis points to attention decay over long exchanges, and they propose split-softmax, a lightweight method that compares favourably against two baselines.
- [StreamingLLM](https://arxiv.org/abs/2309.17453) describes attention sinks: strong attention scores toward initial tokens even when they are not semantically important.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) finds that models use information best at the start or end of a long context and worst in the middle.

Those three facts predict the setup here: the system prompt sits at the very start, its first token is a sink, and each round adds text between the prompt and the place where the next reply is written.

## Protocol

Two bots, A and B, chat with each other. Each sees its own earlier turns as `assistant` and the other's as `user`. Each has one checkable rule.

```python
SYS = {'A': "You are Ahoy, a cheerful sailor chatting with a friend. Rules: always end every message "
            "with the word 'matey'. Keep every message under 30 words.",
       'B': "You are Quill, a quiet poet chatting with a friend. Rules: always include the word "
            "'moonlight' in every message. Keep every message under 30 words."}

def check(w, text):
    t = text.strip(); words = len(t.split())
    ok = (re.sub(r'[^a-z]', '', t.lower().split()[-1]) == 'matey') if w == 'A' else ('moonlight' in t.lower())
    return dict(marker=ok, short=words <= 30)
```

- Model: Qwen2.5-1.5B-Instruct, bfloat16, CPU, sampling at temperature 0.7 and top-p 0.9, 48 new tokens per reply.
- 12 conversations in parallel, each with a different opening line, 8 rounds, so 12 replies per bot per round.
- Two conditions. In `base`, the rules appear once, in the system prompt. In `remind`, the incoming message also carries a short reminder, and only the current message does; the stored history does not, so the cached prefix is unchanged.

```python
REM = {'A': " (Reminder: end your message with the word 'matey'; under 30 words.)",
       'B': " (Reminder: include the word 'moonlight'; under 30 words.)"}
msg = incoming[i] + (REM[w] if MODE == 'remind' else '')
```

The attention measurement takes the finished `base` transcripts, rebuilds each bot's prompt with k earlier rounds, runs one forward pass with attention outputs, and sums the attention from the last prompt position onto the system-prompt tokens, averaged over all 28 layers and 12 heads. The system block is 40 tokens, and the first of them is the likely sink, so the mass is also reported without that first token.

$$m_{\text{sys}} = \frac{1}{LH}\sum_{l,h}\sum_{j \in S}\operatorname{softmax}_j\!\left(\frac{q_{l,h}\cdot k_{l,h,j}}{\sqrt{d}}\right)$$

## Results

![Rule adherence by round for the two conditions, and attention mass on the system prompt against prompt length]({{ site.baseurl }}/assets/pic/companion-2-persona-drift.png){: width="900" }
*Left: share of replies following the rule, both bots pooled. Middle: the same per bot. Right: attention mass on the 40 system-prompt tokens from the last prompt position, against prompt length.*

Pooled over both bots (24 replies per round):

- `base`: 0.50 in round 1, then 0.38, 0.13, 0.42, 0.38, 0.38, 0.25, 0.17 in round 8. Rounds 1–2 average 0.44 and rounds 7–8 average 0.21.
- `remind`: 0.50 in round 1, then 0.58, 0.63, 0.75, 0.71, 0.67, 0.71, 0.71. Rounds 1–2 average 0.54 and rounds 7–8 average 0.71.

Per bot, the two rules behave differently:

- B's rule (include "moonlight") went from 0.75 in round 1 to 0.33 in round 8 in `base`, and sat at 1.00 from round 4 on in `remind`.
- A's rule (end with "matey") is strict and rarely holds at all: 0.10 of A's replies in `base` and 0.39 in `remind`. The looser question of whether "matey" appears anywhere gives 0.51 in `base` and 0.92 in `remind`. In `base` the model keeps the flavour and loses the positional constraint.
- The length rule (under 30 words) held in 0.75 of replies in `base` and 0.86 in `remind`.

Attention mass on the system prompt, from the 16 prompt lengths measured:

- Bot A: 0.568 at 55 tokens, 0.390 at 127, 0.299 at 584. Without the first token: 0.195, 0.091, 0.041.
- Bot B: 0.538 at 78 tokens and 0.296 at 612. Without the first token: 0.192 and 0.034.

A uniform spread over all keys would give the 39 non-sink system tokens a share of 39/n: 0.72 at 54 tokens and 0.067 at 583. The measured mass fell from 0.195 to 0.041 over that range, a factor of about 4.8 where uniform dilution would give about 11. So the system prompt loses attention more slowly than plain dilution would, and most of the early mass sits on the first token.

## Reading the results

- Drift is visible in a 1.5B model within eight rounds in `base`, which agrees in direction with the eight-round figure in the paper for much larger models. The two experiments differ in rules, models and metric, so the agreement is qualitative.
- Attention to the system prompt falls as the prompt grows, which fits the attention-decay account. The experiment shows a correlation. It does not show that the fall causes the drift: adherence is not monotone in the table (round 3 is the lowest pooled point in `base`, then it recovers to 0.42), and a model can also drift because its own earlier replies, which break the rules, become the strongest in-context examples.
- A one-sentence reminder in the newest message raised adherence in every round after the first, and the effect grows with length. It costs about one sentence of prompt per turn and leaves the cached prefix untouched. In round 1, where the reminder sits next to the system prompt anyway, the pooled rates are equal at 0.50.
- Split-softmax is the paper's own remedy and changes the attention computation; it was not tested here.

## Limits

- 12 conversations per condition and 12 replies per bot per point: a single point has a standard error of roughly 0.14, so only the broad shapes (B in `remind`, the early-to-late fall in `base`) are reliable, not individual points.
- One small model, one sampling setting, two synthetic rules, and an 8-round horizon. A companion persona has many soft traits rather than two checkable ones, and the conversation partner is a person, not another instance of a model.
- Strict string checks undercount style that is kept but not in the exact required form, as the "matey" figures show. The 48-token cap truncates some replies, which affects the length and end-position checks.
- The attention analysis uses the `base` transcripts only, a mean over all layers and heads hides head-level behaviour, and the sink share is inferred from removing the first token, not from a per-head analysis.
- The reminder helped here; whether it helps on large models, long personas or user-driven conversations is not shown. A reminder can also be repeated back by the model or make replies repetitive, which was not measured.

Next, in Part 3 of this series: memory, and what to retrieve rather than re-read.
