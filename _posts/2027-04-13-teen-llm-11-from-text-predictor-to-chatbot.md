---
title: "How Language Models Work, Part 11: From Text Predictor to Chatbot"
date: 2027-04-13 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, fine-tuning, chatbot, chat template, beginners, teens]
description: "Part 11 of a 12-part series for readers aged 14 to 17. A chatbot is a text predictor trained a second time on conversations. This post fine-tunes the Part 8 Transformer on ten tiny question-and-answer pairs, measures what it gains and what it loses, and cites the InstructGPT paper and the Hugging Face chat-template docs."
---

Parts 1 to 10 built a text predictor. A chatbot looks different, but the Hugging Face documentation states the central fact plainly: "All causal LMs, whether chat-trained or not, continue a sequence of tokens." This post turns the Part 8 model into a very small chatbot and measures what the change costs.

## What changes: the data, not the machine

Training happens in two steps.

1. **Pre-training.** What Parts 2 to 8 did: predict the next word on a huge pile of text. The result is called a **base model**.
2. **Fine-tuning.** Keep training the same model, with the same loss, on a much smaller set of examples that look like the behaviour you want. In the [Hugging Face guide](https://huggingface.co/docs/transformers/main/en/chat_templating), chat fine-tuning means "training them on data that is formatted as a sequence of messages".

A conversation has to become a stream of tokens. The guide says the list of roles and messages "get converted to a token sequence, often with control tokens like `<|user|>` or `<|assistant|>`", and that "different models may use different formats". The format is called a **chat template**. In the code below, three new control words stand for it: `<user>`, `<assistant>` and `<end>`, added to the vocabulary the way Part 3 added tokens.

## The maths tool: a mask

Which words should the model be graded on? Not the question; the user wrote that. Only the answer. A **mask** is a list of 0s and 1s, one per word: multiply each word's loss by its mask value, add up, and divide by the number of 1s.

![Left: a chat stream of 15 words, with control words in blue, the question in grey and the answer in orange, and a row of masks under it, 0 for the question and 1 for the answer. Right: after fine-tuning, replies end correctly 100 percent of the time but the loss on held-out book text rises from 3.89 to 6.66]({{ site.baseurl }}/assets/pic/teen-llm-finetune.png){: width="900" }
*Left: one training example and its mask. Right: measured, below.*

## The experiment

Start from the Part 8 Transformer, stopped at 250 steps (near its lowest test loss). Ten question-and-answer pairs, hand-written with words it knows; seven for fine-tuning, three held back as new questions. The core code:

```python
def enc(text): return [idx.get(w, 500) for w in re.findall(r"[a-z']+|[.,!?;]", text)]
train_pairs, new_pairs = pairs[:7], pairs[7:]          # 3 questions held back

def chat_batch(ps, n=16):
    xs, ys, ms = [], [], []
    for _ in range(n):
        q, a = ps[torch.randint(len(ps), (1,)).item()]
        seq = [USER] + enc(q) + [ASSIST] + enc(a) + [END]
        start = len(enc(q)) + 2                          # first answer position in seq
        seq += [0] * (T + 1 - len(seq))
        x, y = seq[:T], seq[1:T + 1]
        m = [1 if i + 1 >= start and y[i] != 0 else 0 for i in range(T)]   # 1 = count this word's loss
        xs.append(x); ys.append(y); ms.append(m)
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(ms, dtype=torch.float)

def masked_loss(model, x, y, m):
    per_word = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction="none")
    return (per_word * m.reshape(-1)).sum() / m.sum()   # average over answer words only

# m is the Part 8 model after 250 pre-training steps
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
for t in range(1, 301):
    x, y, mk = chat_batch(train_pairs); loss = masked_loss(m, x, y, mk); opt.zero_grad(); loss.backward(); opt.step()
    if t in (50, 100, 300): report(m, f"after {t} fine-tune steps")
```

(`reply` samples words after `<assistant>` until the model writes `<end>` or 20 words pass; `report` runs it 20 times per question. The full scripts are in the same style as Part 8.)

Real output:

```
before fine-tuning     test loss on the book 3.89 | reply ends with <end>: trained q 0%, new q 0%
after 50 fine-tune steps test loss on the book 6.34 | reply ends with <end>: trained q 99%, new q 100%
after 100 fine-tune steps test loss on the book 6.38 | reply ends with <end>: trained q 100%, new q 100%
after 300 fine-tune steps test loss on the book 6.66 | reply ends with <end>: trained q 100%, new q 100%
```

Two things happened.

- **The habit arrived in 50 steps.** Before, no reply ever ended with `<end>`, because the base model had never seen that word used. After, almost every reply stops where an answer should stop. A base model is a text predictor; fine-tuning taught it the shape of a turn.
- **The older skill dropped.** Loss on held-out book text rose from 3.89 to 6.66. That is close to the 6.86 the Part 8 model reached when it had memorised the book. This is called **forgetting**: training on a tiny set pulled the knobs away from what pre-training had set.

## The failure to reproduce

Each line shows a question and the model's reply after 300 fine-tuning steps:

```
user: who said that ?               | reply: alice said that .
user: what did she say ?            | reply: she said that she said that rate .
user: who is the cat ?              | reply: the queen is not here .
user: what did the hatter say ?     | reply: the rabbit was at the door .
user: where is the door ?           | reply: the king is very old .
```

The first two were fine-tuning questions; the last three were held back. The model answers them in the right format and with the wrong content: it picked a memorised answer. It learned that a reply comes after `<assistant>` and ends with `<end>`. It never learned to look up facts. With 7 examples, that is expected. Real fine-tuning sets are much larger, but the lesson transfers: the format is learned quickly, and being correct is a separate matter, which Part 10 covered.

## What real chatbots add

The [InstructGPT paper](https://arxiv.org/abs/2203.02155) describes one real recipe. The authors start with "labeler-written prompts and prompts submitted through the OpenAI API" and collect "labeler demonstrations of the desired model behavior" to fine-tune GPT-3 "using supervised learning". That is the same step as the experiment above. They then collect "a dataset of rankings of model outputs" and fine-tune further "using reinforcement learning from human feedback". In their human evaluations, "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters". They also report "minimal performance regressions on public NLP datasets", which is the opposite of what the toy model did, and the paper still says InstructGPT "still makes simple mistakes". The paper's opening line gives the reason for all this work: "Making language models bigger does not inherently make them better at following a user's intent."

## Try it

1. Change `lr=1e-3` to `1e-4` in the fine-tuning optimiser. How many steps until replies end properly? How far does the book loss rise?
2. Add three more pairs to `pairs` using words from the book. Do the answers to new questions improve?
3. Print `reply(m, "who is the queen ?", g)` ten times. How often is the answer the same? Why?

## Limits

- Everything here is one 166,000-knob model, ten examples, and one book. The size of the forgetting would be different at scale.
- The InstructGPT paper reports what its authors measured on their own prompts and data. It is not a description of every chatbot, and companies do not all publish their recipes.
- Rankings and reinforcement learning (the second half of that recipe) are not implemented here.
