---
title: "How Language Models Work, Part 8: Putting It Together"
date: 2027-04-10 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, transformer, overfitting, PyTorch, beginners, teens]
description: "Part 8 of a 12-part series for readers aged 14 to 17. This post assembles tokens, embeddings, attention and a small network into a Transformer, trains a real one on a single book, and shows why on that much text it learns to memorise and not to generalise."
---

Parts 3 to 7 built the pieces. This post stacks them into a **Transformer**, trains one, and measures it honestly, including the failure.

## The stack

The 2017 paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) describes layers that each have two sub-layers: multi-head self-attention, and "a simple, position-wise fully connected feed-forward network" (what this series calls a small network). A residual connection goes around each: the layer's input is added to its output, so each layer only has to learn a change. Its base model stacks 6 of these layers. The figure shows the same recipe, drawn with 2.

![Left: a vertical stack of boxes. Words come in as numbers, are combined with position numbers, pass twice through an attention box followed by an add step and a small network, then produce scores and softmax chances. Right: line chart of loss against training steps for a one-word network and a 2-layer Transformer, on the book it trained on and on new text]({{ site.baseurl }}/assets/pic/teen-llm-transformer.png){: width="900" }
*Left: the recipe. Right: real losses from the program below.*

In order, for each word of text:

1. **Word numbers + position numbers.** Part 4's table gives each word a list of numbers. Attention alone ignores order ("dog bites man" and "man bites dog" would look the same), so a second table gives each position (1st word, 2nd word, ...) its own list, and the two are added. The paper says it must "inject some information about the relative or absolute position of the tokens"; it uses a fixed formula for this, and the code below uses a learned table instead.
2. **Attention** (Part 7): each word collects information from earlier words.
3. **Small network** (Part 6): each word's numbers are processed on their own.
4. **Repeat** steps 2 and 3 for each layer.
5. **Scores and softmax** (Part 6): numbers become chances for the next word.
6. **Training** (Part 5): compare with the real next word, compute the loss, move every knob a little.

Nothing new is added here. A Transformer is Parts 3 to 7 in a particular order.

## The program

It uses PyTorch, a free library that computes the slopes automatically (so the backpropagation lines from Part 6 disappear). It needs `pip install torch` and runs in about a minute on a laptop CPU. It uses the same book as Parts 2 to 6, but this time the **last 10% of the book is kept hidden**: the model never trains on it, and we measure the loss on it as "new text". A one-word network, as in Part 6, is the comparison.

```python
import re, math, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter

torch.manual_seed(0)
book = open("alice.txt", encoding="utf-8").read().replace("’", "'")
start = book.index("Down the Rabbit-Hole", book.index("CHAPTER I."))
words = re.findall(r"[a-z']+|[.,!?;]", book[start:book.index("*** END OF")].lower())
vocab = [w for w, _ in Counter(words).most_common(500)] + ["<other>"]
idx = {w: i for i, w in enumerate(vocab)}
data = torch.tensor([idx.get(w, 500) for w in words])
cut = int(len(data) * 0.9)
train_data, test_data = data[:cut], data[cut:]        # last 10% of the book is never trained on
V, T, D = 501, 32, 64                                 # vocabulary, context length, numbers per word

def batch(src, n=64):
    i = torch.randint(0, len(src) - T - 1, (n,))
    return (torch.stack([src[j:j + T] for j in i]), torch.stack([src[j + 1:j + T + 1] for j in i]))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))
    def forward(self, x):
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)       # no looking ahead
        h = self.ln1(x)
        x = x + self.att(h, h, h, attn_mask=mask, need_weights=False)[0]   # attention (Part 7)
        return x + self.mlp(self.ln2(x))                               # small network (Part 6)

class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)    # word numbers + position numbers
        self.blocks = nn.Sequential(*[Block() for _ in range(layers)])
        self.out = nn.Linear(D, V)
    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        return self.out(self.blocks(h))

def evaluate(model, src):
    model.eval(); tot = 0
    with torch.no_grad():
        for _ in range(50):
            x, y = batch(src); tot += F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
    model.train(); return tot / 50

class Bigram(nn.Module):                              # baseline: sees one word, like Part 6
    def __init__(self):
        super().__init__()
        self.tok, self.out = nn.Embedding(V, D), nn.Linear(D, V)
    def forward(self, x):
        return self.out(self.tok(x))

def train(model, name, steps=1500):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for t in range(steps + 1):
        if t % 100 == 0:
            print(name, t, "train", round(evaluate(model, train_data), 3), "test", round(evaluate(model, test_data), 3), flush=True)
        x, y = batch(train_data)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

for name, model in (("bigram", Bigram()), ("transformer", Model(2))):
    print(name, "knobs:", sum(p.numel() for p in model.parameters()))
    train(model, name)
```

Real output (every 100 steps in the full output; selected rows here; loss on the book it trained on, then on the hidden last 10%):

```
bigram knobs: 64629
bigram       100  train 4.118  test 4.305
bigram       300  train 3.455  test 3.877
bigram       1500 train 3.135  test 4.113
transformer knobs: 166645
transformer  100  train 3.751  test 3.977
transformer  200  train 3.372  test 3.872
transformer  500  train 2.684  test 4.170
transformer  1000 train 1.768  test 5.426
transformer  1500 train 1.225  test 6.863
```

## What happened

- The Transformer fits the training text much better: its training loss falls to 1.22, while the one-word network stays at 3.14.
- On the hidden text, the best loss of each model is almost the same: 3.87 for the Transformer (at step 200) and 3.88 for the one-word network (at step 300). The difference, 0.005, is smaller than the step-to-step wobble in the table.
- After that, the Transformer's loss on new text rises steadily to 6.86, close to where it started (6.58). It is learning the exact sequences in this book, not patterns that hold in other text. This is called **overfitting**, and the figure shows it as the gap between a solid and a dashed red line.

The reason is data. The book has about 31,000 words, and the Transformer has 166,645 knobs: more knobs than words, so it can store the text. The one-word network has few ways to memorise and holds up better. A Transformer's advantage is that it can use long context, but learning to use it needs far more text than one book.

## Where it fails: no new text, no win

The failure here is the main result. If you stopped at the training loss, you would conclude that the Transformer is far better. Only the hidden text shows that it is not (yet). Always measure on text the model has not seen. Real language models are trained on far more text than any one book, and that is the topic of Part 9.

## Try it

1. Run the program and find the step where the Transformer's test loss is lowest. Stopping there is called early stopping.
2. Change `Model(2)` to `Model(1)`, then `Model(4)`. Does the pattern change?
3. Change `D = 64` to 16. A smaller model has fewer ways to memorise. What happens to the gap between train and test?
4. Change the 90% cut (`0.9`) to 0.5, so there is less to train on. What happens to the test loss?

## Limits

- This Transformer is tiny and trained on one book. Real models have billions of knobs and trained on far more text; this post shows the mechanism, not what such a model does.
- The figure leaves out details that real Transformers use: normalisation steps (`LayerNorm` in the code) and several parallel attention heads (4 here).
- The test numbers come from random samples of the hidden text, so they wobble by a few hundredths. Another seed changes them slightly.
- Sampling text from this tiny model gives nonsense in both versions. Neither model is a chatbot.
