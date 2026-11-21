---
title: "How Language Models Work, Part 6: A Small Network"
date: 2027-04-08 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, neural network, softmax, backpropagation, beginners, teens]
description: "Part 6 of a 12-part series for readers aged 14 to 17. A two-layer neural network with thousands of knobs learns next-word chances on a whole book by the loop from Part 5. This post explains weighted sums and softmax, shows real loss curves, and shows what a network that is too small cannot learn."
---

Part 5 trained one knob. A neural network is the same loop with many knobs, arranged in layers. This post builds the smallest useful one: it reads one word and outputs a chance for every possible next word. By the end you will have seen every ingredient of a language model except one (attention, Part 7).

## The maths tool: weighted sums and softmax

A **weighted sum** multiplies each input by a knob (a weight) and adds the results. It is the dot product from Part 4, with the weights doing the matching:

```
inputs  (0.5, 2.0)      weights (3, 1)
weighted sum = 0.5*3 + 2.0*1 = 3.5
```

A layer does this many times, with different weights each time, and each result is one output number. A network is layers stacked one after another.

The final layer produces one score per possible next word, and scores can be any size, even negative. We need chances that are positive and add up to 1. The **softmax** function does this: raise `e` (about 2.718) to the power of each score, then divide each result by the total.

```
scores        2.0    1.0    0.1
e^score       7.39   2.72   1.11     total 11.21
chances       0.659  0.242  0.099    (add up to 1)
```

Higher score, higher chance, and the chances always sum to 1.

## The shape of the network

- **Input.** One word, from the 500 most common in the book (everything else is lumped together as `<other>`, 501 symbols).
- **Layer 1.** Look up that word's row in a table: `d` numbers per word (here 16). This is a learned version of the word vectors from Part 4. Nobody counts neighbours; the numbers start random and training sets them.
- **Layer 2.** 501 weighted sums of those 16 numbers: one score for each possible next word.
- **Softmax.** Scores become chances.
- **Loss.** The surprise from Part 5, `-ln(p)`, averaged.

That is about 16 x 501 + 501 x 16 = 16,032 knobs.

## Training: slopes for many knobs

Part 5 had one slope. Here each knob gets its own slope, and the loop is the same: `knob = knob - step * slope`. The slopes are found by working backwards from the loss, layer by layer, which is called **backpropagation**. In the code it is a handful of lines. The first one is worth reading: `dz = p` then subtract 1 at the correct word. The slope of the loss at the scores is "the chance the model gave, minus 1 for the word that actually came". It is the same `p - k/n` from Part 5, now for all 501 words at once.

## The program

This is the longest program in the series. The comments carry the meaning; you can read them and skip the rest. It needs NumPy, and takes about half a minute on a laptop.

```python
import re, numpy as np
from collections import Counter

book = open("alice.txt", encoding="utf-8").read().replace("’", "'")
start = book.index("Down the Rabbit-Hole", book.index("CHAPTER I."))
words = re.findall(r"[a-z']+|[.,!?;]", book[start:book.index("*** END OF")].lower())

vocab = [w for w, _ in Counter(words).most_common(500)] + ["<other>"]   # 501 symbols
idx = {w: i for i, w in enumerate(vocab)}
ids = np.array([idx.get(w, 500) for w in words])
x, y = ids[:-1], ids[1:]                      # input word -> next word

def full_loss(E, W):                          # loss over the whole book
    z = E[x] @ W; z -= z.max(1, keepdims=True)
    return -(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(len(x)), y].mean()

def train(d=16, steps=8000, lr=2.0, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.1, (501, d))          # layer 1: one row of d numbers per word
    W = rng.normal(0, 0.1, (d, 501))          # layer 2: turns d numbers into 501 scores
    for t in range(steps):
        b = rng.integers(0, len(x), 512)      # a random batch of 512 word pairs
        h = E[x[b]]                           # look up the word's numbers
        z = h @ W                             # weighted sums: one score per possible next word
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)   # softmax
        loss = -np.log(p[np.arange(512), y[b]]).mean()
        dz = p; dz[np.arange(512), y[b]] -= 1; dz /= 512      # slope of loss at the scores
        dW = h.T @ dz                         # slope for layer 2
        dh = dz @ W.T                         # pass the slope back to layer 1
        W -= lr * dW
        np.add.at(E, x[b], -lr * dh)          # slope for layer 1
        if t % 1000 == 0 or t == steps - 1: print(d, t, round(full_loss(E, W), 3))
    return E, W

E, W = train(d=16)

def next_word(word, rng):
    z = E[idx.get(word, 500)] @ W
    p = np.exp(z - z.max()); p /= p.sum()
    return vocab[rng.choice(501, p=p)]

rng = np.random.default_rng(3)
text = ["alice"]
for _ in range(20):
    text.append(next_word(text[-1], rng))
print(" ".join(text))
```

Output from a real run (columns: knob size `d`, step, loss over the whole book):

```
16 0 6.217
16 1000 3.845
16 2000 3.648
16 3000 3.558
16 4000 3.503
16 5000 3.463
16 6000 3.431
16 7000 3.416
16 7999 3.394
alice , it saw on , you see the <other> , i can't take them off <other> to him so it
```

At step 0 the loss is 6.217, which is `ln(501)`: the network gives every one of the 501 symbols the same chance. After 8,000 steps it is 3.39. The text it writes is, again, nonsense: it only looks at one word. But each step of the nonsense is plausible: "you see", "take them off", "to him".

![Line chart of loss against training steps for networks with 2, 16 and 64 numbers per word. All start at 6.22. The 2-number network flattens at about 4.28, the 16-number network reaches about 3.4, the 64-number network about 3.2. A dashed line marks the counting-table loss of 3.08]({{ site.baseurl }}/assets/pic/teen-llm-small-network.png){: width="900" }
*Loss on the whole book for three network sizes, from real runs. The dashed line is what the counting table from Part 2 scores on the same book.*

## Where it fails: too small

Run the same program with `d=2`: only two numbers per word. The loss falls to 4.28 and stays there, however long you train (the red line). Two numbers cannot hold enough about 501 different words to separate them. With `d=64` it reaches 3.18 after 8,000 steps and is still falling slowly.

## Why can't it beat the dashed line?

The dashed line, 3.08, is the loss of the counting table from Part 2: for each word, the exact fractions of what followed it. For a model that sees one word of context, that table is the best possible on this book. The network's job is to find those same numbers by gradient descent, and it gets close (3.18 with 64 numbers) but never below. So why build a network at all, if a table is better here? Because the table needs one row for every context, and the number of contexts explodes as you look further back (Part 2: 14,366 rows for two words). A network's number of knobs grows slowly, and it is the only known way to scale up. Part 7 gives it a way to look at many earlier words.

## Try it

1. Run `train(d=2)`, `train(d=16)` and `train(d=64)` and compare the final losses with the figure.
2. Change `lr=2.0` to `0.1`, then `20.0`. What happens to the loss in 8,000 steps? (Part 5 predicted both outcomes.)
3. Print the chances for the word after "the": use `E[idx["the"]] @ W`, softmax them, and show the top five words. Compare with the counted top words from Part 2.
4. Change the sampling start from `"alice"` to `"the"`.

## Limits

- The loss is measured on the same text the network was trained on. This says how well it fitted the book, not how well it would do on a new one.
- The numbers (3.39, 4.28, 3.18, 3.08) are from one run with one random seed and one book. Another seed gives slightly different values.
- This network reads one word. It cannot know what came before it, which is why its text wanders.
- Real networks have more layers and usually add small extra steps between layers that this one omits.
