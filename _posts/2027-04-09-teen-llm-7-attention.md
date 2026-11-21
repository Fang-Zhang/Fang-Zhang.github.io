---
title: "How Language Models Work, Part 7: Attention, Which Earlier Words Matter"
date: 2027-04-09 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, attention, transformer, softmax, beginners, teens]
description: "Part 7 of a 12-part series for readers aged 14 to 17. Attention lets a model weigh every earlier word when it predicts the next one. This post works one example by hand, then runs it in Python, and shows with a simulation why the formula divides by a square root."
---

The network in Part 6 saw one word. Real text needs more: in "the cat sat because it ...", the word "it" points back to "cat", four words earlier. **Attention** is the step that lets the model look back at earlier words and decide how much each one matters right now. It was the key idea of the 2017 paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) from Google researchers, which proposed the Transformer, a network "based solely on attention mechanisms". Part 8 assembles a full Transformer; this post covers the one new piece.

## The idea in three steps

Each word, at each position, does three things:

1. **Ask.** The current word makes a **query**: a list of numbers that says what it is looking for.
2. **Match.** Every earlier word offers a **key**: numbers that say what it has. Compare the query with each key by a dot product (Part 4). A big dot product means a good match. Softmax (Part 6) turns the match scores into weights that add up to 1.
3. **Collect.** Every word also carries a **value**: the information it hands over. The output is the weighted sum of the values.

The paper describes it in the same terms. The output "is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key". For the version used in Transformers: "We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values." Here `dk` is the length of the key vectors, and the divide step is explained at the end.

## The program

Five words, each with a hand-chosen 2-number vector: how animal-like it is, and how much it is a small function word. In a real model these vectors, and the queries, are learned during training (Part 5's loop). Here they are set by hand so you can follow each number. To keep it short, keys and values are the same vectors, and the query is set by hand to "where is an animal?".

```python
import numpy as np

words = ["the", "cat", "sat", "because", "it"]
# hand-chosen 2-number vectors: (animal-ness, function-word-ness). A real model learns these.
X = np.array([[0.0, 1.0],     # the
              [2.0, 0.0],     # cat
              [0.5, 0.0],     # sat
              [0.0, 1.5],     # because
              [0.8, 0.8]])    # it

q = np.array([1.0, 0.0])      # the question "it" asks: "where is an animal?"
K = V = X                     # keys = what each word offers; values = what it hands over

scores = K @ q                                 # one dot product per word
scores = scores / np.sqrt(len(q))              # scale (explained below)
weights = np.exp(scores) / np.exp(scores).sum()   # softmax
output = weights @ V                           # weighted sum of the values

for w, s, a in zip(words, scores, weights):
    print(f"{w:8s} score {s:5.2f}  weight {a:.3f}")
print("output", output.round(3))
```

Output from a real run:

```
the      score  0.00  weight 0.108
cat      score  1.41  weight 0.442
sat      score  0.35  weight 0.153
because  score  0.00  weight 0.108
it       score  0.57  weight 0.189
output [1.113 0.42 ]
```

Walk through "cat": its key is `(2.0, 0.0)`, the query is `(1.0, 0.0)`, so the dot product is `2.0*1.0 + 0.0*0.0 = 2.0`. Divide by the square root of 2 (the vector length) and you get 1.41. The word "the" has key `(0.0, 1.0)`: dot product 0. Softmax turns the five scores into the five weights, with "cat" largest at 0.44. The output `(1.11, 0.42)` is mostly the value of "cat", mixed with smaller amounts of the others. That output vector is what the rest of the network now uses to predict the next word.

![Left: bar chart of attention weights for five words, with cat highest at 0.44. Right: grouped bar chart showing the average largest softmax weight with and without dividing by the square root of the vector length, for lengths 4, 64 and 512]({{ site.baseurl }}/assets/pic/teen-llm-attention.png){: width="900" }
*Left: the weights from the program above. Right: why the formula divides by a square root (see the next section).*

## Where it fails: without the scaling step

Why divide by `√dk`? When the vectors are long, the dot products become large, and softmax then gives almost all the weight to one word. The paper gives this as its suspected reason: "for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients". Small gradients mean the training loop from Part 5 has almost no slope to follow. You can see the effect with random vectors:

```python
import numpy as np
rng = np.random.default_rng(0)
for d in (4, 64, 512):
    big, small = [], []
    for _ in range(2000):
        q = rng.normal(size=d); K = rng.normal(size=(5, d))
        s = K @ q
        for sc, store in ((s, big), (s / np.sqrt(d), small)):
            e = np.exp(sc - sc.max()); store.append((e / e.sum()).max())
    print(d, "largest weight, no scaling:", round(np.mean(big), 2), " with scaling:", round(np.mean(small), 2))
```

```
4 largest weight, no scaling: 0.61  with scaling: 0.44
64 largest weight, no scaling: 0.9  with scaling: 0.46
512 largest weight, no scaling: 0.96  with scaling: 0.46
```

With 512 numbers per vector and no scaling, the winning word takes 96% of the weight on average, no matter how the query compares to the keys. With scaling it stays near 46% at every length.

## Two details to know

- **Only look back.** A model that writes text must not look at words that come after the current one, since at writing time they do not exist. The paper handles this with a mask that stops positions "from attending to subsequent positions".
- **Many looks at once.** The Transformer runs several attention steps in parallel, each with its own learned queries, keys and values, so that one can track "which noun does this pronoun point to" while another tracks something else. The paper calls this multi-head attention.

## Try it

1. Change the query `q` to `[0.0, 1.0]` (looking for function words). Which word gets the largest weight now?
2. Set `q = np.array([5.0, 0.0])`. What happens to the weights? Compare with the scaling failure above.
3. Add a sixth word to `words` and `X`, for example "mouse" with vector `[1.8, 0.2]`. Who gets the weight?
4. Remove the line `scores = scores / np.sqrt(len(q))` and compare the weights.

## Limits

- The vectors here were chosen by hand to make the example readable. Real models learn queries, keys and values from data, as separate weighted sums of the word's vector; "cat" does not get its weight because someone labelled it an animal.
- A model computes weights; it does not decide to pay attention. The word "attention" names the weighted sum, nothing more.
- The figure on the right uses random vectors and averages 2,000 trials per length. The exact numbers depend on the random seed; the pattern does not.
- Attention weights show where information is taken from, which is not always a full explanation of why a model produced a word.
