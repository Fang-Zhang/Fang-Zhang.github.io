---
title: "The LLM Engineer Path, Part 11: Interview Questions on the Fundamentals"
date: 2027-04-25 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, interview, transformer, attention, tokenization, sampling, python]
description: "Part 11 of a 14-part series for developers moving into LLM engineering. Six classic fundamentals questions, each answered in a sentence, backed by the source paper and a short experiment you can run: why attention divides by the square root of d, why position must be injected, what the causal mask costs, how BPE splits words, what temperature and top-p do, and how to count parameters."
---

These are six questions about Transformer fundamentals that come up again and again in LLM engineering interviews. They are classic topics, not questions from any particular company. Each answer is one sentence an interviewer can accept, then the evidence: the paper's own words and a small experiment. An answer you have run yourself is easier to defend under follow-up questions than one you have memorised.

## Q1. Why does attention divide the scores by the square root of d?

**Answer.** The dot product of two d-dimensional vectors has a spread that grows with the square root of d; without the division the softmax saturates, one key takes nearly all the weight, and gradients vanish.

The original paper ([Vaswani et al.](https://arxiv.org/abs/1706.03762)) says it as a suspicion: "We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients." The check: draw queries and keys with unit-variance entries, 16 keys per query, 2,000 trials.

```python
q = rng.standard_normal((2000, 1, d)); k = rng.standard_normal((2000, 16, d))
s = (q * k).sum(-1)                                  # raw scores
p = softmax(s / np.sqrt(d))                          # scaled; compare with softmax(s)
```

```
d       std of raw scores   largest weight, raw   largest weight, scaled
4              2.00                0.43                  0.24
16             4.05                0.68                  0.25
64             8.04                0.85                  0.25
256           16.03                0.92                  0.25
1024          31.95                0.96                  0.25
```

The raw spread equals the square root of d (2, 4, 8, 16, 32), and the scaled spread stays at 1. At d = 1024 the unscaled softmax puts 96% of the weight on one key out of 16. The sum of p(1 - p), the diagonal of the softmax Jacobian, falls from 0.70 to 0.06 without scaling and stays at 0.87 with it, which is the "extremely small gradients" in numbers.

## Q2. Why does a Transformer need positional encodings?

**Answer.** Attention treats its input as a set: shuffling the tokens just shuffles the outputs, so order has to be added to the inputs.

The paper: "in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens". The check with a single attention layer in numpy (6 tokens, d = 8):

```
max |att(X[perm]) - att(X)[perm]| = 1.8e-15                     (no positions: pure permutation)
with sinusoidal positions added to the inputs: the same test gives a difference of 8.22
```

Without positions, permuting the input permutes the output and nothing else changes, to floating-point rounding. With positions added the output depends on where each token is. The paper uses sine and cosine waves, hypothesising this "would allow the model to easily learn to attend by relative positions"; many current models use other schemes, but the reason is the same.

## Q3. What does the causal mask do, and why is attention quadratic?

**Answer.** The mask stops position i from attending to later positions, so a decoder can be trained on whole sequences at once and still predict only from the past; the quadratic cost comes from the n x n score matrix.

The paper's wording: the decoder's self-attention is modified "to prevent positions from attending to subsequent positions", so "the predictions for position i can depend only on the known outputs at positions less than i". Cost: the score matrix has n x n entries per head per layer. Stored at 2 bytes each, if fully materialised:

```
n =   4,096:   0.03 GB per head per layer
n =  32,768:   2.15 GB
n = 131,072:  34.36 GB
```

Multiplying n by 8 multiplies the matrix by 64. [FlashAttention](https://arxiv.org/abs/2205.14135) starts from the same observation, that "the time and memory complexity of self-attention are quadratic in sequence length", and computes exact attention in tiles, which cuts the reads and writes between the GPU's main memory and its on-chip memory; it is an IO-aware rewrite, not an approximation. The cost is still quadratic in compute.

## Q4. How does BPE tokenisation work, and why do models miscount letters?

**Answer.** Byte-pair encoding ([Sennrich et al.](https://arxiv.org/abs/1508.07909)) starts from characters and repeatedly merges the most frequent adjacent pair into one new token; a model reads tokens, not letters, so letter-level questions ask about units it never sees.

A 20-line version trained on half the Python standard library documentation (the other half held out):

```python
for _ in range(1500):
    pairs = Counter()
    for word, count in vocab.items():                 # vocab: tuple of symbols -> frequency
        for a, b in zip(word, word[1:]): pairs[a, b] += count
    (a, b), _ = pairs.most_common(1)[0]               # the most frequent adjacent pair
    merges.append((a, b))                             # ...becomes a new symbol, and every word is rewritten
```

```
function        -> ['function']
tokenization    -> ['token', 'iz', 'ation']
strawberry      -> ['str', 'a', 'w', 'ber', 'r', 'y']
12345678        -> ['1', '2', '3', '4', '5', '6', '7', '8']       (digits are kept single here by choice)

merges   tokens per word on unseen documents
     0   4.79        (characters)
    50   3.08
   200   2.14
   500   1.63
 1,500   1.22
```

"How many r in strawberry?" asks about an object the model gets as pieces; this tokenizer happens to split the word into six. Real tokenizers have far larger vocabularies and split differently, so the point carries over, the exact splits do not. More merges mean fewer tokens per word, which means shorter sequences and lower cost, but a bigger embedding table, and rare words still break into pieces: 328 of the 944 word types in the held-out half had never been seen in training.

## Q5. What do temperature, top-k and top-p do?

**Answer.** Temperature T divides the logits before the softmax (below 1 sharpens, above 1 flattens); top-k keeps the k most likely tokens; top-p keeps the smallest set whose probability adds up to p. Both truncations remove the unreliable tail before sampling.

[Holtzman et al.](https://arxiv.org/abs/1904.09751) found that "using likelihood as a decoding objective leads to text that is bland and strangely repetitive", and that "decoding strategies alone can dramatically effect the quality of machine text, even when generated from exactly the same neural language model". Their nucleus sampling samples "from the dynamic nucleus of the probability distribution", truncating "the less reliable tail". "Dynamic" is the point: top-k keeps k tokens whether the model is sure or not, top-p keeps few tokens when one dominates and many when the distribution is flat. For eight logits (4.0, 3.2, 2.5, 1.0, 0.5, 0.0, -1.0, -2.0):

```
T=0.5: top probability 0.80   entropy 0.90 bits   nucleus (top-p 0.9) size 2
T=1.0: top probability 0.56   entropy 1.70 bits   nucleus (top-p 0.9) size 3
T=2.0: top probability 0.36   entropy 2.46 bits   nucleus (top-p 0.9) size 5
```

Greedy decoding is the T-approaches-0 limit. Use it for extraction and structured output (Part 3), where one right answer exists, and sampling for tasks where variety is wanted.

## Q6. How many parameters does a decoder-only Transformer have?

**Answer.** About 12 L d^2 for the L layers (4 d^2 for the attention projections, 8 d^2 for a feed-forward block with hidden size 4d), plus (V + context length) x d for the token and position embeddings; biases and norms are small.

```python
def formula(L, d, V, ctx): return 12 * L * d * d + (V + ctx) * d
```

Compared with a real PyTorch model built with `nn.TransformerEncoder` layers (the architecture is the same; a causal mask changes the behaviour, not the parameter count):

```
L= 2  d= 128  V= 1,000:  formula   0.55M   PyTorch   0.56M   difference 0.60%
L=12  d= 768  V=50,257:  formula 124.32M   PyTorch 124.44M   difference 0.10%
```

![Left: mean attention weight on the largest of 16 keys against head dimension, rising from 0.43 to 0.96 without scaling and flat at 0.25 with scaling. Middle: probabilities of eight tokens at temperatures 0.5, 1 and 2. Right: parameter breakdown for a 12-layer, 768-wide model: 39.4M embeddings, 28.3M attention, 56.6M feed-forward]({{ site.baseurl }}/assets/pic/llmeng-interview1.png){: width="900" }

The formula assumes the output layer shares the embedding table; a model with a separate output matrix adds V x d. Memory for the weights is then the parameter count times bytes per parameter (2 in 16-bit), the starting point for the estimates in Parts 2, 6 and 9.

## Try it

1. Q1 with 64 keys instead of 16. Does the unscaled softmax saturate faster or slower, and why?
2. Q2: replace the sinusoids with a learned random matrix. Is the output still position-dependent?
3. Q4: train the tokenizer on code instead of prose and compare tokens per word on the same held-out text.
4. Q5: sample 10,000 tokens at each temperature and check the empirical frequencies against the printed probabilities.

## Limits

- The numerical experiments use random vectors and toy sizes; trained models have structured scores, so the saturation shown is the starting condition, not what a trained model looks like.
- The tokenizer is trained on about 5,100 words of documentation, so its splits illustrate the method and do not reproduce any production tokenizer.
- Only single-head attention and a decoder-only parameter count were covered; multi-head attention, grouped-query attention and mixture-of-experts change the count and are not in this post.
