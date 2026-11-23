---
title: "From Zero to ML Researcher, Part 6: Language as Prediction"
date: 2027-05-16 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, language model, n-gram, bits per character, ML researcher]
description: "Part 6 of a 12-part series for complete beginners. A language model predicts the next character. A counting model with no neural network sets the baseline: 2.19 bits per character on held-out text, against 5.58 for guessing, and the way it fails explains why the next parts need attention."
---

## What you will do, and what you get

You will build a language model that uses nothing but counting, and measure it with the number researchers use for text: *bits per character* (how many yes/no questions, on average, the model needs to pin down the next character). Trained on 439,000 characters of English technical text that ships with Python, a model that looks at the previous 3 characters needs 2.19 bits per character on text it never saw, against 5.58 for guessing uniformly among the 48 characters. You will see the same model fail in an instructive way: lengthen the context to 5 characters and the score gets worse (2.51), because the table of contexts is almost empty. That failure is the problem that the next two parts solve.

![Left: bits per character against context length 0 to 5. Validation falls from 4.66 to a minimum of 1.85 at length 4; test falls from 4.65 to 2.19 at length 3 and rises to 2.51 at length 5; a dotted line marks guessing at 5.585. Middle: bars on a log scale comparing the number of contexts seen in training (1, 48, 1,405, 9,404, 28,679, 58,476) with the number possible, 48 to the power of the context length. Right: three text samples generated from the models with context 1, 3 and 5, becoming more word-like and then collapsing into noise at 5]({{ site.baseurl }}/assets/pic/mlr-6-language.png){: width="900" }

## How: the steps

1. **Get a text.** Python ships an English reference text inside the standard library (`pydoc_data.topics`); it needs no download. Lowercase it and keep 48 distinct characters.
2. **Split by position, not at random:** the first 80% for training, the next 10% for validation, the last 10% for testing. Text is a sequence, and a random split would put neighbouring characters on both sides.
3. **Count.** For a context of length n, count how often each next character follows each context in the training text.
4. **Turn counts into probabilities,** adding a small constant `k` to every count so that nothing has probability zero (add-k smoothing).
5. **Measure** the average of `-log2(probability of the true next character)` on held-out text: bits per character.
6. **Choose** `k` on the validation block, then report the test block.
7. **Sample** text from the model to see what it has learned.

## Why it works: the principle

**A language model is a next-character predictor.** Given the text so far, it outputs a probability for each possible next character. Everything that follows in this series, including the transformer of Part 8, is a better predictor of this kind. Generating text is repeated prediction: draw a character from the probabilities, append it, repeat.

**Bits per character is the loss, in different units.** The cross-entropy of Part 5 is `-log p` of the true class. Using `log2` instead of the natural logarithm gives bits instead of nats (divide nats by `ln 2 = 0.693`). The number has a plain meaning: `2` to the power of the score is the number of equally likely choices the model is effectively guessing among (its *perplexity*, described in [Jurafsky and Martin, chapter 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf)). Guessing uniformly among 48 characters costs `log2(48) = 5.585` bits. This kind of counting model is an *n-gram model*; [Shannon (1948)](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) already used such approximations to English to explain information and redundancy.

**Reading the results.**
- Context 0 (just letter frequencies) costs 4.65 bits. Context 1 costs 3.39, context 2 costs 2.68, context 3 costs 2.19: each extra character of context removes information that was predictable.
- The 2.19 is about 4.55 equally likely choices per character (2 to the power 2.187).
- **Then it breaks.** Context 4 costs 2.21 and context 5 costs 2.51 on the test block. The middle panel shows why: with 48 characters there are `48^5 = 254,803,968` possible contexts of length 5, and the training text contains only 58,476 distinct ones (0.02%). Most contexts met at test time were never seen, and the model can only fall back to the constant `k`, close to uniform guessing. The samples in the figure show it: with context 3 the output looks like words, and with context 5 it turns into noise whenever it steps into an unseen context.
- **Validation and test disagree on the best length.** Validation picks context 4 (1.853 bits), the test block says context 3 (2.187 against 2.212). The test block is harder at every length, because the text is ordered by topic and the three blocks cover different topics. Choosing by validation cost 0.025 bits here: small, and a reminder that the validation block is also one sample.

**Why this is the baseline for what follows.** To predict well the model needs a long context, and counting cannot give one, because the table grows exponentially with the context length and never fills. The remedy is a model that does not store a table, but computes a probability from the context by a function with shared parameters. That is what a neural network does, and what attention (Part 7) does better than a fixed window.

**Pitfalls and limits.**
- The text depends on the Python version: the corpus printed in the output has 548,462 characters and a hash starting `4581d6c163b4` on Python 3.14.7. Other versions give slightly different numbers.
- A model can score worse than guessing on a surprising string. The worked example below shows 11.23 bits per character for a nonsense string, because the model is confident and wrong.
- A single split in order is a single sample of the text. The numbers are for this corpus, a mix of prose, code snippets and grammar.
- Counting contexts gives sharp but brittle predictions; add-k smoothing is the crudest fix. Better schemes exist (see the smoothing sections of the same chapter).

## Example, worked through

How much does the bigram model (context 1, `k = 0.1`) pay for the three characters after the `t` in `the `? Each probability comes from counting in the training text.

```
P('h' after 't') = (8070 + 0.1) / (32944 + 0.1 * 48) = 0.2449  ->  2.03 bits
P('e' after 'h') = (6029 + 0.1) / (11214 + 0.1 * 48) = 0.5374  ->  0.90 bits
P(' ' after 'e') = (10261 + 0.1) / (42299 + 0.1 * 48) = 0.2426  ->  2.04 bits
'the ' costs 4.97 bits for 3 predicted characters = 1.66 bits per character (uniform guessing: 5.58)
'tqx ' costs 33.70 bits = 11.23 bits per character
```

- After `t`, the character `h` follows 8,070 times in 32,944 occurrences: probability 0.2449, so 2.03 bits.
- After `h`, the character `e` follows 6,029 times in 11,214: 0.5374, so 0.90 bits.
- After `e`, a space follows 10,261 times in 42,299: 0.2426, so 2.04 bits.
- Total 4.97 bits for 3 characters, 1.66 bits per character, far under the 5.58 of guessing. The string `tqx ` pays 11.23 bits per character, because `q` almost never follows `t`.

## Implementation

Save as `part6.py` and run it (a few minutes on a CPU; it counts 6 models). It needs numpy and the Python standard library.

```python
import re, math, hashlib, collections
import numpy as np
import pydoc_data.topics as topics            # English reference text that ships with every Python installation

text = "\n".join(topics.topics[k] for k in sorted(topics.topics)).lower()
text = re.sub(r"[^a-z0-9 .,\n()'\"=:_-]", " ", text); text = re.sub(r" +", " ", text)
chars = sorted(set(text)); V = len(chars)
n = len(text); train, val, test = text[: int(.8 * n)], text[int(.8 * n): int(.9 * n)], text[int(.9 * n):]   # contiguous blocks, in order
print(f"corpus: {n:,} characters, {V} distinct, sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]}")
print(f"train {len(train):,} | validation {len(val):,} | test {len(test):,}")
print(f"guessing uniformly: {math.log2(V):.3f} bits per character\n")

def fit(n_ctx):
    counts = collections.defaultdict(collections.Counter)
    for i in range(n_ctx, len(train)): counts[train[i - n_ctx:i]][train[i]] += 1
    return counts

def bits_per_char(counts, n_ctx, k, data):
    total = 0.0; m = 0
    for i in range(n_ctx, len(data)):
        c = counts.get(data[i - n_ctx:i]); tot = sum(c.values()) if c else 0
        p = ((c[data[i]] if c else 0) + k) / (tot + k * V)          # add-k smoothing: pretend every character was seen k extra times
        total -= math.log2(p); m += 1
    return total / m

print("context length   best k   validation bits/char   test bits/char   contexts stored")
rows = []
for n_ctx in range(0, 6):
    counts = fit(n_ctx)
    best = min(((bits_per_char(counts, n_ctx, k, val), k) for k in (0.001, 0.01, 0.1, 1.0)))
    t = bits_per_char(counts, n_ctx, best[1], test); rows.append((n_ctx, best[1], best[0], t, len(counts)))
    print(f"{n_ctx:14d}   {best[1]:6}   {best[0]:20.3f}   {t:14.3f}   {len(counts):15,}")

def sample(n_ctx, length, k, seed):
    counts = fit(n_ctx); r = np.random.default_rng(seed); out = text[:n_ctx]
    for _ in range(length):
        c = counts.get(out[len(out) - n_ctx:] if n_ctx else "", collections.Counter())
        w = np.array([c[ch] + k for ch in chars], float); out += chars[r.choice(V, p=w / w.sum())]
    return out[n_ctx:]
print()
for n_ctx in (1, 3, 5):
    print(f"sample, context {n_ctx}: " + repr(sample(n_ctx, 110, 0.01, 0)))
np.save("rows6.npy", np.array(rows))
```

Output:

```
corpus: 548,462 characters, 48 distinct, sha256 4581d6c163b4
train 438,769 | validation 54,846 | test 54,847
guessing uniformly: 5.585 bits per character

context length   best k   validation bits/char   test bits/char   contexts stored
             0    0.001                  4.661            4.654                 1
             1      0.1                  3.200            3.393                48
             2      0.1                  2.401            2.682             1,405
             3      0.1                  1.886            2.187             9,404
             4      0.1                  1.853            2.212            28,679
             5     0.01                  2.021            2.506            58,476

sample, context 1: 'he "gthelus\nn r(se. " oponywiore pleettenard tar tofr cans " ablepate nthondeythefis_foucout tyt wore stands r'
sample, context 3: ' cally retuplement if the writs are is exception stated in add__(name numentext possions oriatitestribut not\ne'
sample, context 5: 'if" statement\nexpression, wnkjl9,md5buw8f6g79u1h(rp2u"7-_q2":0(f5k0x8)iw_xb=hzxaobdp:nmw)mwy\ntzx-yurb2qw3d_w n'
```

The worked example, saved as `example6.py`:

```python
import re, math, collections
import pydoc_data.topics as topics
text = "\n".join(topics.topics[k] for k in sorted(topics.topics)).lower()
text = re.sub(r"[^a-z0-9 .,\n()'\"=:_-]", " ", text); text = re.sub(r" +", " ", text)
V = len(set(text)); train = text[: int(.8 * len(text))]
pairs = collections.defaultdict(collections.Counter)
for a, b in zip(train, train[1:]): pairs[a][b] += 1
def p(a, b, k=0.1): return (pairs[a][b] + k) / (sum(pairs[a].values()) + k * V)
word = "the "; total = 0
for a, b in zip(word, word[1:]):
    q = p(a, b); total -= math.log2(q); print(f"P({b!r} after {a!r}) = ({pairs[a][b]} + 0.1) / ({sum(pairs[a].values())} + 0.1 * {V}) = {q:.4f}  ->  {-math.log2(q):.2f} bits")
print(f"'{word}' costs {total:.2f} bits for 3 predicted characters = {total/3:.2f} bits per character (uniform guessing: {math.log2(V):.2f})")
bad = "tqx "; total = 0
for a, b in zip(bad, bad[1:]): total -= math.log2(p(a, b))
print(f"'{bad}' costs {total:.2f} bits = {total/3:.2f} bits per character")
```

## Try it

1. Change the smoothing constants to `(0.5, 2.0)` and see how the best `k` and the length-5 score move.
2. Make the split random instead of in order (shuffle blocks of 1,000 characters). Does the gap between validation and test shrink, and why?
3. Replace `k` smoothing by a fallback: if the 5-character context is unseen, use the 3-character one. What happens to the length-5 score?
