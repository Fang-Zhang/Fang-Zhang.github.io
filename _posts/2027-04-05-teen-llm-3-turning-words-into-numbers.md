---
title: "How Language Models Work, Part 3: Turning Words into Numbers"
date: 2027-04-05 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, tokenization, byte pair encoding, beginners, teens]
description: "Part 3 of a 12-part series for readers aged 14 to 17. A model works on numbers, not letters. This post builds a byte-pair tokenizer in Python on a whole book, shows the first merges and the trade between vocabulary size and text length, and ties it to the 2016 paper that brought the method to language models."
---

Part 2's model stored words in a table. A neural network, which starts in Part 5, cannot read words at all. It reads lists of numbers. The step that turns text into numbers is called tokenization, and the pieces are called tokens. This post builds a tokenizer from scratch and runs it on the same book as Part 2.

## Three ways to cut text

1. **Letters.** Every character is a token. The vocabulary is tiny (here, 33 different symbols) but the text becomes very long.
2. **Whole words.** Every word is a token. Text is short, but the vocabulary is huge, and any word not in the vocabulary, such as a new name or a typo, has no token at all.
3. **Pieces in between.** Common words get one token; rare words are built from smaller pieces. This is what language models use.

A 2016 paper, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), introduced the version used here to translation systems. Its abstract states the problem as translation being "an open-vocabulary problem" and proposes "encoding rare and unknown words as sequences of subword units", using segmentation "based on the byte pair encoding compression algorithm". The idea is that you do not choose the pieces by hand: you let the data decide.

## The algorithm in one sentence

Start with single letters. Find the pair of neighbouring tokens that occurs most often in the book, and glue them into one new token. Repeat.

## The program

It uses the same `alice.txt` as Part 2. Each word starts as separate letters followed by an end marker `_`, so the tokenizer can tell "the end of a word" from "the letters t-h-e inside a longer word".

```python
import re
from collections import Counter

book = open("alice.txt", encoding="utf-8").read().replace("’", "'")
start = book.index("Down the Rabbit-Hole", book.index("CHAPTER I."))
text = book[start:book.index("*** END OF")].lower()

# every word starts as a tuple of single characters, with "_" marking its end
freq = Counter(re.findall(r"[a-z']+|[.,!?;]", text))
vocab = {tuple(w) + ("_",): n for w, n in freq.items()}

def count_pairs(vocab):
    pairs = Counter()
    for word, n in vocab.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += n
    return pairs

def merge(vocab, pair):
    new = {}
    for word, n in vocab.items():
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1]); i += 2
            else:
                out.append(word[i]); i += 1
        new[tuple(out)] = n
    return new

merges = []
for step in range(1000):
    pair = count_pairs(vocab).most_common(1)[0][0]
    merges.append(pair)
    vocab = merge(vocab, pair)
    if step < 15 or step + 1 in (100, 500, 1000):
        total = sum(len(w) * n for w, n in vocab.items())
        print(step + 1, pair, "-> tokens in book:", total)
```

The loop runs 1,000 times. Each time, `count_pairs` counts every neighbouring pair (weighted by how often the word occurs), `most_common(1)` picks the winner, and `merge` glues that pair everywhere. Here are some of the first merges from a real run (merges 6 and 8 left out to save space):

```
1 ('e', '_')   -> tokens in book: 138238
2 ('t', 'h')   -> tokens in book: 134753
3 ('t', '_')   -> tokens in book: 131453
4 ('d', '_')   -> tokens in book: 128212
5 (',', '_')   -> tokens in book: 125785
7 ('i', 'n')   -> tokens in book: 121513
9 ('th', 'e_') -> tokens in book: 118026
10 ('a', 'n')  -> tokens in book: 116419
```

Before any merge, the book is 144,032 tokens. The first merge glues "e" and the end marker: words ending in "e" now finish with one token. By merge 9 the word "the" is a single token, `the_`, built from `th` and `e_`. The algorithm found it without knowing English.

## Using the tokenizer

To tokenize a new word, apply the merges in the order they were learned:

```python
def encode(word, merges):
    toks = list(word) + ["_"]
    for a, b in merges:
        i = 0
        while i < len(toks) - 1:
            if toks[i] == a and toks[i + 1] == b:
                toks[i:i + 2] = [a + b]
            else:
                i += 1
    return toks
```

With all 1,000 merges, from a real run:

```
alice        ['alice_']                      common: one token
rabbit       ['rabbit_']                     common: one token
wonderland   ['won', 'der', 'l', 'and_']     rarer: four pieces
unwonderful  ['un', 'won', 'der', 'ful_']    not in the book: built from pieces
xyzzy        ['x', 'y', 'z', 'z', 'y_']      nonsense: falls back to letters
```

"unwonderful" never appears in the book, yet it gets a sensible split. "xyzzy" falls back to single letters, which is why a model with this kind of tokenizer can still read any word, even if inefficiently.

## From tokens to numbers

Give each token a number: its position in a list that holds the 33 starting symbols and then each merge in order. That list has 1,033 entries. Then:

```
alice        -> [87]
wonderland   -> [463, 362, 18, 51]
unwonderful  -> [185, 463, 362, 413]
```

That is what the model receives. The word "wonderland" and "unwonderful" share two numbers (463 and 362), because they share two pieces.

![Line chart: tokens needed to write the whole book falls as vocabulary size grows, with marked points at 33 letters, 1,033 tokens after 1,000 merges, and one token per word]({{ site.baseurl }}/assets/pic/teen-llm-vocab-vs-length.png){: width="800" }
*The trade. Letters only: 33 different tokens, 144,032 tokens to write the book. After 1,000 merges: 1,033 tokens, 40,102. One token per word: 2,644 different tokens, 31,070.*

## The trade-off

The chart shows the choice every model designer makes. A bigger vocabulary means shorter text, and a model's work grows with the length of the text it must read. A bigger vocabulary also means more numbers to learn about, and rare tokens that the model sees very few times. Real models choose a vocabulary of tens of thousands of tokens trained on far more text than one book; the size is a design choice, not a law.

## Try it

1. Run the program and print `merges[:50]`. Which merges look like real English pieces (`th`, `ing`)? Which are odd?
2. Encode your own name and three words from a language you speak. Which falls back to single letters?
3. Change `range(1000)` to `range(200)`, then `range(3000)`. How do the pieces of "wonderland" change?
4. Delete the `.lower()` call. How many tokens are the first merges now?

## Limits

- Many real tokenizers work on bytes, not letters, so they can write any character, including emoji and every alphabet. The version here starts from the 33 symbols in this one English book.
- Vocabulary size here is counted as starting symbols plus merges. Some intermediate pieces are no longer used after later merges, so the working vocabulary is smaller than the stated size. The curve ends above the "one token per word" point because after 3,000 merges some rare words are still split into pieces.
- The numbers 87, 463, and so on belong to this one run on this one book. Another text gives other merges and other numbers.
- Tokens are not words and not syllables. Splits such as `won der l and` come from counting, not from any rule of English.
