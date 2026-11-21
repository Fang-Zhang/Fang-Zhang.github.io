---
title: "How Language Models Work, Part 4: Meaning as Position"
date: 2027-04-06 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, embeddings, vectors, beginners, teens]
description: "Part 4 of a 12-part series for readers aged 14 to 17. Words become points on a map. This post builds word vectors from counting on a whole book, uses the dot product to measure closeness, shows the clusters on a map, and shows where the method fails."
---

Part 3 turned words into ID numbers. An ID like 463 carries no information: 463 and 464 are not "close" in any useful way. This post gives each word a list of numbers that does carry information, so that words used in similar ways get similar lists. Such a list is called an **embedding** (or word vector).

## The maths tool: points and the dot product

A point on a map is two numbers, `(x, y)`. A word vector is the same idea with more numbers: 50 instead of 2. To compare two points, you can multiply matching numbers and add them up. This is the **dot product**:

```
a = (3, 4)    b = (4, 3)    c = (4, -3)

a . b = 3*4 + 4*3  = 24
a . c = 3*4 + 4*-3 = 0
```

Arrows `a` and `b` point in nearly the same direction, and their dot product is large. Arrows `a` and `c` are at right angles, and their dot product is 0. Divide by the two arrow lengths (5 and 5) and you get the **cosine similarity**: 24 / 25 = 0.96 for `a` and `b`, 0 for `a` and `c`. A value near 1 means "same direction", 0 means "unrelated". Everything below uses this one number.

## An idea from linguistics: words are known by their neighbours

The words "said" and "replied" are used in the same places: after a name, before a quote. So give each word a description of what appears around it, and two words used in similar places will get similar descriptions. This is how [Mikolov and colleagues (2013)](https://arxiv.org/abs/1301.3781) trained word vectors, though with a small neural network rather than the counting used here. Their paper reports that learning high-quality word vectors from a 1.6-billion-word data set took less than a day.

## The program

It counts, for every pair among the 1,000 most common words in the same book as Parts 2 and 3, how often they appear within two places of each other. Then it converts the counts into scores of "how much more often than chance", and uses a standard maths operation called SVD to squeeze each word's row of 1,000 scores into 50 numbers. You do not need to follow the SVD step; treat it as "squeeze, keeping what matters most". It needs the NumPy package.

```python
import re, numpy as np
from collections import Counter
book = open("alice.txt", encoding="utf-8").read().replace("’", "'")
start = book.index("Down the Rabbit-Hole", book.index("CHAPTER I."))
words = re.findall(r"[a-z']+", book[start:book.index("*** END OF")].lower())   # no punctuation here
freq = Counter(words)
vocab = [w for w, n in freq.most_common(1000)]            # keep the 1,000 most common words
idx = {w: i for i, w in enumerate(vocab)}

# count how often each pair of words appears within 2 places of each other
C = np.zeros((len(vocab), len(vocab)))
for i, w in enumerate(words):
    if w not in idx: continue
    for j in range(max(0, i - 2), min(len(words), i + 3)):
        if j != i and words[j] in idx:
            C[idx[w], idx[words[j]]] += 1

# turn counts into "surprise" scores (positive PMI), then squeeze to 50 numbers per word with SVD
total = C.sum(); pw = C.sum(1, keepdims=True) / total; pc = C.sum(0, keepdims=True) / total
with np.errstate(divide="ignore", invalid="ignore"):
    ppmi = np.maximum(np.log((C / total) / (pw * pc)), 0)
ppmi = np.nan_to_num(ppmi)
U, S, Vt = np.linalg.svd(ppmi)
E = U[:, :50] * S[:50]
E = E / np.linalg.norm(E, axis=1, keepdims=True)           # unit length, so dot product = cosine

def neighbours(w, k=6):
    sims = E @ E[idx[w]]
    return [vocab[i] for i in np.argsort(-sims)[1:k + 1]]

for w in ["queen", "alice", "said", "rabbit", "she", "very", "little"]:
    print(w, "->", neighbours(w))
```

Output from a real run:

```
queen -> ['gryphon', 'king', 'off', 'the', 'alice', 'dormouse']
alice -> ['said', 'but', 'mouse', 'exclaimed', 'it', 'she']
said -> ['alice', 'the', 'replied', "that's", 'continued', 'repeated']
rabbit -> ['white', 'hole', 'mock', 'gryphon', 'lory', 'hare']
she -> ['it', 'as', 'and', 'was', 'that', 'alice']
very -> ['quite', 'alice', 'indeed', 'a', 'feeling', 'remarks']
little -> ['tiny', 'a', 'kid', 'golden', 'thing', 'key']
```

The program was given no definitions. It only counted neighbours. Even so, "said" sits next to "replied", "continued" and "repeated"; "very" next to "quite" and "indeed"; "little" next to "tiny".

## The map

Fifty numbers per word cannot be drawn, so the figure below squashes them to two with the same kind of SVD. Words from four hand-picked groups (characters, speech verbs, size and degree words, pronouns) are coloured by group.

![Scatter plot of 29 words on a flat map. Size words such as tiny, little, large and small sit at the left, speech verbs such as said, replied, asked and cried sit at the bottom, characters such as queen, king, hatter and gryphon sit at the bottom right, and pronouns such as she, it, they and you sit at the top right]({{ site.baseurl }}/assets/pic/teen-llm-word-map.png){: width="850" }
*Word vectors computed from one book, flattened to two directions. The colours were added by hand afterwards; the program never saw them.*

The groups separate into regions. The dots for "queen", "king" and "hatter" are close to each other because they appear in the same kinds of sentences in this book, not because the program knows they are characters.

## Where it fails

Look again at the neighbours of "queen": `gryphon, king, off, the, alice, dormouse`. The words "off" and "the" are not related in meaning to "queen". They are here because "off with her head" and "the queen" are common phrases in this book. Position records how a word is used, which is only partly the same as what it means.

A second failure: the 2013 paper describes an example where `vector("King") - vector("Man") + vector("Woman")` gives a vector closest to "Queen". Try it here and it cannot even start: "woman" appears only twice in this book, so it is not in the 1,000-word vocabulary. One book is too small to give reliable vectors for rare words.

## Try it

1. Change the word in `neighbours("...")` to `"hatter"`, `"alice"`, `"rabbit"`, `"cried"`. Which lists look sensible? Which contain surprises?
2. Change the window `i - 2` and `i + 3` to `i - 1` and `i + 2` (neighbours within 1 place). Do the lists change?
3. Replace 50 in `U[:, :50]` with 5, then with 300. What happens to the lists?
4. Compute the cosine similarity of "queen" and "king", then "queen" and "she": `float(E[idx["queen"]] @ E[idx["king"]])`.

## Limits

- This method is simpler than the neural-network methods in the 2013 paper. It uses counting and SVD. The paper is cited for the idea, not as a description of this code.
- The 2D map throws away 48 of 50 numbers, so some distances in the figure are misleading. Words can look close on the map and not be close in 50 dimensions.
- The groups were chosen by hand, after looking at the results, to show a pattern that exists. Another selection of words would show less tidy groups.
- A real model learns its word vectors during training (Parts 5 and 6) rather than by counting first.
