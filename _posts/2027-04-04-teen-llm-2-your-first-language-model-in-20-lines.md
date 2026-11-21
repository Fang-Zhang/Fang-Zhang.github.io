---
title: "How Language Models Work, Part 2: Your First Language Model in 21 Lines"
date: 2027-04-04 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, Python, beginners, teens]
description: "Part 2 of a 12-part series for readers aged 14 to 17. The next-word table from Part 1 becomes a short Python program trained on a whole book, with its real output, and a one-number change that shows why more context means more copying."
---

Part 1 built a next-word machine with a pencil and a die. This post turns it into a Python program and trains it on a whole book. You do not need to have programmed before: every line is explained, and you only need to run it. The output shown is from a real run.

## What you need

- Python 3 on any computer. Search for "run Python online" if you have not installed it; any online runner that lets you upload a file works, and none of this needs an account or personal data.
- The book: [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11), from Project Gutenberg, which states that the text may be copied and re-used under the Project Gutenberg License. Download the "Plain Text UTF-8" version and save it as `alice.txt` next to your program.

## The program

```python
import random, re
from collections import defaultdict

book = open("alice.txt", encoding="utf-8").read().replace("’", "'")
start = book.index("Down the Rabbit-Hole", book.index("CHAPTER I."))
end = book.index("*** END OF")
words = re.findall(r"[a-z']+|[.,!?;]", book[start:end].lower())

follows = defaultdict(list)             # word -> list of the words that came next
for a, b in zip(words, words[1:]):
    follows[a].append(b)

def write(first, n=25, seed=1):
    random.seed(seed)
    out = [first]
    for _ in range(n):
        out.append(random.choice(follows[out[-1]]))
    return " ".join(out)

print(len(words), "words,", len(set(words)), "different words")
print(write("alice"))
```

## What each part does

- **Lines 1-2.** Load two tools: `random` for dice, `defaultdict` for a dictionary that starts empty.
- **Lines 4-7.** Read the book, cut off the licence text before and after the story, make everything lowercase, and split it into "words". The pattern `[a-z']+|[.,!?;]` means: a run of letters, or one punctuation mark. Punctuation counts as a word, as the full stop did in Part 1.
- **Lines 9-11.** This is the whole "training". For every neighbouring pair `(a, b)`, add `b` to the list of words that followed `a`. If "the" is followed by "queen" 65 times, then "queen" appears 65 times in the list for "the". The list is Part 1's table, stored as repeats instead of fractions.
- **Lines 13-18.** Writing. Start with a word, then repeat: look at the last word, pick one entry at random from its list, add it to the text. Picking at random from a list with repeats gives each word the right share. `random.seed(seed)` fixes the dice so you get the same text every time.
- **Lines 20-21.** Print counts, then write 25 words starting from "alice".

## The output

```
31070 words, 2644 different words
alice . the place , the race ? persisted the time she gave a simple rules their paws and all the footman seemed too late ,
```

The book has 31,070 words and punctuation marks, of which 2,644 are different. The "model" is a table with 2,644 rows. The writing is nonsense, but each neighbouring pair of words occurs somewhere in the book. Change `seed=1` to 2, 3 or 4 and you get different nonsense.

![Left: the eight most common words after the in the book. Right: the share of contexts followed by only one possible word, for models that look back one word and two words]({{ site.baseurl }}/assets/pic/teen-llm-bigram-stats.png){: width="900" }
*Left: what follows "the" in this book (1,650 occurrences, 451 different next words). Right: how often the table offers no choice.*

The left chart is the table row for "the". In this book it is most often followed by "queen" (65 times), "king" (60), "mock" (54), "gryphon" (53) and "hatter" (51). This is a fact about this book, not about English. A model trained on a different book has a different table.

## One change: look back two words

The program looks at one word. Change it to look at two, so the table key is a pair of words:

```python
follows = defaultdict(list)
for i in range(len(words) - 2):
    follows[(words[i], words[i + 1])].append(words[i + 2])

def write(first, second, n=25, seed=1):
    random.seed(seed)
    out = [first, second]
    for _ in range(n):
        out.append(random.choice(follows[(out[-2], out[-1])]))
    return " ".join(out)

print(write("the", "queen"))
```

On the same book this gave:

```
the queen , pointing to alice , and she did it at last , and put back into the roof off . alice said very politely did
```

Better: longer stretches read like sentences ("and she did it at last"). Also a warning sign, shown on the right of the figure. With one word of context, 48% of the table's rows have only one possible next word. With two words of context, 76% do, so in three out of four situations the machine has no choice and copies the book.

That is the central trade of this whole series. More context makes the text more sensible and, in this kind of table, makes the machine copy more. The tables also get huge: 14,366 rows for two words, and the number grows with every extra word. A table cannot be the answer for looking back hundreds of words. Parts 3 to 8 build what replaces it.

## Try it

1. Run the program. Change the starting word `"alice"` to `"the"`, `"she"` or `"queen"`.
2. Change the seed and find a sentence from your run that you think the book contains. Then search the file for it. Is it really there?
3. Count how long a copied stretch is: find the longest run of words in the two-word output that appears unchanged in the book.
4. Train it on a different text, saved as `alice.txt`, and see what its table for "the" looks like.

## Limits

- This is a counting model. It does not know what the words mean and does not understand the story.
- Output depends on the random seed and Python version, so your text may differ from the text printed here, but the counts (31,070 and 2,644) should match if you use the same file.
- The two-word version is meant for teaching. It can stop with an error if the last two words only occur at the very end of the book; a real program would handle that case.
- The 48% and 76% figures are for this one book and this cleaning of the text.
