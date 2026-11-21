---
title: "How Language Models Work, Part 1: A Machine That Guesses the Next Word"
date: 2027-04-03 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, probability, beginners, teens]
description: "Part 1 of a 12-part series for readers aged 14 to 17. A chatbot is built on one job, guessing the next word. This post does that job with a pencil, a table of counts and a die, and shows what the 1948 paper that first did it by hand got."
---

This is the first of twelve posts that explain how a language model works, for readers aged 14 to 17. You need school maths (fractions) and nothing else. No code in this post. By the end you will have built, on paper, a machine that writes text, and you will know exactly what it does and what it cannot do.

## The one job

A chatbot looks like it answers questions. Underneath, a language model has one job: given some text, give a score to every possible next word. Pick one, add it to the text, and repeat. A reply of 200 words is that job done 200 times.

Start with a small version you can do by hand.

## A tiny library

Here is the entire library. It has four sentences. The full stop counts as a word.

```
the cat sat on the mat .
the dog sat on the rug .
the cat saw the dog .
the dog saw the cat .
```

For each word, count which words come right after it. The word "the" appears 8 times. What followed it:

- cat: 3 times
- dog: 3 times
- mat: 1 time
- rug: 1 time

So after "the", "cat" came 3 times out of 8. The other words get the same treatment:

![Four bar charts of what follows the words the, cat, sat and saw in the tiny library, as shares of the time]({{ site.baseurl }}/assets/pic/teen-llm-next-word-table.png){: width="900" }
*Counts from the four-sentence library. After "the": cat 3/8, dog 3/8, mat 1/8, rug 1/8. After "sat": always "on".*

That table is the model. It is a list of "after this word, these words follow, this often". Nothing else is stored.

## Make it write

Writing means rolling a die and looking up the table. Start from "the". Use a die with 8 sides, and give "cat" three faces, "dog" three faces, "mat" one and "rug" one:

| Roll | Next word |
|---|---|
| 1, 2, 3 | cat |
| 4, 5, 6 | dog |
| 7 | mat |
| 8 | rug |

Say you roll a 5, so the text is "the dog". Now look up "dog". It was followed by "sat", "." and "saw", once each. Use a 3-sided die. Keep going. A run of this procedure produced:

```
the dog sat on the mat . the cat . the cat sat on the
```

Check it against the library. "the dog sat on the mat" is not a sentence in the library, but each pair of neighbouring words is. That is the whole trick: the machine only ever looks at the last word, and it still produces sentences that look right.

## This is not new

In 1948 Claude Shannon published [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf). In section 3 he builds "a series of approximations to English" by a closely related method, from tables of letter and word frequencies (or an equivalent shortcut he describes, reading the next letter out of a book). His two word-level examples:

- Words chosen independently with their normal frequencies (the machine ignores the previous word): "REPRESENTING AND SPEEDILY IS AN GOOD APT OR COME CAN DIFFERENT NATURAL HERE HE THE A IN CAME THE TO OF TO EXPERT GRAY COME TO FURNISHES THE LINE MESSAGE HAD BE THESE."
- Words chosen using the probability of each word given the previous word (this post's machine): "THE HEAD AND IN FRONTAL ATTACK ON AN ENGLISH WRITER THAT THE CHARACTER OF THIS POINT IS THEREFORE ANOTHER METHOD FOR THE LETTERS THAT THE TIME OF WHO EVER TOLD THE PROBLEM FOR AN UNEXPECTED."

He notes that "the resemblance to ordinary English text increases quite noticeably at each of the above steps", and that in the second example "sequences of four or more words can easily be placed in sentences without unusual or strained constructions". Look at the second example again: short stretches make sense, and the whole thing does not.

## What this machine cannot do

Run the machine long enough and it loses the thread. It knows only the last word, so it cannot remember that the sentence started with "the cat". Shannon's second example shows the same problem. Fixing it means giving the machine more of the earlier words to look at, and a better way of using them. Most of the next eleven posts are about doing that.

One more thing to notice. The machine does not know what a cat is. "cat" is a label that is followed by other labels some fraction of the time. A real language model is far larger and cleverer, but its job is the same as this one: score the next word, and the scores come from patterns in text it was trained on.

## Try it

1. Write four sentences of your own, about anything. Count what follows each word, and build the table for the two most common words.
2. Use a die or coin to write 12 words. Does your machine ever write something that is not in your library as a whole sentence?
3. Change one sentence in the library. Which row of the table changes? The next post puts this into 20 lines of Python so you can run it on a whole book.

## Limits

- The machine here looks back one word. Real models look back thousands of words and do not use a lookup table; the next posts show how.
- The 8-sided die in the table is a teaching device. Real systems use computer random numbers.
- The sample output above is one run of the procedure with one random seed; another run gives different text.
- Shannon's frequency tables cite sources from 1923 and 1939, and he made the samples to illustrate his theory, not to build a text generator.
