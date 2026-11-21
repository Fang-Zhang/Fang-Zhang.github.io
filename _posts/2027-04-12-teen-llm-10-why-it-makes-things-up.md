---
title: "How Language Models Work, Part 10: Why It Makes Things Up"
date: 2027-04-12 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, hallucination, temperature, sampling, softmax, beginners, teens]
description: "Part 10 of a 12-part series for readers aged 14 to 17. One dial, temperature, changes how a model picks its next word. This post shows it in code and in a measurement on the Part 8 Transformer, then explains why a model continues with fluent text even when it has no facts, citing two research papers."
---

This post does two things. It shows the dial that controls how random a model's output is, and it shows why a model writes fluent text even about things it never saw. You can see both in about 20 lines of code.

## The maths tool: temperature

A model does not output a word. It outputs a score for every word in its vocabulary, and softmax (Part 6) turns the scores into chances. **Temperature** `T` is one extra step: divide every score by `T` before softmax.

`chance of word i = e^(score_i / T) / sum of e^(score_j / T) over all words j`

- `T` below 1 makes the gaps between scores larger, so the top word takes more of the chance.
- `T` above 1 shrinks the gaps, so the chances move toward equal.
- `T = 1` changes nothing.

Run this with made-up scores for four words:

```python
import math

scores = {"not": 2.0, "going": 1.9, "no": 1.4, "quite": 1.3}      # made-up scores for 4 words

def chances(scores, T):
    e = {w: math.exp(s / T) for w, s in scores.items()}           # divide by T, then e**score
    total = sum(e.values())
    return {w: v / total for w, v in e.items()}                   # softmax, as in Part 6

for T in (0.5, 1, 2, 100):
    print("T =", T, {w: round(100 * p) for w, p in chances(scores, T).items()})

def nucleus(chances, top_p):                                      # fewest words that add up to top_p
    kept, running = {}, 0
    for w, p in sorted(chances.items(), key=lambda x: -x[1]):
        if running >= top_p: break
        kept[w] = p; running += p
    return kept

print(list(nucleus(chances(scores, 1), 0.6)))
```

Output, from a real run:

```
T = 0.5 {'not': 42, 'going': 35, 'no': 13, 'quite': 10}
T = 1 {'not': 34, 'going': 31, 'no': 19, 'quite': 17}
T = 2 {'not': 29, 'going': 28, 'no': 22, 'quite': 21}
T = 100 {'not': 25, 'going': 25, 'no': 25, 'quite': 25}
['not', 'going']
```

The scores never changed. Only the picking rule did. The last function, `nucleus`, is a second rule: keep the fewest words whose chances add up to `top_p`, drop the rest, and pick among those. It is called nucleus sampling, from [a 2019 paper](https://arxiv.org/abs/1904.09751) that found "using likelihood as a decoding objective leads to text that is bland and strangely repetitive", and that "decoding strategies alone can dramatically effect the quality of machine text, even when generated from exactly the same neural language model".

## The dial on a real model

Same method as Part 8: the 2-layer Transformer, trained for 200 steps on the Alice text (the step where its test loss was lowest, 3.87). Next word after "alice was", at three temperatures:

```
T=0.5: not 26%  going 22%  no 14%  quite 12%
T=1:   not 11%  going 10%  no  8%  quite  8%
T=2:   not  3%  going  3%  no  3%  quite  3%
```

At T = 0.5, two words share almost half the chance. At T = 2, the model's top words are nearly equally likely, and so are hundreds of others. Now 25 words from the same prompt:

```
T=0.2: alice was going to herself , and <other> to the <other> , and <other> to <other> , and <other> , and <other> <other> <other> , <other> <other>
T=1:   alice was this <other> curious as <other> the door , said alice , not a <other> of feet , and waited of their go to his tea
T=3:   alice was afraid yet through dinah and birds twinkle poor think it's seem that idea sort one indeed reason live three tail may is , tree over
```

Low `T` loops (", and <other> to"). High `T` is a list of words with no sentence. In between, it is sometimes grammatical and often not, which is what a small model deserves.

![Left: chance of the 8 likeliest next words after "alice was" at temperatures 0.5, 1 and 2: the lower the temperature, the steeper the curve. Right: as temperature rises from 0.2 to 3, the share of word pairs found in the book falls from 100 percent to 16 percent while the share of distinct words rises from 22 percent to 96 percent]({{ site.baseurl }}/assets/pic/teen-llm-temperature.png){: width="900" }
*Left: the same scores at three temperatures. Right: 60 samples of 30 words from random held-out prompts at each temperature. "Pairs found in the book" counts adjacent word pairs that also appear in the training text. "Distinct words" is the share of different words among the 30.*

The right panel is a trade. At T = 0.2 every word pair came from the book, but only 22% of the words were different: the output repeats. At T = 3 the words are varied, but only 16% of the pairs ever occurred in the book. With `top_p = 0.9` at T = 1, the same test gave 86% pairs found and 74% distinct words, against 80% and 79% without it. That is a small gain in fluency for a small loss in variety. Real chat products choose settings like these behind the scenes.

## Why the model keeps going

The model above has no way to say "I have no information". Its last step is softmax, and softmax always gives chances that add up to 100%. Ask it something the book never mentions:

```
prompt: the capital of france is
   the <other> of <other> is which <other> , in the baby ? the <other> ; any use , it
   the <other> of <other> is <other> at my dear ? would rather . a gardeners dinah is his tea
   the <other> of <other> is <other> at the right size on and the hatter ; but off <other>
```

(The words "capital" and "france" are not among the 500 it knows, so the model sees `<other>`.) This model is too small to sound convincing, but the mechanism is the same in a large one: it continues with whatever scores highest, and a wrong continuation can look as fluent as a right one.

Researchers call this **hallucination**: output that reads well and is false or unsupported. A [survey of the field](https://arxiv.org/abs/2202.03629) says that deep-learning text generation "is prone to hallucinate unintended text". A [2025 paper](https://arxiv.org/abs/2509.04664) gives a cause in two parts. First, "if incorrect statements cannot be distinguished from facts, then hallucinations in pretrained language models will arise through natural statistical pressures". Second, tests are usually graded so that guessing beats admitting doubt: "language models are optimized to be good test-takers, and guessing when uncertain improves test performance". Their abstract puts it as models producing "plausible yet incorrect statements instead of admitting uncertainty".

Notice what temperature does not do. Setting `T` near zero removes randomness; it does not add facts. The model then repeats its single likeliest word, which can be the wrong one, every time.

## Try it

1. In the first code block, change the scores so `not` is 3.0 and the others are 1.0. At what `T` does `not` drop below 50%?
2. Run `nucleus(chances(scores, 1), 0.9)`. Which words survive? What happens at 0.3?
3. Ask any chatbot you are allowed to use (check its age limit) for a fact you can look up in a book, such as the date of a historic event. Then ask it for something that does not exist, such as a quote from an invented person. Compare how confident the two answers sound.

## Limits

- The measurements use one small model, 200 training steps, and one book. The numbers show a trade-off, not how a large model behaves.
- "Pairs found in the book" is a crude fluency test: a pair can be in the book and still make a bad sentence.
- The hallucination papers are about large models; this post's tiny model shows the missing "I don't know" step, not the whole cause.
- Temperature and nucleus sampling are two of several picking rules, and products differ in what they use.
