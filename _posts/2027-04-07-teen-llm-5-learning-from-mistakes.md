---
title: "How Language Models Work, Part 5: Learning from Mistakes"
date: 2027-04-07 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, loss, gradient descent, beginners, teens]
description: "Part 5 of a 12-part series for readers aged 14 to 17. A model with one adjustable number learns by measuring how surprised it is and nudging the number downhill. This post runs that loop in Python, recovers the counting answer from Part 2, and shows what happens when the step size is wrong."
---

So far the "model" learned by counting. Counting stops working once a model has millions of adjustable numbers and cannot keep a table. Instead, models are trained: start with a guess, measure how wrong it is, nudge the numbers to be a little less wrong, and repeat. This post does that with exactly one adjustable number.

## The task

In the Part 2 book, the word "the" is followed by "queen" some fraction of the time. Counting gives the answer directly: 65 of the 1,650 times, or 3.94%. Pretend we cannot count, and let a model find it by trial and error.

The model has one knob, `w`, and gives the chance of "queen" as

```
p = 1 / (1 + e^(-w))
```

This is the **sigmoid** function. It turns any number `w` into a chance between 0 and 1: `w = 0` gives `p = 0.5`; a large negative `w` gives `p` close to 0; a large positive `w` gives `p` close to 1.

## The maths tool 1: surprise

How wrong is a guess? Measure how surprised the model is by what actually happened. If it gave an outcome chance `p`, the surprise is `-ln(p)`, where `ln` is the natural logarithm (a button on any calculator):

```
p = 0.9   -> surprise 0.11   (not surprised)
p = 0.5   -> surprise 0.69
p = 0.1   -> surprise 2.30
p = 0.04  -> surprise 3.22   (very surprised)
```

Average the surprise over every real "the ..." in the book and you get the **loss**. Low loss means the model gave high chances to what really happened. This is called cross-entropy loss, and it is the standard loss for language models, which are trained to give high chances to the words that actually come next.

## The maths tool 2: the slope

Picture the loss as a hill on a graph: the knob `w` along the bottom, the loss up the side. If the slope at your position points uphill to the right, move left. The size of the slope tells you how far from the bottom you probably are. For this particular model and loss, the slope works out to a short formula: `p - k/n`, where `k/n` is the real fraction (65/1650). If the model's chance is too high, the slope is positive and you decrease `w`. If too low, the slope is negative and you increase it.

The loop, called **gradient descent**, is:

```
w = w - step * slope
```

The `step` (often called the learning rate) says how far to move each time.

## The program

```python
import math, re

book = open("alice.txt", encoding="utf-8").read().replace("’", "'")
start = book.index("Down the Rabbit-Hole", book.index("CHAPTER I."))
words = re.findall(r"[a-z']+|[.,!?;]", book[start:book.index("*** END OF")].lower())

after_the = [b for a, b in zip(words, words[1:]) if a == "the"]
n = len(after_the)                       # how many times "the" appears
k = after_the.count("queen")             # how many of those were "queen"
print(n, k, k / n)

def loss(w):
    p = 1 / (1 + math.exp(-w))           # the model's chance of "queen"
    return -(k * math.log(p) + (n - k) * math.log(1 - p)) / n

def slope(w):
    p = 1 / (1 + math.exp(-w))
    return p - k / n                     # how loss changes if w grows

w, step = 0.0, 10.0                      # start with a guess: p = 50%
for i in range(41):
    if i in (0, 1, 2, 3, 5, 10, 20, 40):
        print(i, round(w, 3), round(1 / (1 + math.exp(-w)), 4), round(loss(w), 4))
    w = w - step * slope(w)
```

Output from a real run (columns: step number, `w`, `p`, loss):

```
1650 65 0.0394
0 0.0    0.5    0.6931
1 -4.606 0.0099 0.1914
2 -4.311 0.0132 0.1832
3 -4.05  0.0171 0.1768
5 -3.646 0.0254 0.1694
10 -3.25  0.0373 0.1661
20 -3.194 0.0394 0.166
40 -3.194 0.0394 0.166
```

The model starts at 50% and ends at 3.94%, the same number that counting gave. Nobody told it 65 or 1,650 in the training loop; the loop only had the slope. The lowest loss it can reach is 0.166 because even the perfect guess cannot remove the real uncertainty about which word comes next.

![Left: the loss curve as a function of the knob, with red dots showing gradient-descent steps of size 10 that jump left in step 1 and then creep to the bottom. Right: loss versus number of steps for step sizes 1, 10 and 200]({{ site.baseurl }}/assets/pic/teen-llm-loss-descent.png){: width="900" }
*Left: the descent on the loss hill. Right: three step sizes. Size 1 is slow, size 10 gets there in a few steps, size 200 starts worse than where it began.*

## Where it fails: the step is too big

Change `step` from 10 to 200 and rerun. The first step throws `w` far past the bottom of the hill, and the loss goes from 0.693 to 3.63, worse than the starting guess. It then slowly comes back down (the red line). Change it to 1 and it works, but slowly: after 40 steps `p` is still 5.0%, not 3.9%. Choosing the step size is part of training a real model, and a wrong choice can waste days of computer time.

## Why this matters for real language models

A real model has billions of knobs instead of one. The same loop applies: for each knob, the slope says which direction reduces the loss, and every knob moves a little. Computing all the slopes at once is what the word **backpropagation** names; Part 6 builds a small network where it is needed.

## Try it

1. Run the program with `step` set to 1, 10, 50 and 200. Record the loss after 10 steps for each.
2. Change `"queen"` to `"king"` or `"hatter"` and see what `p` it settles on. Check against `k / n`.
3. Change the starting `w` from 0 to 5 (a bad guess: 99% for "queen"). Set `step` to 1 and count how many steps it takes to reach a loss under 0.2.
4. Replace "the" with "alice". Does it still reach the counted fraction?

## Limits

- One knob is a toy. Real models have many knobs that interact, and their loss has hills and valleys, not a single smooth bowl.
- The formula `p - k/n` is specific to this sigmoid-plus-cross-entropy case. In general, slopes are computed automatically by software.
- This model always gives the same chance for "queen" after "the", whatever came earlier in the sentence. Parts 7 and 8 fix that.
