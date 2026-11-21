---
title: "How Language Models Work, Part 12: Using One Well"
date: 2027-04-14 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, verification, self-consistency, study habits, beginners, teens]
description: "Part 12 of a 12-part series for readers aged 14 to 17. Asking the same question many times and trusting the majority works only when errors are independent. This post tests it on the Part 11 chatbot, where agreement was high and the answers were wrong, and turns the series into a short checklist for using a language model for study. Sources: a 2022 paper, UNESCO guidance, and one provider's terms."
---

Eleven parts explained what a language model computes. This one turns that into habits. It tests one popular check, "ask several times and see whether the answers agree", and shows when it fails.

## What the series established

- A model assigns chances to the next word and samples one (Parts 1, 2, 10).
- The chances come from patterns in training text, with no database of facts behind them (Parts 6 to 9).
- It always produces an answer, true or not, because softmax always sums to 100% (Part 10).
- A chat format is a second round of training on top of that (Part 11).

So the question to ask of any answer is not "does it sound right?" Sounding right is the one thing the model is trained to do.

## The maths tool: majority vote

Suppose each answer is right with chance `p` and answers are independent, like separate coin flips. The chance that more than half of `n` answers are right is a sum of binomial terms (basic probability: choose which `k` answers are right, multiply by `p^k` and `(1-p)^(n-k)`).

```python
from math import comb

def majority_right(p, n):
    """Chance that more than half of n independent answers are right, if each is right with chance p."""
    return sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n // 2 + 1, n + 1))

for p in (0.6, 0.4):
    print(f"p = {p}:", {n: round(majority_right(p, n), 2) for n in (1, 5, 11, 51)})
```

Output:

```
p = 0.6: {1: 0.6, 5: 0.68, 11: 0.75, 51: 0.93}
p = 0.4: {1: 0.4, 5: 0.32, 11: 0.25, 51: 0.07}
```

If a model is right more often than not, voting makes it more reliable. If it is wrong more often than not, voting makes it more reliably wrong. A [2022 paper](https://arxiv.org/abs/2203.11171) built a method on the first case, called self-consistency: sample many reasoning paths and take the most consistent answer. Its abstract reports gains on arithmetic and commonsense tests such as GSM8K (+17.9%). That is a result on those benchmarks, with a large model. It does not say agreement is proof.

![Left: chance that the majority of n independent answers is right, rising toward 100 percent when each answer is right with chance 0.6 and falling toward 0 when it is 0.4. Right: for the Part 11 chatbot, the share of 50 replies equal to the most common reply for each of ten questions; correct most-common replies in blue, wrong ones in red; three new questions have wrong most-common replies at 82, 96 and 92 percent agreement]({{ site.baseurl }}/assets/pic/teen-llm-agreement.png){: width="900" }
*Left: the maths of voting. Right: measured on the Part 11 chatbot, 50 samples per question.*

## The failure to reproduce: agreeing and wrong

Take the Part 11 chatbot, ask each of the ten questions 50 times, and count how often the replies are identical to the most common one:

```
who said that ?            | alice said that .            | 100% | yes
what did she say ?         | she said that .              |  20% | NO
where did she go ?         | she went to the door .       | 100% | yes
who is the queen ?         | the queen is not here .      | 100% | yes
what is the king ?         | the king is very old .       | 100% | yes
who was at the door ?      | the rabbit was at the door . | 100% | yes
what did alice see ?       | alice saw the white rabbit . |  98% | yes
who is the cat ?           | the queen is not here .      |  82% | NO  (new question)
what did the hatter say ?  | the rabbit was at the door . |  96% | NO  (new question)
where is the door ?        | the king is very old .       |  92% | NO  (new question)
```

The three new questions got the same wrong reply 82%, 96% and 92% of the time. Agreement measures how peaked the model's chances are. It does not measure whether the answer is true. The votes here are not independent, because every sample comes from the same chances and the same wrong pattern: voting changed nothing. The one question where replies disagreed ("what did she say ?", 20%) was also wrong. Disagreement is a useful warning; agreement is only weak evidence.

## A checklist

1. **Check anything you will rely on against a second source.** A textbook, an official site, a teacher. Part 10 gave the reason: the model produces fluent text whether or not it has the fact, and a [2025 paper](https://arxiv.org/abs/2509.04664) on why this happens says models "guess when uncertain" because tests reward guessing.
2. **Treat sources and quotes the model gives you as unchecked.** Open the link. Search for the quote. If you cannot find it, assume it was made up.
3. **Re-do any calculation yourself.** The model handles digits as tokens (Part 3). It does not run an arithmetic routine unless a tool does it for it.
4. **Ask for steps, then check the steps.** A wrong step is easier to see than a wrong final answer.
5. **Use it to practise, not to skip practice.** Ask it to quiz you, to explain a mistake you made, or to give a second explanation of a topic. Then try a problem without it.
6. **Keep personal details out.** Part 9 covered why: text you enter may be stored or used. UNESCO's [2023 guidance on generative AI in education](https://www.unesco.org/en/articles/guidance-generative-ai-education-and-research) calls for "mandating the protection of data privacy" and for "setting an age limit for the independent conversations with GenAI platforms". The age is a recommendation to governments; [UN News](https://news.un.org/en/story/2023/09/1140477) reports that the guidance suggests considering an age limit of 13. Providers set their own rules: one provider's [terms of use](https://openai.com/policies/terms-of-use/) say "You must be at least 13 years old or the minimum age required in your country", and "if you are under 18 you must have your parent or legal guardian's permission". Read the terms of whichever service you use.
7. **Know the rules before you hand work in.** Whether using a model on homework counts as cheating depends on your school and your teacher, so ask them, and say when you used one. Handing in text you did not write and cannot explain does not help you, because the point of the work is that you can do it.

## Try it

1. Run the voting code with `p = 0.5`. What does the chance of a right majority do as `n` grows? Explain why in one sentence.
2. Pick a topic you know well. Ask a chatbot you are allowed to use ten factual questions about it. Count right, wrong and unsure. Ask each wrong one again. Does the answer change?
3. Choose one habit from the checklist and use it for a week of homework.

## Limits

- The agreement test uses one tiny model with ten questions. It shows that agreement and correctness can be separate. It does not give a rate for any real chatbot.
- The self-consistency gains are from one paper's benchmarks and models; results differ by task and model.
- This series explains the core machinery of text prediction. Products add searching, calculators and other tools around the model, which change what is reliable. Ask what a particular product does before trusting it.
- The terms and guidance quoted here were read while this series was written and may change. Read the current version.
