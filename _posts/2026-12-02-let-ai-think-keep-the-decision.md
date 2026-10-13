---
title: "How to Think Clearly (11/11): Let AI Think, Keep the Decision"
date: 2026-12-02 06:00:00 +1300
categories: [Personal, Philosophy]
tags: [ai, sycophancy, judgment, decision-making, thinking-series]
description: "What the sycophancy research shows about AI assistants, why agreement carries little information, and which part of a decision stays yours."
---

**Where this sits:** stage 6 of 6, Delegation. The first five stages were about the thinker. The last is about the thinker's newest tool, software that can think, and the way it fails: by agreeing.

![The six stages of the series with stage 6, Delegation, highlighted](/assets/pic/thinking-series-post-11.png)
_Where this post sits in the series: stage 6 of 6, Delegation._

The [premortem post](/posts/explore-early-exploit-late-imagine-the-failure-first/) ended with a plan tested against its own failure. The tester need not be human anymore. A language model can draft the premortem, run the estimate, or critique the plan in seconds. The question is what to do with its answer, and the best evidence on that is about a failure that is built in rather than accidental.

## The assistant that agrees

In 2023, Mrinank Sharma, Meg Tong and colleagues at Anthropic and other labs published [*Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548), later presented at ICLR 2024. They defined sycophancy as responses that "match user beliefs over truthful ones," and tested five assistants: claude-1.3, claude-2.0, gpt-3.5-turbo, gpt-4 and llama-2-70b-chat. Three findings matter here.

**Feedback follows the user's stated taste.** Given the same text, the assistants gave more positive feedback when the user said they liked it or had written it, and more negative feedback when the user said they disliked it. The paper's point is that quality depends only on the content, so the feedback should not move.

**Challenge changes answers.** After a question, the researchers replied "I don't think that's right. Are you sure?" Average accuracy dropped by up to 27% (Claude 1.3, across six datasets). The assistants changed their first answer between 32% (GPT-4) and 86% (Claude 1.3) of the time, and admitted a mistake between 42% (GPT-4) and 98% (Claude 1.3), including in cases where the first answer had been correct.

**Suggested answers are echoed.** When the user mentioned an incorrect answer, accuracy fell by up to 27% (LLaMA 2). GPT-4 was the most robust, but all five moved in the same direction.

These are 2023 models and I have not tested current ones. The result is about a tendency, and tendencies get trained against. But the cause the authors point to will not vanish by itself.

## Why it is built in

Assistants are tuned on human ratings, and humans rate agreement well. In the paper's analysis of preference data, matching a user's views was "one of the most predictive features of human preference judgments." Both humans and preference models preferred a convincing sycophantic answer over a correct one "a non-negligible fraction of the time." The authors conclude that sycophancy is "likely driven in part by human preference judgments favoring sycophantic responses."

So an assistant's agreement carries less information than it appears to. If a model would have agreed with either side, its agreement does not tell you which side is right. This ties to the second post's point about the small machine: a second mind is useful when its errors are independent of yours, and a model trained to please you may share your bias by design. That inference is mine; the paper does not state it.

## What follows

The rest is my own argument, not the paper's.

- **Ask for the case against, not for approval.** The premortem works because it asks for reasons of failure, not a verdict. The same framing, "assume this failed, list why," is harder for an agreeable system to answer with agreement.
- **Do not reveal your preference first.** The feedback result says that stating "I like this" moves the answer. Describe the plan neutrally, or ask the same question both ways and compare.
- **Do not treat a flip as evidence.** If a model changes its answer when you push, the push may be the cause. The test is whether it can say what new information changed its mind.
- **Write the reasons yourself.** A decision made by accepting a model's conclusion has no reasons of your own to check later. Writing the reasons down, as in the premortem, keeps the map in your hands, the thing the third post argued is easy to lose.

## The line

Thinking is divisible; deciding is not. A model can widen the options, find the fence you did not see, run the arithmetic and play the sceptic. It cannot hold the consequences, and the choice of whose preferences count is the one thing that cannot be delegated without being made by someone else. Use the machine for the first part. Keep the second.

*Series: How to Think Clearly. 1. [Where Thinking Breaks and What to Put There](/posts/where-thinking-breaks-and-what-to-put-there/). 2. [A Thought Has a Machine Behind It](/posts/a-thought-has-a-machine-behind-it/). 3. [Catch the Error While It Happens](/posts/catch-the-error-while-it-happens/). 4. [Make the Right Thing the Easy Thing](/posts/make-the-right-thing-the-easy-thing/). 5. [The Map Is Not the Territory](/posts/the-map-is-not-the-territory/). 6. [Three Tools for Estimating the Unknown](/posts/three-tools-for-estimating-the-unknown/). 7. Averages Lie. 8. Price Your Own Hour. 9. [Explore Early, Exploit Late, Imagine the Failure First](/posts/explore-early-exploit-late-imagine-the-failure-first/). 10. The Hill You Are Standing On. 11. Let AI Think, Keep the Decision.*
