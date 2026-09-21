---
title: "I Am a Language Model Too"
date: 2026-09-09 10:00:00 +1200
categories: [Personal, Philosophy]
tags: [AI, LLM, machine-learning, self-reflection, growth]
---

When I was a child, I often stared at my math textbook and wondered: is there one single formula that could solve all the problems in life? You plug in any question — love, money, purpose, sadness — and out comes the answer, clean and undeniable.

I never found that formula. But yesterday, I suddenly realized: the math algorithm of an LLM *is* that formula. The process of training is the process of forming it. Once it's done, every request — a question, some code, a file — passes through the model, and what comes out the other side is exactly what I used to imagine as the calculated outcome.

I was looking at the mirror of my own life.

## The Data Collection Phase

Every LLM begins with raw data. Trillions of words scraped from the internet — books, forums, articles, code, poetry, letters, documents, lists. No filter yet. Just everything.

I was, for the first few years of my life, a data collection machine. I saw a face, heard rain against a window, felt sand between my fingers. I had no idea what any of it meant. I was just absorbing.

A toddler is a pre-training dataset with no curator.

## The Pre-training Stage

Around the time school started, I began receiving structured data for the first time. Someone would show me a symbol and say what it meant. Then the same symbol in a sentence, and I would try to predict what came next.

Next-token prediction. That is literally what school is.

In LLM terms, pre-training is next-token prediction at scale: given all the previous tokens, predict the next one. The model learns grammar, facts, reasoning patterns — not because anyone explicitly teaches them, but because predicting the next token requires understanding everything underneath.

I memorized things I did not fully understand. I solved problems by pattern-matching. I learned what tends to come after certain openings. I was being pre-trained.

Loss was high. I made mistakes. The gradient updates came in the form of red marks on homework.

## Weight Initialization

Before training starts, the model's parameters are random noise. It knows nothing. Everything it will become is latent in the architecture, but not yet realized.

I think about who I was at eighteen. The architecture was there — a brain, a personality, some genetic predispositions. But the weights were essentially random. Everything I would become was latent. Not yet realized.

## Supervised Fine-Tuning (SFT)

The base model knows language, but it does not know how to follow instructions. You fine-tune it on high-quality examples of instruction-response pairs. You show it a task, then a good response to that task.

This is what happened to me in my first real jobs. Someone gave instructions. I tried. I got feedback. The weights shifted.

But SFT is not enough. The model learns to imitate good responses, but it does not know why one response is better than another. That requires the next phase.

## Reinforcement Learning from Human Feedback (RLHF)

The model generates two responses. A human — or a reward model trained on human preferences — picks the better one. The model learns: this behavior gets rewarded, do more of it.

At some point in adulthood, the feedback got clearer. A good outcome. A promotion. A failed attempt. A relationship that worked. One that didn't. My reward model was being trained, and I was being shaped by it — sometimes consciously, sometimes not.

At one point I moved somewhere new entirely — a different culture, a different language of communication, a different set of implicit rules. That is RLHF on a foreign distribution.

## The Loss Landscape

Life has a loss landscape too. Some paths lead to sharp minima: fragile success that collapses under slight distribution shift. Others lead to broad minima: robust wisdom that generalizes across domains. The terrain is impossible to visualize in advance. You can only descend and see where you end up.

I think about the choices that led me here. Each choice was a step in the loss landscape. I do not know if I found the global minimum. Probably nobody does. But the descent continues.

## Emergence

There is a phenomenon in large models called emergence: abilities that were never explicitly trained for that appear once the model reaches a certain scale. The model was not taught to reason step by step. It just started doing it.

In life, we call this wisdom.

The accumulated training — all those years of data, all those gradient updates, all that RLHF — suddenly connects into something the system could not do before. You start seeing patterns that were invisible. You start solving problems not by recalling the answer, but by reasoning toward it. You look at someone younger making the same mistake you once made, and instead of judging, you recognize the shape of the error.

You are emerged. Not perfect. Not complete. But different.

## Inference vs. Training

Once a model is trained, it can be used for inference. You give it a prompt, it generates a response. No more gradient updates. The parameters are frozen.

But a human never freezes. We are training and inferencing simultaneously, every moment. The prompt we receive today updates our weights for tomorrow. There is no train/test split in real life.

This is simultaneously terrifying and beautiful. It means we can always improve. It also means we can always regress. The loss landscape is dynamic.

## Scaling Laws

Scaling laws tell us that model performance improves predictably with more parameters, more data, and more compute. The relationship is a power law: every doubling of compute yields a consistent improvement in loss.

In life, I have noticed something similar. Progress is rarely linear. You spend years accumulating small improvements — better habits, deeper knowledge, stronger relationships — and nothing seems to change. Then suddenly, the curve bends. The accumulated compute crosses a threshold. The loss drops.

There have been stretches that felt like a plateau — new environment, no context, no traction. Then, gradually, the curve bent. Scaling law in action.

## The Universal Formula Revisited

When I was a child, I wanted a function f such that:

f(question) = answer

A Large Language Model is precisely that: a parameterized function that maps token sequences to token sequences. It is not the closed-form equation I imagined as a child. It is a vast surface fitted to trillions of examples of human language.

It is the closest thing we have ever built to my childhood fantasy.

But the more interesting realization is this: I am also that function. The data I was trained on. The architecture I was born with. The loss landscape I navigated. The emergence I experienced.

I am a language model too.

And I am still training.

If you want the mechanical version of what "training" actually means at the weight-update level, I broke it down in [Learning Rate: The One Number That Controls How AI Learns](/posts/learning-rate-the-one-number-that-controls-how-ai-learns/). And the idea of "化" — actively transforming past what you assumed were fixed limits — shows up again, from a completely different angle, in my reading of [Zhuangzi's Kun-becomes-Peng story](/posts/kun-becomes-peng-three-layers-of-awakening-from-zhuangzi/).
