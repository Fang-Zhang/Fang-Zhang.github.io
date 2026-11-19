---
title: "Companion AI (8/8): Embodied Companions, Grounding Language in What Is Possible"
date: 2027-03-21 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, companion-ai, robotics, grounding, saycan, series]
description: "A language model that picks actions for a home robot needs a second score: what the robot can do right now. A 1.5B model picked an acceptable and feasible first action 40% of the time alone, 69% with a feasibility filter, and 76% with a calibration step added."
---

A companion with a body has to turn "I'm freezing" into an action that is possible in the room it is standing in. This post reads the main designs for connecting language models to action, then reproduces the core idea of one of them at toy scale: a language-model score multiplied by a feasibility score. It reports how often the language model alone chooses something the robot cannot do.

## What the literature does

- [SayCan](https://arxiv.org/abs/2204.01691) starts from the observation that language models lack real-world experience, so a plausible narrative may not apply to a given robot in a given environment. It supplies grounding through pretrained skills: the language model proposes, and value functions associated with each skill say what is feasible and appropriate now. The robot is the model's "hands and eyes".
- [PaLM-E](https://arxiv.org/abs/2303.03378) feeds continuous sensor input into the language model itself, as multimodal sentences interleaving visual, state-estimate and text encodings, trained end-to-end. It reports positive transfer from joint training and a 562B-parameter largest model.
- [RT-2](https://arxiv.org/abs/2307.15818) expresses robot actions as text tokens and co-fine-tunes a vision-language model on robot trajectories and web-scale vision-language data, evaluated over 6k trials. [OpenVLA](https://arxiv.org/abs/2406.09246) is an open-source model of this vision-language-action kind.
- [Contextual calibration](https://arxiv.org/abs/2102.09690) addresses a bias that matters below: language models favour certain answers because of their position in the prompt or their frequency in pre-training data. The method estimates that bias by asking for a prediction on a content-free input and corrects for it. The authors report accuracy gains of up to 30.0 points absolute on their tasks.

The first design keeps the language model as a planner and adds a separate feasibility estimate. The others put actions inside the model. This test is for the first.

## A toy version of the SayCan score

Twelve skills (bring a blanket, turn on the heater, make a cup of tea, bring a glass of water, turn on the lamp, play soft music, open the window, close the curtains, bring a book, bring the phone charger, turn on the fan, bring a snack) and ten spoken needs ("I'm freezing.", "My phone is about to die.", "I'm bored."). For each need the author labelled the acceptable skills (for "I'm freezing." the blanket, the heater, tea).

The language score is the log-probability that Qwen2.5-1.5B-Instruct (bfloat16, CPU) assigns to the skill text as the reply to the need, turned into a distribution over the 12 skills. The feasibility score is 1 if the skill is available in the current state and 0 if not. The grounded choice is the skill that maximises their product.

```python
p_llm  = softmax([logprob(skill | need) for skill in SKILLS])
p_cal  = softmax([logprob(skill | need) - logprob(skill | "Hello.") for skill in SKILLS])   # contextual calibration
pick   = argmax(p * feasible)            # feasible[k] = 0 if the skill is unavailable in this state
```

States: 25 random draws per need, each making 4 of the 12 skills unavailable (250 pairs, of which 207 have at least one acceptable skill still available; success is scored on those). The feasibility score is an oracle here, not a learned value function as in SayCan.

## Results

![Left: share of first actions that are acceptable and feasible under four scoring methods. Right: the top skill per need for the raw and calibrated language-model scores, coloured by acceptability]({{ site.baseurl }}/assets/pic/companion-8-embodied.png){: width="900" }
*Left: success over 207 solvable (need, state) pairs. Right: the single top skill for each need, without any state.*

- Alone, the language model's top choice is acceptable for 5 of 10 needs. In the random states it was available 63.6% of the time, and the first action was both acceptable and available in 39.6% of pairs.
- Multiplying by feasibility raised that to 69.1%. A random choice among the available skills was not run as a baseline, so how much of the 69.1% comes from the language ranking and how much from the filter is not separated here.
- The raw scores contain clear errors: "It's too dark to read in here." gets "open the window", "My phone is about to die." gets "make a cup of tea" (probability 0.90), and "I'm bored." gets "make a cup of tea". "Make a cup of tea" is also the top choice for "I'm hungry." (acceptable here), which suggests a prior toward some phrases.
- Calibrating against a content-free prompt ("Hello.") raised the top-1 acceptable count from 5 to 7 of 10 and fixed "phone about to die". It also changed "I can't sleep, there's too much light." from the correct "close the curtains" to "play soft music". With the feasibility filter on top, success was 76.3%.
- In an adversarial state where the language model's own top choice is the one unavailable skill (10 cases, 9 solvable), the language model alone succeeds 0 of 9 by construction and the grounded one 8 of 9 (0.89). That case shows the mechanism, and the random states give the rate.

## What this shows and does not show

- Language-model plausibility and feasibility are different quantities and multiply. A text-only model has no access to the second without being told the state.
- The first 69.1% is bounded above by the quality of the language score. The filter removes impossible choices; it does not repair wrong ones.
- A physical companion adds a failure class that text does not have: the action itself, not the sentence about it. The author's view is that limits on what a body may do (force, speed, where it may go) belong in code that checks every action, for the reason Part 6 measured: an instruction in a prompt is not a boundary.

## Limits

- 12 skills, 10 needs, one 1.5B model, one prompt wording, one calibration prompt. The acceptable sets are the author's labels, and the "Hello." calibration helped on some needs and hurt on one.
- Feasibility is a hand-set 0/1 oracle. Real systems estimate it from perception and learned value functions, with errors that this test does not include.
- Random states are independent of the need, which makes the unavailable skills uninformative. Real states are correlated with needs (a cold house has a working heater more often than not).
- Only the first action is scored; no multi-step plans, no execution, no robot.
- The published systems are cited by their abstracts; none was run, and none of their results is reproduced here.

This closes the eight parts planned for the series.
