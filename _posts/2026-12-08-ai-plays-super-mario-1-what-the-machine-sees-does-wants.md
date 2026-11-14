---
title: "AI Plays Super Mario, Part 1: What the Machine Sees, Does and Wants"
date: 2026-12-08 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, games, super-mario, stable-retro, series, tutorial]
description: "Part 1 of a 10-post series that ends with a working AI playing Super Mario Bros level 1 on an ordinary CPU. This post defines the three things every game AI needs: what it sees, what it can do, and what it is rewarded for."
---

This is the first post of a ten-part series. By the end, a reader who follows the posts in order will have built and run an AI that plays level 1 of Super Mario Bros on their own computer, with no GPU. This post does not install anything. It defines the problem in the three terms every game-playing AI needs: what it sees, what it can do, and what counts as winning. Every later post is a more careful answer to one of those three.

## The target

The clip below is the goal of the series: a trained agent clearing World 1-1. The bar above the game shows the decision step, Mario's horizontal position, the furthest position reached so far, and which buttons the AI is pressing at that moment.

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-level-1-clear.mp4' | relative_url }}" type="video/mp4">
</video>

*A trained agent clears level 1 in 517 decisions, about 35 seconds of game time. Each decision holds one button combination for 4 video frames. The green boxes show the buttons held.*

Nobody wrote rules for this agent: no "jump when a pipe is near", no "stomp the Goomba". It learned from three ingredients, described next.

## The loop

Every environment built for reinforcement learning, including the ones used here, runs the same loop. The [Gymnasium documentation](https://gymnasium.farama.org/introduction/basic_usage/) describes it with four functions: `make()` creates the environment, `reset()` starts an episode, `step(action)` advances the game, and `render()` draws it. The call to `step` returns an observation, a reward, two flags that say whether the episode ended (`terminated`, `truncated`), and an info dictionary.

In plain terms, the AI looks at the screen, presses buttons for a moment, receives a number that says how well that went, and repeats. Training means adjusting the AI so that action sequences with higher total reward become more likely. The same loop trained the Atari agents in [the 2015 DQN paper](https://www.nature.com/articles/nature14236). We covered the broader landscape in [How AI Learns to Play Video Games](/posts/how-ai-learns-to-play-video-games/); this series stays with one game and goes deep.

## What the AI sees

The agent does not receive "a pipe at position 40" or "an enemy to the right". It receives pixels. The emulator produces a 240 × 224 colour frame, and the pipeline in this series shrinks it to 84 × 84 grayscale, because a smaller input trains much faster on a CPU.

![The same Mario frame in three forms: the original colour image, a grayscale version, and the 84 by 84 version the network receives]({{ "/assets/pic/mario-ai-input.png" | relative_url }})

*One frame from level 1 at three stages. The right panel is what the neural network receives. Coins, score and timer are barely legible, but Mario, the ground and the question block survive.*

A single frame cannot show which way Mario is moving, so the agent is given the last four frames stacked together. Motion becomes a difference between channels. Everything else it needs to know, such as what a pit or a pipe is, it has to infer from patterns in these numbers.

## What the AI can do

The environment exposes nine binary inputs: B, an empty slot, SELECT, START, UP, DOWN, LEFT, RIGHT and A. Almost all of the 512 possible combinations are useless in level 1. The series narrows them to seven choices:

- no button
- right
- right + A (walk and jump)
- right + B (run)
- right + A + B (run and jump)
- A (jump in place)
- left

Fewer choices mean less to learn. The cost is that anything not on the list, such as ducking, can never happen.

## What the AI wants

The game has a score, but score is not the same as progress: points come from coins and enemies, while clearing the level requires reaching the flag. The reward is therefore designed, not taken from the game. The version used here has three parts:

- a small positive reward for every unit of progress to the right
- a penalty for dying
- a bonus for clearing the level

The progress number comes from reading the game's memory, not from analysing the picture, which is the subject of Part 3. The reward looks trivial, and it is one of the places where a game AI most often goes wrong. Part 4 changes it and compares the results.

The difficulty is that reward arrives late. A jump pressed now pays off a second later, when Mario lands on the other side of a gap. Working out which earlier button press deserves credit is the central problem of reinforcement learning, and Parts 5 and 7 deal with it directly.

## What the series builds

1. What the machine sees, does and wants (this post)
2. Setup: Mario moving on screen from a few lines of Python
3. The environment: emulator, memory and the data the game exposes
4. Reward design: the same level with different rewards
5. Algorithms: DQN and PPO compared
6. Learning from a demonstrator: cloning and DAgger
7. Training the level 1 agent
8. Language-model players: how they work and how they are evaluated
9. What game companies use AI players for
10. Review, and what carries over to other games and software

Every post carries runnable code that builds on the previous one, and ends with a checkpoint: the output to expect, and what to check if the numbers differ. Hardware and time figures in the series come from real runs on a 32-core CPU with no GPU. Training the level 1 agent took about 1.5 hours there.

One thing the series keeps in the open is what went wrong. The first trained agent cleared level 1 most of the time when it picked buttons at random according to its own preferences, and never cleared it when it always picked its single most likely button. That failure turned out to have a precise cause, and tracing it is a better lesson than the success.

## What you need

- A computer with Python and a multi-core CPU. No GPU is required.
- Your own legal copy of the Super Mario Bros ROM. The series does not provide ROMs and does not link to any, and the [code repository](https://github.com/Fang-Zhang/ai-plays-super-mario) will not contain one.
- Patience for a training run of one to two hours in Part 7.

Part 2 sets up the tools and ends with Mario walking right on your screen.
