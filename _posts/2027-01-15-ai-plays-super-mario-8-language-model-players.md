---
title: "AI Plays Super Mario, Part 8: Language-Model Players, How They Work and How They Are Evaluated"
date: 2027-01-15 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, vision-language-models, games, benchmarks, super-mario, series]
description: "Part 8 of the AI Plays Super Mario series. How a language model plays a video game through a harness, what four research papers measured, how to read their numbers, and where the approach sits next to the PPO agent built in Parts 5 to 7."
---

This post covers a different kind of player: a general-purpose language model that was never trained on Mario and is asked to play it through a screenshot and a prompt. It describes the loop around the model (the harness), summarises four papers that measured such players (VideoGameBench, lmgame-Bench, RL4VLM and BALROG), and gives a short list of questions for reading any benchmark of this kind. Unlike Parts 3 to 7 there is no new training here: no language model was run on Mario for this post, so every number about language-model players is quoted from the papers, with the version that was read, and every number about our own agent is from the earlier posts. Where the post does arithmetic of its own, it says so.

## The loop, and what a harness is

In the series so far the player was a small convolutional network. It received four gray frames, returned one of seven button combinations, and the game waited for it ([Part 3](/posts/ai-plays-super-mario-3-the-environment/)). A language model takes text and images and returns text, so something has to translate in both directions. That translating layer is usually called the harness (the papers also say scaffold or agent framework).

![Two rows of boxes. Top: game, 4 stacked gray frames, neural network trained by PPO, 7 button combos. Bottom: game, harness with screenshot, notes and prompt, language model, harness that parses the reply into key presses, keys held for a time]({{ '/assets/pic/mario-ai-llm-vs-ppo-loop.png' | relative_url }})
*The loop from the series (top) next to the loop for a language-model player (bottom). Own drawing. The harness (green) is code that someone has to write for each game.*

What the harness contains differs between papers, and that difference turns out to matter more than it looks:

- **Observation:** the raw screenshot, or text that the harness extracts from the game.
- **Memory:** notes the model writes for itself, passed back in the next prompt.
- **Action format:** free text such as "press_key space", parsed into key presses.
- **Timing:** whether the game keeps running while the model thinks.

## What four papers measured

**VideoGameBench** ([Zhang et al.](https://arxiv.org/abs/2505.18134), read in its version 3 of May 2026) gives vision-language models only the raw screen of 23 games on emulators, and forbids overlays, hints and game-state information such as the emulator's memory. The agent outputs actions in natural language, for example "press_key space", which is held for a default of one tenth of a second. The authors' agent follows the ReAct pattern with notes kept in context. They report that all models struggle to progress: the best, Gemini 2.5 Pro and Claude Sonnet 3.7, reach 0.48% on the benchmark's score, which is the average amount of each game completed, and each model was run once per game because of budget. They name two failures: wrong reading of the screen (an agent in Doom II spending all its ammunition on an enemy it believed alive but which was dead) and weak planning and memory. They also name latency. A model takes so long to answer that the game has moved on, so the action it returns is stale. Their variant VideoGameBench Lite pauses the emulator while the model thinks, and every model does considerably better there. The paper contrasts this with a Gemini run that completed Pokemon Blue using tailored tools for pathfinding, game-specific hints and memory, which is the kind of help VideoGameBench disallows.

**lmgame-Bench** ([Hu et al.](https://arxiv.org/abs/2505.15146), version 2 of June 2025) starts from the opposite observation: dropping a model straight into a game gives unreliable evaluations because of brittle vision, sensitivity to the prompt wording and possible contamination from game material in the training data. The suite has platformer, puzzle and narrative games, including Super Mario Bros, scored by the distance Mario travels along the x axis, the same idea as the progress reward in [Part 4](/posts/ai-plays-super-mario-4-reward-design/). It evaluates 13 models on 6 games with a unified Gym-style interface and adds lightweight harness modules:

- a perception module, which for grid games turns the picture into a text table such as "Box at (2,3)";
- a memory module;
- a standardised prompt, tuned to reduce the difference between two candidate prompts (reported reduction of 33.8% to 63.5% on three models).

With the harness, 86.7% of game runs beat a random-play baseline, and for five of the games the paper reports significantly higher scores than without it. Its contamination check found that models' scores on Ace Attorney followed how closely their text resembled the game's public script, until the authors changed the prompts; for Super Mario Bros level 1-1 the models could reconstruct only the first and last frames of a shuffled sequence, and the paper concludes that vision-level contamination is negligible. The last result concerns training and is covered below.

**RL4VLM** ([Zhai et al.](https://arxiv.org/abs/2405.10292)) is the one of the four that changes the model. The prompt asks a 7-billion-parameter vision-language model (LLaVA-1.6 with a Mistral base) to describe the state, reason, and give a text action. The environment parses the action and returns a reward, and PPO, the algorithm from [Part 5](/posts/ai-plays-super-mario-5-dqn-and-ppo/), updates the whole model with it. On their card-game tasks (NumberLine, EZPoints, Points24, Blackjack) and on ALFWorld, a household-task environment, they report that the trained 7B model beats GPT-4V and Gemini, and that removing the written reasoning from the output makes the result significantly worse. There is no Mario in it: the paper shows that the training loop from this series works with a language model as the policy, on those tasks.

**BALROG** ([Paglieri et al.](https://arxiv.org/abs/2411.13543), ICLR 2025) uses six procedurally generated environments: BabyAI, Crafter, TextWorld, Baba Is AI, MiniHack and NetHack. Because each episode is a new instance, memorising one level does not help. Scores are the percentage of progress, averaged over environments. In the language-only format Claude 3.5 Sonnet leads with 32.64 ± 1.93 and GPT-4o follows with 32.34 ± 1.49; with images added GPT-4o falls to 22.56 ± 1.44 while Claude 3.5 Sonnet rises to 35.48 ± 2.02. Every model stays near zero on NetHack, where the best, o1-preview, reaches 1.5%. The authors note a model that correctly says eating rotten food is dangerous and still dies from it, which is a gap between stating knowledge and using it.

## Training a language model on a game

lmgame-Bench also reports an experiment in the direction of RL4VLM. It trains Qwen2.5-7B-Instruct with reinforcement learning on one game, 6×6 Sokoban or a two-piece Tetris, using the RAGEN framework's StarPO-S method. Training on Sokoban improved results on larger Sokoban boards and on the planning task Blocksworld, and the game-trained models improved by at least 6% on the shopping task WebShop. Neither game transferred to the maths task GSM8K or the database task BIRD. So what carries over from the game is a planning and search skill, and not a general one.

## Why latency matters: arithmetic

The series' environment decides once every four frames of a 60-frame-per-second game, which is 15 decisions per second. A level-1 clear took about 520 decisions in the runs of this series. The script below is the whole calculation:

```python
FPS = 60          # NES frames per second
SKIP = 4          # frames per decision in the series' environment
DECISIONS_PER_S = FPS / SKIP
CLEAR_DECISIONS = 520   # about one level-1 clear in the series' runs
TRAIN_STEPS = 3_000_000 # one full PPO run
print(f"real-time game: {DECISIONS_PER_S:.0f} decisions per second")
for latency in (0.5, 1, 2, 5):
    missed = DECISIONS_PER_S * latency
    clear_min = CLEAR_DECISIONS * latency / 60
    train_days = TRAIN_STEPS * latency / 86400
    print(f"reply time {latency:>3} s: {missed:5.1f} game decisions pass per reply | "
          f"one clear = {clear_min:5.1f} min of replies | 3M training steps one after another = {train_days:6.1f} days")
print(f"measured in seed 0: 3,000,000 steps in 5,484 s = {3_000_000/5484:.0f} decisions per second (training included)")
```

Output:

```
real-time game: 15 decisions per second
reply time 0.5 s:   7.5 game decisions pass per reply | one clear =   4.3 min of replies | 3M training steps one after another =   17.4 days
reply time   1 s:  15.0 game decisions pass per reply | one clear =   8.7 min of replies | 3M training steps one after another =   34.7 days
reply time   2 s:  30.0 game decisions pass per reply | one clear =  17.3 min of replies | 3M training steps one after another =   69.4 days
reply time   5 s:  75.0 game decisions pass per reply | one clear =  43.3 min of replies | 3M training steps one after another =  173.6 days
measured in seed 0: 3,000,000 steps in 5,484 s = 547 decisions per second (training included)
```

The reply times are illustrative values, not measurements of any model, and the arithmetic assumes one model call per decision. Harnesses can return several key presses per call, which VideoGameBench's action format allows, and training runs can be parallelised. What the numbers show is the order of magnitude: a model that answers in seconds cannot play a real-time game decision by decision, and 3 million model calls is not a training budget for the loop in this series. This is why Lite-style pausing exists, which our environment does already: in Gym the game only advances when `step` is called.

## How to read a benchmark of language-model players

These are my own questions, drawn from what the four papers disagree on:

1. **What did the harness provide?** VideoGameBench forbids text extracted from game state; lmgame-Bench's perception module supplies it. The same model can score near zero in one and above random in the other, so the numbers are not comparable across the two.
2. **Does the game wait?** Real-time and paused results measure different things.
3. **How many runs?** VideoGameBench ran each model once per game; BALROG reports an uncertainty band; this series reports five seeds for its own agent and says so.
4. **Could the model have seen the game?** lmgame-Bench measured this for two of its games and found it mattered for one.
5. **What is a point?** Progress along the level, completed fraction of the game and percentage of achievements are not the same quantity.

## Where this sits next to the PPO agent

The two approaches trade costs differently, as the papers and the earlier parts show:

- **The trained agent** needs a reward written for the task ([Part 4](/posts/ai-plays-super-mario-4-reward-design/)), hours of compute, and gives a player that acts 15 times a second. Five seeds of the same setup cleared level 1 between 15 and 20 times in 20 when sampling ([Part 7](/posts/ai-plays-super-mario-7-training-the-level-1-agent/)). It knows nothing outside this level.
- **The language-model player** needs no training and no reward, but needs a harness for each game, and in the papers above it makes very little progress in real time. It can read instructions, and a reward or task is described in words.
- **RL on a language model** combines the two and gets the training loop of this series with a much larger network and a much higher cost per step; the evidence in the papers read here is for small tasks and 7-billion-parameter models, not for Mario.

No head-to-head comparison on Mario was run, and none of the papers' numbers can be placed next to the five-seed table of Part 7, because the games, harnesses, models and scores differ.

Part 9 turns from research to practice: where game companies actually use AI players and non-player characters today.
