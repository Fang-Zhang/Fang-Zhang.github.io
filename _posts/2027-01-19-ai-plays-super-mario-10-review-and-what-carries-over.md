---
title: "AI Plays Super Mario, Part 10: Review, and What Carries Over to Other Games and Software"
date: 2027-01-19 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, games, generalization, stable-retro, super-mario, series]
description: "Part 10 of the AI Plays Super Mario series. What each part found, what a reader should expect from the finished pipeline, which parts of it carry over to other games and software and which have to be rebuilt, and what the series did not test."
---

This post closes the series. It does three things: it lists what each part found, with the numbers from the parts themselves; it sets out what a reader who follows the series should expect; and it separates what carries over to another game or another piece of software from what has to be defined again. The last section is a hook, not a promise: the extensions it names were not built or tested for this series, and the post says where the evidence comes from for each claim about them.

## What the series found

Each line is one result, with the part where it is shown and the evidence behind it.

- **Part 3, the environment.** Reading position from the game's memory gives a clean progress number. An agent that only presses right collected a reward of 19.50 and died at position 295 after 100 decisions, which shows that reward alone does not say whether the task is done.
- **Part 4, reward.** With the same algorithm, setting and seed, a reward for progress cleared the level in 20 of 20 evaluation episodes, a reward for score in 10 of 20, and a reward only for clearing in 0 of 20 (one clear in 4,796 training episodes). Single seed each, so these are observations of three runs and not a ranking.
- **Part 5, algorithms.** DQN and PPO were explained from the original papers, with small scripts of the author's own. No head-to-head was run, and the post says so.
- **Part 6, learning from a demonstrator.** The demonstrator was a scripted search, not a person. Behavioural cloning and two rounds of DAgger all cleared 0 of 20 episodes, and held-out accuracy stayed below the always-run baseline. This is a negative result for this setup, and the causes named in the post are untested guesses.
- **Part 7, training.** Five seeds of the same setup, 20 evaluation episodes each: sampling the agent's own action probabilities cleared 15 to 20 of 20, while always taking the most probable action cleared 20 of 20 for two seeds and 0 of 20 for the other three. For the seed traced in detail, the cause was a staircase where holding the jump button does nothing and releasing it for one step does.
- **Part 8, language-model players.** Four papers measured language-model players that act through a harness. Their scores depend on what the harness provides and on whether the game waits for the model, so they cannot be compared with each other or with the five-seed table. No language model was run on Mario.
- **Part 9, industry.** Five systems use AI players for difficulty prediction, coverage testing, a goalkeeper, a teammate and research. In all but the research agent the model is one part of a larger system, and every number is the company's or lab's own.

## What to expect from the finished pipeline

This repeats Part 7 for readers who start here. With the same code, expect an agent that clears level 1 most of the time when it samples its actions, and not every time. In five runs the final sampled clear count was 15 to 20 of 20, and training took between 1.5 and 4 hours on a CPU with no GPU. Evaluate both sampling and always-take-the-most-probable, because the same network scored 20 of 20 and 0 of 20 depending on the mode. Evaluate several checkpoints, because neighbouring ones differed from 0 to 20 of 20. Two readers with the same code can get different results.

Those results come from the training copy of the environment, and not from a rerun with the file written in Part 3 (same actions and rewards, per Part 7). The series did not run that rerun.

## What carries over

The series built an agent from five definitions. Which of them transfer depends on how far the new task is from Mario.

![A table with three columns (Mario in Parts 3 to 7, another emulated game, other software) and five rows (start state, what it sees, what it can do, reward, episode end), showing what has to be supplied for each.]({{ "/assets/pic/mario-ai-what-to-define.png" | relative_url }})
*What has to be defined for each kind of task. The Mario column is what the series built. The other two columns are the author's reading of what must be supplied and were not built or tested here.*

**The loop and the training code carry over.** The observe, act, reward loop of Part 1 is the same one every reinforcement-learning environment exposes, and the PPO settings listed in Part 7 (network type, batch size, learning rate and so on) are generic ones, with nothing in them specific to Mario beyond the environment the agent is given.

**Another emulated game needs three things defined again.** The documentation of [Stable-Retro](https://stable-retro.farama.org/integration), the library used in Part 2, describes integrating a game as defining a starting state, a reward function and a done condition, with the memory locations of the variables written in a data file. That is the work Parts 3 and 4 did for Mario by hand. The library lists many consoles as supported. Whether a given game is easy to train depends on its reward, which Part 4 showed can decide the outcome by itself.

**A different level of the same game is not the same task.** The series trained and evaluated on level 1 only. The research on generalisation points in one direction: [Quantifying Generalization in Reinforcement Learning](https://arxiv.org/abs/1812.02341) reports that agents overfit to surprisingly large sets of training levels, and the [Procgen benchmark](https://arxiv.org/abs/1912.01588) argues that diverse training environments are needed to train and evaluate generalisation. The [Sonic the Hedgehog benchmark](https://arxiv.org/abs/1804.03720) was built to measure transfer between levels of a retro game. None of these papers is about Mario, and the series did not test this agent on any other level, so what a level-1 agent does on level 2 is unknown here.

**Other software shares the idea and little else.** An agent can in principle be given the screen of any program and a set of clicks and key presses. The benchmark [WebArena](https://arxiv.org/abs/2307.13854) evaluates language-model agents on tasks in realistic websites and checks whether the task was completed; in its paper the best GPT-4-based agent completed 14.41% of the tasks, against 78.24% for humans. That is a different design from the pixel-and-PPO agent of this series, and the figure is from an earlier model generation, so it shows how such tasks are evaluated, not what current agents do. The structural difference is the one in the figure: a game provides a score and a game-over signal in its memory, and most software provides neither, so the reward and the end of the episode have to be written for each task.

## What this series did not test

- A single environment file: the training copy and the Part 3 file are described as equivalent, but no rerun was made with the latter.
- The cause of the staircase stall: one model, one state, and a sequence that works. The rule that explains it is consistent with the result and was not tested separately.
- Parts 4 and 6 each rest on one seed.
- A person as the demonstrator in Part 6.
- Any language model playing Mario, and any comparison between a language-model player and the trained agent.
- Any level other than level 1, any other game, and any software other than the emulator.

## Where to go from here

Each item is a question, not a result, and each can be answered with the code of the series:

1. Does the Part 7 agent clear anything on level 2 without further training, and how many steps does it take to train there?
2. Does training on several levels at once improve evaluation on a level held out of training?
3. Does imitation work when the demonstrator is a person and not a search?
4. How does a language-model player, driven by the harness of Part 8, compare with five PPO seeds on level 1 when both use the same scoring?

The scripts of Parts 2 to 8 are in the [code repository](https://github.com/Fang-Zhang/ai-plays-super-mario), without a ROM. The sources linked in Parts 3 to 9 are the places to continue. The series ends here.
