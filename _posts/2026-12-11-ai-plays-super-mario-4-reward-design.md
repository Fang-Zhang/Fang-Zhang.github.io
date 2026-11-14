---
title: "AI Plays Super Mario, Part 4: The Reward Decides What the AI Learns"
date: 2026-12-11 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, games, super-mario, reward-design, ppo, tutorial, series]
description: "Part 4 of the AI Plays Super Mario series. The same level, the same algorithm and the same budget, trained three times with three different rewards. Progress reward clears the level, game score clears it less often, and clear-only reward learns nothing."
---

[Part 3](/posts/ai-plays-super-mario-3-the-environment/) built the environment, including a reward that pays for moving right. This post changes only that reward and trains the same AI three times: once paid for progress, once paid for the game's own score, and once paid only for clearing the level. It reports what each version learned, with videos, and says which of the results can be trusted and which cannot. The training algorithm is PPO, which Part 5 explains. Here it is a fixed tool.

## The setup

Everything except the reward is identical across the three runs: the environment from Part 3, the PPO implementation from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) with the same settings, 16 parallel copies of the game, and 3,000,000 decisions of training. Each run took 1.5 to 3.5 hours on a 32-core CPU with no GPU. Each reward was trained once, so every number below comes from a single training run. Differences of a few episodes are not evidence of anything. Large gaps are.

The three rewards, as code:

```python
def reward(mode, x, last_x, score_gain, died, cleared):
    if mode == "progress":                    # the Part 3 reward
        r = min(max((x - last_x) * 0.1, -1), 3) - 0.01
        if died:
            r -= 5
        if cleared:
            r += 10
    elif mode == "score":                     # the game's own score, nothing else
        r = min(max(score_gain * 0.01, 0), 5)
    elif mode == "sparse":                    # nothing until the level is cleared
        r = 10 if cleared else 0
    return r
```

`score_gain` is how much the score on the game's top bar rose during the last decision. Stomping an enemy, breaking a block and collecting a coin all raise it.

## Evaluation

After training, each model played 20 full episodes of level 1 and the post counts how many reached the flag. The model outputs a probability for each of the seven actions, and an evaluation can either sample from those probabilities or always pick the most likely one. The numbers below sample, which is how the model was trained. Always picking the most likely action behaves differently, and for the progress reward it failed on the end-of-level staircase; Part 7 covers that. Here the question is which reward leads to a model that can clear the level at all.

## Result 1: progress reward clears the level

The progress reward is the one from Part 3, and the model trained with it is the one in the video from Part 1.

- In the last 50 training episodes, 49 cleared the level.
- In evaluation, 20 of 20 episodes cleared it.
- In 30 further episodes, 29 cleared it, taking a median of 523 decisions (range 507 to 591), roughly 35 seconds of game time.

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-level-1-clear.mp4' | relative_url }}" type="video/mp4">
</video>

*The model trained with the progress reward clearing level 1. The bar above the game shows the decision number, Mario's position, the furthest position so far, and which keys the AI is pressing (green).*

## Result 2: the game's own score also works, less often

The game's score is what a human sees as "how well am I doing", so it is a natural reward to try. A reward that pays for stomping enemies and breaking blocks could, in principle, lead an agent to farm points and ignore the exit. The expectation going in was that the AI would do exactly this.

It did not. The AI learned to clear the level:

- In the last 50 training episodes, 30 cleared it.
- In evaluation, 10 of 20 episodes cleared it. In 30 further episodes, 16 cleared it, taking a median of 580 decisions (range 554 to 749). That is slower than the progress model's 523, by about 11 percent.

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-score-reward-clear.mp4' | relative_url }}" type="video/mp4">
</video>

*A successful episode of the model trained with the game's score. Roughly half of its episodes end earlier, in a pit or against an enemy.*

The data does not show why the score reward leads to the exit. One explanation is that in level 1 the points are spread along the route, so a model collects them while moving right. That explanation is a guess: it would need a second level or a different score layout to test, and neither was run. The takeaway is narrower: in this level, with this algorithm and this budget, the score reward worked, but less reliably than the progress reward.

## Result 3: pay only for clearing, and nothing is learned

A reward that pays only when the goal is reached is the cleanest statement of what is wanted, and it is the hardest to learn from. In the sparse run:

- The training log counts clears among the most recent 50 episodes. That count was 0 at 145 of its 146 log lines, including the last one. The exception was a single clear, around 1.8 million steps, among 4,796 training episodes in total.
- In evaluation, 0 of 20 episodes cleared it, whether sampling or always picking the most likely action.
- When always picking the most likely action, the model's average best position was 40, the start of the level. In the video, Mario stands in place.

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-sparse-reward-stands-still.mp4' | relative_url }}" type="video/mp4">
</video>

*The sparse-reward model, always picking its most likely action. No key lights up and Mario stays at the start.*

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-sparse-reward-dies.mp4' | relative_url }}" type="video/mp4">
</video>

*The same model sampling from its probabilities. It runs right and dies to the first Goomba, about 8 seconds in.*

The model does move right when sampling, because a policy that has received no signal stays close to random, and random play does move. But the reward was non-zero in at most one of the 4,796 episodes, so almost nothing told it that any action was better than any other. Clearing the level takes around 500 well-chosen decisions in a row, and chance produced one such episode in the whole run. An earlier attempt at this run stopped partway, after 1,351,680 steps, because the computer restarted. Its log showed the same picture: 0 clears in the last-50 window at every one of its logged points.

## What the three runs say about reward design

- **A reward must give feedback before the goal is reached.** Progress does; clearing alone does not. This is the usual reason people add a shaped reward, a stepwise signal on the way to the goal.
- **A reward can work without being the goal.** The game's score is not "reach the flag", and it still led to clears in this level. There is no guarantee it would in another.
- **The AI optimises what is paid, not what is meant.** OpenAI described a case of this in the boat-racing game CoastRunners, where [an agent found a lagoon in which it could keep hitting respawning targets](https://openai.com/index/faulty-reward-functions/), scoring about 20 percent more than human players without finishing the race. The authors of [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) list this under "reward hacking". Level 1 of Super Mario did not produce that behaviour with the score reward, but the possibility is why the reward needs checking against the behaviour, not only against the number.

## Try it

Part 3's `mario_env.py` has the progress reward. To run the comparison, change the reward in `step()` as in the function above and train for 3,000,000 decisions with PPO as set up in Part 7. In the progress run, the last-50 clear count was 3 of 50 at 1.5 million steps and 46 of 50 at 2.56 million. Expect hours, not minutes, on a CPU.

Part 5 opens the training algorithm that these runs shared.
