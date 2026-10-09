---
title: "Train an AI to Play Breakout on a Laptop CPU"
date: 2026-10-27 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, deep-learning, games, pytorch, stable-baselines3, tutorial]
description: "A hands-on walkthrough: train a PPO agent to play Atari Breakout from raw pixels in about 30 minutes without a GPU. Every command and number here comes from a real run."
---

The [previous post](/posts/how-ai-learns-to-play-video-games/) surveyed four routes by which AI learns to play games. This one takes the first route, learning from pixels and score alone, and does it end to end on an ordinary CPU. Everything below was run on one machine (32 cores, no GPU) with the library versions listed at the end, and every number is from that run.

## Step 0: pick the tools

- **[Gymnasium](https://gymnasium.farama.org/)** defines the standard environment interface: `reset`, then repeated `step(action)` returning an observation, a reward, and end-of-episode flags.
- **[Arcade Learning Environment (ALE)](https://ale.farama.org/)** supplies the Atari games behind that interface. The `ale-py` package ships the game ROMs, so there is nothing to download separately.
- **[Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html)** supplies the algorithms in a few lines of code.
- **PPO** is the algorithm. The [original paper](https://arxiv.org/abs/1707.06347) describes it as sampling data through interaction with the environment and then optimising a "surrogate" objective with stochastic gradient ascent. The change from standard policy gradients is that it allows multiple epochs of minibatch updates on the same data. The SB3 documentation adds the key point in plain terms: "after an update, the new policy should be not too far from the old policy," which PPO enforces by clipping.

```bash
python -m venv rl && source rl/bin/activate
pip install gymnasium "stable-baselines3[extra]" ale-py
```

## Step 1: see what the agent sees

Raw Atari frames are 210×160 colour images. An agent trained on them directly would waste most of its capacity on irrelevant pixels, so a standard preprocessing stack sits between the game and the network. SB3's [`AtariWrapper`](https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html) bundles it, with these defaults:

- **Frame skip of 4:** the agent chooses an action every fourth frame, and the same action is repeated in between.
- **Grayscale and 84×84:** `WarpFrame` converts frames to grayscale and resizes them.
- **Reward clipping:** `ClipRewardEnv` clips every reward to its sign, so +1, 0 or −1. Games with very different score scales then give the optimiser similar-sized signals.
- **Episodic life:** losing a life ends the episode as far as the learner is concerned.
- **Random no-ops at reset,** up to 30, so that each episode starts slightly differently.

One more wrapper stacks the last four frames into a single observation. A single frame cannot show which way the ball is moving; four consecutive frames can. The resulting observation shape is `(84, 84, 4)`, which I confirmed by printing it.

```python
import gymnasium as gym, ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv

gym.register_envs(ale_py)

if __name__ == "__main__":
    env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=8, seed=0,
                         vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=4)
    print(env.observation_space, env.action_space)
    # Box(0, 255, (84, 84, 4), uint8) Discrete(4)
```

Eight copies of the game run in parallel in separate processes. Parallel copies give the learner more varied experience per update, and they use more of the CPU.

## Step 2: train

```python
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

torch.set_num_threads(8)

model = PPO(
    "CnnPolicy", env,
    n_steps=128, batch_size=256, n_epochs=4,
    learning_rate=lambda f: 2.5e-4 * f,   # decays to zero
    clip_range=lambda f: 0.1 * f,
    ent_coef=0.01, vf_coef=0.5,
    seed=0, device="cpu", verbose=1,
)
model.learn(
    total_timesteps=1_000_000,
    callback=CheckpointCallback(100_000 // 8, "ckpt", name_prefix="ppo"),
)
model.save("ppo_breakout_1m")
```

`CnnPolicy` is the convolutional network that turns four stacked frames into action probabilities and a value estimate. The hyperparameters follow the commonly used Atari PPO settings; I did not tune them. The checkpoint callback saves a snapshot every 100,000 steps, which is how the comparison below was made.

On this machine the run took 1,824 seconds, about 30 minutes, at roughly 550 steps per second, as reported by the training log. Without a GPU the practical cost is time, not feasibility. In short speed tests of 16,384 steps, the default single-process setup managed about 130 steps per second, while `SubprocVecEnv` with 8 environments reached 158 to 257 depending on the number of PyTorch threads, so it pays to use it.

![Rolling average training score on Breakout over one million steps](/assets/pic/breakout-training-curve.png)

The curve is the rolling mean score during training, reported by SB3. It rises from near 0 to 17.3 over the million steps and has not flattened, so more steps would very likely have helped.

## Step 3: evaluate honestly

Training scores are not a fair test, because the policy is still exploring and rewards are clipped. I loaded saved checkpoints and played 10 complete games each, with the unclipped game score and a fresh seed.

One trap cost me time. `EpisodicLife` makes the environment report "done" when a life is lost, so a naive loop that stops at the first `done` measures a single life, not a game. The first evaluation I ran did exactly this and showed many scores of 0. The fix is to wait for the `episode` entry in the `info` dictionary, which the `Monitor` wrapper adds only when a full game ends:

```python
def play(model, n=10):
    scores = []
    env = VecFrameStack(make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=100), 4)
    for _ in range(n):
        obs, score = env.reset(), None
        while score is None:
            action = model.predict(obs, deterministic=False)[0]
            obs, _, _, info = env.step(action)
            if "episode" in info[0]:
                score = info[0]["episode"]["r"]
        scores.append(score)
    return scores
```

Results, 10 full games each:

| Agent | Mean score | Best game |
| --- | --- | --- |
| Random actions | 2.0 | 5 |
| After 100,000 steps | 3.9 | 9 |
| After 500,000 steps | 13.6 | 19 |
| After 1,000,000 steps | 21.0 | 28 |

Three observations from the numbers:

- Random play scores about 2, because paddle and ball occasionally meet by accident.
- Early learning is slow. At 100,000 steps the agent has barely beaten random.
- The agent is clearly learning but is not good. A score of 21 means it returns the ball several times and clears some bricks, not that it has mastered the game. DeepMind's [Nature paper](https://www.nature.com/articles/nature14236) trained far longer on far more computation to reach human-level scores across 49 games.

## A cheaper game to start with

If 30 minutes is too long for a first try, [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) trains faster, because its observation is eight numbers rather than pixels. With PPO and `MlpPolicy`, 16 parallel environments and one million steps, training took 936 seconds. Over 20 test episodes the agent scored an average of 248.1 (standard deviation 46.8), against −172.8 for random actions. The environment is considered solved at 200.

It needs the Box2D physics library, which on my Python 3.14 environment did not install with a plain `pip install "gymnasium[box2d]"`. Three things were needed: install `swig`, put the environment's `bin` directory on `PATH`, and build with `pip install --no-build-isolation box2d-py` while pointing `CC` and `CXX` at `gcc` and `g++`. On Python versions with prebuilt wheels the plain install may just work.

## What can go wrong

- **Training score is not game score.** The rolling mean in the logs uses clipped rewards, and counts a life as an episode. Evaluate with full games, as in Step 3.
- **Single runs are noisy.** Every number here comes from one seed. Reinforcement learning results vary a lot between seeds, so a comparison between two settings needs several runs each before it means anything.
- **SubprocVecEnv needs a `__main__` guard.** The worker processes are started as new Python processes that re-import the script, so the training code must sit under `if __name__ == "__main__":`, which is what the code above does.
- **Check the hardware before the plan.** On a CPU, measure steps per second first with a short run, then choose the step budget.

## Where to go next

- Train longer. The curve was still rising at one million steps.
- Try [DQN](https://arxiv.org/abs/1312.5602), the algorithm of the original Atari paper, by swapping `PPO` for `DQN` in the code.
- Move to [Route 4](/posts/how-ai-learns-to-play-video-games/) and let a language model write the player.

## Versions used

gymnasium 1.4.0, stable-baselines3 2.9.0, ale-py 0.12.1, torch 2.14.1 (CPU), Python 3.14. The SB3 documentation linked above describes the same wrappers and parameters for the current release.
