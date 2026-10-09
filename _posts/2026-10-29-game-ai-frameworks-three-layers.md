---
title: "Beyond Gymnasium: The Three Layers of Game-Playing AI Frameworks"
date: 2026-10-29 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, games, gymnasium, pettingzoo, cleanrl, frameworks]
description: "Game-playing AI tooling splits into environments, algorithms and agents. A map of the main projects in each layer, with Gymnasium, PettingZoo and CleanRL actually run as examples."
---

[Gymnasium](https://gymnasium.farama.org/) is the name that comes up first for game-playing AI, and the [previous post](/posts/train-an-ai-to-play-breakout/) used it. But Gymnasium is only one layer of a three-layer stack, and most confusion about "which framework" comes from mixing the layers up.

- **Environment layer:** the game itself, wrapped in a standard interface (`reset`, `step`).
- **Algorithm layer:** the learning code that turns experience into a policy (PPO, DQN and friends).
- **Agent layer:** systems built around a model that plans, remembers and acts, most visibly LLM-based players.

Star counts and update dates below were read from each GitHub repository on 2026-10-10. Only the projects marked "run here" (plus Stable-Baselines3 and the Arcade Learning Environment, used in the previous posts) were installed and executed; the rest are described from their own READMEs and repository descriptions.

## Layer 1: environments

An environment answers one question: given an action, what does the world do next? Gymnasium defines the single-agent contract, and the rest of the layer is mostly games that implement it.

- **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)** (12.6k stars): the standard API for single-agent RL plus reference environments such as CartPole and LunarLander. Run here.
- **[PettingZoo](https://github.com/Farama-Foundation/PettingZoo)** (3.5k stars): "a multi-agent version of Gymnasium," per its README. Run here.
- **[Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)**: Atari 2600 games, used in the previous post.
- **[ViZDoom](https://github.com/Farama-Foundation/ViZDoom)** (2.1k): play Doom from first-person pixels.
- **[Minigrid](https://github.com/Farama-Foundation/Minigrid)** (2.5k): small grid worlds that train in minutes.
- **[Procgen](https://github.com/openai/procgen)** (1.2k): procedurally generated games for measuring generalisation. Last pushed 2026-03.
- **[Craftax](https://github.com/MichaelTMatthews/Craftax)** (460): Crafter plus NetHack rewritten in JAX so that the whole game runs on an accelerator.
- **[OpenSpiel](https://github.com/google-deepmind/open_spiel)** (5.5k): DeepMind's collection of board, card and matrix games, with algorithms included.
- **[RLCard](https://github.com/datamllab/rlcard)** (3.6k): poker, Dou Dizhu, Mahjong, UNO. No pushes since 2024-06.
- **[Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)** (19.7k) and **[Godot RL Agents](https://github.com/edbeeching/godot_rl_agents)** (1.6k): train agents inside a game you build yourself in Unity or Godot.
- **[PySC2](https://github.com/google-deepmind/pysc2)** (StarCraft II, 8.3k) and **[MineDojo](https://github.com/MineDojo/MineDojo)** (Minecraft): large, heavy environments; both last pushed in 2024.
- **[NLE](https://github.com/facebookresearch/nle)** (NetHack) and **[MiniHack](https://github.com/facebookresearch/minihack)**: archived, so read-only.

### Example: the Gymnasium contract

Every environment in this layer is driven by the same loop. A random agent on CartPole is the whole contract in six lines:

```python
import gymnasium as gym
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
```

### Example: PettingZoo's turn-based variant

PettingZoo cannot reuse that loop, because in a two-player game someone else moves in between. It models games as [Agent Environment Cycle (AEC)](https://pettingzoo.farama.org/api/aec/) games, where agents act one at a time, and `env.last()` returns the observation for whoever is up. Its tic-tac-toe observation includes an `action_mask` of the legal squares:

```python
import numpy as np
from pettingzoo.classic import tictactoe_v3
env = tictactoe_v3.env()
env.reset(seed=0)
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        env.step(None); continue
    legal = np.flatnonzero(obs["action_mask"])
    env.step(int(np.random.choice(legal)))  # swap in a learned policy here
```

To check that the API does what it says, I trained a tabular Q-learning player (about 40 lines, no neural network) as `player_1` against a random opponent. Results over 1,000 test games per trained row (2,000 for the random baseline), one seed:

| Player 1 | Win | Draw | Loss |
| --- | --- | --- | --- |
| Random moves | 58.6% | 12.3% | 29.2% |
| After 5,000 training games | 96.5% | 2.4% | 1.1% |
| After 20,000 training games | 99.1% | 0.9% | 0.0% |

Training took 7.6 seconds and the table holds 8,398 states. The caveat matters: the opponent was random, so this shows the agent learned to beat a random player, not to play tic-tac-toe perfectly. It also learned only the first-player seat. For genuinely adversarial training, PettingZoo pairs with self-play, as in the AlphaStar and OpenAI Five routes from the [first post](/posts/how-ai-learns-to-play-video-games/).

## Layer 2: algorithms

This layer takes an environment and returns a trained policy. The choice here is mostly about how much you want to see.

- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** (13.9k stars): a few lines to train, a large algorithm menu, a stable API. The default for getting something working, used in the previous two posts.
- **[CleanRL](https://github.com/vwxyzjn/cleanrl)** (10.5k): the opposite philosophy. Its README describes "high-quality single-file implementation with research-friendly features." Each algorithm is one self-contained script. Run here.
- **[Tianshou](https://github.com/thu-ml/tianshou)** (11k) and **[TorchRL](https://github.com/pytorch/rl)** (3.6k, [docs](https://pytorch.org/rl/)): full libraries with modular building blocks; TorchRL is the PyTorch project's own.
- **[Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)**: distributed training across many machines, the choice when one CPU is no longer enough.
- **[Sample Factory](https://github.com/alex-petrenko/sample-factory)** (1k) and **[EnvPool](https://github.com/sail-sg/envpool)** (1.5k): throughput tools. EnvPool is a C++ engine that runs many environment copies in parallel to feed the learner faster.

### Example: CleanRL's single file

SB3 hides PPO behind `model.learn()`. CleanRL's [`ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) is 312 lines where the network, rollout loop, advantage computation and clipped loss all sit in front of you, with settings as command-line flags:

```bash
python ppo.py --env-id CartPole-v1 --total-timesteps 100000 --seed 1
```

One compatibility note from running it. CleanRL's `pyproject.toml` pins `gymnasium==0.29.1` and Python below 3.11, and the current Gymnasium 1.4.0 returns episode statistics in a different `infos` format, so the script's logging lines found nothing until I changed a handful of lines to read `infos["episode"]` and `infos["_episode"]`. With a matching Gymnasium version this is not needed. The old `cleanrl` package on PyPI (0.4.8, from 2021) is a different, outdated codebase; use the GitHub repository.

Result on CartPole, 4 parallel environments, one seed, CPU only: the run took 17.5 seconds at about 6,400 steps per second. Across 590 training episodes the mean return went from 20.0 over the first 20 episodes to 293.6 over the last 20 (the maximum is 500). Because those are training episodes with a stochastic policy, they are a progress measure, not a final evaluation.

The trade-off against SB3 is clear from the same task. SB3 is faster to start, and CleanRL is easier to read and modify, because nothing is behind an abstraction.

## Layer 3: agents

The top layer does not train a network with a reward signal in a game. It builds a system around a pretrained model that reads the game state, plans, and acts.

- **Voyager** ([arXiv 2305.16291](https://arxiv.org/abs/2305.16291)): an LLM agent in Minecraft that writes code as its actions and keeps a growing skill library, covered in the first post.
- **[BALROG](https://github.com/balrog-ai/BALROG)** (273 stars, last pushed 2026-04): a benchmark whose description reads "Benchmarking Agentic LLM and VLM Reasoning On Games." It is the closest thing to a standard test for this layer.

Frameworks here are thinner and move fast, so the main signal is the benchmark: it tells you how well a model plays, where layers 1 and 2 tell you how to train one.

## Choosing

- **First project:** Gymnasium plus Stable-Baselines3.
- **Understanding the algorithm:** CleanRL, read the file next to the paper.
- **Two or more players:** PettingZoo, or OpenSpiel for board and card games.
- **Your own game:** Unity ML-Agents or Godot RL Agents.
- **More speed:** EnvPool or Craftax for the environment, RLlib for more machines.
- **An LLM as the player:** Voyager-style agents, scored with BALROG.

The layers are independent: the same algorithm family can drive Atari, a Doom level or a game built in Unity, with a different network and preprocessing for each, which is the real reason to learn the stack by layer rather than by library.

## Versions used

Python 3.14, gymnasium 1.4.0, PettingZoo 1.27.0, torch 2.14.1 (CPU), CleanRL master (last commit 2026-04-20), on a 32-core machine with no GPU.
