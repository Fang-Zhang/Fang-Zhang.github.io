---
title: "How AI Learns to Play Video Games: Four Routes"
date: 2026-10-25 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, deep-learning, games, LLM, agents, reading-list]
description: "Reinforcement learning from pixels, self-play at scale, imitation from video, and LLM agents that write code: four routes by which AI learns to play games, with a runnable example."
---

Every game-playing AI answers the same three questions. What does it see? What can it do? What counts as winning? The research history is mostly a sequence of different answers to those questions, and each answer opened a new route.

This post covers four routes, from the one that started the field to the one still being explored, then ends with a short program that can be run in under a minute.

## The loop underneath everything

All of the systems below share one structure. The agent receives an *observation* (pixels, game state, text), picks an *action* (a button press, a mouse move, a line of code), and gets a *reward* (the score, a win, an item collected). Then the game advances and the loop repeats. This is reinforcement learning, and the open-source [Arcade Learning Environment](https://ale.farama.org/) states the interface plainly: a `step` call takes an action and returns the next observation, a reward, and a flag for whether the episode has ended. It is built for exactly this purpose, "a framework that allows researchers and hobbyists to develop AI agents for Atari 2600 roms."

What changes between routes is how the agent turns observations into actions.

## Route 1: learn from pixels and score alone

In 2013, DeepMind's [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) presented "the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning." The model was a convolutional neural network, trained with a variant of Q-learning, whose input was raw pixels and whose output was a value function estimating future rewards. It outperformed all previous approaches on six of seven Atari games and surpassed a human expert on three.

The follow-up in *Nature*, [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236), scaled this to 49 games. The agent, a deep Q-network, received "only the pixels and the game score as inputs" and reached "a level comparable to that of a professional human games tester", using the same algorithm, network architecture and hyperparameters across all 49 games. The same design played every game; only the training experience differed.

Two later papers removed assumptions from this recipe:

- [MuZero](https://arxiv.org/abs/1911.08265) combines tree search with a learned model of the game, reaching a new state of the art on 57 Atari games. On Go, chess and shogi it matched AlphaZero "without any knowledge of the game rules." The agent no longer needs to be told how the game works.
- [DreamerV3](https://arxiv.org/abs/2301.04104) learns a model of the environment and improves its behaviour "by imagining future scenarios". With a single configuration it outperforms specialised methods across more than 150 tasks, and it is the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula.

## Route 2: self-play at very large scale

Games with two sides allow a different trick: the opponent can be a copy of the agent. [OpenAI Five](https://arxiv.org/abs/1912.06680) used existing reinforcement learning techniques, scaled to learn from batches of about 2 million frames every 2 seconds, with a distributed system that trained for 10 months. On 13 April 2019 it became "the first AI system to defeat the world champions at an esports game", beating Team OG at Dota 2. The paper's conclusion is that self-play reinforcement learning can reach superhuman performance on a difficult task, with the difficulties named as long time horizons, imperfect information and complex action spaces.

Self-play has a known failure mode, which DeepMind describes in its [AlphaStar write-up](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning): forgetting. An agent can keep improving yet lose its ability to beat earlier versions, and the strategies cycle like rock, paper, scissors. AlphaStar's answer was the League, a group of agents in which some play to win and others exist to expose the weaknesses of the main agents. Playing under human constraints (a camera view and capped action rates), it was ranked above 99.8% of active players on Battle.net and reached Grandmaster level with all three StarCraft II races. The *Nature* version is [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z).

## Route 3: watch humans play

Reward signals can be too sparse to learn from, and humans have already produced millions of hours of gameplay video. The catch, as the [Video PreTraining (VPT)](https://arxiv.org/abs/2206.11795) paper states, is that such videos do not contain the action labels needed to train a behavioural prior.

VPT's workaround: train an inverse dynamics model on a small amount of labelled data, accurate enough to label a huge unlabelled source of online Minecraft videos, then train a general behavioural prior on that. The agent uses the native human interface, mouse and keyboard at 20 Hz. Fine-tuned with imitation and reinforcement learning, it handles hard-exploration tasks "impossible to learn from scratch via reinforcement learning", and the authors report the first computer agents that can craft diamond tools, which take proficient humans upwards of 20 minutes (24,000 environment actions).

## Route 4: let a language model write the player

The newest route replaces the neural network policy with a large language model. [Voyager](https://arxiv.org/abs/2305.16291) plays Minecraft through GPT-4 with black-box queries, no fine-tuning of model parameters. It has three parts: an automatic curriculum that maximises exploration, a growing skill library of executable code, and an iterative prompting loop that feeds back environment results, execution errors and self-verification.

The skill library is the notable design choice. A learned behaviour is stored as a program, so it is, in the paper's words, "temporally extended, interpretable, and compositional." Voyager obtains 3.3 times more unique items, travels 2.3 times longer distances and unlocks key tech-tree milestones up to 15.3 times faster than the prior state of the art, and the skill library transfers to a new Minecraft world.

Compare the routes: Routes 1 to 3 change numbers inside a network, and Route 4 changes a text file of code. What the agent has learned can be read, edited and reused.

## A minimal version to run

The first route in miniature needs two libraries. [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html) provides the algorithms with an sklearn-like interface, and Gymnasium provides the environments, including ALE's Atari games. This example uses CartPole, a pole balanced on a cart, because it trains in seconds on a CPU and needs no emulator ROMs:

```python
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")
model = DQN("MlpPolicy", env, seed=0)
model.learn(total_timesteps=50_000)

def mean_return(policy, episodes=10):
    total = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    return total / episodes

print("trained:", mean_return(lambda o: model.predict(o, deterministic=True)[0]))
print("random: ", mean_return(lambda o: env.action_space.sample()))
```

On gymnasium 1.4.0 and stable-baselines3 2.9.0, CPU only, this ran in about 25 seconds and printed a mean return of 219.8 for the trained agent against 28.7 for random play (the maximum is 500). That is one seed and ten test episodes, so it demonstrates that learning happens, not how well the algorithm performs in general.

To move to Atari, install ALE and replace the environment name with `"ALE/Breakout-v5"` as shown in the [ALE documentation](https://ale.farama.org/). Pixel inputs need a convolutional policy instead of `MlpPolicy` and far more training steps, hours rather than seconds.

## What the four routes have in common

Each route made the same trade. It gave the agent less pre-built knowledge about the game in exchange for more experience of it: no hand-made features (route 1), no rules (MuZero), no opponent except itself (route 2), no action labels (route 3), no fixed skill set (route 4). Progress came from removing what had been supplied by hand and letting the loop of observe, act and score supply it instead.
