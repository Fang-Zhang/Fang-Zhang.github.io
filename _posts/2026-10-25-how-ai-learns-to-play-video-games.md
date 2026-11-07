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

![Agent and game exchanging action, observation and reward in a loop]({{ "/assets/pic/games-agent-loop.png" | relative_url }})
_Figure 1. The loop shared by every route below._

## The loop in miniature: Q-learning on a 20-cell grid

Before the Atari systems, here is the whole idea at a scale that can be followed by hand. The example is mine, not from the papers below, and every number was computed in code.

**The scenario.** A grid of 5 by 4 cells. The agent starts in the bottom-left corner and can move up, down, left or right. Reaching the goal cell scores +1, falling into the pit next to it scores −1, and every other move costs 0.04, so wasted moves are penalised. A wall blocks one cell. The agent is not told any of this. It only sees which cell it is in, picks a move, and gets a reward.

**The problem it solves.** The reward for a good move often arrives much later than the move. Q-learning, the algorithm DeepMind's 2013 paper used, handles this by learning a number Q for every (cell, move) pair: the total reward the agent expects to collect from there on, if it plays well afterwards. The update after each move is the Bellman rule the 2013 paper writes out:

Q(s, a) ← Q(s, a) + α × [ r + γ × max Q(s′, a′) − Q(s, a) ]

Here s′ is the cell the move led to, α = 0.5 is how far to move toward the new estimate, and γ = 0.95 discounts future reward. In words: nudge the old estimate toward "the reward I just got, plus the best value I think is available from the next cell".

**Worked example.** All Q values start at 0. In the first episode, the agent happens to step from the cell next to the goal into the goal:

- Q = 0 + 0.5 × (1 − 0) = **0.5**. Only the final move knew anything.
- In the next episode, the agent steps from the cell below that one into it. The reward is just −0.04, but the best value in the next cell is now 0.5: Q = 0 + 0.5 × (−0.04 + 0.95 × 0.5 − 0) = **0.2175**.

The news of the goal travels backwards one cell per visit, and after 400 episodes it has reached the start. At that point the agent's best move from every cell points along a path to the goal and around the pit, and the start cell carries a value of 0.52. The greedy route takes 7 moves and would score 1 − 6 × 0.04 = 0.76. The average over the last 50 training episodes is lower, 0.49, because during training the agent still makes a random move 20% of the time (the ε-greedy rule from the DQN paper), and some of those moves end in the pit. That is the dips in the right-hand chart. The first 10 episodes averaged −0.18.

![Left: a 5 by 4 grid with start, wall, goal and pit. Middle: the same grid after training, with an arrow in each cell for the best move and its Q value. Right: reward per episode rising from negative to about 0.5]({{ "/assets/pic/games-gridworld.png" | relative_url }})
_Figure 2. Tabular Q-learning on a toy grid. Left: the game. Middle: the best move and Q value in each cell after 400 episodes. Right: reward per episode, with the dips caused by random exploration. Computed in code._

The grid has 20 cells, so a table with one row per cell works. An Atari screen has no such table. Route 1 is what happens when the table is replaced by a neural network.

## Route 1: learn from pixels and score alone

In 2013, DeepMind's [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) presented "the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning." The model was a convolutional neural network, trained with a variant of Q-learning, whose input was raw pixels and whose output was a value function estimating future rewards. It outperformed all previous approaches on six of seven Atari games and surpassed a human expert on three.

**The problem it solves.** The grid above stores one number per cell. A game screen has far too many possible images for a table, and two screens that differ by a few pixels should get similar values. A neural network generalises across screens where a table cannot. The 2013 paper's network works like this: the raw 210×160 colour frames are converted to grey, cropped to 84×84, and the last 4 frames are stacked, so the network can see motion. Two convolutional layers (16 filters of 8×8 with stride 4, then 32 filters of 4×4 with stride 2) and a fully connected layer of 256 units feed an output layer with one value per valid action, between 4 and 18 depending on the game. One pass through the network gives the Q value of every button at once.

![A pipeline of boxes: raw frame, grey-scale and 4 frames stacked, two convolutional layers, a fully connected layer, and one Q-value output per action, with an experience replay memory feeding training]({{ "/assets/pic/games-dqn-network.png" | relative_url }})
_Figure 3. The 2013 DQN network, redrawn from the paper's description. The action names on the outputs are illustrative; the games use between 4 and 18 actions._

Two more details make the training work. The agent stores each experience (state, action, reward, next state) in a replay memory and trains on random samples from it, which, in the paper's words, breaks the correlations between consecutive samples. And it picks a random action with probability ε, annealed from 1 to 0.1 over the first million frames, so it keeps exploring. Rewards were clipped to +1 and −1 during training, so one learning rate worked across games with very different score scales.

**The result.** The paper's Table 1 compares average scores on seven games, with one network design and one set of hyperparameters:

![Seven small bar charts comparing random play, the DQN agent and a human on Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest and Space Invaders]({{ "/assets/pic/games-dqn-scores.png" | relative_url }})
_Figure 4. Average scores from Table 1 of the 2013 paper. DQN beats the human score on Breakout, Enduro and Pong, and falls short on the other four, the longer-horizon games._

On Breakout the agent averaged 168 against 31 for the human and 1.2 for random play. On Q*bert, Seaquest and Space Invaders it is far below the human score. The authors attribute that to games that "require the network to find a strategy that extends over long time scales."

The follow-up in *Nature*, [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236), scaled this to 49 games. The agent, a deep Q-network, received "only the pixels and the game score as inputs" and reached "a level comparable to that of a professional human games tester", using the same algorithm, network architecture and hyperparameters across all 49 games. The same design played every game; only the training experience differed.

Two later papers removed assumptions from this recipe:

- [MuZero](https://arxiv.org/abs/1911.08265) combines tree search with a learned model of the game, reaching a new state of the art on 57 Atari games. On Go, chess and shogi it matched AlphaZero "without any knowledge of the game rules." The agent no longer needs to be told how the game works.
- [DreamerV3](https://arxiv.org/abs/2301.04104) learns a model of the environment and improves its behaviour "by imagining future scenarios". With a single configuration it outperforms specialised methods across more than 150 tasks, and it is the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula.

## Route 2: self-play at very large scale

Games with two sides allow a different trick: the opponent can be a copy of the agent. [OpenAI Five](https://arxiv.org/abs/1912.06680) used existing reinforcement learning techniques, scaled to learn from batches of about 2 million frames every 2 seconds, with a distributed system that trained for 10 months. On 13 April 2019 it became "the first AI system to defeat the world champions at an esports game", beating Team OG at Dota 2. The paper's conclusion is that self-play reinforcement learning can reach superhuman performance on a difficult task, with the difficulties named as long time horizons, imperfect information and complex action spaces.

**The problem it solves.** In a two-sided game there is no fixed environment to learn from; the right move depends on what the opponent does. A copy of the agent is always available, always roughly the right strength, and costs nothing to produce. The price is the failure mode below.

Self-play has a known failure mode, which DeepMind describes in its [AlphaStar write-up](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning): forgetting. An agent can keep improving yet lose its ability to beat earlier versions, and the strategies cycle like rock, paper, scissors. AlphaStar's answer was the League, a group of agents in which some play to win and others exist to expose the weaknesses of the main agents. Playing under human constraints (a camera view and capped action rates), it was ranked above 99.8% of active players on Battle.net and reached Grandmaster level with all three StarCraft II races. The *Nature* version is [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z).

**The cycle in miniature.** Rock-paper-scissors shows the problem with no game engine at all. This example is mine, and it illustrates the failure DeepMind describes; it is not AlphaStar's algorithm. Start with a version that plays rock. The next version learns to beat it by playing paper, the next beats paper with scissors, the next beats scissors with rock, and so on. Every generation beats the one before it, yet nothing is gained: against all earlier versions combined, each new generation scores about 0.5 (win = 1, tie = 0.5, loss = 0), the same as a coin flip. Now change the rule: each new version best-responds to the average of everything played so far, as in fictitious play. After 300 rounds the mix is 32% rock, 35% paper and 33% scissors, close to the unbeatable one-third each. Keeping the history in play is what removes the cycle, and the League is the large-scale version of the same instinct: keep other opponents around so that no weakness stays hidden.

![Left: a step chart of strategies over 12 generations cycling through rock, paper and scissors. Right: three lines of strategy shares converging to one third each over 300 rounds]({{ "/assets/pic/games-selfplay-cycle.png" | relative_url }})
_Figure 5. Left: each version beats only its predecessor, and the strategies cycle. Right: best-responding to the whole history settles at an even mix. Rock-paper-scissors simulation, computed in code._

## Route 3: watch humans play

Reward signals can be too sparse to learn from, and humans have already produced millions of hours of gameplay video. The catch, as the [Video PreTraining (VPT)](https://arxiv.org/abs/2206.11795) paper states, is that such videos do not contain the action labels needed to train a behavioural prior.

**The problem it solves.** In a game like Minecraft the reward for a hard goal, such as crafting a diamond tool, is far off, and the paper reports that such tasks are impossible to learn from scratch with reinforcement learning. Human gameplay video is plentiful and shows how people reach those goals. But the video shows the screen, not which keys were pressed, and imitation needs both.

VPT's workaround: train an inverse dynamics model on a small amount of labelled data, accurate enough to label a huge unlabelled source of online Minecraft videos, then train a general behavioural prior on that. The agent uses the native human interface, mouse and keyboard at 20 Hz.

![A pipeline: a small labelled video set trains an inverse dynamics model, which adds predicted actions to a large unlabelled video set, which trains a behavioural prior that is then fine-tuned]({{ "/assets/pic/games-vpt-pipeline.png" | relative_url }})
_Figure 6. The VPT recipe: use a small amount of labelled data to label a very large amount of unlabelled data._

The paper gives the reason this works: the inverse dynamics task is simpler than directly imitating behaviour because it is non-causal, meaning it can look at both past and future frames to infer an action. A policy must guess intent from past frames alone. Fine-tuned with imitation and reinforcement learning, it handles hard-exploration tasks "impossible to learn from scratch via reinforcement learning", and the authors report the first computer agents that can craft diamond tools, which take proficient humans upwards of 20 minutes (24,000 environment actions).

## Route 4: let a language model write the player

The newest route replaces the neural network policy with a large language model. [Voyager](https://arxiv.org/abs/2305.16291) plays Minecraft through GPT-4 with black-box queries, no fine-tuning of model parameters. It has three parts: an automatic curriculum that maximises exploration, a growing skill library of executable code, and an iterative prompting loop that feeds back environment results, execution errors and self-verification.

**The problem it solves.** The first three routes store what the agent has learned inside network parameters, which are hard to read and, when trained further on new tasks, can lose earlier abilities (catastrophic forgetting, which the Voyager paper says its design alleviates). Voyager instead keeps the learned behaviour outside the model, as programs that can be read, saved and reused.

![A loop: an automatic curriculum proposes a task, GPT-4 writes code, the code runs in Minecraft, feedback and self-verification return, successful programs are saved in a skill library that feeds later tasks]({{ "/assets/pic/games-voyager-loop.png" | relative_url }})
_Figure 7. Voyager's three parts around one loop, from the paper's description: the curriculum proposes, the model writes code, the game answers, and working code is saved._

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

![Timeline of five systems, each paired with what it no longer needed to be given by hand]({{ "/assets/pic/games-four-routes.png" | relative_url }})
_Figure 8. What each route stopped supplying by hand. Years are first-paper years._
