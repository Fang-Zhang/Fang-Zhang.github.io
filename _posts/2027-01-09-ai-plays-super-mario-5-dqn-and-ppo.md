---
title: "AI Plays Super Mario, Part 5: Two Ways to Learn, DQN and PPO"
date: 2027-01-09 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, dqn, ppo, games, super-mario, tutorial, series]
description: "Part 5 of the AI Plays Super Mario series. DQN learns how good each button is; PPO learns which button to press. Two runnable toy examples show the update rule of each, and the PPO settings used in this series are explained one by one."
---

[Part 4](/posts/ai-plays-super-mario-4-reward-design/) showed that the reward decides what the AI learns, and treated the training algorithm as a fixed tool. This post opens the tool. It explains the two algorithms most game-playing AIs are built on, DQN and PPO, with one tiny runnable example for each, and then lists the PPO settings used in the Mario runs and what each one does. No new Mario training was run for this post, so there is no DQN-versus-PPO result on Mario here, and none is claimed.

## Two questions an algorithm can answer

The AI sees a screen and must pick one of seven button combinations. There are two ways to turn experience into a decision rule.

- **Estimate how good each button is, then press the best one.** The estimate is called the action value, written Q(state, action): the total reward the AI expects from here on if it presses this button now and plays well afterwards. This is the DQN family.
- **Learn the probability of pressing each button directly.** The network outputs seven probabilities, and the AI samples from them. Training nudges the probabilities toward buttons that worked. This is the policy-gradient family, and PPO belongs to it.

## DQN: learn a score for every button

### The update rule on a corridor

The idea of DQN comes from Q-learning. The 2013 paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) builds the method on the Bellman equation: the value of a button now is the reward it brings, plus the discounted value of the best button available next. In code, one update is a single line: move the stored estimate a step toward `reward + gamma * (best next estimate)`.

The smallest world where this can be seen is a corridor of five cells. The AI starts in cell 0, can press left or right, and gets a reward of 1 on reaching cell 4. This toy is the author's own, not from a paper. Save it as `corridor_q.py`:

```python
import random

random.seed(0)
N = 5                       # cells 0..4, reward 1 for reaching cell 4
GAMMA, ALPHA, EPS = 0.9, 0.5, 0.3
Q = [[0.0, 0.0] for _ in range(N)]       # Q[state][action], action 0 = left, 1 = right

def step(s, a):
    s2 = max(0, s - 1) if a == 0 else s + 1
    return s2, (1.0 if s2 == N - 1 else 0.0), s2 == N - 1

for episode in range(300):
    s, done = 0, False
    while not done:
        if random.random() < EPS:
            a = random.randrange(2)                      # explore
        else:
            a = max((0, 1), key=lambda x: Q[s][x])       # exploit (ties go to left)
        s2, r, done = step(s, a)
        target = r + (0 if done else GAMMA * max(Q[s2]))     # Bellman target
        Q[s][a] += ALPHA * (target - Q[s][a])
        s = s2

for s in range(N - 1):
    print(f"cell {s}: Q(left)={Q[s][0]:.3f}  Q(right)={Q[s][1]:.3f}  exact Q(right)={GAMMA ** (N - 2 - s):.3f}")
```

Output:

```
cell 0: Q(left)=0.656  Q(right)=0.729  exact Q(right)=0.729
cell 1: Q(left)=0.656  Q(right)=0.810  exact Q(right)=0.810
cell 2: Q(left)=0.729  Q(right)=0.900  exact Q(right)=0.900
cell 3: Q(left)=0.810  Q(right)=1.000  exact Q(right)=1.000
```

The learned value of pressing right matches the exact value, which is 0.9 raised to the number of cells still to go after this step: 1.000 next to the goal, 0.900 one cell further back, and so on. The reward information has spread backwards from the goal, one cell further with each pass. The AI's rule is then trivial: in every cell, pick the larger number, which here is always right.

Two lines of the code are the two ideas of the whole family. The `EPS` branch is exploration: with probability 0.3 the AI presses a random button, because an AI that always follows its current estimates never finds out whether another button is better. The `target` line is the Bellman update.

### From a table to a network

Mario has no table. A screen is 84 × 84 numbers, and the same screen almost never occurs twice. The 2013 paper replaces the table with a convolutional network that takes the last four frames as input and outputs one Q value per button. Its input is 84 × 84 × 4, its first layer has 16 filters of 8 × 8 with stride 4, its second has 32 filters of 4 × 4 with stride 2, then a 256-unit layer, then one output per valid action. The Nature paper of 2015, [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236), reports that this agent, using only pixels and the game score, reached a level comparable to a professional human games tester across 49 Atari games with one algorithm, one architecture and one set of hyperparameters.

Training a network on a stream of game frames is unstable, because consecutive frames are nearly identical and the targets move as the network changes. The DQN paper adds an **experience replay** memory: every step is stored, and each update trains on a random batch of old steps. The paper notes that this requires learning off-policy, meaning the stored steps were produced by older versions of the network, which is why it builds on Q-learning. The [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) lists the stabilising tricks of the DQN implementation as a replay buffer, a target network (a slowly updated copy of the network used to compute the targets) and gradient clipping.

## PPO: learn which button to press

### What gets learned

A PPO network has two outputs: seven probabilities (the policy) and one number estimating how good the current screen is. After the AI has played a batch of steps, each step gets an **advantage**: how much better the outcome was than that estimate predicted. A positive advantage means "that button did better than expected here", a negative one means worse. Training raises the probability of buttons with positive advantage and lowers the probability of the others.

### Why it is "proximal"

The paper that introduced the method, [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), starts from a practical problem: doing several gradient steps on the same batch of data often leads to destructively large changes to the policy. PPO's answer is a clipped objective. For each step it computes the ratio of the new probability to the old probability of the button that was pressed, and clips that ratio to the range from 1 − ε to 1 + ε, so there is no benefit in moving a probability by more than that.

The following script, the author's own illustration of the paper's formula with the clip range ε = 0.1 used in the Mario runs, evaluates it for one step where the old policy gave the pressed button a probability of 0.40. Save it as `ppo_clip.py`:

```python
EPS = 0.1                    # the clip range used in this series' PPO runs

def objective(ratio, adv):
    clipped = min(max(ratio, 1 - EPS), 1 + EPS)
    return min(ratio * adv, clipped * adv)

old_p = 0.40                 # probability the old policy gave to the sampled action
for adv in (+2.0, -2.0):
    print(f"advantage {adv:+.1f}")
    for new_p in (0.30, 0.36, 0.40, 0.44, 0.50, 0.60):
        r = new_p / old_p
        print(f"  new prob {new_p:.2f}  ratio {r:.3f}  objective {objective(r, adv):+.3f}")
```

Output:

```
advantage +2.0
  new prob 0.30  ratio 0.750  objective +1.500
  new prob 0.36  ratio 0.900  objective +1.800
  new prob 0.40  ratio 1.000  objective +2.000
  new prob 0.44  ratio 1.100  objective +2.200
  new prob 0.50  ratio 1.250  objective +2.200
  new prob 0.60  ratio 1.500  objective +2.200
advantage -2.0
  new prob 0.30  ratio 0.750  objective -1.800
  new prob 0.36  ratio 0.900  objective -1.800
  new prob 0.40  ratio 1.000  objective -2.000
  new prob 0.44  ratio 1.100  objective -2.200
  new prob 0.50  ratio 1.250  objective -2.500
  new prob 0.60  ratio 1.500  objective -3.000
```

For the good button (advantage +2), the objective rises as its probability rises, up to a ratio of 1.1, which is a probability of 0.44. Beyond that it stays at +2.2: raising the probability to 0.60 earns nothing more, so the training has no reason to push further. For the bad button (advantage −2), the objective stops improving once the probability has fallen to a ratio of 0.9, which is 0.36. The figure plots the same function.

![Two plots of the PPO objective against the probability ratio. With positive advantage the objective rises and then flattens above a ratio of 1.1. With negative advantage it is flat below a ratio of 0.9 and falls above it.]({{ "/assets/pic/mario-ai-ppo-clip.png" | relative_url }})

*The clipped objective for one step, with ε = 0.1. The grey dashed line is the same objective without clipping. Drawn from the formula in the PPO paper.*

The flat parts are the whole trick. The AI can reuse one batch of data for several epochs without the policy lurching, and it needs no replay memory: PPO alternates between collecting fresh data with the current policy and optimising on it, then throws the data away.

## The two side by side

- **What is learned.** DQN: a value per button. PPO: a probability per button, plus a value for the screen.
- **How data is used.** DQN keeps old steps in a replay memory and trains on random samples of them (off-policy). PPO uses each batch of fresh steps for a few epochs and discards it (on-policy).
- **How it explores.** DQN presses a random button with some probability ε. PPO samples from its own probabilities, and an entropy bonus rewards keeping them spread out.
- **How it acts when finished.** DQN presses the highest-valued button, with a small ε left on (0.05 in the 2013 paper's evaluation). A PPO model can either sample or always press its most probable button. Part 4 noted that for Mario these two ways of acting can give very different results from the same trained network, and Part 7 shows how.

Each design has a known weakness stated by its own literature. The PPO paper says Q-learning with function approximation "fails on many simple problems" and is poorly understood; that is the PPO authors' view in 2017, not a measurement made here. The DQN paper says its approach lacks theoretical convergence guarantees. And PPO's paper compares itself with A2C and ACER on Atari, not with DQN, so the paper does not rank the two.

## Which one this series uses, and why

This series uses PPO from Stable-Baselines3. This is a choice, not a finding of a comparison. The reasons are practical:

- PPO runs naturally on many copies of the game at once. The Mario runs use 16 emulator processes, which suits a CPU-only machine with many cores.
- It has no replay memory to hold. By the author's arithmetic, one stored 84 × 84 × 4 observation takes 28,224 bytes, so a buffer of 100,000 of them takes about 2.8 GB for the observations alone.
- Part 4's runs show that this PPO setup learns to clear level 1 under a good reward.

None of this shows that DQN could not do the same. It was not run here.

## The PPO settings in the Mario runs

These are the settings every Mario run in this series used. The names are those of the Stable-Baselines3 `PPO` class.

- `n_envs = 16`: sixteen copies of the game run in parallel.
- `n_steps = 128`: each copy plays 128 decisions before an update, so one batch is 16 × 128 = 2,048 decisions.
- `batch_size = 256` and `n_epochs = 4`: the batch is cut into 8 mini-batches of 256, and the optimiser passes over them 4 times, which makes 32 gradient steps per batch.
- `learning_rate = 2.5e-4`: the size of each gradient step. The PPO paper uses it as the starting value in its Atari experiments, where it is then decayed to 0.
- `clip_range = 0.1`: ε in the formula above, also the paper's starting Atari value (the paper decays it during training; this setup keeps both constant).
- `ent_coef = 0.01`: the weight of the entropy bonus that keeps the probabilities from collapsing too early. The paper's Atari table lists the same value.
- `gamma = 0.99`: the discount, how much a reward one decision later is worth compared with a reward now. The corridor example used 0.9 to keep its numbers short.
- 4 stacked frames of 84 × 84 grayscale, as in Part 3. The random seed is fixed at 0, so each run is a single sample.

With these numbers, a 3,000,000-decision run is about 1,465 batches (3,000,000 divided by 2,048), or about 47,000 gradient steps (1,465 × 32).

## Checkpoint

Run both scripts with plain Python 3, no extra packages. `corridor_q.py` should print four rows in which the learned `Q(right)` equals the exact value. If the numbers differ slightly from the output above, check that the `random.seed(0)` line is present. `ppo_clip.py` should print exactly the two blocks shown, with the objective flat at +2.200 for the positive advantage once the ratio reaches 1.1.

Part 6 leaves reinforcement learning for a while and asks a different question: if a human plays the level and the machine records the screen and the keys, can the AI learn to copy it?
