---
title: "AI Plays Super Mario, Part 3: The Environment, From Memory Addresses to Rewards"
date: 2027-01-05 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, games, super-mario, stable-retro, gymnasium, tutorial, series]
description: "Part 3 of the AI Plays Super Mario series. Read Mario's position from the game's memory, then wrap the emulator in a Gymnasium environment with 7 actions, frame skipping, 84x84 grayscale input and a reward. Includes the code and the outputs it produced."
---

[Part 2](/posts/ai-plays-super-mario-2-setup-and-mario-moving/) ended with Mario running into the first Goomba. This post turns the raw emulator into the environment a training algorithm can use. It reads Mario's position from the game's memory, shrinks the picture, reduces the buttons to seven choices, holds each choice for four frames, and computes a reward. The result is one file, `mario_env.py`, which every later post imports. The checkpoint at the end shows what two fixed strategies score in it.

All outputs below come from the same machine as Part 2 (Ubuntu 24.04 under WSL2, Python 3.14.7, no GPU).

## What `info` gives, and when

The environment returns an `info` dictionary with numbers the game keeps in memory. There is a catch that costs beginners an hour: `reset()` returns an empty one.

```python
import numpy as np
import stable_retro as retro

env = retro.make("SuperMarioBros-Nes-v0", render_mode="rgb_array")
obs, info = env.reset()
print("info right after reset():", info)

obs, reward, terminated, truncated, info = env.step(np.zeros(9, dtype=np.int8))   # one step, no buttons
print("info after one step:", info)
```

```
info right after reset(): {}
info after one step: {'coins': 0, 'score': 0, 'levelHi': 0, 'lives': 2, 'levelLo': 0, 'time': 400, 'scrolling': 16, 'xscrollHi': 0, 'xscrollLo': 0}
```

Reading `info["lives"]` straight after `reset()` raises `KeyError: 'lives'`. When training runs the environment in several worker processes, the same error can surface as a less helpful message from the worker pipe, so it is worth fixing at the source: take one step with no buttons after every reset.

## Where is Mario? Two numbers, one distance

`info` has a scroll position, `xscrollHi * 256 + xscrollLo`, which says how far the screen has moved right. Scroll position is not Mario's position, though. Mario can walk on the screen without scrolling it, and the screen stops scrolling near the end of the level.

Mario's own position is also in memory. The community [RAM map for Super Mario Bros.](https://datacrystal.tcrf.net/wiki/Super_Mario_Bros.:RAM_map) lists address `0x006D` as the player's horizontal position in the level (which screen-wide page he is on) and `0x0086` as his x position on the screen. A page is 256 pixels wide, so the position in the level is `page * 256 + x_on_screen`. The emulator exposes memory through `env.get_ram()`:

```python
def mario_x():
    ram = env.get_ram()
    return int(ram[0x6D]) * 256 + int(ram[0x86])
```

Holding RIGHT for 300 frames and printing both numbers every 100 frames gave:

```
step 100  mario_x= 160  scroll=  53  gap=107  score=0  time=396
step 200  mario_x= 295  scroll= 183  gap=112  score=0  time=392
step 300  mario_x= 295  scroll= 183  gap=112  score=0  time=392
```

Mario's position stays ahead of the scroll by a similar gap (107 and 112 here), because the screen follows him at a fixed offset. At step 200 he reaches 295 and stops: the first Goomba has hit him. The two numbers move together, so either would do for the first part of the level. Mario's own position is the one used in this series, because it stays correct on the last screen where scrolling stops.

## Seven buttons instead of nine inputs

Part 1 narrowed the nine inputs to seven useful combinations. In code, a combination is a list of nine zeros and ones, built from button names so that nobody has to count positions:

```python
BUTTONS = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]

def combo(*pressed):
    return [1 if b in pressed else 0 for b in BUTTONS]

ACTIONS = [
    combo(),                       # 0 no button
    combo("RIGHT"),                # 1 right
    combo("RIGHT", "A"),           # 2 right + A
    combo("RIGHT", "B"),           # 3 right + B
    combo("RIGHT", "A", "B"),      # 4 right + A + B
    combo("A"),                    # 5 A
    combo("LEFT"),                 # 6 left
]
```

The environment's action space becomes `Discrete(7)`: the AI picks a number from 0 to 6.

## Frame skipping

The game runs at 60 frames per second. A decision every frame would mean 60 decisions per second of play, and consecutive frames hardly differ. The wrapper repeats each chosen action for 4 frames, so the AI decides 15 times per second. Training sees four times less data per second of game time, and a jump (which lasts many frames) is still a handful of decisions.

## The reward, in code

The reward has the three parts described in Part 1, plus a small cost per decision so that standing still is not free:

- progress: `0.1 × (x - previous x)`, clipped to the range −1 to 3 so that one odd frame cannot dominate
- time: −0.01 per decision
- death: −5, when the lives counter drops
- clear: +10, when the level number changes

An episode ends when Mario dies, clears the level, or reaches 4,000 decisions, whichever is first.

## The file

Save this as `mario_env.py`:

```python
import cv2
import gymnasium as gym
import numpy as np
import stable_retro as retro

BUTTONS = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]

def combo(*pressed):
    return [1 if b in pressed else 0 for b in BUTTONS]

ACTIONS = [
    combo(),                       # 0 no button
    combo("RIGHT"),                # 1 right
    combo("RIGHT", "A"),           # 2 right + A
    combo("RIGHT", "B"),           # 3 right + B
    combo("RIGHT", "A", "B"),      # 4 right + A + B
    combo("A"),                    # 5 A
    combo("LEFT"),                 # 6 left
]


class Mario(gym.Env):
    def __init__(self, skip=4, size=84, max_steps=4000):
        self.env = retro.make("SuperMarioBros-Nes-v0", render_mode="rgb_array")
        self.skip, self.size, self.max_steps = skip, size, max_steps
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, (size, size, 1), np.uint8)

    def _picture(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return small[:, :, None]

    def _x(self):
        ram = self.env.get_ram()
        return int(ram[0x6D]) * 256 + int(ram[0x86])

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        obs, _, _, _, info = self.env.step(np.array(ACTIONS[0], dtype=np.int8))  # reset() returns an empty info
        self.steps = 0
        self.last_x = self.best_x = self._x()
        self.last_lives = info["lives"]
        self.start_level = (info["levelHi"], info["levelLo"])
        return self._picture(obs), info

    def step(self, action):
        done = False
        for _ in range(self.skip):                       # hold the buttons for 4 frames
            obs, _, term, trunc, info = self.env.step(np.array(ACTIONS[int(action)], dtype=np.int8))
            done = done or term or trunc
            if done:
                break
        self.steps += 1
        x = self._x()
        reward = float(np.clip((x - self.last_x) * 0.1, -1, 3)) - 0.01
        self.last_x = x
        self.best_x = max(self.best_x, x)
        died = info["lives"] < self.last_lives
        cleared = (info["levelHi"], info["levelLo"]) != self.start_level
        if died:
            reward -= 5.0
        if cleared:
            reward += 10.0
        terminated = died or cleared or done
        truncated = self.steps >= self.max_steps
        return self._picture(obs), reward, terminated, truncated, dict(info, best_x=self.best_x, cleared=cleared)
```

The class follows the Gymnasium convention from Part 1: `reset` returns an observation and info, `step` returns observation, reward, `terminated`, `truncated` and info. The extra `best_x` and `cleared` entries in `info` are for logging only.

## Checkpoint

Save this as `try_wrapper.py`. It runs two fixed strategies, always RIGHT and always RIGHT + A, then times random play:

```python
import time
import numpy as np
from mario_env import Mario

env = Mario()
obs, info = env.reset()
print("observation:", obs.shape, obs.dtype, "| actions:", env.action_space)

for name, action in [("right (1)", 1), ("right + A (2)", 2)]:
    obs, info = env.reset()
    total, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward; steps += 1
        if terminated or truncated:
            break
    print(f"always {name:14s}: {steps:4d} decisions, total reward {total:7.2f}, best_x {info['best_x']}")

rng = np.random.default_rng(0)
obs, info = env.reset(); t0 = time.time(); n = 0
while n < 2000:
    obs, r, term, trunc, info = env.step(rng.integers(7)); n += 1
    if term or trunc:
        obs, info = env.reset()
print(f"random play: {n / (time.time() - t0):.0f} decisions per second on one core")
```

Output on the test machine:

```
observation: (84, 84, 1) uint8 | actions: Discrete(7)
always right (1)     :  100 decisions, total reward   19.50, best_x 295
always right + A (2) :  100 decisions, total reward   18.50, best_x 295
random play: 650 decisions per second on one core
```

Check these:

- The observation is `(84, 84, 1)` and there are 7 actions.
- Both fixed strategies die at the same place, `best_x` 295, after 100 decisions: the first Goomba. Holding A as well changes the reward slightly, not the result.
- Speed is in the hundreds of decisions per second per core. Yours will differ with the CPU.

The reward number is worth a second look. Always-right earns 19.50 even though the level is not cleared: progress from the start (about 40) to 295 adds up to 25.5, minus 5 for dying and 1.0 for the 100 decisions of time cost. A strategy that fails at the first enemy still scores well on this reward, and any real agent must beat it. Part 4 changes the reward and measures how far this choice matters.

Part 4 keeps the environment fixed and varies the reward, to see what an AI does with each version.
