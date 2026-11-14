---
title: "AI Plays Super Mario, Part 2: Setup, and Mario Moving on Screen"
date: 2026-12-09 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, games, super-mario, stable-retro, gymnasium, tutorial, series]
description: "Part 2 of the AI Plays Super Mario series. Install Stable-Retro and Gymnasium, import a ROM you own, and write two short Python scripts that make Mario move. Includes the three errors most likely to appear and what they mean."
---

[Part 1](/posts/ai-plays-super-mario-1-what-the-machine-sees-does-wants/) defined what the AI sees, does and wants. This post builds the ground it stands on: a Python environment where the game runs, accepts button presses and returns pictures. It ends with two short scripts, one that presses random buttons and one that holds RIGHT and records a video. No learning happens yet. If the checkpoints below match, everything later in the series will run.

## What was tested

All commands and outputs in this post come from one machine: Ubuntu 24.04 under WSL2, Python 3.14.7, 32 CPU cores, no GPU, no display. Package versions were Stable-Retro 1.0.1, Gymnasium 1.4.0, NumPy 2.5.4 and OpenCV 5.0.0.93. Other systems were not tested here, so differences on macOS or native Windows are possible.

## The tools

[Stable-Retro](https://github.com/Farama-Foundation/stable-retro) is maintained under the Farama Foundation, which also maintains Gymnasium. It wraps game emulators in the Gymnasium interface from Part 1, and it ships definitions that say where in the game's memory to find the score, lives and position. It does not ship the games themselves.

That means one thing you must supply: a legal copy of the Super Mario Bros ROM, as a `.nes` file. This series does not provide ROMs, does not link to any, and the code that accompanies it will not contain one.

## Install

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install stable-retro gymnasium numpy opencv-python-headless
```

On the test machine, pip installed prebuilt packages and nothing was compiled.

## Import the ROM

Create a folder called `roms` next to your scripts and copy your `.nes` file into it. Then run:

```bash
python -m stable_retro.import roms
```

Expected output:

```
Importing SuperMarioBros-Nes-v0
Imported 1 games
```

The importer checks each file in the folder against a checksum stored with the game definition, which is why the ROM has to match a known release. The module is `stable_retro`; the older name `retro` belongs to gym-retro, the project this one is forked from.

## Script 1: random buttons

Save this as `hello_mario.py`:

```python
import stable_retro as retro

env = retro.make("SuperMarioBros-Nes-v0", render_mode="rgb_array")
obs, info = env.reset()

print("screen shape:", obs.shape)
print("buttons:", env.buttons)
print("action space:", env.action_space)

for step in range(300):
    action = env.action_space.sample()            # random buttons
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

print("steps done, last info:", info)
env.close()
```

Expected output on the test machine:

```
screen shape: (224, 240, 3)
buttons: ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
action space: MultiBinary(9)
steps done, last info: {'coins': 0, 'score': 0, 'levelHi': 0, 'lives': 2, 'levelLo': 0, 'time': 387, 'scrolling': 16, 'xscrollHi': 0, 'xscrollLo': 0}
```

Each line connects to Part 1:

- The observation is 224 rows by 240 columns of colour pixels. This is the raw picture before it is shrunk for the network.
- The action is nine on/off values, one per entry in `env.buttons`. `action_space.sample()` fills them in at random, so Mario flails.
- The `info` dictionary is not part of the picture. It holds numbers read from the game's memory: coins, score, lives, the countdown timer and the horizontal scroll position. The timer dropped from 400 to 387 here. Part 3 uses these numbers to build the reward.

## Script 2: hold RIGHT and record a video

Save this as `walk_right.py`:

```python
import cv2
import numpy as np
import stable_retro as retro

env = retro.make("SuperMarioBros-Nes-v0", render_mode="rgb_array")
env.reset()

# env.buttons = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
RIGHT = np.zeros(9, dtype=np.int8)
RIGHT[env.buttons.index("RIGHT")] = 1

writer = cv2.VideoWriter("walk_right.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 60, (240, 224))
for step in range(600):
    obs, reward, terminated, truncated, info = env.step(RIGHT)
    writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if step % 100 == 0:
        print(step, "time left:", info["time"], "lives:", info["lives"])
writer.release()
env.close()
```

Expected output:

```
0 time left: 400 lives: 2
100 time left: 396 lives: 2
200 time left: 392 lives: 2
300 time left: 392 lives: 2
400 time left: 392 lives: 1
500 time left: 392 lives: 1
```

The script writes `walk_right.mp4` and opens no window, which is why it also works on a server or inside WSL. The clip below is that output, 600 frames or 10 seconds of game time.

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-walk-right.mp4' | relative_url }}" type="video/mp4">
</video>

*Holding RIGHT for 600 frames. Mario runs into the first Goomba, the lives counter drops from 2 to 1, and the level restarts.*

This clip is the reason the series exists. A program that only knows "go right" cannot get past the first enemy. Everything from Part 3 onward is about replacing that fixed rule with behaviour that depends on what is on the screen.

## Three errors and what they mean

**`ImportError: Library "GLU" not found`** appears when `retro.make` is called without `render_mode="rgb_array"`. Without it the library tries to open a window through OpenGL, which a machine without a display does not have. The error text suggests installing OpenGL or a virtual display; passing `render_mode="rgb_array"` avoids the problem, because the frames are returned as arrays and nothing is drawn.

**`FileNotFoundError: No romfiles found for game: SuperMarioBros-Nes`** appears when the game id is `SuperMarioBros-Nes`. The id must end in `-v0`: `SuperMarioBros-Nes-v0`. If the id is right and the error persists, check that the import step printed `Imported 1 games`.

**Nothing imported (`Imported 0 games`)** means the files in `roms` did not match any known release. Check that the file is a complete, unmodified copy of a known release. On the test machine, a random file renamed to `.nes` gave `Imported 0 games`, while the original zip archive of the ROM was accepted as it was.

## Checkpoint

- `pip install` finishes without errors.
- `python -m stable_retro.import roms` prints `Imported 1 games`.
- `hello_mario.py` prints a shape of `(224, 240, 3)` and a `MultiBinary(9)` action space.
- `walk_right.py` writes a video in which Mario moves right and meets the first Goomba.

Part 3 looks inside the environment: which numbers the game exposes, how to turn them into a distance, and how to wrap the emulator so that a training algorithm can use it.
