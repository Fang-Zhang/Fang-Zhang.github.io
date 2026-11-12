---
title: "AI Plays Super Mario, Part 7: Training the Level 1 Agent"
date: 2027-01-13 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement-learning, ppo, games, super-mario, tutorial, series]
description: "Part 7 of the AI Plays Super Mario series. The full PPO training script, how to evaluate it, and what five seeds of the same setup produced: the level gets cleared in all of them, but how often depends on the seed, the checkpoint and the way the agent picks its button."
---

This post trains the agent. It gives the complete PPO training script, the evaluation script, and the results of running the same setup with five different random seeds, because one run turned out not to be enough to say what a reader will get. The answer is that every run learns to clear level 1 most of the time when it samples its buttons, and that the stricter test, always pressing the single most likely button, passes for some seeds and fails completely for others. The post also traces one such failure to its cause, on a staircase near the end of the level.

## What the training does

The pieces come from earlier parts. The environment is the `Mario` class from [Part 3](/posts/ai-plays-super-mario-3-the-environment/) with the progress reward from [Part 4](/posts/ai-plays-super-mario-4-reward-design/): the change in Mario's x position times 0.1, a penalty of 0.01 per decision, minus 5 for dying and plus 10 for clearing the level. The algorithm is PPO from [Part 5](/posts/ai-plays-super-mario-5-dqn-and-ppo/), run through Stable-Baselines3 with a convolutional policy. Sixteen copies of the game run in parallel, each decision holds the buttons for 4 frames, and the policy sees the last four 84×84 gray frames.

Save this as `train.py`. The fourth argument is the random seed; it fixes the network's starting weights and the sampling inside PPO:

```python
import sys, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from mario_env import Mario

N_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
OUT = sys.argv[3] if len(sys.argv) > 3 else "ppo_mario"
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0


class Log(BaseCallback):
    def __init__(self):
        super().__init__()
        self.t0 = time.time()
        self.bests, self.clears = [], []

    def _on_step(self):
        for i, d in enumerate(self.locals["dones"]):
            if d:
                info = self.locals["infos"][i]
                self.bests.append(info.get("best_x", 0))
                self.clears.append(bool(info.get("cleared", False)))
        if self.num_timesteps % 20480 < N_ENVS and self.bests:
            b, c = self.bests[-50:], self.clears[-50:]
            print(f"steps={self.num_timesteps} t={time.time()-self.t0:.0f}s "
                  f"eps={len(self.bests)} mean_best_x={np.mean(b):.0f} max_x={max(b)} clears={sum(c)}/{len(c)}", flush=True)
        if self.num_timesteps % 491520 < N_ENVS * 128:
            self.model.save(f"{OUT}_{self.num_timesteps//1000}k")
        return True


if __name__ == "__main__":
    env = VecFrameStack(VecMonitor(SubprocVecEnv([Mario for _ in range(N_ENVS)])), 4, channels_order="last")
    model = PPO("CnnPolicy", env, n_steps=128, batch_size=256, n_epochs=4, learning_rate=2.5e-4,
                gamma=0.99, ent_coef=0.01, clip_range=0.1, device="cpu", seed=SEED, verbose=0)
    model.learn(TOTAL, callback=Log())
    model.save(OUT)
    print("saved", OUT, flush=True)
```

The settings are the ones listed in Part 5: 128 steps per environment per update, batches of 256, 4 passes over each batch, learning rate 0.00025, discount 0.99, entropy bonus 0.01 and clip range 0.1. The `Log` callback prints one line per 20,480 steps with the average of the furthest x reached over the last 50 finished episodes and how many of those 50 cleared the level, and saves a checkpoint about every 491,520 steps.

### Checkpoint

A full run takes hours, so first check that the script works with a short one:

```
python train.py 16 61440 ppo_test 0
```

On the machine used here this printed three log lines and saved `ppo_test.zip` after 4 minutes 27 seconds in total:

```
steps=20480 t=81s eps=31 mean_best_x=653 max_x=1540 clears=0/31
steps=40960 t=169s eps=91 mean_best_x=838 max_x=1944 clears=0/50
steps=61440 t=255s eps=168 mean_best_x=846 max_x=1677 clears=0/50
```

Your numbers will differ in detail because the 16 processes do not run in lockstep, but the average furthest x should be a few hundred to about a thousand and no episode should be a clear yet. The full run is `python train.py 16 3000000 ppo_mario 0`. The first run of this setup (seed 0) took 5,484 seconds, about 1.5 hours, on a CPU with no GPU, with nothing else running. The training runs below used the same code, with the environment file kept in my working folder; Part 3's `mario_env.py` is a rewritten, tidier copy of it with the same actions and reward, and only the 61,440-step trial above was run with Part 3's copy.

## When the level starts being cleared

The log of each run shows the same shape. Almost nothing is cleared for the first 1.5 million steps, then the clear count rises quickly. For seed 0 the last-50 count was 0 of 50 at 983,040 steps, 7 of 50 at 1,966,080, 43 of 50 at 2,457,600 and 48 of 50 at 2,949,120.

![Five line curves of clears among the last 50 training episodes against training steps, one per seed, all rising from zero between about 2 and 3 million steps]({{ '/assets/pic/mario-ai-seeds-curve.png' | relative_url }})
*Clears among the last 50 training episodes, for five seeds of the identical setup (seed 0 is the first run, seeds 1 to 4 were added later). Single run per seed. The curves rise between 2.0 and 3.0 million steps, at slightly different times: the first log line with at least 25 of 50 comes at 2.40 million steps for seed 0, 2.36 for seed 1, 2.23 for seed 2, 2.38 for seed 3 and 2.64 for seed 4.*

Seeds 1 to 4 were trained two at a time on the same machine, which took about 4 hours each, so the times are not comparable with the 1.5 hours of seed 0. By the end of training the last-50 count was 49, 43, 45, 44 and 37 of 50 for seeds 0 to 4.

## Evaluating

The training log counts episodes in which the agent was still exploring. To measure the finished agent, play it without learning. Save this as `evaluate.py`:

```python
import sys, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from mario_env import Mario

path = sys.argv[1]; n_ep = int(sys.argv[2]); det = sys.argv[3] == "det"
env = VecFrameStack(DummyVecEnv([Mario]), 4, channels_order="last")
model = PPO.load(path, device="cpu")
res = []
for ep in range(n_ep):
    obs = env.reset(); done = False
    while not done:
        a, _ = model.predict(obs, deterministic=det)
        obs, r, d, info = env.step(a); done = d[0]
    res.append((info[0]["best_x"], info[0]["cleared"]))
print(path, "det" if det else "stochastic", "clears", sum(c for _, c in res), "/", n_ep, "mean_best_x", round(np.mean([x for x, _ in res])))
```

It plays 20 episodes with one of two ways of choosing the button.

`stochastic` samples from the probabilities the network outputs, which is how it was trained. `det` always presses the most probable button. Run both for the saved model:

```
python evaluate.py ppo_mario 20 sto
python evaluate.py ppo_mario 20 det
```

(Any third argument other than `det` samples.) The output is one line each, for example `ppo_mario stochastic clears 20 / 20 mean_best_x 3266`. The level ends at x = 3,266.

## Five seeds

Final models, 20 episodes per cell, single run per seed:

| seed | sampled | most probable button | last-50 clears in training |
|---|---|---|---|
| 0 | 20 / 20 | 0 / 20 | 49 / 50 |
| 1 | 15 / 20 | 20 / 20 | 43 / 50 |
| 2 | 20 / 20 | 0 / 20 | 45 / 50 |
| 3 | 15 / 20 | 20 / 20 | 44 / 50 |
| 4 | 16 / 20 | 0 / 20 | 37 / 50 |

Reading the table:

- **Sampling clears the level most of the time in every seed:** 86 of 100 episodes in total, between 15 and 20 of 20 per seed. Seeds 0 and 2 clear it every time; seeds 1, 3 and 4 do not.
- **The most probable button splits the seeds into two groups.** Seeds 1 and 3 clear 20 of 20, and seeds 0, 2 and 4 clear 0 of 20. In these cells the 20 episodes ended identically (all clears or all stuck), which suggests the game and the choice of button leave little room for variation.
- **The two modes can disagree in either direction.** For seeds 1 and 3 the most probable button beats sampling, for seeds 0, 2 and 4 it loses completely.

Checkpoints saved during training show that the final model is not a safe summary either. Sampled clears out of 20 at each checkpoint, one entry per seed 0 to 4:

- 491k and 983k steps: 0, 0, 0, 0, 0 at both
- 1.47M: 0, 0, 0, 1, 0
- 1.97M: 1, 3, 2, 0, 0
- 2.46M: 16, 11, 18, 12, 8
- 2.95M: 20, 16, 16, 17, 17
- final (3.0M): 20, 15, 20, 15, 16

With the most probable button, every checkpoint of every seed clears 0 of 20 except three entries: seed 3 at 2.95M (20 of 20), and the final models of seeds 1 and 3 (20 of 20). The sampled results rise steadily, the most-probable-button results do not.

## One failure, traced

Seed 0's final model clears 20 of 20 when sampling and 0 of 20 with the most probable button, always ending at x = 2,978. Playing it step by step shows why: Mario stops part-way up the large staircase just before the flagpole, and the most probable button, run right plus jump, does not move him.

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-argmax-stuck-on-stairs.mp4' | relative_url }}" type="video/mp4">
</video>

*Seed 0's model pressing its most probable button until it stops at x = 2,978 on the staircase, then the same state continued with one release and three RIGHT+A presses, repeated. The second half clears the level.*

The test that localises it uses the emulator's save and restore from [Part 6](/posts/ai-plays-super-mario-6-learning-from-a-demonstrator/). Play the model to the stuck moment, save the state, and from that same state try different button sequences. Save this as `stairs.py` and run `python stairs.py ppo_mario`:

```python
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from mario_env import Mario

env = VecFrameStack(DummyVecEnv([Mario]), 4, channels_order="last")
m = env.venv.envs[0]                      # the Mario object inside the wrapper
model = PPO.load(sys.argv[1], device="cpu"); obs = env.reset()

for t in range(450):                      # play 450 decisions, always pressing the most probable key
    a, _ = model.predict(obs, deterministic=True)
    obs, _, done, info = env.step(a)
state = m.env.em.get_state()              # save the emulator state at the stuck moment
x_here = m.last_x
print("stuck at x =", x_here, "| most probable key now:", int(a[0]))

def try_sequence(label, seq):
    m.env.em.set_state(state); best = x_here; cleared = False
    for key in seq:
        _, _, term, _, info = m.step(key)
        best = max(best, m.last_x); cleared = cleared or info["cleared"]
        if term: break
    print(f"{label:34s} furthest x {best}  cleared {cleared}")

try_sequence("hold RIGHT+A (key 2) 40 steps", [2] * 40)
try_sequence("hold RIGHT+A+B (key 4) 40 steps", [4] * 40)
try_sequence("6 no-ops, then RIGHT+A 20 steps", [0] * 6 + [2] * 20)
try_sequence("1 RIGHT, 3 RIGHT+A, repeated", ([1] + [2] * 3) * 150)
```

Output:

```
stuck at x = 2978 | most probable key now: 2
hold RIGHT+A (key 2) 40 steps      furthest x 2978  cleared False
hold RIGHT+A+B (key 4) 40 steps    furthest x 2978  cleared False
6 no-ops, then RIGHT+A 20 steps    furthest x 3068  cleared False
1 RIGHT, 3 RIGHT+A, repeated       furthest x 3266  cleared True
```

Holding RIGHT+A, with or without B, does not move Mario at all. Letting go of A for one decision and then pressing it again does, and repeating that pattern reaches the flag. The pattern fits a rule of the game that a new jump needs a fresh press of the jump button. This test shows which sequences work from this state, not the rule itself, and it was done for this one model only; whether the stalls of seeds 2 and 4 have the same cause was not checked. A model that samples sometimes picks a button without A by chance, which would release it; this is the likely reason sampling clears the level and the most probable button does not here, and it was not tested separately.

An earlier attempt at this test, which released A every few steps without first fixing the game state, was inconclusive. The saved state is what makes the comparison fair: the only thing that differs between the four lines is the buttons.

## What to expect

A reader following the series with the same code should expect an agent that clears level 1 most of the time when it samples its buttons, and not an agent that clears it every time. In five runs, the final sampled clear count was 15 to 20 of 20, and the training took between 1.5 and 4 hours on a CPU with no GPU. A few practical points follow from the tables.

- **Evaluate both ways of acting.** The same trained network scored 20 of 20 and 0 of 20 depending on the mode.
- **Evaluate more than the last checkpoint.** Between neighbouring checkpoints the results moved from 0 to 20 of 20 and back.
- **Expect no clears for the first 1.5 million steps.** That is normal and not a sign of a bug: in all five runs the first clears appear after 1.5 million steps.
- **Expect that two readers with the same code get different results.** Seeds 0 to 4 are the same setup and ended between 37 and 49 of 50 in training.

These are five runs with 20 evaluation episodes each, so the numbers describe these runs and not a general clear rate. They also come from the training copy of the environment, not from a rerun with Part 3's rewritten file.

Part 8 moves from training to language models and asks how a model that was never trained on Mario can still be made to play it.
