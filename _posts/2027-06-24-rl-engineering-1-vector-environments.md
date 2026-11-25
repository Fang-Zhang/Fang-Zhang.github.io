---
title: "Reinforcement Learning Engineering (1/6): Parallel Environments Lost to a Plain Loop at 16 Microseconds a Step and Won 11-Fold at 1 Millisecond"
date: 2027-06-24 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement learning, engineering, gymnasium, vectorised environments, throughput, series]
description: "Part 1 of 6 on reinforcement learning engineering. A CPU benchmark of Gymnasium's synchronous and asynchronous vector environments across three per-step costs, plus the autoreset behaviour that silently changes what a training loop sees."
---

## What this post finds

Every reinforcement learning system repeats one loop: the policy picks actions, the environment steps, and the learner updates from the collected data. Which of the three dominates the wall-clock time depends on the environment. The [EnvPool paper](https://arxiv.org/abs/2206.10558) describes parallel environment execution as "often the slowest part of the whole system", and the [IMPALA paper](https://arxiv.org/abs/1802.01561) separates acting from learning for the same reason: to use hardware efficiently. This post measures the first half of that statement on one machine.

Setup: an AMD Ryzen 9 5950X (16 cores, 32 threads), Python 3.14, Gymnasium 1.4.0, PyTorch 2.15 on CPU with one thread. CartPole-v1 is the environment. To stand in for slower simulators, a wrapper adds a fixed busy-wait of 0, 100 or 1,000 microseconds to every step. Throughput is environment steps per second over 3 seconds, with random actions.

- **A plain synchronous loop beats parallel processes for a cheap environment.** On unmodified CartPole, `SyncVectorEnv` with 32 environments reached 169,248 steps per second. `AsyncVectorEnv` with 32 environments reached 26,583, a sixth of that. Every asynchronous step pays for inter-process communication that a 16-microsecond step cannot hide. (The raw CartPole step alone measured 5.2 microseconds in a bare loop, so the vector wrapper itself adds the rest.)
- **At 100 microseconds per step the picture flips only at four environments.** Synchronous throughput stays near 9,500 steps per second at any count, since one process does all the work. Asynchronous throughput grows from 3,954 (1 env) to 25,299 (32 envs). It is below synchronous at 1 and 2 environments and above it from 4 environments on (12,357 against 9,390).
- **At 1 millisecond per step parallelism is the whole story.** Synchronous stays at about 1,030 steps per second. Asynchronous reaches 11,105 at 32 environments, an 11-fold gain on a 32-thread machine, with 9,882 at 16 environments and 5,757 at 8. The curve flattens beyond 16 environments, which matches 16 physical cores.
- **The policy network adds a second reason to batch.** With a 64x64 policy in the loop, synchronous throughput rose from 12,239 steps per second (1 env) to 125,316 (32 envs), because one batched forward pass serves all environments. Asynchronous vector environments still pay for the message passing: 24,907 at 32 envs.
- **Autoreset is a correctness setting, not a speed setting.** When an episode ends, Gymnasium's default `next-step` mode resets the sub-environment on the following step, and that step returns reward 0 with the new observation. `same-step` mode resets immediately and hands over the last observation in `info["final_obs"]`. Training code written for one mode and run on the other mislabels transitions.

![Two panels. Left: environment steps per second against the number of environments for three per-step costs, synchronous (dashed) and asynchronous (solid) vector environments; for plain CartPole the synchronous loop reaches 169 thousand steps per second at 32 environments against 27 thousand asynchronous, while with 1,000 microseconds added per step the asynchronous version reaches 11 thousand against 1 thousand. Right: with a small policy network in the loop, synchronous CartPole reaches 125 thousand steps per second at 32 environments and asynchronous 25 thousand]({{ site.baseurl }}/assets/pic/rl-eng-1-vector-throughput.png){: width="900" }
_Figure: left, steps per second against environment count for three per-step costs (dashed: synchronous, solid: asynchronous); right, CartPole with a policy network in the loop. Measured here, log scale on the left._

## What to do with this

1. **Measure the environment before choosing a vectoriser.** A step cost below roughly 100 microseconds argues for synchronous vectorisation, or for a purpose-built parallel engine such as EnvPool, which its paper reports at 2.8 times the speed of Python subprocesses on a laptop. A step cost near a millisecond (heavier simulation or rendering) argues for processes or threads.
2. **Batch the policy even when the environment cannot be sped up.** The synchronous rows show the gain from one forward pass per batch instead of one per environment.
3. **Write down the autoreset mode.** The Gymnasium [vector API](https://gymnasium.farama.org/api/vector/) exposes it as `metadata["autoreset_mode"]`, and the Farama [deep dive](https://farama.org/Vector-Autoreset-Mode) states that the modes differ in the number of step calls per episode, so training code has to be written for one specific mode. The listing below shows the difference on the same environment.
4. **Parallelism is capped by cores, not by the environment count.** Past the physical core count the asynchronous curves flatten.

## Limits

- This is one machine, one environment family and a synthetic cost: a busy-wait loop in Python is not a physics engine, and real simulators vary in memory use, GIL release and start-up time.
- Random actions are used for the environment tests, so no learner time is included. The policy test uses a small network with one thread; a GPU policy or a learner update changes the balance and is not measured here.
- Each configuration ran once for 3 seconds. Differences of a few percent, for example between 16 and 32 synchronous environments, are inside the noise of one run.
- The 16 physical cores claim rests on the CPU model (Ryzen 9 5950X, 16 cores and 32 threads per AMD's specification); the benchmark does not isolate physical cores.
- EnvPool and IMPALA are cited for what their abstracts state; neither system was run here.

## Code and output

Throughput benchmark:

```python
import time, json, sys, os
import numpy as np, gymnasium as gym
import torch
torch.set_num_threads(1)

class Busy(gym.Wrapper):
    """Adds a fixed CPU-bound cost to every step, standing in for a simulator that is slower than CartPole."""
    def __init__(self, env, micros):
        super().__init__(env); self.micros = micros
    def step(self, a):
        out = self.env.step(a)
        end = time.perf_counter() + self.micros * 1e-6
        while time.perf_counter() < end: pass
        return out

def make(micros):
    def f():
        e = gym.make("CartPole-v1")
        return Busy(e, micros) if micros else e
    return f

def run(mode, n, micros, seconds=3.0, policy=False):
    fns = [make(micros) for _ in range(n)]
    envs = gym.vector.AsyncVectorEnv(fns) if mode == "async" else gym.vector.SyncVectorEnv(fns)
    net = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2))
    obs, _ = envs.reset(seed=0)
    for _ in range(20):  # warm-up
        envs.step(envs.action_space.sample())
    steps = 0; t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if policy:
            with torch.no_grad(): a = net(torch.as_tensor(obs, dtype=torch.float32)).argmax(-1).numpy()
        else: a = envs.action_space.sample()
        obs, *_ = envs.step(a); steps += n
    dt = time.perf_counter() - t0; envs.close()
    return steps / dt

if __name__ == "__main__":
    res = {}
    for micros in (0, 100, 1000):
        for mode in ("sync", "async"):
            for n in (1, 2, 4, 8, 16, 32):
                v = run(mode, n, micros)
                res[f"{micros}|{mode}|{n}"] = v
                print(f"env cost +{micros:4d} us | {mode:5s} | {n:2d} envs | {v:10.0f} steps/s", flush=True)
    # with a policy network in the loop (batched forward, 1 thread)
    for mode in ("sync", "async"):
        for n in (1, 8, 32):
            v = run(mode, n, 0, policy=True); res[f"policy|{mode}|{n}"] = v
            print(f"policy in loop | CartPole | {mode:5s} | {n:2d} envs | {v:10.0f} steps/s", flush=True)
    json.dump(res, open("e1_res.json", "w"))
```

Output:

```
env cost +   0 us | sync  |  1 envs |      60770 steps/s
env cost +   0 us | sync  |  2 envs |      87572 steps/s
env cost +   0 us | sync  |  4 envs |     116742 steps/s
env cost +   0 us | sync  |  8 envs |     137271 steps/s
env cost +   0 us | sync  | 16 envs |     160544 steps/s
env cost +   0 us | sync  | 32 envs |     169248 steps/s
env cost +   0 us | async |  1 envs |       6927 steps/s
env cost +   0 us | async |  2 envs |      12745 steps/s
env cost +   0 us | async |  4 envs |      18426 steps/s
env cost +   0 us | async |  8 envs |      24418 steps/s
env cost +   0 us | async | 16 envs |      26141 steps/s
env cost +   0 us | async | 32 envs |      26583 steps/s
env cost + 100 us | sync  |  1 envs |       8678 steps/s
env cost + 100 us | sync  |  2 envs |       9128 steps/s
env cost + 100 us | sync  |  4 envs |       9390 steps/s
env cost + 100 us | sync  |  8 envs |       9568 steps/s
env cost + 100 us | sync  | 16 envs |       9642 steps/s
env cost + 100 us | sync  | 32 envs |       9728 steps/s
env cost + 100 us | async |  1 envs |       3954 steps/s
env cost + 100 us | async |  2 envs |       7404 steps/s
env cost + 100 us | async |  4 envs |      12357 steps/s
env cost + 100 us | async |  8 envs |      17816 steps/s
env cost + 100 us | async | 16 envs |      23567 steps/s
env cost + 100 us | async | 32 envs |      25299 steps/s
env cost +1000 us | sync  |  1 envs |       1012 steps/s
env cost +1000 us | sync  |  2 envs |       1027 steps/s
env cost +1000 us | sync  |  4 envs |       1022 steps/s
env cost +1000 us | sync  |  8 envs |       1027 steps/s
env cost +1000 us | sync  | 16 envs |       1026 steps/s
env cost +1000 us | sync  | 32 envs |       1030 steps/s
env cost +1000 us | async |  1 envs |        891 steps/s
env cost +1000 us | async |  2 envs |       1685 steps/s
env cost +1000 us | async |  4 envs |       3197 steps/s
env cost +1000 us | async |  8 envs |       5757 steps/s
env cost +1000 us | async | 16 envs |       9882 steps/s
env cost +1000 us | async | 32 envs |      11105 steps/s
policy in loop | CartPole | sync  |  1 envs |      12239 steps/s
policy in loop | CartPole | sync  |  8 envs |      69216 steps/s
policy in loop | CartPole | sync  | 32 envs |     125316 steps/s
policy in loop | CartPole | async |  1 envs |       5105 steps/s
policy in loop | CartPole | async |  8 envs |      20613 steps/s
policy in loop | CartPole | async | 32 envs |      24907 steps/s
```

Autoreset check on the same environment, always pushing left so the episode ends at step 11:

```python
import gymnasium as gym, numpy as np
for name, mode in (("NEXT_STEP", gym.vector.AutoresetMode.NEXT_STEP), ("SAME_STEP", gym.vector.AutoresetMode.SAME_STEP)):
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")], autoreset_mode=mode)
    obs, _ = envs.reset(seed=0)
    print(f"--- {name} (metadata autoreset_mode = {envs.metadata['autoreset_mode']})")
    ended = False
    for t in range(1, 40):
        obs, r, term, trunc, info = envs.step(np.zeros(1, dtype=int))  # always push left, so the episode ends quickly
        if ended or term[0] or trunc[0]:
            print(f"step {t:2d}: reward {r[0]:.0f}  terminated {bool(term[0])!s:5}  obs[0] {obs[0][0]: .3f}  info keys {sorted(info.keys())}")
        if ended: break
        ended = bool(term[0] or trunc[0])
```

Output:

```
--- NEXT_STEP (metadata autoreset_mode = AutoresetMode.NEXT_STEP)
step 11: reward 1  terminated True   obs[0] -0.206  info keys []
step 12: reward 0  terminated False  obs[0]  0.031  info keys []
--- SAME_STEP (metadata autoreset_mode = AutoresetMode.SAME_STEP)
step 11: reward 1  terminated True   obs[0]  0.031  info keys ['_final_info', '_final_obs', 'final_info', 'final_obs']
step 12: reward 1  terminated False  obs[0]  0.032  info keys []
```

In `NEXT_STEP` mode the terminal step is followed by a step with reward 0 and the reset observation, so one episode costs 12 step calls and the last one carries no learning signal. In `SAME_STEP` mode the terminal step already returns the reset observation, and the true final observation sits in `info["final_obs"]`.

Next: Part 2 looks at what keeps policy-gradient training stable, by removing implementation details one at a time across random seeds.

## References

- [EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine](https://arxiv.org/abs/2206.10558), NeurIPS 2022.
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561), 2018.
- [Gymnasium vector environments](https://gymnasium.farama.org/api/vector/) and [Deep Dive: Autoreset Modes in Gymnasium v1.1 Vector Environments](https://farama.org/Vector-Autoreset-Mode).
