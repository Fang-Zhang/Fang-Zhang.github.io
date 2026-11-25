---
title: "Reinforcement Learning Engineering (5/6): Simulator Randomisation Showed No Measurable Gain on CartPole at 40 Seeds, While 10 Seeds Showed Up to +60"
date: 2027-06-28 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement learning, sim-to-real, domain randomization, robotics, recommender systems, scheduling, series]
description: "Part 5 of 6 on reinforcement learning engineering. How applications outside games get trained and deployed: simulation for legged robots, logged data for recommenders, learned schedulers. Plus a 40-seed test of simulator randomisation on CartPole."
---

## Three deployment patterns in the sources

Outside games, a reinforcement learning system meets a hard engineering constraint: the real world is slow, risky or both, so training happens somewhere else. The sources below show three answers.

- **Train in simulation, transfer to hardware.** [Hwangbo et al.](https://arxiv.org/abs/1901.08652) train a neural network policy in simulation and transfer it to the ANYmal quadruped, which then follows body velocity commands precisely and energy-efficiently, runs faster than before and recovers from falling. [Rudin et al.](https://arxiv.org/abs/2109.11978) make the simulation side cheap: massive parallelism on a single workstation GPU trains ANYmal policies for flat terrain in under four minutes and for uneven terrain in twenty (Part 1 measured why environment throughput is the lever). [OpenAI](https://arxiv.org/abs/1910.07113) report that models trained only in simulation solve a Rubik's cube with a humanoid robot hand, using automatic domain randomization, which "generates a distribution over randomized environments of ever-increasing difficulty". The earlier randomisation papers cited in Part 3 ([Tobin et al.](https://arxiv.org/abs/1703.06907), [Peng et al.](https://arxiv.org/abs/1710.06537)) set up this approach.
- **Learn from logs, correct for the logging policy.** [Chen et al.](https://arxiv.org/abs/1812.02353) describe a REINFORCE-based top-K recommender in production at YouTube, with an action space "on the orders of millions" of items. Learning from logged feedback is "subject to biases caused by only observing feedback on recommendations selected by the previous versions of the recommender", so the contributions are an off-policy correction, a top-K variant of it, and the value of exploration, shown "through a series of simulations and multiple live experiments". The offline failure in Part 3 is the small-scale version of the same problem.
- **Replace a hand-tuned heuristic.** [Decima](https://arxiv.org/abs/1810.01963) learns scheduling policies for Spark clusters. The authors state that off-the-shelf reinforcement learning "cannot handle the complexity and scale of the scheduling problem", so they built new representations of job dependency graphs, scalable models and training methods for continuous stochastic job arrivals. On a prototype with Spark on a 25-node cluster, average job completion time improved over hand-tuned heuristics by at least 21 percent, up to 2 times in periods of high load.

None of these results was reproduced here. The experiment below tests one claim that all the simulation-based work leans on: that randomising the simulator makes a policy robust to the gap between simulation and reality.

## Experiment: how large is the gap, and does randomisation close it?

Setup: PPO from Part 2 (baseline settings, 150,000 steps) on CartPole, trained on three simulator regimes. The simulator is changed by a wrapper that, at each episode start, draws multiplicative factors for actuator force, pole length and pole mass uniformly from 1 +- w, and optionally an action delay of 0 to 2 steps and observation noise with standard deviation up to 0.1. The "real system" is the same simulator at fixed shifted settings. Regimes: nominal only; dynamics +-50 percent; dynamics +-50 percent plus delay and noise. Scores are the greedy return over 20 episodes of each test condition.

- **First look, 10 seeds, dynamics only.** Nominal-only training scored 440.5 on the nominal system; training with +-20 percent randomisation scored 486.3 and with +-50 percent scored 490.0. The gain looked like +46 to +50, and similar on shifted systems (for example 416.6 against 467.4 and 460.6 with the actuator at half strength).
- **The same regime, drawn again, disagreed.** A second 10-seed run with the delay and noise wrapper (the +-50 percent dynamics regime is identical in both runs; only the random stream differs) scored 424.6 on the nominal system, against 490.0 in the first run for the same regime, a 65-point gap from noise alone. In that run the full regime appeared to gain up to +59.7 on the combined shift (440.8 against 381.1).
- **Forty seeds per arm settle it.** On the combined shift (actuator half strength, pole mass tripled, action delay of 2 steps, observation noise 0.1), nominal-only training scored 423.3 against 467.4 on the nominal system, a loss of 44 points (9 percent). Full randomisation scored 441.7, a gain of +18.4 with Welch p = 0.393. Across the six test conditions the gains were +10.0 to +22.0 with p between 0.189 and 0.607. The seed-to-seed standard deviation was 56 to 109 points.
- **Randomisation did not come free in training return.** Return during training (last 10 percent of steps) was 430.2 for nominal only, 432.2 for +-50 percent dynamics, and 403.4 for the full regime. This difference was not tested for significance.
- **What a k-seed experiment would have reported.** Resampling the 40 seeds on the combined shift, the measured gain has a 5th to 95th percentile range of -78.4 to +119.3 at 5 seeds per arm, -50.5 to +89.0 at 10, -29.6 to +67.8 at 20 and -16.8 to +53.3 at 40, with a median of about +17 to +18 throughout.

![Two panels. Left: grouped bars of greedy return for policies trained on three simulator regimes, tested under six shifted conditions, 40 seeds each; all bars are within a few tens of points and the error bars overlap. Right: the gain from full randomisation that a k-seed experiment would report, as a range from about minus 78 to plus 119 at 5 seeds narrowing to minus 17 to plus 53 at 40 seeds]({{ site.baseurl }}/assets/pic/rl-eng-5-randomisation.png){: width="900" }
_Figure: left, 40-seed means with 95 percent intervals (the vertical axis starts at 300); right, the range of gains a k-seed experiment would report. Measured here._

## Engineering reading

1. **Measure the gap before paying to close it.** Here the nominal-only policy lost at most 9 percent under a combined shift. An audit of this kind (evaluate the trained policy on deliberately shifted simulators) costs minutes and shows whether randomisation is needed at all.
2. **Randomise what the gap is made of.** The systems in the sources randomise rendering, dynamics or task difficulty because those are where the real system differs. Wider ranges have a price, visible here as a lower training return for the widest regime.
3. **Throughput makes randomisation affordable.** Training in minutes on one GPU, as in the locomotion work, allows many seeds and many ranges; with 10 seeds this experiment produced two contradictory stories.
4. **Logged-data systems need a correction for who collected the data.** The recommender paper builds its method around that bias; Part 3 showed what happens on a fixed dataset without it.
5. **A strong non-learning baseline defines the gain.** The scheduler's headline number is relative to hand-tuned heuristics, and the paper's own claim is bounded to a 25-node Spark prototype.

## Limits

- CartPole is a low-dimensional system whose controller has wide margins, which is why the gap is small. A legged robot or a hand differs in kind, and nothing here supports or refutes the cited results.
- The gaps tested are the ones built into this wrapper (scaled dynamics, delay, noise); real gaps include contact models, sensor calibration and unmodelled effects.
- The no-significant-difference result is a statement about power at 40 seeds: the interval at 40 seeds still includes gains up to about +53 and losses up to about -17.
- Evaluation uses the greedy policy and 20 episodes per condition, with return capped at 500.
- An earlier version of the delay wrapper had a bug (a deque with a maximum length dropped the wrong action, and every delayed policy scored 9.4). It was fixed before any table reported here, and the code below is the fixed version.
- The application results come from abstracts; none of these papers was read in full for this post.

## Code and output

Wrapper used in the first look (dynamics only) and its output (10 seeds):

```python
import json, time, numpy as np, torch, gymnasium as gym
from multiprocessing import Pool
from ppo import train
def set_dyn(env, f, l, m):
    u = env.unwrapped; u.force_mag = 10.0 * f; u.length = 0.5 * l; u.masspole = 0.1 * m
    u.total_mass = u.masspole + u.masscart; u.polemass_length = u.masspole * u.length
class Rand(gym.Wrapper):
    def __init__(s, env, w): super().__init__(env); s.w = w
    def reset(s, **kw):
        if s.w > 0: set_dyn(s.env, *np.random.uniform(1 - s.w, 1 + s.w, 3))
        return s.env.reset(**kw)
REGIMES = {"nominal only": 0.0, "randomised +-20%": 0.2, "randomised +-50%": 0.5}
TESTS = {"nominal": (1, 1, 1), "actuator x0.5": (0.5, 1, 1), "actuator x0.4": (0.4, 1, 1), "pole length x2": (1, 2, 1), "pole length x0.4": (1, 0.4, 1), "pole mass x3": (1, 1, 3), "actuator x0.6 + length x0.5": (0.6, 0.5, 1)}
def evaluate(ag, dyn, seed, n=20):
    env = gym.make("CartPole-v1"); out = []
    for i in range(n):
        o, _ = env.reset(seed=5000 + 100 * seed + i); set_dyn(env, *dyn); R = 0; d = False
        while not d:
            with torch.no_grad(): a = int(ag.actor(torch.as_tensor(o, dtype=torch.float32)).argmax())
            o, r, te, tr, _ = env.step(a); R += r; d = te or tr
        out.append(R)
    return float(np.mean(out))
def job(a):
    reg, seed = a; w = REGIMES[reg]; np.random.seed(seed)
    rets, ag = train("baseline", seed, make_env=lambda: Rand(gym.make("CartPole-v1"), w), return_agent=True)
    train_ret = float(np.mean([x for g, x in rets if g > 0.9 * 150_000]))
    return reg, seed, train_ret, {k: evaluate(ag, v, seed) for k, v in TESTS.items()}
if __name__ == "__main__":
    t = time.time()
    with Pool(14) as p: out = p.map(job, [(r, s) for r in REGIMES for s in range(10)], chunksize=1)
    json.dump(out, open("e5_dr.json", "w"))
    print("Greedy return (20 episodes per test, mean over 10 seeds; 500 = solved) after training on each regime")
    print(f"{'test dynamics':30s}" + "".join(f"{r:>22s}" for r in REGIMES))
    for k in TESTS:
        print(f"{k:30s}" + "".join(f"{np.mean([o[3][k] for o in out if o[0] == r]):22.1f}" for r in REGIMES))
    print(f"{'(training-time return)':30s}" + "".join(f"{np.mean([o[2] for o in out if o[0] == r]):22.1f}" for r in REGIMES))
    print("elapsed", time.time() - t)
```

```
Greedy return (20 episodes per test, mean over 10 seeds; 500 = solved) after training on each regime
test dynamics                           nominal only      randomised +-20%      randomised +-50%
nominal                                        440.5                 486.3                 490.0
actuator x0.5                                  416.6                 467.4                 460.6
actuator x0.4                                  415.5                 463.1                 458.6
pole length x2                                 422.4                 478.2                 477.2
pole length x0.4                               454.9                 472.1                 483.5
pole mass x3                                   433.6                 483.6                 487.7
actuator x0.6 + length x0.5                    419.9                 463.1                 467.0
(training-time return)                         380.6                 446.5                 447.3
elapsed 46.252893924713135
```

Wrapper with delay and noise (the version below ran with 40 seeds per arm):

```python
import json, time, numpy as np, torch, gymnasium as gym
from multiprocessing import Pool
from collections import deque
from ppo import train
from e5_dr import set_dyn
class Gap(gym.Wrapper):
    """Dynamics scales (force, pole length, pole mass), action delay in steps, observation noise."""
    def __init__(s, env, w=0.0, max_delay=0, max_noise=0.0, fixed=None):
        super().__init__(env); s.w, s.md, s.mn, s.fixed = w, max_delay, max_noise, fixed
    def reset(s, **kw):
        if s.fixed: f, l, m, s.delay, s.noise = s.fixed
        else:
            f, l, m = np.random.uniform(1 - s.w, 1 + s.w, 3) if s.w > 0 else (1, 1, 1)
            s.delay = int(np.random.randint(0, s.md + 1)); s.noise = float(np.random.uniform(0, s.mn))
        set_dyn(s.env, f, l, m); s.q = deque([0] * s.delay)
        o, i = s.env.reset(**kw); return o + s.noise * np.random.randn(4).astype(np.float32), i
    def step(s, a):
        if s.delay: s.q.append(int(a)); a = s.q.popleft()
        o, r, te, tr, i = s.env.step(a); return o + s.noise * np.random.randn(4).astype(np.float32), r, te, tr, i
REGIMES = {"nominal only": dict(), "dynamics +-50%": dict(w=0.5), "dynamics +-50% + delay 0-2 + noise": dict(w=0.5, max_delay=2, max_noise=0.1)}
TESTS = {"nominal": (1, 1, 1, 0, 0), "dynamics: actuator x0.5, pole mass x3": (0.5, 1, 3, 0, 0), "action delay 1 step": (1, 1, 1, 1, 0), "action delay 2 steps": (1, 1, 1, 2, 0), "observation noise 0.1": (1, 1, 1, 0, 0.1), "all: x0.5, mass x3, delay 2, noise 0.1": (0.5, 1, 3, 2, 0.1)}
def evaluate(ag, fixed, seed, n=20):
    env = Gap(gym.make("CartPole-v1"), fixed=fixed); out = []
    for i in range(n):
        o, _ = env.reset(seed=5000 + 100 * seed + i); R = 0; d = False
        while not d:
            with torch.no_grad(): a = int(ag.actor(torch.as_tensor(o, dtype=torch.float32)).argmax())
            o, r, te, tr, _ = env.step(a); R += r; d = te or tr
        out.append(R)
    return float(np.mean(out))
def job(a):
    reg, seed = a; kw = REGIMES[reg]; np.random.seed(seed)
    rets, ag = train("baseline", seed, make_env=lambda: Gap(gym.make("CartPole-v1"), **kw), return_agent=True)
    return reg, seed, float(np.mean([x for g, x in rets if g > 0.9 * 150_000])), {k: evaluate(ag, v, seed) for k, v in TESTS.items()}
if __name__ == "__main__":
    t = time.time()
    with Pool(14) as p: out = p.map(job, [(r, s) for r in REGIMES for s in range(40)], chunksize=1)
    json.dump(out, open("e5_dr2.json", "w"))
    print("Greedy return (20 episodes per test, mean over 40 seeds; 500 = solved) after training on each regime")
    print(f"{'test condition':42s}" + "".join(f"{r:>38s}" for r in REGIMES))
    for k in TESTS:
        print(f"{k:42s}" + "".join(f"{np.mean([o[3][k] for o in out if o[0] == r]):38.1f}" for r in REGIMES))
    print(f"{'(return during training)':42s}" + "".join(f"{np.mean([o[2] for o in out if o[0] == r]):38.1f}" for r in REGIMES))
    from scipy import stats
    print("\nDifference (full randomisation minus nominal only), per test: mean difference, Welch p, seed-to-seed sd of each arm")
    for k in TESTS:
        a = np.array([o[3][k] for o in out if o[0] == "nominal only"]); b = np.array([o[3][k] for o in out if o[0] == list(REGIMES)[2]])
        print(f"{k:42s} {b.mean() - a.mean():+6.1f}  p = {stats.ttest_ind(b, a, equal_var=False)[1]:.3f}   sd {a.std(ddof=1):5.1f} vs {b.std(ddof=1):5.1f}")
    print("elapsed", time.time() - t)
```

Output of the 40-seed run:

```
Greedy return (20 episodes per test, mean over 40 seeds; 500 = solved) after training on each regime
test condition                                                      nominal only                        dynamics +-50%    dynamics +-50% + delay 0-2 + noise
nominal                                                                    467.4                                 469.3                                 485.6
dynamics: actuator x0.5, pole mass x3                                      452.2                                 465.4                                 473.7
action delay 1 step                                                        454.1                                 467.8                                 476.1
action delay 2 steps                                                       440.2                                 459.3                                 454.0
observation noise 0.1                                                      445.4                                 457.0                                 455.4
all: x0.5, mass x3, delay 2, noise 0.1                                     423.3                                 434.9                                 441.7
(return during training)                                                   430.2                                 432.2                                 403.4

Difference (full randomisation minus nominal only), per test: mean difference, Welch p, seed-to-seed sd of each arm
nominal                                     +18.2  p = 0.189   sd  66.6 vs  55.6
dynamics: actuator x0.5, pole mass x3       +21.5  p = 0.244   sd  95.0 vs  65.9
action delay 1 step                         +22.0  p = 0.208   sd  90.1 vs  62.2
action delay 2 steps                        +13.8  p = 0.526   sd 109.3 vs  82.9
observation noise 0.1                       +10.0  p = 0.607   sd  94.9 vs  77.2
all: x0.5, mass x3, delay 2, noise 0.1      +18.4  p = 0.393   sd 105.6 vs  84.7
elapsed 160.1650779247284
```

Output of the 10-seed run of the same script (before changing the seed count to 40):

```
Greedy return (20 episodes per test, mean over 10 seeds; 500 = solved) after training on each regime
test condition                                                      nominal only                        dynamics +-50%    dynamics +-50% + delay 0-2 + noise
nominal                                                                    440.5                                 424.6                                 497.6
dynamics: actuator x0.5, pole mass x3                                      415.3                                 445.1                                 480.5
action delay 1 step                                                        418.8                                 436.3                                 483.5
action delay 2 steps                                                       395.5                                 439.4                                 456.1
observation noise 0.1                                                      406.7                                 430.2                                 459.7
all: x0.5, mass x3, delay 2, noise 0.1                                     381.1                                 394.4                                 440.8
(return during training)                                                   380.6                                 404.7                                 403.9
```

Figure script output (k-seed resampling: seeds per arm, 5th, 50th, 95th percentile of the gain):

```
[(5, np.float64(-78.4), np.float64(16.5), np.float64(119.3)), (10, np.float64(-50.5), np.float64(18.1), np.float64(89.0)), (20, np.float64(-29.6), np.float64(17.4), np.float64(67.8)), (40, np.float64(-16.8), np.float64(18.1), np.float64(53.3))]
```

Next: [Part 6](/posts/rl-engineering-6-llm-infrastructure/) covers reinforcement learning in language models from the infrastructure side: generation cost, reward models, and why training and generation are separated.

## References

- [Learning agile and dynamic motor skills for legged robots](https://arxiv.org/abs/1901.08652), 2019.
- [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978), 2021.
- [Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113), 2019.
- [Top-K Off-Policy Correction for a REINFORCE Recommender System](https://arxiv.org/abs/1812.02353), 2018.
- [Learning Scheduling Algorithms for Data Processing Clusters](https://arxiv.org/abs/1810.01963), 2018.
- [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907), 2017.
- [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537), 2017.
