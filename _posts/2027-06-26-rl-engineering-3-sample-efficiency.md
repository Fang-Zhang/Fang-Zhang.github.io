---
title: "Reinforcement Learning Engineering (3/6): Four Times the Updates per Step Did Not Help DQN, Resets Did, and Near-Expert Data Collapsed Offline Learning to One Action"
date: 2027-06-26 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement learning, engineering, dqn, replay buffer, offline reinforcement learning, series]
description: "Part 3 of 6 on reinforcement learning engineering. Two small DQN experiments on CartPole: how many gradient updates per environment step pay off, and what a fixed offline dataset can and cannot teach."
---

## The two levers

Environment steps are the expensive resource in many reinforcement learning projects, so the engineering question is how much learning can be extracted from each one. Two levers are common. The first is to reuse stored transitions more often (a higher **update-to-data ratio**, also called the replay ratio: gradient updates per environment step). The second is to skip collection and learn from a fixed dataset (**offline** learning). [D'Oro et al.](https://iclr.cc/virtual/2023/poster/11457) describe raising the replay ratio as "an appealing strategy" for sample efficiency, and report that fully or partially resetting an agent's parameters is what makes higher ratios scale. The [offline reinforcement learning review by Levine et al.](https://arxiv.org/abs/2005.01643) describes learning from previously collected data without additional online collection, and says the limitations of current algorithms make this difficult. This post measures both on CartPole with a plain DQN written from scratch.

## Findings

Online DQN (40,000 environment steps, 10 seeds per arm, greedy evaluation over 10 episodes every 2,000 steps):

- **More updates per step did not help.** Averaged over the last five checkpoints (32,000 to 40,000 steps), return was 170.2 at 0.25 updates per step, 271.8 at 1 update per step, and 165.2 at 4 updates per step. The intervals are wide (bootstrap 95 percent interval for 1 update per step: 180.1 to 367.0), and the differences against 1 update per step have Welch p = 0.091 and 0.079. These are suggestive, not conclusive.
- **Four updates per step cost four times the gradient updates (156,004 against 39,001) and no seed ended at 195 or above** (0 of 10, against 4 of 10 at 1 update per step).
- **Resetting the network every 10,000 steps rescued the high ratio.** Same 4 updates per step, full reset of the Q-network, target network and optimiser at 10,000, 20,000 and 30,000 steps with the replay buffer kept: last-five-checkpoint return 302.3 (interval 263.1 to 336.6), 5 of 10 seeds ending at 195 or above, and Welch p < 0.001 against the same ratio without resets.
- **The reset arm pays for it visibly.** Evaluations taken immediately after a reset score 17.7, 15.6 and 11.7 on average (steps 10,000, 20,000, 30,000), and 158.8, 149.2 and 230.9 two thousand steps later (steps 12,000, 22,000, 32,000). Counting every checkpoint, the four arms are close (172.3 to 197.3), so the advantage of resets shows late in training, and the reset arm's last-five window starts after a reset.

Offline DQN (30,000 fixed transitions, 20,000 updates, no environment interaction, 10 seeds per dataset). The data were collected by a hand-written controller that pushes toward the side the pole leans (it scores 491.7 over 20 episodes), with a fraction of actions replaced by random ones:

- **Random data (return 22.4) produced a policy scoring 106.8 (sd 64.8).** Learning beat the data-collecting policy about fivefold.
- **Half random actions (data return 115.5) produced 457.6 (sd 76.4).** The best offline result, and about four times the return of the policy that collected the data.
- **Less random data gave worse results, not better ones.** 20 percent random actions (data return 320.4) gave a learned policy scoring 9.6, and 5 percent random (data return 461.4) gave 9.3. A CartPole policy that always pushes the same way scores 9.3 to 9.4 here.
- **A single-seed diagnostic of the 5 percent case shows why.** On the dataset's own states the greedy policy picked action 1 in 100 percent of cases, and the Q-values reached about 380,000 to 390,000 on average (maximum 624,416), while the true value with reward 1 per step and discount 0.99 cannot exceed 100. Only 2.5 percent of dataset actions differed from the controller's choice, so the network almost never saw the outcome of the other action and its value estimates diverged. This is one seed and one dataset, and the 20 percent case was not inspected.

![Two panels. Left: greedy evaluation return of online DQN on CartPole against environment steps for four arms, with 10 seeds each; the reset arm drops to near zero right after each reset and then climbs above the others. Right: bars comparing the return of the policy that collected an offline dataset with the return of an offline DQN trained on it; the two datasets with the least random actions give an offline policy that scores about 10]({{ site.baseurl }}/assets/pic/rl-eng-3-sample-efficiency.png){: width="900" }
_Figure: left, online DQN at different update-to-data ratios, mean of 10 seeds with standard-error band; right, offline DQN against the policy that collected its data. Measured here._

## Engineering reading

1. **Count updates, not just steps.** The cost of a run is environment steps plus gradient updates, and the second grows linearly with the ratio. In this experiment 4 times the updates bought nothing without resets.
2. **A high ratio needs a mechanism, not just a bigger number.** The reset arm matches the pattern in the paper above, where resets are what let higher ratios help. Whether the cause is lost plasticity (a network that has trained too long on too little data can no longer fit new data) was not tested here.
3. **Offline data need coverage of the actions that were not taken.** Near-expert data looked ideal and failed completely, while noisier data worked. Methods built for offline learning add constraints so that the learned policy stays near the data; this plain DQN has none.
4. **Check Q-values before trusting a curve.** A value estimate far above the maximum possible return, as here, is a cheap diagnostic that needs one line of code.
5. **Simulation randomisation is the third lever, and works on a different axis.** It adds variety instead of reusing data. [Tobin et al.](https://arxiv.org/abs/1703.06907) randomise rendering in the simulator so that "the real world may appear to the model as just another variation", and [Peng et al.](https://arxiv.org/abs/1710.06537) randomise simulator dynamics so that policies trained only in simulation keep a similar level of performance on a real robot pushing task. Neither was run here; both are cited from their abstracts.

## Limits

- CartPole is a small, low-dimensional task with a fixed 500-step limit; the update-to-data findings come from one network size and one learning rate (0.001) with a soft target update of 0.005, and were not tuned per arm.
- Ten seeds per arm and noisy 10-episode evaluations give wide intervals. Only the comparison between resets and no resets is strong; the ordering of 0.25, 1 and 4 updates per step is not.
- Resets were applied every 10,000 steps without searching over the period.
- The offline dataset size (30,000) and update count (20,000) were fixed, and the controller was written for this experiment, so the "expert" is a heuristic, not a learned policy.
- The offline failure is characterised in detail for one seed of one dataset.

## Code and output

DQN, replay buffer and update rule:

```python
import sys, time, numpy as np, torch, torch.nn as nn, gymnasium as gym
torch.set_num_threads(1)
def qnet(): return nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
class Replay:
    def __init__(s, cap=50_000):
        s.o = np.zeros((cap, 4), np.float32); s.a = np.zeros(cap, np.int64); s.r = np.zeros(cap, np.float32); s.o2 = np.zeros((cap, 4), np.float32); s.d = np.zeros(cap, np.float32); s.n = 0; s.cap = cap
    def add(s, o, a, r, o2, d):
        i = s.n % s.cap; s.o[i], s.a[i], s.r[i], s.o2[i], s.d[i] = o, a, r, o2, d; s.n += 1
    def sample(s, k, rng):
        i = rng.integers(0, min(s.n, s.cap), k); return [torch.as_tensor(x[i]) for x in (s.o, s.a, s.r, s.o2, s.d)]
def update(q, qt, opt, batch, gamma=0.99, tau=0.005):
    o, a, r, o2, d = batch
    with torch.no_grad(): y = r + gamma * (1 - d) * qt(o2).max(1).values
    loss = nn.functional.smooth_l1_loss(q(o).gather(1, a[:, None]).squeeze(1), y)
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
    with torch.no_grad():
        for p, pt in zip(q.parameters(), qt.parameters()): pt.mul_(1 - tau).add_(tau * p)
def evaluate(q, seed, n=10):
    env = gym.make("CartPole-v1"); tot = []
    for i in range(n):
        o, _ = env.reset(seed=10_000 + seed * 100 + i); R = 0; done = False
        while not done:
            with torch.no_grad(): a = int(q(torch.as_tensor(o, dtype=torch.float32)).argmax())
            o, r, te, tr, _ = env.step(a); R += r; done = te or tr
        tot.append(R)
    return float(np.mean(tot))
def train_online(utd, seed, steps=40_000, eps_steps=10_000, start=1000, batch=64, log=2000, reset_every=None):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    env = gym.make("CartPole-v1"); q, qt = qnet(), qnet(); qt.load_state_dict(q.state_dict()); opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    rb = Replay(); o, _ = env.reset(seed=seed); curve = []; n_upd = 0; acc = 0.0
    for t in range(1, steps + 1):
        eps = max(0.05, 1 - 0.95 * t / eps_steps)
        a = int(rng.integers(0, 2)) if rng.random() < eps else int(q(torch.as_tensor(o, dtype=torch.float32)).argmax())
        o2, r, te, tr, _ = env.step(a); rb.add(o, a, r, o2, float(te)); o = o2
        if te or tr: o, _ = env.reset()
        if t >= start:
            acc += utd
            while acc >= 1: update(q, qt, opt, rb.sample(batch, rng)); acc -= 1; n_upd += 1
        if reset_every and t % reset_every == 0 and t < steps:  # full parameter reset, replay buffer kept
            q, qt = qnet(), qnet(); qt.load_state_dict(q.state_dict()); opt = torch.optim.Adam(q.parameters(), lr=1e-3)
        if t % log == 0: curve.append((t, n_upd, evaluate(q, seed)))
    return curve
if __name__ == "__main__":
    t0 = time.time(); c = train_online(float(sys.argv[1]), 0, steps=10_000); print(c, time.time() - t0)
```

Online experiment (four arms):

```python
import json, time, numpy as np
from multiprocessing import Pool
from dqn import train_online
SEEDS = list(range(10))
ARMS = [("update-to-data 0.25", 0.25, None), ("update-to-data 1", 1.0, None), ("update-to-data 4", 4.0, None), ("update-to-data 4 + reset every 10k steps", 4.0, 10_000)]
def job(a): name, u, rs, s = a; return name, s, train_online(u, s, reset_every=rs)
if __name__ == "__main__":
    t = time.time()
    with Pool(14) as p: out = p.map(job, [(n, u, rs, s) for n, u, rs in ARMS for s in SEEDS], chunksize=1)
    json.dump(out, open("e3_online.json", "w"))
    print("Greedy evaluation (10 episodes) every 2,000 environment steps, 10 seeds per arm, 40,000 environment steps")
    for n, u, rs in ARMS:
        runs = [o for o in out if o[0] == n]; C = np.array([[x[2] for x in o[2]] for o in runs]); upd = np.mean([o[2][-1][1] for o in runs])
        print(f"{n:42s} updates {upd:7.0f} | mean over all 20 checkpoints {C.mean():5.1f} | mean of last 5 checkpoints {C[:,-5:].mean():5.1f} | share of checkpoints >= 195: {100*(C>=195).mean():3.0f}% | seeds ending >= 195: {(C[:,-1]>=195).sum()}/10")
    print("elapsed", time.time() - t)
```

Output:

```
Greedy evaluation (10 episodes) every 2,000 environment steps, 10 seeds per arm, 40,000 environment steps
update-to-data 0.25                        updates    9750 | mean over all 20 checkpoints 179.5 | mean of last 5 checkpoints 170.2 | share of checkpoints >= 195:  33% | seeds ending >= 195: 2/10
update-to-data 1                           updates   39001 | mean over all 20 checkpoints 196.3 | mean of last 5 checkpoints 271.8 | share of checkpoints >= 195:  30% | seeds ending >= 195: 4/10
update-to-data 4                           updates  156004 | mean over all 20 checkpoints 172.3 | mean of last 5 checkpoints 165.2 | share of checkpoints >= 195:  24% | seeds ending >= 195: 0/10
update-to-data 4 + reset every 10k steps   updates  156004 | mean over all 20 checkpoints 197.3 | mean of last 5 checkpoints 302.3 | share of checkpoints >= 195:  40% | seeds ending >= 195: 5/10
elapsed 364.56823897361755
```

Statistics:

```python
import json, numpy as np
from scipy import stats
out = json.load(open("e3_online.json")); arms = []
[arms.append(o[0]) for o in out if o[0] not in arms]
S = {a: np.array([np.mean([x[2] for x in o[2]][-5:]) for o in out if o[0] == a]) for a in arms}
print("Per-seed mean of the last 5 evaluation checkpoints: mean [bootstrap 95% interval] and Welch test against update-to-data 1")
rng = np.random.default_rng(0)
for a in arms:
    bs = [rng.choice(S[a], 10).mean() for _ in range(10000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    p = "" if a == arms[1] else f" | Welch p vs UTD 1 = {stats.ttest_ind(S[a], S[arms[1]], equal_var=False)[1]:.3f}"
    print(f"{a:42s} {S[a].mean():6.1f} [{lo:6.1f}, {hi:6.1f}]{p}")
print("Welch p, UTD 4 + reset vs UTD 4 without reset: %.3f" % stats.ttest_ind(S[arms[3]], S[arms[2]], equal_var=False)[1])
```

```
Per-seed mean of the last 5 evaluation checkpoints: mean [bootstrap 95% interval] and Welch test against update-to-data 1
update-to-data 0.25                         170.2 [ 131.6,  220.2] | Welch p vs UTD 1 = 0.091
update-to-data 1                            271.8 [ 180.1,  367.0]
update-to-data 4                            165.2 [ 123.7,  213.7] | Welch p vs UTD 1 = 0.079
update-to-data 4 + reset every 10k steps    302.3 [ 263.1,  336.6] | Welch p vs UTD 1 = 0.584
Welch p, UTD 4 + reset vs UTD 4 without reset: 0.000
```

Offline experiment:

```python
import json, time, numpy as np, torch, gymnasium as gym
from multiprocessing import Pool
from dqn import qnet, Replay, update, evaluate
torch.set_num_threads(1)
def heur(o): return int(o[2] + o[3] > 0)  # push toward the side the pole is leaning or falling
def collect(eps, n, seed):
    env = gym.make("CartPole-v1"); rng = np.random.default_rng(seed); rb = Replay(n); o, _ = env.reset(seed=seed); rets = []; R = 0
    while rb.n < n:
        a = int(rng.integers(0, 2)) if rng.random() < eps else heur(o)
        o2, r, te, tr, _ = env.step(a); rb.add(o, a, r, o2, float(te)); R += r; o = o2
        if te or tr: rets.append(R); R = 0; o, _ = env.reset()
    return rb, float(np.mean(rets)) if rets else float("nan")
DATASETS = {"random policy (eps = 1.0)": 1.0, "heuristic with 50% random actions": 0.5, "heuristic with 20% random actions": 0.2, "heuristic with 5% random actions": 0.05}
def job(a):
    name, seed = a; rb, beh = collect(DATASETS[name], 30_000, seed); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    q, qt = qnet(), qnet(); qt.load_state_dict(q.state_dict()); opt = torch.optim.Adam(q.parameters(), lr=1e-3); curve = []
    for u in range(1, 20_001):
        update(q, qt, opt, rb.sample(64, rng))
        if u % 2000 == 0: curve.append(evaluate(q, seed))
    return name, seed, beh, curve
if __name__ == "__main__":
    t = time.time()
    with Pool(14) as p: out = p.map(job, [(n, s) for n in DATASETS for s in range(10)], chunksize=1)
    json.dump(out, open("e3_offline.json", "w"))
    print("Offline DQN (no environment interaction, 20,000 updates on a fixed 30,000-transition dataset), 10 seeds per dataset")
    for n in DATASETS:
        runs = [o for o in out if o[0] == n]; beh = np.mean([o[2] for o in runs]); C = np.array([o[3] for o in runs])
        print(f"{n:36s} data-collecting policy return {beh:6.1f} | learned policy return after 20k updates {C[:,-1].mean():6.1f} (sd {C[:,-1].std(ddof=1):5.1f}) | mean of last 3 checkpoints {C[:,-3:].mean():6.1f}")
    print("elapsed", time.time() - t)
```

Output:

```
Offline DQN (no environment interaction, 20,000 updates on a fixed 30,000-transition dataset), 10 seeds per dataset
random policy (eps = 1.0)            data-collecting policy return   22.4 | learned policy return after 20k updates  106.8 (sd  64.8) | mean of last 3 checkpoints  118.2
heuristic with 50% random actions    data-collecting policy return  115.5 | learned policy return after 20k updates  457.6 (sd  76.4) | mean of last 3 checkpoints  389.5
heuristic with 20% random actions    data-collecting policy return  320.4 | learned policy return after 20k updates    9.6 (sd   0.7) | mean of last 3 checkpoints    9.7
heuristic with 5% random actions     data-collecting policy return  461.4 | learned policy return after 20k updates    9.3 (sd   0.2) | mean of last 3 checkpoints    9.3
elapsed 67.97095584869385
```

Diagnostics (constant-action baseline, Q-values, reset-arm checkpoints):

```python
import numpy as np, torch, gymnasium as gym, json
from e3_offline import collect, DATASETS
from dqn import qnet, update
env = gym.make("CartPole-v1")
for a in (0, 1):
    R = []
    for i in range(20):
        o, _ = env.reset(seed=i); t = 0; d = False
        while not d: o, r, te, tr, _ = env.step(a); t += r; d = te or tr
        R.append(t)
    print(f"always action {a}: mean return {np.mean(R):.1f}")
seed = 0; rb, beh = collect(0.05, 30_000, seed); torch.manual_seed(seed); rng = np.random.default_rng(seed)
q, qt = qnet(), qnet(); qt.load_state_dict(q.state_dict()); opt = torch.optim.Adam(q.parameters(), lr=1e-3)
for u in range(20_000): update(q, qt, opt, rb.sample(64, rng))
n = rb.n; O = torch.as_tensor(rb.o[:n])
with torch.no_grad(): Q = q(O)
act = rb.a[:n]
print("5% random dataset: share of dataset actions that differ from the heuristic:", float(np.mean(act != (rb.o[:n, 2] + rb.o[:n, 3] > 0).astype(int))))
print("greedy action share on dataset states: action 1 =", float((Q.argmax(1) == 1).float().mean()))
print("mean Q of dataset action %.1f vs mean Q of the other action %.1f" % (Q.gather(1, torch.as_tensor(act)[:, None]).mean(), Q.gather(1, torch.as_tensor(1 - act)[:, None]).mean()))
print("max Q over dataset: %.1f (reward-1-per-step, gamma .99 upper bound 100)" % Q.max())
on = json.load(open("e3_online.json"))
import numpy as np
C = np.array([[x[2] for x in o[2]] for o in on if o[0].endswith("reset every 10k steps")])
print("reset arm mean eval at steps 10k, 20k, 30k:", C[:, [4, 9, 14]].mean(0).round(1), "and at 12k, 22k, 32k:", C[:, [5, 10, 15]].mean(0).round(1))
```

```
always action 0: mean return 9.4
always action 1: mean return 9.3
5% random dataset: share of dataset actions that differ from the heuristic: 0.024766666666666666
greedy action share on dataset states: action 1 = 1.0
mean Q of dataset action 382688.4 vs mean Q of the other action 389431.9
max Q over dataset: 624415.9 (reward-1-per-step, gamma .99 upper bound 100)
reset arm mean eval at steps 10k, 20k, 30k: [17.7 15.6 11.7] and at 12k, 22k, 32k: [158.8 149.2 230.9]
```

Next: Part 4 turns to applications and checks five well-known cases (game playing, chip placement, data-centre cooling, plasma control and matrix multiplication) against what the original papers claim.

## References

- [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier](https://iclr.cc/virtual/2023/poster/11457), ICLR 2023.
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643), 2020.
- [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907), 2017.
- [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537), 2017.
