---
title: "Reinforcement Learning Engineering (2/6): Removing Advantage Normalisation Helped PPO on CartPole, and One Seed Gets the Sign Wrong 23 Percent of the Time"
date: 2027-06-25 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement learning, engineering, ppo, random seeds, reproducibility, series]
description: "Part 2 of 6 on reinforcement learning engineering. A from-scratch PPO ablation on CartPole with 20 seeds per variant: which implementation details mattered here, how wide the seed spread is, and how many seeds a comparison needs."
---

## What this post finds

Policy-gradient code carries many small choices that papers describe as side notes. [Engstrom et al.](https://arxiv.org/abs/2005.12729) call them "code-level optimizations" and report that they account for most of PPO's gain in cumulative reward over TRPO. [Henderson et al.](https://arxiv.org/abs/1709.06560) show that variance intrinsic to the methods, together with non-determinism in the benchmarks, makes reported results hard to interpret without significance measures. This post tests both ideas on a small case: a PPO written from scratch in PyTorch, CartPole-v1, six standard details removed one at a time, 20 random seeds per variant (140 runs, 150,000 environment steps each, about 11 to 13 seconds per run on one CPU thread).

The baseline uses advantage normalisation, ratio clipping at 0.2, GAE with lambda 0.95, gradient-norm clipping at 0.5, learning-rate annealing and orthogonal initialisation. The score is the mean return of episodes that finished in the last 10 percent of training (500 is the maximum).

- **The baseline is not the best variant here, and its seed spread is wide.** Final return averaged 417.6 with a standard deviation of 96.6 across seeds, from 197.9 for the worst seed to 500.0 for the best, and 12 of 20 seeds reached at least 450.
- **Removing advantage normalisation helped most.** Mean 487.3 (standard deviation 24.0), 19 of 20 seeds at 450 or above, a difference of +69.7 against the baseline (bootstrap 95 percent interval +29.8 to +114.2, Welch p = 0.005).
- **Four other removals look like improvements of about 48 to 49 points, but sit near the noise.** No ratio clipping (+48.4, p = 0.066), no GAE with lambda = 1 (+48.8, p = 0.045), no learning-rate annealing (+49.4, p = 0.039). With six comparisons, a Bonferroni threshold is 0.0083, and only the advantage-normalisation result passes it.
- **Two removals changed nothing measurable.** No gradient-norm clipping (+3.0, p = 0.929) and default weight initialisation (+0.6, p = 0.983). Default initialisation did have the lowest average return over the whole run (214.0 against 230.8 for the baseline), so it may slow early learning even where final scores agree; this was not tested further.
- **Few seeds mislead.** Resampling the 20 runs, a one-seed-per-arm experiment shows "advantage normalisation removal is better" in 77 percent of cases, so the sign is wrong 23 percent of the time. With 3 seeds per arm that rises to 93 percent, with 5 to 97 percent and with 10 to 100 percent. For the 48-point effects the corresponding numbers are 60 to 71 percent at one seed and 92 to 96 percent at 10 seeds. A variant with no real effect (default initialisation) looks better in 46 to 51 percent of experiments, at any seed count.

![Two panels. Left: final mean episode return of PPO on CartPole for seven variants, 20 seeds each, as dots with a black mean bar; the baseline mean is 418 with a wide spread from 198 to 500, while removing advantage normalisation gives 487. Right: the share of simulated experiments in which each variant looks better than the baseline as the number of seeds per arm grows from 1 to 20; for advantage normalisation it rises from 77 to 100 percent]({{ site.baseurl }}/assets/pic/rl-eng-2-ppo-seeds.png){: width="900" }
_Figure: left, final return per seed for each variant (black bar: mean); right, how often a k-seed comparison ranks the variant above the baseline. Measured here._

## What to do with this

1. **Run a comparison with at least 10 seeds per arm, and report the spread.** The effects above are half to three quarters of the baseline's own standard deviation. The seed spread is part of the result, and a single learning curve hides it.
2. **Treat "standard" details as hypotheses for the task at hand.** Advantage normalisation, ratio clipping and learning-rate annealing are common defaults. On this task and these untuned settings none of them improved final return. This result is specific to CartPole with a learning rate of 0.001, and does not argue against the details elsewhere; the paper above studied continuous-control benchmarks, which were not run here.
3. **Ablate one detail at a time, with the same seeds, and keep the metric fixed before looking at results.** The final-10-percent score here was chosen in advance; a different score (for example area under the curve) ranks the variants differently, as the training-average column shows.
4. **Count the comparisons.** Four of six comparisons had p below 0.1, and only one survives a correction for six tests.

## Limits

- One environment with a short horizon and a small network. CartPole is solved at 500, so many seeds hit the ceiling and distributions are skewed; means and standard deviations understate that.
- Hyperparameters (learning rate 0.001, 8 environments, 128 steps, 4 epochs, 4 minibatches) were fixed for all variants, not tuned per variant. Some removals may interact with the learning rate. For example, one untested guess is that normalised advantages combined with a large learning rate make the late updates too big.
- Truncated episodes are treated as terminal (no bootstrapping at the time limit), which is simple and slightly biased. Same-step autoreset is used, as in Part 1.
- The "k seeds" experiment resamples from the same 20 runs with replacement, so it describes these runs and not the whole population of seeds.
- The effect sizes are from one run of 20 seeds per variant. Another batch of seeds would move the numbers; the 95 percent intervals above show by how much.
- The six details are those listed above, not an exhaustive list of PPO implementation choices, and the code is a compact re-implementation, not a library.

## Code and output

PPO and the variants (about 80 lines):

```python
import sys, json, time, numpy as np, torch, torch.nn as nn, gymnasium as gym
torch.set_num_threads(1)
BASE = dict(adv_norm=True, clip=0.2, lam=0.95, grad_clip=0.5, anneal=True, ortho=True)
VARIANTS = {
    "baseline": {},
    "no advantage normalisation": dict(adv_norm=False),
    "no ratio clipping": dict(clip=None),
    "no GAE (lambda = 1)": dict(lam=1.0),
    "no gradient-norm clipping": dict(grad_clip=None),
    "no learning-rate annealing": dict(anneal=False),
    "default weight init": dict(ortho=False),
}
def layer(i, o, std, ortho):
    l = nn.Linear(i, o)
    if ortho: nn.init.orthogonal_(l.weight, std); nn.init.zeros_(l.bias)
    return l
class Agent(nn.Module):
    def __init__(s, ortho):
        super().__init__()
        s.actor = nn.Sequential(layer(4, 64, 2**.5, ortho), nn.Tanh(), layer(64, 64, 2**.5, ortho), nn.Tanh(), layer(64, 2, .01, ortho))
        s.critic = nn.Sequential(layer(4, 64, 2**.5, ortho), nn.Tanh(), layer(64, 64, 2**.5, ortho), nn.Tanh(), layer(64, 1, 1., ortho))
def train(name, seed, total=150_000, n_env=8, n_steps=128, epochs=4, n_mb=4, lr=1e-3, gamma=0.99):
    cfg = {**BASE, **VARIANTS[name]}
    torch.manual_seed(seed); np.random.seed(seed)
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(n_env)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    ag = Agent(cfg["ortho"]); opt = torch.optim.Adam(ag.parameters(), lr=lr, eps=1e-5)
    obs, _ = envs.reset(seed=seed); obs = torch.as_tensor(obs, dtype=torch.float32)
    B = n_env * n_steps; n_upd = total // B; ep_ret = np.zeros(n_env); done_rets = []  # (global step, return)
    gstep = 0
    for u in range(n_upd):
        if cfg["anneal"]:
            for g in opt.param_groups: g["lr"] = lr * (1 - u / n_upd)
        O = torch.zeros(n_steps, n_env, 4); A = torch.zeros(n_steps, n_env, dtype=torch.long); LP = torch.zeros(n_steps, n_env); R = torch.zeros(n_steps, n_env); D = torch.zeros(n_steps, n_env); V = torch.zeros(n_steps, n_env)
        for t in range(n_steps):
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=ag.actor(obs)); a = dist.sample(); v = ag.critic(obs).squeeze(-1)
            O[t] = obs; A[t] = a; LP[t] = dist.log_prob(a); V[t] = v
            o, r, te, tr, _ = envs.step(a.numpy()); gstep += n_env
            d = np.logical_or(te, tr); R[t] = torch.as_tensor(r, dtype=torch.float32); D[t] = torch.as_tensor(d, dtype=torch.float32)
            ep_ret += r
            for i in np.where(d)[0]: done_rets.append((gstep, ep_ret[i])); ep_ret[i] = 0
            obs = torch.as_tensor(o, dtype=torch.float32)
        with torch.no_grad():
            nv = ag.critic(obs).squeeze(-1); adv = torch.zeros(n_steps, n_env); last = 0
            for t in reversed(range(n_steps)):
                nxt = nv if t == n_steps - 1 else V[t + 1]; nd = 1.0 - D[t]
                delta = R[t] + gamma * nxt * nd - V[t]; last = delta + gamma * cfg["lam"] * nd * last; adv[t] = last
            ret = adv + V
        bO, bA, bLP, bAdv, bRet = O.reshape(-1, 4), A.reshape(-1), LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        for _ in range(epochs):
            perm = torch.randperm(B)
            for mb in perm.chunk(n_mb):
                dist = torch.distributions.Categorical(logits=ag.actor(bO[mb])); lp = dist.log_prob(bA[mb]); ratio = (lp - bLP[mb]).exp()
                ad = bAdv[mb]
                if cfg["adv_norm"]: ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                if cfg["clip"] is None: pg = -(ratio * ad).mean()
                else: pg = -torch.min(ratio * ad, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * ad).mean()
                vl = 0.5 * ((ag.critic(bO[mb]).squeeze(-1) - bRet[mb]) ** 2).mean()
                loss = pg + 0.5 * vl - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                if cfg["grad_clip"]: nn.utils.clip_grad_norm_(ag.parameters(), cfg["grad_clip"])
                opt.step()
    envs.close()
    return done_rets
if __name__ == "__main__":
    t = time.time(); r = train(sys.argv[1] if len(sys.argv) > 1 else "baseline", 0)
    last = [x for g, x in r if g > 0.9 * 150_000]
    print(len(r), "episodes", "last-10% mean return", np.mean(last), "time", time.time() - t)
```

Experiment driver (20 seeds per variant, 14 processes):

```python
import json, numpy as np, time
from multiprocessing import Pool
from ppo import train, VARIANTS
SEEDS = list(range(20)); TOTAL = 150_000
def job(a):
    name, seed = a; r = train(name, seed, total=TOTAL)
    gs = np.array([g for g, _ in r]); rt = np.array([x for _, x in r])
    final = rt[gs > 0.9 * TOTAL]; bins = np.linspace(0, TOTAL, 16); curve = [float(rt[(gs > bins[i]) & (gs <= bins[i + 1])].mean()) if ((gs > bins[i]) & (gs <= bins[i + 1])).any() else None for i in range(15)]
    return name, seed, float(final.mean()) if len(final) else None, float(rt.mean()), curve
if __name__ == "__main__":
    t = time.time()
    with Pool(14) as p: out = p.map(job, [(n, s) for n in VARIANTS for s in SEEDS], chunksize=1)
    json.dump(out, open("e2_res.json", "w"))
    for n in VARIANTS:
        f = np.array([o[2] for o in out if o[0] == n], dtype=float); m = np.array([o[3] for o in out if o[0] == n])
        print(f"{n:28s} final return: mean {f.mean():6.1f} sd {f.std(ddof=1):5.1f} min {f.min():6.1f} max {f.max():6.1f} | seeds >= 450: {(f >= 450).sum():2d}/20 | mean return over training {m.mean():6.1f}", flush=True)
    print("elapsed", time.time() - t)
```

Output:

```
baseline                     final return: mean  417.6 sd  96.6 min  197.9 max  500.0 | seeds >= 450: 12/20 | mean return over training  230.8
no advantage normalisation   final return: mean  487.3 sd  24.0 min  396.0 max  500.0 | seeds >= 450: 19/20 | mean return over training  241.2
no ratio clipping            final return: mean  465.9 sd  59.6 min  303.2 max  500.0 | seeds >= 450: 16/20 | mean return over training  254.9
no GAE (lambda = 1)          final return: mean  466.4 sd  36.4 min  356.1 max  500.0 | seeds >= 450: 17/20 | mean return over training  236.9
no gradient-norm clipping    final return: mean  420.6 sd 117.4 min  152.2 max  500.0 | seeds >= 450: 13/20 | mean return over training  238.4
no learning-rate annealing   final return: mean  467.0 sd  27.1 min  420.5 max  500.0 | seeds >= 450: 14/20 | mean return over training  236.4
default weight init          final return: mean  418.2 sd  95.6 min  173.4 max  500.0 | seeds >= 450: 10/20 | mean return over training  214.0
elapsed 134.09871196746826
```

Statistics and the k-seed resampling:

```python
import json, numpy as np
from scipy import stats
out = json.load(open("e2_res.json")); names = []
for o in out:
    if o[0] not in names: names.append(o[0])
F = {n: np.array([o[2] for o in out if o[0] == n]) for n in names}
rng = np.random.default_rng(0)
print("Difference in mean final return against the baseline (20 seeds each), Welch t-test and bootstrap 95% interval")
for n in names[1:]:
    d = F[n].mean() - F["baseline"].mean(); t, p = stats.ttest_ind(F[n], F["baseline"], equal_var=False)
    bs = [rng.choice(F[n], 20).mean() - rng.choice(F["baseline"], 20).mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); print(f"{n:28s} diff {d:+6.1f}  95% interval [{lo:+6.1f}, {hi:+6.1f}]  Welch p = {p:.3f}")
print("\nShare of experiments in which the variant looks better than the baseline when only k seeds per arm are run (10,000 resampled experiments)")
res = {}
for n in ("no advantage normalisation", "no ratio clipping", "no GAE (lambda = 1)", "no learning-rate annealing", "default weight init"):
    row = []
    for k in (1, 3, 5, 10, 20):
        w = np.mean([rng.choice(F[n], k).mean() > rng.choice(F["baseline"], k).mean() for _ in range(10000)]); row.append(w)
    res[n] = row; print(f"{n:28s} " + "  ".join(f"k={k:2d}: {100*w:3.0f}%" for k, w in zip((1, 3, 5, 10, 20), row)))
json.dump(res, open("e2_flip.json", "w"))
```

Output:

```
Difference in mean final return against the baseline (20 seeds each), Welch t-test and bootstrap 95% interval
no advantage normalisation   diff  +69.7  95% interval [ +29.8, +114.2]  Welch p = 0.005
no ratio clipping            diff  +48.4  95% interval [  +1.9,  +97.2]  Welch p = 0.066
no GAE (lambda = 1)          diff  +48.8  95% interval [  +6.9,  +95.0]  Welch p = 0.045
no gradient-norm clipping    diff   +3.0  95% interval [ -63.9,  +66.4]  Welch p = 0.929
no learning-rate annealing   diff  +49.4  95% interval [  +9.3,  +94.6]  Welch p = 0.039
default weight init          diff   +0.6  95% interval [ -58.1,  +57.6]  Welch p = 0.983

Share of experiments in which the variant looks better than the baseline when only k seeds per arm are run (10,000 resampled experiments)
no advantage normalisation   k= 1:  77%  k= 3:  93%  k= 5:  97%  k=10: 100%  k=20: 100%
no ratio clipping            k= 1:  71%  k= 3:  78%  k= 5:  84%  k=10:  92%  k=20:  98%
no GAE (lambda = 1)          k= 1:  60%  k= 3:  78%  k= 5:  86%  k=10:  94%  k=20:  99%
no learning-rate annealing   k= 1:  61%  k= 3:  78%  k= 5:  86%  k=10:  96%  k=20:  99%
default weight init          k= 1:  46%  k= 3:  50%  k= 5:  51%  k=10:  51%  k=20:  51%
```

Next: [Part 3](/posts/rl-engineering-3-sample-efficiency/) looks at sample efficiency: replay buffers, how many gradient updates per environment step are worth it, and what offline data and simulation randomisation add.

## References

- [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729), 2020.
- [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), 2017.
- [Gymnasium vector environments and autoreset modes](https://farama.org/Vector-Autoreset-Mode).
