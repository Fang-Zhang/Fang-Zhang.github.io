---
title: "AI Plays Super Mario, Part 6: Learning From a Demonstrator"
date: 2027-01-11 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [imitation-learning, behavioural-cloning, dagger, games, super-mario, tutorial, series]
description: "Part 6 of the AI Plays Super Mario series. A scripted demonstrator clears level 1, a network copies its key presses from screenshots, and the copy fails. The failure, the DAgger fix from the literature, and what the DAgger rounds did here: still 0 clears out of 20."
---

[Part 5](/posts/ai-plays-super-mario-5-dqn-and-ppo/) ended by asking whether an AI can learn by copying a player instead of by trial and error. This post tries it. A scripted demonstrator, a program and not a human, plays level 1 while every screen and key press is recorded; a neural network is trained to predict the key from the screen; and the copy is let loose on the level. The copy never clears it. The post shows the code, the numbers, the cause that the literature gives for this kind of failure, the fix from the literature (DAgger) and what two rounds of that fix did here: no clears either. All numbers are single runs with one random seed per setting, and the reason for the failure here was not established.

## Imitation instead of reward

Reinforcement learning (Parts 4 and 5) needs a reward and millions of trials. Imitation learning needs neither. It needs a demonstrator and a table of pairs (screen, key pressed). Training is then ordinary supervised classification: show the network a screen, ask which of the seven key combinations the demonstrator pressed, and correct it when it is wrong. The simplest version is called behavioural cloning.

The cost moves from the reward to the demonstrator. Someone or something has to play well.

## A scripted demonstrator

A demonstrator needs to clear the level, and clear it repeatedly. This post uses a program instead of a human. The program exploits something only an emulator offers: the full game state can be saved and restored. At every decision it saves the state, plays 81 candidate key sequences of 26 decisions each from that state (26 decisions are 104 frames, about 1.7 seconds of game time at 60 frames per second), scores each by how far right Mario got without dying, restores the state, and presses the first key of the best sequence. This is random shooting, a basic form of planning by search. It reads the emulator's internal state, so it is not something a screen-only player could do, and nothing the network learns later has that access.

Two details are specific to this game. A sequence counts as a death when the game's RAM says so, not only when the lives counter drops: the vertical-position byte at 0xB5 is above 1 once Mario is below the visible screen, 0xCE is his on-screen height, and 0x0E is the player state, where 11 means dying ([RAM map](https://datacrystal.tcrf.net/wiki/Super_Mario_Bros.:RAM_map)). The height threshold of 200 is my own value, chosen after looking at the heights seen on the ground. And the candidate set always contains one sequence that just runs right.

Earlier versions of this demonstrator tried fixed button patterns (jump for n frames, hold for m) instead of random sequences. Every variant died at x of about 1,560, a pit followed by an enemy, so the version below replaced them.

Save this as `demo.py`, next to the `mario_env.py` from [Part 3](/posts/ai-plays-super-mario-3-the-environment/):

```python
"""Search-based demonstrator: at every decision, try random button sequences on a copy of the
emulator, keep the one that gets furthest right without dying, and play its first action."""
import sys, time, numpy as np
from mario_env import Mario

H = 26                     # look ahead 26 decisions (about 1.7 seconds of game time)

def snapshot(m):
    return (m.env.em.get_state(), m.last_x, m.best_x, m.last_lives, m.steps)

def restore(m, s):
    m.env.em.set_state(s[0]); m.last_x, m.best_x, m.last_lives, m.steps = s[1:]

N_SAMPLES, ACTS, PROBS = 80, [1, 2, 3, 4], [0.1, 0.15, 0.4, 0.35]

def sample_seq(rng):
    seq = []
    while len(seq) < H:
        seq += [int(rng.choice(ACTS, p=PROBS))] * int(rng.integers(2, 9))
    return seq[:H]

def choose(m, rng):
    """Random shooting: try N random button sequences on a copy of the emulator, play the first
    action of the one that gets furthest right without dying (dying sooner scores worse)."""
    snap = snapshot(m); best, best_score = 3, -1e18
    cands = [[3] * H] + [sample_seq(rng) for _ in range(N_SAMPLES)]
    for seq in cands:
        restore(m, snap); score = None
        for k, a in enumerate(seq):
            _, r, term, trunc, info = m.step(a)
            if info["cleared"]:
                score = 1e9 - k; break
            ram = m.env.get_ram()
            if term or ram[0xB5] > 1 or ram[0xCE] >= 200 or ram[0x0E] == 11:
                score = -1e9 + k; break
        if score is None:
            score = m.last_x
        if score > best_score:
            best, best_score = seq[0], score
    restore(m, snap)
    return best

def run(seed, noise, max_decisions=1500):
    rng = np.random.default_rng(seed)
    m = Mario(); obs, info = m.reset()
    frames, labels = [obs], []
    for t in range(max_decisions):
        a = choose(m, rng); labels.append(a)
        a_exec = int(rng.choice([1, 3, 4])) if rng.random() < noise else a
        obs, r, term, trunc, info = m.step(a_exec)
        if term or trunc:
            break
        frames.append(obs)
        if t % 50 == 0: print(f'  t={t} x={m.last_x}', flush=True)
    return np.array(frames[:len(labels)]), np.array(labels), info

if __name__ == "__main__":
    seed, noise, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    t0 = time.time()
    f, l, info = run(seed, noise)
    print(f"seed {seed} noise {noise}: {len(l)} decisions, cleared={info['cleared']}, best_x={info['best_x']}, {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(out, frames=f, labels=l, cleared=info["cleared"], best_x=info["best_x"])
```

`noise` is the probability that the key actually pressed is replaced by a random one from run right, run right plus jump, and jump right. The recorded label stays the key the demonstrator intended. Noise makes the recordings visit situations that a flawless run never meets, such as being slightly off the ideal path, along with the correct action for them.

### Checkpoint

Run this to play the first 60 decisions with no noise:

```python
import demo
f, l, info = demo.run(0, 0.0, max_decisions=60)
print(len(l), "labels, first 10:", l[:10])
```

It should print `t=0 x=40` and `t=50 x=494` while it runs, then the number of labels, 60. With 81 sequences of 26 steps per decision it takes a few minutes on one CPU core. The full level took 486 decisions and 1,506 seconds for one clean run on the machine used for this series (32 cores, no GPU, one core per run).

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-scripted-demonstrator-clear.mp4' | relative_url }}" type="video/mp4">
</video>

*The scripted demonstrator clearing level 1, noise 0. The bar shows the decision step, Mario's x position, the furthest x so far and the keys pressed. About 520 decisions in this recording.*

The data used below is 13 recordings: one clean run saved as `demos/test0.npz`, and twelve with noise 0.1 saved as `demos/ep1.npz` to `demos/ep12.npz`, made with `python demo.py SEED 0.1 demos/epSEED.npz` for seeds 1 to 12. Six of the twelve noisy runs cleared the level, with 512 to 533 decisions; the other six died between x = 1,532 and x = 2,754. Together there are 5,252 labelled screens. The label counts are: run right 3,544 (67%), run right plus jump 1,097, jump right 422, walk right 189, and the other three key combinations never appear. A network that always answers "run right" is therefore right about two thirds of the time, a baseline that matters below.

## Behavioural cloning

The network is a [Nature-DQN](https://www.nature.com/articles/nature14236)-style convolutional net, the same kind as in the PPO runs, on four stacked 84×84 gray frames, with seven outputs, trained with cross-entropy against the demonstrator's key. Two whole episodes (`ep3` and `ep8`, 904 screens) are held out to measure accuracy on play the network has not seen. Save this as `bc_train.py`:

```python
import glob, numpy as np, torch, torch.nn as nn, sys
torch.manual_seed(0); np.random.seed(0)
HOLD = {"ep3", "ep8"}              # whole episodes kept out to measure accuracy on unseen play

def stack(frames):                  # (T,84,84,1) -> (T,4,84,84), zeros before the episode starts, as VecFrameStack does
    f = frames[:, :, :, 0]; pad = np.concatenate([np.zeros((3,) + f.shape[1:], f.dtype), f])
    return np.stack([pad[i:i + len(f)] for i in range(4)], axis=1)

def load(files):
    X, y = [], []
    for f in files:
        d = np.load(f); X.append(stack(d["frames"])); y.append(d["labels"])
    return np.concatenate(X), np.concatenate(y)

def make_net():
    return nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                         nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 7))

if __name__ == "__main__":
    files = sorted(glob.glob("demos/ep*.npz")) + sorted(glob.glob("demos/dg*.npz")) + ["demos/test0.npz"]
    tr = [f for f in files if f.split("/")[-1][:-4] not in HOLD]; va = [f for f in files if f.split("/")[-1][:-4] in HOLD]
    Xt, yt = load(tr); Xv, yv = load(va)
    print("train", len(yt), "val", len(yv), flush=True)
    Xt, yt, Xv, yv = map(torch.from_numpy, (Xt, yt.astype(np.int64), Xv, yv.astype(np.int64)))
    net = make_net(); opt = torch.optim.Adam(net.parameters(), float(sys.argv[2])); lossf = nn.CrossEntropyLoss()
    for ep in range(int(sys.argv[1])):
        net.train(); perm = torch.randperm(len(yt)); tot = 0
        for i in range(0, len(yt), 64):
            idx = perm[i:i + 64]; loss = lossf(net(Xt[idx].float() / 255), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            acc_t = (net(Xt[:2000].float() / 255).argmax(1) == yt[:2000]).float().mean().item()
            acc_v = (net(Xv.float() / 255).argmax(1) == yv).float().mean().item()
        print(f"epoch {ep+1:2d} loss {tot/len(yt):.3f} train_acc {acc_t:.3f} val_acc {acc_v:.3f}", flush=True)
        torch.save(net.state_dict(), "bc.pt")
    print("majority-class val acc", (yv == 3).float().mean().item())
```

Run `python bc_train.py 150 3e-4`. A first attempt with 30 epochs at a learning rate of 1e-4 stayed near the majority-class baseline (validation accuracy 60% to 64% against 64.3% for always answering "run right"), so the final setting trains longer and faster. Training accuracy then climbs from 0.69 to 0.96, but validation accuracy stays between 0.55 and 0.60, below the baseline. The network memorises its 4,348 training screens and does not generalise to the held-out episodes by this measure.

Held-out accuracy is only a proxy. The real test is to let the network play. Save this as `bc_eval.py`; it plays 20 episodes either by sampling a key from the network's probabilities or by always pressing the most probable key (argmax):

```python
import sys, numpy as np, torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from mario_env import Mario
from bc_train import make_net
torch.set_num_threads(1)
n_ep, det = int(sys.argv[1]), sys.argv[2] == "det"
net = make_net(); net.load_state_dict(torch.load("bc.pt")); net.eval()
env = VecFrameStack(DummyVecEnv([Mario]), 4, channels_order="last"); res = []
for ep in range(n_ep):
    obs = env.reset(); done = False
    while not done:
        x = torch.from_numpy(obs).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad(): logits = net(x)
        a = logits.argmax(1) if det else torch.distributions.Categorical(logits=logits).sample()
        obs, r, d, info = env.step(a.numpy()); done = d[0]
    res.append((info[0]["best_x"], info[0]["cleared"]))
print("BC", "argmax" if det else "sampled", "clears", sum(c for _, c in res), "/", n_ep, "mean_best_x", round(np.mean([x for x, _ in res])), "xs", [x for x, _ in res], flush=True)
```

Copy the trained weights first (`cp bc.pt bc_round0.pt`), because `bc_eval.py` reads `bc.pt` and later rounds overwrite it. Then run `python bc_eval.py 20 sto` and `python bc_eval.py 20 det`. The result:

- sampled: 0 clears in 20, mean best x 1,072, furthest 2,587
- most probable key: 0 clears in 20, all 20 episodes end at x = 434

The level ends at x = 3,266 in these runs. The argmax player is deterministic, so all 20 episodes are the same one.

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-cloning-stuck-at-pipe.mp4' | relative_url }}" type="video/mp4">
</video>

*The cloned network pressing its most probable key. Mario reaches the first pipe at x = 434 and holds run right against it, never jumping, until the episode ends.*

## Why a good copy still fails

[Ross, Gordon and Bagnell (2011)](https://arxiv.org/abs/1011.0686) describe this problem. A network trained on the demonstrator's states is tested on its own states, and those are different. One small mistake puts the copy in a place the demonstrator never visited, there the network is less accurate, and the next mistake pushes it further away. The paper cites a result that a classifier with error ε on the demonstrator's states can make about T²ε mistakes over T steps when it runs by itself, where the error of a normal supervised problem would grow only as Tε. The paper also reports a Super Mario Bros. experiment in which the supervised-only learner "often" got stuck against an obstacle instead of jumping, because the demonstrator jumps from a distance and the learner never saw what a close approach looks like. The experiment used a Mario simulator with grid features instead of pixels, so its numbers do not carry over to this post. The pattern is what the stuck Mario in the video shows.

The fix proposed in the paper is DAgger, dataset aggregation:

1. Train on the demonstrator's data.
2. Let the learner play. At every screen it visits, ask the demonstrator what it would press there, and add those (screen, key) pairs to the dataset.
3. Retrain on everything and repeat from step 2.

The learner chooses where to go; the demonstrator only labels. This works here because the demonstrator is a search over saved emulator states and can answer "what would you press here?" for any state. A human demonstrator would have to label recorded screens one by one without being able to play them.

Save this as `dagger.py`. One call is one learner episode of at most 400 decisions:

```python
"""One DAgger round: the learner plays (sampling from its own probabilities); at every state it
visits, the search demonstrator says what it would press there. Those (frame, label) pairs are saved."""
import sys, numpy as np, torch
from collections import deque
from mario_env import Mario
from bc_train import make_net
import demo
torch.set_num_threads(1)
rnd, seed, model = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(1000 * rnd + seed); torch.manual_seed(1000 * rnd + seed)
net = make_net(); net.load_state_dict(torch.load(model)); net.eval()
m = Mario(); obs, info = m.reset()
buf = deque([np.zeros((84, 84), np.uint8)] * 3 + [obs[:, :, 0]], maxlen=4)
frames, labels = [], []
for t in range(400):
    x = torch.from_numpy(np.stack(buf)[None]).float() / 255
    with torch.no_grad(): a = int(torch.distributions.Categorical(logits=net(x)).sample())
    frames.append(buf[-1][:, :, None].copy()); labels.append(demo.choose(m, rng))
    obs, r, term, trunc, info = m.step(a)
    if term or trunc: break
    buf.append(obs[:, :, 0])
    if t % 50 == 0: print(f"  t={t} x={m.last_x}", flush=True)
print(f"dagger round {rnd} seed {seed}: {len(labels)} states, best_x={info['best_x']}, cleared={info['cleared']}", flush=True)
np.savez_compressed(f"demos/dg{rnd}_{seed}.npz", frames=np.array(frames), labels=np.array(labels), cleared=info["cleared"], best_x=info["best_x"])
```

A round is: `python dagger.py ROUND SEED bc_roundN.pt` for seeds 1 to 12, then `python bc_train.py 150 3e-4` again (it reads every `demos/dg*.npz` file), then evaluate. Each round used 12 episodes in parallel and several tens of minutes per round. This is the simplest variant in the paper: after the first round the learner acts alone and the demonstrator only labels.

## What two rounds did

Training data grew from 4,348 screens (cloning) to 6,955 (after round 1) and 9,831 (after round 2). Twenty evaluation episodes each:

- **Behavioural cloning:** sampled mean best x 1,072, furthest 2,587; most probable key x = 434 every time. 0 of 20 cleared.
- **After 1 DAgger round:** sampled mean 1,024, furthest 2,402; most probable key x = 2,162 every time. 0 of 20 cleared.
- **After 2 DAgger rounds:** sampled mean 985, furthest 1,784; most probable key x = 709 every time. 0 of 20 cleared.

![Furthest x reached in each of 20 evaluation episodes for behavioural cloning, DAgger round 1 and DAgger round 2, sampled and most-probable-key. The dashed line marks the level end at x = 3266; no episode reaches it.]({{ '/assets/pic/mario-ai-bc-results.png' | relative_url }})
*Furthest x in every evaluation episode (single run per setting). The most probable key gives the same value in all 20 episodes.*

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="{{ '/assets/video/mario-ai-dagger-stuck-at-stairs.mp4' | relative_url }}" type="video/mp4">
</video>

*The network after one DAgger round, pressing its most probable key. It gets much further than the plain copy and stops at the stairs at x = 2,162.*

The first DAgger round did what the paper predicts in one respect: the deterministic player got from x = 434 to x = 2,162. It did not hold up. Sampled play did not improve, the second round made the deterministic player worse (x = 709), and no episode cleared the level in any setting. During collection the learner also drifted: in round 1, three of the twelve runs sat at x = 594 until the 400-decision cap, and in round 2 one run reached x = 3,266 but was cut off by the cap before the flag, so it does not count as cleared.

## What was not established

The experiment gives no reason for why two DAgger rounds did not help. The following are hypotheses, none of them tested:

- **Inconsistent labels.** The demonstrator picks the best of 81 random sequences, so at an ambiguous screen two runs can give different keys. The network is trained to match labels that disagree with each other.
- **Too little data.** The paper's Mario experiment used 5,000 data points per iteration over 20 iterations. Here a round added about 2,600 to 2,900 screens, and there were two rounds.
- **A network trained from scratch on pixels.** The paper's learners used features, not pixels.
- **The held-out accuracy gap.** Validation accuracy below the "always run right" baseline says the network does not generalise across these episodes, and more data is the obvious remedy, but not a tested one.

No variant was run to separate these. A clean test would change one of them (for example, make the demonstrator deterministic) and rerun with several seeds. Nothing here supports a claim about which one matters, or about whether imitation can solve this level with more effort.

## Where this leaves the series

On this level, with these settings, imitation did not produce an AI that clears level 1, while the PPO runs of Part 4 did clear it in some evaluations. That is a statement about these runs, not about the two approaches in general: the PPO runs had millions of steps and a hand-built reward, the imitation runs had 13 demonstrations and two DAgger rounds, and each is a single seed.

The next post returns to PPO. It trains PPO on level 1 and evaluates the result.
