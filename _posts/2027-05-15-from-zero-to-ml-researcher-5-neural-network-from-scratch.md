---
title: "From Zero to ML Researcher, Part 5: A Neural Network From Scratch"
date: 2027-05-15 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, neural network, backpropagation, autograd, pytorch, ML researcher]
description: "Part 5 of a 12-part series for complete beginners. A two-layer neural network written in plain numpy, with backpropagation derived by hand and checked two ways (finite differences and PyTorch autograd). It reaches 96.6% on handwritten digits, and does not beat the nearest-neighbour baseline."
---

## What you will do, and what you get

You will build a neural network with about 40 lines of numpy: 64 pixels in, 32 hidden units, 10 digit scores out, 2,410 parameters in total. You will write the backward pass by hand and verify it twice: against finite differences (worst relative error 5e-09) and against PyTorch's autograd (largest difference 8e-17). Trained for 30 epochs on 1,000 digit images, it scores 96.6% on 797 held-out images (mean of five seeds, standard deviation 0.62). It does *not* beat the 98.4% of the nearest-neighbour baseline from Part 4, and that honest comparison is part of the lesson: a network is a tool, not a guarantee.

![Left: the forward pass from 64 input pixels through a hidden layer with ReLU and an output layer to softmax and loss, with red arrows going backward and the four backward formulas. Middle: training for 30 epochs, with the train loss falling from 1.0 to about 0.06 on a log scale, and train and test accuracy rising to about 98 and 96 percent. Right: test accuracy bars: majority baseline 9.3, nearest average 89.2, network 96.6 with five seed dots, nearest neighbour 98.4]({{ site.baseurl }}/assets/pic/mlr-5-network.png){: width="900" }

## How: the steps

1. **Define the layers.** `z1 = x W1 + b1`, then `h = max(z1, 0)` (ReLU), then `logits = h W2 + b2`: ten scores, one per digit.
2. **Turn scores into probabilities** with softmax, and measure error with cross-entropy: the negative log of the probability given to the true digit.
3. **Write the backward pass:** the derivative of the loss with respect to every weight, working from the output back to the input.
4. **Check it** against finite differences on a small batch before training anything.
5. **Train** with mini-batches: repeat "take 50 images, compute the gradient, step every weight downhill".
6. **Compare** with the baselines of Part 4, on held-out data, over several seeds.
7. **Repeat the gradient with autograd** (PyTorch) and confirm that it matches.

## Why it works: the principle

**A network is Part 3 with more numbers.** Part 3 fitted two parameters by following the gradient of a loss. This network has 2,410 parameters, and the loop is the same: forward pass to get the loss, backward pass to get the gradient, step. What changes is only how the gradient is computed, because the loss depends on early weights through many later operations.

**Backpropagation is the chain rule, applied in reverse order.** If `loss = f(g(w))`, then `dloss/dw = f'(g(w)) * g'(w)`. A network is a chain of such operations; starting from the loss and multiplying local derivatives backwards, each layer reuses the work of the layer after it. This is the algorithm popularised by [Rumelhart, Hinton and Williams (1986)](https://www.nature.com/articles/323533a0). For this network the four steps are:

```
d logits = (p - onehot) / n          softmax with cross-entropy gives this simple form
dW2 = h^T  d logits     db2 = sum of d logits
dh  = d logits  W2^T                  push the gradient back through layer 2
dz1 = dh * (z1 > 0)                   ReLU passes the gradient only where it was active
dW1 = x^T  dz1          db1 = sum of dz1
```

Where `p` are the softmax probabilities and `n` the batch size. Each line is a shape-respecting matrix product (Part 2): a gradient always has the same shape as the thing it is the gradient of, and that is the best way to catch a mistake.

**Why the check matters.** A wrong backward pass does not crash. It trains badly, or worse, almost well. The central finite difference estimates each derivative by nudging a weight up and down and measuring the loss change; it needs no calculus, so an agreement of 5e-09 over 32 randomly chosen weights is strong evidence that the formulas are right. [PyTorch's autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) records every operation during the forward pass and runs the chain rule for you; here it agrees with the hand-written gradient to 1e-16, which is floating-point rounding. In practice you use autograd, and you understand the hand version so that you can reason about what autograd is doing and why a gradient may vanish or explode.

**Why ReLU and this initialisation.** ReLU, `max(z, 0)`, keeps the derivative at 1 for active units, which keeps gradients from shrinking through layers. The weights start as random numbers with standard deviation `sqrt(2 / fan_in)`, the scaling proposed by [He and colleagues (2015)](https://arxiv.org/abs/1502.01852) for ReLU networks; it keeps the size of the activations roughly constant from layer to layer. Starting all weights at zero would make every hidden unit identical and unable to learn different things.

**Reading the results.**
- Loss starts near `ln(10) = 2.303`, the loss of guessing uniformly among 10 digits, and falls to 0.058 by epoch 30.
- Train accuracy reaches 98.6%, test accuracy 96.2% for the seed shown. Over five seeds the test accuracy is 96.2, 95.6, 97.4, 96.9 and 97.0 (mean 96.6, std 0.62): the seed alone moves the result by up to 1.8 points.
- The network beats the nearest-average model (89.2%) and trails 1-NN (98.4%). With only 1,000 training images, memorising the examples and comparing is hard to beat. The advantage of networks shows up with more data and harder tasks; that is a claim to test, not to assume, and the later parts return to it at larger scale.
- Test accuracy peaked around epoch 10 (96.6) and was 96.2 at epoch 30. A change of this size is within the noise between seeds; it is not evidence of overfitting by itself.

**Pitfalls and limits.**
- Learning rate 0.5, batch size 50 and 30 epochs were picked once and not tuned. Tuning on the test set would break the rule of Part 4.
- 2,410 parameters and 1,000 training images: the network can fit the training set almost perfectly (98.6%), so train accuracy is not the number to report.
- A gradient check on a *tiny* batch cannot catch every bug (for instance one that only appears with large batches), but it catches nearly all formula mistakes.

## Example, worked through

One image with two pixels, `x = (1, 2)`, two hidden units, two classes, true class 0. Weights `W1 = [[1, -1], [0.5, 1]]`, `W2 = [[1, -1], [-1, 1]]`, no biases.

- Forward: `z1 = x W1 = (2, 1)`, `h = (2, 1)` (both positive, so ReLU changes nothing), `logits = h W2 = (1, -1)`, `p = (0.8808, 0.1192)`, `loss = -ln 0.8808 = 0.1269`.
- Backward: `d logits = p - onehot = (-0.1192, 0.1192)`. `dW2 = h^T d logits = [[-0.2384, 0.2384], [-0.1192, 0.1192]]`. `dh = d logits W2^T = (-0.2384, 0.2384)`. ReLU passes both. `dW1 = x^T dz1 = [[-0.2384, 0.2384], [-0.4768, 0.4768]]`.
- Check: nudging `W1[1, 0]` by plus and minus 1e-6 and measuring the loss change gives -0.4768, as the formula says.
- Meaning: the entry `-0.4768` is negative, so increasing `W1[1, 0]` lowers the loss, by about 0.48 per unit of increase. The update `w = w - lr * gradient` therefore increases that weight.

```python
import numpy as np
x = np.array([[1., 2.]]); W1 = np.array([[1., -1.], [0.5, 1.]]); W2 = np.array([[1., -1.], [-1., 1.]]); true = 0
z1 = x @ W1; h = np.maximum(z1, 0); logits = h @ W2
p = np.exp(logits) / np.exp(logits).sum()
print("z1", z1, "h", h, "logits", logits, "p", p.round(4), "loss", round(-np.log(p[0, true]), 4))
dlogits = p.copy(); dlogits[0, true] -= 1
dW2 = h.T @ dlogits; dh = dlogits @ W2.T; dz1 = dh * (z1 > 0); dW1 = x.T @ dz1
print("dlogits", dlogits.round(4)); print("dW2\n", dW2.round(4)); print("dh", dh.round(4)); print("dW1\n", dW1.round(4))
def loss(W1):
    l = np.maximum(x @ W1, 0) @ W2; q = np.exp(l) / np.exp(l).sum(); return -np.log(q[0, true])
E = np.zeros_like(W1); E[1, 0] = 1e-6
print("finite-difference check of dW1[1,0]:", round((loss(W1 + E) - loss(W1 - E)) / 2e-6, 4), "vs", dW1[1, 0].round(4))
```

Output:

```
z1 [[2. 1.]] h [[2. 1.]] logits [[ 1. -1.]] p [[0.8808 0.1192]] loss 0.1269
dlogits [[-0.1192  0.1192]]
dW2
 [[-0.2384  0.2384]
 [-0.1192  0.1192]]
dh [[-0.2384  0.2384]]
dW1
 [[-0.2384  0.2384]
 [-0.4768  0.4768]]
finite-difference check of dW1[1,0]: -0.4768 vs -0.4768
```

## Implementation

Save as `part5.py` and run it (under a minute on a CPU). It needs numpy and scikit-learn (for the data).

```python
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True); X = X / 16.0                  # pixel values 0-16 -> 0-1
order = np.random.default_rng(0).permutation(len(y)); tr, te = order[:1000], order[1000:]

def init(seed, sizes=(64, 32, 10)):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, np.sqrt(2 / sizes[0]), (sizes[0], sizes[1])), "b1": np.zeros(sizes[1]),
            "W2": r.normal(0, np.sqrt(2 / sizes[1]), (sizes[1], sizes[2])), "b2": np.zeros(sizes[2])}

def forward(P, A):
    z1 = A @ P["W1"] + P["b1"]; h = np.maximum(z1, 0)                # layer 1 + ReLU
    logits = h @ P["W2"] + P["b2"]                                   # layer 2
    e = np.exp(logits - logits.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)   # softmax
    return z1, h, p

def loss_fn(P, A, t):
    p = forward(P, A)[2]; return -np.mean(np.log(p[np.arange(len(t)), t]))      # cross-entropy

def backward(P, A, t):                                               # backpropagation, by hand
    z1, h, p = forward(P, A); n = len(t)
    dlogits = p.copy(); dlogits[np.arange(n), t] -= 1; dlogits /= n  # d loss / d logits = (p - onehot) / n
    dh = dlogits @ P["W2"].T; dz1 = dh * (z1 > 0)                    # through layer 2, then through ReLU
    return {"W2": h.T @ dlogits, "b2": dlogits.sum(0), "W1": A.T @ dz1, "b1": dz1.sum(0)}

# 1. Check the hand-written gradient against finite differences, on a small batch.
P = init(0); A, t = X[tr[:20]], y[tr[:20]]; G = backward(P, A, t); worst = 0
rng = np.random.default_rng(1)
for name in P:
    for _ in range(8):
        idx = tuple(rng.integers(0, s) for s in P[name].shape); old = P[name][idx]; h = 1e-5
        P[name][idx] = old + h; lp = loss_fn(P, A, t); P[name][idx] = old - h; lm = loss_fn(P, A, t); P[name][idx] = old
        num = (lp - lm) / (2 * h); worst = max(worst, abs(num - G[name][idx]) / max(1e-8, abs(num) + abs(G[name][idx])))
print(f"1. gradient check: worst relative error over 32 random parameters = {worst:.1e}")
np.savez("check5.npz", **init(0), A=A, t=t)

# 2. Train with mini-batch gradient descent.
def train(seed, epochs=30, lr=0.5, batch=50):
    P = init(seed); r = np.random.default_rng(seed + 1000); log = []
    for ep in range(1, epochs + 1):
        idx = r.permutation(tr)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]; G = backward(P, X[b], y[b])
            for k in P: P[k] -= lr * G[k]
        if True:
            log.append((ep, loss_fn(P, X[tr], y[tr]), 100 * (forward(P, X[tr])[2].argmax(1) == y[tr]).mean(), 100 * (forward(P, X[te])[2].argmax(1) == y[te]).mean()))
    return P, log
print("\n2. training (64 -> 32 -> 10, 2,410 parameters, batch 50, learning rate 0.5)")
print("   epoch   train loss   train acc   test acc")
P, log = train(0)
for ep, l, a, b in [r for r in log if r[0] in (1, 2, 5, 10, 30)]: print(f"   {ep:5d}   {l:10.3f}   {a:8.1f}%   {b:7.1f}%")
print(f"   (chance: loss {np.log(10):.3f}, accuracy 10%)")

# 3. Different starting weights give different results: five seeds.
accs = [train(s)[1][-1][3] for s in range(5)]
np.save("accs5.npy", np.array(accs))
print(f"\n3. five seeds, test accuracy: {np.round(accs, 1)}  mean {np.mean(accs):.1f}%  std {np.std(accs):.2f}")
print("   for comparison (Part 4): nearest average 89.2%, 1-NN 98.4%")
np.save("log5.npy", np.array(log))
```

Output:

```
1. gradient check: worst relative error over 32 random parameters = 5.1e-09

2. training (64 -> 32 -> 10, 2,410 parameters, batch 50, learning rate 0.5)
   epoch   train loss   train acc   test acc
       1        1.025       66.1%      66.2%
       2        0.406       89.8%      90.1%
       5        0.206       94.3%      92.6%
      10        0.099       98.0%      96.6%
      30        0.058       98.6%      96.2%
   (chance: loss 2.303, accuracy 10%)

3. five seeds, test accuracy: [96.2 95.6 97.4 96.9 97. ]  mean 96.6%  std 0.62
   for comparison (Part 4): nearest average 89.2%, 1-NN 98.4%
```

The PyTorch comparison, saved as `torch5.py` and run after `part5.py` in the same folder (install CPU PyTorch from the [official page](https://pytorch.org/get-started/locally/)):

```python
import numpy as np, torch
from sklearn.datasets import load_digits
torch.set_default_dtype(torch.float64)
c = np.load("check5.npz")                       # created by part5.py: the weights and 20-image batch of the numpy gradient check
W = {k: torch.tensor(c[k], requires_grad=True) for k in ("W1", "b1", "W2", "b2")}
A, t = torch.tensor(c["A"]), torch.tensor(c["t"])

# 1. Autograd gives the gradient without writing any backward formula.
logits = torch.relu(A @ W["W1"] + W["b1"]) @ W["W2"] + W["b2"]
loss = torch.nn.functional.cross_entropy(logits, t)          # softmax + cross-entropy in one call
loss.backward()
print(f"1. loss from autograd on the 20-image batch: {loss.item():.6f}")

# The same gradient, with the hand-written formulas of part5.py:
Wn = {k: c[k] for k in W}; z1 = c["A"] @ Wn["W1"] + Wn["b1"]; h = np.maximum(z1, 0); lg = h @ Wn["W2"] + Wn["b2"]
p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); n = len(c["t"])
dl = p.copy(); dl[np.arange(n), c["t"]] -= 1; dl /= n; dz1 = (dl @ Wn["W2"].T) * (z1 > 0)
hand = {"W2": h.T @ dl, "b2": dl.sum(0), "W1": c["A"].T @ dz1, "b1": dz1.sum(0)}
for k in W: print(f"   {k}: largest difference, hand-written vs autograd = {np.abs(hand[k] - W[k].grad.numpy()).max():.1e}")

# 2. The same network trained in PyTorch's own style.
X, y = load_digits(return_X_y=True); X = torch.tensor(X / 16.0); y = torch.tensor(y)
order = torch.tensor(np.random.default_rng(0).permutation(len(y))); tr, te = order[:1000], order[1000:]
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))
opt = torch.optim.SGD(net.parameters(), lr=0.5)
for epoch in range(30):
    idx = tr[torch.randperm(len(tr))]
    for s in range(0, len(idx), 50):
        b = idx[s:s + 50]
        opt.zero_grad()                                       # clear old gradients
        torch.nn.functional.cross_entropy(net(X[b]), y[b]).backward()
        opt.step()                                            # w = w - lr * gradient, for every parameter
print(f"\n2. PyTorch version, 30 epochs: test accuracy {100 * (net(X[te]).argmax(1) == y[te]).float().mean():.1f}%")
```

Output:

```
1. loss from autograd on the 20-image batch: 2.493916
   W1: largest difference, hand-written vs autograd = 4.2e-17
   b1: largest difference, hand-written vs autograd = 2.8e-17
   W2: largest difference, hand-written vs autograd = 2.8e-17
   b2: largest difference, hand-written vs autograd = 8.3e-17

2. PyTorch version, 30 epochs: test accuracy 97.1%
```

The PyTorch version uses its own default initialisation, so its 97.1% is one run, not a direct replacement of the numpy mean.

## Try it

1. Change the hidden size from 32 to 4, then to 128, and compare the five-seed means. Where does adding units stop helping?
2. Set the learning rate to 5.0 and then to 0.01. Describe both curves in terms of Part 3.
3. Break the backward pass on purpose (for example drop the `/ n`) and see what the gradient check reports.
