---
title: "Inside the Math, Part 2: Softmax Meets Cross-Entropy, and the Gradient p - y"
date: 2027-05-24 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, softmax, cross-entropy, gradient, backpropagation, transformer, inside the math]
description: "Part 2 of an 8-part series on the classic maths inside language models. The Jacobian of softmax is diag(p) - p p^T, and when it meets the cross-entropy loss everything collapses into the gradient p - y. Checked against finite differences and autograd on a trained transformer, in float64."
---

## What you will do, and what you get

You will compute, by hand and in code, how the loss of a language model changes when each of the 48 next-character scores changes. Four results, all checked on a trained 110,384-parameter character transformer in float64:

- **The Jacobian of softmax is `diag(p) - p p^T`.** Against finite differences the largest difference is 1.1e-10, and every row sums to zero (the probabilities keep summing to 1).
- **The gradient of the cross-entropy loss with respect to the scores is `p - y`**, the predicted probabilities minus the one-hot truth. PyTorch autograd agrees to 2.8e-17.
- **That gradient is large when the model is confidently wrong** (norm 1.01 at a probability of 0.0000 for the true character) and small when it is confidently right (0.016 at 0.98). The same quantity with squared error on the probabilities is 0.00001 in the confidently-wrong case: the model would learn almost nothing from its worst mistakes.
- **`log(softmax(s))` returns `-inf`** for a score gap of 200 in float32, while `log_softmax` returns -203.85.

![Left: the 6 by 6 corner of the softmax Jacobian for the six most probable characters of one real context, with 0.053 for the top character on the diagonal and negative values off the diagonal. Middle: for the same context the model probability, the one-hot truth and their difference p - y over the true character e (probability 0.944, gradient -0.056) and four others with small positive gradients. Right: gradient norm against the score margin of the true character on a log scale, for cross-entropy (about 1 until the margin is 0, then falling to 0.016 at +8) and for squared error on the probabilities (0.00001 at margin -8, peaking at 0.24 at +4)]({{ site.baseurl }}/assets/pic/mm-2-softmax-cross-entropy.png){: width="900" }

## How: the steps

1. **Take the scores `z` and the true next character `y`** (an index from 0 to 47). `p = softmax(z)` as in Part 1.
2. **The loss** for one position is the negative log probability of the true character: `L = -log p_y`. In bits, divide by `ln 2`. Averaged over positions, this is the bits-per-character number used throughout the series.
3. **Differentiate in two stages.** First `dL/dp`, which is `-1 / p_y` at the true character and zero elsewhere. Then multiply by the Jacobian of softmax, `dp/dz`.
4. **Simplify.** The product collapses to `dL/dz = p - y`.
5. **Verify numerically.** Perturb each score by plus and minus a small number, recompute the loss, and compare; also compare with `loss.backward()`.
6. **Use the result.** `p - y` is the first thing backpropagation sends into the network, starting at the output layer; Part 7 follows it back.

## Why it works: the principle

**Softmax couples its outputs.** Raising one score raises its probability and, since the probabilities sum to 1, lowers all the others. The Jacobian `J_ij = dp_i / dz_j` is `p_i (1 - p_i)` on the diagonal and `-p_i p_j` off it, i.e. `J = diag(p) - p p^T`. Each row sums to zero. For two scores it reduces to the familiar `p (1 - p)` of the sigmoid.

**Why the chain rule collapses.** With `L = -log p_y`, `dL/dp_i = -y_i / p_i`. Then `dL/dz_j = sum_i (dL/dp_i) J_ij = -(1 / p_y) * p_y (y_j - p_j) = p_j - y_j`: the `p_y` in the denominator cancels the `p_y` in the Jacobian. The result has a simple reading: for every character, push its score down by its predicted probability, and push the true character's score up by `1 - p_y`. When `p` already equals `y` the push is zero.

**Why this pairing is the standard one.** The [Deep Learning book (section 6.2)](https://www.deeplearningbook.org/contents/mlp.html) states the design principle: the gradient of the cost "must be large and predictable enough to serve as a good guide for the learning algorithm", and functions that saturate (become very flat) undermine this because the gradient becomes very small. It adds that the log in the negative log-likelihood undoes the exp of output units such as softmax. The experiment shows the principle in numbers: with cross-entropy, the gradient norm is about 1.01 for any strongly wrong prediction and falls only as the model becomes right; with squared error on the probabilities it is 0.00001 at a margin of -8, because the same softmax saturates and flattens the loss exactly where the model is most wrong.

**Where `-log` meets `exp` numerically.** Computing `log(softmax(s))` as two steps first rounds the tiny probability to zero (`exp(-200)` underflows in float32), and the log of zero is `-inf`. `log_softmax` uses `z_i - logsumexp(z)` from Part 1 and returns the correct -203.85. This is why `torch.nn.functional.cross_entropy` takes *scores*, not probabilities.

**Soft targets.** If the target is a distribution `q` instead of a one-hot `y` (for example a smoothed label or a teacher model's probabilities), the same algebra gives `p - q`, provided `q` sums to 1.

**Pitfalls and limits.**
- **Passing probabilities to `cross_entropy`.** The function applies `log_softmax` itself; giving it softmax output applies softmax twice and flattens the training signal.
- **The gradient sums to zero over the 48 scores** (3e-16 in the experiment). That reflects the shift invariance of Part 1: moving all scores together does not change the loss, so no gradient flows in that direction.
- **Nats and bits.** PyTorch reports the loss in nats (natural log). The conversion to bits divides by `ln 2 = 0.693`. The gradient `p - y` is the gradient of the nats loss.
- **Float32 is enough for training but not for a derivative check.** The checks above run in float64, where finite differences with `eps = 1e-6` are accurate to about 1e-10; in float32 the same check is noisy.
- **One position, one context.** The numbers shown are for one real context; the sums over a batch add up the `p - y` vectors of all positions (divided by their number for a mean loss).

## Example, worked through

Scores `z = [2, 1, 0]`, true class 1.

- `p = [0.665, 0.245, 0.090]`; the loss is `-ln 0.245 = 1.408` nats, or 2.031 bits.
- `dL/dp = [0, -1/0.245, 0] = [0, -4.08, 0]`.
- Jacobian row for class 1: `[-0.163, 0.185, -0.022]` (`-p_1 p_0`, `p_1 (1 - p_1)`, `-p_1 p_2`).
- Chain rule: `dL/dz_0 = -4.08 * (-0.163) = 0.665`, `dL/dz_1 = -4.08 * 0.185 = -0.755`, `dL/dz_2 = -4.08 * (-0.022) = 0.090`.
- Shortcut: `p - y = [0.665, -0.755, 0.090]`. Same numbers. The true class's score is pushed up, the other two down, and the three entries sum to zero.

```python
import numpy as np
z = np.array([2.0, 1.0, 0.0]); y = np.array([0, 1, 0])                   # the true class is number 1
p = np.exp(z) / np.exp(z).sum()
print("p =", p.round(3), " loss =", round(-np.log(p[1]), 3), "nats =", round(-np.log2(p[1]), 3), "bits")
J = np.diag(p) - np.outer(p, p); dL_dp = -y / p                           # d(-log p_y)/dp: only the true class is non-zero
print("Jacobian:\n", J.round(3))
print("chain rule  J^T dL/dp =", (J.T @ dL_dp).round(3))
print("shortcut    p - y     =", (p - y).round(3))
eps = 1e-6; L = lambda s: -np.log((np.exp(s) / np.exp(s).sum())[1])
print("finite differences    =", np.array([(L(z + eps * np.eye(3)[j]) - L(z - eps * np.eye(3)[j])) / (2 * eps) for j in range(3)]).round(3))
```

Output:

```
p = [0.665 0.245 0.09 ]  loss = 1.408 nats = 2.031 bits
Jacobian:
 [[ 0.223 -0.163 -0.06 ]
 [-0.163  0.185 -0.022]
 [-0.06  -0.022  0.082]]
chain rule  J^T dL/dp = [ 0.665 -0.755  0.09 ]
shortcut    p - y     = [ 0.665 -0.755  0.09 ]
finite differences    = [ 0.665 -0.755  0.09 ]
```

## Implementation

`part2.py` loads the model saved by `part8.py` from Part 1 of this series (`tiny8.pt`, a 110,384-parameter character transformer, about 4 minutes to train on a CPU) and runs the four checks in float64. Keep `part8.py`, `tiny8.pt` and this script in one folder.

```python
import json, math, numpy as np, torch, torch.nn.functional as F
from part8 import Tiny, chars, windows, test, T
torch.set_default_dtype(torch.float64)
m = Tiny().double(); m.load_state_dict({k: v.double() for k, v in torch.load("tiny8.pt").items()}); m.eval()
xb, yb = windows(test, [0, 500, 1000, 1500])                              # 4 windows of 64 characters
with torch.no_grad(): z = m(xb)[:, -1]                                     # scores for the last position of each window: (4, 48)
y = yb[:, -1]                                                              # the true next characters

print("1. Jacobian of softmax: diag(p) - p p^T, versus finite differences (one context)")
p = F.softmax(z[0], -1); J = torch.diag(p) - torch.outer(p, p)
eps = 1e-6; Jn = torch.stack([(F.softmax(z[0] + eps * torch.eye(48)[j], -1) - F.softmax(z[0] - eps * torch.eye(48)[j], -1)) / (2 * eps) for j in range(48)], 1)
print(f"   largest absolute difference {(J - Jn).abs().max().item():.2e}; each row sums to {J.sum(1).abs().max().item():.1e} (probabilities stay summing to 1)")

print("\n2. gradient of cross-entropy with respect to the scores: p - onehot(y)")
zz = z.clone().requires_grad_(True); loss = F.cross_entropy(zz, y, reduction="sum"); loss.backward()
manual = F.softmax(z, -1) - F.one_hot(y, 48)
print(f"   autograd vs p - y: largest difference {(zz.grad - manual).abs().max().item():.2e}")
i = 0; print(f"   context 0: true character {chars[y[i]]!r}, probability {F.softmax(z[i], -1)[y[i]].item():.3f}, gradient at the true character {manual[i, y[i]].item():+.3f}, sum of gradient over all 48 scores {manual[i].sum().item():.1e}")

print("\n3. the gradient vanishes when the model is confidently RIGHT and stays large when confidently WRONG")
rows = []
for margin in (-8, -4, 0, 4, 8):                                           # score of the true class minus the others (others all 0), 48 classes
    s = torch.zeros(48); s[7] = margin; pr = F.softmax(s, -1); g = pr.clone(); g[7] -= 1
    sq = s.clone().requires_grad_(True); ((F.softmax(sq, -1) - F.one_hot(torch.tensor(7), 48)) ** 2).sum().backward()      # squared error on the probabilities, for contrast
    rows.append({"margin": margin, "p_true": pr[7].item(), "loss_bits": -math.log2(pr[7].item()), "grad_norm": g.norm().item(), "mse_grad_norm": sq.grad.norm().item()})
    print(f"   margin {margin:+d}: probability of the true character {pr[7].item():.4f}, cross-entropy {rows[-1]['loss_bits']:.3f} bits with gradient norm {g.norm().item():.4f}; squared-error gradient norm {sq.grad.norm().item():.5f}")
print("\n4. log-softmax, not log(softmax): scores of size 200")
s = torch.zeros(48, dtype=torch.float32); s[7] = -200.
print("   log(softmax(s))[7] in float32:", torch.log(F.softmax(s, -1))[7].item(), "  log_softmax(s)[7]:", F.log_softmax(s, -1)[7].item())
json.dump({"rows": rows, "p": F.softmax(z[0], -1).tolist(), "J": J.tolist(), "chars": chars, "y": int(y[0])}, open("results2.json", "w"))
```

Output:

```
1. Jacobian of softmax: diag(p) - p p^T, versus finite differences (one context)
   largest absolute difference 1.08e-10; each row sums to 3.2e-16 (probabilities stay summing to 1)

2. gradient of cross-entropy with respect to the scores: p - onehot(y)
   autograd vs p - y: largest difference 2.78e-17
   context 0: true character 'e', probability 0.944, gradient at the true character -0.056, sum of gradient over all 48 scores 3.0e-16

3. the gradient vanishes when the model is confidently RIGHT and stays large when confidently WRONG
   margin -8: probability of the true character 0.0000, cross-entropy 17.096 bits with gradient norm 1.0106; squared-error gradient norm 0.00001
   margin -4: probability of the true character 0.0004, cross-entropy 11.326 bits with gradient norm 1.0102; squared-error gradient norm 0.00080
   margin +0: probability of the true character 0.0208, cross-entropy 5.585 bits with gradient norm 0.9895; squared-error gradient norm 0.04123
   margin +4: probability of the true character 0.5374, cross-entropy 0.896 bits with gradient norm 0.4675; squared-error gradient norm 0.23739
   margin +8: probability of the true character 0.9845, cross-entropy 0.023 bits with gradient norm 0.0157; squared-error gradient norm 0.00049

4. log-softmax, not log(softmax): scores of size 200
   log(softmax(s))[7] in float32: -inf   log_softmax(s)[7]: -203.8501434326172
```

## Try it

1. Replace the true character by the model's *most likely* wrong character and look at `p - y`: which entries are large?
2. Verify `p - q` for a soft target `q = 0.9 * onehot + 0.1 / 48` with autograd.
3. Compute the gradient of the loss for the whole window of 64 positions at once and check that its shape is `(64, 48)` and that every row sums to zero.
