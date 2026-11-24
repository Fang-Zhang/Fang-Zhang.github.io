---
title: "Inside the Math, Part 5: The Feed-Forward Block, Two Matrices and a Bend"
date: 2027-05-27 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, feed-forward network, mlp, gelu, transformer, inside the math]
description: "Part 5 of an 8-part series on the classic maths inside language models. The feed-forward block of a transformer is two matrix products with a nonlinearity between them and holds two thirds of each block's parameters (66.2 percent in a 110,384-parameter model). Written out by hand, matched to PyTorch, and read as keys and values."
---

## What you will do, and what you get

You will take apart the feed-forward block (also called the MLP block) of a trained 110,384-parameter character transformer, rebuild it by hand, and read what its hidden units do. Results:

- **It is two matrix products with a bend in between.** `output = W2 * GELU(W1 * x + b1) + b2`, with `W1` of size 256 x 64 (widening each 64-number vector to 256 numbers) and `W2` of size 64 x 256. The hand-written version matches PyTorch to 3.6e-06.
- **It holds two thirds of the block.** One block has 33,088 MLP parameters against 16,640 for attention and 256 for the two LayerNorms: the MLP is 66.2 percent of a block.
- **It is sparse in use.** For 31.8 percent of (position, unit) pairs the pre-activation is positive, and on average 58 of the 256 units are strongly active (activation above 0.5 in size) at a position.
- **Each hidden unit reads a pattern and writes a vector.** A row of `W1` is a "key" the input is matched against; the matching column of `W2` is a "value" added to the output in proportion to the activation. In the last block, unit 118 fires most when the current character is `h`, `t` or `c`; its value raises the scores of `e`, `o`, `y` and lowers `t`, `f`, `c`.
- **One hidden layer can fit a curve, and more units fit it better.** For `sin(3x)` on [-2, 2], the mean squared error is 0.438 with 1 unit, 0.077 with 4 units, 0.00015 with 16 units and 0.00002 with 64.

![Left: the GELU and ReLU curves, with a histogram of the real pre-activations in the last block of the model, mostly negative. Middle: a table of six hidden units; for each, the characters on which it fires most (its key) and the characters whose scores its value raises and lowers. Right: the curve sin(3x) in grey with fits from 1, 4 and 16 hidden units, with mean squared errors 0.4379, 0.0767 and 0.0001]({{ site.baseurl }}/assets/pic/mm-5-feedforward.png){: width="900" }

## How: the steps

1. **Take one vector `x` of 64 numbers**, the output of the LayerNorm in front of the MLP (Part 3), one for every position.
2. **Widen:** `u = W1 x + b1`, a vector of 256 numbers. In the original transformer the inner layer is four times wider than the model: [Vaswani and colleagues (2017)](https://arxiv.org/abs/1706.03762) use 2,048 inner units for a model width of 512, and this model uses 256 for 64.
3. **Bend:** `h = GELU(u)`, applied to each of the 256 numbers on its own. Without it, the two matrices would multiply into one matrix and the block would be a single linear map.
4. **Narrow:** `out = W2 h + b2`, back to 64 numbers, which are *added* to the residual stream.
5. **Apply the same function to every position separately.** Positions do not talk to each other in this block (attention does that); the original paper describes the block as applied to each position separately and identically.
6. **Read the block as memory.** Write `out` as a sum over hidden units, `sum_k h_k * (column k of W2)`, with each `h_k` set by how well `x` matches row `k` of `W1`.
7. **Test the capacity** of such a block in isolation on a curve.

## Why it works: the principle

**The nonlinearity.** GELU is `x * Phi(x)`, where `Phi` is the standard Gaussian cumulative distribution function, proposed by [Hendrycks and Gimpel](https://arxiv.org/abs/1606.08415), who describe it as weighting inputs by their value, where ReLU gates them by their sign. It looks like ReLU (`max(0, x)`) but is smooth around zero and slightly negative for small negative inputs (GELU(-1) = -0.159). Without any such bend, two stacked linear maps are one linear map; with it, the block can compute functions no single matrix can.

**Why the inner layer is wide.** A wider hidden layer gives more directions to switch on and off. The curve fit shows the effect in miniature: with 1 unit the fit is nearly a straight line; with 4 it follows the first waves only roughly; with 16 it is indistinguishable from the target by eye. The [Deep Learning book (section 6.4)](https://www.deeplearningbook.org/contents/mlp.html) states the universal approximation theorem: a feedforward network with a linear output layer and at least one hidden layer with a "squashing" activation can approximate any Borel measurable function with any desired nonzero error, provided it has enough hidden units, and it adds that this does not guarantee that training will find that function (optimisation may fail, or the model may overfit).

**Where the parameters are.** Two matrices of `64 x 256` give `2 * 16,384 = 32,768` weights plus 320 biases: 33,088. [Geva and colleagues (2021)](https://arxiv.org/abs/2012.14913) note that feed-forward layers constitute two thirds of a transformer model's parameters, which this block reproduces (66.2 percent of a block, 66.5 percent of MLP plus attention).

**Keys and values.** Geva and colleagues show that feed-forward layers in transformer language models operate as key-value memories: each key correlates with textual patterns in the training examples, and each value induces a distribution over the output vocabulary. The table in the figure is a small version of their analysis in this model. For each unit, the key side is the character on which its average activation is highest; the value side is the unit's column of `W2` multiplied by the output layer's weights, which gives the direct effect of the unit on each character's score. Unit 118 is a pattern one could name: after `h`, `t` or `c`, make `e`, `o` or `y` more likely and `t`, `f`, `c` less likely.

**Gated variants.** [Shazeer (2020)](https://arxiv.org/abs/2002.05202) tests variants of this block with a gate (the component-wise product of two linear projections, one passed through a nonlinearity) and reports quality improvements over the usual ReLU or GELU feed-forward layers in the Transformer. Many recent models use one of these variants; the block in this post is the plain version.

**Pitfalls and limits.**
- **The table is a heuristic.** It reads only the *direct* effect of the unit on the next-character scores, ignoring the final LayerNorm and the other 255 units, and it picks units by the spread of their activations. It is a description, not a causal test; removing the unit and measuring the loss would be the test.
- **Character level.** In a model over characters the keys are often single letters; in a word-level model the patterns are larger. The picture in Geva and colleagues' analysis covers models trained on words and subwords.
- **A fit is not a proof.** The curve fit shows that more units fit better on 200 points of one function with one seed; it does not show how many units a real task needs.
- **Representable is not learnable.** The theorem says an approximating network exists. Whether gradient descent finds it, and from how much data, is a separate question (Part 8 covers the optimiser).
- **Biases are part of the count.** `b1` has 256 numbers and `b2` has 64; they are small next to the matrices, but they are parameters.
- **One block of one model.** 31.8 percent positive and 58 of 256 strongly active are numbers for the last block on 16 windows of test text.

## Example, worked through

A block with 2 input features and 3 hidden units. `x = [1, 2]`. Rows of `W1`: `[1, 0]`, `[0, 1]`, `[1, 1]` with `b1 = [0, 0, 0.5]`. Columns of `W2`: `[1, 0]`, `[0, 1]`, `[1, -1]`.

- Pre-activations (key times input, plus bias): `[1, 2, 3.5]`.
- GELU: `[0.841, 1.954, 3.499]`.
- Output as a sum of values: `0.841 * [1, 0] + 1.954 * [0, 1] + 3.499 * [1, -1] = [4.341, -1.545]`.
- The matrix form `W2 h + b2` gives the same.
- Contrast: GELU(-1) = -0.159 while ReLU(-1) = 0; near zero and below, the two differ.

```python
import numpy as np
from math import erf, sqrt
gelu = lambda u: np.array([0.5 * a * (1 + erf(a / sqrt(2))) for a in u])
x = np.array([1.0, 2.0])                                                    # input vector, 2 features
W1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]); b1 = np.array([0.0, 0.0, 0.5])      # 3 hidden units; each ROW of W1 is a key
W2 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]); b2 = np.zeros(2)       # each COLUMN of W2 is a value
pre = W1 @ x + b1; act = gelu(pre)
print("pre-activations (key . x + bias):", pre, "-> GELU:", act.round(3))
print("output, matrix form:", (W2 @ act + b2).round(3))
print("output, sum of activation * value:", sum(act[k] * W2[:, k] for k in range(3)).round(3), " = ", " + ".join(f"{act[k]:.3f}*{W2[:, k]}" for k in range(3)))
print("ReLU instead of GELU:", np.maximum(pre, 0), "(for positive inputs the two are close; they differ near 0)", " GELU(-1) =", gelu(np.array([-1.0]))[0].round(3), " ReLU(-1) = 0")
```

Output:

```
pre-activations (key . x + bias): [1.  2.  3.5] -> GELU: [0.841 1.954 3.499]
output, matrix form: [ 4.341 -1.545]
output, sum of activation * value: [ 4.341 -1.545]  =  0.841*[1. 0.] + 1.954*[0. 1.] + 3.499*[ 1. -1.]
ReLU instead of GELU: [1.  2.  3.5] (for positive inputs the two are close; they differ near 0)  GELU(-1) = -0.159  ReLU(-1) = 0
```

## Implementation

`part5.py` loads the model trained by `part8.py` (see Part 1 of this series; 110,384 parameters, about 4 minutes on a CPU), rebuilds the last block's MLP by hand, measures the activation statistics, builds the key-value table and fits the curves. Keep `part8.py`, `tiny8.pt` and the script in one folder.

```python
import json, math, numpy as np, torch, torch.nn.functional as F
from part8 import Tiny, chars, windows, test, T, D, V
m = Tiny(); m.load_state_dict(torch.load("tiny8.pt")); m.eval()

print("1. where the parameters are (one block)")
b = m.blocks[0]; mlp = sum(p.numel() for p in b.mlp.parameters()); att = sum(p.numel() for p in list(b.qkv.parameters()) + list(b.proj.parameters())); ln = sum(p.numel() for p in list(b.ln1.parameters()) + list(b.ln2.parameters()))
print(f"   MLP {mlp:,} (two matrices {D}x{4*D} and {4*D}x{D} plus biases), attention {att:,}, two LayerNorms {ln:,}; MLP is {100*mlp/(mlp+att+ln):.1f}% of a block, {100*mlp/(mlp+att):.1f}% of MLP+attention")

print("\n2. the MLP is two matrix products and a nonlinearity: hand-written vs PyTorch")
xb, yb = windows(test, list(range(0, 64 * 16, 64)))
store = {}; hs = [m.blocks[1].ln2.register_forward_hook(lambda mod, i, o: store.update(inp=o.detach())), m.blocks[1].mlp.register_forward_hook(lambda mod, i, o: store.update(out=o.detach()))]
with torch.no_grad(): m(xb)
mlp2 = m.blocks[1].mlp; W1, b1, W2, b2 = mlp2[0].weight.detach(), mlp2[0].bias.detach(), mlp2[2].weight.detach(), mlp2[2].bias.detach()
gelu = lambda u: 0.5 * u * (1 + torch.erf(u / math.sqrt(2)))
pre = store["inp"] @ W1.T + b1; hid = gelu(pre); out = hid @ W2.T + b2
print(f"   shapes: input {tuple(store['inp'].shape)} -> hidden {tuple(hid.shape)} -> output {tuple(out.shape)};  largest difference to nn.Sequential {(out - store['out']).abs().max().item():.1e}")

print("\n3. how many of the 256 hidden units are 'on' (block 2)")
on = (pre > 0).float().mean().item(); strong = (hid.abs() > 0.5).float().mean().item()
print(f"   pre-activation positive for {100*on:.1f}% of (position, unit) pairs; |activation| > 0.5 for {100*strong:.1f}%; on average {strong*256:.0f} of 256 units are strongly active per position")

print("\n4. keys and values: what a hidden unit reads and what it writes (block 2, direct effect on the next-character scores)")
ids = xb.reshape(-1); H = hid.reshape(-1, 4 * D)
U = m.out.weight.detach()                                                   # 48 x 64 unembedding
effect = (W2.T @ U.T)                                                       # (256 units) x (48 characters): direct logit effect of each unit's value vector
order = H.std(0).argsort(descending=True)[:6]; table = []
show = lambda c: "space" if c == " " else ("newline" if c == "\n" else c)
for k in order.tolist():
    act = torch.stack([H[ids == c, k].mean() if (ids == c).sum() > 20 else torch.tensor(-9.) for c in range(V)])
    reads = [show(chars[c]) for c in act.topk(3).indices.tolist()]; writes = [show(chars[c]) for c in effect[k].topk(3).indices.tolist()]; avoids = [show(chars[c]) for c in (-effect[k]).topk(3).indices.tolist()]
    table.append({"unit": k, "reads": reads, "writes": writes, "avoids": avoids, "std": H[:, k].std().item()})
    print(f"   unit {k:3d}: fires most when the current character is {reads}; its value raises {writes} and lowers {avoids}")

print("\n5. one hidden layer can fit a curve: y = sin(3x) on [-2, 2], GELU, 1500 Adam steps")
xs = torch.linspace(-2, 2, 200)[:, None]; ys = torch.sin(3 * xs); fits = {}
for h in (1, 2, 4, 16, 64):
    torch.manual_seed(0); net = torch.nn.Sequential(torch.nn.Linear(1, h), torch.nn.GELU(), torch.nn.Linear(h, 1)); opt = torch.optim.Adam(net.parameters(), 1e-2)
    for _ in range(1500): loss = F.mse_loss(net(xs), ys); opt.zero_grad(); loss.backward(); opt.step()
    fits[h] = {"mse": loss.item(), "curve": net(xs).detach().flatten().tolist(), "params": sum(p.numel() for p in net.parameters())}
    print(f"   {h:3d} hidden units ({fits[h]['params']:3d} parameters): mean squared error {loss.item():.5f}")
json.dump({"table": table, "fits": fits, "xs": xs.flatten().tolist(), "ys": ys.flatten().tolist(), "frac_on": on, "frac_strong": strong, "mlp": mlp, "att": att, "ln": ln,
           "pre_hist": np.histogram(pre.flatten().numpy(), bins=60, range=(-6, 6))[0].tolist()}, open("results5.json", "w"))
```

Output:

```
1. where the parameters are (one block)
   MLP 33,088 (two matrices 64x256 and 256x64 plus biases), attention 16,640, two LayerNorms 256; MLP is 66.2% of a block, 66.5% of MLP+attention

2. the MLP is two matrix products and a nonlinearity: hand-written vs PyTorch
   shapes: input (16, 64, 64) -> hidden (16, 64, 256) -> output (16, 64, 64);  largest difference to nn.Sequential 3.6e-06

3. how many of the 256 hidden units are 'on' (block 2)
   pre-activation positive for 31.8% of (position, unit) pairs; |activation| > 0.5 for 22.5%; on average 58 of 256 units are strongly active per position

4. keys and values: what a hidden unit reads and what it writes (block 2, direct effect on the next-character scores)
   unit  34: fires most when the current character is ['space', 'newline', 'l']; its value raises ['"', 'o', 'b'] and lowers ['space', '-', ':']
   unit 118: fires most when the current character is ['h', 't', 'c']; its value raises ['e', 'o', 'y'] and lowers ['t', 'f', 'c']
   unit 127: fires most when the current character is ['space', 'newline', 'i']; its value raises ['c', 'p', '('] and lowers ['y', 'l', 'r']
   unit 168: fires most when the current character is ['space', '"', 'e']; its value raises ['e', 'o', 'n'] and lowers ['y', '-', 'c']
   unit  30: fires most when the current character is ['newline', 'n', 's']; its value raises ['o', 'i', 'y'] and lowers ['p', 's', 'j']
   unit  38: fires most when the current character is ['c', 'h', 't']; its value raises ['r', 'l', 'o'] and lowers ['z', 'g', '8']

5. one hidden layer can fit a curve: y = sin(3x) on [-2, 2], GELU, 1500 Adam steps
     1 hidden units (  4 parameters): mean squared error 0.43786
     2 hidden units (  7 parameters): mean squared error 0.28051
     4 hidden units ( 13 parameters): mean squared error 0.07666
    16 hidden units ( 49 parameters): mean squared error 0.00015
    64 hidden units (193 parameters): mean squared error 0.00002
```

## Try it

1. Replace `GELU` by `ReLU` in the curve fit and compare the errors for 4 and 16 hidden units.
2. Zero out hidden unit 118 of the last block and measure the change in bits per character on the test text: does the unit matter as much as the table suggests?
3. Print the same table for the first block. Do its units read and write simpler or more complex patterns?
