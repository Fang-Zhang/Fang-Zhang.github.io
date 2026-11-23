---
title: "From Zero to ML Engineer, Part 8: Your First Neural Network"
date: 2027-05-06 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, neural network, pytorch, backpropagation, ML engineer]
description: "Part 8 of a 12-part series for complete beginners. A neuron is a weighted sum plus a bend; a network stacks them. Train one in PyTorch with the standard four-line loop, see it solve a problem a straight line cannot, and see it read handwritten digits at 97.4%."
---

## What you will do, and what you get

You will build a neural network from nothing but its definition, train it with PyTorch on a CPU in under a minute, and watch it do two things. First, it separates two interleaved half-circles where logistic regression, which can only draw a straight line, reaches 87.2% and the network reaches 95.0%. Second, it learns to read handwritten digits from 64 pixel values, climbing from 64.6% after one pass to 97.4% after forty. Nothing new is learned about *learning*: the loop is Part 3's loop, applied to many more weights at once.

![Three panels. Left: two interleaved half-circle classes with a straight dashed boundary from logistic regression and a bent solid boundary from the network. Middle: training loss falling from about 1.2 to 0.01 over 40 epochs while test accuracy rises from 65 to 97 percent. Right: ten 8 by 8 digit images with true and predicted labels, five correct and five mistakes]({{ site.baseurl }}/assets/pic/mlpath-8-network.png){: width="900" }

## How: the steps

1. **Define the network** as layers: `Linear` (weighted sums) alternating with an activation such as `ReLU` (the bend).
2. **Pick a loss.** For choosing one of several classes, cross-entropy.
3. **Pick an optimiser.** Stochastic gradient descent (SGD): gradient descent on small random batches of rows.
4. **Run the loop**, for each batch:
   - forward pass: compute predictions and the loss;
   - `opt.zero_grad()`: clear the previous batch's gradients (PyTorch accumulates them otherwise);
   - `loss.backward()`: compute the gradient for every weight;
   - `opt.step()`: move every weight a little downhill;
5. **Watch the loss and a held-out score** after each epoch (one pass over all training rows).

## Why it works: the principle

**A neuron** is the logistic-regression recipe from Part 5 without the squashing: take a weighted sum of the inputs, add a bias, and apply a simple bend. The most common bend is ReLU, `max(0, z)`: pass positive values, cut negative ones to zero. With weights (0.4, 0.3, -0.2), bias 0.1 and inputs (2.0, -1.0, 0.5), the sum is 0.8 - 0.3 - 0.1 + 0.1 = 0.50 and ReLU leaves it at 0.50. With inputs (-2.0, -1.0, 0.5) the sum is -1.10 and the output is 0. See the [PyTorch ReLU reference](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html).

**A network** feeds the outputs of many neurons into the next layer of neurons. The bend is what makes this worth doing: without it, a stack of weighted sums collapses into a single weighted sum, and the network is just logistic regression with extra steps. With the bend, each layer can build on the last: the left panel shows the effect on the half-circles, where the network's boundary curves around the data and the straight line cannot. The network used there has 354 weights (2 inputs, two layers of 16 neurons, 2 outputs).

**Backpropagation** is how the gradient for every weight is computed: apply the chain rule from calculus layer by layer, starting at the loss and working backwards, reusing each intermediate result. It was popularised by [Rumelhart, Hinton and Williams in 1986](https://www.nature.com/articles/323533a0). In PyTorch you never write it: `loss.backward()` does it for any combination of layers. What you write is the model, the loss and the loop; the [PyTorch optimisation tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) walks through the same loop.

**Why mini-batches and epochs.** Computing the gradient on all rows for each step is slow; computing it on 32 random rows is a noisy estimate that is much cheaper, and the noise averages out over many steps. One pass through all the training rows is an *epoch*. The digits network takes 1,257 images, so one epoch is 40 steps.

**What the digits data is.** 1,797 images of handwritten digits, each 8 by 8 pixels with grey levels from 0 to 16 (the program divides by 16 to scale to 0-1, for the reason in Part 5). Each image is a row of 64 numbers; there are 10 classes. See the [scikit-learn dataset page](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). The network is 64 inputs, 32 hidden neurons, 10 outputs: 2,410 weights in total.

**Do not expect networks to beat everything.** On this dataset, plain logistic regression already scores 97.0%, and the network reaches 97.4% on the test images: within noise of each other, as the spread in Part 6 would suggest. Networks earn their place when the problem is hard for a straight boundary (the half-circles), or when there is much more data. They also have more settings (layer sizes, learning rate, batch size, epochs), so there is more to get wrong.

## Example, worked through

- **Neuron:** inputs (2.0, -1.0, 0.5) give 0.50; inputs (-2.0, -1.0, 0.5) give -1.10, which ReLU cuts to 0.
- **Half-circles (180 test points):** logistic regression 87.2%, network 95.0%.
- **Digits, per epoch (train loss, test accuracy):** epoch 1: 1.195 and 64.6%; epoch 2: 0.303 and 91.9%; epoch 5: 0.144 and 94.8%; epoch 20: 0.032 and 96.5%; epoch 40: 0.010 and 97.4%.
- **Overfitting signal:** the training loss keeps falling towards 0 while the test accuracy flattens near 97%. That is the Part 6 gap in a new setting. A longer run would mostly memorise the training images.
- **The mistakes:** the five misclassified images in the right panel (a 1 read as 8, an 8 as 1, and so on) are drawn in only 8 by 8 pixels. Some are hard to read by eye.
- **A caveat on the numbers:** the program prints the test accuracy after every epoch only to draw the curve. If you used it to decide when to stop, it would become a validation set, and you would need a fresh test set (Part 4).

## Implementation

Save as `part8.py` and run it. It needs numpy, scikit-learn and PyTorch (`pip install torch scikit-learn`; the CPU build is enough). It takes under a minute.

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0); np.random.seed(0)

# 1. A neuron, by hand: weighted sum, then a bend (ReLU = max(0, z))
x = np.array([2.0, -1.0, 0.5]); w = np.array([0.4, 0.3, -0.2]); b = 0.1
z = w @ x + b
print(f"1. neuron: z = 0.4*2.0 + 0.3*(-1.0) + (-0.2)*0.5 + 0.1 = {z:.2f},  ReLU(z) = {max(0, z):.2f}")
x2 = np.array([-2.0, -1.0, 0.5]); z2 = w @ x2 + b
print(f"   other input (-2.0, -1.0, 0.5): z = {z2:.2f},  ReLU(z) = {max(0, z2):.2f}  (negative z is cut to 0)")

def train(model, Xtr, ytr, Xva, yva, epochs, lr, batch=None):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9); loss_fn = nn.CrossEntropyLoss(); hist = []
    Xtr, ytr, Xva, yva = map(torch.as_tensor, (Xtr, ytr, Xva, yva))
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr)) if batch else torch.arange(len(Xtr))
        for i in range(0, len(Xtr), batch or len(Xtr)):
            idx = perm[i:i + (batch or len(Xtr))]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])     # 1. forward pass and loss
            opt.zero_grad(); loss.backward()              # 2. gradients by backpropagation
            opt.step()                                    # 3. step downhill (Part 3, for every weight at once)
        with torch.no_grad():
            hist.append((loss_fn(model(Xtr), ytr).item(), (model(Xva).argmax(1) == yva).float().mean().item()))
    return hist

# 2. Two interleaved half-circles: a straight line cannot separate them, a small network can
Xm, ym = make_moons(600, noise=0.2, random_state=0)
Xa, Xb, ya, yb = train_test_split(Xm, ym, test_size=0.3, random_state=0)
lin = LogisticRegression().fit(Xa, ya)
net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2))
train(net, Xa.astype("float32"), ya, Xb.astype("float32"), yb, 300, 0.05)
with torch.no_grad(): acc_net = (net(torch.as_tensor(Xb.astype("float32"))).argmax(1).numpy() == yb).mean()
print(f"\n2. moons (180 test points): logistic regression {100*lin.score(Xb, yb):.1f}%   network 2-16-16-2 ({sum(p.numel() for p in net.parameters())} parameters) {100*acc_net:.1f}%")

# 3. Handwritten digits, 8x8 pixels, 10 classes
D = load_digits(); Xd = D.data / 16.0; yd = D.target
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, stratify=yd, random_state=0)
lr_ = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
print(f"\n3. digits: {len(Xtr)} train / {len(Xte)} test images, 64 pixels each. Logistic regression test accuracy {100*lr_.score(Xte, yte):.1f}%")
torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
hist = train(mlp, Xtr.astype("float32"), ytr, Xte.astype("float32"), yte, 40, 0.05, batch=32)
print(f"   network 64-32-10 ({sum(p.numel() for p in mlp.parameters())} parameters), minibatches of 32")
for ep in (0, 1, 2, 4, 9, 19, 39): print(f"   epoch {ep+1:2d}: train loss {hist[ep][0]:.3f}   test accuracy {100*hist[ep][1]:.1f}%")
print(f"   chance level (10 classes): 10.0%;  untrained network: ~10%")
np.save("hist8.npy", np.array(hist))
```

Output:

```
1. neuron: z = 0.4*2.0 + 0.3*(-1.0) + (-0.2)*0.5 + 0.1 = 0.50,  ReLU(z) = 0.50
   other input (-2.0, -1.0, 0.5): z = -1.10,  ReLU(z) = 0.00  (negative z is cut to 0)

2. moons (180 test points): logistic regression 87.2%   network 2-16-16-2 (354 parameters) 95.0%

3. digits: 1257 train / 540 test images, 64 pixels each. Logistic regression test accuracy 97.0%
   network 64-32-10 (2410 parameters), minibatches of 32
   epoch  1: train loss 1.195   test accuracy 64.6%
   epoch  2: train loss 0.303   test accuracy 91.9%
   epoch  3: train loss 0.207   test accuracy 93.5%
   epoch  5: train loss 0.144   test accuracy 94.8%
   epoch 10: train loss 0.086   test accuracy 95.6%
   epoch 20: train loss 0.032   test accuracy 96.5%
   epoch 40: train loss 0.010   test accuracy 97.4%
   chance level (10 classes): 10.0%;  untrained network: ~10%
```

## Try it

1. Change the hidden layer of the digits network from 32 neurons to 4, then to 256. How do the parameter count and the test accuracy move?
2. Remove the `nn.ReLU()` layers from the half-circles network. What accuracy do you get, and why does it match the straight line?
3. Set the learning rate to 2.0. What does the loss do, and which part of Part 3 explains it?
