---
title: "ML Engineer Interviews, With Reference Answers (6/18): Deep Learning and Training"
date: 2027-01-01 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, deep-learning, pytorch, optimization, series]
description: "Part 6 of 18. Ten interview questions on initialisation, gradients, optimisers, learning rates, normalisation, dropout and debugging a network that does not learn, with reference answers and PyTorch experiments whose code and output are shown."
---

This post covers the deep learning questions of the technical round: why initialisation matters, how optimisers and learning rates behave, what normalisation and dropout do, and how to debug a network that does not learn. It gives ten questions with what each tests, a reference answer, a common weak answer and a follow-up. Numbers in the answers come from code that was run for this post, shown with its output. The answers are the author's, for hypothetical candidates. [Part 5](/posts/ml-engineer-interviews-5-data-sampling-and-features/) covered data and features.

The code ran on a CPU-only machine with Python 3.14.7, PyTorch 2.15.0 and NumPy, with fixed seeds. Most experiments use a small multilayer perceptron on the 8x8 handwritten-digits data set that ships with scikit-learn (1,200 training rows, 597 test rows), because the point is the mechanism, not the benchmark. Other models and data will give other numbers.

![Left: activation standard deviation by layer in a 30-layer ReLU network on a log scale. With weights drawn from N(0, 0.01) it falls to about 1e-24, with Xavier initialisation to about 3e-5, and with He initialisation it stays near 0.85. Right: test accuracy after 30 epochs against learning rate from 1e-4 to 1 for SGD, SGD with momentum and Adam; each optimiser has a different working range.]({{ "/assets/pic/ml-interview-6-init-learning-rate.png" | relative_url }})
*Left: initialisation scale decides whether the signal survives depth (questions 1 and 2). Right: working learning-rate ranges differ by optimiser (questions 3 and 4).*

## 1. "Why does weight initialisation matter?"

**What it tests.** Whether the candidate can reason about the scale of activations and gradients through depth.

**Reference answer.** Each layer multiplies the signal by a weight matrix. If the weights are too small, the activations shrink layer by layer until the signal and the gradients vanish; if too large, they grow until they explode or saturate the nonlinearities. Initialisation should keep the variance of activations, and of gradients, roughly constant across layers. [Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) study why standard gradient descent from random initialisation does poorly in deep networks and propose a scheme that brings substantially faster convergence, motivated by keeping the singular values of each layer's Jacobian near 1. [He et al. (2015)](https://arxiv.org/abs/1502.01852) derive an initialisation that specifically accounts for rectifier nonlinearities and show that it lets them train extremely deep rectified models directly from scratch. For ReLU, a layer with n inputs uses weights with variance 2/n; the factor 2 compensates for ReLU zeroing about half the units.

An experiment with a 30-layer ReLU network, 512 units per layer and random inputs, tracking the standard deviation of the activations:

```python
import numpy as np
rng = np.random.default_rng(0)
x0 = rng.normal(size=(1000, 512)); L = 30
def run(name, std_fn):
    x = x0.copy(); stds = []
    for _ in range(L):
        W = rng.normal(0, std_fn(512), size=(512, 512)); x = np.maximum(x @ W, 0); stds.append(x.std())
    print(f"{name:28} activation std after layer 1: {stds[0]:.2e}, 10: {stds[9]:.2e}, 30: {stds[-1]:.2e}")
run("N(0, 0.01)", lambda n: 0.01)
run("Xavier/Glorot sqrt(1/n)", lambda n: np.sqrt(1/n))
run("He sqrt(2/n)", lambda n: np.sqrt(2/n))
```

```
N(0, 0.01)                   activation std after layer 1: 1.32e-01, 10: 7.97e-09, 30: 8.24e-25
Xavier/Glorot sqrt(1/n)      activation std after layer 1: 5.84e-01, 10: 2.73e-02, 30: 3.01e-05
He sqrt(2/n)                 activation std after layer 1: 8.24e-01, 10: 8.61e-01, 30: 8.44e-01
```

With a fixed small standard deviation the signal is gone by layer 10. The Glorot scale (variance 1/n for square layers) was designed without ReLU in mind and still loses more than four orders of magnitude over 30 layers. The He scale keeps the activation scale at 0.84 at layer 30. Zero initialisation is the extreme case: all units in a layer receive identical gradients and stay identical, so the symmetry is never broken.

**Common weak answer.** "Random initialisation avoids local minima." The issue is signal scale and symmetry, not minima.

**Follow-up.** "What changes with normalisation layers?" They re-normalise activations and make training less sensitive to initial scale (question 6).

## 2. "What are vanishing and exploding gradients? How do you fix them?"

**What it tests.** Linking depth, activation functions and the chain rule.

**Reference answer.** Backpropagation multiplies one Jacobian per layer. If the typical factor is below 1, the gradient shrinks exponentially with depth; if above 1, it grows. Saturating activations are a common cause of shrinkage: the derivative of a sigmoid is at most 0.25. Glorot and Bengio report that the logistic sigmoid is unsuited for deep networks with random initialisation because of its mean value, which can drive the top hidden layer into saturation. Fixes: non-saturating activations such as ReLU, an initialisation matched to the activation (question 1), normalisation layers, residual connections, and gradient clipping for the exploding case. (Residual connections and clipping are named from general practice here; the sources read for this post do not cover them.)

An experiment with a 20-layer network of width 64, Xavier-initialised, and the gradient norm of the weight matrices at three depths:

```python
import torch, torch.nn as nn
torch.manual_seed(0)
def grads(act, depth=20, width=64):
    layers = []
    for _ in range(depth):
        l = nn.Linear(width, width); nn.init.xavier_normal_(l.weight); layers += [l, act()]
    m = nn.Sequential(*layers, nn.Linear(width, 1)); x = torch.randn(256, width)
    m(x).pow(2).mean().backward()
    ws = [p.grad.norm().item() for n, p in m.named_parameters() if n.endswith("weight")]
    return ws[0], ws[depth // 2], ws[-1]
for name, act in (("sigmoid", nn.Sigmoid), ("tanh", nn.Tanh), ("ReLU", nn.ReLU)):
    a, b, c = grads(act); print(f"{name:8} gradient norm, first layer {a:.2e}, middle layer {b:.2e}, last hidden layer {c:.2e}")
```

```
sigmoid  gradient norm, first layer 6.35e-14, middle layer 1.06e-06, last hidden layer 2.54e+00
tanh     gradient norm, first layer 1.14e-02, middle layer 1.03e-01, last hidden layer 3.62e-01
ReLU     gradient norm, first layer 5.88e-05, middle layer 2.71e-03, last hidden layer 5.45e-02
```

In the sigmoid network the gradient at the first layer is 13 orders of magnitude smaller than at the last: the early layers do not learn. The tanh and ReLU networks lose far less. (ReLU still loses about three orders of magnitude here because Xavier scaling is not matched to it; question 1.)

**Common weak answer.** "Use ReLU." One of several fixes, and the ReLU network above still shows decay.

**Follow-up.** "How do you detect exploding gradients in training?" Log the global gradient norm; spikes and NaN losses are the signals; clip the norm.

## 3. "SGD, momentum, Adam: what is the difference, and which would you choose?"

**What it tests.** Knowing what each optimiser does and the practical defaults.

**Reference answer.** SGD steps along the negative minibatch gradient. Momentum accumulates a running average of past gradients, which smooths the noise and speeds progress along consistent directions. Adam, from [Kingma and Ba (2015)](https://arxiv.org/abs/1412.6980), keeps adaptive estimates of the first and second moments of the gradient; the paper says it is invariant to diagonal rescaling of the gradients and that its hyper-parameters typically require little tuning. In practice, [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/) suggests Adam with a learning rate of 3e-4 for early baselines because it is much more forgiving to hyper-parameters, including a bad learning rate, and notes that for ConvNets a well-tuned SGD will almost always slightly outperform Adam, with a narrower and more problem-specific optimal learning-rate region. The author's reading: choose Adam or AdamW as the default to get a working baseline; consider SGD with momentum when there is budget to tune it and the architecture is known to prefer it.

An experiment: the same 3-layer MLP, 30 epochs, batch size 64, five learning rates:

```python
for name, mk in (("SGD", lambda p, lr: torch.optim.SGD(p, lr=lr)),
                 ("SGD+momentum 0.9", lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9)),
                 ("Adam", lambda p, lr: torch.optim.Adam(p, lr=lr))):
    row = []
    for lr in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        torch.manual_seed(0); m = mlp(); h = train(m, mk(m.parameters(), lr))
        row.append(f"lr={lr:g}: {acc(m, Xte, yte):.3f}")
    print(f"{name:18}", "  ".join(row))
```

(`mlp` builds a 3-hidden-layer, 128-unit ReLU network and `train` is a standard minibatch loop; their code is in the appendix at the end of the post, and the `digits.npz` file holds the scikit-learn digits data.)

```
SGD                lr=0.0001: 0.099  lr=0.001: 0.099  lr=0.01: 0.293  lr=0.1: 0.951  lr=1: 0.154
SGD+momentum 0.9   lr=0.0001: 0.099  lr=0.001: 0.271  lr=0.01: 0.943  lr=0.1: 0.978  lr=1: 0.104
Adam               lr=0.0001: 0.915  lr=0.001: 0.966  lr=0.01: 0.951  lr=0.1: 0.372  lr=1: 0.104
```

Test accuracy is above 0.9 for SGD at one of five learning rates, for momentum at two and for Adam at three. Each optimiser has a different working range: Adam's is lower than plain SGD's by one to three orders of magnitude, and the best number overall (0.978) came from SGD with momentum at 0.1, after a sweep. This is a small problem and a single seed, so the comparison is a demonstration of the sweep, not a ranking of optimisers.

**Common weak answer.** "Adam is always better." Not in this sweep's best case, and not in Karpathy's note about ConvNets.

**Follow-up.** "What does Adam's second-moment estimate do?" It scales each parameter's step by a running estimate of its gradient magnitude, so parameters with large gradients take smaller steps.

## 4. "How do you set the learning rate and its schedule?"

**What it tests.** Treating the learning rate as the main hyper-parameter, and knowing the standard devices.

**Reference answer.** The learning rate is the first thing to tune; [Smith (2017)](https://arxiv.org/abs/1506.01186) states that it is known to be the most important hyper-parameter, and proposes a range test: increase the rate linearly over a few epochs and read off reasonable bounds. Common schedules: a short warm-up, then a decay (step, cosine, linear), or cyclical rates; [Loshchilov and Hutter (2017)](https://arxiv.org/abs/1608.03983) propose warm restarts for SGD to improve its anytime performance. Two warnings. First, Karpathy's recipe says not to trust learning-rate-decay defaults: schedules based on the epoch number can silently drive the rate to zero too early when the data set is a different size from the one the default was designed for. Second, the learning rate depends on the batch size. [Goyal et al. (2017)](https://arxiv.org/abs/1706.02677) state the linear scaling rule: when the minibatch size is multiplied by k, multiply the learning rate by k, keeping other hyper-parameters unchanged, and add a warm-up with a low rate at the start for very large batches, where the rule breaks down early in training. Their results are for ImageNet with ResNet-50 up to a batch size of 8,192.

A check of the rule on the digits MLP, plain SGD, 10 epochs, base setting batch 16 with learning rate 0.01:

```python
for bs in (16, 64, 256):
    row = []
    for tag, lr in (("same lr", 0.01), ("linear scaling", 0.01 * bs / 16)):
        torch.manual_seed(0); m = mlp(); train(m, torch.optim.SGD(m.parameters(), lr=lr), epochs=10, bs=bs)
        row.append(f"{tag} (lr={lr:g}): {acc(m, Xte, yte):.3f}")
    print(f"batch {bs:>3}: ", "   ".join(row))
```

```
batch  16:  same lr (lr=0.01): 0.410   linear scaling (lr=0.01): 0.410
batch  64:  same lr (lr=0.01): 0.176   linear scaling (lr=0.04): 0.417
batch 256:  same lr (lr=0.01): 0.099   linear scaling (lr=0.16): 0.404
```

With the learning rate fixed, accuracy falls from 0.410 to 0.099 as the batch grows, because there are fewer updates. With the rate scaled linearly, the three batch sizes end within 0.013 of each other. (The accuracy values are low because training is short; the comparison across rows is the point.)

**Common weak answer.** "Try 1e-3, and lower it if the loss is bad." It skips the sweep, the schedule and the batch-size dependence.

**Follow-up.** "The loss spikes midway through training." A rate that is too high for the current phase, a bad batch, or a missing warm-up; lower the rate, clip gradients, inspect the batch.

## 5. "What is weight decay, and how does it differ from L2 regularisation?"

**What it tests.** Whether the candidate knows the two coincide for SGD and not for Adam.

**Reference answer.** L2 regularisation adds a penalty on the squared weights to the loss; weight decay shrinks the weights directly at each step. [Loshchilov and Hutter (2019)](https://arxiv.org/abs/1711.05101) state that the two are equivalent for standard SGD (when rescaled by the learning rate) but not for adaptive gradient algorithms such as Adam: with an L2 term in the loss, the penalty's gradient passes through Adam's per-parameter scaling. They propose decoupling the decay from the gradient step, which is AdamW, and report that it decouples the best weight decay from the learning rate and substantially improves Adam's generalisation. In PyTorch, `torch.optim.Adam(weight_decay=...)` is the L2 form and `torch.optim.AdamW` is the decoupled form.

An experiment with 300 training rows, 200 epochs, learning rate 1e-3 and the same weight-decay value in both forms:

```python
Xs, ys = Xtr[:300], ytr[:300]
for name, mk in (("Adam, no decay", lambda p: torch.optim.Adam(p, lr=1e-3)),
                 ("Adam + L2 (weight_decay=0.1)", lambda p: torch.optim.Adam(p, lr=1e-3, weight_decay=0.1)),
                 ("AdamW (weight_decay=0.1)", lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.1)),
                 ("AdamW (weight_decay=1.0)", lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1.0))):
    torch.manual_seed(0); m = mlp(); h = train(m, mk(m.parameters()), epochs=200, X=Xs, y=ys)
    wn = sum(p.pow(2).sum() for n, p in m.named_parameters() if n.endswith("weight")).sqrt().item()
    print(f"{name:30} test acc {acc(m, Xte, yte):.3f}  weight norm {wn:.1f}  final train loss {h[-1][0]:.4f}  test loss {h[-1][1]:.3f}")
```

```
Adam, no decay                 test acc 0.928  weight norm 17.7  final train loss 0.0007  test loss 0.335
Adam + L2 (weight_decay=0.1)   test acc 0.087  weight norm 0.0  final train loss 2.2871  test loss 2.312
AdamW (weight_decay=0.1)       test acc 0.928  weight norm 16.8  final train loss 0.0013  test loss 0.296
AdamW (weight_decay=1.0)       test acc 0.926  weight norm 13.0  final train loss 0.0105  test loss 0.235
```

The same nominal value, 0.1, collapses the L2 form (the weights shrink to zero and accuracy is chance) and leaves the decoupled form training normally, with a lower test loss than no decay (0.296 against 0.335). Even a decay of 1.0 in AdamW only reduces the weight norm to 13.0 and lowers the test loss further to 0.235. The values are therefore not interchangeable between the two forms, which is the practical point. This is one small data set and seed; it shows the non-equivalence, not an optimal setting.

**Common weak answer.** "They're the same thing." True for SGD, which is what makes the answer tempting.

**Follow-up.** "Do you apply decay to biases and normalisation parameters?" A common choice is not to; the sources read for this post do not settle it, so the candidate should say it is a design choice to test.

## 6. "Batch normalisation: what does it do, and what are the alternatives?"

**What it tests.** Mechanics, the train/test difference, and awareness that the standard explanation is disputed.

**Reference answer.** Batch normalisation normalises each unit's input using the mean and variance of the current minibatch, then applies a learned scale and shift. [Ioffe and Szegedy (2015)](https://arxiv.org/abs/1502.03167) report that it allows much higher learning rates, makes initialisation less critical, acts as a regulariser (in some cases eliminating the need for dropout), and reached the same accuracy as a baseline image model with 14 times fewer training steps. They attribute the effect to reducing "internal covariate shift". [Santurkar et al. (2018)](https://arxiv.org/abs/1805.11604) argue that this distributional stability has little to do with the success of BatchNorm, and that its fundamental impact is making the optimisation landscape significantly smoother, which gives more predictive and stable gradients. A good answer states the benefit and flags that the explanation is contested.

Because the statistics come from the batch, behaviour depends on the batch size, and at inference a running average from training is used instead. [Ba et al. (2016)](https://arxiv.org/abs/1607.06450) propose layer normalisation, which computes the statistics from all the summed inputs to the neurons in a layer on a single training case, so it does not depend on the minibatch size, performs exactly the same computation at training and test time, and is straightforward to apply to recurrent networks. The choice in practice: batch normalisation for convolutional networks with reasonably large batches; layer normalisation for sequence models and small batches. (That last sentence is the author's summary of the two papers' stated trade-offs, not a quotation.)

**Common weak answer.** "It fixes internal covariate shift." That is the original claim and the disputed one.

**Follow-up.** "What happens with batch size 1?" See the next question: the training-mode computation fails or is meaningless.

## 7. "A model works in training and fails at inference. What is the first thing you check?"

**What it tests.** Train versus eval mode, and other differences in how the same network runs at the two times.

**Reference answer.** Check `model.eval()`. Batch normalisation and dropout behave differently in the two modes: in training, batch normalisation uses batch statistics and dropout drops units; in evaluation, batch normalisation uses running averages and dropout is the identity. Forgetting `eval()` is a classic, quiet bug. The dropout paper describes the intent: at test time, a single unthinned network with smaller weights approximates averaging all the thinned networks. PyTorch implements this by scaling the kept units by 1/(1-p) in training, so no scaling is needed at test time.

An experiment with a batch-normalised MLP trained 20 epochs on the digits:

```python
torch.manual_seed(0); m = mlp(norm=nn.BatchNorm1d)
train(m, torch.optim.Adam(m.parameters(), lr=1e-3), epochs=20)
print(f"eval mode, test accuracy {acc(m, Xte, yte):.3f}")
m.train()
try:
    with torch.no_grad(): m(Xte[:1])
except Exception as e: print("train mode, batch of 1:", type(e).__name__, str(e)[:70])
with torch.no_grad(): a1 = (m(Xte).argmax(1) == yte).float().mean().item()
print(f"train mode, whole test set as one batch: accuracy {a1:.3f}")
with torch.no_grad():
    a2 = torch.cat([m(Xte[i:i+2]).argmax(1) for i in range(0, len(Xte) - 1, 2)]).eq(yte[:len(Xte)//2*2]).float().mean().item()
print(f"train mode, batches of 2: accuracy {a2:.3f}")
m.eval()
with torch.no_grad(): o1 = m(Xte[:1]); o2 = m(Xte[:8])[:1]
print(f"eval mode: same row alone vs in a batch, max output difference {(o1-o2).abs().max().item():.2e}")
m.train()
with torch.no_grad(): o1 = m(Xte[:2])[:1]; o2 = m(Xte[:8])[:1]
print(f"train mode: same row in different batches, max output difference {(o1-o2).abs().max().item():.2e}")
d = nn.Dropout(0.5); x = torch.ones(100000); d.train()
print(f"dropout p=0.5, train mode: mean output {d(x).mean().item():.3f}, share of zeros {(d(x)==0).float().mean().item():.3f}, kept value {d(x).max().item():.1f}")
d.eval(); print(f"dropout, eval mode: mean output {d(x).mean().item():.3f}")
```

```
eval mode, test accuracy 0.985
train mode, batch of 1: ValueError Expected more than 1 value per channel when training, got input size t
train mode, whole test set as one batch: accuracy 0.983
train mode, batches of 2: accuracy 0.503
eval mode: same row alone vs in a batch, max output difference 1.43e-06
train mode: same row in different batches, max output difference 4.55e+00
dropout p=0.5, train mode: mean output 1.001, share of zeros 0.500, kept value 2.0
dropout, eval mode: mean output 1.000
```

In eval mode the output for a row does not depend on its batch (difference 1.4e-6, float rounding). In train mode, a single row cannot be processed at all, batches of 2 give 0.503 accuracy against 0.985, and the same row gives outputs that differ by 4.55 depending on its batch companions. Dropout keeps the expected output unchanged by doubling the kept units at p = 0.5.

Other places where training and inference differ: preprocessing and tokenisation code, data augmentation applied at test time, and numerical precision; Part 5, question 9 covers the data side.

**Common weak answer.** "Retrain with more data." It skips the cheap check.

**Follow-up.** "You serve one request at a time on a BatchNorm model. What do you do?" Run in eval mode with running statistics; the training-mode result depends on the other rows in the batch.

## 8. "The model overfits. What do you do?"

**What it tests.** A ranked list of remedies, starting from the cheapest and the most likely.

**Reference answer.** Confirm it is overfitting: training loss keeps falling while validation loss rises. Then, in order of cost: get more data or augment it; stop early; add weight decay; add dropout; reduce the model's capacity; and check for leakage that makes the validation set look better than it is (Part 4). Karpathy's recipe lists the same ladder (more real data, data augmentation, a smaller batch size with batch normalisation, dropout, weight decay, early stopping, and a larger model last) and adds two cautions: use dropout sparingly because it does not seem to play nice with batch normalisation, and a larger model combined with early stopping can sometimes beat a smaller one. [Srivastava et al. (2014)](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) present dropout as randomly dropping units during training, which prevents units from co-adapting too much and samples from an exponential number of "thinned" networks.

Early stopping on a deliberately small data set, 150 training rows and a 256-unit MLP:

```python
Xs, ys = Xtr[:150], ytr[:150]
torch.manual_seed(0); m = mlp(width=256); h = train(m, torch.optim.Adam(m.parameters(), lr=1e-3), epochs=150, X=Xs, y=ys)
best = min(range(len(h)), key=lambda i: h[i][1])
print(f"best test loss {h[best][1]:.3f} at epoch {best+1}; at epoch 150 test loss {h[-1][1]:.3f}, train loss {h[-1][0]:.5f}")
for e in (5, 10, 20, 50): print(f"  epoch {e}: train {h[e-1][0]:.4f} test {h[e-1][1]:.3f}")
```

```
best test loss 0.335 at epoch 32; at epoch 150 test loss 0.373, train loss 0.00047
  epoch 5: train 1.7813 test 1.978
  epoch 10: train 0.7276 test 0.973
  epoch 20: train 0.1064 test 0.379
  epoch 50: train 0.0060 test 0.344
```

The training loss goes to 0.0005 while the test loss is lowest at epoch 32 (0.335) and slightly higher at epoch 150 (0.373). The effect is mild here: the 11% rise in test loss is the cost of training far past the best epoch. (The test set stands in for validation data in this demonstration; in a real project, early stopping needs a validation set separate from the final test set.)

**Common weak answer.** "Add dropout." It is one remedy of several, and is applied before checking for the more likely causes.

**Follow-up.** "Validation loss rises but accuracy still improves. Is that overfitting?" It can be overconfidence: loss punishes confident wrong predictions; look at calibration (Part 4, question 5).

## 9. "Your network does not learn. How do you debug it?"

**What it tests.** A systematic procedure instead of random changes.

**Reference answer.** Karpathy's recipe, which the author uses as the checklist, starts with the data and then builds up from simple checks. Four of them are cheap to run:

1. **Loss at initialisation.** For a softmax classifier with n classes it should be about -log(1/n). A different value indicates a bug in the last layer, the loss or the labels.
2. **Input-independent baseline.** Train with the inputs zeroed; it should do worse than the real inputs. If not, the model is not extracting information from the input.
3. **Overfit one small batch.** A few examples should reach near-zero loss. If they do not, there is a bug and the next stage cannot proceed.
4. **Verify decreasing training loss** on the toy model, then increase capacity a little.

The recipe also advises complexifying one thing at a time, and reminds that the failure is often a logical bug (labels not flipped with an augmentation, an off-by-one in an autoregressive model, clipping the loss instead of the gradient) which a unit test will not catch.

The four checks on the digits MLP:

```python
import math
torch.manual_seed(0); m = mlp(); lf = nn.CrossEntropyLoss()
with torch.no_grad(): print(f"loss at init {lf(m(Xtr), ytr).item():.3f}  (ln 10 = {math.log(10):.3f})")
bad = mlp()
with torch.no_grad():
    for p in bad[-1].parameters(): p.mul_(30)
    print(f"final layer scaled x30: loss at init {lf(bad(Xtr), ytr).item():.2f}")
torch.manual_seed(0); m = mlp(width=256); h = train(m, torch.optim.Adam(m.parameters(), lr=1e-3), epochs=100, X=Xtr[:16], y=ytr[:16])
print(f"overfit 16 rows: train loss after 100 epochs {h[-1][0]:.5f}, accuracy on those rows {acc(m, Xtr[:16], ytr[:16]):.2f}")
torch.manual_seed(0); m = mlp(); train(m, torch.optim.Adam(m.parameters(), lr=1e-3), epochs=30, X=torch.zeros_like(Xtr), y=ytr)
print(f"zeroed inputs: test accuracy {acc(m, torch.zeros_like(Xte), yte):.3f} (majority class share {torch.bincount(ytr).max().item()/len(ytr):.3f})")
torch.manual_seed(0); m = mlp(); train(m, torch.optim.Adam(m.parameters(), lr=1e-3), epochs=30)
print(f"real inputs: test accuracy {acc(m, Xte, yte):.3f}")
```

```
loss at init 2.303  (ln 10 = 2.303)
final layer scaled x30: loss at init 3.13
overfit 16 rows: train loss after 100 epochs 0.00012, accuracy on those rows 1.00
zeroed inputs: test accuracy 0.089 (majority class share 0.104)
real inputs: test accuracy 0.966
```

The healthy network starts at 2.303, which is ln 10 to three decimals; an over-scaled last layer starts at 3.13 and flags itself before any training. The 16-row overfit reaches a loss of 0.00012. With zeroed inputs the model is at chance (0.089, against a majority share of 0.104), and with real inputs it reaches 0.966.

**Common weak answer.** "Lower the learning rate." Sometimes right, and not a method.

**Follow-up.** "The loss is NaN after 200 steps." Inspect the first batch where it appears, the gradient norm before it, the learning rate and the data for inf or NaN values; clip gradients as a stopgap. (The author's list; no source read for this post gives a NaN procedure.)

## 10. "Derive the gradients for a two-layer network and check them."

**What it tests.** Backpropagation as the chain rule applied layer by layer, and the habit of verifying a derivation numerically.

**Reference answer.** For a network `z1 = XW1 + b1`, `h = relu(z1)`, `z2 = hW2 + b2` with softmax cross-entropy loss averaged over n rows, the gradient with respect to the logits is `(softmax(z2) - onehot(y)) / n`. Then `dW2 = h^T dz2`, `db2` is the column sum of `dz2`, `dh = dz2 W2^T`, `dz1 = dh * (z1 > 0)`, `dW1 = X^T dz1`, and `db1` is the column sum of `dz1`. Each step multiplies the upstream gradient by the local Jacobian; the activations from the forward pass are the reason training needs memory proportional to depth. The check is to compare against automatic differentiation and against finite differences.

```python
import numpy as np, torch
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3)); y = rng.integers(0, 4, 5)
W1 = rng.normal(size=(3, 6)) * 0.5; b1 = np.zeros(6); W2 = rng.normal(size=(6, 4)) * 0.5; b2 = np.zeros(4)
def forward(W1, b1, W2, b2):
    z1 = X @ W1 + b1; h = np.maximum(z1, 0); z2 = h @ W2 + b2
    p = np.exp(z2 - z2.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return z1, h, p, -np.log(p[np.arange(5), y]).mean()
z1, h, p, loss = forward(W1, b1, W2, b2)
dz2 = p.copy(); dz2[np.arange(5), y] -= 1; dz2 /= 5
dW2 = h.T @ dz2; db2 = dz2.sum(0); dh = dz2 @ W2.T; dz1 = dh * (z1 > 0); dW1 = X.T @ dz1; db1 = dz1.sum(0)
t = [torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, b2)]
zt = torch.relu(torch.tensor(X) @ t[0] + t[1]) @ t[2] + t[3]
torch.nn.functional.cross_entropy(zt, torch.tensor(y)).backward()
for name, mine, th in zip(("W1", "b1", "W2", "b2"), (dW1, db1, dW2, db2), t):
    print(f"{name}: max |manual - autograd| = {np.abs(mine - th.grad.numpy()).max():.2e}")
eps = 1e-6; num = np.zeros_like(W1)
for i in range(3):
    for j in range(6):
        a = W1.copy(); a[i, j] += eps; c = W1.copy(); c[i, j] -= eps
        num[i, j] = (forward(a, b1, W2, b2)[3] - forward(c, b1, W2, b2)[3]) / (2 * eps)
print(f"W1: max |manual - finite difference| = {np.abs(dW1 - num).max():.2e}")
```

```
W1: max |manual - autograd| = 4.16e-17
b1: max |manual - autograd| = 2.78e-17
W2: max |manual - autograd| = 1.39e-17
b2: max |manual - autograd| = 5.55e-17
W1: max |manual - finite difference| = 1.40e-10
```

The hand-derived gradients agree with PyTorch's autograd to about 1e-17 and with central finite differences to 1e-10, which is the expected truncation error for a step of 1e-6.

**Common weak answer.** "PyTorch does it for me." True in practice, and it fails the question: the interviewer wants to see the chain rule and the shapes.

**Follow-up.** "Why does the backward pass cost about as much as the forward pass, and what is the memory cost?" Each layer needs one extra matrix product per input and weight gradient, and the forward activations must be stored. (The author's reasoning; not taken from a source read for this post.)

## Takeaways

- Initialise to keep the activation scale constant through depth; match the scale to the nonlinearity.
- Every optimiser has a working learning-rate range; sweep it, and scale it with the batch size.
- L2 and weight decay are the same for SGD and different for Adam.
- Always check train versus eval mode; BatchNorm in training mode makes outputs depend on the batch.
- Debug in a fixed order: loss at initialisation, an input-independent baseline, overfitting a tiny batch.

## Limits

- The experiments use a small MLP on a small data set, one seed each; they show mechanisms, not the behaviour of large models.
- Transformers, convolutional architectures, mixed precision, distributed training and large-scale learning-rate recipes are not covered here; the LLM-specific questions are the subject of later parts.
- Items marked as the author's reading or reasoning (the optimiser choice in question 3, the normalisation summary in question 6, the NaN checklist and the cost of backpropagation) have no single source behind them.
- The reference answers describe one reasonable answer. Interviewers differ, and the candidates in the examples are hypothetical.

## Appendix: helper code used above

```python
import numpy as np, torch, torch.nn as nn
d = np.load("digits.npz"); X = torch.tensor(d["X"], dtype=torch.float32); y = torch.tensor(d["y"])   # scikit-learn digits, pixels / 16
g = torch.Generator().manual_seed(0); perm = torch.randperm(len(X), generator=g)
tr, te = perm[:1200], perm[1200:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
def mlp(depth=3, width=128, act=nn.ReLU, norm=None, p=0.0):
    layers, i = [], 64
    for _ in range(depth):
        layers.append(nn.Linear(i, width))
        if norm: layers.append(norm(width))
        layers.append(act())
        if p: layers.append(nn.Dropout(p))
        i = width
    layers.append(nn.Linear(i, 10)); return nn.Sequential(*layers)
def acc(m, X, y):
    m.eval()
    with torch.no_grad(): return (m(X).argmax(1) == y).float().mean().item()
def train(m, opt, epochs=30, bs=64, seed=0, X=Xtr, y=ytr):
    torch.manual_seed(seed); lf = nn.CrossEntropyLoss(); hist = []
    for ep in range(epochs):
        m.train(); idx = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = idx[i:i+bs]; opt.zero_grad(); loss = lf(m(X[b]), y[b]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): hist.append((lf(m(X), y).item(), lf(m(Xte), yte).item()))
    return hist
```

Next: Part 7, on LLM engineering.
