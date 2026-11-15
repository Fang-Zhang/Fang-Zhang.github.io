---
title: "Maths for LLMs, Tested in Code (3/10): Calculus, Backpropagation and the Residual Path"
date: 2027-01-16 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, calculus, backpropagation, residual-connections, series]
math: true
description: "Part 3 of 10. Backpropagation is the chain rule applied to a transformer block in reverse order. The post writes it by hand for a residual feed-forward block with a cross-entropy head, checks it against autograd, shows what the skip connection does to gradients across 50 layers, and what curvature does to the learning rate."
---

Chapter 3 of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) covers derivatives, partial derivatives, the gradient, the chain rule, and second derivatives. Training an LLM uses these through one algorithm, backpropagation, and one structural choice, the residual connection. This post writes the backward pass of one transformer-style block by hand, compares it with PyTorch autograd, measures the gradient across 50 layers with and without the skip path, and shows the role of curvature in the step size. All numbers are from scripts run on this machine (CPU, PyTorch 2.15). [Part 2](/posts/maths-for-llms-2-linear-algebra-attention-ffn-lora/) defined the matrices; here they are differentiated.

![Three panels. Left: the ratio of the gradient norm at layer 1 to that at layer 50 against weight gain, for plain and residual stacks of 50 tanh layers; the plain stack falls to 4e-15 at gain 0.5 and the residual stack stays above 8. Middle: the norm of the weights under gradient descent with learning rates 0.01, 0.019 and 0.021 on a quadratic with eigenvalues 100 and 1; only 0.021 diverges. Right: maximum absolute difference between hand-written and autograd gradients, about 5e-17 for all four tensors.](/assets/pic/maths-llms-3-calculus.png)
*Left: gradient across depth. Middle: the step-size limit set by curvature. Right: hand-written backward pass against autograd.*

## 1. The chain rule through a block

For a function built by composition, $L = \ell(g_n(\dots g_1(x)))$, the chain rule multiplies Jacobians from the output backwards. For the block used here (a residual feed-forward layer and a softmax cross-entropy head, the structure of [one transformer sub-layer](https://arxiv.org/abs/1706.03762) with the attention part removed), the forward pass is

$$a = xW_1,\quad r=\max(0,a),\quad f = rW_2,\quad h = x + f,\quad z = hW_o,\quad L = -\tfrac1T\sum_t \log \operatorname{softmax}(z_t)_{y_t}.$$

Each line has a local derivative, and the backward pass applies them in reverse order:

$$\frac{\partial L}{\partial z}=\frac{\operatorname{softmax}(z)-\text{onehot}(y)}{T},\qquad \frac{\partial L}{\partial W_o}=h^{\top}\frac{\partial L}{\partial z},\qquad \frac{\partial L}{\partial h}=\frac{\partial L}{\partial z}W_o^{\top},$$

$$\frac{\partial L}{\partial x}=\underbrace{\frac{\partial L}{\partial h}}_{\text{skip}}+\underbrace{\Big(\frac{\partial L}{\partial h}W_2^{\top}\odot \mathbb{1}[a>0]\Big)W_1^{\top}}_{\text{through the block}} .$$

The first equation is the compact form of softmax followed by cross-entropy, one of the places where two derivatives simplify into subtraction. The last equation has two terms because $h = x + f$ has two paths from $x$. The code writes each line, then compares with autograd in float64, and with a central finite difference for one entry.

```python
import torch
torch.manual_seed(0)
# One pre-norm residual MLP block followed by a cross-entropy head, back-propagated by hand.
d, dff, V, T = 6, 12, 10, 4
x = torch.randn(T, d, dtype=torch.float64)
y = torch.randint(0, V, (T,))
W1 = (torch.randn(d, dff, dtype=torch.float64) / d**.5).requires_grad_()
W2 = (torch.randn(dff, d, dtype=torch.float64) / dff**.5).requires_grad_()
Wo = (torch.randn(d, V, dtype=torch.float64) / d**.5).requires_grad_()

# forward, keeping intermediates:  h = x + relu(x W1) W2 ;  logits = h Wo ;  L = mean CE
a = x @ W1; r = a.clamp(min=0); f = r @ W2; h = x + f; z = h @ Wo
p = torch.softmax(z, -1); L = -torch.log(p[torch.arange(T), y]).mean()

# backward by the chain rule, one line per forward line, in reverse order
dz = p.clone(); dz[torch.arange(T), y] -= 1; dz /= T           # dL/dz = (softmax - onehot)/T
dWo = h.T @ dz;      dh = dz @ Wo.T                              # z = h Wo
dx_skip = dh                                                     # h = x + f : identity branch
df = dh;             dW2 = r.T @ df;  dr = df @ W2.T             # f = r W2
da = dr * (a > 0);   dW1 = x.T @ da;  dx_mlp = da @ W1.T         # r = relu(a), a = x W1
dx = dx_skip + dx_mlp

# the same quantities by autograd
x2 = x.clone().requires_grad_()
h2 = x2 + torch.relu(x2 @ W1) @ W2
L2 = torch.nn.functional.cross_entropy(h2 @ Wo, y); L2.backward()
print("loss  hand vs autograd:", float(L), float(L2))
for n, mine, ref in (("dW1", dW1, W1.grad), ("dW2", dW2, W2.grad), ("dWo", dWo, Wo.grad), ("dx", dx, x2.grad)):
    print(f"{n}: max abs diff {float((mine - ref).abs().max()):.2e}")
# finite-difference check of one entry
eps = 1e-6; i, j = 2, 5
def loss_at(W1_):
    hh = x + torch.relu(x @ W1_) @ W2
    return torch.nn.functional.cross_entropy(hh @ Wo, y)
W1p = W1.detach().clone(); W1m = W1.detach().clone(); W1p[i, j] += eps; W1m[i, j] -= eps
fd = float((loss_at(W1p) - loss_at(W1m)) / (2 * eps))
print(f"dW1[{i},{j}]: backprop {float(dW1[i, j]):.8f}  finite difference {fd:.8f}")
print("share of dx that came through the skip path:", f"{float(dx_skip.norm() / dx.norm()):.3f}", "(skip norm / total norm)")
```

```
loss  hand vs autograd: 4.623347765084449 4.62334776508445
dW1: max abs diff 5.55e-17
dW2: max abs diff 5.55e-17
dWo: max abs diff 5.55e-17
dx: max abs diff 4.16e-17
dW1[2,5]: backprop -0.22163061  finite difference -0.22163061
share of dx that came through the skip path: 0.617 (skip norm / total norm)
```

All four gradients agree with autograd to about $10^{-17}$, i.e. float64 rounding (right panel), and the single-entry check agrees to eight decimals with a finite difference, which is the definition of the derivative applied numerically. The last line shows the two paths in numbers: for this random block, the skip path carries a gradient of norm 0.617 times the total. Autograd does this bookkeeping for a graph of any size; the hand version shows that nothing beyond matrix products, an indicator function and a subtraction is involved.

## 2. What the skip connection does to gradients

Through $L$ layers of plain composition $h_{l+1}=\phi(W_lh_l)$ the gradient at layer 1 is a product of $L$ Jacobians; if their typical size is below 1 the product vanishes, if above 1 it explodes. With a residual layer $h_{l+1}=h_l+\phi(W_lh_l)$ each Jacobian is $I + J_l$, and the identity term gives the gradient a path that does not shrink. [He et al.](https://arxiv.org/abs/1603.05027) analyse this: with identity skip connections, the signal "could be directly propagated from one unit to any other units, in both forward and backward passes". The Transformer applies the same device around every sub-layer. The experiment measures $\lVert\partial L/\partial h_1\rVert / \lVert\partial L/\partial h_{50}\rVert$ in a 50-layer tanh stack, for several weight scales (gain 1 is the usual $1/\sqrt{d}$ initialisation).

```python
import torch, torch.nn as nn
torch.manual_seed(0)
d, depth = 64, 50
def ratio(gain, residual):
    layers = [nn.Linear(d, d).double() for _ in range(depth)]
    for l in layers: nn.init.normal_(l.weight, std=gain / d**.5); nn.init.zeros_(l.bias)
    h = torch.randn(32, d, dtype=torch.float64); hs = []
    for l in layers:
        h = (h + torch.tanh(l(h))) if residual else torch.tanh(l(h))
        h.retain_grad(); hs.append(h)
    h.pow(2).mean().backward()
    g = [float(t.grad.norm()) for t in hs]
    return g[0] / g[-1]          # gradient norm at layer 1 divided by gradient norm at layer 50
print("depth 50, tanh layers; ratio = |dL/dh_1| / |dL/dh_50|  (1 = gradient preserved)")
print(f"{'weight gain':>12}{'plain':>14}{'residual':>14}")
for gain in (0.5, 1.0, 1.5, 2.5):
    print(f"{gain:>12}{ratio(gain, False):>14.3e}{ratio(gain, True):>14.3e}")
```

```
depth 50, tanh layers; ratio = |dL/dh_1| / |dL/dh_50|  (1 = gradient preserved)
 weight gain         plain      residual
         0.5     3.993e-15     8.539e+00
         1.0     6.425e-01     3.330e+01
         1.5     1.794e+01     1.288e+02
         2.5     5.282e+04     1.254e+03
```

Read the table by columns. In the plain stack the ratio is $4\times10^{-15}$ at gain 0.5 (the gradient has vanished), 0.64 at gain 1, and $5\times10^{4}$ at gain 2.5 (it explodes): the plain stack works only in a narrow band of weight scale. In the residual stack the ratio is at least 8.5 in every row, so the gradient never vanishes, and it grows with the gain at a much slower rate. The growth is the other side of the same identity term: each layer adds to the signal, so without a scaling factor or a normalisation layer the activations and gradients grow with depth (the author's reading of this output; Part 1's LayerNorm and RMSNorm are the usual remedies). The experiment is a toy, but the contrast is the one the identity-mapping paper describes.

## 3. Second derivatives: curvature sets the step size

Gradient descent updates $w \leftarrow w - \eta\nabla L(w)$. Near a minimum the loss is approximately quadratic, $L \approx \tfrac12 w^{\top}Hw$, with $H$ the matrix of second derivatives (the Hessian). Then each step multiplies the component along eigenvector $i$ by $1-\eta\lambda_i$, so the iteration converges for all components exactly when

$$\eta < \frac{2}{\lambda_{\max}(H)} .$$

The smallest eigenvalue sets the rate along the flattest direction, so the ratio $\lambda_{\max}/\lambda_{\min}$ (the condition number) decides how slow the slowest direction is when $\eta$ is held under the limit.

```python
import torch
# Curvature: loss L(w) = 0.5 w^T H w, gradient descent with step lr converges iff lr < 2/lambda_max
H = torch.diag(torch.tensor([100.0, 1.0]))
lam = torch.linalg.eigvalsh(H)
print("eigenvalues of the Hessian:", lam.tolist(), "-> condition number", float(lam.max() / lam.min()))
for lr in (0.001, 0.01, 0.019, 0.021):
    w = torch.tensor([1.0, 1.0])
    for _ in range(200): w = w - lr * (H @ w)
    print(f"lr {lr:<6} (2/lambda_max = {2/float(lam.max()):.3f}): |w| after 200 steps = {float(w.norm()):.3e}")
```

```
eigenvalues of the Hessian: [1.0, 100.0] -> condition number 100.0
lr 0.001  (2/lambda_max = 0.020): |w| after 200 steps = 8.186e-01
lr 0.01   (2/lambda_max = 0.020): |w| after 200 steps = 1.340e-01
lr 0.019  (2/lambda_max = 0.020): |w| after 200 steps = 2.157e-02
lr 0.021  (2/lambda_max = 0.020): |w| after 200 steps = 1.899e+08
```

With eigenvalues 100 and 1 the limit is $\eta<0.02$. Rates of 0.001, 0.01 and 0.019 converge, more slowly the further below the limit; 0.021, only 5% above, diverges to $10^{8}$ in 200 steps (middle panel). Part 7 uses this to explain warmup, momentum and Adam; the point here is that the learning rate is a statement about a second derivative.

## Which maths, which component

- Chain rule: backpropagation through attention, FFN, norm and the output head; one local derivative per forward line.
- Softmax plus cross-entropy: the gradient with respect to the logits is $\operatorname{softmax}(z)-\text{onehot}(y)$.
- The identity term in $h = x + f$: a second path for gradients, which the residual connections around every sub-layer provide.
- Hessian eigenvalues: the largest bounds the stable learning rate, the ratio to the smallest bounds the speed.

Next: [Part 4, probability, softmax, temperature and top-p](/posts/maths-for-llms-4-probability-softmax-sampling/).

## Limits

- The backward pass covers a feed-forward block and a head, not attention or LayerNorm; their derivatives follow the same rule but need the softmax and normalisation Jacobians.
- The depth experiment uses tanh layers, random weights and no normalisation, with one seed and one input batch; it illustrates the Jacobian-product argument and does not measure a trained transformer.
- The quadratic model is exact only for a quadratic loss. Real loss surfaces have changing curvature, and the largest Hessian eigenvalue moves during training.
- Not covered: second-order optimisers, saddle points (Part 7), and the memory cost of storing activations for the backward pass.
