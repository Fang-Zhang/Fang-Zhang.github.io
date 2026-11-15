---
title: "Maths for LLMs, Tested in Code (7/10): Optimisation, Adam, AdamW and Warmup"
date: 2027-01-20 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, mathematics, optimisation, adam, adamw, learning-rate, series]
math: true
description: "Part 7 of 10. Training an LLM is minimising a loss with Adam or AdamW and a learning-rate schedule. Adam is written from its formulas and matched to torch.optim.Adam to 1e-16; the effects of bias correction, curvature, decoupled weight decay and warmup are each measured."
---

Chapter 7 of [The Little Book of Maths for LLMs](https://github.com/little-book-of/maths-for-llms/blob/main/books/en-US/book.md) is optimisation: gradient descent, momentum, adaptive methods, schedules and regularisation. In an LLM, all of the maths in [Part 3](/posts/maths-for-llms-3-calculus-backprop-residual/) ends here: the gradient is handed to an optimiser, almost always Adam or AdamW with warmup and decay. The script below (PyTorch for the cross-check, otherwise NumPy, CPU) implements the update from the formulas and tests five claims, one block each.

![Three panels. Left: loss against step on a quadratic with curvatures 100 and 1 for SGD, momentum and Adam at their best learning rates; SGD declines slowly while momentum and Adam oscillate and fall further. Middle: a learning-rate schedule with linear warmup over 100 steps and cosine decay to 10 percent of the peak. Right: the schedule of Vaswani et al., rising linearly to a peak of 7e-4 at step 4000 and decaying as the inverse square root.](/assets/pic/maths-llms-7-optimisation.png)
*Left: curvature and the three optimisers. Middle and right: two warmup-and-decay schedules.*

## The update rules

Gradient descent: $\theta\leftarrow\theta-\eta g$. Adam ([Kingma and Ba](https://arxiv.org/abs/1412.6980)) keeps exponential moving averages of the gradient and its square, and corrects their initialisation bias:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad \hat m_t=\frac{m_t}{1-\beta_1^t},\ \hat v_t=\frac{v_t}{1-\beta_2^t},\quad \theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

The paper's suggested defaults are $\beta_1=0.9$ and $\beta_2=0.999$. AdamW ([Loshchilov and Hutter](https://arxiv.org/abs/1711.05101)) decouples weight decay: instead of adding $\lambda\theta$ to the gradient, where Adam's normalisation then rescales it, it subtracts $\eta\lambda\theta$ from the weights directly. The paper's abstract says the two are equivalent for SGD but not for adaptive methods such as Adam.

```python
import numpy as np, torch
# 1. Adam from the formulas, checked against torch.optim.Adam
def adam(grad_fn, th, steps, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8, correct=True):
    m = np.zeros_like(th); v = np.zeros_like(th); out = []
    for t in range(1, steps + 1):
        g = grad_fn(th); m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh, vh = (m / (1 - b1 ** t), v / (1 - b2 ** t)) if correct else (m, v)
        th = th - lr * mh / (np.sqrt(vh) + eps); out.append(th.copy())
    return th, out
A = np.diag([100.0, 1.0]); grad = lambda x: A @ x; f = lambda x: 0.5 * x @ A @ x
x0 = np.array([1.0, 1.0]); mine, _ = adam(grad, x0.copy(), 50)
p = torch.tensor(x0, requires_grad=True); opt = torch.optim.Adam([p], lr=1e-2)
for _ in range(50):
    opt.zero_grad(); (0.5 * p @ torch.tensor(A) @ p).backward(); opt.step()
print("1. hand-written Adam vs torch.optim.Adam after 50 steps, max |diff| =", float(np.abs(mine - p.detach().numpy()).max()))
# 2. Bias correction: the first update has size lr whatever the gradient scale
for gs in (1e-3, 1.0, 1e3):
    g = lambda x, gs=gs: gs * np.ones_like(x)
    _, o1 = adam(g, np.zeros(1), 1); _, o2 = adam(g, np.zeros(1), 1, correct=False)
    print(f"2. gradient scale {gs:g}: first step with correction {abs(o1[0][0]):.2e} (lr 1e-2); without {abs(o2[0][0]):.2e}")
# 3. Ill-conditioned quadratic, H = diag(100, 1): loss after 100 steps
def run(kind, lr, steps=100):
    x = x0.copy(); v = np.zeros(2)
    if kind == 'adam': return f(adam(grad, x, steps, lr=lr)[0])
    for _ in range(steps):
        g = grad(x)
        if kind == 'sgd': x = x - lr * g
        else: v = 0.9 * v + g; x = x - lr * v
    return f(x)
print("\n3. loss after 100 steps on 0.5 x^T diag(100,1) x from (1,1); best learning rate on a grid:")
grid = [10 ** e for e in np.linspace(-4, -0.5, 36)]
for kind in ('sgd', 'momentum', 'adam'):
    r = [(run(kind, lr), lr) for lr in grid]; r = [t for t in r if np.isfinite(t[0])]; b = min(r)
    print(f"   {kind:<9} best lr {b[1]:.4f} -> loss {b[0]:.3e}")
print("   SGD stable only for lr < 2/100 = 0.02; its rate is set by the small eigenvalue: factor (1 - lr*1) per step on that axis.")
# 4. AdamW vs Adam+L2: parameter with pure-noise gradients of scale s (no signal), 2000 steps, wd = 0.1, lr = 0.01
def decay_run(kind, s, steps=2000, lr=1e-2, wd=0.1, b1=0.9, b2=0.999, eps=1e-8):
    rng = np.random.default_rng(0); w = 1.0; m = v = 0.0
    for t in range(1, steps + 1):
        g = s * rng.standard_normal()
        if kind == 'l2': g = g + wd * w
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if kind == 'adamw': w = w - lr * wd * w
    return w
print("\n4. weight left after 2000 steps (start 1.0), noise-only gradients of scale s, wd = 0.1, lr = 0.01:")
for s_ in (1e-3, 1e-1, 1.0, 10.0):
    print(f"   s = {s_:<6g} Adam+L2: {decay_run('l2', s_):+.4f}   AdamW: {decay_run('adamw', s_):+.4f}   (AdamW pure decay: exp(-lr*wd*steps) = {np.exp(-1e-2*0.1*2000):.4f})")
# 5. Schedules: linear warmup + cosine decay, and the Vaswani et al. schedule
def cosine(t, peak=1.0, warm=100, total=1000, floor=0.1):
    return peak * t / warm if t < warm else floor * peak + 0.5 * (1 - floor) * peak * (1 + np.cos(np.pi * (t - warm) / (total - warm)))
vas = lambda t, d=512, w=4000: d ** -0.5 * min(t ** -0.5, t * w ** -1.5)
print("\n5. schedules: cosine at t=0,50,100,550,1000 ->", [round(cosine(t), 3) for t in (0, 50, 100, 550, 1000)])
print(f"   Vaswani lrate (d_model=512, warmup 4000): peak at t=4000 = {vas(4000):.2e}; t=1000: {vas(1000):.2e}; t=16000: {vas(16000):.2e}")
```

```
1. hand-written Adam vs torch.optim.Adam after 50 steps, max |diff| = 1.1102230246251565e-16
2. gradient scale 0.001: first step with correction 1.00e-02 (lr 1e-2); without 3.16e-02
2. gradient scale 1: first step with correction 1.00e-02 (lr 1e-2); without 3.16e-02
2. gradient scale 1000: first step with correction 1.00e-02 (lr 1e-2); without 3.16e-02

3. loss after 100 steps on 0.5 x^T diag(100,1) x from (1,1); best learning rate on a grid:
   sgd       best lr 0.0158 -> loss 2.048e-02
   momentum  best lr 0.0316 -> loss 7.040e-05
   adam      best lr 0.0398 -> loss 2.707e-05
   SGD stable only for lr < 2/100 = 0.02; its rate is set by the small eigenvalue: factor (1 - lr*1) per step on that axis.

4. weight left after 2000 steps (start 1.0), noise-only gradients of scale s, wd = 0.1, lr = 0.01:
   s = 0.001  Adam+L2: -0.0021   AdamW: +0.4418   (AdamW pure decay: exp(-lr*wd*steps) = 0.1353)
   s = 0.1    Adam+L2: +0.1207   AdamW: +0.4418   (AdamW pure decay: exp(-lr*wd*steps) = 0.1353)
   s = 1      Adam+L2: +0.4369   AdamW: +0.4418   (AdamW pure decay: exp(-lr*wd*steps) = 0.1353)
   s = 10     Adam+L2: +1.3134   AdamW: +0.4418   (AdamW pure decay: exp(-lr*wd*steps) = 0.1353)

5. schedules: cosine at t=0,50,100,550,1000 -> [0.0, 0.5, 1.0, 0.55, 0.1]
   Vaswani lrate (d_model=512, warmup 4000): peak at t=4000 = 6.99e-04; t=1000: 1.75e-04; t=16000: 3.49e-04
```

## Reading the output

1. **The implementation is right.** The hand-written Adam matches `torch.optim.Adam` after 50 steps to $1.1\times10^{-16}$.
2. **Bias correction fixes the first step.** With $m_0=v_0=0$ the first uncorrected step is $\eta\,(1-\beta_1)/\sqrt{1-\beta_2}=0.0316$ for $\eta=0.01$, whatever the gradient scale; corrected, it is exactly $\eta=0.01$. In both cases the step size does not depend on the gradient scale (1e-3 to 1e3), which is the invariance to gradient scale the Adam paper claims: the update is a ratio of two quantities in the same units.
3. **Curvature sets the learning rate.** On $f=\tfrac12x^{\top}\mathrm{diag}(100,1)x$, SGD diverges for $\eta>2/100$, so its step is limited by the stiffest direction while progress along the flat one goes as $(1-\eta)^t$. Under the best-lr grid search, SGD reached a loss of $2.0\times10^{-2}$ after 100 steps, momentum $7.0\times10^{-5}$ and Adam $2.7\times10^{-5}$. The left panel shows why this is not a clean ranking: momentum and Adam oscillate, and the numbers are the loss at step 100 of an oscillating curve. The robust reading is the one from the plot: SGD is monotone but slow, and the other two trade smoothness for speed on this toy.
4. **Decoupled decay behaves the same at every gradient scale.** With pure-noise gradients of scale $s$ and $\lambda=0.1$, AdamW ends at 0.4418 for every $s$. Adam with $L^2$ ends between $-0.002$ and $+1.31$: when the gradient is tiny the penalty term dominates and shrinks the weight to nearly 0; when the gradient is large the same penalty is a small part of the normalised update and barely acts. The 0.4418 is above the pure-decay value $e^{-\eta\lambda t}=0.135$ because the noise steps (each about $\eta$ in size after normalisation) random-walk the weight.
5. **Schedules.** Warmup raises the rate from 0 to the peak over the first steps, and a decay follows. The Vaswani et al. formula, $\text{lrate}=d_{\text{model}}^{-0.5}\min(t^{-0.5},\,t\cdot\text{warmup}^{-1.5})$ with warmup 4000, is a linear increase for the first 4,000 steps and then an inverse-square-root decay; for $d_{\text{model}}=512$ it peaks at $7.0\times10^{-4}$ (right panel). The paper's own description is that it increases the rate linearly for the first warmup steps and decreases it proportionally to the inverse square root of the step number.

Why warmup helps is not tested here. One common explanation is that $\hat v_t$ is a poor estimate early in training, which is plausible from item 2 but is the author's reading, not something this script measures.

## Which maths, which component

- Exponential moving averages and their bias correction: Adam's $m_t$, $v_t$.
- Eigenvalues of the Hessian: the stable learning-rate bound $2/\lambda_{\max}$ and the slow direction $\lambda_{\min}$.
- Ratio of expectations with the square root: Adam's invariance to gradient scale.
- Regularisation as a penalty versus as a decay on the weights: $L^2$ versus AdamW.
- A piecewise function of the step number: warmup and decay schedules.

Next: Part 8, discrete maths and graphs: the causal mask as a directed acyclic graph, and counting in the KV cache.

## Limits

- All experiments are two-parameter or one-parameter toys; they show mechanisms, not behaviour on a Transformer with billions of weights.
- The best-learning-rate search compares losses at one step of oscillating curves, and a different step budget would change the ordering of momentum and Adam.
- Gradient clipping, batch size and second-order methods (the rest of the book's chapter) are not exercised.
- The decay experiment uses noise-only gradients, a deliberate simplification: it isolates the interaction of decay and normalisation, and does not say which variant generalises better on real data.
