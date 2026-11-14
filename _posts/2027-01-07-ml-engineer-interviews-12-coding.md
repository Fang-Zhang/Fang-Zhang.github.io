---
title: "ML Engineer Interviews, With Reference Answers (12/18): Coding"
date: 2027-01-07 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, coding, python, numpy, pytorch, series]
description: "Part 12 of 18. Ten coding questions for ML interviews: data structures, API design, k-means, AUC, attention, numerical stability, vectorisation and debugging a broken training loop, each solved, tested against a library and timed."
---

This post covers the coding round, after [Part 11](/posts/ml-engineer-interviews-11-deployment-and-mlops/): classic data-structure questions, class and API design, and the ML routines that interviewers ask for from scratch (k-means, AUC, attention, a stable softmax), ending with a broken training loop. Each question has what it tests, a reference answer with code, a common weak answer and a follow-up. Every solution was run here, and the ML ones were checked against scikit-learn, SciPy or PyTorch; the output is shown. Claims about documentation are limited to the passages that were read. Timings depend on the machine and vary between runs.

The code ran on a CPU-only machine (32 cores) with Python 3.14.7, NumPy, SciPy, scikit-learn and PyTorch 2.15.0, with fixed seeds.

![Left: time to compute 300 by 1200 Euclidean distances: Python loops 0.529 s, NumPy broadcasting 0.135 s, matrix-product form 0.005 s. Right: k-means inertia over 20 seeds on 8 blobs: random initialisation min 3568.9, median 4326.7, max 8000.0; k-means++ min 3568.2, median 3721.6, max 4970.4; scikit-learn k-means++ min 3568.4, median 3695.0, max 5112.1.](/assets/pic/ml-interview-12-vectorise-kmeans.png)
*Left: vectorisation (question 10). Right: initialisation of k-means (question 5).*

## 1. "Find the k most frequent items in a large stream."

**What it tests.** Choosing the right data structure and stating the complexity, not typing speed.

**Reference answer.** Count with a hash map, then select the top k. Counting is O(n) time and O(d) memory for d distinct values (hash-table get and set are O(1) on average, per the [Python time-complexity table](https://wiki.python.org/moin/TimeComplexity)). For the selection, sorting all d counts costs O(d log d); a heap of size k costs O(d log k), which wins when k is much smaller than d. The standard library provides it: [`heapq.nlargest(n, iterable, key=...)`](https://docs.python.org/3/library/heapq.html), which the documentation says is equivalent to `sorted(iterable, key=key, reverse=True)[:n]` (the equivalence is in results; the heap version is the one that avoids the full sort). An interviewer may then ask what to do when the stream does not fit in memory or the distinct count is huge: use a sketch (count-min) or the space-saving algorithm for approximate counts, or partition by hash and merge; that follow-up is the author's addition and was not coded here.

```python
import heapq, time, numpy as np
from collections import Counter
rng = np.random.default_rng(0); stream = rng.zipf(1.3, 2_000_000).tolist(); k = 10
t0 = time.perf_counter(); c = Counter(stream); t_count = time.perf_counter() - t0
t0 = time.perf_counter(); a = heapq.nlargest(k, c.items(), key=lambda kv: kv[1]); t_heap = time.perf_counter() - t0
t0 = time.perf_counter(); b = sorted(c.items(), key=lambda kv: kv[1], reverse=True)[:k]; t_sort = time.perf_counter() - t0
print(f"{len(stream)} items, {len(c)} distinct values")
print(f"count {t_count:.3f} s, nlargest {t_heap:.4f} s, full sort {t_sort:.4f} s")
print("same top-10 counts:", [v for _, v in a] == [v for _, v in b], [v for _, v in a][:5])
```

```
2000000 items, 96684 distinct values
count 0.058 s, nlargest 0.0038 s, full sort 0.0159 s
same top-10 counts: True [508639, 206279, 121914, 83730, 62490]
```

The heap selection was about four times faster than the full sort on 96,684 distinct values, and the two give the same counts (ties may order differently, which is why the check compares counts, not items). The counting step dominates, as the complexity predicts.

**Common weak answer.** "Sort the whole stream and count runs." That is O(n log n) time and needs the whole stream in memory.

**Follow-up.** "What if two items tie at the k-th place?" Decide the rule first (return all, or break ties by a stable key), state it, and test it.

## 2. "Implement an LRU cache with O(1) get and put."

**What it tests.** Combining a hash map with an ordering structure, and writing a small class with tests.

**Reference answer.** The two operations need constant-time lookup and constant-time reordering. A hash map plus a doubly linked list does it, and Python's `collections.OrderedDict` already is that: its documentation says it has a `move_to_end()` method to efficiently reposition an element to an endpoint, and `popitem(last=False)` removes the oldest entry. A plain list for the order would make each access O(n), since the Python time-complexity table gives O(n) for removing from the middle of a list.

```python
import time
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        assert capacity > 0; self.cap, self.d = capacity, OrderedDict()
    def get(self, key, default=None):
        if key not in self.d: return default
        self.d.move_to_end(key); return self.d[key]                  # most recently used goes to the end
    def put(self, key, value):
        self.d[key] = value; self.d.move_to_end(key)
        if len(self.d) > self.cap: self.d.popitem(last=False)         # evict the least recently used
class ListLRU:                                                         # O(n) per access: a list keeps the order
    def __init__(self, capacity): self.cap, self.keys, self.vals = capacity, [], {}
    def get(self, key, default=None):
        if key not in self.vals: return default
        self.keys.remove(key); self.keys.append(key); return self.vals[key]
    def put(self, key, value):
        if key in self.vals: self.keys.remove(key)
        self.keys.append(key); self.vals[key] = value
        if len(self.keys) > self.cap: del self.vals[self.keys.pop(0)]
c = LRUCache(2); c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)          # "b" is the least recently used
assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3; c.put("a", 10); assert c.get("a") == 10 and len(c.d) == 2
print("unit tests passed")
import random; random.seed(0); ops = [(random.random() < 0.7, random.randrange(20000)) for _ in range(100000)]
for cls in (LRUCache, ListLRU):
    cache = cls(10000); t0 = time.perf_counter()
    for is_get, k in ops: cache.get(k) if is_get else cache.put(k, k)
    print(f"{cls.__name__:9s} capacity 10000, 100000 operations: {time.perf_counter() - t0:.3f} s")
```

```
unit tests passed
LRUCache  capacity 10000, 100000 operations: 0.016 s
ListLRU   capacity 10000, 100000 operations: 1.115 s
```

The assertions cover a hit that refreshes recency, an eviction of the least recently used key, and an update of an existing key. The timing shows why the structure matters: with 10,000 slots and 100,000 operations the O(1) version took 0.016 s and the list-based version 1.1 s, about 70 times slower, and the gap grows with capacity. In an interview, also say what is not handled: thread safety (a lock or sharding), a time-to-live, and a `capacity` of zero (here rejected by an assertion).

**Common weak answer.** "Use a dictionary and a list of keys." It is correct and O(n) per operation, which the interviewer will ask about.

**Follow-up.** "Write it without OrderedDict." A dictionary from key to node, a doubly linked list with sentinel head and tail nodes, and the same two operations as pointer updates; the common bugs are forgetting to update both neighbours on removal and forgetting to remove the evicted key from the dictionary.

## 3. "Sample from a categorical distribution given unnormalised weights, many times."

**What it tests.** Cumulative sums and binary search, which appear in sampling, in negative sampling (Part 9) and in beam-search variants.

**Reference answer.** Build the cumulative sum of the weights once (O(n)), then for each draw pick a uniform number in `[0, total)` and binary-search the first cumulative value that reaches it (O(log n) per draw), with `bisect`. A linear scan costs O(n) per draw. Check the sampler empirically: the frequencies of many draws should match the probabilities.

```python
import bisect, itertools, time, numpy as np
rng = np.random.default_rng(0); w = rng.random(1000) ** 3; cum = list(itertools.accumulate(w)); total = cum[-1]
def draw(r): return bisect.bisect_left(cum, r * total)               # first index whose cumulative weight reaches the target
r = np.random.default_rng(1).random(200000); t0 = time.perf_counter(); idx = [draw(x) for x in r]; t_bis = time.perf_counter() - t0
freq = np.bincount(idx, minlength=1000) / len(idx); p = w / w.sum()
print(f"max |empirical - true probability| over 1000 items: {np.abs(freq - p).max():.4f} (largest probability {p.max():.4f})")
cum_np = np.array(cum); t0 = time.perf_counter()
for x in r[:2000]: int(np.argmax(cum_np >= x * total))              # linear scan per draw
t_lin = (time.perf_counter() - t0) * 100
print(f"200000 draws: bisect {t_bis:.2f} s; linear scan (extrapolated from 2000 draws) {t_lin:.2f} s")
```

```
max |empirical - true probability| over 1000 items: 0.0004 (largest probability 0.0038)
200000 draws: bisect 0.05 s; linear scan (extrapolated from 2000 draws) 0.32 s
```

The largest deviation between empirical and true probability over 1,000 items is 0.0004 against a largest probability of 0.0038 (roughly the sampling error expected with 200,000 draws). The bisect sampler is about 6 times faster than a vectorised linear scan at 1,000 items, and the gap widens with more items. The alias method gives O(1) per draw after O(n) set-up; it is the standard answer to "can you do better", and it was not implemented here.

**Common weak answer.** "Repeat random draws until one lands on an item with probability proportional to its weight" (rejection sampling without bounds), or a loop that subtracts weights from the draw; both can be made to work and neither has the O(log n) per draw.

**Follow-up.** "The weights change after every draw." Then a cumulative array is stale after each update; a Fenwick tree gives O(log n) updates and draws (the author's suggestion).

## 4. "Design the class for a preprocessing step. What is the contract?"

**What it tests.** API design: separating `fit` from `transform`, and knowing why the separation exists.

**Reference answer.** A preprocessing object has two phases: `fit` learns statistics from the training data only and stores them as attributes; `transform` applies them to any data, without learning. The separation prevents leakage: statistics computed on the test data make test examples depend on each other and on the future (Part 5 covers this in the data section). A good class also handles degenerate input (a constant column has a standard deviation of zero) and returns `self` from `fit` so calls can be chained. Matching the scikit-learn convention (`fit`, `transform`, `fit_transform`, trailing underscore for learned attributes) lets the object work in a pipeline.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
class Standardizer:
    def fit(self, X): self.mean_, self.std_ = X.mean(0), X.std(0); self.std_[self.std_ == 0] = 1.0; return self
    def transform(self, X): return (X - self.mean_) / self.std_
    def fit_transform(self, X): return self.fit(X).transform(X)
rng = np.random.default_rng(0); X = rng.normal(5, 3, (1000, 4)); X[:, 3] = 7.0            # the last column is constant
mine, ref = Standardizer().fit_transform(X), StandardScaler().fit_transform(X)
print("matches sklearn StandardScaler (max abs difference):", np.abs(mine - ref).max())
train, test = rng.normal(0, 1, (900, 1)), rng.normal(0, 1, (100, 1)) + 2.0                # the test period has shifted
leak = Standardizer().fit(np.vstack([train, test])).transform(test).mean(); ok = Standardizer().fit(train).transform(test).mean()
print(f"test mean after scaling: fit on train only {ok:.2f}, fit on train+test {leak:.2f} (the shift is partly hidden by the leak)")
```

```
matches sklearn StandardScaler (max abs difference): 0.0
test mean after scaling: fit on train only 2.03, fit on train+test 1.56 (the shift is partly hidden by the leak)
```

The result is identical to `StandardScaler` including the constant column. The second experiment shows the leak: the test data are shifted by 2, and scaling with statistics fitted on train and test together moves the test mean to 1.56, hiding nearly a quarter of the shift, whereas fitting on train alone preserves it (2.03). In production the shift is exactly what has to remain visible.

**Common weak answer.** A function `standardize(X)` that computes the mean and standard deviation of whatever it receives. It works in a notebook and fails at serving time, where a single row has no meaningful mean.

**Follow-up.** "What do you persist for serving?" The learned attributes and a version of the code, saved with the model so the training-time and serving-time transformations are the same (Part 10, question 5).

## 5. "Implement k-means. What are its weaknesses?"

**What it tests.** An iterative algorithm written with vectorised NumPy, and awareness of local minima.

**Reference answer.** The algorithm alternates two steps: assign each point to its nearest centroid, then move each centroid to the mean of its points. The [scikit-learn documentation](https://scikit-learn.org/stable/modules/clustering.html) describes the objective as the inertia, the within-cluster sum of squares, states that the algorithm always converges, but possibly to a local minimum that depends strongly on the initialisation, so the computation is often repeated with different initialisations, and recommends the k-means++ scheme, which initialises the centroids to be generally distant from each other. In k-means++ the first centre is chosen at random and each next centre is drawn with probability proportional to the squared distance to the nearest centre already chosen. Other weaknesses to name: the number of clusters must be given, clusters are assumed to be roughly spherical and of similar variance, and the mean is sensitive to outliers.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
def kmeans(X, k, seed, plus=True, iters=100):
    rng = np.random.default_rng(seed)
    if plus:                                                       # k-means++: next centre drawn with probability proportional to squared distance
        C = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d2 = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), axis=1); C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(C)
    else: C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(new, C): break
        C = new
    return ((X - C[lab]) ** 2).sum(), lab
X, _ = make_blobs(2000, centers=8, cluster_std=1.0, random_state=3)
for name, plus in (("random init", False), ("k-means++ init", True)):
    inertias = [kmeans(X, 8, s, plus)[0] for s in range(20)]
    print(f"mine, {name:14s}: inertia over 20 seeds min {min(inertias):7.1f}  median {np.median(inertias):7.1f}  max {max(inertias):7.1f}")
sk = [KMeans(8, init="k-means++", n_init=1, random_state=s).fit(X).inertia_ for s in range(20)]
print(f"sklearn k-means++ , n_init=1: min {min(sk):7.1f}  median {np.median(sk):7.1f}  max {max(sk):7.1f}")
```

```
mine, random init   : inertia over 20 seeds min  3568.9  median  4326.7  max  8000.0
mine, k-means++ init: inertia over 20 seeds min  3568.2  median  3721.6  max  4970.4
sklearn k-means++ , n_init=1: min  3568.4  median  3695.0  max  5112.1
```

All three versions reach the same best inertia (about 3,568) over 20 seeds, so the good solution exists for every method; the difference is in how often a run lands there. Random initialisation has a median of 4,327 and a worst case of 8,000, more than twice the best, while k-means++ has a median of 3,722 and a worst case of 4,970. The scikit-learn version with a single initialisation behaves like mine (median 3,695, worst 5,112), which is the check that the implementation is right; this test does not show that mine matches it in every case (for example empty clusters, which are handled here by leaving the centroid in place).

**Common weak answer.** "Run it once and report the clusters." The run-to-run spread above is the reason to restart or to use k-means++ and several initialisations.

**Follow-up.** "How do you choose k?" There is no single answer: inspect the inertia curve, use a silhouette or stability measure, and above all ask what the clusters are for, since the decision downstream fixes the useful number.

## 6. "Compute AUC from scratch. Why is it a ranking metric?"

**What it tests.** Understanding what a metric means, and handling ties and scale.

**Reference answer.** ROC AUC is the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative one, counting ties as one half. That definition gives an O(P x N) algorithm over all pairs, which is fine for 3,000 rows and impossible for 400,000. The equivalent rank form needs one sort: rank all scores (ties get the average rank), then `AUC = (sum of positive ranks - P(P+1)/2) / (P x N)`, where P and N are the numbers of positives and negatives. This is the Mann-Whitney U statistic divided by P x N; the code checks the equivalence numerically against the pair definition and against [scikit-learn's `roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), which is the evidence, not a source claim.

```python
import time, numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
def auc_rank(y, s):                                                # AUC = (sum of positive ranks - n_pos (n_pos + 1) / 2) / (n_pos n_neg); ties get the average rank
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def auc_pairs(y, s):                                               # definition: P(score of a positive > score of a negative), ties count 1/2
    p, n = s[y == 1], s[y == 0]; return ((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n))
rng = np.random.default_rng(0); y = (rng.random(3000) < 0.1).astype(int); s = rng.normal(size=3000) + 1.0 * y
print(f"continuous scores: rank {auc_rank(y, s):.6f}  pairs {auc_pairs(y, s):.6f}  sklearn {roc_auc_score(y, s):.6f}")
s2 = np.round(s * 2) / 2                                           # many tied scores
print(f"tied scores:       rank {auc_rank(y, s2):.6f}  pairs {auc_pairs(y, s2):.6f}  sklearn {roc_auc_score(y, s2):.6f}")
print(f"constant scores:   rank {auc_rank(y, np.zeros(3000)):.6f}  (a model with no information)")
yb, sb = (rng.random(400000) < 0.1).astype(int), rng.normal(size=400000)
t0 = time.perf_counter(); auc_rank(yb, sb); print(f"400000 rows, rank method {time.perf_counter() - t0:.3f} s (the pair method would need {int(yb.sum()) * int((1 - yb).sum()):,} comparisons)")
```

```
continuous scores: rank 0.791631  pairs 0.791631  sklearn 0.791631
tied scores:       rank 0.788057  pairs 0.788057  sklearn 0.788057
constant scores:   rank 0.500000  (a model with no information)
400000 rows, rank method 0.051 s (the pair method would need 14,444,460,679 comparisons)
```

The three agree to six digits on continuous scores and on heavily tied scores (where average ranks are what makes the formula right), a constant score gives exactly 0.5, and the rank method handles 400,000 rows in 0.05 s where the pair method would need 14.4 billion comparisons. Because AUC depends only on the order of the scores, it is unchanged by any monotone transformation of them and says nothing about calibration or about performance at a particular threshold or alert budget (Part 10, question 3).

**Common weak answer.** "Draw the ROC curve at many thresholds and add up trapezoids." It is correct and slower; the interviewer is usually looking for the ranking interpretation.

**Follow-up.** "AUC is 0.99 but the product still fails." On very imbalanced data a tiny false-positive rate can still be too many alerts; look at precision at the operating point and at the precision-recall curve (Part 4).

## 7. "Implement scaled dot-product attention, with a causal mask."

**What it tests.** Shapes, the scaling factor, masking, and a correctness test against a reference.

**Reference answer.** For queries Q, keys K and values V of dimension d, attention is `softmax(Q K^T / sqrt(d)) V`, with the softmax over the key dimension. The reference implementation in the [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) uses the same scale factor, `1 / sqrt(d)`, and implements the causal option by adding minus infinity to the entries above the diagonal of the score matrix, so that position i attends only to positions up to i. Write the shapes in the comment (batch, heads, sequence, dimension), mask before the softmax (never after), and test against the library function; also test the property that matters, that changing future values cannot change earlier outputs.

```python
import math, torch, torch.nn.functional as F
torch.manual_seed(0)
def attention(q, k, v, causal=False):                              # q, k, v: (batch, heads, seq, dim)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))       # scale by sqrt(d) so the softmax inputs keep unit-order variance
    if causal: scores = scores.masked_fill(torch.ones(scores.shape[-2:], dtype=torch.bool).triu(1), float("-inf"))   # position i may not see j > i
    return torch.softmax(scores, dim=-1) @ v
q, k, v = (torch.randn(2, 4, 16, 32) for _ in range(3))
for causal in (False, True):
    d = (attention(q, k, v, causal) - F.scaled_dot_product_attention(q, k, v, is_causal=causal)).abs().max().item(); print(f"causal={causal!s:5s}: max abs difference from torch's scaled_dot_product_attention {d:.2e}")
v2 = v.clone(); v2[:, :, 10:] += 100.0                              # change the 'future' (positions 10 and later)
print("causal output at positions 0-9 unchanged when later values change:", torch.allclose(attention(q, k, v, True)[:, :, :10], attention(q, k, v2, True)[:, :, :10]))
big = torch.randn(1, 1, 1000, 512); s = (big @ big.transpose(-2, -1)); print(f"score std without scaling {s.std():.1f}, with 1/sqrt(d) {(s / math.sqrt(512)).std():.2f}")
```

```
causal=False: max abs difference from torch's scaled_dot_product_attention 2.98e-07
causal=True : max abs difference from torch's scaled_dot_product_attention 2.38e-07
causal output at positions 0-9 unchanged when later values change: True
score std without scaling 27.9, with 1/sqrt(d) 1.23
```

The implementation matches PyTorch's function to about 3e-7, which is float32 rounding, with and without the mask. The causal test changes the values at positions 10 and later by 100 and the outputs at positions 0 to 9 do not change. The last line shows why the scale exists: with 512-dimensional random vectors the score standard deviation is 27.9 without scaling and 1.23 with it; large inputs push the softmax toward a one-hot output with tiny gradients (the explanation is the author's; the numbers are the measurement).

**Common weak answer.** Masking after the softmax, or forgetting to renormalise. Masking has to happen before the softmax so that the probabilities over the allowed positions sum to one.

**Follow-up.** "What is the memory cost, and how would you reduce it?" The score matrix is sequence-length squared per head; fused kernels that avoid materialising it (Part 8, question on FlashAttention) and key-value caching at inference (Part 7) address the memory and the repeated work respectively.

## 8. "Implement softmax and cross-entropy so they do not overflow."

**What it tests.** Numerical stability, and the interface of library losses.

**Reference answer.** The softmax is unchanged by subtracting a constant from all logits, so subtract the maximum before exponentiating; the largest exponent is then zero and nothing overflows. For cross-entropy, never form the probabilities: compute `logsumexp(logits) - logit_of_target` with the same shift, so that very unlikely classes give a finite loss instead of `log(0)`. The [PyTorch documentation for `CrossEntropyLoss`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) states that the input is expected to contain unnormalised logits and that the case with class indices is equivalent to applying `LogSoftmax` followed by `NLLLoss`; therefore applying a softmax yourself before this loss is a bug (it is one of the broken loops in question 9).

```python
import numpy as np, torch
x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
with np.errstate(all="ignore"): naive = np.exp(x) / np.exp(x).sum()
stable = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
print("naive softmax :", naive); print("stable softmax:", stable)
def logsumexp(x): m = x.max(); return m + np.log(np.exp(x - m).sum())
def cross_entropy(logits, y): return np.mean([logsumexp(l) - l[t] for l, t in zip(logits, y)])    # -log softmax(l)[t] without forming probabilities
rng = np.random.default_rng(0); L = rng.normal(0, 30, (5, 4)); y = rng.integers(0, 4, 5)
print(f"cross entropy from logits {cross_entropy(L, y):.6f}  torch {torch.nn.functional.cross_entropy(torch.tensor(L), torch.tensor(y)).item():.6f}")
z = np.array([0.0, -200.0], dtype=np.float32); p = np.exp(z) / np.exp(z).sum()
with np.errstate(divide="ignore"): print("float32 probabilities", p, "log of the unlikely one:", np.log(p[1]), "| logit form:", z[1] - logsumexp(z))
```

```
naive softmax : [nan nan nan]
stable softmax: [0.09003057 0.24472846 0.66524094]
cross entropy from logits 42.301622  torch 42.301622
float32 probabilities [1. 0.] log of the unlikely one: -inf | logit form: -200.0
```

The naive softmax on logits of 1000 returns `nan` for every entry, the shifted version returns the right probabilities, and the logit-form cross-entropy matches PyTorch to six digits (42.301622, on logits with a standard deviation of 30). The last line shows the other failure: in float32 the probability of a class 200 logits below the top underflows to 0, so its log is minus infinity, while the logit form gives the exact value, minus 200.

**Common weak answer.** `np.exp(x) / np.exp(x).sum()`, which works on the textbook example and fails on real logits; the next weak step is clipping probabilities with an epsilon, which hides the problem and changes the loss.

**Follow-up.** "What about the sigmoid and binary cross-entropy?" The same idea: use the logit form (`BCEWithLogitsLoss` in PyTorch) rather than a sigmoid followed by a log.

## 9. "A training loop does not learn. Find the bug."

**What it tests.** Debugging method, and knowing the usual suspects.

**Reference answer.** Start with a method, not a guess. Karpathy's [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) gives one: begin by inspecting the data (he reports finding duplicates, corrupted labels and imbalances this way); check that the loss at initialisation is the value expected for the task, `-log(1/n_classes)` for a softmax classifier; and overfit a single batch of a few examples, which should reach the lowest achievable loss, otherwise something is wrong and there is no point in training further (the author's reading of his advice). He also lists subtler failures, such as an autoregressive model that takes the quantity it predicts as an input through an off-by-one error, and gradient clipping applied to the loss instead of the gradients.

The experiment below trains a small classifier on a synthetic three-class problem with a correct loop and four common bugs (one change each), reporting the first-batch loss, the final train loss and the test accuracy after 8 epochs.

```python
import torch, torch.nn as nn
torch.manual_seed(0); torch.set_num_threads(4)
X = torch.randn(4000, 20); W = torch.randn(20, 3); y = (X @ W + 0.5 * torch.randn(4000, 3)).argmax(1)
Xtr, ytr, Xte, yte = X[:3000], y[:3000], X[3000:], y[3000:]
print(f"initial loss expected -log(1/3) = {-torch.log(torch.tensor(1 / 3)).item():.3f}")
def run(bug=None, epochs=8, lr=0.1):
    torch.manual_seed(1); net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3)); opt = torch.optim.SGD(net.parameters(), lr=lr)
    first = None
    for ep in range(epochs):
        net.train(); perm = torch.randperm(3000)
        for i in range(0, 3000, 100):
            idx = perm[i:i + 100]; xb, yb = Xtr[idx], ytr[idx]
            if bug == "shuffled labels": yb = ytr[torch.randperm(3000)[:100]]       # features and labels no longer aligned
            if bug != "no zero_grad": opt.zero_grad()
            out = net(xb)
            if bug == "softmax before loss": out = torch.softmax(out, 1)            # CrossEntropyLoss expects raw logits
            loss = nn.functional.cross_entropy(out, yb); first = first or loss.item(); loss.backward(); opt.step()
    if bug != "dropout left on": net.eval()
    with torch.no_grad(): acc = (net(Xte).argmax(1) == yte).float().mean().item(); trl = nn.functional.cross_entropy(net(Xtr), ytr).item()
    return first, trl, acc
for bug in (None, "no zero_grad", "softmax before loss", "shuffled labels", "dropout left on"):
    f, trl, acc = run(bug); print(f"{str(bug or 'correct loop'):20s} first-batch loss {f:5.3f}  final train loss {trl:5.3f}  test accuracy {acc:.3f}")
net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3)); opt = torch.optim.Adam(net.parameters(), 1e-2); xb, yb = Xtr[:8], ytr[:8]
for _ in range(200): opt.zero_grad(); l = nn.functional.cross_entropy(net(xb), yb); l.backward(); opt.step()
print(f"overfit one batch of 8: loss after 200 steps {l.item():.4f}")
```

```
initial loss expected -log(1/3) = 1.099
correct loop         first-batch loss 1.097  final train loss 0.205  test accuracy 0.923
no zero_grad         first-batch loss 1.097  final train loss 7.326  test accuracy 0.820
softmax before loss  first-batch loss 1.095  final train loss 0.324  test accuracy 0.899
shuffled labels      first-batch loss 1.140  final train loss 1.089  test accuracy 0.370
dropout left on      first-batch loss 1.097  final train loss 0.266  test accuracy 0.880
overfit one batch of 8: loss after 200 steps 0.0000
```

Reading the table like a debugger would: all loops start near 1.099 (= -log(1/3)), so the initial-loss check does not flag any of them; the correct loop reaches 92.3% test accuracy. Forgetting `zero_grad` accumulates gradients across steps, and the train loss is 7.3 against 0.205. Labels shuffled relative to the features show a first-batch loss of 1.140, slightly above 1.099 and a final loss that has not moved from the starting value (1.089), with a test accuracy of 37.0%, near the 33% of chance. Applying a softmax before the loss still learns, but worse (89.9% against 92.3% and a train loss of 0.324), because the squashed outputs shrink the gradients; this bug is easy to miss for that reason. Leaving dropout active at evaluation costs accuracy (88.0%) and makes the evaluation noisy. The single-batch test passes, as it should for the correct model (loss 0.0000 after 200 steps); a model with the label-shuffling bug would fail it, which is the point of running it first.

**Common weak answer.** "Lower the learning rate, add layers, train longer." Each of those changes several things at once; the recipe's argument is to add one verified piece of complexity at a time.

**Follow-up.** "The loss decreases but validation accuracy is at chance." Suspect the evaluation path: a different preprocessing, `eval()` not called, the labels misaligned in the validation loader, or leakage in the training split.

## 10. "This function is too slow. How would you speed it up?"

**What it tests.** Vectorisation, algebra, and memory awareness.

**Reference answer.** Profile first, then remove Python-level loops with array operations, then look for an algebraic reformulation. The example is the matrix of Euclidean distances between two sets of points, a step in k-nearest neighbours, k-means (question 5) and retrieval (Part 9). Three versions: nested loops (O(nmd) operations, each in interpreted Python), NumPy broadcasting (same operations in compiled code, but it creates an intermediate array of shape n x m x d), and the expansion `||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b`, which turns the distance computation into one matrix product that uses optimised linear-algebra routines, with a clamp at zero against rounding.

```python
import time, numpy as np
rng = np.random.default_rng(0); A, B = rng.normal(size=(1500, 64)), rng.normal(size=(1200, 64))
def loops(A, B):
    D = np.empty((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)): D[i, j] = np.sqrt(((A[i] - B[j]) ** 2).sum())
    return D
def broadcast(A, B): return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))                 # builds an (n, m, d) array
def gram(A, B): return np.sqrt(np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0))   # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
for name, f in (("python loops", loops), ("broadcast", broadcast), ("matrix product", gram)):
    t0 = time.perf_counter(); f(A[:300], B); print(f"{name:15s}, 300 x 1200 distances: {time.perf_counter() - t0:7.3f} s")
t0 = time.perf_counter(); D = gram(A, B); print(f"matrix product, 1500 x 1200 distances: {time.perf_counter() - t0:7.3f} s")
print("max abs difference, broadcast vs matrix product:", np.abs(broadcast(A[:300], B) - gram(A[:300], B)).max())
print(f"extra memory of the broadcast intermediate at 1500 x 1200 x 64: {A.shape[0] * B.shape[0] * A.shape[1] * 8 / 1e9:.2f} GB")
```

```
python loops   , 300 x 1200 distances:   0.529 s
broadcast      , 300 x 1200 distances:   0.135 s
matrix product , 300 x 1200 distances:   0.005 s
matrix product, 1500 x 1200 distances:   0.026 s
max abs difference, broadcast vs matrix product: 3.552713678800501e-15
extra memory of the broadcast intermediate at 1500 x 1200 x 64: 0.92 GB
```

On 300 x 1200 distances the loops took 0.53 s, broadcasting 0.14 s and the matrix product 0.005 s, a factor of about 100 between the first and the last, and the full 1500 x 1200 problem takes 0.026 s with the last. The versions agree to 4e-15. The memory line explains why the middle version does not scale: the broadcast intermediate at the full size would hold 0.92 GB. The matrix-product form has a cost of its own, which should be stated: subtracting nearly equal large numbers loses precision, so for points far from the origin with tiny distances it can return a few units of rounding error (hence the clamp), and centring the data first helps.

**Common weak answer.** "Rewrite it in C++" or "use multiprocessing" before profiling or vectorising; both can help later, and neither is the first step.

**Follow-up.** "Now 10 million points." Do not build the full matrix: process in blocks, or use an index for approximate nearest neighbours (Part 9, question 5).

## Takeaways

- State the structure and the complexity first, then code: hash map plus heap for top-k (heap selection 4 times faster than a full sort in the test), hash map plus ordering for an LRU cache (70 times faster than the list-based one), cumulative sum plus bisect for sampling.
- Preprocessing classes keep `fit` and `transform` apart; fitting on the test data hid a quarter of a shift of 2.
- k-means converges to a local minimum: the worst of 20 random initialisations had twice the best inertia; k-means++ cut the worst case from 8,000 to 4,970.
- AUC is a rank statistic: the rank formula equals the pair definition and scikit-learn, and handles 400,000 rows in 0.05 s.
- Attention is `softmax(QK^T / sqrt(d)) V` with the mask applied before the softmax; test against the library and test causality.
- Compute softmax and cross-entropy in the logit domain; the naive softmax returns `nan` for logits of 1000.
- Debug with a method: check data, initial loss, and an overfit single batch; of the four bugs tried, only the missing `zero_grad` and misaligned labels were obvious from the final loss.
- Profile, vectorise, then reformulate: the matrix-product distance was about 100 times faster than loops.

## Limits

- Every solution was run on one machine with synthetic data and checked against a library where one exists; this shows the code is correct on those inputs, not that it handles every edge case. Timings are single runs and vary between runs.
- Claims about documentation and articles are limited to the passages read: the Python `heapq` and `collections` documentation and time-complexity table, the scikit-learn clustering and `roc_auc_score` pages, the PyTorch attention and `CrossEntropyLoss` pages, and Karpathy's recipe. The statements on count-min sketches, the alias method, Fenwick trees, thread safety and the precision loss of the matrix-product distance are the author's and were not tested.
- The training-loop bugs were each applied alone on one small synthetic problem; the size of their effects depends on the problem, the learning rate and the number of epochs.
- Not covered: graph and dynamic-programming problems, string problems, concurrency, SQL (usually asked separately), and writing backpropagation by hand.
- The candidates in the examples are hypothetical, and the reference answers are one reasonable answer among several.

Next: Part 13, mathematics and statistics.
