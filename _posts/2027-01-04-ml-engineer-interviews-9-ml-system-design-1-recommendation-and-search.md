---
title: "ML Engineer Interviews, With Reference Answers (9/18): ML System Design I, Recommendation and Search"
date: 2027-01-04 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, system-design, recommender-systems, search, ranking, series]
description: "Part 9 of 18. An answer framework for ML system design and ten questions on recommendation and search ranking: two-stage architecture, negative sampling, nearest-neighbour search, learning to rank, position bias and offline evaluation, with runnable code."
---

This post covers the first half of the system-design round: a framework for answering, then recommendation and search ranking. It gives ten questions with what each tests, a reference answer, a common weak answer and a follow-up. The numbers in the answers come from code that was run for this post, on synthetic data, with the output shown; claims about papers and guides are limited to the passages that were read. The candidates are hypothetical. [Part 8](/posts/ml-engineer-interviews-8-rl-generative-models-and-distributed-training/) covered the advanced topics. Part 10 continues with fraud, forecasting, real-time versus batch, and data modelling.

The code ran on a CPU-only machine (32 cores) with Python 3.14.7, NumPy, PyTorch 2.15.0 and faiss-cpu 1.16.0, with fixed seeds. The data are synthetic, so the experiments check mechanisms; absolute numbers will differ on real data.

![Left: recall at 20 on held-out items for a synthetic recommender. Popularity 0.024, BPR 0.226, full softmax 0.271, sampled softmax with 100 negatives 0.272, oracle 0.317. Right: recall at 10 against milliseconds per query on a log scale for 200,000 vectors. HNSW reaches 0.327, 0.676 and 0.950 at 0.022, 0.063 and 0.211 milliseconds for efSearch 16, 64 and 256; exact search has recall 1.0 at 0.75 milliseconds.]({{ "/assets/pic/ml-interview-9-recall-ann.png" | relative_url }})
*Left: training objectives for retrieval (questions 3 and 4). Right: the speed and recall trade-off of approximate search (question 5).*

## 1. "Design a recommendation system. How do you structure your answer?"

**What it tests.** Whether the candidate clarifies the problem before choosing a model, and whether the answer is organised.

**Reference answer.** An order that works for most system-design questions (the author's framework; it is a checklist, not a standard):

1. Clarify the goal and the constraints: what is recommended to whom, what the product wants (watch time, purchases, retention), the latency budget, the catalogue size, and what data exist.
2. Define metrics: one online metric the business cares about, a few guardrails, and offline metrics (recall@k, NDCG) used to iterate quickly.
3. Describe the data and the labels: what counts as a positive, how negatives are obtained, how the time split works.
4. Start with a baseline and the overall architecture (below: retrieval, then ranking).
5. Discuss serving, monitoring and iteration: latency, feature consistency between training and serving, experiments, drift.

Google's [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) supports several of these steps. Rule 1: "Don't be afraid to launch a product without machine learning", because a heuristic such as install rate can get part of the way. Rule 2: "First, design and implement metrics", before formalising what the system will do. Rule 4: "Keep the first model simple and get the infrastructure right", because the first model gives the biggest boost and infrastructure problems will be more numerous than expected. Rule 12: "Don't overthink which objective you choose to directly optimize"; early on, metrics that are not optimised directly tend to rise too. An interviewer hears these as judgement: a candidate who proposes the baseline first and the deep model second sounds like someone who has shipped.

**Common weak answer.** Opening with "I would use a two-tower transformer with ..." before asking what the product is for.

**Follow-up.** "What would you ship in the first week?" A popularity or recency heuristic with the logging and metrics in place, which also produces the training data for the first model.

## 2. "Why do large recommenders use two stages, candidate generation then ranking?"

**What it tests.** Understanding of the scale constraint, and of the different job of each stage.

**Reference answer.** The catalogue is too large to score every item with an expensive model for every request. [Covington et al. (2016)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) describe the YouTube system as the classic two-stage information-retrieval split: a deep candidate generation model, and a separate deep ranking model. Candidate generation takes the user's history and selects a small subset from the very large corpus, favouring recall and speed. Ranking scores that subset with many more features describing the video and the user's relation to it; as the paper puts it, only a few hundred videos are being scored rather than millions, so a richer model is affordable, and it uses impression data to specialise and calibrate predictions for the particular interface (a user may often watch a video but be unlikely to click on a specific impression because of the thumbnail). Their ranking model is a modified logistic regression trained to predict expected watch time rather than click probability. The paper also reports that live A/B results are not always correlated with offline experiments, which is why the final decision rests on live experiments (question 9).

The design consequence is the budget split: the first stage must be cheap enough to run over the whole corpus (hence approximate nearest-neighbour search, question 5), and the second may be expensive because its input is small. Multiple retrievers can feed one ranker, and the ranker can also apply business rules and diversity.

**Common weak answer.** "One big model scores everything." It ignores the latency and cost of scoring millions of items.

**Follow-up.** "What happens if the retrieval stage misses the right item?" Nothing downstream can recover it, so retrieval is judged on recall of the eventually clicked or watched items, and the ranker on ordering quality.

## 3. "How would you train the retrieval model? There are millions of items."

**What it tests.** The cost of a full softmax and how sampling changes the training.

**Reference answer.** Covington et al. pose recommendation as extreme multiclass classification: predict which video among millions (the classes) the user watches next, given a user and context embedding u and video embeddings v_j, with a softmax over e^(v_j . u). A full softmax over millions of classes is too slow, so they sample negative classes from a background distribution ("candidate sampling"), correct for the sampling with importance weighting, and minimise cross-entropy on the true label and the sampled negatives; several thousand negatives were sampled, which they say is more than a 100 times speedup over the traditional softmax. At serving time, calibrated probabilities are not needed, so scoring reduces to a nearest-neighbour search in the dot-product space.

A toy check on synthetic implicit feedback: 3,000 users and 2,000 items; each user's 80 interactions are drawn according to a latent taste-plus-popularity score; 70 go to training and 10 are held out; recall@20 is measured over items not seen in training. The "oracle" ranks by the true score, so it bounds what any model can reach.

```python
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); rng = np.random.default_rng(0)
U, I, D = 3000, 2000, 8
pu = rng.normal(size=(U, D)); qi = rng.normal(size=(I, D)); pop = rng.normal(size=I) * 1.0
score = pu @ qi.T / np.sqrt(D) * 3.0 + 0.5 * pop                      # true affinity: personal taste + item popularity
probs = np.exp(score - score.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
inter = [rng.choice(I, size=80, replace=False, p=probs[u]) for u in range(U)]   # 80 items per user, sampled by affinity
train = [x[:70] for x in inter]; test = [set(x[70:]) for x in inter]            # 70 for training, 10 held out per user
pairs = np.array([(u, i) for u in range(U) for i in train[u]]); K = 20
def recall(S):                                                    # S: users x items scores; exclude training items
    S = S.copy()
    for u in range(U): S[u, train[u]] = -1e9
    top = np.argsort(-S, 1)[:, :K]; return np.mean([len(test[u] & set(top[u])) / len(test[u]) for u in range(U)])
cnt = np.bincount(pairs[:, 1], minlength=I); print(f"popularity baseline recall@{K}: {recall(np.tile(cnt, (U, 1)).astype(float)):.3f}")
print(f"oracle (true affinity) recall@{K}: {recall(score):.3f}")
class MF(nn.Module):
    def __init__(s): super().__init__(); s.u = nn.Embedding(U, D); s.i = nn.Embedding(I, D); nn.init.normal_(s.u.weight, std=0.1); nn.init.normal_(s.i.weight, std=0.1)
    def scores(s, u): return s.u(u) @ s.i.weight.T
def run(kind, epochs=40, nneg=100):
    m = MF(); opt = torch.optim.Adam(m.parameters(), lr=0.02); P = torch.tensor(pairs)
    for ep in range(epochs):
        perm = torch.randperm(len(P))
        for b in range(0, len(P), 2048):
            bt = P[perm[b:b+2048]]; u, i = bt[:, 0], bt[:, 1]
            if kind == "full": loss = F.cross_entropy(m.scores(u), i)
            elif kind == "sampled":
                neg = torch.randint(0, I, (len(u), nneg)); pos = (m.u(u) * m.i(i)).sum(1, keepdim=True); ns = torch.einsum("bd,bnd->bn", m.u(u), m.i(neg))
                loss = F.cross_entropy(torch.cat([pos, ns], 1), torch.zeros(len(u), dtype=torch.long))
            else:
                j = torch.randint(0, I, (len(u),)); loss = -F.logsigmoid((m.u(u) * m.i(i)).sum(1) - (m.u(u) * m.i(j)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return recall(m.scores(torch.arange(U)).numpy())
for kind in ("full", "sampled", "bpr"): print(f"{kind:8s} recall@{K}: {run(kind):.3f}")
```

```
popularity baseline recall@20: 0.024
oracle (true affinity) recall@20: 0.317
full     recall@20: 0.271
sampled  recall@20: 0.272
bpr      recall@20: 0.226
```

A matrix-factorisation model trained with the full softmax reaches 0.271 and one trained with 100 uniformly sampled negatives reaches 0.272: the same quality, with a cost per example that no longer grows with the catalogue. (Uniform sampling needs no importance correction because the 100 negatives are drawn uniformly; the paper's correction is for sampling from a non-uniform background distribution. With only 2,000 items the full softmax is cheap, so this toy shows equivalence of quality, not the speed-up.) The popularity baseline reaches 0.024 and the oracle 0.317, so the learned model recovers most of the available signal.

**Common weak answer.** "Train a classifier on clicks." It does not say what the negatives are, which is the central design decision for retrieval.

**Follow-up.** "What is wrong with random negatives?" They are mostly easy; harder negatives sharpen the model but risk including items the user would have liked, and sampling from the item-frequency distribution changes what the model learns about popularity (the importance correction above).

## 4. "What is BPR and when would you use a pairwise loss for recommendation?"

**What it tests.** Implicit feedback: the observed positives are not the only good items, and the loss should reflect the ranking goal.

**Reference answer.** [Rendle et al. (2009)](https://arxiv.org/abs/1205.2618) study item recommendation from implicit feedback (clicks, purchases), where models such as matrix factorisation and k-nearest-neighbour are designed for personalised ranking but are not directly optimised for it. They derive a criterion, BPR-Opt, as the maximum posterior estimator from a Bayesian analysis of the problem, with a learning algorithm based on stochastic gradient descent with bootstrap sampling of (user, positive item, negative item) triples. Their experiments show that it outperforms standard learning techniques for both matrix factorisation and adaptive kNN, and their conclusion is the importance of optimising for the right criterion. The loss per triple is `-log sigmoid(score(u, i) - score(u, j))` for an observed item i and an unobserved item j: it asks only that the observed item outranks the unobserved one.

In the toy experiment above the BPR-trained model (one uniformly sampled negative per positive, same epochs) reached 0.226 against 0.271 for the softmax objectives. That is a result of this synthetic setting and training budget, not a general ranking of the losses: BPR sees one negative per step, so it learns more slowly, and the data here are generated from a softmax model, which favours the softmax losses. A fair answer says the choice depends on the metric and on the sampling budget, and that both should be compared on a held-out set.

**Common weak answer.** "Treat unobserved items as negatives." Unobserved items include items the user never saw; the loss above treats them only as less preferred than the observed one, which is weaker.

**Follow-up.** "How does the choice of negatives change the model?" Exposure matters: negatives drawn only from items shown to the user say "seen but not chosen", while uniform negatives say "unseen or not chosen", which mixes preference with exposure.

## 5. "How do you retrieve the nearest items from millions of embeddings in a few milliseconds?"

**What it tests.** Approximate nearest-neighbour search, and the recall/latency trade-off.

**Reference answer.** Exact search compares the query with every vector, so its cost grows linearly with the catalogue. Approximate methods accept a small loss of recall for a large drop in latency. [Malkov and Yashunin (2016)](https://arxiv.org/abs/1603.09320) describe Hierarchical Navigable Small World graphs (HNSW): a multi-layer structure of proximity graphs, with each element's maximum layer drawn from an exponentially decaying distribution, which separates links by scale; search starts in the sparse top layer and descends. They report logarithmic complexity scaling and strong performance against the open-source baselines of the time. [Johnson et al. (2017)](https://arxiv.org/abs/1702.08734) describe the GPU library Faiss, covering brute-force, approximate and compressed-domain (product quantisation) search, and report building a k-nearest-neighbour graph of 1 billion vectors in under 12 hours on four GPUs. The trade-off is a knob (here `efSearch`, the size of the candidate list during search), and the answer should include measuring it.

A measurement on 200,000 random 64-dimensional vectors, 1,000 queries, top 10, with inner-product similarity, against exact search as ground truth (random Gaussian vectors have no cluster structure, which makes this a hard case for graph methods; real embeddings usually behave differently):

```python
import numpy as np, faiss, time
rng = np.random.default_rng(0); n, d, nq, k = 200000, 64, 1000, 10
X = rng.normal(size=(n, d)).astype("float32"); Qs = rng.normal(size=(nq, d)).astype("float32")
faiss.omp_set_num_threads(4)
flat = faiss.IndexFlatIP(d); flat.add(X); t = time.perf_counter(); _, gt = flat.search(Qs, k); tf = time.perf_counter() - t
print(f"exact (flat inner product): {tf / nq * 1000:.2f} ms/query, recall@{k} 1.000")
h = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 100; t = time.perf_counter(); h.add(X); tb = time.perf_counter() - t
print(f"HNSW build (M=32, efConstruction=100): {tb:.1f}s")
for ef in (16, 64, 256):
    h.hnsw.efSearch = ef; t = time.perf_counter(); _, I = h.search(Qs, k); ts = time.perf_counter() - t
    rec = np.mean([len(set(I[i]) & set(gt[i])) / k for i in range(nq)]); print(f"HNSW efSearch={ef:3d}: {ts / nq * 1000:.3f} ms/query, recall@{k} {rec:.3f}")
```

```
exact (flat inner product): 0.75 ms/query, recall@10 1.000
HNSW build (M=32, efConstruction=100): 13.7s
HNSW efSearch= 16: 0.022 ms/query, recall@10 0.327
HNSW efSearch= 64: 0.063 ms/query, recall@10 0.676
HNSW efSearch=256: 0.211 ms/query, recall@10 0.950
```

At `efSearch` = 256 the graph index reaches recall 0.95 at 0.211 ms per query, 3.6 times faster than exact search; at 16 it is 34 times faster and finds a third of the true neighbours (right panel of the figure). Timing figures depend on the machine and thread count, and the exact baseline here is a brute-force search that is already vectorised; the gap grows with the catalogue size. The build took 13.7 seconds, which is the other cost: an index must be built and refreshed as items and embeddings change. For a recommender, the right recall target comes from the downstream metric: Covington et al. report that their A/B results were not particularly sensitive to the choice of nearest-neighbour algorithm.

**Common weak answer.** "Use a vector database." That names a product, not the trade-off or the index.

**Follow-up.** "The item embeddings are retrained daily. What happens to the index?" It is rebuilt or updated, and the user-side model and the item index must come from the same model version, or the dot products are meaningless.

## 6. "How do you handle new items and new users? What do Wide & Deep models address?"

**What it tests.** Cold start, and the memorisation versus generalisation distinction.

**Reference answer.** [Cheng et al. (2016)](https://arxiv.org/abs/1606.07792) define the tension: memorisation is learning the frequent co-occurrence of items or features and exploiting the correlation in the historical data; generalisation is based on the transitivity of correlation and explores feature combinations that have never or rarely occurred. Memorisation tends to give topical, directly relevant recommendations; generalisation improves diversity. Their abstract describes the two failure modes: wide models with cross-product features memorise well but need feature engineering to generalise, and deep models with embeddings generalise to unseen combinations but can over-generalise and recommend less relevant items when user-item interactions are sparse and high-rank. Wide & Deep trains a wide linear model and a deep network jointly; on Google Play, with over a billion active users and over one million apps, online experiments showed a significant increase in app acquisitions over wide-only and deep-only models.

Cold start is a related problem, and the standard remedies are the author's summary rather than claims from these papers. For a new item with no interactions: use content features (text, category, image embeddings) in the model so it has a representation before it has clicks, reserve a share of exposure to gather data, and boost recency. For a new user: start from context (device, location, time, referrer), popular items and a short onboarding signal, and move to personalised retrieval after a few interactions. In each case, say how exposure is measured, because an item that is never shown never produces a click (the logging problem in question 8). Covington et al. add one more design point for freshness: the model should include the age of the training example as a feature, so that it can represent time-dependent popularity (their Figure 4); without it the model predicts roughly the average likelihood over the training window.

**Common weak answer.** "Recommend popular items to new users." It is a reasonable first step and says nothing about how the system transitions to personalisation, or about new items.

**Follow-up.** "How do you evaluate cold-start performance?" Slice the offline and online metrics by the age of the item and by the user's history length; an overall average hides it.

## 7. "How would you build search ranking? Explain pointwise versus pairwise learning to rank and NDCG."

**What it tests.** The retrieval-then-ranking structure for search, the loss families, and a ranking metric.

**Reference answer.** Search has the same two stages as recommendation, with the query as the user signal. First-stage retrieval classically uses sparse lexical scoring such as TF-IDF or BM25; [Karpukhin et al. (2020)](https://arxiv.org/abs/2004.04906) call these the de facto method for open-domain question answering and show that a dense dual-encoder retriever, trained from a small number of question-passage pairs, outperforms a strong Lucene-BM25 system by 9 to 19 points absolute in top-20 passage retrieval accuracy on the datasets they test. Whether dense, sparse or a combination of the two is best depends on the corpus and the queries, and should be measured (the combination is the author's suggestion). A second-stage ranker then scores the top few hundred candidates with richer features: text-match scores, document quality, freshness, click history.

The learning-to-rank loss families: pointwise (regress or classify each document's grade independently), pairwise (learn that a more relevant document scores higher than a less relevant one for the same query) and listwise (optimise a list-level objective directly). The metric is normalised discounted cumulative gain: gain `2^grade - 1`, discounted by `1/log2(position + 1)`, divided by the best possible value for that query. A worked example, and a comparison of pointwise and pairwise training of the same small network on synthetic data (600 training queries, 400 test queries, 30 documents each, five relevance grades):

```python
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); rng = np.random.default_rng(0); Q, L, Fd = 600, 30, 8
def make(nq):
    X = rng.normal(size=(nq, L, Fd)); s = X[..., 0] * 1.0 + X[..., 1] * X[..., 2] + np.maximum(X[..., 3], 0) + 0.3 * rng.normal(size=(nq, L))
    rel = np.digitize(s - s.mean(1, keepdims=True), np.quantile(s - s.mean(1, keepdims=True), [0.6, 0.8, 0.93, 0.98])); return torch.tensor(X, dtype=torch.float32), torch.tensor(rel)
def ndcg(scores, rel, k=10):
    top = scores.argsort(1, descending=True)[:, :k]; g = torch.gather(rel, 1, top).float(); disc = 1 / torch.log2(torch.arange(2, k + 2).float())
    ideal = rel.float().sort(1, descending=True)[0][:, :k]; return (((2 ** g - 1) * disc).sum(1) / ((2 ** ideal - 1) * disc).sum(1).clamp(min=1e-9)).mean().item()
Xtr, rtr = make(Q); Xte, rte = make(400)
def net(): return nn.Sequential(nn.Linear(Fd, 32), nn.ReLU(), nn.Linear(32, 1))
def train(kind):
    m = net(); opt = torch.optim.Adam(m.parameters(), lr=0.01)
    for ep in range(150):
        s = m(Xtr).squeeze(-1)
        if kind == "pointwise": loss = F.mse_loss(s, rtr.float())
        else:
            d = s.unsqueeze(2) - s.unsqueeze(1); mask = (rtr.unsqueeze(2) > rtr.unsqueeze(1)).float(); loss = (F.softplus(-d) * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return ndcg(m(Xte).squeeze(-1), rte)
print(f"random ranking NDCG@10: {ndcg(torch.rand(400, L), rte):.3f}")
print(f"pointwise (MSE on grade) NDCG@10: {train('pointwise'):.3f}"); print(f"pairwise (logistic on score differences) NDCG@10: {train('pairwise'):.3f}")
print("worked example: grades in ranked order [3,2,3,0,1,2], k=6 ->", end=" ")
g = np.array([3, 2, 3, 0, 1, 2]); disc = 1 / np.log2(np.arange(2, 8)); dcg = ((2 ** g - 1) * disc).sum(); idcg = ((2 ** np.sort(g)[::-1] - 1) * disc).sum(); print(f"DCG {dcg:.3f}, ideal {idcg:.3f}, NDCG {dcg / idcg:.3f}")
```

```
random ranking NDCG@10: 0.266
pointwise (MSE on grade) NDCG@10: 0.967
pairwise (logistic on score differences) NDCG@10: 0.965
worked example: grades in ranked order [3,2,3,0,1,2], k=6 -> DCG 13.848, ideal 14.595, NDCG 0.949
```

Random ranking scores 0.266 and both trained rankers about 0.97, with no meaningful difference between pointwise and pairwise on this easy synthetic task. The result is worth stating plainly: the claim that pairwise or listwise losses win is not something this experiment shows; the choice should be made on real data and the real metric. In the worked example the ranking [3, 2, 3, 0, 1, 2] scores 0.949 because the second and third positions are swapped relative to the ideal order and the later grades are not sorted.

**Common weak answer.** "Use BERT to rank." It names a model without the candidate stage, the label source or the metric.

**Follow-up.** "Where do the relevance labels come from?" Human raters, clicks (biased, see the next question), or a mix; the label source determines what the model learns to optimise.

## 8. "Clicks are biased by position. How do you train on them?"

**What it tests.** Understanding of implicit-feedback bias, and a correction.

**Reference answer.** A result at rank 1 is clicked more than the same result at rank 10 because users examine the top more, not because it is more relevant. [Joachims et al. (2017)](https://arxiv.org/abs/1608.04468) state that position bias in search rankings strongly influences how many clicks a result receives, so using click data directly as a training signal in learning to rank yields suboptimal results. They present a counterfactual inference framework that gives the theoretical basis for unbiased learning to rank from biased data, derive a propensity-weighted Ranking SVM, and use click models to estimate the propensities. The correction is inverse propensity scoring: weight each click by one over the probability that the result was examined at its position.

A simulation with a known examination probability of 1/rank: documents have a true relevance, a logging policy ranks them with noise (so some relevant documents sit low), and a user clicks with probability examination x relevance:

```python
import numpy as np
rng = np.random.default_rng(0); n_docs, n_q, K = 10, 40000, 10
rel = rng.uniform(0.05, 0.9, size=(n_q, n_docs)); examine = 1 / (np.arange(1, K + 1)) ** 1.0        # chance a user even looks at rank r
logged_order = np.argsort(-(rel + rng.normal(0, 0.35, size=rel.shape)), 1)    # noisy logging policy: some good documents sit low
pos_of = np.empty_like(logged_order); np.put_along_axis(pos_of, logged_order, np.arange(K)[None, :].repeat(n_q, 0), 1)
click = (rng.random(rel.shape) < examine[pos_of] * rel)
bins = np.digitize(rel, [0.25, 0.5, 0.75]); tail = pos_of >= 7
print("documents ranked 8-10 by the logging policy, grouped by true relevance:")
for b, name in enumerate(["0.05-0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90"]):
    msk = tail & (bins == b); print(f"  relevance {name}: true mean {rel[msk].mean():.3f}, naive click rate {click[msk].mean():.3f}, IPS-weighted click estimate {(click[msk] / examine[pos_of[msk]]).mean():.3f}")
top = pos_of == 0; print(f"documents at rank 1: true relevance mean {rel[top].mean():.3f}, naive click rate {click[top].mean():.3f}")
print(f"all documents: naive click rate at rank 1 vs rank 10 for equal relevance (0.5): {0.5 * examine[0]:.3f} vs {0.5 * examine[9]:.3f}")
```

```
documents ranked 8-10 by the logging policy, grouped by true relevance:
  relevance 0.05-0.25: true mean 0.144, naive click rate 0.016, IPS-weighted click estimate 0.143
  relevance 0.25-0.50: true mean 0.362, naive click rate 0.041, IPS-weighted click estimate 0.362
  relevance 0.50-0.75: true mean 0.605, naive click rate 0.070, IPS-weighted click estimate 0.607
  relevance 0.75-0.90: true mean 0.816, naive click rate 0.092, IPS-weighted click estimate 0.789
documents at rank 1: true relevance mean 0.684, naive click rate 0.683
all documents: naive click rate at rank 1 vs rank 10 for equal relevance (0.5): 0.500 vs 0.050
```

At ranks 8 to 10 the naive click rate is 0.016 to 0.092 for documents whose true relevance is 0.14 to 0.82: a click-trained model would conclude that these documents are poor. The inverse-propensity-weighted estimate recovers 0.143, 0.362, 0.607 and 0.789 (true: 0.144, 0.362, 0.605, 0.816). The correction works here because the examination probabilities are known exactly; in practice they must be estimated, for instance with a click model or a small randomisation experiment, and low propensities make the weighted estimate noisy (the last row's 0.789 against 0.816 is a sign of this). Joachims et al. also note that the same framework can handle other biases, such as trust bias and document popularity, with appropriate click models.

**Common weak answer.** "Use clicks as labels." It trains the next model on the last model's ranking.

**Follow-up.** "How would you measure the propensities?" By a small randomised swap of adjacent positions, or by fitting a click model; both cost some user experience, which has to be weighed.

## 9. "How do you evaluate a recommender offline, and what can go wrong?"

**What it tests.** Evaluation design: splits, leakage, baselines, and the relationship between offline and online results.

**Reference answer.** Three pitfalls, each with a source. First, the split. Covington et al. report that many collaborative-filtering systems choose labels and context by holding out a random item and predicting it from the rest of the user's history, which leaks future information and ignores asymmetric consumption patterns (users often watch episodes in order, for instance); they instead "roll back" the history, using only actions before the held-out watch, and found much better performance predicting the next watch than a random held-out one. A simulation shows the size of the effect when item popularity is short-lived: items are popular for only a few days, and the "model" is the list of the ten most-viewed items in the training data.

```python
import numpy as np
rng = np.random.default_rng(0); n_items, days, n_events = 500, 60, 200000
start = rng.integers(0, days - 5, size=n_items); life = rng.integers(3, 8, size=n_items)           # each item is "hot" for a few days only
day = rng.integers(0, days, size=n_events)
alive = (day[:, None] >= start[None]) & (day[:, None] < (start + life)[None]); w = alive * 1.0 + 1e-3
item = np.array([rng.choice(n_items, p=w[i] / w[i].sum()) for i in range(20000)]); day = day[:20000]
def hit_rate(train_idx, test_idx, k=10):                                                         # recommend the k most-viewed items in training, count test events they cover
    cnt = np.bincount(item[train_idx], minlength=n_items); top = np.argsort(-cnt)[:k]; return np.isin(item[test_idx], top).mean()
idx = rng.permutation(len(item)); cut = int(0.8 * len(item)); print(f"random 80/20 split: hit rate of 'top-10 most viewed' {hit_rate(idx[:cut], idx[cut:]):.3f}")
order = np.argsort(day); print(f"temporal split (train on the first 80% of time, test on the last 20%): {hit_rate(order[:cut], order[cut:]):.3f}")
```

```
random 80/20 split: hit rate of 'top-10 most viewed' 0.048
temporal split (train on the first 80% of time, test on the last 20%): 0.001
```

The random split reports a hit rate of 0.048 and the temporal split 0.001, because the items popular in training are no longer popular afterwards. The effect is exaggerated by the construction (every item is hot for only a few days), but the direction is what the paper warns about: a random split rewards a model for knowing the future.

Second, the baselines. [Rendle et al. (2019)](https://arxiv.org/abs/1905.01395) show on the Movielens 10M benchmark that results reported for baselines over five years were suboptimal: a carefully set up vanilla matrix factorisation outperforms the reported results of any newly proposed method, and they conclude that empirical findings in papers are questionable unless obtained on standardised benchmarks where baselines have been extensively tuned. The interview lesson: tune the baseline as hard as the new model before claiming an improvement.

Third, offline versus online. Covington et al. rely on live A/B tests for the final decision, because live results are not always correlated with offline experiments. A complete answer lists offline metrics (recall@k for retrieval, NDCG for ranking, with a temporal split), then a staged online test with guardrail metrics, and checks segments (new users, new items, regions) for effects the average hides. Part 4 covers experimentation in detail.

**Common weak answer.** "Compute accuracy on a random test set." For a ranking task with a huge item space, accuracy over a random split is neither the right metric nor the right split.

**Follow-up.** "The offline metric improved by 3% and the A/B test is flat. What do you check?" The offline label may not match the online goal, the offline split may leak, the new model may change exposure, and the A/B test may be underpowered; check each in turn.

## 10. "Your recommender has hundreds of millions of items and users. How do embeddings affect the system design?"

**What it tests.** Memory arithmetic for embedding tables, and training/serving consistency.

**Reference answer.** Categorical features (user ID, item ID, category) enter the model as embeddings: rows in a table of size (number of categories) x (embedding dimension). [Naumov et al. (2019)](https://arxiv.org/abs/1906.00091), in the DLRM paper, note that these networks differ significantly from other deep learning networks because of the need to handle categorical features, and design a parallelisation scheme that uses model parallelism on the embedding tables to mitigate memory constraints, with data parallelism for the fully connected layers. The memory arithmetic shows why:

```python
for n, d, name in ((10e6, 64, "10 million items"), (100e6, 64, "100 million items"), (1e9, 128, "1 billion IDs, 128 dims")):
    for bytes_, tag in ((4, "fp32"), (2, "fp16")):
        print(f"{name:25s} {tag}: {n * d * bytes_ / 2**30:8.1f} GiB")
print(f"hashing 1 billion IDs into 50 million rows at 128 dims fp32: {50e6 * 128 * 4 / 2**30:.1f} GiB")
```

```
10 million items          fp32:      2.4 GiB
10 million items          fp16:      1.2 GiB
100 million items         fp32:     23.8 GiB
100 million items         fp16:     11.9 GiB
1 billion IDs, 128 dims   fp32:    476.8 GiB
1 billion IDs, 128 dims   fp16:    238.4 GiB
hashing 1 billion IDs into 50 million rows at 128 dims fp32: 23.8 GiB
```

A table for a billion IDs at 128 dimensions in fp32 takes 477 GiB, so it has to be sharded across machines (model parallelism, as in the DLRM design), compressed (lower precision, smaller dimension) or reduced (hashing IDs into fewer rows at the price of collisions; the hashing remedy is the author's note, not a claim from the papers read). Each choice changes the quality, so the answer should end with an evaluation.

The second half of the answer is consistency. Rule 29 of the Rules of Machine Learning says the best way to make sure you train like you serve is to save the features used at serving time and log them for training; the guide reports that the YouTube home page switched to logging features at serving time with significant quality improvements and reduced code complexity. Rule 37 asks you to measure training/serving skew, including the gap between holdout and next-day performance. For embeddings, the same rule applies to the model version: the user embedding computed online and the item embeddings in the index must come from the same training run.

**Common weak answer.** "Use a bigger embedding for better quality." It ignores the memory bill and the collision and freshness trade-offs.

**Follow-up.** "How do you serve an embedding for an item that appeared an hour ago?" Compute it from content features at ingestion until it has enough interactions, and refresh the index incrementally; the staleness window is a design parameter.

## Takeaways

- Start from the goal and the metrics, ship a heuristic first, and add the model second (Rules of Machine Learning 1, 2, 4).
- Two stages: cheap, high-recall retrieval over the corpus, then an expensive ranker over a few hundred candidates.
- Sampled negatives matched the full softmax in the toy test (0.272 against 0.271 recall@20); the loss choice should be compared on held-out data, not assumed.
- Approximate search is a dial: 0.95 recall at 3.6 times the speed of exact search in the test above, 0.33 recall at 34 times.
- Clicks are position-biased by construction; inverse propensity weights recovered the true relevance in the simulation when the propensities were known.
- Evaluate with a temporal split, tune the baseline, and let an online experiment make the final decision.
- Embedding tables dominate memory at scale; log serving features and keep model versions consistent.

## Limits

- All experiments use synthetic data on one CPU machine; absolute numbers (recall, speed, hit rates) do not transfer to real catalogues. The temporal-split simulation is deliberately extreme.
- Claims about papers and guides are limited to the passages read (abstracts, the sections quoted, and the Rules of Machine Learning rules cited), at the time they were written.
- Items marked as the author's own (the answer framework, cold-start remedies, the dense-plus-sparse suggestion, hashing of IDs) have no source behind them.
- The pairwise and pointwise comparison and the BPR comparison are results of toy setups and do not show that one method is better in general.
- Not covered: multi-objective ranking and re-ranking for diversity, reinforcement-learning recommenders, graph-based models, session-based models such as the recurrent approach of Hidasi et al. (not read in depth), privacy, and fairness in ranking.
- The candidates in the examples are hypothetical, and the reference answers are one reasonable answer among several.

Next: Part 10, the second part on system design: fraud detection, forecasting, real-time versus batch, and data modelling.
