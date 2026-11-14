---
title: "ML Engineer Interviews, With Reference Answers (7/18): LLM Engineering"
date: 2027-01-02 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, llm, transformers, fine-tuning, rag, series]
description: "Part 7 of 18. Ten interview questions on how large language models are built, trained, adapted and served: attention, tokenisation, scaling laws, memory, KV cache, LoRA, alignment and retrieval, with reference answers and runnable code."
---

This post covers the LLM questions that now appear in most machine learning interviews: how attention and tokenisation work, what scaling laws say, where training and serving memory goes, how fine-tuning and alignment methods differ, and when to use retrieval. It gives ten questions with what each tests, a reference answer, a common weak answer and a follow-up. Numbers in the answers come from code that was run for this post, shown with its output; claims about papers are limited to what the abstracts and the sections read for this post say. The candidates in the examples are hypothetical. [Part 6](/posts/ml-engineer-interviews-6-deep-learning-and-training/) covered deep learning and training.

The code ran on a CPU-only machine with Python 3.14.7, NumPy and PyTorch 2.15.0, with fixed seeds. The toy models are small by design: they show the mechanism and the arithmetic, not the behaviour of a production model.

![Left: mean largest softmax weight over 64 random keys against key dimension from 16 to 1024. Without scaling it rises from 0.58 to 0.95; with scaling by 1 over square root of d_k it stays near 0.107. Right: KV cache size in GiB against tokens per sequence for 1, 8 and 32 sequences, for a model with 32 layers, 32 heads and 128 dimensions per head in fp16; at 32,768 tokens one sequence needs 16 GiB, more than the 14 GB of fp16 weights of a 7B-parameter model.]({{ "/assets/pic/ml-interview-7-attention-kv-cache.png" | relative_url }})
*Left: why attention scores are scaled (question 1). Right: how the KV cache grows with context length and batch size (question 6).*

## 1. "Walk me through self-attention. Why divide by the square root of the key dimension?"

**What it tests.** Whether the candidate can write the computation and explain each part, not recite it.

**Reference answer.** Each token is projected to a query, a key and a value. The score between two tokens is the dot product of one's query with the other's key; a softmax over the scores gives weights, and the output is the weighted sum of the values: `softmax(QK^T / sqrt(d_k)) V`. [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) give the reason for the scale factor: for large d_k the dot products grow large in magnitude, pushing the softmax into regions where it has extremely small gradients, so they scale the products by 1/sqrt(d_k). The footnote's argument is that if the components of q and k are independent with mean 0 and variance 1, their dot product has mean 0 and variance d_k.

That claim is easy to check. The experiment draws random queries and 64 random keys per query and measures the spread of the scores and how peaked the softmax is:

```python
import numpy as np
rng = np.random.default_rng(0)
def softmax(z): e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
print(" d_k   std of q.k   std of q.k/sqrt(d_k)   mean max softmax weight (64 keys): unscaled / scaled")
for dk in (16, 64, 256, 1024):
    q = rng.normal(size=(2000, dk)); K = rng.normal(size=(2000, 64, dk))
    s = np.einsum("nd,nkd->nk", q, K)
    print(f"{dk:5d}   {s.std():9.2f}   {(s/np.sqrt(dk)).std():18.2f}   {softmax(s).max(-1).mean():.3f} / {softmax(s/np.sqrt(dk)).max(-1).mean():.3f}")
q = rng.normal(size=(1, 1024)); K = rng.normal(size=(64, 1024)); s = (K @ q.T)[:, 0]
for name, z in (("unscaled", s), ("scaled", s / 32)):
    p = softmax(z); J = np.diag(p) - np.outer(p, p)
    print(f"d_k=1024 {name:9}: Frobenius norm of the softmax Jacobian {np.linalg.norm(J):.4f}")
```

```
 d_k   std of q.k   std of q.k/sqrt(d_k)   mean max softmax weight (64 keys): unscaled / scaled
   16        3.99                 1.00   0.581 / 0.107
   64        8.00                 1.00   0.793 / 0.108
  256       16.01                 1.00   0.896 / 0.107
 1024       32.00                 1.00   0.950 / 0.107
d_k=1024 unscaled : Frobenius norm of the softmax Jacobian 0.0000
d_k=1024 scaled   : Frobenius norm of the softmax Jacobian 0.1928
```

The standard deviation of the scores equals sqrt(d_k), as the footnote says. Unscaled, the largest softmax weight is 0.95 at d_k = 1024, so the softmax is nearly one-hot and its Jacobian, which is what gradients flow through, is zero to four decimals (0.0000, against 0.1928 with scaling). Scaled, the largest weight stays at about 0.107 whatever d_k is.

**Common weak answer.** "It normalises the scores." True and not the reason: the point is keeping the softmax out of its saturated region.

**Follow-up.** "What is the cost of attention in sequence length?" Quadratic in time and memory for the full score matrix; question 6 covers what inference does about it.

## 2. "Why multiple heads, and how does the model know word order?"

**What it tests.** Two design facts that are often stated without a reason.

**Reference answer.** Multi-head attention projects queries, keys and values h times with different learned projections to smaller dimensions and runs attention in parallel; the paper says it lets the model jointly attend to information from different representation subspaces at different positions, and that with a single head, averaging inhibits this. The original model uses h = 8 heads. Attention itself has no notion of order: since the model contains no recurrence and no convolution, the paper states that positional information must be injected, and adds positional encodings to the input embeddings. The original used fixed sinusoids; learned encodings are another option the paper mentions. Newer models use other position schemes, which this post does not cover.

**Common weak answer.** "Heads let the model look at different words." Close, but it skips the subspace point and the cost: with the dimension split across heads, the total cost is similar to a single head of full size (the author's reading of the paper's design, not a quoted claim).

**Follow-up.** "What does a causal mask do?" It stops each position from attending to later positions, which is what lets a decoder be trained to predict the next token on all positions in parallel.

## 3. "How does tokenisation work, and why subwords?"

**What it tests.** Whether the candidate understands the open-vocabulary problem and can run the algorithm.

**Reference answer.** A fixed word vocabulary cannot cover rare words, names and new compounds. [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909) adapt byte pair encoding to word segmentation: start from characters (plus an end-of-word symbol), repeatedly count symbol pairs and merge the most frequent into a new symbol, for a chosen number of merges. Rare words end up as sequences of subword units; frequent words become single tokens. The number of merges sets the vocabulary size.

Their algorithm on the toy vocabulary from the paper (low x5, lower x2, newest x6, widest x3):

```python
import re, collections
corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
vocab = collections.Counter(" ".join(w) + " </w>" for w in corpus.split())
def pairs(v):
    c = collections.Counter()
    for w, f in v.items():
        s = w.split()
        for a, b in zip(s, s[1:]): c[(a, b)] += f
    return c
def merge(pair, v):
    pat = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
    return {pat.sub("".join(pair), w): f for w, f in v.items()}
for i in range(10):
    p = pairs(vocab); best = max(p, key=p.get); vocab = merge(best, vocab); print(f"merge {i+1}: {best} (count {p[best]})")
print(dict(vocab))
```

```
merge 1: ('e', 's') (count 9)
merge 2: ('es', 't') (count 9)
merge 3: ('est', '</w>') (count 9)
merge 4: ('l', 'o') (count 7)
merge 5: ('lo', 'w') (count 7)
merge 6: ('n', 'e') (count 6)
merge 7: ('ne', 'w') (count 6)
merge 8: ('new', 'est</w>') (count 6)
merge 9: ('low', '</w>') (count 5)
merge 10: ('w', 'i') (count 3)
{'low</w>': 5, 'low e r </w>': 2, 'newest</w>': 6, 'wi d est</w>': 3}
```

The first merges join the most frequent pair, `e s` (nine occurrences: six in "newest", three in "widest"), then build `est</w>`, then `low` and `newest`. After ten merges, "low" and "newest" are single tokens, while "lower" is `low` plus `e r </w>` and "widest" is `wi d est</w>`: the rare word is split into pieces that appear elsewhere. (The order of merges between ties may differ from the paper's figure, which uses a different toy dictionary.) Production tokenisers, many of them byte-level variants of this idea, are not covered by the sources read for this post.

Consequences an interviewer expects: token counts, not characters, set cost and context limits; models handle rare or misspelled words as several tokens; and text that tokenises into many tokens (some languages, long numbers) costs more.

**Common weak answer.** "Words are split into tokens." It misses the frequency-driven merge and the open-vocabulary motivation.

**Follow-up.** "Why can models be bad at character counting?" The model sees tokens, not characters (the author's explanation; no source read here tests it).

## 4. "What do scaling laws say, and how would you use them?"

**What it tests.** Reading empirical results correctly and without overclaiming.

**Reference answer.** [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) find that the loss scales as a power law with model size, dataset size and training compute, with trends spanning more than seven orders of magnitude, and that other architectural details such as width or depth have minimal effects within a wide range. Their conclusion on allocation: larger models are significantly more sample-efficient, so compute-efficient training involves very large models trained on a relatively modest amount of data and stopped well before convergence. [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) revisit the question over more than 400 models, and conclude that model size and training tokens should be scaled equally: for every doubling of model size, double the training tokens. They tested it with Chinchilla, which uses the same compute budget as the 280B-parameter Gopher with 70B parameters and four times more data, and which outperforms Gopher, GPT-3 and others on a large range of downstream tasks. The practical use is to plan a training run: given a compute budget, pick model size and token count together, and use small runs to fit the curve before committing. Because a smaller model is also cheaper at inference, the paper notes it is easier to use downstream.

The two papers disagree on how to split a fixed budget, which is a good thing for a candidate to mention: the later work concluded that current large models were too big and under-trained.

**Common weak answer.** "Bigger is always better." Scaling laws are about loss on a particular distribution and a particular budget; the 70B model beat the 280B one.

**Follow-up.** "Does a lower loss mean a better product?" Not by itself. Loss is measured on the training distribution; user-facing quality depends on alignment, retrieval and evaluation (questions 8 to 10).

## 5. "How much GPU memory does it take to train a 7B-parameter model?"

**What it tests.** Back-of-envelope estimation, and knowing that the weights are the smallest part.

**Reference answer.** Count bytes per parameter, then add activations. [Rajbhandari et al. (2020)](https://arxiv.org/abs/1910.02054) (the ZeRO paper) give the accounting for mixed-precision training with Adam: an fp16 copy of the parameters (2 bytes each) and of the gradients (2 bytes), plus optimiser states, which are an fp32 copy of the parameters, the momentum and the variance (4 + 4 + 4 bytes), written as K = 12. The total is 2Ψ + 2Ψ + KΨ = 16Ψ bytes for Ψ parameters. For GPT-2 with 1.5 billion parameters the paper gives at least 24 GB, against 3 GB for the fp16 parameters alone. Activations, temporary buffers and fragmentation come on top; the paper treats them separately, and the activation share depends on batch size, sequence length and checkpointing.

```python
for psi in (1.5e9, 7e9, 70e9):
    print(f"{psi/1e9:>5.1f}B params: fp16 weights {2*psi/1e9:7.0f} GB | mixed-precision Adam model states (2+2+12 bytes/param) {16*psi/1e9:8.0f} GB")
```

```
  1.5B params: fp16 weights       3 GB | mixed-precision Adam model states (2+2+12 bytes/param)       24 GB
  7.0B params: fp16 weights      14 GB | mixed-precision Adam model states (2+2+12 bytes/param)      112 GB
 70.0B params: fp16 weights     140 GB | mixed-precision Adam model states (2+2+12 bytes/param)     1120 GB
```

A 7B model needs about 112 GB for model states alone, which is eight times the 14 GB of the weights, so the states must be partitioned or reduced. ZeRO partitions the optimiser states, gradients and parameters across data-parallel workers; the other levers are lower-precision optimiser states, activation checkpointing, and parameter-efficient fine-tuning (question 7). The distributed-training side is the subject of Part 8.

**Common weak answer.** "7B parameters at 2 bytes is 14 GB, so a 16 GB card is enough." That is inference-only arithmetic.

**Follow-up.** "What changes if you freeze the base model?" Gradients and optimiser states exist only for the trainable parameters, which is why adapters are cheap.

## 6. "Why does generation use a KV cache, and what does it cost?"

**What it tests.** The difference between training and autoregressive inference, and serving-memory arithmetic.

**Reference answer.** When generating token t, attention needs the keys and values of all earlier tokens. Without a cache, every step recomputes them for the whole prefix; with a cache, each step projects only the new token and appends its key and value. The outputs are identical; the cost moves from recomputation to memory. The cache holds, per token, 2 (keys and values) x layers x heads x head dimension x bytes.

```python
import torch, time
torch.manual_seed(0)
d, T = 256, 512
Wq, Wk, Wv = (torch.randn(d, d) / d**0.5 for _ in range(3)); X = torch.randn(T, d)
def attend(q, K, V): return torch.softmax(q @ K.T / d**0.5, -1) @ V
t0 = time.perf_counter(); outs = []
for t in range(T):                       # no cache: recompute K and V for the whole prefix each step
    H = X[:t+1]; outs.append(attend(H[-1] @ Wq, H @ Wk, H @ Wv))
t1 = time.perf_counter(); no_cache = torch.stack(outs)
K = torch.empty(0, d); V = torch.empty(0, d); outs = []
for t in range(T):                       # cache: project only the new token, append
    K = torch.cat([K, X[t:t+1] @ Wk]); V = torch.cat([V, X[t:t+1] @ Wv]); outs.append(attend(X[t] @ Wq, K, V))
t2 = time.perf_counter(); cached = torch.stack(outs)
print(f"max |no cache - cache| = {(no_cache - cached).abs().max().item():.2e}")
print(f"time without cache {t1 - t0:.3f}s, with cache {t2 - t1:.3f}s, ratio {(t1 - t0) / (t2 - t1):.1f}x (single head, d={d}, {T} tokens)")
layers, heads, dh, bytes_ = 32, 32, 128, 2
per_tok = 2 * layers * heads * dh * bytes_
print(f"KV cache per token for 32 layers x 32 heads x 128 dims in fp16: {per_tok} bytes = {per_tok / 2**20:.2f} MiB")
for n in (4096, 32768): print(f"  {n} tokens: {per_tok * n / 2**30:.1f} GiB per sequence")
```

```
max |no cache - cache| = 1.37e-06
time without cache 0.133s, with cache 0.073s, ratio 1.8x (single head, d=256, 512 tokens)
KV cache per token for 32 layers x 32 heads x 128 dims in fp16: 524288 bytes = 0.50 MiB
  4096 tokens: 2.0 GiB per sequence
  32768 tokens: 16.0 GiB per sequence
```

The outputs match to float rounding (1.4e-6). On this toy single head the cache is 1.8 times faster at 512 tokens (3.4 times in an earlier run with a different thread setting), so the ratio depends on the machine; it grows with length because the uncached cost per step grows with the prefix. The memory side, for a configuration of that shape (the shape of a 7B-class model, chosen here as an example), is 0.5 MiB per token: 2 GiB for a 4,096-token sequence and 16 GiB for 32,768 tokens, more than the 14 GB of fp16 weights, and it scales with the number of concurrent sequences (figure above). [Kwon et al. (2023)](https://arxiv.org/abs/2309.06180) report that the KV cache of each request is huge, grows and shrinks dynamically, and is wasted by fragmentation and duplication in existing systems, which limits the batch size. For a 13B model on a 40 GB A100, their figure shows the parameters at 26 GB (65%) and the KV cache above 30%. Their vLLM uses PagedAttention, inspired by operating-system paging, to get near-zero waste in KV cache memory and to share cache within and across requests. The levers a candidate can name: grouping keys and values across heads, quantising the cache, paging, and limiting context length; the sources read here cover only paging.

**Common weak answer.** "The cache speeds things up." It misses the memory bill, which is what limits throughput.

**Follow-up.** "Prefill and decoding behave differently. How?" Prefill processes the prompt in parallel; decoding produces one token per step and is limited by memory traffic (the author's summary; no source read here measures it).

## 7. "Full fine-tuning, LoRA and QLoRA: when would you use each?"

**What it tests.** The trade-offs of adaptation methods, with the arithmetic behind them.

**Reference answer.** Full fine-tuning updates all parameters and needs the memory in question 5. [Hu et al. (2022)](https://arxiv.org/abs/2106.09685) freeze the pretrained weight W0 and learn a low-rank update, h = W0 x + BAx, with B of size d x r, A of size r x k and r much smaller than min(d, k); A is initialised randomly and B to zero, so the update starts at zero. Compared with fine-tuning GPT-3 175B with Adam, they report 10,000 times fewer trainable parameters and 3 times less GPU memory, quality on par or better than full fine-tuning on the models they test, and, unlike adapters, no additional inference latency (the update can be merged into W0). [Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314) go further with QLoRA: backpropagate through a frozen 4-bit quantised model into the LoRA adapters, which lets them fine-tune a 65B-parameter model on a single 48 GB GPU while preserving full 16-bit fine-tuning task performance, as reported.

A minimal LoRA layer, checked for the three properties the paper claims:

```python
import torch, torch.nn as nn
torch.manual_seed(0)
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__(); self.base = base; self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01); self.B = nn.Parameter(torch.zeros(base.out_features, r)); self.s = alpha / r
    def forward(self, x): return self.base(x) + self.s * (x @ self.A.T) @ self.B.T
d = 4096; base = nn.Linear(d, d, bias=False); l = LoRALinear(base, r=8); x = torch.randn(4, d)
print(f"B = 0 at init -> output identical to base: {torch.equal(l(x), base(x))}")
tr = sum(p.numel() for p in l.parameters() if p.requires_grad); fr = base.weight.numel()
print(f"one {d}x{d} layer: frozen {fr:,}, trainable {tr:,} ({fr / tr:.0f}x fewer)")
opt = torch.optim.Adam([l.A, l.B], lr=1e-3); target = torch.randn(4, d)
for _ in range(20): opt.zero_grad(); ((l(x) - target)**2).mean().backward(); opt.step()
W = base.weight + l.s * l.B @ l.A
print(f"after 20 steps, merged weight gives the same output: max diff {(x @ W.T - l(x)).abs().max().item():.2e}; rank of update {torch.linalg.matrix_rank(l.s * l.B @ l.A).item()}")
print(f"r=8 on 4 square {d}x{d} matrices in each of 32 layers: {32 * 4 * 2 * d * 8:,} trainable parameters")
```

```
B = 0 at init -> output identical to base: True
one 4096x4096 layer: frozen 16,777,216, trainable 65,536 (256x fewer)
after 20 steps, merged weight gives the same output: max diff 1.67e-06; rank of update 8
r=8 on 4 square 4096x4096 matrices in each of 32 layers: 8,388,608 trainable parameters
```

The layer starts identical to the base model, trains 256 times fewer parameters than the frozen matrix holds, and the learned update has rank 8 and can be folded into the weight with no change in output (1.7e-6, float rounding), which is why there is no inference latency. The last line is an illustration for a 7B-class shape (adapters on four matrices per layer), about 8.4 million trainable parameters, 0.12% of 7B.

Choosing: LoRA when memory or the number of task-specific variants matters (one frozen base, many small adapters); QLoRA when the base model does not fit in 16-bit; full fine-tuning when the budget allows it and the task needs changes that a low-rank update may not capture. That last trade-off is the author's judgement; the sources read here report parity on the tasks they tested, not a guarantee.

**Common weak answer.** "LoRA is just a smaller model." The base model is full size and frozen; only the update is small.

**Follow-up.** "How do you choose the rank?" Start small, sweep, and compare on a held-out set; the answer comes from validation, not from theory.

## 8. "How do you align a model with human preferences? RLHF versus DPO."

**What it tests.** The pipeline, and whether the candidate knows the newer simplifications and what they claim.

**Reference answer.** [Ouyang et al. (2022)](https://arxiv.org/abs/2203.02155) describe the pipeline behind InstructGPT: collect labeler demonstrations of desired behaviour and fine-tune GPT-3 with supervised learning; collect rankings of model outputs and use them to fine-tune further with reinforcement learning from human feedback. Their headline result: on their prompt distribution, outputs from the 1.3B-parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite 100 times fewer parameters, with improvements in truthfulness and reductions in toxic output, and minimal regressions on public NLP datasets. The paper's framing: making models bigger does not make them better at following the user's intent.

RLHF in the usual form fits a reward model to the preference data, then optimises the language model against it with reinforcement learning, with a penalty for drifting from the starting model. [Rafailov et al. (2023)](https://arxiv.org/abs/2305.18290) call this complex and often unstable, and propose Direct Preference Optimization: a reparameterisation of the reward in which the optimal policy has a closed form, so the problem reduces to a classification loss on preference pairs, with no sampling from the model during fine-tuning. For a preferred response y_w and a rejected one y_l, the loss is `-log sigmoid(beta * [log(pi(y_w)/pi_ref(y_w)) - log(pi(y_l)/pi_ref(y_l))])`. The paper reports that it matches or improves on PPO-based RLHF on the tasks tested (sentiment control, summarisation, single-turn dialogue). A later paper, [Ahmadian et al. (2024)](https://arxiv.org/abs/2402.14740), argues that much of PPO is unnecessary in this setting and that simpler REINFORCE-style methods outperform PPO and "RL-free" methods such as DPO. The sources therefore disagree about which method is best; a good answer says the comparison depends on the task and the budget.

A numeric check of the DPO loss on one pair, with beta = 0.1 and made-up log-probabilities:

```python
import math
b = 0.1; lw, ll = -12.0, -14.0; rw, rl = -12.5, -13.0    # policy and reference log-probs of chosen / rejected
z = b * ((lw - rw) - (ll - rl)); print(f"margin {z:.3f}, loss {-math.log(1/(1+math.exp(-z))):.4f}; with policy = reference the loss is {-math.log(0.5):.4f}")
```

```
margin 0.150, loss 0.6210; with policy = reference the loss is 0.6931
```

When the policy equals the reference the loss is ln 2 = 0.6931 whatever the data; it drops below that as the policy raises the chosen response's likelihood relative to the rejected one, compared with the reference.

**Common weak answer.** "RLHF makes the model safer." The papers claim preference and truthfulness gains on specific evaluations, not safety in general.

**Follow-up.** "What goes wrong with reward models?" The policy can exploit errors in the learned reward. (The author's note; the DPO paper's complaint about RLHF's instability is the nearest source read.)

## 9. "When would you use retrieval-augmented generation instead of fine-tuning, and what are the failure modes?"

**What it tests.** Choosing between putting knowledge in weights or in the prompt, and seeing retrieval's own failure modes.

**Reference answer.** [Lewis et al. (2020)](https://arxiv.org/abs/2005.11401) motivate RAG by the limits of parametric memory: the ability of language models to access and precisely manipulate knowledge is limited, and providing provenance and updating world knowledge are open problems. They combine a pretrained seq2seq generator with a dense vector index of Wikipedia accessed by a neural retriever, and report state-of-the-art results on three open-domain question answering tasks and more specific, diverse and factual generation than a parametric-only baseline. The introduction adds that with retrieval, knowledge can be directly revised and expanded and the accessed knowledge inspected. Hence the standard answer: use retrieval when knowledge changes often, must be cited, or is too large for the weights; use fine-tuning to change behaviour, format or style. The two combine.

Retrieval has failure modes of its own. If the retriever returns the wrong passages, the generator has nothing correct to use; if it returns many passages, position matters. [Liu et al. (2023)](https://arxiv.org/abs/2307.03172) study multi-document question answering and key-value retrieval and find that performance can degrade significantly when the position of the relevant information changes: it is often highest when the information is at the beginning or end of the context and degrades when it is in the middle, even for models built for long contexts. Practical consequences, as the author's design advice rather than claims from these papers: measure retrieval quality separately from generation quality (recall of the right passage in the top k), keep the number of passages small, order them with the best at the edges, and check that answers are supported by the retrieved text.

**Common weak answer.** "RAG removes hallucination." The RAG paper describes more factual generation, not none, and the generator can still ignore or misread the passages.

**Follow-up.** "How would you evaluate a RAG system?" Build a labelled set of questions with the passages that answer them; report retrieval recall and answer correctness separately, so that a failure can be placed in one stage.

## 10. "What is chain-of-thought prompting, and when does it help?"

**What it tests.** Knowing a prompting technique and its scope.

**Reference answer.** [Wei et al. (2022)](https://arxiv.org/abs/2201.11903) show that giving a few exemplars that include intermediate reasoning steps improves performance on arithmetic, commonsense and symbolic reasoning tasks. With eight exemplars, a 540B-parameter PaLM reached state-of-the-art accuracy on GSM8K, a benchmark of math word problems, surpassing a fine-tuned GPT-3 with a verifier. The scope matters: the paper reports that chain-of-thought prompting is an emergent ability of model scale. It does not positively impact performance for small models and only gives gains with models of about 100B parameters, and smaller models produced fluent but illogical chains of thought, leading to lower performance than standard prompting. These results are for the models and years the paper tested; whether the threshold has moved with newer models is not covered by the source.

The engineering corollaries (the author's): longer outputs cost more tokens and latency, so test whether the reasoning improves accuracy on your own task before shipping it; and intermediate text is not guaranteed to be a faithful explanation of how the answer was produced, which the paper does not claim either.

**Common weak answer.** "Just add 'think step by step' to every prompt." That skips the evaluation and the cost.

**Follow-up.** "How would you decide whether it is worth it?" Compare accuracy, token cost and latency with and without it on a held-out set of real inputs.

## Takeaways

- Attention scores are scaled by 1/sqrt(d_k) because the variance of a dot product grows with d_k; the experiment above shows the softmax going one-hot without it.
- Memory arithmetic wins interviews: 16 bytes per parameter for mixed-precision Adam, and roughly 0.5 MiB of KV cache per token for a 7B-class shape.
- LoRA trains a small update on a frozen model and can be merged with no inference cost; QLoRA adds a 4-bit base.
- Alignment methods and their comparisons are contested; state what each paper measured.
- Retrieval fixes knowledge access, not reasoning; test retrieval and generation separately.

## Limits

- The experiments are toy-sized and CPU-only; the timing ratio in question 6 varies between runs and machines.
- Claims about papers are limited to the sections read: abstracts, the passages quoted, and the figure captions. Results in those papers are for the models, data and years they report; they may not hold for current models.
- Items marked as the author's own advice or reading (the head-count design note in question 2, the prefill/decode note, the choice between adapters and full fine-tuning, the RAG design rules, the chain-of-thought corollaries) have no single source behind them.
- Mixture-of-experts, speculative decoding, quantisation of weights for inference, long-context position schemes and agents are not covered. Distributed training and reinforcement learning are the subject of Part 8.
- The candidates in the examples are hypothetical, and the reference answers are one reasonable answer among several.

Next: Part 8, on advanced topics: reinforcement learning, generative models and distributed training.
