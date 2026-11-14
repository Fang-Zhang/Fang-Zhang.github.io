---
title: "Deep Learning, Domain by Domain, Part 2: Foundations, from Convolution and Recurrence to the Transformer"
date: 2026-12-19 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deep-learning, convolutional-networks, recurrent-networks, transformer, attention, series]
description: "Part 2 of Deep Learning, Domain by Domain. The three network designs every later part builds on (convolutional, recurrent, attention-based), what problem each was built for, the figures their papers report, and a twelve-line attention function that can be run."
---

This post does one thing: it explains the three network designs that the rest of the series relies on, in the order they appeared. For each it states the problem it was built for, what it changed, and the figures its own paper reports. [Part 1](/posts/deep-learning-domain-by-domain-1-a-short-history/) gave the dates; this part gives the mechanisms. It ends with a short program that computes the one formula at the centre of the Transformer.

## The problem: two shapes of data

Most of the data in the later parts has one of two shapes. An image is a grid, where nearby cells are related and the same pattern can appear anywhere. Text, speech and a game's moves are sequences, where order matters and what comes later may depend on something that came much earlier. The three designs below are three answers to how a layer should connect its inputs given those shapes.

## Convolution: use the structure of the grid

In 1980, Fukushima proposed the [neocognitron](https://pubmed.ncbi.nlm.nih.gov/7370364/), a network for visual pattern recognition that is self-organised by "learning without a teacher" and recognises patterns by the similarity of their shapes without being affected by their position. A fuller description of the design came from [LeCun and colleagues in 1998](https://hal.science/hal-03926082/document). They write that convolutional networks combine three architectural ideas: local receptive fields, shared weights, and spatial or temporal subsampling. Each unit looks only at a small neighbourhood of the previous layer, the same weights are reused across positions, and resolution is reduced layer by layer while the number of feature maps grows.

The paper gives a figure that shows what weight sharing buys: its LeNet-5 network has 345,308 connections but only 60,000 trainable free parameters. It adds that, because all weights are learned with backpropagation, such a network can be seen as synthesising its own feature extractor. Part 1 covered the 2012 ImageNet result.

The next step was depth. Kaiming He and colleagues' [residual learning framework](https://arxiv.org/abs/1512.03385) (arXiv, December 2015) starts from the statement that deeper networks are more difficult to train, and reformulates layers as learning residual functions with reference to the layer inputs. Their abstract reports residual networks up to 152 layers deep, 8 times deeper than VGG nets, and a 3.57% error for an ensemble on the ImageNet test set, which won the ILSVRC 2015 classification task.

## Recurrence: carry a state along the sequence

For sequences, the earlier idea was to feed a layer's own output back to itself. [Elman's 1990 report](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1) uses recurrent links so that "hidden unit patterns are fed back to themselves", giving the network a dynamic memory; its simulations range from a temporal version of XOR to discovering syntactic and semantic features of words. The weakness, and the 1997 [LSTM](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) that addressed it, were covered in Part 1: the error signal decays as it flows back through time.

Recurrent networks were then applied to translation. Sutskever, Vinyals and Le's [sequence-to-sequence paper](https://arxiv.org/abs/1409.3215) (September 2014) uses one multilayer LSTM to map the input sentence to a vector of fixed dimensionality and another to decode the target sentence from it. On the WMT'14 English-to-French task their abstract reports a BLEU score of 34.8, against 33.3 for a phrase-based statistical system. The fixed-length vector became the stated problem in the [Bahdanau, Cho and Bengio paper](https://arxiv.org/abs/1409.0473), posted the same month: they conjecture that it is a bottleneck, and let the model "(soft-)search" for the parts of the source sentence that are relevant to each target word. That is an attention mechanism added to a recurrent network.

## Attention: let every position look at every other

The [Transformer paper](https://arxiv.org/abs/1706.03762) (June 2017) removes the recurrence. Its starting point, in the introduction, is that recurrent models compute along the positions of the sequence, which "precludes parallelization within training examples". Its section 3.2 defines an attention function as mapping a query and a set of key-value pairs to an output, where the output is a weighted sum of the values and each weight comes from comparing the query with the corresponding key. The version it uses is scaled dot-product attention: take the dot products of the query with all keys, divide each by the square root of the key dimension, and apply a softmax to get the weights. For matrices of queries Q, keys K and values V:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

In self-attention, "all of the keys, values and queries come from the same place", so each position can attend to every position of the previous layer. The paper's Table 1 compares layer types on two measures, and the figure below redraws the sequential-operation and path-length entries.

![Three diagrams of seven input positions connected to seven output positions: a recurrent layer with a chain of links (sequential operations O(n), maximum path length O(n)), a convolutional layer with links to neighbours only (O(1) and O(log_k n)), and a self-attention layer with every position linked to every other (O(1) and O(1)).]({{ "/assets/pic/dl-domain-2-layer-types.png" | relative_url }})
*Connection patterns of the three layer types for a sequence of n = 7 positions. The complexity entries are those of Table 1 in the Transformer paper; the drawing is the author's.*

The paper's own reading of the table: a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer needs O(n), and self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimension d. The reported results are 28.4 BLEU on WMT 2014 English-to-German, and a single-model 41.8 BLEU on English-to-French after training for 3.5 days on eight GPUs; the introduction adds that a new state of the art in translation quality can be reached after as little as twelve hours on eight P100 GPUs.

The design then left translation. Dosovitskiy and colleagues' [Vision Transformer paper](https://arxiv.org/abs/2010.11929) (October 2020) shows that a pure transformer applied directly to sequences of image patches can perform very well on image classification, and that when pre-trained on large amounts of data it attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train. The grid, which was convolution's home territory, can be handled by the attention design once it is cut into patches.

## Run the formula

The function below is the formula above, and nothing else: no multiple heads, no position information, no masking, no training. It was run in a clean virtual environment with Python 3.14.7 and NumPy 2.5.4.

```python
import numpy as np

def attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                 # compare every query with every key
    w = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)           # softmax: each row sums to 1
    return w @ V, w                                 # weighted sum of the values

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 8))                         # 4 positions, 8 numbers each
out, w = attention(X, X, X)                         # self-attention: Q = K = V = X
np.set_printoptions(precision=2, suppress=True)
print(w)                                            # 4 x 4 weights: every position sees every other
print(w.sum(axis=1))                                # [1. 1. 1. 1.]
print(out.shape)                                    # (4, 8)
```

The output of that run:

```
[[0.6  0.1  0.2  0.1 ]
 [0.02 0.89 0.04 0.05]
 [0.15 0.2  0.56 0.1 ]
 [0.08 0.3  0.11 0.51]]
[1. 1. 1. 1.]
(4, 8)
```

Each row is one position's weights over all four positions, and each row sums to 1. The input here is random numbers, so the weights mean nothing; in a trained network, learned projections produce Q, K and V, and the weights come to reflect which positions are relevant to which.

## What the sources show

These points are the author's reading of the sources above, not claims the sources make.

- **Each design is a choice of what a unit may look at.** Convolution restricts it to a neighbourhood and reuses the weights; recurrence passes a state along one step at a time; attention allows all positions and lets the data decide the weights.
- **Parallelism is the practical difference the Transformer paper stresses.** Its argument against recurrence is about sequential computation, and its results are stated together with training time.
- **The designs overlap in use.** The 2014 attention was added to a recurrent network, and the 2020 vision result splits an image into patches so that a sequence model can read it.

## Limits of this summary

- It covers three designs, not the many variants of each. Multi-head attention, positional encodings and the decoder's masking are in the Transformer paper and are not explained here.
- The Transformer paper was read in full; the neocognitron, residual-network, sequence-to-sequence, attention and Vision Transformer papers were read as abstracts, and the 1998 paper as far as its description of convolutional networks. Figures quoted are the ones those sources state.
- The program is an illustration of the formula on random numbers. It is not code from any of the papers and does not reproduce any of their results.
