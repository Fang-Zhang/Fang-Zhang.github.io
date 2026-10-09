---
title: "The Math Behind LLMs Is Small: Five Operations"
date: 2026-10-23 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, machine-learning, mathematics, transformers, reading-list]
description: "A large language model is a short list of mathematical operations composed many times: dot product, matrix projection, softmax, weighted averaging, and cross-entropy with gradient descent."
---

The mathematics of LLMs looks large from outside. Papers carry pages of notation, and the models carry billions of parameters. Read the explainers written for newcomers, though, and they converge on the same claim: the list of ideas is short.

Giles Thomas, in [The maths you need to start understanding LLMs](https://www.gilesthomas.com/2025/09/maths-for-llms), puts it bluntly: "if you studied it at high-school at any time since the 1960s, you did all of the groundwork then: vectors, matrices, and so on." Joseph Breeden, in [The Simple Mathematics of Large Language Models](https://www.crc.business-school.ed.ac.uk/sites/crc/files/2026-02/Mathematics_of_LLMs_2026.pdf), a working paper written for readers with statistics and linguistics backgrounds, says the core ideas "are not exotic: conditional probability, weighted averages, and learned linear transformations. The sophistication lies in how these simple pieces are composed."

So the useful question is which pieces, and in what order. Here are five.

## 1. The dot product: similarity as arithmetic

Multiply two vectors element by element and add the results. That is the whole operation.

```
a · b = a1·b1 + a2·b2 + ... + ad·bd
```

Its meaning: the dot product is large when two vectors point in similar directions and small or negative when they do not. Giles Thomas notes that when the vectors are scaled to length one, it equals the cosine of the angle between them, which is why it is called cosine similarity. Breeden gives the reason it was chosen: it is the simplest measure of association that is cheap to compute and differentiable, so parameters can be learned by gradient-based optimization.

Every later step uses it. Embeddings, where words become points in a high-dimensional space with similar concepts clustered together, only become useful once there is a cheap way to ask how close two points are.

## 2. Matrix multiplication: moving between spaces

A matrix multiplication projects points from one space into another. In Giles Thomas's example, a matrix of shape 50,257 × 768 projects from GPT-2's 50,257-token vocabulary space down to a 768-dimensional space, and the transpose shape projects back. He adds a caveat worth keeping: projections to fewer dimensions are lossy, so no matrix can recover what was discarded.

A single neural-network layer, ignoring the activation function and bias, is exactly this: a matrix multiplication. The weights are the numbers in the matrix, and "learning" means adjusting them.

Rohit Patel's [Understanding LLMs from Scratch Using Middle School Math](https://towardsdatascience.com/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876/) makes the same point from the other end. He starts from a small leaf-or-flower classifier, where every neuron is a weighted sum of the previous layer, and builds up to Llama 3.1 without introducing anything beyond sums and products.

## 3. Softmax: scores into probabilities

A network's raw outputs are arbitrary real numbers. Softmax converts a vector of them into non-negative numbers that sum to one:

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Giles Thomas points out a consequence: many different "messy" logit vectors map to the same probability distribution, so softmax gives a tidy normalised space. Breeden argues the name hides the purpose. For a statistician it is the inverse multinomial logit, the same function used in multinomial logistic regression, and "softmax" names "a computational property (it is a smooth approximation to the argmax function) rather than the statistical purpose."

## 4. Attention: a learned weighted average

Here the first three pieces combine. Breeden's formulation is the plainest one available: to build a context-aware representation of position *i*, take a weighted average of the vectors at positions *j*,

```
h_i = sum_j  alpha_ij · v_j        (alpha_ij >= 0, sum_j alpha_ij = 1)
```

where the weights come from dot products between learned projections of the vectors (pieces 1 and 2), scaled by the square root of the projection dimension and passed through softmax (piece 3). The standard compact form, as [Anjali T. writes it](https://medium.com/@anjalitanikella/understanding-the-math-behind-llm-models-and-fine-tuning-them-ee4ca222823f), is `softmax(QKᵀ/√d_k)V`.

Breeden also objects to the usual vocabulary. Query, key and value are borrowed from databases, but "Nothing is being 'looked up' in the sense of database retrieval." His neutral names are influence scores and influence weights. He also stresses that influence is asymmetric: how much "cat" matters when updating "sat" differs from how much "sat" matters when updating "cat", which is why a plain dot product of the raw vectors is not enough and separate projections are needed.

Stacking many such layers, with nonlinear transformations, normalisation and position information between them, gives the transformer. Patel's article walks through each of those additions in its own section.

## 5. Cross-entropy and gradient descent: how the numbers get learned

After the final layer, each vocabulary token gets a score, softmax turns the scores into a probability for every possible next token, and training asks one question: how much probability did the model give the token that actually came next?

The loss is cross-entropy. Breeden shows why it is simpler than it sounds. The "true distribution" is a one-hot vector, because only one word actually occurred, and its entropy is zero. Cross-entropy therefore reduces to `−log q(actual word)`, the negative log-likelihood of the observed word. Minimising it is the same objective as maximum likelihood. The information-theoretic framing adds intuition, as he puts it, "you are measuring how surprised your model is by reality", but the mathematics is identical.

The optimiser is gradient descent. Compute the derivative of the loss with respect to every parameter and move each one slightly in the direction that lowers it. The chain rule makes this feasible through dozens of layers, and software libraries do it automatically. Breeden's verdict: "If anything in LLMs seems like magic, it is that billions of parameters can be estimated simultaneously."

Patel adds a detail that explains where embeddings come from. The numbers that represent each token start out arbitrary, so gradient descent is applied to them too, not only to the weights. An embedding is a trained input.

Two further stages sit on top. Breeden lists supervised fine-tuning and reinforcement learning from human feedback after pre-training, and the Medium article above covers the policy-gradient method [PPO](https://arxiv.org/abs/1707.06347) used in that last stage. They change the objective, not the toolkit: the same derivatives, a different reward.

## What this list leaves out

Giles Thomas scopes his post to inference, using a trained model, and says training is "also not much beyond high-school maths" but defers it to later posts. Position encodings, normalisation layers, mixture-of-experts routing and scaling laws each add their own mathematics, and the sources above treat them in separate sections. The five operations are the spine. The rest are refinements that make it train faster and generalise better.

## Where to read, by depth

- **Shortest path:** [Giles Thomas](https://www.gilesthomas.com/2025/09/maths-for-llms), one post covering vectors, embeddings, the dot product and matrices as projections, with a direct line to how an LLM works in his following post.
- **From zero to a full transformer:** [Rohit Patel](https://towardsdatascience.com/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876/), a 43-minute read, built on addition and multiplication alone.
- **For statisticians and linguists:** [Joseph Breeden](https://www.crc.business-school.ed.ac.uk/sites/crc/files/2026-02/Mathematics_of_LLMs_2026.pdf), a PDF that renames the jargon into standard statistical terms and shows where each piece comes from.
- **As a reference text:** [The Little Book of Maths for LLMs](https://little-book-of.github.io/maths-for-llms/books/en-US/book.html), about a hundred short sections from numbers and algebra through calculus, probability, geometry, optimization and information theory. Its opening lines compress the whole subject: "Every prediction is a probability. Every representation is a vector. Every update is a derivative. Every layer is a transformation."
- **Underlying mathematics in book form:** freeCodeCamp's [The Math Behind Artificial Intelligence](https://www.freecodecamp.org/news/the-math-behind-artificial-intelligence-book/) devotes chapters to linear algebra, multivariable calculus, probability and statistics, and optimization theory. It describes itself as "not a math book filled with complex formulas."
- **A book in progress:** [Understanding LLMs – A Mathematical Approach](https://actionbridge.io/en-US/llmtutorial/p/llm-mathematics-introduction), whose introduction argues the foundations will outlast any architecture: "These requirements are mathematical, not architectural."

## Small pieces, composed

Each of the five operations fits on one line. Nothing in the list needs more than a school-level grasp of vectors, sums, derivatives and probabilities. What looks like intelligence is the composition: dot products inside weighted averages, inside layers, inside a loss, repeated across billions of parameters and trillions of tokens.

Understanding a system of that kind means finding the primitives, and checking that each one is simple enough to hold in your head. After that, the size of the system is only repetition.
