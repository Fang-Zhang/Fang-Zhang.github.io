---
title: "The Math Behind LLMs Is Small: Five Operations"
date: 2026-10-23 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, machine-learning, mathematics, transformers, reading-list]
math: true
description: "A large language model is a short list of mathematical operations composed many times: dot product, matrix projection, softmax, weighted averaging, and cross-entropy with gradient descent."
---

The mathematics of LLMs looks large from outside. Papers carry pages of notation, and the models carry billions of parameters. Read the explainers written for newcomers, though, and they converge on the same claim: the list of ideas is short.

Giles Thomas, in [The maths you need to start understanding LLMs](https://www.gilesthomas.com/2025/09/maths-for-llms), puts it bluntly: "if you studied it at high-school at any time since the 1960s, you did all of the groundwork then: vectors, matrices, and so on." Joseph Breeden, in [The Simple Mathematics of Large Language Models](https://www.crc.business-school.ed.ac.uk/sites/crc/files/2026-02/Mathematics_of_LLMs_2026.pdf), a working paper written for readers with statistics and linguistics backgrounds, says the core ideas "are not exotic: conditional probability, weighted averages, and learned linear transformations. The sophistication lies in how these simple pieces are composed."

So the useful question is which pieces, and in what order. Here are five.

![Five operations as boxes: dot product, matrix multiplication, softmax, attention, cross-entropy and gradient descent, with gradient descent feeding back into the weights]({{ "/assets/pic/math-five-operations.png" | relative_url }})
_Figure 1. The five operations and how they connect. Attention combines the first three; training closes the loop._

## One toy model to run all five

Formulas alone do not show where each operation sits, so this post runs one tiny model through all five. The model is mine, not taken from any of the sources, and it is deliberately trivial: a vocabulary of six words (*the, cat, sat, on, mat, dog*), two numbers per word, one attention step, and parameters set by hand rather than trained. Every number below was computed in code from those parameters.

The task is the one an LLM performs billions of times: given **"the cat sat on the"**, assign a probability to every word that could come next. The sentence is a stand-in for any prompt, and the actual next word will be *mat*. Each section below states the problem the operation solves, then shows the arithmetic on this example.

## 1. The dot product: similarity as arithmetic

Multiply two vectors element by element and add the results. That is the whole operation.

$$\mathbf{a}\cdot\mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_d b_d$$

Its meaning: the dot product is large when two vectors point in similar directions and small or negative when they do not. Giles Thomas notes that when the vectors are scaled to length one, it equals the cosine of the angle between them, which is why it is called cosine similarity. Breeden gives the reason it was chosen: it is the simplest measure of association that is cheap to compute and differentiable, so parameters can be learned by gradient-based optimization.

Every later step uses it. Embeddings, where words become points in a high-dimensional space with similar concepts clustered together, only become useful once there is a cheap way to ask how close two points are.

**The problem it solves.** A computer cannot compare the meanings of "cat" and "dog" directly. It can compare two lists of numbers. Once each word is a vector, "how related are these two words?" becomes one multiplication-and-addition. The same question appears in semantic search, in recommendation, and, as section 4 shows, in every attention layer.

**Worked example.** In the toy model "cat" is the vector (0.9, 0.4), "dog" is (0.8, 0.5) and "mat" is (0.3, −0.8):

- cat · dog = 0.9×0.8 + 0.4×0.5 = **0.92**
- cat · mat = 0.9×0.3 + 0.4×(−0.8) = **−0.05**
- cat · the = 0.9×0.1 + 0.4×0.9 = **0.45**

Cat and dog score high because their arrows point almost the same way; cat and mat score near zero because they do not. Dividing by the two vector lengths turns these into cosines: 0.99 for cat and dog, −0.06 for cat and mat. The values themselves are invented. In a trained model, similar words end up close together because training pushes them there; nothing in the toy model learned this.

![Left: six words drawn as arrows in a plane, cat and dog nearly parallel. Right: the dot products of cat with dog, mat and the, worked out]({{ "/assets/pic/math-dot-product.png" | relative_url }})
_Figure 2. Words as arrows, and the dot product as a measure of how closely two arrows align. Invented 2-D numbers, computed in code._

**Why direction decides the size.** The element-by-element formula has an equivalent geometric form:

$$\mathbf{a}\cdot\mathbf{b} = |\mathbf{a}|\,|\mathbf{b}|\cos\theta$$

where θ is the angle between the two vectors. The lengths |a| and |b| are never negative, so the sign and relative size of the result come from cos θ:

- θ near 0° (similar direction): cos θ is near 1, and the product is close to its maximum, |a||b|.
- θ = 90° (perpendicular): cos θ = 0, and the product is 0.
- θ above 90° (pointing away): cos θ is negative, and so is the product. At 180° it reaches its minimum, −|a||b|.

The two forms agree. In two dimensions, let a point at angle α and b at angle β. Then

a·b = |a||b|(cos α cos β + sin α sin β) = |a||b| cos(α − β),

which is the cosine difference identity, and α − β is the angle between them.

Another way to read it: |b| cos θ is the length of b's shadow along a, so a·b is that shadow times |a|. If b leans toward a, the shadow is positive; if b is perpendicular, there is none; if b leans away, the shadow points backwards and the product is negative.

A second small example, separate from the toy model: take a = (1, 0), b = (1, 1) and c = (−1, 0.2). The angle between a and b is 45°, and a·b = **1**. The angle between a and c is about 169°, and a·c = **−1**, because c points almost the opposite way.

One caution: the dot product mixes direction with length. Two long vectors at a modest angle can score higher than two short vectors that point exactly the same way. To compare direction alone, divide by both lengths, which gives cos θ, the cosine similarity used in the worked example above (0.99 for cat and dog).

## 2. Matrix multiplication: moving between spaces

A matrix multiplication projects points from one space into another. In Giles Thomas's example, a matrix of shape 50,257 × 768 projects from GPT-2's 50,257-token vocabulary space down to a 768-dimensional space, and the transpose shape projects back. He adds a caveat worth keeping: projections to fewer dimensions are lossy, so no matrix can recover what was discarded.

A single neural-network layer, ignoring the activation function and bias, is exactly this: a matrix multiplication. The weights are the numbers in the matrix, and "learning" means adjusting them.

**The problem it solves.** One vector is rarely the right view of a word for every job. The word "cat" may need to be looked at one way when it asks "what am I looking for?" and another way when it answers "what do I contain?" A matrix gives the model a learned way to re-express the same vector for each job. Matrices also do the plainest job of all: turning a word into a vector in the first place.

**Worked example, lookup.** Write "cat" as a one-hot vector: a 1 in the cat position and 0 elsewhere. Multiplying it by the 6×2 embedding matrix E picks out exactly one row, (0.9, 0.4). That is how a token ID becomes a vector. During training every number in E is adjusted, as the cross-entropy section notes.

**Worked example, projection.** Take the toy model's query matrix Wq = [[1.0, 0.3], [−0.2, 0.9]]. Multiplying the cat vector by it gives (0.9×1.0 + 0.4×(−0.2), 0.9×0.3 + 0.4×0.9) = **(0.82, 0.63)**. The matrix rotated and stretched the whole plane, and "cat" moved with it. Section 4 uses three such matrices (query, key, value), each giving the same word a different role.

![Left: a one-hot vector times the embedding matrix selects the row for cat. Right: a square grid and the cat arrow before and after being transformed by a matrix]({{ "/assets/pic/math-matrix.png" | relative_url }})
_Figure 3. Two uses of matrix multiplication: selecting a word's vector, and reshaping the space a vector lives in. Toy numbers, computed in code._

Rohit Patel's [Understanding LLMs from Scratch Using Middle School Math](https://towardsdatascience.com/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876/) makes the same point from the other end. He starts from a small leaf-or-flower classifier, where every neuron is a weighted sum of the previous layer, and builds up to Llama 3.1 without introducing anything beyond sums and products.

## 3. Softmax: scores into probabilities

A network's raw outputs are arbitrary real numbers. Softmax converts a vector of them into non-negative numbers that sum to one:

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**The problem it solves.** At the end of the model, the final vector must become a decision among thousands of words. Raw scores cannot be compared across prompts, cannot be negative probabilities, and do not sum to anything useful. Softmax makes them a probability distribution, which is what both training (section 5) and sampling a reply need.

**Worked example.** After the attention step in section 4 produces a vector h = (0.38, 0.65), one more matrix multiplication (h times the 6×2 output matrix) gives six scores, one per vocabulary word: the 1.08, cat 1.02, sat 1.19, on 1.30, mat 0.09, dog 1.21. Softmax exponentiates each and divides by the total, giving **0.17, 0.16, 0.19, 0.21, 0.06, 0.20**, which sum to 1. "On" is now the likeliest next word, and the target word "mat" gets the lowest probability, 0.064. That is expected: this model has not learned anything yet.

A common extra step divides the scores by a number T before softmax; [one explainer](https://www.fanout.sh/blog/softmax-temperature-explained) describes it in one line as dividing every logit by T. In this example, T = 0.5 widens the gap (on 0.25, mat 0.02) and T = 2 flattens it (on 0.19, mat 0.11). That is the knob behind the "temperature" setting in many LLM apps: low values make the output more predictable, high values more varied. This paragraph is my reading of the arithmetic, not a claim from Thomas or Breeden.

![Three bar charts: the six logits, the six probabilities after softmax with mat highlighted, and the probabilities at temperatures 0.5, 1 and 2]({{ "/assets/pic/math-softmax.png" | relative_url }})
_Figure 4. Softmax turns scores into probabilities; dividing by a temperature first sharpens or flattens them. Toy numbers, computed in code._

Giles Thomas points out a consequence: many different "messy" logit vectors map to the same probability distribution, so softmax gives a tidy normalised space. Breeden argues the name hides the purpose. For a statistician it is the inverse multinomial logit, the same function used in multinomial logistic regression, and "softmax" names "a computational property (it is a smooth approximation to the argmax function) rather than the statistical purpose."

## 4. Attention: a learned weighted average

Here the first three pieces combine. Breeden's formulation is the plainest one available: to build a context-aware representation of position *i*, take a weighted average of the vectors at positions *j*,

$$h_i = \sum_j \alpha_{ij}\, v_j, \qquad \alpha_{ij}\ge 0,\quad \sum_j \alpha_{ij}=1$$

where the weights come from dot products between learned projections of the vectors (pieces 1 and 2), scaled by the square root of the projection dimension and passed through softmax (piece 3). The standard compact form, as [Anjali T. writes it](https://medium.com/@anjalitanikella/understanding-the-math-behind-llm-models-and-fine-tuning-them-ee4ca222823f), is $$\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$.

Breeden also objects to the usual vocabulary. Query, key and value are borrowed from databases, but "Nothing is being 'looked up' in the sense of database retrieval." His neutral names are influence scores and influence weights. He also stresses that influence is asymmetric: how much "cat" matters when updating "sat" differs from how much "sat" matters when updating "cat", which is why a plain dot product of the raw vectors is not enough and separate projections are needed.

**The problem it solves.** A word's meaning depends on the words around it. In "the cat sat on the", the final "the" is just an article until the model knows what came before it. Attention lets each position pull in information from earlier positions, in proportion to how relevant they are, instead of treating each word alone.

**Worked example.** Take the last word, "the". Its query vector is (−0.08, 0.84). Dot it with each word's key vector and divide by √2 (the square root of the projection dimension, 2 here): the scores for *the, cat, sat, on, the* are 0.48, −0.03, 0.26, 0.28, 0.48. Softmax turns them into the weights **0.24, 0.14, 0.19, 0.19, 0.24**. Those weights multiply the five value vectors, and the weighted sum is the output h = (0.38, 0.65). In words: the last "the" took roughly a fifth of its new representation from each word before it, and slightly less from "cat". The heatmap shows the same computation for every position at once; the upper triangle is blank because a word may only look at earlier words (and itself) when predicting the next one.

The weights here come from hand-set matrices, so they are nearly even and carry no real linguistic meaning. In a trained model, training would shape the matrices until the weights concentrate on the words that help.

![Left: a lower-triangular heatmap of attention weights for the five words, last row outlined. Right: the five value vectors as arrows with their weighted average h in red]({{ "/assets/pic/math-attention-full.png" | relative_url }})
_Figure 5. One attention step on "the cat sat on the". Left: weights for every position. Right: for the last word, the value vectors and their weighted average. Hand-set toy parameters, computed in code._

Stacking many such layers, with nonlinear transformations, normalisation and position information between them, gives the transformer. Patel's article walks through each of those additions in its own section.

## 5. Cross-entropy and gradient descent: how the numbers get learned

After the final layer, each vocabulary token gets a score, softmax turns the scores into a probability for every possible next token, and training asks one question: how much probability did the model give the token that actually came next?

**The problem it solves.** The parameters start as arbitrary numbers, and nobody writes the right ones by hand. Training needs two things: a single number that says how wrong the model's prediction was, and a rule for changing the parameters to make that number smaller.

The loss is cross-entropy. Breeden shows why it is simpler than it sounds. The "true distribution" is a one-hot vector, because only one word actually occurred, and its entropy is zero. Cross-entropy therefore reduces to $$-\log q(\text{actual word})$$, the negative log-likelihood of the observed word. Minimising it is the same objective as maximum likelihood. The information-theoretic framing adds intuition, as he puts it, "you are measuring how surprised your model is by reality", but the mathematics is identical.

The optimiser is gradient descent. Compute the derivative of the loss with respect to every parameter and move each one slightly in the direction that lowers it. The chain rule makes this feasible through dozens of layers, and software libraries do it automatically. Breeden's verdict: "If anything in LLMs seems like magic, it is that billions of parameters can be estimated simultaneously."

**Worked example.** In the toy model the actual next word is "mat", and the model gave it probability 0.064. The loss is −ln(0.064) = **2.75**. Had the model given "mat" probability 0.9, the loss would be 0.11. Gradient descent then computes how the loss changes with each of the model's 36 numbers (the embeddings, output matrix and three attention matrices) and nudges each against its slope. Here I computed the slopes numerically, by wiggling each number slightly; real libraries obtain them exactly with the chain rule. With a step size of 0.15, after 10 steps the probability of "mat" is 0.59 and the loss 0.52; after 40 steps the probability is 0.99 and the loss 0.014.

One caveat on that result. The toy model was trained on this single sentence, so it memorises it. A real model sees trillions of tokens and cannot memorise them all, so it is pushed to find patterns that generalise. The mechanics, however, are the same: predict, measure the surprise, adjust.

![Three charts: loss falling over 40 steps, probability of mat rising from 0.06 to 0.99, and loss plotted against one parameter with five descent steps marked]({{ "/assets/pic/math-training.png" | relative_url }})
_Figure 6. Training the toy model on one example. Left and middle: loss and the probability of "mat" over 40 gradient-descent steps. Right: loss as a function of one parameter with the others fixed; each step moves downhill. Toy numbers, computed in code._

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
