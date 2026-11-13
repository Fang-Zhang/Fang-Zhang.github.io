---
title: "Deep Learning, Domain by Domain, Part 1: A Short History, 1943 to 2024"
date: 2027-01-21 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deep-learning, neural-networks, history, series]
description: "Part 1 of Deep Learning, Domain by Domain. A chronological history of deep learning in fourteen dated steps, from the first threshold-unit networks in 1943 to the 2024 Nobel Prize in Physics, each tied to the paper or official page that states it."
---

This post does one thing: it sets out, in date order, how deep learning got from a logical model of neurons to the large pretrained models of today. Each step is tied to the paper or official page that states it, and the figures quoted are the ones those sources give. The series that follows takes the field one domain at a time (images, language, games, medicine and so on); this first post is the common history they all draw on.

The nine parts of the series:

1. A short history (this post)
2. [Foundations: from convolutional and recurrent networks to the Transformer](/posts/deep-learning-domain-by-domain-2-foundations-from-convolution-and-recurrence-to-the-transformer/)
3. [Images and video](/posts/deep-learning-domain-by-domain-3-images-and-video/)
4. [Language and speech](/posts/deep-learning-domain-by-domain-4-language-and-speech/)
5. [Reinforcement learning, robotics and self-driving](/posts/deep-learning-domain-by-domain-5-reinforcement-learning-robots-and-driving/)
6. [Recommendation and e-commerce, with fraud, security and forecasting](/posts/deep-learning-domain-by-domain-6-recommendation-fraud-security-and-forecasting/)
7. [Games, art and music](/posts/deep-learning-domain-by-domain-7-games-art-and-music/)
8. [Medicine, biomedicine and drug discovery](/posts/deep-learning-domain-by-domain-8-medicine-biology-and-drug-discovery/)
9. [Business impact: cost, evaluation, risk and when not to use deep learning](/posts/deep-learning-domain-by-domain-9-business-impact-cost-evaluation-and-risk/)

## What deep learning is

The 2015 review by LeCun, Bengio and Hinton gives the definition most later work uses: [deep learning](https://pubmed.ncbi.nlm.nih.gov/26017442) lets computational models "composed of multiple processing layers" learn representations of data "with multiple levels of abstraction", and it uses the backpropagation algorithm to indicate how each layer's internal parameters should change. The history below is the history of those three pieces: a unit that computes something simple, layers of such units, and a way to adjust all the layers from the error at the output.

![A horizontal timeline with fourteen dated events from 1943 to 2024, grouped into three eras: ideas (1943 to 1969), learning methods (1986 to 2006), and data, GPUs and scale (2009 to 2024).]({{ "/assets/pic/dl-domain-1-timeline.png" | relative_url }})
*The fourteen steps covered in this post, in date order. The selection is the author's; it is not a complete list of contributors.*

## 1943 to 1969: units, and the limits of one layer

**1943.** McCulloch and Pitts show in [A logical calculus of the ideas immanent in nervous activity](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf) that, because nervous activity is "all-or-none", neural events and their relations can be treated by propositional logic. Their abstract states that the behaviour of every such net can be described in these terms, and that for any logical expression meeting certain conditions a net can be found that behaves as it describes. This is a model of threshold units, not a learning method.

**1958.** Rosenblatt's [perceptron](https://pubmed.ncbi.nlm.nih.gov/13602029) (*Psychological Review* 65(6), 386 to 408) is, in the title's words, a probabilistic model for information storage and organisation in the brain.

**1969.** Minsky and Papert publish [*Perceptrons*](https://mitpress.mit.edu/9780262631112/perceptrons/) (first edition 1969, per the [book's publication history](https://en.wikipedia.org/wiki/Perceptrons_(book))). The MIT Press description calls it the first mathematical analysis carried far enough to show the exact limitations of a class of computing machines that could seriously be considered as models of the brain. Its subject is the perceptron model of 1958.

## 1986 to 2006: learning in several layers

**1986.** Rumelhart, Hinton and Williams describe [back-propagation](https://www.nature.com/articles/323533a0) in *Nature*: a procedure that repeatedly adjusts the weights of a network so as to minimise the difference between the actual output vector and the desired one. Their abstract says that as a result, internal "hidden" units come to represent important features of the task, and that this ability to create new features is what distinguishes back-propagation from earlier, simpler methods such as the perceptron-convergence procedure.

**1997.** Hochreiter and Schmidhuber's [long short-term memory](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) (LSTM) addresses a specific failure of recurrent networks trained by backpropagation: the error signal decays as it flows back through time, so learning to store information over long intervals takes very long. Their abstract reports that LSTM can bridge time lags in excess of 1000 discrete time steps.

**1998.** LeCun, Bottou, Bengio and Haffner publish [Gradient-based learning applied to document recognition](https://hal.science/hal-03926082/document). The paper's stated main message is that better pattern-recognition systems can be built by relying more on automatic learning and less on hand-designed heuristics, shown on character recognition by replacing hand-crafted feature extraction with learning machines that operate directly on pixel images.

**2006.** Hinton, Osindero and Teh's [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf) gives a greedy algorithm that learns a deep, directed belief network one layer at a time. Their abstract reports that after fine-tuning, a network with three hidden layers forms a very good generative model of handwritten digits and gives better digit classification than the best discriminative learning algorithms they compared against.

## 2009 to 2017: data, GPUs and a new architecture

**2009.** Deng and colleagues introduce [ImageNet](https://ieeexplore.ieee.org/document/5206848), a database that aims to populate most of the 80,000 WordNet synsets with an average of 500 to 1000 clean, full-resolution images each. A yearly benchmark built on it, the [ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/abs/1409.0575), has run since 2010.

**2012.** Krizhevsky, Sutskever and Hinton's [ImageNet classification with deep convolutional neural networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) trains a network with five convolutional layers and 60 million parameters. A variant entered in ILSVRC-2012 won with a top-5 test error of 15.3%, against 26.2% for the second-best entry. The paper states that training took five to six days on two GTX 580 3GB GPUs, and that its results "can be improved simply by waiting for faster GPUs and bigger datasets to become available".

**2016.** DeepMind's [AlphaGo](https://www.nature.com/articles/nature16961) combines policy and value networks, trained by supervised learning from expert games and by self-play reinforcement learning, with tree search. The *Nature* abstract reports a 99.8% winning rate against other Go programs and a 5 to 0 win over the European Go champion, a feat the authors describe as previously thought to be at least a decade away.

**2017.** Vaswani and colleagues propose the [Transformer](https://arxiv.org/abs/1706.03762), a network based solely on attention mechanisms that dispenses with recurrence and convolutions. They report 28.4 BLEU on the WMT 2014 English-to-German translation task, with more parallelisable computation and less training time than the recurrent and convolutional models it was compared with.

## 2018 onward: pretraining and scale

**2018.** [BERT](https://arxiv.org/abs/1810.04805) pre-trains deep bidirectional representations from unlabelled text and is then fine-tuned with one additional output layer. The paper reports new state-of-the-art results on eleven natural-language tasks, including a GLUE score of 80.5%. In the same year Bengio, Hinton and LeCun receive the [ACM A.M. Turing Award](https://amturing.acm.org/award_winners/hinton_4791679.cfm), with the citation "for conceptual and engineering breakthroughs that have made deep neural networks a critical component of computing".

**2020.** Kaplan and colleagues report [empirical scaling laws](https://arxiv.org/abs/2001.08361): language-model loss scales as a power law with model size, dataset size and training compute, with some trends spanning more than seven orders of magnitude. Months later, [GPT-3](https://arxiv.org/abs/2005.14165) is a 175-billion-parameter language model, ten times larger than any previous non-sparse language model according to its abstract, tested on tasks specified purely through text and a few examples, with no gradient updates.

**2024.** The Royal Swedish Academy of Sciences awards the [Nobel Prize in Physics](https://www.nobelprize.org/prizes/physics/2024/press-release/) to John Hopfield and Geoffrey Hinton "for foundational discoveries and inventions that enable machine learning with artificial neural networks".

## What the dates show

Three patterns are visible in the sources above, and they are the author's reading of them rather than claims the sources make.

- **The means came after the ideas.** Layered networks with hidden units are described in 1986; the 2012 ImageNet result comes with a large labelled dataset (2009) and GPU training, and the AlexNet paper itself says its results can be improved by faster GPUs and bigger datasets.
- **The architecture changed while the training idea stayed.** The 1998 paper replaces hand-crafted features, 2017 replaces recurrence with attention, and the 2015 definition quoted at the top still names backpropagation as the way each layer is adjusted.
- **After 2017 the main lever changed from architecture to scale.** The 2020 papers measure what happens when model size, data and compute grow together.

## Limits of this summary

- It follows one thread: the line of work that leads to the networks used in the later parts. Many contributors and parallel lines are left out, and this post does not assign credit for individual ideas.
- It rests on the papers' own abstracts and introductions, and on the publishers' and awarding bodies' pages. Where a number appears it is the number the source states; none was reproduced here.
- It stops at 2024 because that is the last step this post checked against an official page. Parts 2 to 9 cover later developments in each domain, each with its own sources.
