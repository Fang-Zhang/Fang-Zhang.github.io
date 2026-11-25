---
title: "Reinforcement Learning Engineering (4/6): Three of Five Famous Applications Carry a Qualifier the Headline Leaves Out"
date: 2027-06-27 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [reinforcement learning, applications, alphago, alphazero, alphatensor, chip design, series]
description: "Part 4 of 6 on reinforcement learning engineering. Board games, data-centre cooling, chip floorplanning, plasma control and matrix multiplication, each checked against the original source and the later work that tested it."
---

## What was checked

Five results are cited again and again as proof that reinforcement learning works outside benchmarks. This post reads each against its original source and against any later work that tested it. No experiment is run here: the question is what each source claims, what method sits behind the claim, and what a later source adds. The sources are public; how much of each was read is listed at the end.

![A table with five rows (board games, data-centre cooling, chip floorplanning, plasma control, matrix multiplication) and three columns: the claim in the source, where reinforcement learning sits in the system, and a qualification found in the sources]({{ site.baseurl }}/assets/pic/rl-eng-4-applications-claims.png){: width="900" }
_Figure: the five cases, summarised from the sources linked below. The last column of the first row is this series' own reading, not a claim from a source._

## Board games: the best-documented case

- **AlphaGo (2016).** The [Nature paper](https://www.nature.com/articles/nature16961) trains policy and value networks "by a novel combination of supervised learning from human expert games, and reinforcement learning from games of self-play", combines them with Monte Carlo search, and reports a 99.8 percent winning rate against other Go programs and a 5 to 0 win against the European Go champion.
- **AlphaZero (2017).** The [follow-up](https://arxiv.org/abs/1712.01815) drops the human data. Starting from random play and given no domain knowledge except the game rules, it "achieved within 24 hours a superhuman level of play" in chess, shogi and Go, and "convincingly defeated a world-champion program in each case".
- **Reading.** Rules are known and games can be simulated without limit and without risk. None of the other four cases has all three properties, which is why the later rows need more engineering around the learner.

## Data-centre cooling: two numbers, two systems

- **2016 blog.** [DeepMind reported](https://deepmind.google/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/) that the system "was able to consistently achieve a 40 percent reduction in the amount of energy used for cooling", which "equates to a 15 percent reduction in overall PUE overhead", at one data centre. The method described there is an ensemble of deep neural networks trained on historical sensor data to predict future PUE, plus two more ensembles predicting temperature and pressure over the next hour, used to check recommended actions against operating constraints. The post does not describe it as reinforcement learning.
- **2022 paper.** [Luo et al.](https://arxiv.org/abs/2211.07357) describe live experiments on two commercial cooling facilities with a reinforcement learning system, report "energy savings of approximately 9% and 13% respectively at the two live experiment sites", and name the challenges: "evaluation, learning from offline data, and constraint satisfaction".
- **Qualifier.** The 40 percent figure and the 9 and 13 percent figures come from different systems, sites and measurement conditions. Quoting one as the result of the other mixes a prediction-model system with a reinforcement learning one.

## Chip floorplanning: a result still contested

- **Claim.** The [Nature paper](https://www.nature.com/articles/s41586-021-03544-w) reports that "in under six hours" the method generates floorplans "superior or comparable to those produced by humans in all key metrics, including power consumption, performance and chip area", and that it was used to design the next generation of Google's AI accelerators.
- **Independent assessment.** [Cheng et al.](https://arxiv.org/abs/2302.11014) write that evaluation "has been hampered because neither data nor code in these works is, to date, fully available" (version 2), that inability to reproduce the methods and results "has led to controversy", and, in the latest version's abstract, that a stronger simulated-annealing baseline was used and that "still-missing confirmations (e.g., of CT's scalability and pre-training methodology) remain open questions". The paper was published in IEEE Transactions on CAD.
- **Response.** A [2024 Nature addendum](https://www.nature.com/articles/s41586-024-08032-5) restates that "in 2020, we introduced a deep reinforcement learning method capable of generating superhuman chip layouts" and names the method AlphaChip. The addendum text read here is its opening lines and reference list; its full argument was not read.
- **Qualifier.** The strength of the baseline decides how large the gain looks, and the open questions are about reproduction and scale, not whether the method runs.

## Plasma control: sim-to-real in one paragraph

[Degrave et al.](https://www.nature.com/articles/s41586-021-04301-9) report a learned controller commanding the full set of magnetic coils of the TCV tokamak, producing elongated, negative-triangularity and snowflake shapes, and sustained "droplets" with two plasmas at once. The text states that the policies are "learned through interaction with a tokamak simulator and are shown to be directly capable of tokamak magnetic control on hardware", and the controller runs as compiled code at a 10 kHz control rate. The abstract reports the demonstrations on that one device, and it stresses constraints: the architecture meets high-level objectives "at the same time satisfying physical and operational constraints".

## Matrix multiplication: a verifiable reward

- **Claim.** [AlphaTensor](https://www.nature.com/articles/s41586-022-05172-4) plays a single-player game whose goal is a tensor decomposition, and is built on AlphaZero. It finds a 4 by 4 algorithm with 47 multiplications in modular arithmetic against 49 for two-level Strassen, and a rank-76 decomposition for 4 by 5 times 5 by 5 in standard arithmetic against the previous 80. The paper stresses that the algorithms are provably correct.
- **Qualifier 1.** The 47 holds in a finite field. A [commentary on the Computational Complexity blog](https://blog.computationalcomplexity.org/2022/10/alpha-tensor.html) points out that it does not work for general fields such as the real numbers and that press coverage often blurred this.
- **Qualifier 2.** In December 2022, [Kauers and Moosbauer](https://arxiv.org/abs/2212.01175) introduced a random-walk method on a "flip graph" that reduced the number of multiplications for formats (4,4,5) and (5,5,5), both in characteristic two and for arbitrary ground fields. The abstract does not use reinforcement learning.

## Engineering reading

These points are this series' reading of the sources above, not claims from them.

1. **A cheap, safe place to practise is the common ingredient.** Games, a tokamak simulator and a tensor game can all be run without consequences. Cooling and chip layout needed historical data and a careful evaluation protocol instead; the cooling paper lists evaluation, offline data and constraints as the hard parts, which are the topics of the previous post.
2. **A checkable score helps.** The matrix algorithms are verified exactly, and a game ends with a win or a loss. Where the score is a proxy or a long-horizon measurement (cooling energy, post-route chip metrics), the dispute moves to how it is measured.
3. **Baselines carry the claim.** In the chip case the disagreement is largely about how strong the non-learning baseline was. In the matrix case a non-learning search later improved the same formats.
4. **Quote the number with its setting.** The same headline can refer to a prediction system or a reinforcement learning system (cooling), to one field or all fields (matrices), or to a result under reproduction (chips).

## Limits and how much was read

- Read beyond the abstract: the 2016 cooling blog, the relevant parts of the plasma paper's Nature page, and the Cheng et al. version 2 introduction. Abstracts only (plus extended-data captions for AlphaTensor): the Luo et al., AlphaZero, AlphaTensor and flip-graph papers. For AlphaGo and the chip paper only the abstracts were read, because the full texts are behind a subscription; the AlphaGo figures above are those in the abstract.
- No result was reproduced. The 2016 cooling system is described only by its blog, and this post does not judge whether it was reinforcement learning in a broader sense.
- The five cases were chosen because they are widely cited. They are not a sample of all applications, and failures are underrepresented in published sources.
- The chip dispute has several parties and years of updates; this post cites only the Nature paper, one independent assessment and the Nature addendum.

Next: Part 5 covers applications that run in the physical world or at service scale: locomotion and manipulation, sim-to-real transfer, recommenders and resource scheduling, with the focus on engineering instead of algorithms.

## References

- [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961), Nature, 2016.
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), 2017.
- [DeepMind AI Reduces Google Data Centre Cooling Bill by 40%](https://deepmind.google/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/), 2016.
- [Controlling Commercial Cooling Systems Using Reinforcement Learning](https://arxiv.org/abs/2211.07357), 2022.
- [A graph placement methodology for fast chip design](https://www.nature.com/articles/s41586-021-03544-w), Nature, 2021, and its [addendum](https://www.nature.com/articles/s41586-024-08032-5), 2024.
- [An Updated Assessment of Reinforcement Learning for Macro Placement](https://arxiv.org/abs/2302.11014), 2023 to 2026.
- [Magnetic control of tokamak plasmas through deep reinforcement learning](https://www.nature.com/articles/s41586-021-04301-9), Nature, 2022.
- [Discovering faster matrix multiplication algorithms with reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4), Nature, 2022.
- [Flip Graphs for Matrix Multiplication](https://arxiv.org/abs/2212.01175), 2022.
- [Alpha Tensor](https://blog.computationalcomplexity.org/2022/10/alpha-tensor.html), Computational Complexity blog, 2022.
