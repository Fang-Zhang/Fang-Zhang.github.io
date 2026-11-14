---
title: "Deep Learning, Domain by Domain, Part 7: Games, Art and Music"
date: 2026-12-24 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deep-learning, game-ai, generative-models, music-generation, series]
description: "Part 7 of Deep Learning, Domain by Domain. Four game-playing systems, two art papers and four music papers, each with what it reports, and what the three fields share."
---

This post does one thing: it covers deep learning where the result is a game won, a picture made or a piece of music generated, through ten dated papers, and states what each reports. Games here means the systems that reached or passed top human players; the game-playing agents built in the [AI Plays Super Mario series](/posts/ai-plays-super-mario-8-language-model-players/), and what game companies use such agents for ([Mario Part 9](/posts/ai-plays-super-mario-9-what-game-companies-use-ai-players-for/)), are not repeated. Atari and the method behind it are in [Part 5](/posts/deep-learning-domain-by-domain-5-reinforcement-learning-robots-and-driving/), and AlphaGo's 2016 match in [Part 1](/posts/deep-learning-domain-by-domain-1-a-short-history/).

## The field: three kinds of result

A game gives a clear score, which makes it the easiest of the three to measure. Art and music have no such score, and the papers below rely on human judgement and on comparisons with earlier systems. The figure places the ten papers on one time axis, grouped by field.

![A timeline in three lanes. Games: AlphaGo Zero 2017, AlphaZero 2017, OpenAI Five 2019, AlphaStar 2019. Art: Gatys et al. 2015, DALL-E 2021. Music: WaveNet 2016, Jukebox 2020, MusicLM and MusicGen 2023. Each box carries one claim from its source.]({{ "/assets/pic/dl-domain-7-games-art-music.png" | relative_url }})
*Games, art and music, with the claim each source's abstract makes. Lane layout and selection are the author's.*

## Games

**AlphaGo Zero (2017).** The [Nature paper](https://www.nature.com/articles/nature24270) starts from AlphaGo, whose networks were trained by supervised learning from human expert moves and by reinforcement learning from self-play. AlphaGo Zero uses reinforcement learning alone, without human data, guidance or domain knowledge beyond the rules. A single network is trained to predict the program's own move selections and the winner of its games, and it improves the tree search that produces the next round of games. The abstract reports a 100-0 win against the previously published, champion-defeating AlphaGo.

**AlphaZero (2017).** Silver and colleagues' [AlphaZero](https://arxiv.org/abs/1712.01815) generalises that approach into one algorithm. The abstract says the strongest chess programs rely on search techniques, domain-specific adaptations and handcrafted evaluation functions refined by human experts over decades. Starting from random play and given only the rules, AlphaZero reached a superhuman level in chess, shogi and Go within 24 hours, and convincingly defeated a world-champion program in each.

**OpenAI Five (2019).** The [OpenAI Five report](https://arxiv.org/abs/1912.06680) states that on April 13, 2019, the system became the first AI to defeat the world champions at an esports game, Dota 2, beating Team OG. It names the difficulties: long time horizons, imperfect information, and complex continuous state-action spaces. The method was existing reinforcement learning, scaled to batches of about 2 million frames every 2 seconds, with a distributed training system that allowed training for 10 months.

**AlphaStar (2019).** The [AlphaStar paper in Nature](https://www.nature.com/articles/s41586-019-1724-z) addresses StarCraft II. It says the strongest earlier agents had simplified aspects of the game, used superhuman capabilities or relied on hand-crafted sub-systems. AlphaStar uses a multi-agent reinforcement learning algorithm with data from both human and agent games, within a league of continually adapting strategies and counter-strategies. In online games against human players it was rated at Grandmaster level for all three races, above 99.8% of officially ranked players.

## Art

Gatys, Ecker and Bethge's [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (August 2015) opens with the observation that in painting, humans compose a "complex interplay between the content and style of an image", and that no artificial system had comparable capabilities. Their system uses the representations of a deep convolutional network to separate and recombine the content and style of arbitrary images.

Ramesh and colleagues' [DALL-E paper](https://arxiv.org/abs/2102.12092) (February 2021) takes the problem of text-to-image generation, which they say had focused on better modelling assumptions for a fixed dataset, and describes a simple alternative: a transformer that autoregressively models the text and image tokens as a single stream. With sufficient data and scale it is competitive with previous domain-specific models in a zero-shot evaluation. Diffusion-based image generation is in [Part 3](/posts/deep-learning-domain-by-domain-3-images-and-video/).

## Music

**WaveNet (2016).** Van den Oord and colleagues' [WaveNet](https://arxiv.org/abs/1609.03499) generates raw audio waveforms: a fully probabilistic, autoregressive model in which each audio sample is conditioned on all previous ones, trained on data with tens of thousands of samples per second. Applied to text-to-speech, human listeners rated it significantly more natural than the best parametric and concatenative systems for English and Mandarin. Trained on music, it generates "novel and often highly realistic musical fragments".

**Jukebox (2020).** Dhariwal and colleagues' [Jukebox](https://arxiv.org/abs/2005.00341) generates music with singing in the raw audio domain. To handle the long context of raw audio it compresses it to discrete codes with a multi-scale VQ-VAE and models those codes with autoregressive Transformers. The abstract reports songs coherent over multiple minutes, steerable by artist, genre and unaligned lyrics, and says the samples released were not cherry-picked.

**MusicLM and MusicGen (2023).** Agostinelli and colleagues' [MusicLM](https://arxiv.org/abs/2301.11325) generates 24 kHz music from text descriptions, treating generation as hierarchical sequence-to-sequence modelling, and stays consistent over several minutes. The authors report that it outperforms previous systems in audio quality and adherence to the text, and they release MusicCaps, 5.5 thousand music-text pairs described by human experts. Copet and colleagues' [MusicGen](https://arxiv.org/abs/2306.05284) uses a single-stage transformer over several streams of compressed discrete tokens, with token interleaving patterns that remove the need to cascade several models, and reports results superior to the evaluated baselines on a standard text-to-music benchmark.

## What the sources show

These points are the author's reading of the sources above, not claims the sources make.

- **Games became less dependent on human data, with an exception.** AlphaGo Zero and AlphaZero use none beyond the rules, while AlphaStar's abstract states that it uses data from both human and agent games.
- **Art and music converged on tokens and autoregressive models.** DALL-E models text and image tokens as one stream, Jukebox models discrete audio codes with Transformers, and MusicGen models music tokens with one Transformer. This is the language-model recipe of [Part 4](/posts/deep-learning-domain-by-domain-4-language-and-speech/) applied to other signals.
- **Compute is stated as part of the result in games.** OpenAI Five reports 10 months of training and AlphaZero reports 24 hours, so the claim comes with its cost.
- **Success is easy to state only in games.** The game papers give a win rate or a rank, and the art and music papers give descriptions and human or benchmark comparisons.

## Limits of this summary

- It covers ten papers. Procedural content generation, game design tools, music transcription and recommendation, voice cloning, and questions of copyright and credit for generated work are left out.
- Every paper was read as an abstract page, except the Gatys paper, whose abstract was read from the PDF's first page. The figures are those the sources state.
- Match results and rankings are from the dates of the papers, and the systems have not been re-tested here.
