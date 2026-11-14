---
title: "Deep Learning, Domain by Domain, Part 4: Language and Speech"
date: 2026-12-21 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deep-learning, nlp, speech-recognition, machine-translation, language-models, series]
description: "Part 4 of Deep Learning, Domain by Domain. Eight dated steps in text and speech, from word vectors in 2013 to Whisper in 2022, each tied to the paper that reports it."
---

This post does one thing: it follows deep learning in text and speech through eight dated papers, from word vectors in 2013 to a speech model trained on 680,000 hours of audio in 2022, and states what each paper reports. The architectures (recurrence, attention, the Transformer) were introduced in [Part 2](/posts/deep-learning-domain-by-domain-2-foundations-from-convolution-and-recurrence-to-the-transformer/); the images side of the same period is in [Part 3](/posts/deep-learning-domain-by-domain-3-images-and-video/).

## The field: sequences of symbols and of sound

Text is a sequence of discrete symbols, and speech is a sampled waveform. The work below splits along that line, with the two meeting in the last step: a model that transcribes speech by treating it as text prediction.

![A timeline from 2013 to 2022. Above the axis, text: word2vec 2013, GNMT 2016, BERT 2018, RAG 2020, InstructGPT 2022. Below the axis, speech: Deep Speech 2014, wav2vec 2.0 2020, Whisper 2022. Each box carries one figure reported by its source.]({{ "/assets/pic/dl-domain-4-language-speech.png" | relative_url }})
*Eight papers on one axis, each with a figure its own source reports. The figures come from different benchmarks and are not comparable. The timeline is the author's.*

## Text

**Word vectors (2013).** Mikolov and colleagues' [word2vec](https://arxiv.org/abs/1301.3781) proposes two architectures for computing continuous vector representations of words from very large data sets. The abstract reports that it takes less than a day to learn high quality word vectors from a 1.6 billion word data set, with large accuracy improvements at much lower computational cost than earlier neural approaches. A follow-up, [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546), adds subsampling of frequent words and negative sampling, and states an inherent limit of word representations: their indifference to word order and their inability to represent idiomatic phrases, as in "Air Canada".

**Translation (2016).** Wu and colleagues' [Google's Neural Machine Translation system](https://arxiv.org/abs/1609.08144) is a deep LSTM network with 8 encoder and 8 decoder layers, using attention and residual connections. It divides words into sub-word units called wordpieces to handle rare words, and uses low-precision arithmetic at inference for speed. On WMT'14 English-to-French and English-to-German it reaches results competitive with the state of the art, and in a human side-by-side evaluation on isolated simple sentences it reduces translation errors by an average of 60% compared with Google's phrase-based production system.

**Pre-training (2018).** Devlin and colleagues' [BERT](https://arxiv.org/abs/1810.04805) pre-trains deep bidirectional representations from unlabeled text by conditioning on left and right context in all layers. With one added output layer it is fine-tuned for tasks such as question answering and language inference. The abstract reports new best results on eleven tasks, including a GLUE score of 80.5%, a 7.7 point absolute improvement.

**Retrieval (2020).** Lewis and colleagues' [retrieval-augmented generation](https://arxiv.org/abs/2005.11401) starts from a stated limit: pre-trained language models store factual knowledge in their parameters, but their ability to access and manipulate it precisely is limited, and providing provenance and updating knowledge remain open problems. Their models pair a pre-trained sequence-to-sequence model with a dense vector index of Wikipedia, reached through a pre-trained retriever. The abstract reports state-of-the-art results on three open-domain question answering tasks, and more specific, diverse and factual generated language than a parametric-only baseline.

**Following instructions (2022).** Ouyang and colleagues' [InstructGPT](https://arxiv.org/abs/2203.02155) opens with the claim that making language models bigger does not inherently make them better at following a user's intent. They fine-tune GPT-3 on labeler demonstrations, then further with reinforcement learning from human feedback on labeler rankings of outputs. In human evaluations on their prompt distribution, outputs of the 1.3B-parameter InstructGPT are preferred to those of the 175B GPT-3. The abstract also reports improvements in truthfulness and reductions in toxic output, and says the models still make simple mistakes.

## Speech

**End-to-end recognition (2014).** Hannun and colleagues' [Deep Speech](https://arxiv.org/abs/1412.5567) replaces the engineered processing pipelines of traditional systems with a recurrent network trained end to end. The abstract says it needs no phoneme dictionary, "nor even the concept of a phoneme", and does not use hand-designed components to model background noise, reverberation or speaker variation. It reports 16.0% error on the full Switchboard Hub5'00 test set, and better handling of noisy environments than the commercial systems it compared against.

**Self-supervised speech (2020).** Baevski and colleagues' [wav2vec 2.0](https://arxiv.org/abs/2006.11477) learns representations from speech audio alone, masking the input in the latent space and solving a contrastive task, then fine-tunes on transcribed speech. With all the labelled Librispeech data it reaches word error rates of 1.8 on the clean and 3.3 on the other test sets. With ten minutes of labelled data and pre-training on 53,000 hours of unlabelled audio it still reaches 4.8 and 8.2.

**Weak supervision at scale (2022).** Radford and colleagues' [Whisper](https://arxiv.org/abs/2212.04356) trains models simply to predict transcripts of audio from the internet, 680,000 hours of multilingual and multitask data. The abstract reports that the models generalise well to standard benchmarks, often competitively with fully supervised results in a zero-shot setting without fine-tuning, and that they approach human accuracy and robustness.

## What the sources show

These points are the author's reading of the sources above, not claims the sources make.

- **Both columns moved from labelled to unlabelled data.** Text went from word vectors learned from raw text, to BERT's pre-training on unlabeled text. Speech went from Deep Speech, to wav2vec 2.0 on audio alone, to Whisper's internet transcripts.
- **Each system dropped a hand-built component.** Deep Speech dropped the phoneme dictionary, GNMT's wordpieces dropped the fixed word list, and BERT dropped most task-specific architectures.
- **The later text papers are about the model's use, not its accuracy.** RAG addresses where knowledge lives, and InstructGPT addresses what the model is asked to optimise. Their abstracts are written in terms of provenance, truthfulness and user intent.

## Limits of this summary

- It covers eight papers. Dialogue systems, summarisation, sentiment analysis, speech synthesis and speaker recognition are left out.
- Every paper was read as an abstract page only. The figures quoted are those the abstracts state, on benchmarks that differ between rows and that have been replaced since.
- The Part 2 material is not repeated, and this post makes no claim about which current model is best.
