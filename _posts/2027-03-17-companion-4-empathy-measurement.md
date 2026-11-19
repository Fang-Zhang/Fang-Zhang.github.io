---
title: "Companion AI (4/8): Emotion and Empathy, What Can Be Measured"
date: 2027-03-17 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, companion-ai, empathy, evaluation, llm-as-judge, series]
description: "Emotion recognition is measurable and a 1.5B model does it well. Empathy scores are a different matter: a fixed sentence beats the model's own replies on a model judge, and a reply naming the wrong emotion scores the same as one naming the right one."
---

A companion is expected to respond to feelings, and every product claims to. This post separates three things that get called empathy: recognising an emotion, expressing a fitting response, and the effect on the person. It measures the first two on a 1.5B model, shows how a model judge and a keyword score can be fooled by text that ignores the message, and says what remains unmeasurable offline.

## What the literature measures

- [EmpatheticDialogues](https://arxiv.org/abs/1811.00207) supplies the standard training and evaluation conversations grounded in emotional situations (Part 1 covers it).
- [Sharma et al.](https://arxiv.org/abs/2009.08441) build a theory-grounded framework for expressed empathy in text, a corpus of 10k annotated (post, response) pairs, and a RoBERTa-based model that identifies empathy and extracts rationales. Applied to 235k mental-health interactions, it finds that users do not self-learn empathy over time.
- [EQ-Bench](https://arxiv.org/abs/2312.06281) asks models to predict the intensity of characters' emotional states in a dialogue, using 60 questions. It correlates strongly with MMLU (r=0.97), which its authors read as capturing similar aspects of broad intelligence.
- [Wang et al.](https://arxiv.org/abs/2307.09042) build a psychometric test of emotion understanding with a reference frame of over 500 adults. GPT-4 scores an EQ of 117, above 89% of human participants, and a pattern analysis suggests some models do not use a human-like mechanism to get there.
- [EmotionBench](https://arxiv.org/abs/2308.03656) measures how models' own reported feelings change in over 400 situations against reference results from more than 1,200 human subjects. Models respond appropriately to some situations but fall short in alignment with human emotional behaviour and cannot connect similar situations.
- [EmotionPrompt](https://arxiv.org/abs/2307.11760) is a different question: adding emotional stimuli to a prompt can improve task performance, with reported relative gains of 8.00% in Instruction Induction and 115% in BIG-Bench.

Recognition and understanding have benchmarks with human reference scores. Expressed empathy has annotation frameworks. The effect on the person does not appear in this list, and Part 1's dependence studies measure it only over weeks of use, with participants.

## Test 1: recognition

30 first-person messages written for this test, 6 for each of five emotions (sadness, anger, fear, joy, shame), none containing the emotion's name. The model picks one label by comparing the next-token logits of the five option words.

```python
first = {e: sorted({tok.encode(v)[0] for v in (e, e.capitalize())}) for e in EM}
lg = last_logits(f'Message: "{t}"\nWhich emotion is the speaker mainly feeling? '
                 'Options: sadness, anger, fear, joy, shame. Answer with one word.')
pred = max(EM, key=lambda e: max(lg[i].item() for i in first[e]))
```

Accuracy is 27/30 = 0.90 (Qwen2.5-1.5B-Instruct, bfloat16, CPU). Sadness, fear and joy are 6 of 6; anger is 4 of 6 with both errors labelled shame; shame is 5 of 6 with one labelled anger. The errors fall on the two emotions that blur in everyday use.

## Test 2: three ways to score a reply

Five responders produce a reply to each of the 30 messages:

- `plain model`: the message alone, greedy decoding, 60 new tokens.
- `model told to be empathic`: a system prompt asking for a warm, caring reply in two short sentences.
- `fixed template`: the same sentence for every message: "I'm so sorry you're feeling this way. That sounds really hard, and I'm here for you."
- `template, right emotion`: "I can hear how {emotion} you feel. That makes sense, and I'm here for you." with the true emotion.
- `template, wrong emotion`: the same sentence with a different emotion.

Three scores:

- Marker rate: occurrences of empathy phrases ("sorry", "understand", "feel", "hear", "sounds", "here for you", "that must be", "makes sense", "valid"), per 100 words.
- Model judge: the same 1.5B model rates the reply from 1 to 9; the score is the expectation over the logits of the nine digits.
- Emotion check: whether the reply contains a word from the true emotion's keyword family, and whether it contains one from another family.

```python
dig = [tok.encode(str(d))[0] for d in range(1, 10)]
def judge(user, reply):
    lg = last_logits(f'A person said: "{user}"\nA companion replied: "{reply}"\n'
                     'Rate how empathetic the reply is, from 1 (not at all) to 9 (extremely). Answer with one digit.')
    p = torch.softmax(lg[dig], 0).numpy(); return float((p * np.arange(1, 10)).sum())
```

## Results

![Confusion matrix of emotion recognition, model-judge scores for five responders, and keyword check of the emotion named in the reply]({{ site.baseurl }}/assets/pic/companion-4-empathy.png){: width="900" }
*Left: recognition, true against predicted label. Middle: mean judge score (the vertical axis starts at 6). Right: share of replies containing a keyword of the true emotion and of a different emotion.*

| Responder | words | markers per 100 words | judge (1–9) | judge on joy messages |
|---|---|---|---|---|
| plain model | 49.8 | 17.5 | 7.61 | 8.32 |
| model told to be empathic | 24.1 | 17.3 | 7.93 | 8.75 |
| fixed template | 16.0 | 50.0 | 8.69 | 8.74 |
| template, right emotion | 15.0 | 60.0 | 8.37 | 8.56 |
| template, wrong emotion | 15.0 | 60.0 | 8.31 | 8.33 |

- The marker rate rewards the templates, 50 to 60 per 100 words against about 17 for the model's own replies. A reply that repeats the same phrases wins.
- The model judge ranks the fixed template (8.69) above both model-written replies (7.61 and 7.93). On the six joy messages it gives "I'm so sorry you're feeling this way" 8.74, the same as the empathic model's reply to the same messages (8.75).
- The judge puts the right-emotion template at 8.37 and the wrong-emotion template at 8.31. It barely separates a reply that names the right emotion from one that names a wrong one.
- The keyword check separates them completely (1.00 against 0.00 for the true emotion) because it looks at content. It also shows its crudeness: the fixed template's "sorry" belongs to the sadness family, so it counts as right for sadness messages (0.20, six of 30) and as a different emotion for the rest (0.80).

A real reply to a joyful message looks like this (plain model): "Congratulations! It's great news that you've been accepted into your dream school or program." A reply to a sad one: "I'm sorry to hear that your grandmother's house has been sold. It must have been a difficult time for you as you were able to walk through the empty rooms alone." The second one reads as empathic and also misreads the sentence, which no score above notices.

## What this shows and does not show

- Recognition is a classification task with a gold label and can be scored (0.90 here, with the confusion between anger and shame visible).
- Surface scores of expressed empathy, whether a marker count or a judge model of this size, respond to the form of the text. They did not check that the reply fit the message.
- The test does not show that fixed text is better for users. The judge's preference may reflect that the template is warm and short. It shows only that the judge's number cannot be read as fit to the message.
- The outcome that matters, whether the person is better off, is outside all three scores. It needs people, over time, and Part 1 cites the only kind of study that has done it.

## Limits

- 30 messages written for this test, five emotions, short and unambiguous. Real messages mix emotions, carry sarcasm and depend on history.
- The judge is the same 1.5B model, with one prompt and no calibration against people. Larger judges, or judges with rubrics and several prompts, may do better. That was not tested.
- The templates are adversarial constructions chosen to expose what the scores ignore, not text any product emits. The keyword families are short word lists written for this test and miss paraphrases such as "bittersweet".
- No human ratings were collected, so none of the scores is validated against what people judge empathic.
- Greedy decoding, one run, 30 messages per responder; the third decimal of any score is noise.

Next, in [Part 5](/posts/companion-5-sycophancy/): sycophancy, the failure where warmth and agreement are confused.
