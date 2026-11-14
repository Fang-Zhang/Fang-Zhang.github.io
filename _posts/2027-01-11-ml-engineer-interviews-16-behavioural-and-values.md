---
title: "ML Engineer Interviews, With Reference Answers (16/18): Behavioural and Values Questions"
date: 2027-01-11 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, behavioural, star-method, responsible-ai, fairness, series]
description: "Part 16 of 18. Ten behavioural and values questions for ML engineer interviews: why they are asked, the STAR structure, conflict, failure, feedback, responsible AI with a fairness simulation, ambiguity, influence, company values as a rubric and hypothetical questions."
---

The last rounds of an ML engineer loop usually include behavioural questions: what the candidate did in past situations, how they worked with others and how they handle questions with no technical answer. After [Part 15](/posts/ml-engineer-interviews-15-project-deep-dive/), which followed one project in depth, this post covers ten questions of that kind with what each tests, a reference answer, a weak answer and a follow-up. These answers are stories, and stories cannot be verified here, so every story below is hypothetical and is marked as such; the evidence the post does use is limited to sources that were read: the personnel-selection meta-analyses of [Sackett et al. (2022)](https://gwern.net/doc/statistics/meta-analysis/2021-sackett.pdf) and [Schmidt and Hunter (1998)](https://psycnet.apa.org/record/1998-10661-006), Indeed's description of the [STAR technique](https://www.indeed.com/career-advice/interviewing/how-to-use-the-star-interview-response-technique), three fairness papers, and two published AI-governance pages. Two questions come with code on this machine (CPU, fixed seeds). Advice that no source supports is marked as the author's.

![Left: mean job performance of the top 20% selected by a score, in standard-deviation units, against the validity of the score; 0.19 gives 0.27 and 0.42 gives 0.59. Right: with one threshold, group A has a true-positive rate of 0.52 and precision 0.56, and group B has 0.10 and 0.50; lowering group B's threshold gives B a true-positive rate of 0.52 and a precision of 0.25.](/assets/pic/ml-interview-16-validity-fairness.png)
*Left: what a validity coefficient buys (question 1). Right: equalising the true-positive rate in a calibrated model costs precision (question 6).*

## 1. "Why do companies ask behavioural questions, and how are the answers scored?"

**What it tests.** Whether the candidate understands the format well enough to prepare for it.

**Reference answer.** The format exists because structured interviews, in which every candidate gets the same questions and is scored against a set rubric, predict job performance better than unstructured conversations. Sackett et al. re-estimated the validity of selection procedures and report a mean validity of 0.42 for structured interviews and 0.19 for unstructured ones; their estimate for structured interviews is lower than the 0.51 of the earlier Schmidt and Hunter analysis, because they dropped a range-restriction correction that relied on an assumed distribution, but it is still the highest on their list. Schmidt and Hunter report that the combination of general mental ability and a structured interview has a multivariate validity of 0.63. What a validity coefficient means in practice is easy to compute: if a score has correlation r with later performance, the average performance of the top 20% selected on it is r times the average standardised score of that group.

```python
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(0); n = 2_000_000; sr = 0.20                      # hire the top 20% by score
z = norm.ppf(1 - sr); expected_z = norm.pdf(z) / sr                             # mean of a standard normal above its 80th percentile
print(f"selection ratio {sr:.0%}; the mean standardised score of those selected is {expected_z:.3f}")
print("validity r | mean job performance of those selected (SD units): simulated, formula r x mean score | selected above median performance")
perf = rng.normal(size=n)
for r in (0.0, 0.19, 0.31, 0.42, 0.51):
    score = r * perf + np.sqrt(1 - r**2) * rng.normal(size=n); sel = score > np.quantile(score, 1 - sr)
    print(f"{r:10.2f} | {perf[sel].mean():+.3f}, {r * expected_z:+.3f} | {np.mean(perf[sel] > 0):.1%}")
```

```
selection ratio 20%; the mean standardised score of those selected is 1.400
validity r | mean job performance of those selected (SD units): simulated, formula r x mean score | selected above median performance
      0.00 | -0.001, +0.000 | 49.9%
      0.19 | +0.266, +0.266 | 60.7%
      0.31 | +0.435, +0.434 | 67.4%
      0.42 | +0.589, +0.588 | 73.7%
      0.51 | +0.714, +0.714 | 78.8%
```

The mean standardised score of the top 20% is 1.400, so a score with validity 0.19 selects people whose average performance is 0.27 standard deviations above the mean, and a score with validity 0.42 selects people at 0.59, more than twice as far. The simulation matches the formula to the third decimal. Among those selected, 60.7% are above the median performer at validity 0.19 and 73.7% at 0.42. The interviews are noisy even when structured: a validity of 0.42 leaves most of the variation in performance unexplained, which is why loops use several interviewers. For preparation, the consequence is that each answer is scored against a rubric of the competencies the role needs (collaboration, ownership, handling ambiguity, technical judgement), so an answer has to show evidence of those, not charm (the author's inference from the format).

**Common weak answer.** Treating the behavioural round as small talk, or preparing one generic story for every question.

**Follow-up.** "What would a rubric score on your story?" The answer lists the behaviours that the story shows: what the candidate decided, what they did and what changed.

## 2. "Tell me about a time you ... " (the STAR structure)

**What it tests.** Whether the answer is a story with a decision and a result, in a size that fits the time.

**Reference answer.** Indeed's description of the STAR technique has four parts: situation (context, with two or three important details, and the least time spent on it, because interviewers are more concerned with the actions and results), task (the candidate's own responsibility), action (what the candidate did) and result (two or three main outcomes, what was learned, with concrete examples, and slightly less time than the action). Two additions (the author's): put one number in the result, and use "I" for the actions that were the candidate's own, with named colleagues for the others (Part 15, question 10). A hypothetical example for a question about taking ownership: "Our weekly ranking-model report had been wrong for a month because the label join dropped late events (situation). I was the owner of the model but not of the pipeline (task). I wrote a check that compared the label counts with the source system, found the join problem, convinced the data team to fix it, and added the check to the weekly job (action). The label count error went from 6% to under 0.5%, the model's reported precision dropped from 0.34 to 0.29, which was the honest number, and two other teams adopted the check (result)." It takes about 90 seconds.

**Common weak answer.** A long situation, a vague action ("we worked on it") and no result, or a result without the candidate's part in it.

**Follow-up.** "What would you do if it happened again?" This probes whether the story ended in a changed practice.

## 3. "Tell me about a disagreement with a teammate or stakeholder."

**What it tests.** Whether disagreement is handled with evidence and respect, and whether the candidate can describe the other side fairly.

**Reference answer.** Choose a real, technical disagreement in which both positions were defensible. Describe the other person's view in the way they would have, state what each side wanted (not who was right), and say how it was settled: usually by an experiment, a metric agreed in advance or an escalation with a clear question. A hypothetical example: a product manager wanted to launch a recommendation change on the strength of a lift in click rate; the candidate worried that the lift came from a positional effect (Part 9). They agreed that a holdout with randomised positions would settle it, ran it for two weeks, found that the lift was half as large, and launched with the smaller expected gain. The reference answer then closes the loop on the relationship: the stakeholder asked for the same check on their next launch. Avoid the stories where the candidate was right and the other party was foolish; the interviewer is also hearing how the candidate will describe them later.

**Common weak answer.** A story with a villain, or "I don't have conflicts".

**Follow-up.** "What if the evidence had gone the other way?" The candidate should say what they would have done: launch, and say so.

## 4. "Tell me about a failure."

**What it tests.** Accountability, and the ability to learn in a specific way.

**Reference answer.** The story has a real cost, the candidate's own share of the cause and a change in practice. Part 15 (question 9) listed technical sources of such stories; the structure is the same: what happened, what it cost, what the candidate had decided that turned out to be wrong, what is done differently now. A hypothetical example: "I launched a model after checking an offline AUC that used a random split. The labels arrived two weeks late, so the test rows overlapped in time with the training rows. Production precision was 0.21 against the 0.30 I had reported. I had not asked how the labels were produced. Now I split by time, state the label delay in the report and ask for a shadow period before any launch." Do not choose a failure that is a strength in disguise, and do not choose one so large that the interviewer will wonder about the candidate's judgement; a real mistake of moderate cost, owned and fixed, is the target (the author's view).

**Common weak answer.** "My biggest weakness is perfectionism", or a failure blamed on a colleague.

**Follow-up.** "What did it cost, and who knew?" The honest answer includes how the candidate told the people affected.

## 5. "How do you handle feedback, given and received?"

**What it tests.** Whether the candidate can take correction and give it usefully.

**Reference answer.** For receiving: describe one piece of critical feedback, whether the candidate agreed with it, what they checked and what changed. The strongest version is one where the first reaction was disagreement and the evidence changed it, or the opposite, where after checking the candidate kept their position and explained why. For giving: say how feedback is delivered so that it can be acted on: specific (the observation, not the person), soon after the event, with the effect and a suggested alternative. A hypothetical ML example: "A junior colleague's evaluation notebook computed the metric on the training set. I told them privately the same day: this cell uses the training data, here is the effect on the number, here is how the pipeline avoids it. I asked them to present the corrected version at the team review, since they had found the fix quickly." The structure (the author's, not from a source) is observation, effect and next step.

**Common weak answer.** "I welcome all feedback" with no example, or feedback that is vague ("be more strategic").

**Follow-up.** "Tell me about feedback you disagreed with." The answer should show a test of the feedback, not just an opinion about it.

## 6. "Tell me about a time you worried that a model could cause harm."

**What it tests.** Whether responsible-AI practice is part of how the candidate works: what they measure, what they decide and what they escalate.

**Reference answer.** Describe the concern, the measurement, the decision and the follow-up. The measurement matters most, because "fairness" has several definitions that conflict. [Kleinberg, Mullainathan and Raghavan](https://arxiv.org/abs/1609.05807) formalise three fairness conditions and prove that, except in highly constrained special cases, no method satisfies all three at once. [Chouldechova](https://arxiv.org/abs/1610.07524) shows that adhering to a fairness criterion that originates in educational and psychological testing can lead to considerable disparate impact when prevalence differs across groups. [Hardt, Price and Srebro](https://arxiv.org/abs/1610.02413) propose equality of opportunity as a criterion and show how to adjust any learned predictor to satisfy it. The simulation makes the conflict concrete: two groups with prevalences of 0.30 and 0.10, the same signal quality, and a model that is calibrated within each group.

```python
import numpy as np
rng = np.random.default_rng(0); n = 400_000
prev = {"A": 0.30, "B": 0.10}; d = {}
sig = lambda t: 1 / (1 + np.exp(-t)); logit = lambda p: np.log(p / (1 - p))
for g, p in prev.items():
    y = rng.random(n) < p; x = rng.normal(size=n) + 1.0 * y                         # same signal quality in both groups
    s = sig(x - 0.5 + logit(p)); d[g] = (y, s)                                        # exact posterior: calibrated within each group
for g, (y, s) in d.items():
    bins = np.digitize(s, np.linspace(0, 1, 11)); gap = max(abs(y[bins == b].mean() - s[bins == b].mean()) for b in range(1, 11) if (bins == b).sum() > 500)
    print(f"group {g}: prevalence {y.mean():.3f}; largest calibration gap over score bins {gap:.3f}")
def row(g, t):
    y, s = d[g]; f = s > t; return f"group {g} threshold {t:.3f}: selected {f.mean():.3f}  TPR {f[y].mean():.3f}  FPR {f[~y].mean():.3f}  PPV {y[f].mean():.3f}"
print("\none threshold for both groups (0.40):"); [print(row(g, 0.40)) for g in d]
yA, sA = d["A"]; tprA = (sA > 0.40)[yA].mean(); yB, sB = d["B"]
tB = np.quantile(sB[yB], 1 - tprA)
print(f"\ngroup B threshold lowered to {tB:.3f} so that its true-positive rate equals group A's:"); [print(row(g, 0.40 if g == 'A' else tB)) for g in d]
```

```
group A: prevalence 0.300; largest calibration gap over score bins 0.005
group B: prevalence 0.100; largest calibration gap over score bins 0.016

one threshold for both groups (0.40):
group A threshold 0.400: selected 0.279  TPR 0.523  FPR 0.174  PPV 0.563
group B threshold 0.400: selected 0.020  TPR 0.099  FPR 0.011  PPV 0.502

group B threshold lowered to 0.146 so that its true-positive rate equals group A's:
group A threshold 0.400: selected 0.279  TPR 0.523  FPR 0.174  PPV 0.563
group B threshold 0.146: selected 0.209  TPR 0.523  FPR 0.174  PPV 0.251
```

The calibration gaps are small (0.005 and 0.016), so a score of 0.4 means about 0.4 in both groups. With one threshold of 0.40, group A has a true-positive rate of 0.523 and group B 0.099: the model finds half of the positives in the first group and a tenth in the second, although it is calibrated and equally precise (0.563 and 0.502). Lowering group B's threshold to 0.146 equalises the true-positive rate at 0.523 (and the false-positive rate at 0.174, since the signal has the same shape in both groups), but precision for group B falls to 0.251 against 0.563 for group A, and 20.9% of group B is selected against 2.0% before. Neither setting is "the fair one"; each satisfies one definition and breaks another, as the theorems say, and the choice depends on the cost of each kind of error to each group. The behaviour that interviewers want is that the candidate measured by group, explained the trade-off to the decision-maker in terms of these costs, and recorded the decision. Documentation practices such as model cards (Part 14, question 9) and [datasheets for datasets](https://arxiv.org/abs/1803.09010) give the record a place.

**Common weak answer.** "The model does not use protected attributes, so it is fair", which ignores proxies, or "fairness is not my job".

**Follow-up.** "Who decided the threshold?" The candidate should name the owner of the decision and what information was given to them.

## 7. "Tell me about working with ambiguity, or too many priorities."

**What it tests.** Whether the candidate can turn a vague request into a plan and say no with reasons.

**Reference answer.** In ML work the vague request is typically "make the model better" or "use AI for X". The steps (the author's framework): restate the goal as a decision and a metric and confirm it with the requester; list what would have to be true for the work to be worth doing; run the smallest test that would show whether it is (a baseline and a data check, as in Part 14); and agree a stopping point. For too many priorities, rank by value, cost and risk, show the ranking to the stakeholders and ask them to change it if they disagree, so that the trade-off is theirs and visible. A hypothetical example: three teams asked for model changes in one sprint; the candidate wrote down the expected value and effort of each, proposed the order, and moved one request to a baseline experiment that took two days and showed there was no signal in the data for that request. The result was a clear "no" that saved a month.

**Common weak answer.** "I work harder", or taking every request.

**Follow-up.** "What did you drop?" A story with nothing dropped is not a story about prioritising.

## 8. "How do you explain a model to someone who has to make a decision with it?"

**What it tests.** Influence without authority, and the ability to communicate uncertainty.

**Reference answer.** Start from the decision, not from the model: what they can do, what it costs, and what the model changes. Give the number with its range and a plain-language reading ("about three in ten of the flagged cases are real, give or take a few"), show one example of a correct and one of a wrong prediction, and say when the model should not be used. Use the same metric every time, so that the audience learns what it means. When the audience disagrees, ask which part of the evidence they doubt, and offer a test. This is the presentation structure of Part 14 (question 9) applied to a smaller audience, and the uncertainty is the part to practise: an interval, not a point (Part 13, question 5). A hypothetical example is the payments model of Part 15 (question 2), where the net monthly value was given as a range, with the unit values stated as assumptions that the finance team could change.

**Common weak answer.** Explaining the algorithm, or a single number presented as certain.

**Follow-up.** "The executive wants to launch anyway. What do you do?" State the risk in their terms, propose a staged launch with a stop condition, and record the decision.

## 9. "Why us? What do our values mean to you?"

**What it tests.** Whether the candidate has read the company's public statements and can connect them to practice.

**Reference answer.** Published values and principles pages work as a rubric because interviewers write their behavioural questions from them. Read the page and extract the verbs. For example, Google's AI Principles include a commitment to "employing rigorous design, testing, monitoring, and safeguards to mitigate unintended or harmful outcomes and avoid unfair bias." A candidate would prepare a story that matches each verb: testing (the leakage check of Part 14), monitoring (Part 11) and bias (question 6 above). NIST's AI Risk Management Framework, released in January 2023 for voluntary use, is another document that interviewers in regulated industries may expect candidates to have heard of. The method (the author's): for each company, list its published values, map each to one of the candidate's stories, and check that the story supports the value with an action and not a slogan. Do not claim a company's values as one's own without a story; the interviewer will ask for it.

**Common weak answer.** "I love your mission" with no specific thing the candidate has read, or reciting the values page.

**Follow-up.** "Which of our values would be hardest for you?" Name one honestly, with an example of how the candidate would handle it.

## 10. "What would you do if ...?" (hypothetical questions)

**What it tests.** Reasoning and judgement in a situation the candidate has not been in. Situational questions describe a problem and ask for the response, where behavioural ones ask for a past instance.

**Reference answer.** Say the assumptions, ask one or two clarifying questions, give the first step and the criteria for the next ones, and name what would change the plan. A typical ML scenario: "The launch is tomorrow, the overall metric is up, and you find that the model does poorly on a small segment." A reference answer: confirm the finding (is it a data error or a real failure? How many users?); size the harm and the benefit (the segment's share and the cost of an error there; Part 14, question 5 shows an overall AUC of 0.810 hiding a group at 0.247); propose options (launch with the segment excluded or on the old model, delay, or launch with a mitigation and a monitor); recommend one and name who decides; and say what would be reported afterwards. The answer shows a method, not a verdict, because the interviewer cannot know whether the candidate's verdict is right and can see whether the reasoning is sound.

**Common weak answer.** An immediate verdict ("I'd delay the launch") with no sizing of the problem, or a refusal to answer without more information.

**Follow-up.** "The VP says launch anyway." Question 8 applies: state the risk in their terms, propose the staged launch, and make the decision visible.

## Takeaways

- Behavioural rounds are structured interviews: same questions, rubric scoring. Structured interviews had a mean validity of 0.42 against 0.19 for unstructured ones; at a 20% selection ratio that is 0.59 against 0.27 SD in the average performance of those selected.
- STAR: little on the situation, most on the action, a number in the result, "I" for your own part.
- Use real, moderate-cost stories with a changed practice; keep the other side of a conflict fair.
- Fairness definitions conflict: in a calibrated model with different prevalences, one threshold gave true-positive rates of 0.523 and 0.099; equalising them dropped group B's precision to 0.251 against 0.563.
- Measure by group, explain the trade-off in terms of costs, and record who decided.
- Read the company's published values and map each verb to a story with an action.
- For hypotheticals, give assumptions, a first step, options and a recommendation, not only a verdict.

## Limits

- The stories are hypothetical and illustrate a structure; none claims to be an actual case, and a candidate's real stories need their own facts. Interviewers differ in what they reward.
- The validity figures are estimates from meta-analyses of selection procedures in general and not specific to ML roles or to any company's process; the simulation assumes normally distributed scores and a 20% selection ratio.
- The fairness simulation uses synthetic data with Gaussian signals and exactly calibrated scores. Real systems have unequal signal quality, proxies and other conditions. The three papers were read at the level of their abstracts for this post, and no derivation from them is reproduced.
- The sources for the values question are two published pages (Google's AI Principles and NIST's AI Risk Management Framework overview), read as published; other companies' pages differ and change.
- The framework for ambiguity, the feedback structure, the advice on choosing a failure and the claims about what interviewers infer are the author's.
- Not covered: culture-fit questions that differ by company, diversity of interviewer styles, and legal limits on what interviewers may ask.

Next: [Part 17, questions to ask, warning signs, offers and a preparation method](/posts/ml-engineer-interviews-17-questions-to-ask-and-offers/).
