---
title: "ML Engineer Interviews, With Reference Answers (1/18): Tell Me About Yourself"
date: 2026-12-27 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, machine-learning, career, series]
description: "Part 1 of 18. Ten versions of the opening question in an ML engineer interview, each with what it tests, a reference answer, a weak answer and a follow-up."
---

This post does one thing: it takes the opening question of an ML engineer interview, "tell me about yourself", and works through ten versions of it, from the general case to a 30-second limit. Each version has what the question tests, a reference answer, a common weak answer and a follow-up. The reference answers are the author's, written for hypothetical candidates with invented details; they are not standard answers and not questions from any named company. The series has 18 parts, in three rounds: screening (Parts 1 and 2), technical (Parts 3 to 14) and final (Parts 15 to 17), plus a reading list in Part 18.

## What the question is for

Interviewers ask it for several reasons. [Big Interview](https://www.biginterview.com/resources/tell-me-about-yourself) lists five: it eases the candidate into the interview, shows how they handle an open-ended question, helps judge overall fit, makes the interview feel like a conversation, and shows the person behind the resume. Its fourth point matters most here: a detail in the answer can catch the interviewer's attention and become the starting point for a string of follow-up questions. [Grow with Google](https://grow.google/grow-your-career/articles/interview-tips) states the interviewer's request as a clear, concise summary of who you are, what you do and why you might fit this specific role, and names the common mistake as telling your entire professional or personal history.

The author's reading of those two sources is that the answer is a menu. Whatever the candidate names, the interviewer is likely to ask about next. A good answer therefore does two jobs: it summarises, and it chooses the next ten minutes of the conversation.

The research on interviews supports the habit of answering with evidence rather than charm. [Sackett, Zhang, Berry and Lievens (2022)](https://gwern.net/doc/statistics/meta-analysis/2021-sackett.pdf) revisited the validity estimates of hiring methods and report that structured interviews emerged as the top-ranked selection procedure, with an operational validity of .42 against .19 for unstructured interviews. The earlier figures from [Schmidt and Hunter (1998)](https://psycnet.apa.org/record/1998-10661-006) were .51 and .38; Sackett and colleagues argue that those were inflated by overcorrection for restriction of range. An opening question is rarely scored on its own, so this is context, not proof about the question. It does explain why a candidate does better to organise the answer around job-relevant facts the interviewer can follow up.

## The formula and the time budget

The most common structure is present, past, future. [Big Interview](https://www.biginterview.com/resources/tell-me-about-yourself) describes it this way: the present is where you are in your career, the past is how earlier roles and experiences shaped you, and the future is your plans and how this role fits them. Its own sample answers follow that order. On length the sources differ. Grow with Google suggests a succinct 30 to 60 second pitch, well practised but not scripted, leading with the strengths most relevant to the role. Big Interview says the answer should probably fall between 30 seconds and 2 minutes, and gives 1 to 2 minutes as the ideal in its FAQ.

The author's suggestion is about 90 seconds for a general case, split as in the figure, with a final sentence that hands the conversation to a specific project. The split is the author's; the ranges above are the cited ones.

![A horizontal bar showing a 90-second self-introduction divided into Present (0 to 25 seconds), Past (25 to 60), Future (60 to 80) and Handoff (80 to 90), each with a one-line content note.]({{ "/assets/pic/ml-interview-1-intro-budget.png" | relative_url }})
*A 90-second introduction. The ranges of 30 to 60 seconds and 30 seconds to 2 minutes are cited; the split inside 90 seconds is the author's.*

Three rules apply to every version below.

- **Lead with the role, not the biography.** Name the current role and one result in the first sentence.
- **Put two or three hooks in.** A hook is a project, a system or a measured result that the candidate is prepared to discuss for ten minutes. Anything named will be asked about, so name nothing that cannot be defended.
- **End with a handoff.** The last sentence points at the project the candidate wants to talk about next.

## 1. "Tell me about yourself."

**What it tests.** Clarity, relevance to the role, and whether the candidate can organise a story without being led.

**Reference answer.** The structure is present, past, future, then a handoff. For a hypothetical mid-level candidate for a recommendation-ranking role, with invented details:

> "I'm a machine learning engineer on the ranking team at an online marketplace. I own the retrieval and ranking models for the home feed and the service that serves them. Last year I replaced a gradient-boosted ranker with a two-stage model and raised click-through rate in an A/B test by a few percent, with latency unchanged. Before that I spent three years as a backend engineer, which is where I learned to care about latency budgets and failure modes; I moved to machine learning after building the logging pipeline that our first ranking model trained on. I now want to work where the ranking problem is larger and the experiment volume is higher, which is why this role on your search team interests me. The project I would most like to talk about is the two-stage ranker, because the hard parts were the offline-to-online gap and the evaluation design."

Why it works: the first sentence gives title, scope and ownership. The result has a metric and a constraint. The past section explains why the career moved, not just what it contained. The future section connects to this role without flattery. The last sentence picks the topic of the next question, and it picks one the candidate has chosen to be strong on.

**Common weak answer.** A chronological reading of the resume from school onwards, with every employer and no result. It uses the whole time budget on the past and leaves the future and the handoff out. A second weak form is a list of tools ("Python, PyTorch, Spark, Kubernetes") with no project attached to any of them.

**Follow-up.** "You mentioned a two-stage ranker. What was the offline metric, and did it predict the online result?" The handoff sentence invited exactly this, so the candidate should have the answer ready.

## 2. "Walk me through your resume."

**What it tests.** The same as question 1, plus whether the candidate can compress a long history and choose what to skip. [Big Interview](https://www.biginterview.com/resources/tell-me-about-yourself) lists this phrasing among the common alternatives to "tell me about yourself", and its advice is to recognise the intent and use the same formula.

**Reference answer.** Keep the present-past-future shape but let the past follow the resume's order, newest to oldest or oldest to newest, whichever the interviewer is holding. Spend the time unevenly: one sentence for early roles, more for the last two, because those are the ones with results. For each role give three things: what the team did, what the candidate owned, and one outcome. Skip a role in one clause if it has nothing to do with the target ("a short role in IT support before I moved to engineering"). End by asking which part the interviewer wants to go into, which hands them the choice in a polite form.

Sample script for the middle of such an answer: "At the second company I joined a team of six that ran fraud models for card payments. I owned the feature pipeline, which moved from daily batch to a streaming job, and I led the retraining schedule after we found the model decayed within weeks of each release. That is the part of the resume I would suggest we go through in the most detail."

**Common weak answer.** Reading each bullet aloud, in the order it is printed, at equal length. The interviewer has the document and learns nothing new.

**Follow-up.** "Which of these roles taught you the most, and why?" The candidate should have chosen it before the interview and should be able to answer in under thirty seconds.

## 3. The new graduate with no or little industry experience

**What it tests.** Whether the candidate can show evidence of working on real problems without a job history to point to.

**Reference answer.** Replace the "present" with the current status and a best piece of work, and the "past" with the two or three projects that show applied skill. Evidence has to be checkable: a repository, a paper, a competition result, a dataset the candidate cleaned. For a hypothetical graduate:

> "I'm finishing a master's degree in computer science with a focus on machine learning. My thesis was on detecting label noise in a medical imaging data set; I implemented three methods, ran them on a public chest X-ray data set, and found that the simplest one, a loss-based filter, matched the more complex ones within the error bars of five seeds. Before that I did a summer internship on a data-engineering team, where I wrote the validation checks that caught a schema change before it reached training. I'm looking for a first role on a team that ships models and lets me work on data quality as well as modelling. The thesis experiments are public, and I'd be glad to go through how I ran them."

The numbers are invented; the pattern is the point. It names a method, a data set, a finding, the care taken (five seeds), and an offer to go deep.

**Common weak answer.** Describing coursework as a list of courses. Courses are not evidence of anything other than enrolment.

**Follow-up.** "Why did the simple filter match the complex ones?" The candidate should explain the result in terms of the data, or say plainly that the experiment did not isolate the cause.

## 4. The career changer moving into machine learning

**What it tests.** Seriousness of the move and whether the earlier career is an asset or a gap. [Big Interview](https://www.biginterview.com/resources/tell-me-about-yourself) has a sample for career changers; it explains why the change is happening, connects the two fields, and shows steps already taken, so that the interviewer does not conclude the candidate lacks the required skills.

**Reference answer.** State the current field in one sentence, then the bridge, then the evidence. The bridge should be a skill the new role needs, not a feeling. For a hypothetical former analyst:

> "I've spent five years as a data analyst in logistics, building the demand reports that planners use. About two years ago I started building the forecasts themselves: first a seasonal baseline, then a gradient-boosted model with calendar and promotion features, which cut weekly error against the baseline in a backtest by a margin I can show you. I moved from reporting to modelling because the forecasting problems were where the decisions were. I've since taken over the retraining job and its monitoring. I'm looking for a role where modelling is the main work rather than a side task, and your team's forecasting product is that."

**Common weak answer.** Apologising for the background ("I don't have a computer science degree, but..."). The apology becomes the topic.

**Follow-up.** "How did you validate the forecast?" A backtest that respects time order, and a baseline, are the expected parts of an answer. This is covered in Part 10.

## 5. The researcher moving from academia to industry

**What it tests.** Whether the candidate can translate research into product terms and has any sign of shipping or collaborating beyond the lab.

**Reference answer.** Lead with the research question in one plain sentence, then what changed in the field because of it or what it was used for, then the engineering side: code the candidate wrote that others ran, scale, data size, reproducibility. For a hypothetical candidate:

> "I'm finishing a PhD on efficient fine-tuning of language models. The question was how little of a model has to change to adapt it to a new domain. My main result was a method that matched full fine-tuning on three benchmarks while training a small fraction of the parameters, and the code is used by two other labs. The engineering part I enjoyed most was making the experiments reproducible across a shared cluster, which meant writing the launch and logging tooling. I want to move to industry because I'd like the problems to come with users and deadlines. The project I'd like to talk about is the cluster tooling, because it is the part closest to this role."

**Common weak answer.** A talk abstract: three minutes of background with no statement of what was achieved and no sign of any practical use.

**Follow-up.** "If this had to run in production with a latency budget, what would you change?" The candidate should name what the research code did not handle: batching, failure recovery, monitoring.

## 6. The senior candidate

**What it tests.** Scope, judgment and leadership: whether the candidate has owned a system and other people's work, not only their own tasks. [Modern Descartes](https://www.moderndescartes.com/essays/ml_eng_interviewing) describes a "career chat" round in which the candidate discusses the career arc, highlights and goals for the next role, and says it signals ambition, agency and growth potential. A senior introduction is close to that conversation.

**Reference answer.** Open with scope in numbers (team size, systems owned, users or requests served, the budget decided), then one decision that shows judgment, then the target. For a hypothetical staff-level candidate:

> "I'm a staff engineer on the ML platform team. For the last four years I've owned the path from experiment to serving for about thirty models across six product teams, and I led the move from per-team pipelines to a shared one. The decision I'm proudest of was to not build a feature store at first: we shipped a shared feature-definition library and measured how many bugs were train-serve skew, and only built the store once the count justified it. I'm looking for a role with a wider mandate over how models are evaluated and deployed. I'd like to talk about the evaluation framework, since that is where the platform and the product teams disagreed most."

**Common weak answer.** A list of technologies at the senior level suggests that the candidate's value is tool knowledge. Another weak form is "we" throughout, with no statement of what the candidate decided.

**Follow-up.** "Who disagreed with you, and how was it resolved?" This leads into the behavioural questions of Part 16.

## 7. The employment gap or a short stay

**What it tests.** Honesty and composure. The interviewer wants a short, factual explanation and a sign that the candidate is ready to work.

**Reference answer.** Give the reason in one clause, give what the period produced, and return to the present. Do not dwell. For a hypothetical gap used for study:

> "After my last role ended in a reorganisation, I spent eight months on a self-directed programme: I completed a deep learning specialisation, reproduced two papers and published the code, and contributed fixes to an open-source evaluation library. I'm ready to return to a team role, and the experience reinforced that I want to work on model evaluation, which is why this position interests me."

For a short stay, say what was learned and what was decided: "The role changed scope after I joined, from modelling to dashboards; I stayed for nine months, finished the migration I had been hired for and moved on."

**Common weak answer.** Silence about the gap, or a long account of the circumstances. Both make the gap the main topic.

**Follow-up.** "What did you reproduce from the papers, and what didn't match?" Reproductions that did not match the paper are a good answer if the candidate can explain what was tried.

## 8. "Tell me about yourself", asked by an engineer in a technical round

**What it tests.** The same, but the audience is a peer who will probe technically, and the time is shared with the technical content.

**Reference answer.** Shorten to 45 to 60 seconds and make the hooks technical. Skip the career motivation and go straight to the system, its scale and the hardest problem. For example:

> "I work on the ranking models for a marketplace feed: a two-stage retrieval and ranking system serving tens of millions of requests a day, trained daily. The hardest problem recently has been the gap between offline NDCG and online click-through, which I traced to position bias in the training logs. I'm happy to start there or with the serving side."

The offer at the end lets the interviewer choose between two areas the candidate knows well, which gives the peer a quick route to the technical part.

**Common weak answer.** Repeating the recruiter-style narrative, including career motivation, to an engineer who has twenty minutes to test the candidate's technical depth.

**Follow-up.** "What's position bias, and how did you correct for it?" The follow-up is only fair if the candidate named it; this is the reason to name only what can be defended.

## 9. "What is not on your resume?"

**What it tests.** Whether the candidate can add something relevant that the document does not show, and judgment about what to share. [Big Interview](https://www.biginterview.com/resources/tell-me-about-yourself) lists this among the alternative phrasings and advises mentioning something that goes beyond the resume or cover letter.

**Reference answer.** Pick something that is professional or close to it and that supports the role: a side project, a habit of reading papers and reproducing them, a talk or a tutorial the candidate wrote, a way of working. For example:

> "Something the resume doesn't show is that I keep a log of every experiment I run, including the failed ones, with the configuration and the one-line reason I thought it would work. It has saved me twice from repeating a dead end, and when a model regressed in production last year, the log let me tell the team what had been tried and when. I've also started writing short notes on papers I reproduce, and I can share one."

The answer shows a habit that is useful in the job and can be checked.

**Common weak answer.** An unrelated hobby offered as the whole answer. [Big Interview](https://www.biginterview.com/resources/tell-me-about-yourself) does say a hobby that aligns with the company culture can show the person behind the resume, so a hobby is not wrong, but it works as a sentence added to a relevant answer and not as the answer.

**Follow-up.** "Can you show me one of those notes?" The candidate needs to have them in a shareable form.

## 10. "You have 30 seconds."

**What it tests.** Compression. This version appears in short screens, panel interviews and video submissions where the time is fixed. It sits inside the 30-second lower bound that both cited sources allow.

**Reference answer.** Keep three sentences: the present with one result, the relevant past in one clause, and the target. Drop the handoff only if there is no time, because it is the first thing the interviewer will want.

> "I'm a machine learning engineer on a marketplace ranking team; last year I led a two-stage ranker that improved click-through in an A/B test without raising latency. Before that I was a backend engineer, which shapes how I think about serving. I'm looking to work on search ranking at larger scale, which is why I'm interested in this role."

**Common weak answer.** The 90-second answer spoken faster. The structure has to be cut, not the speed raised.

**Follow-up.** "Tell me more about the ranker." The short version is a handoff in itself.

## How to prepare

1. **Write the 90-second version, then cut it to 30 and 45.** The 45-second version is for technical rounds, the 30-second one for short screens.
2. **Choose two or three hooks before choosing the words.** Each should have a number, a constraint and a decision the candidate can defend.
3. **Speak it, do not only write it.** [Grow with Google](https://grow.google/grow-your-career/articles/interview-tips) advises a pitch that is well practised but does not feel scripted. Record it and listen for filler and for any sentence that does not serve one of the three sections.
4. **Check the handoff against the project list.** The project named last should be the one prepared for in the most depth; Part 15 covers the project deep dive.
5. **Adapt it to the interviewer.** A recruiter wants the arc and the fit; an engineer wants the system; a hiring manager wants scope and judgment.

## What the sources show

These points are the author's reading, not claims the sources make.

- **The question is a handoff.** The sources agree it is asked to start a conversation, and that a detail in the answer becomes the next question.
- **The sources disagree on length, and the author picks 90 seconds.** The cited range is 30 seconds to 2 minutes.
- **Evidence beats adjectives.** A named project with a metric gives the interviewer something to ask; "passionate" and "hard-working" give nothing.

## Limits of this summary

- The sources on "tell me about yourself" are career-advice articles, not studies of what makes the opening answer succeed. The structured-interview figures come from selection research and are not about this question specifically.
- All scripts above are for hypothetical candidates with invented details. They are examples of shape, not text to copy.
- [Silvia Sapora's guide](https://silviasapora.github.io/blog/ml-interviews.html) describes the recruiter screen as usually a low-stakes chat; interview practice differs by company and by role, and this series does not claim any company's process.
