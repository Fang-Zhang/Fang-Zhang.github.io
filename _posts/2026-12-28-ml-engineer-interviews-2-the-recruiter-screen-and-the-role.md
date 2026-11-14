---
title: "ML Engineer Interviews, With Reference Answers (2/18): The Recruiter Screen and the Role"
date: 2026-12-28 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, machine-learning, career, series]
description: "Part 2 of 18. The recruiter screen of an ML engineer interview: ten questions on motivation, role fit, pay, offers and logistics, with reference answers, plus how the roles and company types differ."
---

This post does one thing: it covers the first conversation after the resume screen, the recruiter or phone screen, in an ML engineer interview. It sets out what the screen is for, how the main ML roles and company types differ, and ten questions with a reference answer, a common weak answer and a follow-up for each. [Part 1](/posts/ml-engineer-interviews-1-tell-me-about-yourself/) covered the opening question, "tell me about yourself". The reference answers are the author's, written for hypothetical candidates with invented details; they are not standard answers and not questions from any named company. The main source is [Chip Huyen's Introduction to Machine Learning Interviews](https://huyenchip.com/ml-interviews-book), which is a practitioner's account, not a study.

## What the screen is for

In Huyen's account, the resume screen comes first; if the candidate applied directly, a recruiter screens it and passes it to the hiring manager. The phone screen then has two purposes: to gauge the candidate's interest and expectations, and to sell the candidate on the company. If the interviewer is an engineer, the call may include technical questions that do not involve coding. Usually there is one phone screen, and sometimes two. [Silvia Sapora's guide](https://silviasapora.github.io/blog/ml-interviews.html) describes the recruiter screen in similar terms, as usually a low-stakes chat in which to show that one's skills are relevant to the job. Meta's [interview preparation page](https://www.metacareers.com/ML-prep-onsite) says its full loop consists of up to six 45-minute conversations with engineers, which places the screen as the door to a longer process.

The author's reading: the screen is a filter on three things, whether the candidate is plausibly qualified, whether the candidate wants this role, and whether the practical terms (pay, timing, location) are compatible. A recruiter cannot judge ML depth in a short call, so the questions test whether the story is coherent and the terms are workable.

## The roles, in one table

Questions in the screen depend on which role the candidate is applying for. Huyen separates roles by goal.

![A table of four roles. Data scientist: generate business insights; tends to be the stronger statistician. ML engineer: turn data into products; tends to be the stronger engineer. Research scientist: come up with original ideas; typically a Ph.D. or first-author papers. Research engineer: set up and run experiments; no Ph.D. required.]({{ "/assets/pic/ml-interview-2-roles.png" | relative_url }})
*Four roles, paraphrased from Huyen. The source says titles overlap in practice.*

Details from the source:

- **ML engineer and software engineer.** Huyen writes that ML engineering is considered a subfield of software engineering, and that the hiring process for ML engineers is often spun out of the software engineering process: some organisations swap in a few ML questions, others add an interview focused on ML. Candidates are expected to code. Several hiring managers told her they would rather hire great engineers who know little ML, because it is easier for them to pick up ML than for ML experts to pick up good engineering practice.
- **ML engineer and data scientist.** Huyen gives three reasons for the overlap: machine learning is part of data science, companies often start ML work in the data science team, and many data science tasks such as demand forecasting can be done with ML models. The difference is in the goal, as in the figure.
- **Research scientist and research engineer.** Huyen says that in some teams there is no difference and that the titles are largely a product of bureaucracy.

Company types matter too. Huyen distinguishes applications companies, which solve a specific business problem such as fraud detection, from tooling companies, which build tools others use to build applications. In her account an application is easier to sell early and a tool needs engineering effort to adopt, so tooling companies have higher risk. She also separates enterprise products, sold to organisations, from consumer products, sold to individuals, and suggests different questions to ask about each: for enterprise, the number of customers, the length of the sales cycle and how hard integration is; for consumer, the number of active users, the cost of acquiring a user and how the product makes money. On startups against big companies she lists the usual trade-off: brand, stability and mentoring against autonomy, speed of growth and breadth.

## 1. "Why are you looking to move?"

**What it tests.** Whether the reasons are specific and forward-looking, and whether the candidate speaks badly of the current employer. Huyen says employers want to know why the candidate is looking and what the candidate is looking for, and that career-history answers should cover why the candidate left each company to join the next.

**Reference answer.** State a reason that points toward the new role and is true. Name what the current role cannot give and what this role would. For a hypothetical candidate:

> "I've learned a lot in the current role, and it has been mostly about keeping one ranking model healthy. The team is small and the roadmap is stable, which means that I'm no longer working on new problems. I'm looking for a role where the models change the product more directly, with more experiments per quarter. Your search team runs that kind of work, which is why I applied."

The answer has a fact (what the current role is), a gap (no new problems) and a destination (this role). It does not criticise people.

**Common weak answer.** A complaint about the manager, pay or the company. Even if true, it makes the candidate the story. A second weak form is "I'm just exploring", which tells the recruiter the candidate is unlikely to accept.

**Follow-up.** "What would make you stay in your current role?" The candidate should answer honestly in one sentence and return to the new role.

## 2. "Why this role, and why this company?"

**What it tests.** Preparation, and whether interest is real. Huyen's list of interest questions includes "What interests you about this position?" and observes that companies want employees who are passionate about their work.

**Reference answer.** Give two reasons, one about the problem and one about the way of working, each backed by something the candidate read or used. For example:

> "Two things. The first is the problem: your job posting describes ranking with delayed feedback, which is a harder version of what I do now, and I've read your engineering blog post on how you handle it. The second is how the team works. The post said that every model change goes through an A/B test with a pre-registered metric, which is how I want to work."

The reference to a real source shows that the preparation happened. It is checked easily and cannot be faked in the follow-up.

**Common weak answer.** "It's an industry leader" or "I love AI." Both fit any employer and give the recruiter nothing to verify.

**Follow-up.** "What do you know about our product?" Use it, name a specific feature, and say one thing the candidate would examine from the outside.

## 3. "What is your experience with [NLP, computer vision, recommendation, ...]?"

**What it tests.** Real level of depth in the domain. Huyen says that interviewers test general expertise with questions like "What level of involvement have you had with [ML|computer vision|natural language processing|etc.]?", and her advice is not to bluff: if a candidate pretends to more experience than they have, the interviewer will find out.

**Reference answer.** Answer in three layers: what was built, what role the candidate had, and what was only read about. Mark the boundary clearly.

> "In recommendation I've owned a ranking model end to end for two years: features, training, evaluation and the serving path. In computer vision I've used pretrained models for a classification task and fine-tuned one, but I haven't designed an architecture or worked on detection. In NLP I've read widely and built a retrieval prototype, not a production system."

A candidate who marks the boundary is more credible about the part that is strong.

**Common weak answer.** Claiming everything with one adjective ("extensive"). Huyen's point about the interviewer asking for details means that the claim gets tested, and a claim that does not survive becomes the story.

**Follow-up.** "Walk me through a design decision you made in the ranking model." The recruiter may pass this to the engineer; the candidate should have a decision ready.

## 4. "What are you looking for in your next role?"

**What it tests.** Fit between the candidate's goals and the role. [Modern Descartes](https://www.moderndescartes.com/essays/ml_eng_interviewing) describes a "career chat" in which the candidate discusses the career arc, highlights and goals for the next role, and says it gives signal on ambition, agency and growth potential. Huyen lists "What do you hope to learn in your next job?" among her interest questions.

**Reference answer.** Name two or three things, in order of priority, and make at least one specific to the work.

> "First, problems where the model's output changes a product metric directly. Second, a team with strong evaluation habits, because I learn most from rigorous experiments. Third, a path to leading a small group in a couple of years. I'm flexible on the last one."

Listing the priorities in order lets the recruiter see which ones are firm. Marking the last one flexible is honest.

**Common weak answer.** "Growth and learning" with nothing underneath. It is true of every candidate.

**Follow-up.** "Where do you see yourself in three years?" Answer with a direction (technical depth, or leading a team), not a title.

## 5. "What are your strengths and weaknesses?"

**What it tests.** Self-awareness. Huyen notes that asking directly often yields poor results because candidates are reluctant to admit flaws and too shy to boast, so interviewers reframe the question: "How would your colleagues describe you?", "If you could go back to five years ago, what skill would you learn?", or "Tell me about a time when you faced a seemingly insurmountable challenge and how you overcame it."

**Reference answer.** Answer the reframed form with evidence. For a strength, name a behaviour and what it produced. For a weakness, name a real one, say what it cost, and say what changed.

> "My colleagues would say I'm careful about evaluation: I write the metric and the baseline before the model. The weakness is the reverse side of that: I used to over-invest in evaluation design before getting a first model out, and a project ran two weeks longer than needed. I now time-box the evaluation work and ship a rough baseline first."

The weakness is real, specific to ML work and bounded, and the fix is observable.

**Common weak answer.** "I work too hard" or "I'm a perfectionist". These are strengths in disguise, and interviewers recognise them.

**Follow-up.** "How do you decide how long to spend on evaluation?" The candidate should have a rule, such as a time-box or a threshold for a decision.

## 6. "What are your compensation expectations?"

**What it tests.** Whether expectations are compatible with the role's range, and how the candidate handles negotiation. Huyen describes this as the question candidates dread and gives two approaches. The first is to ask the recruiter for the range first, with a script: express interest, say the candidate would like to learn more about the position to have a realistic expectation, and ask for the compensation range. The drawback is that the negotiation anchors on the recruiter's number, and a low one is awkward to raise. The second is to give a range without committing to it: take the number wanted, add about 20 percent and give it as the lower end. She notes that once a number is given it can only go down.

[Haseeb Qureshi's ten rules](https://haseebq.com/my-ten-rules-for-negotiating-a-job-offer/) argue that information is power and advise not revealing current pay, with exceptions; if it must be given, he advises stating the total value of the package and framing it as looking for a step up.

**Reference answer.** The author's suggestion is to start with the first approach and move to the second only if pressed:

> "I'd like to understand the scope of the role first, and I'm sure we can find something that works if it's a good match. Could you share the range for this level?"

If the recruiter will not share it, give a range with the lower end set at a number the candidate would be content with, and say it depends on the total package. If asked about current pay, give the total package and say the candidate is looking for a step up, as Haseeb Qureshi suggests, or decline politely. Both authors write from the experience of the US technology market, and the rules on what an employer may ask about pay history differ by country and region; candidates should check local rules.

**Common weak answer.** Naming a low number to seem agreeable. It anchors the whole negotiation, as Huyen warns.

**Follow-up.** "Is that base salary or total compensation?" Know the components: base, equity, bonus. Huyen's compensation section lists these as the three direct components of pay in most tech companies: base salary, equity grants and bonuses.

## 7. "Do you have other offers or interviews in progress?"

**What it tests.** Timeline, and honesty. Huyen says every company asks whether the candidate has any impending offer or deadline. She notes that some candidates feel awkward admitting they have none and lie about impending onsites or offers, and writes that she would not want to be the person who lies to get ahead, leaving it as a personal decision.

**Reference answer.** Tell the truth, give the timeline and do not volunteer names.

> "I'm in process with two other companies; one is at the final round and the other at the first technical interview. I don't have an offer or a deadline. I'd like to finish this process on a similar timeline, so if there's a way to schedule the next rounds within three weeks, that would help."

If there are none: "This is one of the first processes I've started, so I have no deadlines, and I can be flexible on scheduling."

**Common weak answer.** An invented competing offer. It is a lie that is hard to maintain and may become a discussion point at the offer stage, when companies can ask for evidence. Sapora reports that several companies asked her to show proof of other offers before raising their own.

**Follow-up.** "Which company is your first choice?" Sapora reports that recruiters read small signals, such as how often a company is mentioned; the candidate can say honestly that this role is a strong match without ranking.

## 8. "When can you start? Where can you work?"

**What it tests.** Practical compatibility: notice period, location, remote or office, work authorisation. These are factual questions. The sources cited here do not set out advice for them, so this is the author's guidance and depends on country and contract.

**Reference answer.** Answer factually and briefly, and give any condition that affects timing.

> "My notice period is one month; I can start about five weeks after signing. I'm based in this city and am happy to work remotely or hybrid. I have the right to work in this country, and I would not need sponsorship."

Check the contract before the call so the notice period is stated correctly. If the candidate needs sponsorship or relocation, say so in the first call, as it changes the recruiter's process and the company's options.

**Common weak answer.** Vagueness ("it depends") on facts the candidate knows. It costs time on both sides.

**Follow-up.** "Is any of it flexible?" Know in advance which constraints are firm.

## 9. "Which kind of role are you after: ML engineer, data scientist or research?"

**What it tests.** Whether the candidate understands the roles and is applying to the right one. The table above is the base for the answer.

**Reference answer.** Say which goal the candidate's work has served, and use the role definitions.

> "ML engineer. My work has been turning models into services: the pipeline, the serving path and the monitoring. I have done analysis work and can design experiments, but the part I enjoy and am good at is getting a model to production and keeping it healthy. That matches your posting, which lists deployment and monitoring."

Huyen's remark that titles overlap suggests also asking the recruiter what the day-to-day work is, since the title may not say.

**Common weak answer.** "Whatever fits", which signals that the candidate has not read the posting.

**Follow-up.** "How much of your time would you want to spend on modelling against infrastructure?" Have a ratio and a reason.

## 10. "Do you have questions for me?"

**What it tests.** Interest and preparation. Huyen says the interview is a two-way street and that candidates can learn about the company from their questions. For the recruiter, the useful ones are about the process. She advises asking, before each interview, what its scope is; if coding is involved, what platform will be used; and for the interview schedule and the names of the interviewers.

**Reference answer.** Ask three or four, in this order:

1. "What are the stages, and how long does each take?"
2. "Will the technical rounds include coding, ML questions or system design, and on what platform?"
3. "Who will I speak to next, and what is their role?"
4. "What does the team work on, and what is the top priority for the person in this role?"

The fourth sets up the later rounds, and its answer is worth noting. Huyen's advice is not to ask interviewers about compensation unless they raise it, as some hiring managers read it as a sign that the candidate only cares about money; she makes that point about interviewers generally, so ask the recruiter about the range only when the recruiter opens the topic or asks about expectations. A longer list of questions for later rounds is in Part 17.

**Common weak answer.** "No, I think you've covered everything." It misses the chance to prepare for the next round and shows little interest.

**Follow-up.** None usually; the recruiter explains the next steps. Write them down.

## How to prepare

1. **Read the posting and write down its three main requirements.** Map one example from the candidate's work to each. This feeds questions 2, 3 and 9.
2. **Write the reason for leaving in one sentence.** It should be forward-looking and true.
3. **Decide the pay answer before the call.** Know the number wanted, the number that would be acceptable, and what counts in the package.
4. **Check the facts.** Notice period, work authorisation and dates. State them correctly.
5. **Prepare the three process questions.** They take two minutes and make the next round easier.

## What the sources show

These points are the author's reading, not claims the sources make.

- **The screen is a filter for coherence and terms.** The recruiter cannot test ML depth, so a clear story and workable terms are what the screen measures.
- **Honesty is the stable strategy.** Huyen describes interviewers checking claims against the resume, and Sapora says candidates should be able to talk about the papers they authored; the advice on offers and experience is the same: state what is true and mark its boundary.
- **Roles are defined by goal, not title.** Ask what the day-to-day work is.

## Limits of this summary

- Huyen's book and the blog posts cited are individual accounts, mostly from the US technology market. They are not studies, and practice differs by company, country and role.
- The pay advice reflects those accounts. It is not legal or financial advice, and the rules on salary history differ by jurisdiction.
- The guidance on notice period, work authorisation and remote work in question 8 is the author's and comes from no source.
- The chapters of the book were read from the project's public repository; this post quotes only short passages and paraphrases the rest.
