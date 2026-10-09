---
title: "Evaluating LLMs: Two Questions, Not One"
date: 2026-10-21 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM, evaluation, evals, machine-learning, reading-list]
description: "Choosing a model and judging your own application are two different evaluation problems. Benchmarks answer the first. Error analysis, graders and a small eval set answer the second."
---

"How do you evaluate an LLM?" sounds like one question. It is two.

- **Question A: which model should I use?** The tools are benchmarks, leaderboards and published comparisons.
- **Question B: is my application good?** The tools are your own data, your own failure cases and graders you have checked yourself.

A high score on a public benchmark answers A, and only partly. It does not answer B. Most of the confusion in this field comes from using the tools of one question to answer the other. This post covers A briefly and spends most of its length on B, because B is where the work is.

## Question A: choosing a model

Benchmarks fall into a few families, and the family tells you what the number can mean.

Knowledge tests such as [MMLU](https://arxiv.org/abs/2009.03300) cover 57 tasks, from elementary mathematics to US history, computer science and law. Code benchmarks run the generated program against tests. [SWE-bench](https://github.com/swe-bench/SWE-bench) gives a model real GitHub issues and checks whether the repository's tests pass. The pattern is that evaluation is most reliable where the answer can be verified mechanically: a test passes, a number matches.

Sebastian Raschka's [Understanding the 4 Main Approaches to LLM Evaluation](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches) is a good map of how scoring works. He covers multiple-choice accuracy, verifiers that check answers, leaderboards built on preferences, and LLM judges, with code for each. The rest of this post leans on the same split: some scores come from a check that can be run, others from a judgment that has to be trusted.

### Why a benchmark number should be read with suspicion

A leaderboard is a measurement, and measurements have error. One documented example comes from Chatbot Arena, where people vote between two anonymous answers. LMSYS, the team behind it, [controlled for answer length and markdown formatting](https://www.lmsys.org/blog/2024-08-28-style-control) and found that the ranking changed. In their words, style has a strong effect, and some models dropped while others rose once style was separated from substance. The vote measured the content and the packaging together.

So a benchmark can screen out models that clearly will not work. It cannot tell you which model fits your task. For that you need your own data.

## Question B: is my application good?

The shift here is from general ability to specific behavior. [Hamel Husain](https://hamel.dev/blog/posts/evals/index.html) opens his essay *Your AI Product Needs Evals* with an observation from his own work: the products that fail share a common root cause, which is the lack of a robust evaluation system. Vibe checks are useful, he says, but not enough.

### Start by reading outputs

Start with the outputs, not with a metric. Husain's advice is to remove all friction from looking at data, and to read as many traces as you can when starting. His line on the subject is that you can never stop looking at data. Over time you can sample more, but the reading does not end.

What you are doing in that reading is **error analysis**: noting what goes wrong, grouping the failures into types, and then deciding which types deserve a test. The list of failure types is the real asset. A metric chosen before reading the outputs tends to measure what was easy to measure.

### Start small

Anthropic's engineering post [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) says teams delay building evals because they think they need hundreds of tasks. Their view is that 20 to 50 simple tasks drawn from real failures is a great start, because early changes have large effects and large effects show up in small samples. Begin with what you already test by hand before each release, and with the bug reports you already have.

### Three kinds of grader

Anthropic's post lays out three types of grader, and the ordering matters:

- **Code-based:** string matches, test suites, checks on the final state. Fast, cheap, objective and reproducible, but brittle against valid variations.
- **Model-based:** an LLM scores the output against a rubric. Flexible, and able to handle open-ended output, but non-deterministic and in need of calibration against humans.
- **Human:** expert review and spot checks. The gold standard for quality, and the way to calibrate the model-based graders. Expensive and slow.

Their recommendation is deterministic graders where possible, LLM graders where necessary, and human graders used judiciously. Husain's framework has the same shape in three levels of cost: assertions on every code change, model and human evaluation on a cadence, and A/B tests only after significant product changes.

Husain's assertion examples are small and concrete. A listing-search feature should return exactly one result when one listing matches. A generic check should confirm that no internal ID leaks into the reply. These are a few lines each. They can run on every change.

### Binary beats a scale

Husain starts by labeling examples as good or bad. He found that scores and granular ratings are more onerous to manage than a binary rating. Eugene Yan makes a similar point in [Task-Specific LLM Evals that Do & Don't Work](https://eugeneyan.com/writing/evals): simplifying an eval to a binary metric makes human labeling and later active learning easier. The pass or fail question forces you to write down what "good" means.

### Balance the set

Anthropic gives an example of what happens without this. If you only test that an agent searches when it should, you may end up with an agent that searches for everything. Their web-search evals covered both directions: queries that need a search and queries that should be answered from existing knowledge. One-sided evals produce one-sided optimization.

### Match the bar to the risk

Yan adds a point that is easy to skip: the evaluation bar depends on the application. A customer-facing medical or financial chatbot needs a higher bar than an internal tool that classifies products. As a data point from his experience, he writes that the typical factual inconsistency or irrelevance rate is 5 to 10 percent even after grounding and good prompting, and that going below 2 percent may be prohibitively hard. Setting a target of perfection for every task is a way to never ship.

## The judge is also a system that needs evaluating

LLM judges are attractive because they scale. The paper that started the practice, [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685), reports that a strong judge such as GPT-4 reached over 80 percent agreement with human preferences, the same level as agreement between two humans. The same paper lists the known biases: position, verbosity and self-enhancement, along with limited reasoning ability.

So a judge is not an oracle. Husain calls model-based evaluation a meta-problem inside the larger problem. His method is low-tech. Have a human grade 25 to 50 examples, write a critique for each, compare them with the judge's verdicts, and revise the judge's prompt until the two agree. Then repeat the check periodically, because agreement drifts. He also warns that raw agreement is misleading when classes are imbalanced, and that precision and recall should be measured separately.

The practical rule: **an automated grader gets a small evaluation of its own, against human labels, before its scores are trusted.** Without that, a dashboard of judge scores is a measurement of the judge.

## Agents: what changes

Anthropic's post is built around the fact that agents make evaluation harder. An agent takes many turns, calls tools and changes the state of its environment, so mistakes can propagate and compound. Several ideas from the post carry over to anything agent-like.

**Grade the outcome, not the path.** They distinguish the transcript, which is everything the agent said and did, from the outcome, which is the final state of the environment. A flight-booking agent may say "your flight has been booked", but the outcome is whether a reservation exists in the database. They also report that checking for a specific sequence of tool calls is too rigid, because agents often find valid approaches the designer did not anticipate.

**Runs vary, so count trials.** Anthropic describes two numbers. pass@k is the chance of at least one success in k attempts. pass^k is the chance that all k attempts succeed. With a 75 percent success rate per trial, three trials all passing happens about 42 percent of the time. For an assistant that people rely on every time, the second number is the honest one.

**A zero can mean a broken eval.** With frontier models, they write, a 0 percent pass rate across many trials is most often a sign of a broken task, not an incapable agent. Their examples are instructive. Opus 4.5 first scored 42 percent on CORE-Bench until a researcher found rigid grading that penalized "96.12" when it expected "96.124991…", ambiguous specifications and unreproducible tasks. After the fixes the score rose to 95 percent. The model had not changed. The measurement had.

**Read the transcripts.** They state that they do not take eval scores at face value until someone has dug into the details and read some transcripts. A failure should look fair: it should be clear what the agent got wrong and why.

**Watch saturation.** An eval at 100 percent tracks regressions but gives no signal for improvement. They note that frontier models now reach above 80 percent on SWE-bench Verified, so the remaining differences are small, and large capability gains look like small score changes. Capability evals that reach a high pass rate can graduate into regression suites.

## After launch

The same material says evaluation is not a gate at the end. Husain treats evaluation, debugging and changing the system as a cycle, and observes that many people work only on the third step, which keeps them at the demo stage. Anthropic describes teams without evals as flying blind when users say the product feels worse, and teams with evals as able to adopt a new model in days instead of weeks. An eval set grows from real failures and keeps growing with them.

## A path for a newcomer

1. Read Raschka's [four approaches](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches) to see what scoring methods exist.
2. Read Husain's [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/index.html), then his [AI Evals FAQ](https://hamel.dev/blog/posts/evals-faq).
3. Read Yan's [Task-Specific LLM Evals that Do & Don't Work](https://eugeneyan.com/writing/evals) for task-by-task metrics.
4. Read Anthropic's [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) for agents and for the roadmap.
5. Read the [MT-Bench paper](https://arxiv.org/abs/2306.05685) before trusting an LLM judge.
6. For a book-length treatment, Chip Huyen's *AI Engineering* (O'Reilly, 2025) has [supporting material](https://github.com/chiphuyen/aie-book).
7. For standard benchmarks in code, [lm-evaluation-harness](https://github.com/eleutherai/lm-evaluation-harness) from EleutherAI is the common tool, and [Inspect](https://inspect.aisi.org.uk/) from the UK AI Security Institute covers agent and safety evaluations.

Then do the exercise that matters: collect 20 real inputs, read the outputs, mark each good or bad, and write one sentence about why. The list of reasons is the beginning of an evaluation.

## What an evaluation is for

A benchmark compares models against each other. A custom eval compares your system against what you want it to do. Each turns an impression into something that can be checked again, and the second is the only one that tells you whether the thing you built works.
