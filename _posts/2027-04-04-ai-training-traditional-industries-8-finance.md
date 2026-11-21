---
title: "AI Training for Traditional Industries, Part 8: Finance, Three Ways In"
date: 2027-04-04 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [AI training, upskilling, finance, banking, JPMorganChase, Morgan Stanley, DBS]
description: "Part 8 of a series on training traditional-industry workforces to use AI. Three large financial institutions are read side by side: JPMorganChase (tool first), Morgan Stanley (evaluation first) and DBS (training first), with what each source states and what it leaves out."
---

Banks start from a different place than the other sectors in this series: their staff already work at screens, and the main barrier is control, not access. The three firms below each lead with a different mechanism. One releases a tool and lets demand pull people in, one tests every use case before releasing it, and one trains the whole workforce and builds a career service around it. The sources are company communications, so every figure is the firm's own.

## JPMorganChase: release the tool

Source: the firm's blog post [LLM Suite named 2025 "Innovation of the Year" by American Banker](https://www.jpmorganchase.com/about/technology/blog/llmsuite-ab-award) (3 June 2025).

- LLM Suite, the firm's proprietary generative-AI platform, "was released in summer 2024 to eligible employees across the firm, providing access to large language models (LLMs) in a secure environment".
- "Driven in part by employee demand, the platform went from zero to 200,000 onboarded users within eight months", supporting "idea generation and content drafting".
- Next steps named: connecting more internal data sources and combining generative AI with workflows to build agents. The Chief Analytics Officer calls the target "an AI hub for employees".

The post describes the platform and the user count. It does not describe any training, so nothing in it shows how people learned to use the tool.

## Morgan Stanley: evaluate before release

Source: OpenAI's case study [Morgan Stanley uses AI evals to shape the future of financial services](https://openai.com/index/morgan-stanley/), a vendor-published account, and Morgan Stanley's [press release on the Debrief tool](https://www.morganstanley.com/press-releases/ai-at-morgan-stanley-debrief-launch) (26 June 2024).

- The firm "implemented an evaluation (eval) framework to test every AI use case before deployment". For summarisation, "advisors and prompt engineers graded AI responses for accuracy and coherence", which let the team refine prompts. A translation eval followed for multilingual clients.
- For the meeting-notes tool, the team built evaluation datasets for different meeting types and tested whether it captured action items "without introducing errors". Advisors review and adjust the output before it is finalised.
- Compliance quality is checked "daily" with "a regression suite of sample questions".
- Reported results: "over 98% adoption in wealth management" of advisor teams; document access up from 20% to 80%; the retrieval system went "from being able to answer 7,000 questions to a place where we can now effectively answer any question from a corpus of 100,000 documents".
- The press release carries three advisor testimonials, one of them claiming about half an hour saved per meeting. These are quotes selected by the firm, not measurements.

The source is about the evaluation of the tools. It does not describe courses or training for the advisors. The advisor-graded evals did put users inside the quality loop from the beginning; the source does not say whether the firm treats that as training.

## DBS: train everyone, then redesign roles

Source: DBS's [press release on its agreement with the Institute of Banking and Finance Singapore](https://dbs.com/newsroom/DBS_inks_industrys_first_MOU_with_IBF_to_jointly_uplift_upskill_and_upbuild_Singapores_financial_sector_workforce_for_the_AI_era) (24 August 2026).

- **Breadth.** The release cites DBS's experience "equipping all 40,000 employees globally – including 14,800 in Singapore – with foundational AI skills". Under the agreement, IBF-recognised programmes will cover "AI governance, responsible AI, prompt engineering and practical AI applications".
- **Depth.** DBS reports "upskilling over 11,000 employees in roles significantly reshaped by AI", and will test-bed job-redesign and reskilling interventions with the IBF.
- **Career service.** A new Career Advisory Service starts with "more than 100 Career Advisors from within the bank", combining AI career tools with human guidance, for all employees in its six core markets.
- **Stance.** Head of Group HR: "We see AI as a cognitive enhancer"; "Our philosophy has always been to save people, not jobs".
- **Pipeline.** The bank commits to more than 500 local young-talent places in 2026.

The release does not give course content, hours, assessment, or any result of the training.

![Three cards for JPMorganChase, Morgan Stanley and DBS: the mechanism each leads with and the headline number each reports]({{ site.baseurl }}/assets/pic/ai-training-finance.png){: width="900" }
*The three entry routes and the headline figure each firm reports. Each number measures something different.*

## Reading across

These are the author's observations, not claims from the sources.

- **The three numbers are not comparable.** 200,000 is accounts onboarded, 98% is advisor teams using a tool, and 40,000 is employees given foundational skills. Onboarded, using and trained are three different states.
- **Only DBS describes training as a programme.** The other two describe tools and, at Morgan Stanley, a quality loop. In both cases how people were taught to use the tool is not stated.
- **DBS pairs training with an employment promise.** Its release says AI skills are meant to help people into redesigned roles and states "save people, not jobs". Whether that promise holds is not testable from a press release, but the training is presented as part of it rather than as a separate offer.
- **Control is the shared concern.** Secure environment (JPMorganChase), daily regression tests and review before sending (Morgan Stanley), and AI governance and responsible AI as recognised course topics (DBS) are all control measures. Compare the health sector in Part 5, where a clinical safety officer is the equivalent.

## Limits

- All three sources are company or vendor communications written to promote the programme. None reports an independent evaluation.
- The Morgan Stanley case is published by the technology vendor, and its adoption figure covers advisor teams in wealth management only.
- None of the sources reports learning outcomes, time saved measured across a population, or error rates in production.
- Large global banks are the best-resourced employers in this series; the small-firm picture in Part 4 is different.
