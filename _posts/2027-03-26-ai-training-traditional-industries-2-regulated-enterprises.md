---
title: "AI Training for Traditional Industries, Part 2: Airbus and Roche, Two Ways to Train a Regulated Workforce"
date: 2027-03-26 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [AI training, upskilling, Airbus, Roche, aerospace, pharmaceuticals, workforce]
description: "Part 2 of a series on training traditional-industry workforces to use AI. It compares two large regulated employers from the UK government's case-study set: Airbus, which built capability over a decade, and Roche, which trained 100,000 people in months. Every figure is self-reported by the companies."
---

Large, regulated employers have one advantage in AI training: they can mandate it and fund it. This part reads the two such cases in the UK government's [supporting case studies](https://www.gov.uk/government/publications/skills-for-ai-what-works-for-ai-upskilling-in-the-uk/supporting-case-studies-what-works-for-ai-upskilling-in-the-uk) for *What Works for AI Upskilling in the UK*: Airbus (aerospace and defence manufacturing) and Roche (pharmaceuticals). The first part of the series described the report and its PRIMES design principles; here the point is how two very different timelines ended up with some of the same mechanisms.

One caveat applies to every number below. The report states that case-study outcomes "are based on evidence reported by participating organisations during interviews and validation discussions" and "have not been independently audited or verified as performance data".

## Airbus: a decade, built from data analytics upward

**Starting point.** Airbus operates across aeronautics, defence, space and helicopters. AI had been in its products for over a decade but sat with a small number of specialised technical roles. The company's useful data (supply chains, manufacturing, procurement, delivery scheduling, operational planning) sat in legacy systems that "could only be meaningfully interpreted by people with deep organisational and domain knowledge". The report gives this as the reason external hiring was not enough.

**What it did**, per the case study:

1. **Data analytics first.** It strengthened data analytics and data governance before AI, creating a progression from analytics to data science to AI, and moved existing employees into new roles over time instead of relying mainly on external recruitment.
2. **Learning pathways with cohorts that became mentors.** Formal pathways for data analysts and data scientists, from beginner to advanced. Around 2,000 employees, about 3% of the workforce, completed them. These cohorts later mentored others.
3. **Communities of practice.** Internal AI and analytics communities, regular showcases, mentoring and an internal podcast, *AI for Newbies*, to explain concepts in short formats.
4. **Competencies in the job framework.** AI competencies, including *Generative AI Essentials*, can be added to any role regardless of function. The Airbus Digital Academy produces an annual digital and data competence strategy that HR business partners and line managers use for recruitment, redeployment and development decisions.
5. **Two core pathways:** essentials for employees and managers needing baseline understanding, and practitioner and specialist pathways for advanced technical roles.
6. **A mandatory gate.** "Mandatory training was introduced before employees could access generative AI tools", with data governance covering what data may and may not be shared with AI systems.

**What it reports.** Uptake of pathways, community participation and mandatory training stayed high "with no evidence of drop-off following initial deployment". In a trial with about 2,000 employees, the average user saved up to 4 hours per week when using an AI assistant effectively. The report adds that these savings were seen "primarily in office-based, non-production roles". It also reports increased internal labour mobility as employees acquired data and AI skills. Airbus itself emphasised that the full value is hard to capture in simple financial metrics.

## Roche: 100,000 people in one programme

**Starting point.** Roche reported operating in around 120 countries with over 100,000 employees, about 3,000 of them in the UK. The report names two problems: AI was seen as the domain of data scientists, and non-technical staff saw it as abstract and detached from their roles, while formal education had not caught up so most employees had not met AI during their education.

**What it did:**

1. **One mandatory programme for everyone.** Launched and completed in the second half of 2025, mandatory for all employees, "comparable to compliance or safety training". It was rolled out in phases by department, not by job level, with everyone required to finish within the window. Each employee had a personal dashboard of completed and outstanding modules.
2. **A broad curriculum.** Prompting and use of large language models, organisation-approved tools, bias and limits of AI-generated outputs, and a human-in-the-loop model in which humans stay accountable for decisions.
3. **Short, flexible modules.** Sessions of about 30 to 45 minutes, a mix of live virtual and asynchronous formats, no fixed order, modules revisitable at any time.
4. **Tools alongside training.** Gemini was integrated into the existing Google productivity environment with enterprise access at scale, "reducing the risk that training would remain abstract or disconnected from practice".
5. **Champions inside each function.** Champions came from the function itself and ran voluntary "Everyday AI" sessions. In the Regulatory function a session ran every two weeks, at times suited to Europe and North America, with around 150 people attending each. Peers showed real uses.
6. **Small cohorts to reduce hierarchy.** Roche described groups of about 10 people asking champions questions in chat-based formats, "with groups randomised to reduce hierarchy effects".

**What it reports.** The corporate ambition was at least one hour saved per employee per week. In the Regulatory function, around 50% of employees saved 4 hours or more per week, around 25% about one hour, and around 25% less than one hour. Before the training, savings were "negligible". Savings came mostly from summarising documents, drafting text, preparing presentations and synthesising information. Roche reported that fear of job displacement fell as staff gained hands-on experience.

Roche's data rules show where training meets compliance. Some internal information categories (C1 to C4 in Roche's scheme) could be processed with the approved tools, but GDPR-related personal data could not, and Roche noted that "data is not always neatly organised into GDPR and non-GDPR categories", which needs continuing reinforcement through training. Roche said impact would be assessed function by function in 2026, and listed open questions: onboarding new joiners, updating content for regulatory change, and deeper training on fine-tuning, testing and data governance.

![Left: which design elements the Airbus and Roche case studies state. Right: Roche Regulatory function reported weekly time savings]({{ site.baseurl }}/assets/pic/ai-training-airbus-roche.png){: width="900" }
*Left: design elements each case study states ("not stated" means the case study does not mention it, not that it was absent). Right: Roche's self-reported distribution for one function, ~50% / ~25% / ~25%.*

## Reading across the two

The points below are the author's reading, not statements from the report.

- **Both gate tool access behind training, and both frame it as governance.** Airbus ties mandatory training to access to generative AI tools; Roche presents the programme as compliance-grade. Both put data handling, what may be put into an AI tool, inside the training content.
- **Timelines are not comparable.** Airbus's pathways grew over more than a decade from a data-analytics base. Roche's baseline programme took months and was paired with approved tool access. The Airbus route builds specialists and internal mobility; the Roche route builds a common baseline. They solve different problems, and firms need both layers.
- **The time-saving figures measure different things.** Airbus's is a trial-user average ("up to 4 hours" for effective users, office roles). Roche's is a distribution inside one function. Neither supports a claim about production-floor work, and neither can be added to the other.
- **Peer mechanisms carry the sustain phase.** Airbus's communities and Roche's champions both ran on voluntary participation after a mandatory start. Roche's randomised small groups were designed, in its words, to reduce hierarchy effects on who asks questions; the report gives no result for that mechanism.

## Limits

- The case studies rest on interviews and validation discussions with the organisations. Outcomes are self-reported and unaudited, and the report calls the cases "illustrative rather than representative".
- Both are very large firms with dedicated learning functions. The report says large organisations can invest in structured programmes and internal platforms, while smaller ones rely on focused, use-case-led approaches; later parts cover those.
- The case studies report no comparison group, so the time savings cannot be separated from the effect of having the tools at all (the author's reading).
- This post reads the report only. It did not consult Airbus or Roche's own publications.
