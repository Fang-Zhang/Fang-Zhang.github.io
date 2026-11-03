---
title: "When Effort Stops Being a Signal: AI and the Cost of Looking Busy"
date: 2026-12-12 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [ai, signaling, organizations, economics]
image:
  path: /assets/pic/signal-cost-gap.png
  alt: Two panels showing the cost of writing a long report for two worker types, before and after AI
---

A weekly report arrives. It runs twenty pages, with charts, a methods section and a list of risks. Next to it sits a seven-line note that says what changed, what to do, and how sure the author is. The reader has time for one.

Before AI, the twenty pages carried information that had nothing to do with their content: someone had spent two days on them. That spending was legible. It said the author was diligent, the topic mattered, and the work had been checked. When a draft takes two minutes to produce, the information is gone. What remains is a reader who must now judge the content directly.

## The mechanism: a signal must cost the wrong people more

Michael Spence's [*Job Market Signaling*](https://www.jstor.org/stable/1882010) (Quarterly Journal of Economics, 1973) set out the condition. An observable action works as a signal only if it is cheaper for the type you want to identify than for the type you don't. A degree separates workers if finishing it is easier for the more capable. If it costs everyone the same, it separates no one.

Applied to writing, the logic is the same. A long, polished, tailored document used to be costly to produce, and costlier for someone who did not understand the problem. Readers used it as a proxy for quality. The figure below is a schematic of that condition, not data.

![Cost of producing a long report for two types of worker, before and after AI]({{ "/assets/pic/signal-cost-gap.png" | relative_url }})
_Schematic. Before: the careless worker pays much more per page, so length separates the types. After: both costs are near zero, so length no longer separates them._

The step from job applications to internal reports is an analogy, and I flag it as one. The mechanism is the same; the evidence below comes from hiring, not from reporting lines.

## Why organizations lean on effort signals

Three conditions make effort a tempting proxy inside firms.

- **Output is hard to attribute.** In white-collar work the contribution of one person to a result is entangled with many others, and the counterfactual cannot be rerun. Effort is what is left to observe.
- **Principals cannot watch agents.** The principal–agent problem, set out in Jensen and Meckling's [*Theory of the Firm*](https://doi.org/10.1016/0304-405X%2876%2990026-X) (Journal of Financial Economics, 1976), is exactly this: the owner cannot observe what the manager does, so incentives and monitoring fill the gap, and both are imperfect.
- **Work is handed across levels.** Every handoff is a point where someone must show that they did their part. David Graeber's [*Bullshit Jobs*](https://www.simonandschuster.com/books/Bullshit-Jobs/David-Graeber/9781501143335) describes the resulting layers of paperwork. It is a description of the phenomenon, not a measurement of it.

![Chain of handoffs from analyst to VP, each producing a new document]({{ "/assets/pic/handoff-chain.png" | relative_url }})
_Schematic. Each level writes a document for the next level. Only the last step makes a decision._

## The evidence: when writing got cheap, writing stopped predicting outcomes

Two recent working papers study this directly, both using data from Freelancer.com, an online labour platform. Both are preprints and have not been through peer review.

**Cui, Dias and Ye** ([*Signaling in the Age of AI: Evidence from Cover Letters*](https://arxiv.org/abs/2509.25054)) study the platform's launch of an "AI Bid Writer" on 19 April 2023. They report three findings:

- Workers with access to the tool wrote cover letters that matched the job post more closely, and received more callbacks. The effect was 3.56 percentage points for users, against a baseline callback rate of 7.02%, a rise of about 51%.
- After the launch, the correlation between how closely a letter was tailored to the job and whether it received a callback fell by about 51%.
- Employers shifted toward other signals, including workers' prior work histories.

The two 51% figures are different quantities. One is the callback gain for people who used the tool. The other is the drop in how well tailoring predicts a callback.

**Galdin and Silbert** ([*Making Talk Cheap: Generative AI and Labor Market Signaling*](https://arxiv.org/abs/2511.08785)) use an LLM to score how tailored each application is. They find that, before LLMs, employers were willing to pay more for tailored applications, and after LLMs they were not. They then estimate a model of signaling and simulate a market where written applications carry no signal at all. In that simulation, workers in the top fifth of the ability distribution are hired 19% less often, and workers in the bottom fifth 14% more often. The market becomes less meritocratic.

![Model counterfactual: hiring rates change by -19% for the top quintile and +14% for the bottom quintile]({{ "/assets/pic/signal-counterfactual.png" | relative_url }})
_The numbers come from the authors' estimated model, not from observed outcomes._

Two limits apply. Both studies cover one platform, with short-term remote jobs, where the employer sees a text and little else. And both are about hiring; neither measures what happens to internal reports.

## My reading (an inference, not a finding)

The papers show that a signal built on cheap-to-produce text loses value when production gets cheap, and that employers reach for other signals. Taking that to organizations, I would expect the following.

1. **AI multiplies material at the handoffs, not decisions.** A decision is still made once. The summaries, status documents and decks around it can each be generated at almost no cost. So the volume of fake-looking work rises first, and the volume of useful judgment does not.
2. **Readers will shift to signals that are harder to produce.** The platform data suggests work history. In a firm, the candidates are commitments that can be checked later, track records, and stated confidence that the author is willing to be held to.
3. **The new signals can be forged too.** This has not been studied, and I have no evidence for how fast it happens.

## What the sources do not settle

- Whether internal reports lose signal value the way cover letters did. No study I found measures that.
- Whether the 51% and 19%/14% figures hold outside Freelancer.com.
- Whether employers' move to alternative signals improves hiring or only changes who is hired.
- How long the transition takes, and whether readers adapt faster than writers do.

## Sources

- Spence, M. (1973). [Job Market Signaling](https://www.jstor.org/stable/1882010). QJE 87(3), 355–374. Also his [Nobel lecture](https://www.nobelprize.org/prizes/economic-sciences/2001/spence/lecture/).
- Jensen, M. and Meckling, W. (1976). [Theory of the Firm](https://doi.org/10.1016/0304-405X%2876%2990026-X). Journal of Financial Economics.
- Graeber, D. (2013). [On the Phenomenon of Bullshit Jobs](https://strikemag.org/bullshit-jobs/). STRIKE! Magazine; and the [2018 book](https://www.simonandschuster.com/books/Bullshit-Jobs/David-Graeber/9781501143335).
- Cui, J., Dias, G., Ye, J. (2025). [Signaling in the Age of AI](https://arxiv.org/abs/2509.25054).
- Galdin, A. and Silbert, J. (2025). [Making Talk Cheap](https://arxiv.org/abs/2511.08785).
