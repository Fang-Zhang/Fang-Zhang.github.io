---
title: "LLMs on the Farm, Part 6: When a Text Message Is the Better Tool"
date: 2027-06-05 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, agriculture, alternatives, sms, extension, trade-offs, llms on the farm]
description: "Part 6 of 6 on large language models in agriculture. Six text-message experiments, a field-day cost, a voice-menu study and one AI trial give the alternatives to a language model. A break-even calculation shows how large an effect an AI service needs to beat a text message on cost, and where each channel fits."
---

## What this post finds

Parts 1 to 5 took the language-model route from model choice to cost. Part 6 sets it against the alternatives that existed before it and still work.

- **Text messages have a measured, modest, cheap effect.** A [meta-analysis of six text-message experiments](https://www.nber.org/system/files/working_papers/w32048/w32048.pdf) in Kenya and Rwanda (maize, 2015 to 2019, administrative input-purchase data for tens of thousands of farmers) found an odds ratio of 1.22 (95 percent interval 1.16 to 1.29) for following the recommendation. The authors call the effects small and say the low cost can still make the programmes cost-effective.
- **The paper puts a number on that cost.** In its cost table, a text-message programme costs $1.50 per farmer who changed (three messages at $0.01 each), a sales incentive from One Acre Fund $1.89, and a Farmer Field Day $45.92. Those are costs per affected farmer for buying a recommended input, one outcome in one setting.
- **Matching the text message takes a large effect from an AI service.** At $0.35 per farmer, an AI service needs about 23 percent of farmers to change to match $1.50 per changed farmer. The text-message programmes imply about 2 percent. The $0.35 is the early estimate from Part 5; at an assumed $1.61 it cannot match $1.50 at all.
- **The comparison reverses against field days and traditional extension.** At $0.35, an AI service matches the $45.92 field day at under 1 percent and the cited $35 traditional extension at 1 percent.
- **Voice menus have a measured weak point.** In a 2009 study of 45 farmers in Gujarat, task completion was 74 percent with keypad input and 61 percent with isolated-word speech input. Speech input fell to 42 percent after a recognition error.
- **The AI trial's effect sits where a message cannot reach.** Part 5's trial found the AI advisor's effects concentrated on crop protection and problem response, the topics where a farmer asks about a specific situation.

![Two panels. Left: cost per farmer who changed behaviour against the share of farmers who change, on log axes; two falling lines for an AI service at 0.35 and 1.61 dollars per farmer cross the text message line at 1.50 dollars at 23 percent and above 100 percent, and cross the 3.50 dollar video line at 10 and 46 percent. Right: in a 2009 study with 45 farmers, task completion was 61 percent with speech input and 74 percent with keypad input]({{ site.baseurl }}/assets/pic/farm-6-alternatives.png){: width="900" }
_Figure: the left panel is arithmetic on the published costs quoted above and the assumed cost from Part 5; the right panel is the published completion rate from the 2009 study._

## The alternatives, with what is known

**Text messages.** The Fabregas, Kremer, Lowes, On and Zane paper pools six experiments run with three organisations, with different content, design and target populations. It reports that framing of the messages did not significantly increase impacts, that repetition had modest positive effects, and that the effects cannot be distinguished across experiments. The control group followed the recommendation at about 13 percent. It is a working paper and measures purchases of recommended inputs, not yield. The same paper's footnote records that some Kenyan services charged under $0.006 per message in 2020.

**Field days and incentives.** For Farmer Field Days the paper uses the organiser's reported marginal cost of $2,600 per event for 100 to 300 farmers; the resulting $45.92 per affected farmer is high because the effect on purchases was small relative to the cost. The One Acre Fund sales incentive was the strongest effect per farmer in the table, at $1.89 per affected farmer, and it is a price mechanism, not information.

**Video extension.** The Farmer.Chat paper cites a study in which video-enabled extension cut cost per adoption from $35 to $3.50 and reached 30 percent more farmers, and notes that producing localised videos is resource-intensive and relies on human facilitators. [J-PAL's summary](https://www.povertyactionlab.org/case-study/video-based-support-small-scale-farmers-around-world) of the Bihar evaluation estimates US$3 to 6 per adoption at scale. Digital Green's chatbot is built on this video library, so the two are complements as much as competitors.

**Voice menus.** The AIEP report says its services use telephone lines on existing IVR infrastructure alongside SMS, apps and messaging platforms, and that combining WhatsApp or an app for smartphone users with a telephone line for feature-phone users was the most common choice for balancing access. A [CHI 2009 study of Avaaj Otalo](https://tap2k.org/papers/note0202-patel.pdf) randomised 45 Gujarati farmers to speech or keypad input: task completion was 74 percent with keypad and 61 percent with speech, and the gap was largest for users under 30 and those with less than eight years of schooling. Users found speaking single-word commands to a machine unnatural, and recovering from recognition errors hard. That study used isolated-word recognition from 2009. The 2026 AIEP services use ASR, translation and speech synthesis built on newer models, and the sources read here do not report a comparison of the two designs.

**Extension agents with a chatbot.** Digital Green first deployed Farmer.Chat as an assistant to extension agents, with a human reviewing answers before they reached farmers, as the OpenAI case study describes. The AIEP report notes that where the main channel includes intermediaries such as extension agents, outreach rules and costs differ from direct messaging.

## The arithmetic

The script converts a cost per farmer into the share of farmers who must change behaviour for an AI service to equal each published cost per changed farmer. It uses the $0.35 published estimate, and $1.61 per farmer-year from Part 5's high-price assumption at 1,000,000 farmers with a $50,000 team. The cost per message and per event are the published ones; the AI costs are not measurements.

```python
# Break-even arithmetic: how large an effect must an AI service have to match published channel costs?
# Published per-"affected farmer" costs (Fabregas et al., Table G1; one affected farmer = one extra farmer using the recommended input)
bench = {"text message (3 msgs at $0.01)": 1.50, "sales incentive (One Acre Fund)": 1.89, "Farmer Field Day": 45.92}
# Per-adoption figures cited in the Farmer.Chat paper for video-enabled and traditional extension
bench.update({"video extension (as cited)": 3.50, "traditional extension (as cited)": 35.00})
# Assumed AI cost per farmer-year from the Part 5 calculation: $0.05 model bill only, $0.35 published estimate, $1.61 high-price model at 1,000,000 farmers with a $50k team
ai = {"AI at $0.35 (published estimate)": 0.35, "AI at $1.61 (assumed high price, 1M farmers)": 1.61}

print("=== Share of farmers who must change behaviour for the AI service to match each benchmark ===")
print(f"{'benchmark':36s}{'cost':>8s}" + f"{'AI at $0.35':>14s}{'AI at $1.61':>14s}")
for b, c in bench.items():
    print(f"{b:36s}${c:7.2f}" + "".join(f"{v/c:>13.1%} " for v in ai.values()))
print()
print("Reference: the text-message programmes imply 0.03 / 1.50 = {:.1%} of farmers affected.".format(0.03/1.50))
print()
print("=== One query against one text message ===")
for name, q in (("low-price query $0.0025", 0.0025), ("high-price query $0.078", 0.078)):
    print(f"{name}: {q/0.01:.2f} text messages at $0.01")
```

Output:

```
=== Share of farmers who must change behaviour for the AI service to match each benchmark ===
benchmark                               cost   AI at $0.35   AI at $1.61
text message (3 msgs at $0.01)      $   1.50        23.3%        107.3% 
sales incentive (One Acre Fund)     $   1.89        18.5%         85.2% 
Farmer Field Day                    $  45.92         0.8%          3.5% 
video extension (as cited)          $   3.50        10.0%         46.0% 
traditional extension (as cited)    $  35.00         1.0%          4.6% 

Reference: the text-message programmes imply 0.03 / 1.50 = 2.0% of farmers affected.

=== One query against one text message ===
low-price query $0.0025: 0.25 text messages at $0.01
high-price query $0.078: 7.80 text messages at $0.01
```

What the output shows, and what it does not:

- **A cost per farmer reached and a cost per farmer changed are different quantities.** A text message costs $0.03 per farmer and changes about 2 percent, so each changed farmer costs $1.50. An AI service at $0.35 per farmer has to change 23 percent of farmers to equal that. Whether it does is not established by any source read for this series.
- **The break-even depends on the outcome.** The $1.50 is for buying a recommended input such as lime, the 2 percent is a probability difference from a pooled meta-analysis, and the AI trial measured practice indices on a different crop and country. The calculation sets published numbers side by side; it is not a like-for-like trial.
- **The cost per query is not the constraint.** One high-price query costs about 8 text messages and one low-price query a quarter of a message (Part 5's assumptions). The expensive part is the team that keeps the content correct (Part 4).

## Choosing between them

- **A recommendation that is the same for many farmers and has a date** (lime before planting, a spray window) fits a text message. The evidence for it is the strongest of any channel here, and its effect is small and cheap.
- **A question about this farmer's field today** (what is this spot on the leaf, what can be sprayed on this crop now) fits a language model with a curated source and a human check. This is where the AI trial's effects appeared.
- **A farmer who does not read** needs voice, and the keypad evidence argues for a menu with a speech fallback and a fast route to a person, until a current comparison exists.
- **An extension system that already has agents and videos** gains most from the chatbot as a tool for the agents, which also keeps a human review step.
- **Nothing here substitutes for inputs.** The trial's authors suggest that on-demand information alone may not transform outcomes without complementary resources, and a price incentive was among the cheapest ways to change purchases in the cost table.

## Closing the series

The six parts read the public evidence in one order: model choice (Part 1), grounding the model in farm knowledge (Part 2), the pipeline around it (Part 3), upkeep (Part 4), cost and benefit (Part 5), alternatives (Part 6). The pattern across all six is that the model is the easy, replaceable part. The curated content, the evaluation set, the usage logs and the channel decide whether the service is correct and whether anyone benefits. The next series applies the same six questions to manufacturing.

## References

- Fabregas, Kremer, Lowes, On and Zane, [Digital Information Provision and Behavior Change: Lessons from Six Experiments in East Africa](https://www.nber.org/system/files/working_papers/w32048/w32048.pdf), NBER Working Paper 32048, January 2024.
- Patel, Agarwal, Rajput, Nanavati, Dave and Parikh, [A Comparative Study of Speech and Dialed Input Voice Interfaces in Rural India](https://tap2k.org/papers/note0202-patel.pdf), CHI 2009.
- Singh, Wang'ombe, Okanga et al., [Farmer.Chat: Scaling AI-Powered Agricultural Services for Smallholder Farmers](https://arxiv.org/abs/2409.08916), 2024.
- AIEP Initiative, [Building AI-based advisory services for smallholder farmers: Technical learnings from the AIEP Initiative](https://arxiv.org/abs/2601.11537), January 2026.
- J-PAL, [Video-based support for small-scale farmers around the world](https://www.povertyactionlab.org/case-study/video-based-support-small-scale-farmers-around-world).
- Aouad, Sungu, Taneri and Ucel, [AI Advisory for Smallholder Farmers: A Randomized Field Experiment in the Philippines and India](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7572761), SSRN working paper, October 2026.
- OpenAI, [Digital Green: Building agricultural database for farmers](https://openai.com/index/digital-green).
