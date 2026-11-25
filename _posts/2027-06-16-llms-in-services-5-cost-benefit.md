---
title: "LLMs in Services, Part 5: The Saving Is Measured in Seconds"
date: 2027-06-16 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, services, cost, roi, customer service, llms in services]
description: "Part 5 of 6 on large language models in service work. Two randomized field experiments at Alibaba measured a drafting assistant that saved 5.7 seconds per chat and a supervised agent that cut chat duration by 3.2 percent while lowering ratings in the chats it handled. The measured saving sets the most a model call may cost."
---

## What this post finds

Cost-benefit claims in customer service usually start from a headline such as a share of chats handled by AI. Two randomized field experiments on Alibaba's Taobao platform measure what actually moved, and the numbers are small, specific and uneven.

- **A drafting assistant saved seconds.** [Ni and colleagues](https://arxiv.org/abs/2603.29888) (2026) randomly gave 5,940 customer-service agents, all with under a year of tenure, access to an assistant that drafts issue diagnoses and solution proposals in the opening stage only. Over about four weeks (23 January to 19 February 2024) the agents handled about 2.56 million chats. Access cut issue identification time by 8.2 percent (3.2 seconds) and chat duration by 1.1 percent (5.7 seconds), lowered the dissatisfaction rate by 0.012 (3.4 percent of the pre-treatment mean) and raised ratings by 0.042 points on a five-point scale (1.2 percent). It had no significant effect on retrials, the paper's measure of whether the problem was actually solved.
- **The average hides who gains.** Lower-performing agents benefited most; top performers saw ratings and retrials worsen, which the authors link to workflow disruption in a setting where agents juggle 2.68 chats at once on average.
- **A supervised agent cut time and lost ratings where it worked.** In a second experiment, [Wang and colleagues](https://arxiv.org/abs/2605.14830) (2026) had human workers supervise an agentic system that resolved the chats eligible for automation, which were under 10 percent of volume. Average chat duration fell 16.8 percent in eligible chats and 3.2 percent overall, retrials barely moved, and customer ratings in eligible chats fell by 0.412 points. Human intervention preserved quality when the AI hit a technical limit, but was less effective when the customer was already frustrated.
- **The ratings gain came from the other chats.** On the chats the AI did not touch, ratings were 0.091 points higher for treated workers, who adapted their multitasking to give those chats more attention.
- **Analysts expect costs to rise.** [Gartner](https://www.gartner.com/en/newsroom/press-releases/2026-01-26-gartner-predicts-genai-cost-per-resolution-for-customer-service-will-exceed-offshore-human-agent-costs-by-2030) predicted in January 2026 that generative AI cost per resolution will exceed 3 dollars by 2030, above many B2C offshore human agents, and that regulation giving customers the right to reach a human will raise assisted volume by 30 percent by 2028. That is an analyst forecast, not a measurement.

![Two panels. Left: customer rating effects of supervised agentic AI at Alibaba, minus 0.412 points on the AI-eligible 9 percent of chats, plus 0.091 on the other chats, and plus 0.044 volume-weighted. Right: the most a chat can spend on the model before the measured time saving is gone, rising linearly with the assumed human cost per chat; a drafting assistant's line stays near a cost of 1 to 2 model calls at 3 dollars per chat, while the supervised agent's line sits well above it]({{ site.baseurl }}/assets/pic/services-5-cost-benefit.png){: width="900" }
_Figure: left, rating effects from the supervised-agent experiment with the volume-weighted average computed here; right, break-even model spend per chat for assumed human costs._

## What the experiments do and do not show

Both are randomized and run on the same platform with the same operator, which is as strong as field evidence gets, and both are one company's after-sales chat in e-commerce, with Alibaba co-authors on both papers. The first sample is new agents only. The second experiment puts a human in the loop on every automated chat, so its time saving is net of supervision. Neither paper reports the cost of running the models, and neither reports labour cost per chat, so the cost side below is assumed.

## What the arithmetic shows

Check 1 tests the second paper's numbers against each other. Check 2 turns the first paper's per-chat savings into minutes per agent. Check 3 asks the cost question the papers leave open: given a measured share of human work removed, what is the most a chat can spend on the model before the saving is gone. The human cost per chat is an assumption; the 3 dollars case is chosen to match the scale of Gartner's figure, and the model cost of 0.012 dollars per call is the illustrative figure used earlier in this series.

```python
# Part 5 of the services series: cost and benefit from two Alibaba field experiments. Standard library only.
print("=== Check 1: consistency of the agentic experiment (Wang et al. 2026) ===")
d_el, d_inel, d_all = 0.168, 0.018, 0.032     # chat-duration reductions: AI-eligible chats, AI-ineligible chats, all chats
s = (d_all - d_inel) / (d_el - d_inel)
print(f"share of chats that are AI-eligible implied by the three duration effects: {s:.1%}   (the paper says: under 10 percent of volume)")
r_el, r_inel = -0.412, +0.091                 # customer-rating effects, points on a five-point scale
print(f"volume-weighted rating effect at that share: {s*r_el + (1-s)*r_inel:+.3f} points (paper: aggregate effect not significant)")
print(f"the rating loss inside the automated slice: {r_el:+.3f} points; the gain among the other {1-s:.0%} of chats: {r_inel:+.3f} points")
print()
print("=== Check 2: how much time the drafting assistant saves (Ni et al. 2026, ITT effects) ===")
chats, agents, days = 2.56e6, 5940, 28
print(f"chats per agent over the experiment: {chats/agents:.0f} (about {days} days)")
for name, sec in (("issue identification", 3.2), ("chat duration", 5.7)):
    print(f"{name}: {sec} s per chat -> {chats/agents*sec/60:.0f} minutes per agent over the experiment")
print(f"mean chat duration implied by '1.1% = 5.7 s': {5.7/0.011:.0f} s; mean concurrency in the data: 2.68 chats at once")
print(f"ratings: +0.042 on a five-point scale, 1.2% of the pre-treatment mean 3.46")
print()
print("=== Check 3: the most a model call may cost per chat before the saving is gone (assumed human cost per chat) ===")
print("break-even AI spend per chat = share of human work removed x human cost per chat")
print("human cost per chat:     $1.00    $3.00    $6.00")
for name, share in (("drafting assistant, 1.1% of duration", 0.011), ("supervised agent, 3.2% of duration", 0.032)):
    print(f"{name:38s}" + "  ".join(f"${share*h:6.3f}" for h in (1, 3, 6)))
print("assumed model cost: 1 to 2 calls at $0.012 each = $0.012 to $0.024; one call per agent message (11.69) = $%.2f" % (11.69*0.012))

```

Output:

```
=== Check 1: consistency of the agentic experiment (Wang et al. 2026) ===
share of chats that are AI-eligible implied by the three duration effects: 9.3%   (the paper says: under 10 percent of volume)
volume-weighted rating effect at that share: +0.044 points (paper: aggregate effect not significant)
the rating loss inside the automated slice: -0.412 points; the gain among the other 91% of chats: +0.091 points

=== Check 2: how much time the drafting assistant saves (Ni et al. 2026, ITT effects) ===
chats per agent over the experiment: 431 (about 28 days)
issue identification: 3.2 s per chat -> 23 minutes per agent over the experiment
chat duration: 5.7 s per chat -> 41 minutes per agent over the experiment
mean chat duration implied by '1.1% = 5.7 s': 518 s; mean concurrency in the data: 2.68 chats at once
ratings: +0.042 on a five-point scale, 1.2% of the pre-treatment mean 3.46

=== Check 3: the most a model call may cost per chat before the saving is gone (assumed human cost per chat) ===
break-even AI spend per chat = share of human work removed x human cost per chat
human cost per chat:     $1.00    $3.00    $6.00
drafting assistant, 1.1% of duration  $ 0.011  $ 0.033  $ 0.066
supervised agent, 3.2% of duration    $ 0.032  $ 0.096  $ 0.192
assumed model cost: 1 to 2 calls at $0.012 each = $0.012 to $0.024; one call per agent message (11.69) = $0.14
```

- **The reported effects are consistent with each other.** The duration effects imply that 9.3 percent of chats were AI-eligible, in line with the paper's "under 10 percent". At that share the volume-weighted rating effect is +0.044, so the large loss inside the automated slice is offset in the average by the gain among the rest, and the paper reports the aggregate as not significant.
- **The assistant's saving is about 41 minutes per agent over four weeks.** That is 431 chats times 5.7 seconds, and it is an upper bound for labour saved, since agents work on 2.68 chats at once and the seconds partly overlap.
- **A ceiling on model spend follows.** At an assumed 3 dollars of human cost per chat, the drafting assistant removes 1.1 percent of the work, worth 3.3 cents. One or two calls at 1.2 cents fit under it; one call per agent message (11.69 per chat), about 14 cents, does not. The supervised agent removes 3.2 percent, worth 9.6 cents at the same assumption, and still falls short of the per-message figure.
- **Limits.** Human cost per chat, model price and number of calls per chat are assumptions. The volume-weighted average uses the share implied by the duration effects. Time saved per chat is an average over all chats, and the papers do not show it converting into fewer hired agents.

## What this means for cost and benefit

1. **Count the saving in the unit the experiment measured.** Seconds per chat, retrials and ratings are measured; "share automated" is a design choice.
2. **Put the model spend under a ceiling set by the measured saving.** The break-even is the share of work removed times the cost of the work, which for a drafting assistant is a few cents per chat.
3. **Report resolution and rating separately.** Both experiments found that speed and ratings can move while resolution does not, so one metric cannot stand for the others.
4. **Price the slice and the spillover.** The automated slice lost ratings while the other 91 percent gained, so the average hid both.
5. **Keep humans in the cost model.** Supervision is inside the measured saving, and Gartner's forecast of more assisted volume points the same way.

## What comes next

Part 6 closes the series with alternatives and trade-offs: what simpler systems do on the same tasks, and when the model is the wrong tool.

## References

- Ni, Wang, Feng, Lu, Wang and Zhou, [Generative AI in Action: Field Experimental Evidence from Alibaba's Customer Service Operations](https://arxiv.org/abs/2603.29888), 2026.
- Wang, Zhu, Feng, Lu and Jia, [Agentic AI and Human-in-the-Loop Interventions: Field Experimental Evidence from Alibaba's Customer Service Operations](https://arxiv.org/abs/2605.14830), 2026.
- Gartner, [Gartner Predicts GenAI Cost Per Resolution for Customer Service Will Exceed Offshore Human Agent Costs by 2030](https://www.gartner.com/en/newsroom/press-releases/2026-01-26-gartner-predicts-genai-cost-per-resolution-for-customer-service-will-exceed-offshore-human-agent-costs-by-2030), press release, 26 January 2026.
