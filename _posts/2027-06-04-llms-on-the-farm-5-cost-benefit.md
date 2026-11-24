---
title: "LLMs on the Farm, Part 5: A Cost per Farmer Is Not a Cost per Result"
date: 2027-06-04 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, agriculture, cost-benefit, rct, extension, llms on the farm]
description: "Part 5 of 6 on large language models in agriculture. The published cost figure for AI farm advice is per farmer reached, the older benchmark is per adoption, and the first randomized trial found changed practices but no significant yield effect. A small calculation shows what each number needs before it can be compared."
---

## What this post finds

Part 4 covered what keeps the service correct. Part 5 asks what it costs and what it returns, using published claims, one pre-registered randomized trial, and a calculation with labelled assumptions.

- **The headline cost claim is per farmer reached.** [OpenAI's case study](https://openai.com/index/digital-green) says early estimates suggest Farmer.Chat can cut the cost of traditional extension 100 times, from $35 per farmer to $0.35 per farmer. The page calls these early estimates and the basis is not given there.
- **The older benchmark is per adoption.** The [Farmer.Chat paper](https://arxiv.org/abs/2409.08916) cites a study in which video-enabled extension cut the cost per adoption from $35 to $3.50. The two $35 figures count different things, so the 100 times and the 10 times cannot be compared directly.
- **A cost per farmer becomes a cost per result only with an adoption rate.** At $0.35 per farmer, the cost per adoption equals the video benchmark of $3.50 only if 10 percent of farmers adopt, and equals $35 at 1 percent.
- **The only randomized trial found changed practices, not yields.** In a pre-registered trial with 1,935 rice farmers in the Philippines and India, an AI advisor on WhatsApp or Messenger increased deliberate practice changes by 0.10 standard deviations, with no statistically significant effect on costs or yields in the one cycle observed.
- **Whether $0.35 is reachable depends on the model bill and team size.** In the calculation below, with assumed token counts and prices, the model bill alone is $0.05 or $1.56 per farmer-year, and a content team of $200,000 a year needs about 668,000 farmers to reach $0.35 in the cheaper case.

![Two panels. Left: cost per farmer-year against farmers served on log axes, for three assumed team costs and two assumed model prices; the low-price curves cross the 0.35 dollar line at some hundreds of thousands of farmers while the high-price curves flatten at about 1.6 dollars and never reach it. Right: cost per adoption against the share of farmers who adopt, for a service costing 0.35 dollars per farmer; it equals 35 dollars at 1 percent adoption and 3.50 dollars at 10 percent]({{ site.baseurl }}/assets/pic/farm-5-cost-benefit.png){: width="900" }
_Figure: both panels are computed by the script at the end of this post. The left panel uses assumed token counts, prices and team costs; the right panel is arithmetic on the two published figures._

## What the sources say about cost

**Claims by the builders.** Beyond the $35 to $0.35 estimate above, the same page says Farmer.Chat reached extension agents first, and that India's agent-to-farmer ratio is about 1 to 650. A [World Bank blog post](https://blogs.worldbank.org/en/agfood/can-ai-give-small-scale-producers-the-right-advice-) from September 2025 reports that Digital Green's chatbot had reached 460,000 farmers and extension workers across five countries, and describes a second service, iSDA's Virtual Agronomist, as serving over 350,000 plots with profit gains of up to 4.7 times and yield increases of 1.6 times. Those are the blog's reported figures for a service, and the blog gives no trial design for them.

**The report that lists what scales.** The AIEP report, in a section on costs at scale, says development may be manageable at first but scaling adds server usage, data storage, API calls and maintenance, that integrations such as weather, market prices and translation can have usage-based costs, and that user acquisition costs must stay sustainable after initial funding ends. It recommends comparing the cost-benefit ratio of channels such as SMS, IVR and WhatsApp. The report does not publish a cost per query or per farmer.

**The older benchmarks.** The Farmer.Chat paper cites an SMS reminder programme in Kenya that raised sugarcane yields by 8 percent and revenue by US$54 per farmer at US$0.02 per message. A randomized evaluation of Digital Green's video model in Bihar, [summarised by J-PAL](https://www.povertyactionlab.org/case-study/video-based-support-small-scale-farmers-around-world), estimated that scaled to all study villages the programme would cost US$3 to 6 per adoption, and the published paper by [Baul and colleagues](https://www.sciencedirect.com/science/article/pii/S0304387824000166) reports Digital Green as about 10 times more cost-effective than training and visit extension at persuading farmers to adopt a new practice. These are the comparisons any AI service has to be set against, and they are per adoption, measured in a trial.

## What the one randomized trial found

An [SSRN working paper by Aouad, Sungu, Taneri and Ucel](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7572761) (October 2026, 53 pages) reports a pre-registered randomized trial with 1,935 smallholder rice farmers in the Philippines and India over one production cycle. Treated farmers received an AI advisor on WhatsApp or Messenger; controls received a similar chatbot without agronomic content, so the trial isolates the information.

- Deliberate practice changes rose by a fifth (0.10 standard deviations), and the quality of farming operations improved (0.08).
- The effects were concentrated on crop protection and problem response, the most frequent topics in the logs. Treated farmers reported more integrated pest management (0.11) and were suggestively more likely to adopt new practices (0.10) and new pesticides (0.11) in those areas.
- There were no statistically significant effects on the adoption of other technologies or inputs, or on costs or yields in that cycle. Interest in dry direct-seeded rice rose (0.12), but adoption did not significantly.
- The authors' reading is that advice helps with things a farmer can adjust with resources at hand, especially time-sensitive problems, while on-demand information alone may not transform outcomes without complementary resources or more proactive engagement.

This is a working paper, not yet a journal article, and it tests one advisor in one cycle. It is also the only randomized evidence of this kind found for this series; the sources read do not show a trial of Farmer.Chat's effect on income or yield. A reported benefit such as the 24 percent income gain on OpenAI's page belongs to Digital Green's video programme, as that page states, not to the chatbot.

## The arithmetic

The calculation has three parts. Part 1 prices a query as four model calls (the minimum for a voice path, per the AIEP report) of 3,000 input and 300 output tokens each, at two assumed price scenarios, with 20 queries per farmer a year. Part 2 adds a fixed yearly team cost divided by the farmers served. Part 3 converts a cost per farmer into a cost per adoption. The token counts, prices, query volume and team costs are assumptions chosen to span a range; no provider's actual price list was used.

```python
# Cost arithmetic for an AI farm-advice service. All inputs are labelled assumptions, not measured prices.
calls_per_query = 4            # voice path needs at least 4 model calls (AIEP report)
in_tok, out_tok = 3000, 300    # assumed tokens per call (retrieved passages + question; short answer)
queries_per_farmer_year = 20   # assumed

print("=== 1. Model bill per query, two assumed price scenarios (USD per million tokens, in/out) ===")
for name, pin, pout in (("low-price scenario", 0.15, 0.60), ("high-price scenario", 5.0, 15.0)):
    q = calls_per_query * (in_tok * pin + out_tok * pout) / 1e6
    print(f"{name:20s} {pin:>5}/{pout:<5} -> ${q:.4f} per query, ${q*queries_per_farmer_year:.3f} per farmer-year")

print()
print("=== 2. Cost per farmer-year = fixed team cost / farmers + model bill ===")
vars_ = {name: calls_per_query * (in_tok * pin + out_tok * pout) / 1e6 * queries_per_farmer_year
         for name, pin, pout in (("low", 0.15, 0.60), ("high", 5.0, 15.0))}
for name, var in vars_.items():
    print(f"-- {name}-price scenario, model bill ${var:.2f} per farmer-year")
    for fixed in (50_000, 200_000, 500_000):
        row = [f"N={n:>9,}: ${fixed/n+var:8.2f}" for n in (1_000, 10_000, 100_000, 1_000_000)]
        print(f"   fixed ${fixed:>7,}/yr  " + "  ".join(row))
print("Farmers needed for the total to fall to $0.35 per farmer:")
for name, var in vars_.items():
    for fixed in (50_000, 200_000, 500_000):
        n = fixed / (0.35 - var) if var < 0.35 else None
        print(f"  {name}-price, fixed ${fixed:>7,}/yr -> " + (f"{n:>10,.0f} farmers" if n else "never (the model bill alone exceeds $0.35)"))

print()
print("=== 3. Units: cost per farmer reached vs cost per adoption ===")
print("Published: video extension cost per adoption $35 -> $3.50 (Farmer.Chat paper); AI estimate $35 -> $0.35 per FARMER (OpenAI page).")
for c in (0.35, 3.50):
    print(f"At ${c:.2f} per farmer, cost per adoption = ${c}/p. p needed to equal $3.50: {c/3.5:.0%}; to equal $35: {c/35:.1%}")
print("Cost per adoption at $0.35 per farmer for adoption rates p:")
for p in (0.01, 0.02, 0.05, 0.10, 0.20):
    print(f"  p={p:>4.0%}: ${0.35/p:6.2f}")
```

Output:

```
=== 1. Model bill per query, two assumed price scenarios (USD per million tokens, in/out) ===
low-price scenario    0.15/0.6   -> $0.0025 per query, $0.050 per farmer-year
high-price scenario    5.0/15.0  -> $0.0780 per query, $1.560 per farmer-year

=== 2. Cost per farmer-year = fixed team cost / farmers + model bill ===
-- low-price scenario, model bill $0.05 per farmer-year
   fixed $ 50,000/yr  N=    1,000: $   50.05  N=   10,000: $    5.05  N=  100,000: $    0.55  N=1,000,000: $    0.10
   fixed $200,000/yr  N=    1,000: $  200.05  N=   10,000: $   20.05  N=  100,000: $    2.05  N=1,000,000: $    0.25
   fixed $500,000/yr  N=    1,000: $  500.05  N=   10,000: $   50.05  N=  100,000: $    5.05  N=1,000,000: $    0.55
-- high-price scenario, model bill $1.56 per farmer-year
   fixed $ 50,000/yr  N=    1,000: $   51.56  N=   10,000: $    6.56  N=  100,000: $    2.06  N=1,000,000: $    1.61
   fixed $200,000/yr  N=    1,000: $  201.56  N=   10,000: $   21.56  N=  100,000: $    3.56  N=1,000,000: $    1.76
   fixed $500,000/yr  N=    1,000: $  501.56  N=   10,000: $   51.56  N=  100,000: $    6.56  N=1,000,000: $    2.06
Farmers needed for the total to fall to $0.35 per farmer:
  low-price, fixed $ 50,000/yr ->    166,889 farmers
  low-price, fixed $200,000/yr ->    667,557 farmers
  low-price, fixed $500,000/yr ->  1,668,892 farmers
  high-price, fixed $ 50,000/yr -> never (the model bill alone exceeds $0.35)
  high-price, fixed $200,000/yr -> never (the model bill alone exceeds $0.35)
  high-price, fixed $500,000/yr -> never (the model bill alone exceeds $0.35)

=== 3. Units: cost per farmer reached vs cost per adoption ===
Published: video extension cost per adoption $35 -> $3.50 (Farmer.Chat paper); AI estimate $35 -> $0.35 per FARMER (OpenAI page).
At $0.35 per farmer, cost per adoption = $0.35/p. p needed to equal $3.50: 10%; to equal $35: 1.0%
At $3.50 per farmer, cost per adoption = $3.5/p. p needed to equal $3.50: 100%; to equal $35: 10.0%
Cost per adoption at $0.35 per farmer for adoption rates p:
  p=  1%: $ 35.00
  p=  2%: $ 17.50
  p=  5%: $  7.00
  p= 10%: $  3.50
  p= 20%: $  1.75
```

What the output shows, and what it does not:

- **The model bill is small, then it is not.** In the low-price scenario the model costs about a quarter of a cent per query and $0.05 per farmer-year; in the high-price scenario $1.56. Only the first is compatible with $0.35 per farmer at any scale. The assumptions, not the sources, drive that result.
- **Fixed costs dominate until the farmer count is large.** With a $200,000 team, the cost per farmer-year is $20 at 10,000 farmers, $2 at 100,000 and $0.25 at 1,000,000 (low-price case). The $0.35 figure is a statement about a service that has already reached hundreds of thousands of farmers.
- **A cost per farmer is not a cost per adoption.** Dividing $0.35 by an adoption rate gives $35 at 1 percent, $7 at 5 percent and $3.50 at 10 percent. No source read here gives the adoption rate of an AI advisory service, so which side of the video benchmark it falls on is not established.
- **A cost per farmer also ignores the benefit.** The trial above found changed practices and no significant yield effect in one cycle; cost per adoption of a practice that does not move yield is a cost per behaviour, not a cost per unit of income gained.

## What a cost-benefit statement needs

1. The unit: per farmer reached, per active farmer, per query, per adoption, or per unit of yield or income.
2. Both cost lines: the model bill and the people who maintain the content (Part 4).
3. A comparison group: a trial arm or a baseline, not a before-and-after reading.
4. The horizon: one cycle, as in the trial, can miss effects that need purchased inputs or a second season.
5. Who is counted: farmers, or extension agents who then serve farmers, as in the first Farmer.Chat deployments.

## What comes next

Part 6 closes the series with alternatives and trade-offs: SMS, voice menus, video, extension agents and plain search, and where a language model is not the right tool.

## References

- OpenAI, [Digital Green: Building agricultural database for farmers](https://openai.com/index/digital-green).
- Singh, Wang'ombe, Okanga et al., [Farmer.Chat: Scaling AI-Powered Agricultural Services for Smallholder Farmers](https://arxiv.org/abs/2409.08916), 2024.
- AIEP Initiative, [Building AI-based advisory services for smallholder farmers: Technical learnings from the AIEP Initiative](https://arxiv.org/abs/2601.11537), January 2026.
- Shah, Collis and Krishnakumari, [Can AI give small scale producers the right advice?](https://blogs.worldbank.org/en/agfood/can-ai-give-small-scale-producers-the-right-advice-), World Bank blog, September 2025.
- Aouad, Sungu, Taneri and Ucel, [AI Advisory for Smallholder Farmers: A Randomized Field Experiment in the Philippines and India](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7572761), SSRN working paper, October 2026.
- J-PAL, [Video-based support for small-scale farmers around the world](https://www.povertyactionlab.org/case-study/video-based-support-small-scale-farmers-around-world).
- Baul et al., [Improving smallholder agriculture via video-based group extension](https://www.sciencedirect.com/science/article/pii/S0304387824000166), Journal of Development Economics, 2024.
