---
title: "LLMs on the Factory Floor, Part 5: The Token Bill Is the Small Number"
date: 2027-06-10 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, manufacturing, cost, roi, maintenance, llms on the factory floor]
description: "Part 5 of 6 on large language models in manufacturing. At plant scale the model's own cost is a rounding error next to the time spent checking answers, while a dedicated GPU is the wrong buy below about 14,000 queries a day. A worked cost model with labelled assumptions."
---

## What this post finds

Parts 1 to 4 chose, adapted, deployed and monitored an assistant. Part 5 asks whether it pays. The public evidence on plant-level returns is thin, so the post combines three sources with a transparent cost model whose inputs are labelled as assumptions, and shows which inputs decide the outcome.

- **Measured productivity gains exist, but not from plants.** [Brynjolfsson, Li and Raymond](https://www.nber.org/papers/w31161) studied 5,179 customer-support agents given a generative-AI assistant and found a 14 percent average gain in issues resolved per hour, 34 percent for novice and low-skilled workers, and minimal impact for experienced ones (published in the Quarterly Journal of Economics, 2025). A Siemens [press release](https://press.siemens.com/global/en/pressrelease/siemens-expands-industrial-copilot-new-generative-ai-powered-maintenance-offering) says first pilot use cases of its maintenance copilot saved on average 25 percent of reactive maintenance time; it is a vendor statement, and the release gives no sample or method.
- **Self-hosting cost is set by load.** [Patil](https://arxiv.org/abs/2606.11690) (a 2026 preprint) benchmarked vLLM on H100 GPUs and reports effective cost between $0.21 and $15.25 per million output tokens on identical hardware, depending on offered request rate. Typical calculators assume the GPU is busy; at 1 to 10 requests per second the paper measures a 2.5 to 24 times underestimate.
- **Token prices keep falling.** [Epoch AI](https://epoch.ai/data-insights/llm-inference-price-trends) finds the price of reaching a fixed benchmark performance fell between 9 and 900 times per year, 40 times per year for GPT-4-level PhD-science questions, and notes that the fastest drops are recent and may not persist.
- **A plant's volume is small.** Hundreds of queries a day is 0.002 requests per second, hundreds of times below the 1 to 10 requests per second in the cost paper.
- **Checking time decides the outcome.** In the model below, the per-answer cost is dominated by the person checking it, so the price of tokens barely moves the result.

![Two panels. Left: generation cost per query against queries per day on log axes; a rented H100 running around the clock falls from about 8 dollars per query at 20 queries a day to 1 cent at 14,000, where it equals an assumed API cost of 1.2 cents per query. Right: queries per day needed to repay a GPU plus upkeep against the share of right answers, for checking times of 1, 3 and 5 minutes; longer checking pushes the curves up and to the right]({{ site.baseurl }}/assets/pic/factory-5-cost-benefit.png){: width="900" }
_Figure: both panels computed by the script below. Prices, token counts, wage, handling time and upkeep are assumptions; the GPU price comes from Patil._

## What the sources say, and what they do not

The support-agent study is the strongest causal evidence available: staggered roll-out, thousands of workers, measured output. Its heterogeneity matters most for a plant. The gain came mostly from less experienced workers, and the authors give suggestive evidence that the tool spread the practices of the best agents. A maintenance team that is already experienced should expect a smaller effect. Customer support is not maintenance, and the paper does not claim otherwise.

The 25 percent figure is a vendor-reported average from early pilots, so it is an upper-range hypothesis to test, not a planning number.

Patil's cost study measures one thing well. On a rented H100 at $6.98 per GPU-hour (the paper's rate, assuming a co-tenant uses the other GPU in the node; a dedicated node doubles it), cost per token is dominated by how busy the GPU is. The study is a single-author preprint measuring output tokens on one cloud SKU; it does not cover input tokens, which dominate a retrieval prompt.

## What the cost model shows

The script uses the paper's GPU price and, otherwise, round assumptions: a technician costs $60 an hour and takes 10 minutes to find an answer unaided, the assistant answers with 4,000 input and 400 output tokens at $2 and $10 per million, and yearly upkeep (re-running the Part 4 regression set, updating the index) costs $20,000. A wrong answer falls back to doing it by hand, and every answer is checked. The saving per query is accuracy times the manual cost, minus generation and checking.

```python
# Plant-scale cost and benefit of a maintenance assistant. Every input below is an assumption unless marked (paper).
H_MIN, WAGE = 10.0, 60.0                  # assumed: minutes to find an answer unaided; loaded cost of a technician, $/hour
H = H_MIN / 60 * WAGE                     # $ per query answered by hand
GPU_HOUR = 6.98                           # (paper, Patil 2026) cloud H100 price per GPU-hour
GPU_YEAR = GPU_HOUR * 24 * 365
P_OUT, P_IN, T_OUT, T_IN = 10.0, 2.0, 400, 4000   # assumed API prices $/M tokens and tokens per answer (retrieved context included)
G_API = (P_OUT * T_OUT + P_IN * T_IN) / 1e6
UPKEEP = 20_000.0                         # assumed yearly people cost to run the regression set and index (Part 4)

print(f"manual cost per query H = ${H:.2f}; API generation cost per answer G = ${G_API:.3f}; one rented H100 around the clock = ${GPU_YEAR:,.0f}/year")
print()
print("=== Check 1: a dedicated GPU against queries per day ===")
for q in (50, 200, 1000, 5000, 20000):
    print(f"{q:>6} queries/day: GPU cost per query ${GPU_YEAR/(q*365):7.2f}   (API: ${G_API:.3f})   offered rate {q/86400:.4f} requests/s")
bq = GPU_YEAR / (365 * G_API)
print(f"daily queries at which a rented GPU costs the same per query as the API: {bq:,.0f}  ({bq/86400:.2f} requests/s)")

print()
print("=== Check 2: queries per day needed to repay the fixed cost ===")
print("saving per query = a*H - G - V   (a = share of answers that are right and used; V = checking cost; wrong answers fall back to doing it by hand)")
print("break-even queries/day = fixed cost / (365 * saving per query); fixed cost = upkeep, plus a rented GPU for self-hosting")
for V_MIN in (1, 3, 5):
    V = V_MIN / 60 * WAGE
    print(f"\nchecking time {V_MIN} min = ${V:.2f} per answer; the assistant can only pay if a > {(V+G_API)/H:.0%}")
    print(f"{'a':>6} {'saving/query':>14} {'API: queries/day':>18} {'self-hosted: queries/day':>26}")
    for a in (0.5, 0.7, 0.9):
        s = a * H - G_API - V
        if s <= 0: print(f"{a:>6.0%} {s:>14.2f} {'never':>18} {'never':>26}"); continue
        print(f"{a:>6.0%} {s:>14.2f} {UPKEEP/(365*s):>18,.0f} {(UPKEEP+GPU_YEAR)/(365*s):>26,.0f}")

print()
print("=== Check 3: how much does a falling token price matter? ===")
V = 3 / 60 * WAGE
print(f"API generation is {G_API/(G_API+V):.2%} of the per-answer cost at a 3-minute check; a 10x yearly price fall moves that to {G_API/10/(G_API/10+V):.3%}")
print(f"cutting checking time from 3 minutes to 2 changes the per-answer cost by {(1/60*WAGE)/(G_API+V):.0%}")

```

Output:

```
manual cost per query H = $10.00; API generation cost per answer G = $0.012; one rented H100 around the clock = $61,145/year

=== Check 1: a dedicated GPU against queries per day ===
    50 queries/day: GPU cost per query $   3.35   (API: $0.012)   offered rate 0.0006 requests/s
   200 queries/day: GPU cost per query $   0.84   (API: $0.012)   offered rate 0.0023 requests/s
  1000 queries/day: GPU cost per query $   0.17   (API: $0.012)   offered rate 0.0116 requests/s
  5000 queries/day: GPU cost per query $   0.03   (API: $0.012)   offered rate 0.0579 requests/s
 20000 queries/day: GPU cost per query $   0.01   (API: $0.012)   offered rate 0.2315 requests/s
daily queries at which a rented GPU costs the same per query as the API: 13,960  (0.16 requests/s)

=== Check 2: queries per day needed to repay the fixed cost ===
saving per query = a*H - G - V   (a = share of answers that are right and used; V = checking cost; wrong answers fall back to doing it by hand)
break-even queries/day = fixed cost / (365 * saving per query); fixed cost = upkeep, plus a rented GPU for self-hosting

checking time 1 min = $1.00 per answer; the assistant can only pay if a > 10%
     a   saving/query   API: queries/day   self-hosted: queries/day
   50%           3.99                 14                         56
   70%           5.99                  9                         37
   90%           7.99                  7                         28

checking time 3 min = $3.00 per answer; the assistant can only pay if a > 30%
     a   saving/query   API: queries/day   self-hosted: queries/day
   50%           1.99                 28                        112
   70%           3.99                 14                         56
   90%           5.99                  9                         37

checking time 5 min = $5.00 per answer; the assistant can only pay if a > 50%
     a   saving/query   API: queries/day   self-hosted: queries/day
   50%          -0.01              never                      never
   70%           1.99                 28                        112
   90%           3.99                 14                         56

=== Check 3: how much does a falling token price matter? ===
API generation is 0.40% of the per-answer cost at a 3-minute check; a 10x yearly price fall moves that to 0.040%
cutting checking time from 3 minutes to 2 changes the per-answer cost by 33%
```

- **The model call is cheap.** Generation costs $0.012 per answer against a manual cost of $10. At a three-minute check it is 0.40 percent of the per-answer cost, and a tenfold yearly price fall would take it to 0.04 percent. Cutting the check from three minutes to two changes the per-answer cost by 33 percent, so checking time is worth more than any price decline.
- **A dedicated GPU is the wrong buy at plant volume.** A rented H100 around the clock is $61,145 a year. At 200 queries a day that is $0.84 per query, 70 times the API call, and the two cost the same only at about 14,000 queries a day. A plant would reach that only by serving many sites or by running other work on the same hardware. Local deployment has other justifications (Part 3's data-control case) but price at this volume is not one of them.
- **Break-even volume is low when accuracy clears the checking bar.** With a one-minute check and 70 percent useful answers, an API setup repays $20,000 of upkeep at 9 queries a day, and a rented GPU at 37. With a five-minute check and 50 percent accuracy the saving per query is below zero and nothing repays it, which is Part 2's rule that the accuracy bar rises with checking cost.
- **Limits.** The model counts per-query savings and fixed upkeep. It leaves out build and integration cost, adoption, and benefits such as fewer repeat faults, and it treats all queries as worth the same ten minutes. The 10 minutes, $60 and 20,000 dollars are inputs, and the conclusion that checking time dominates holds only while the manual task takes minutes and the model's cost stays pennies per answer.

## What this means for the decision

1. **Measure the manual baseline before anything else.** Time to find an answer, by experience level. The only measured effect in the evidence base is concentrated among novices.
2. **Count checking time as the main cost.** It is the largest line and the one a design can lower, with citations to the source page, short answers and a quick path to the original record.
3. **Default to a hosted model at plant volume, and justify a local one on control, not price.** Re-check the arithmetic if volume rises by two orders of magnitude.
4. **Do not plan on token-price declines for the savings.** They are real, but the token bill is already under one percent of the cost per answer.
5. **Treat vendor percentages as hypotheses.** Run a pilot against a control group of technicians and report the effect by experience level, as the support-agent study did.
6. **Budget upkeep explicitly.** Part 4's regression set and index maintenance are a yearly cost, and in the model above break-even volume scales in proportion to it.

## What comes next

Part 6 closes the series with alternatives and trade-offs: when a classical model, a rule system or a search index is the better answer than a language model.

## References

- Brynjolfsson, Li and Raymond, [Generative AI at Work](https://www.nber.org/papers/w31161), Quarterly Journal of Economics 140(2), 2025.
- Patil, [Beyond Per-Token Pricing: A Concurrency-Aware Methodology for LLM Infrastructure Cost Estimation](https://arxiv.org/abs/2606.11690), 2026 (preprint).
- Cottier, Snodin, Owen and Adamczewski, [LLM inference prices have fallen rapidly but unequally across tasks](https://epoch.ai/data-insights/llm-inference-price-trends), Epoch AI, 2025.
- Siemens, [Siemens expands Industrial Copilot with New generative AI-powered Maintenance Offering](https://press.siemens.com/global/en/pressrelease/siemens-expands-industrial-copilot-new-generative-ai-powered-maintenance-offering), press release, March 2025.
