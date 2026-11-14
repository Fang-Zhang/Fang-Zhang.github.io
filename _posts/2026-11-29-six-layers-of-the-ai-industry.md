---
title: "Six Layers of the AI Industry, and Who Sits at Each"
date: 2026-11-29 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [ai, industry, semiconductors, cloud, energy]
image:
  path: /assets/pic/ai-six-layers.png
  alt: Stack of five layers from power and cooling up to applications, with capital as a side column
---

One chat query starts in an app, goes to a model, runs in a cloud data center, executes on a chip that a single foundry made with a machine only one company builds, and ends up as electricity drawn from a grid. Most coverage of AI covers the top of that chain. The industry is the whole chain, and each link has its own owners, its own numbers and its own degree of competition.

This post lays out the chain as five layers plus a side column for capital, names who sits in each, and attaches one primary-source number per layer. The layering is a way of organizing the field, not an official taxonomy. Merging cloud and networking into one layer, and treating capital as a column rather than a layer, are my choices.

![Five layers of the AI industry from power and cooling to applications, with capital as a side column]({{ "/assets/pic/ai-six-layers.png" | relative_url }})
_The stack, bottom to top. Each layer sells to the one above it. Capital flows through all of them. Sources are listed under the figure and in the sections below._

## Layer 1: Power and cooling

Everything above runs on electricity. The International Energy Agency's [*Energy and AI*](https://www.iea.org/reports/energy-and-ai/executive-summary) projects data-center consumption of around 945 TWh by 2030, slightly more than Japan's total electricity use today, with the United States the largest source of growth.

Who sits here: generators and utilities, electrical-equipment makers, and cooling suppliers. Two examples with primary documents:

- Constellation signed a [20-year power purchase agreement with Microsoft](https://www.constellationenergy.com/news/2024/Constellation-to-Launch-Crane-Clean-Energy-Center-Restoring-Jobs-and-Carbon-Free-Power-to-The-Grid.html) to restart Three Mile Island Unit 1 as the Crane Clean Energy Center, adding about 835 MW.
- Vertiv, which sells power and thermal equipment to data centers, reported [second-quarter 2026 net sales of $3.27 billion, up 24%](https://investors.vertiv.com/news/news-details/2026/Vertiv-Reports-Strong-Second-Quarter-2026-with-Diluted-EPS-Growth-of-53-Adjusted-Diluted-EPS-Growth-of-60-Raises-Full-Year-2026-Guidance-Across-All-Key-Metrics/default.aspx) and guided full-year 2026 to $13.8–14.2 billion.

## Layer 2: Chips and memory

This layer has three separable parts, and they are concentrated differently.

**Design.** Epoch AI's [AI chip sales database](https://epoch.ai/data/ai-chip-sales) estimates compute in H100-equivalents. Summing its quarterly rows for 2025 gives about 13.6 million H100-equivalents added, split roughly Nvidia 65.5%, Google (TPU) 20.2%, Amazon (Trainium) 5.2%, AMD 5.0%, Huawei 3.9%, Cambricon 0.2%. These are Epoch's estimates built from earnings commentary and analyst figures, and the database covers only those six designers.

**Manufacturing.** [TrendForce](https://www.trendforce.com/presscenter/news/20260909-13225.html) puts TSMC at $40.2 billion and 72.5% of global foundry revenue in the second quarter of 2026; Samsung is second at 5.9% and SMIC third at 5.4%. TSMC's own [management report](https://investor.tsmc.com/chinese/encrypt/files/encrypt_file/reports/2026-07/6f49632674bd2d0fd48cb65aaf89ec6ab510b559/2Q26%20ManagementReport.pdf) shows high-performance computing at 66% of its net revenue that quarter, against 60% a year earlier. Upstream, [ASML's annual report](https://ourbrand.asml.com/m/8ab959d4926657b/original/asml-2025-annual-report-strategic-report-section.pdf) describes ASML as the world's only manufacturer of EUV lithography systems.

**Memory.** High-bandwidth memory is the third bottleneck. [Counterpoint](https://counterpointresearch.com/en/insights/global-dram-and-hbm-market-share) reports second-quarter 2026 HBM revenue shares of SK hynix 50%, Samsung 33% and Micron 18%. A year earlier the split was 64%, 15% and 21%. This is the one layer in the stack where the leader's share is visibly falling.

Nvidia, the largest designer, reported [$89.0 billion of data-center revenue](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2027) for the quarter ended in July 2026, up 117% from a year earlier.

## Layer 3: Cloud and networking

Chips become usable only when someone racks them, cools them and connects them. [Synergy Research](https://www.srgresearch.com/articles/q2-cloud-market-passes-143-billion-highest-growth-rate-in-eight-years) estimated enterprise spending on cloud infrastructure services at $143 billion in the second quarter of 2026, up more than $43 billion year over year. Amazon held 28%, Microsoft 20% and Google 15%. Among smaller providers, Synergy lists CoreWeave, OpenAI, Oracle, Crusoe, Nebius, Anthropic and Nscale as the fastest-growing, and says nine neocloud companies are now in the top 40 providers by cloud-infrastructure revenue. Microsoft says [Azure passed $100 billion in annual revenue](https://www.microsoft.com/en-us/investor/earnings/fy-2026-q4/press-release-webcast) for the first time in fiscal 2026.

Inside the cluster, the network matters as much as the chips. [Dell'Oro](https://www.delloro.com/news/ethernet-extends-lead-in-ai-scale-out-networks-despite-strong-infiniband-rebound) reports that Ethernet made up about two-thirds of AI back-end switch sales in the first quarter of 2026, while InfiniBand sales more than tripled on the ramp of Nvidia's Blackwell Ultra platform. In Ethernet, Celestica ranked first, Nvidia second, Arista third and Cisco fourth.

## Layer 4: Models

The Stanford [AI Index 2026](https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_1_research_development.pdf) counts notable model releases by organization. In 2025 the top industry contributors were OpenAI (20), Google (14) and Alibaba (11). In academia the top contributor was Tsinghua University (26). Industry produced the large majority of notable models.

Model builders also appear in other layers: Google designs chips and runs a cloud, and Anthropic and OpenAI appear in Synergy's list of fast-growing cloud providers. The layers describe functions, and a single company can hold several.

## Layer 5: Applications

The applications layer is where revenue is easiest to see and hardest to compare. Menlo Ventures' [2025 survey](https://menlovc.com/perspective/2025-the-state-of-generative-ai-in) puts enterprise generative-AI spending at $37 billion in 2025, with more than half ($19 billion) going to applications and coding the first clear use case at $4 billion. It reports enterprise LLM API share of 40% for Anthropic and 27% for OpenAI. This is a venture firm's survey, not audited data. Microsoft reported more than 30 million paid Microsoft 365 Copilot seats in its [fiscal 2026 fourth-quarter release](https://www.microsoft.com/en-us/investor/earnings/fy-2026-q4/press-release-webcast).

## The side column: Capital

Capital does not sit in one layer. It pays for all of them, and it has become a link between them.

- Nvidia and OpenAI announced a [letter of intent](https://investor.nvidia.com/news/press-release-details/2025/OpenAI-and-NVIDIA-Announce-Strategic-Partnership-to-Deploy-10-Gigawatts-of-NVIDIA-Systems/default.aspx) for at least 10 gigawatts of Nvidia systems, with Nvidia intending to invest up to $100 billion in OpenAI as systems are deployed. A letter of intent is not a completed investment.
- Alphabet raised its 2026 capital-expenditure guidance to [$195–205 billion](https://abc.xyz/investor/events/event-details/2026/2026-Q2-Earnings-Call-2026-GgTAq7Is0z/default.aspx).
- Anthropic [announced a $65 billion Series H](https://www.anthropic.com/news/series-h) at a $965 billion post-money valuation in May 2026 and said run-rate revenue had passed $47 billion.
- The AI Index [reports](https://hai.stanford.edu/ai-index/2026-ai-index-report) U.S. private AI investment of $285.9 billion in 2025.

## Concentration falls as you go up

Putting the largest player's share of each market side by side gives a pattern.

![Largest player's share in six markets, from EUV lithography to cloud infrastructure]({{ "/assets/pic/ai-concentration.png" | relative_url }})
_Share held by the largest player in each market. The metrics differ (revenue, units, compute added), so each bar should be read on its own terms rather than ranked as a like-for-like series._

My reading, which is an interpretation and not a finding: the markets closest to physical manufacturing are the most concentrated, with one EUV supplier, one dominant foundry and three HBM makers. The markets closer to users are more contested. The HBM shift from 64% to 50% for the leader in one year, and the rise of neoclouds into Synergy's top 40, are the clearest examples of concentration loosening in the middle of the stack.

## What the numbers do not show

- Shares measure revenue or estimated units, not bargaining power. A foundry with 72.5% share may still depend on a few customers.
- The sources differ in kind: company filings (TSMC, Vertiv, Microsoft, Nvidia, ASML), analyst estimates (TrendForce, Counterpoint, Synergy, Dell'Oro, Epoch AI) and a venture-firm survey (Menlo). Analyst figures are revisable.
- This post says nothing about whether the investment in these layers will earn its return. The data above describes who is in each layer, not whether the layer pays.
