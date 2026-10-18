---
title: Reading List
icon: fas fa-book
order: 5
---

## Books I want to read

- **Reinforcement Learning from Human Feedback** — Manning (via [AI Book Club](https://theaibookclub.github.io/))
- **AI Product Manager's Handbook** (2nd ed.) — Irene Bratsis, Packt (building/scaling AI products, AI-native vs. evolving products, commercialization)
- **Machine Learning Platform Engineering** — Benjamin Tan Wei Hao, Shanoop Padmanabhan, Varun Mallya, Manning (build an MLOps/LLMOps platform from scratch: Kubeflow, MLflow, BentoML, Feast, model serving & monitoring)
- **Machine Learning System Design** — Valerii Babushkin, Arseny Kravchenko, Manning (end-to-end ML system framework: problem framing, dataset gathering, training pipelines, serving & monitoring)
- **Knowledge Graphs and LLMs in Action** — Giuseppe Futia, Vlastimil Kus, Manning (building knowledge graphs from structured/unstructured sources, integrating with LLM apps & RAG pipelines)
- **Graph Neural Networks in Action** — Keita Broadwater, Manning (building GNNs in Python for node prediction, link prediction, graph classification)
- **Introduction to Machine Learning** — Laurent Younes, arXiv textbook (mathematical foundations of ML: linear algebra/probability, kernel methods, supervised & generative learning, generalization theory)
- **The Little Book of Deep Learning** — François Fleuret (compact free ebook: gradient descent, backprop, model components, architectures, applications)
- **[Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory](https://arxiv.org/abs/2310.20360)** — Kuckuck et al., arXiv (737pp: ANN architectures, optimization theory, approximation/generalization theory, deep learning for PDEs)
- **LLM Customization and Fine-Tuning** — Amit Bahree, Weehyong Tok, Manning (adaptation spectrum from prompting/RAG through LoRA/QLoRA, full SFT, distillation, and DPO alignment; production ops for drift & safety)
- **[A First Course in Causal Inference](https://arxiv.org/abs/2305.18793)** — Peng Ding, UC Berkeley lecture notes (causal inference from basic probability, statistical inference, linear/logistic regression)
- **AI Agents and Applications: With LangChain, LangGraph, and MCP** — Roberto Infante, Manning (build LLM-powered agentic applications: agent workflows, tools, MCP integrations)
- **Multi-Agent AI Engineering: Design, build, and operate AI systems that think and act as coordinated teams** — Dr. Xiao Ma, Dr. Chi Wang, Packt (production-grade multi-agent systems: communication protocols, memory/context, orchestration, evaluation, security, observability)
- **AI Agents in Action** (2nd ed.) — Micheal Lanham, Manning (autonomous agent design/deployment, MCP tools/memory, reasoning & planning patterns — ReAct, Reflexion, Tree-of-Thought, multi-agent patterns)

## AI Plays Games: From Atari to Agents

A study track on how machines learn to play video games, from reinforcement learning on raw pixels to LLM agents that write their own skills.

**Foundations**

- **[Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)** (2nd ed.) — Richard Sutton, Andrew Barto, MIT Press, 2018 (the standard RL textbook; the authors offer a free PDF, linked here via a Stanford course copy)

**Reinforcement learning from pixels**

- **[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)** — Mnih et al., DeepMind, 2013 (the starting point: one network learns Atari games from raw pixels)
- **[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)** — Mnih et al., Nature, 2015 (the DQN paper)
- **[Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)** — Silver et al., Nature, 2016 (AlphaGo)
- **[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)** — Schrittwieser et al., 2019 (MuZero: plans with a model it learns itself, no rules given)

**Large-scale competitive play**

- **[Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z)** — Vinyals et al., Nature, 2019 (AlphaStar)
- **[Dota 2 with Large Scale Deep Reinforcement Learning](https://arxiv.org/abs/1912.06680)** — OpenAI, 2019 (OpenAI Five)

**LLM agents and social games**

- **[Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)** — Wang et al., 2023 (Minecraft agent that builds a reusable skill library as code)
- **[Human-level play in the game of Diplomacy by combining language models with strategic reasoning](https://pubmed.ncbi.nlm.nih.gov/36413172)** — Meta FAIR Diplomacy Team, Science, 2022 (Cicero: negotiates in natural language)

## Articles

- **[Extending Raschka's GPT-2: an MoE trained from scratch on an RTX 3090](https://www.gilesthomas.com/2026/09/gpt-2-to-moe)** — Giles Thomas
- **[AI Infra: 大模型系统设计与工程实践](https://github.com/bojieli/ai-infra-book)** — 李博杰 (open-source book on AI infrastructure: model serving, distributed training/inference, hardware & compute estimation)
- **[James H. Simons, PhD: Using Mathematics to Make Money](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4668072)** — James Simons interview, Journal of Investment Consulting (quant investing at Renaissance Technologies: model-building, hiring scientists over finance veterans, collaboration)

## Blogs

- **[Brendan Gregg's Homepage](https://www.brendangregg.com/)** — systems performance engineer (creator of Flame Graphs, eBPF/BPF tools, DTrace; author of *Systems Performance* and *BPF Performance Tools*); site collects his docs, talks, and tools on Linux/cloud performance analysis
- **[科学空间 (Scientific Spaces)](https://spaces.ac.cn/)** — 苏剑林, blog on math/ML theory (optimizer theory — Adam/Muon, scaling laws, manifold optimization, information theory)
- **[Sebastian Raschka, PhD](https://sebastianraschka.com/)** — LLM research engineer, author of *Build a Large Language Model (From Scratch)*; blog covers LLM research, architecture notes (Kimi K3, Muse Glimmer), and practical/code-driven AI deep dives
- **[Paul Graham's Essays](https://www.paulgraham.com/)** — startups, tech, and how to think, from the Y Combinator co-founder
- **[RLHF Book](https://rlhfbook.com/)** — Nathan Lambert, living online textbook on Reinforcement Learning from Human Feedback (RLHF fundamentals, reward modeling, PPO/DPO, RLHF's role in modern LLM post-training)
- **[Lil'Log (Lilian Weng)](https://lilianweng.github.io/)** — long survey-style posts (20 to 40 minute reads); latest: *Harness Engineering for Self-Improvement* (2026.7), *Scaling Laws, Carefully* (2026.6)
- **[Andrej Karpathy blog](https://karpathy.github.io/)** — infrequent posts; latest: *microgpt* (2026.2), GPT training and inference in 200 lines of pure Python
- **[Tri Dao's blog](https://tridao.me/blog/)** — FlashAttention author; 2026 posts include *Gram Newton-Schulz* (fast Newton-Schulz for Muon), SonicMoE, ReplaySSM
- **[Giles' Blog (Giles Thomas)](https://www.gilesthomas.com/)** — tutorial-style posts on learning LLMs by building them, "the post I wished I'd found when I started learning"; includes a series working through Raschka's from-scratch book; actively updated (2026.10)
- **[Interconnects](https://www.interconnects.ai/)** — Nathan Lambert, newsletter on open models, post-training and the AI industry
- **[Tim Dettmers](https://timdettmers.com/)** — quantization, GPU hardware and open coding agents (*Building SERA*, 2026.1)
- **[Simon Willison's Weblog](https://simonwillison.net/)** — daily notes on LLM tools, coding agents and security
- **[Hamel Husain's Blog](https://hamel.dev/)** — applied AI engineering, focused on evals
- **[Eugene Yan](https://eugeneyan.com/)** — applied LLM systems and recommendation systems
- **[Sasha Rush](https://rush-nlp.com/)** — post-training researcher at Cursor; hands-on exercises: [Thinking like Transformer](https://srush.github.io/raspy), [LLM Training Puzzles](https://github.com/srush/LLM-Training-Puzzles)
- **[Thonk From First Principles (Horace He)](https://www.thonking.ai/)** — ML systems from first principles
- **[Transformer Circuits](https://transformer-circuits.pub/)** — interpretability research thread; Chris Olah's current writing, moved from his older blog [colah.github.io](https://colah.github.io/) (last updated 2021)
- **[The Scaling Hypothesis (Gwern)](https://gwern.net/scaling-hypothesis)** — long essay, not a running blog
- **[LessWrong](https://www.lesswrong.com/)** — community blog on rationality and AI alignment; user-submitted essays, with the Sequences (Eliezer Yudkowsky's rationality essays) as its core reading
- **[Facing the Intelligence Explosion (Luke Muehlhauser)](https://intelligenceexplosion.com/)** — a short online book on what happens when machines become better than humans at doing science and engineering
- **[Recommended Readings (Luke Muehlhauser)](https://lukemuehlhauser.com/recommended-readings/)** — a deliberately short reading list, at most three sources per topic, covering effective altruism, existential risk, extreme-stakes AI risk, forecasting and more

## Code Repositories

- **[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)** — open-source LLM fine-tuning framework (LoRA/QLoRA, distributed training with FSDP2 and DeepSpeed; Apache-2.0)
- **[LitGPT](https://github.com/Lightning-AI/litgpt)** — Lightning AI, 20+ high-performance LLMs with recipes to pretrain, finetune and deploy at scale (Apache-2.0)

## Movies

**Big ideas**

- **[Arrival](https://en.wikipedia.org/wiki/Arrival_(film))** — Denis Villeneuve, 2016 (a linguist communicates with aliens; language, time and family)
- **[Contact](https://en.wikipedia.org/wiki/Contact_(1997_American_film))** — Robert Zemeckis, 1997 (science and faith)
- **[Interstellar](https://en.wikipedia.org/wiki/Interstellar_(film))** — Christopher Nolan, 2014 (space travel and time)
- **[2001: A Space Odyssey](https://en.wikipedia.org/wiki/2001:_A_Space_Odyssey_(film))** — Stanley Kubrick, 1968 (the founding work of the genre; very slow)
- **[The Martian](https://en.wikipedia.org/wiki/The_Martian_(film))** — Ridley Scott, 2015 (survival by hard science, light and optimistic)

**Solitude, vision, reflection**

- **[Oblivion](https://en.wikipedia.org/wiki/Oblivion_(2013_film))** — Joseph Kosinski, 2013 (a lone drone repairman on a post-apocalyptic Earth)
- **[Blade Runner 2049](https://en.wikipedia.org/wiki/Blade_Runner_2049)** — Denis Villeneuve, 2017 (identity and memory)
- **[Moon](https://en.wikipedia.org/wiki/Moon_(2009_film))** — Duncan Jones, 2009 (low budget, one man on a lunar base; identity and clones)
- **[Ex Machina](https://en.wikipedia.org/wiki/Ex_Machina_(film))** — Alex Garland, 2014 (artificial intelligence and what makes us human)
- **[Annihilation](https://en.wikipedia.org/wiki/Annihilation_(film))** — Alex Garland, 2018 (a mutating zone with strange rules; needs patience)
- **[Gravity](https://en.wikipedia.org/wiki/Gravity_(2013_film))** — Alfonso Cuarón, 2013 (survival in space)
- **[Ad Astra](https://en.wikipedia.org/wiki/Ad_Astra_(film))** — James Gray, 2019 (father and son, inward-looking)

**Time and causality**

- **[Tenet](https://en.wikipedia.org/wiki/Tenet_(film))** — Christopher Nolan, 2020 (time inversion; worth planning to watch twice)
- **[Primer](https://en.wikipedia.org/wiki/Primer_(film))** — Shane Carruth, 2004 (ultra-low budget, a rigorously worked-out time machine)
- **[12 Monkeys](https://en.wikipedia.org/wiki/12_Monkeys)** — Terry Gilliam, 1995 (a time loop and fatalism)
- **[Coherence](https://en.wikipedia.org/wiki/Coherence_(film))** — James Ward Byrkit, 2013 (low budget, parallel worlds)

**Reality and perception**

- **[Inception](https://en.wikipedia.org/wiki/Inception)** — Christopher Nolan, 2010 (layered dreams with clear rules)
- **[The Matrix](https://en.wikipedia.org/wiki/The_Matrix)** — the Wachowskis, 1999 (the classic of virtual reality)
- **[The Prestige](https://en.wikipedia.org/wiki/The_Prestige_(film))** — Christopher Nolan, 2006 (not pure science fiction; intricate structure)
- **[Shutter Island](https://en.wikipedia.org/wiki/Shutter_Island_(film))** — Martin Scorsese, 2010 (psychological mystery with a strong twist)

**Closed spaces**

- **[Cube](https://en.wikipedia.org/wiki/Cube_(1997_film))** — Vincenzo Natali, 1997 (strangers trapped in a cube of traps)
