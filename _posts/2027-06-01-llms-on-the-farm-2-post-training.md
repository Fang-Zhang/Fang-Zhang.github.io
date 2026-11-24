---
title: "LLMs on the Farm, Part 2: Retrieval Carries the Facts, Fine-Tuning Carries the Habits"
date: 2027-06-01 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, agriculture, rag, fine-tuning, lora, post-training, llms on the farm]
description: "Part 2 of 6 on large language models in agriculture. A published farm-document study shows retrieval and fine-tuning adding up for GPT-4 (75, 80, 81, 86 percent). Papers on knowledge injection explain why, and a CPU experiment shows the cost: facts trained into weights are hard to update and displace old skills, while facts retrieved into the prompt are edited in place."
---

## What this post finds

Part 1 chose the base model. Part 2 asks how to get a farm's own knowledge into the system: regional planting windows, local pest records, extension bulletins, a company's product rules. The two standard routes are retrieval (put the relevant text into the prompt) and fine-tuning (change the model's weights).

- **Both help, and the gains add up.** In a study on Washington State farm documents, [Balaguer and colleagues](https://arxiv.org/abs/2401.08406) report GPT-4 at 75 percent accuracy as is, 80 with retrieval, 81 fine-tuned and 86 with both. The abstract summarises it as over 6 points from fine-tuning, cumulative with 5 more from retrieval.
- **Retrieval is the better carrier of facts.** [Ovadia and colleagues](https://arxiv.org/abs/2312.05934) find that retrieval consistently beats unsupervised fine-tuning, for knowledge the model saw in pretraining and for entirely new knowledge, and that models struggle to learn new facts by fine-tuning. [Gekhman and colleagues](https://arxiv.org/abs/2405.05904) find new facts are learned slowly in fine-tuning and, once learned, raise the tendency to hallucinate. Their conclusion is that models mostly acquire facts in pretraining and fine-tuning teaches them to use those facts.
- **Fine-tuning is for habits.** Output format, tone, succinctness, the step order of an advisory answer. The farm study's own discussion puts it this way: fine-tuning taught domain-specific skills and gave more precise, shorter answers, at a high initial cost.
- **In a small CPU experiment the trade is visible.** Training a 110,384-parameter model on 24 new facts pushed its loss on ordinary text from 1.99 to 4.43 bits per character; replaying ordinary text during training cut that to 2.12, but the facts were still answered right only 27 to 52 percent of the time. Putting the fact in the prompt answered 90 percent right with no change to the model, and changing a fact took one edit instead of a training run.

![Three panels. Left: GPT-4 accuracy on a farm-document benchmark, 75 percent as is, 80 with retrieval, 81 fine-tuned, 86 fine-tuned with retrieval. Middle: recall at top 3 falls from 81.5 to 75.0 percent as the retrieval index grows from 573 to 3,888 documents. Right: in a toy run, facts answered right and held-out text loss for five setups: no new knowledge 0.06 and 1.99, full fine-tuning on facts only 0.52 and 4.43, full fine-tuning with replay 0.27 and 2.12, LoRA rank 4 with replay 0.42 and 2.19, retrieval in the prompt 0.90 and 1.99]({{ site.baseurl }}/assets/pic/farm-2-post-training.png){: width="900" }
_Figure: left and middle are published numbers (Tables 18 and 17 of the farm study, the left axis starts at 60); right is the experiment at the end of this post._

## The published evidence in detail

**The farm study.** Documents from three crop-producing countries (the United States, Brazil and India) were converted into question-answer pairs by GPT models, filtered for quality, and used both as retrieval sources and as fine-tuning data. Answers were graded by GPT-4 against guideline answers. For GPT-4, accuracy was 75 percent plain, 80 with retrieval, 81 after fine-tuning and 86 with both. For a 13-billion-parameter open model the picture is less tidy: Llama-2-chat scored 76 percent plain and 75 with retrieval, while a Llama 2 13B model fine-tuned directly on the farm data scored 68 and reached 77 with retrieval. The effect of each technique depends on the model it is added to.

**Retrieval scales with a cost.** The same study measured how often the right passage appears among the top three retrieved. Recall was 81.5 percent with 573 documents in the index and fell to 75.0 percent with 3,888, because more similar passages compete. A farm advisory system that keeps adding documents needs this number tracked, not assumed.

**Why this differs from the benchmark in Part 1.** The AI-AgriBench authors report that adding retrieval to frontier models did not significantly improve accuracy on their text questions. That benchmark is close to saturated for the base models, so there is little room for retrieval to add. The farm study asks location-specific questions that a general model answers generically; its motivating example is a planting-time question where GPT-4 gives the same answer for three different states. Retrieval helps where the base model lacks the specific fact, and shows little where it already has it.

**Cost of new knowledge in weights.** Fine-tuning every parameter is expensive and can damage what the model already does. [LoRA](https://arxiv.org/abs/2106.09685) freezes the weights and trains small low-rank matrices added to each layer; for GPT-3 175B the authors report 10,000 times fewer trainable parameters and 3 times less GPU memory than fine-tuning with Adam. [Biderman and colleagues](https://arxiv.org/abs/2405.09673) compare LoRA with full fine-tuning on programming and mathematics: in standard low-rank settings LoRA learns less, and it also forgets less of what the base model could do outside the target domain.

## Steps: decide route by what the knowledge is

1. **Facts that change or must be traceable** (prices, regulations, local pest alerts, product labels): retrieval. A fact is edited in the store, and the answer can cite the passage.
2. **Habits and format** (answer in three steps, ask for the crop stage first, use a fixed warning for pesticide questions, answer in short sentences for voice): fine-tuning, or at first just a prompt, then fine-tuning once the prompt is stable and the behaviour must hold on a small model.
3. **Both, in that order.** Build retrieval first because it is cheap to change, measure it with a local test set (Part 1, step 4), then fine-tune on what retrieval cannot fix.
4. **When fine-tuning, mix in ordinary data** and keep a fixed held-out set of general tasks, so that forgetting shows up in a number.
5. **Re-run the whole evaluation after every change** to the index, the prompt or the weights.

## Pitfalls

- **Synthetic training data inherits its generator.** In the farm study the question-answer pairs, and the grading, come from GPT-4-class models. Pairs not reviewed by an agronomist can encode the generator's errors, and a generator grading its own style is not an independent judge.
- **Retrieval needs a model that can read.** A model must be able to use text placed in its prompt. In the experiment below an earlier run without any reading practice scored 0.00 with retrieval; frontier models have this ability from instruction tuning, small local models may not.
- **Retrieval has its own failure: the wrong passage.** Recall at top 3 sets a ceiling on answer accuracy, and it falls as the index grows.
- **Fine-tuning on facts can teach confident guessing.** The Gekhman result applies to any farm dataset whose facts the base model does not know.
- **Updates are the hidden cost.** Seasonal facts change every year. A retrieval store is edited; weights are retrained.

## Worked example: the same 24 facts three ways

The script below pretrains a 110,384-parameter character transformer on the text that ships with Python's documentation. It is a stand-in for a base model, not a farm model, so the numbers are about the mechanism, not about farms. It then adds 24 invented facts of the form `kora: maize in march.` (an invented zone, a crop, a planting month) five ways, with five random fact tables, and measures how many it answers right (a month answer has a 1 in 12 chance by luck) and how its loss on unrelated text changes. Retrieval means the matching line is looked up by zone name and put in front of the question. Fifteen percent of the pretraining windows are small reading exercises (a fact line followed by a question on it) so that the base model can read its prompt; the test zones and lead-in words are not in them.

```python
# Putting new facts into a small language model three ways: full fine-tuning, LoRA, and retrieval in the prompt.
# CPU only, standard library + PyTorch. The "farm facts" are synthetic (invented zone names), so the model cannot know them from pretraining.
import re, math, random, copy, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pydoc_data.topics as topics                                   # ships with Python: the pretraining text

text = "\n".join(topics.topics[k] for k in sorted(topics.topics)).lower()
text = re.sub(r"[^a-z0-9 .,\n()'\"=:_-]", " ", text); text = re.sub(r" +", " ", text)
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
ids = torch.tensor([stoi[c] for c in text]); n = len(ids)
train, test = ids[: int(.9 * n)], ids[int(.9 * n):]
T, D, H, L = 64, 64, 4, 2

class Lin(nn.Module):                                              # nn.Linear with an optional low-rank update B @ A
    def __init__(s, i, o): super().__init__(); s.base = nn.Linear(i, o); s.A = s.B = None
    def add_lora(s, r, alpha=8.0):
        s.A = nn.Parameter(torch.randn(r, s.base.in_features) / math.sqrt(s.base.in_features)); s.B = nn.Parameter(torch.zeros(s.base.out_features, r)); s.scale = alpha / r
    def forward(s, x): return s.base(x) if s.A is None else s.base(x) + s.scale * (x @ s.A.T) @ s.B.T

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv, s.proj, s.fc1, s.fc2 = Lin(D, 3 * D), Lin(D, D), Lin(D, 4 * D), Lin(4 * D, D)
    def forward(s, x):
        B, t, _ = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, dim=2)
        q, k, v = (a.view(B, t, H, D // H).transpose(1, 2) for a in (q, k, v))
        x = x + s.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D))
        return x + s.fc2(F.gelu(s.fc1(s.ln2(x))))

class Tiny(nn.Module):
    def __init__(s):
        super().__init__(); s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.Sequential(*[Block() for _ in range(L)]); s.ln = nn.LayerNorm(D); s.out = nn.Linear(D, V)
    def forward(s, idx): return s.out(s.ln(s.blocks(s.tok(idx) + s.pos(torch.arange(idx.shape[1])))))

def batch(data, g, bs=64):
    st = torch.randint(0, len(data) - T - 1, (bs,), generator=g); return torch.stack([data[j:j + T] for j in st]), torch.stack([data[j + 1:j + T + 1] for j in st])

MONTHS = "january february march april may june july august september october november december".split()
CROPS = "maize wheat barley oats rye millet sorghum rice beans peas flax hemp".split()
def fact_line(f): return f"{f[0]}: {f[1]} in {f[2]}.\n"
def random_fact(r):
    return ("".join(r.choice("bdgkmnprstvz") + r.choice("aeiou") for _ in range(2)), r.choice(CROPS), r.choice(MONTHS))
def reading_example(r):                                                # a fact in the prompt, then a question about it: teaches the model to READ its context
    f = random_fact(r); lead = r.choice(["", "".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(r.randint(3, 5))) + " "]); return (fact_line(f) + f"{lead}{f[0]}: {f[1]} in {f[2]}.\n").ljust(T)[:T]

def pretrain(steps=3000, lr=3e-3, seed=0, reading=0.15):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); r = random.Random(seed); m = Tiny(); opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for s in range(1, steps + 1):
        for p in opt.param_groups: p["lr"] = lr * (1 - s / (steps + 1))
        x, y = batch(train, g)
        for i in range(len(x)):                                        # 15 percent of windows are reading examples instead of documentation text
            if r.random() < reading: w = torch.tensor([stoi[c] if c in stoi else stoi[" "] for c in reading_example(r) + " "][: T + 1]); x[i], y[i] = w[:T], w[1:T + 1]
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def bpc(m, data=test, k=300):                                       # bits per character on held-out Python-documentation text
    g = torch.Generator().manual_seed(123); x, y = batch(data, g, k); return F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item() / math.log(2)

# ---- synthetic facts: "zone kora: plant maize in march."
def make_facts(seed, k=24):
    r = random.Random(10_000 + seed); zones = set(); out = []
    while len(out) < k:
        f = random_fact(r)
        if f[0] not in zones: zones.add(f[0]); out.append(f)
    return out
q_exact = lambda f: f"{f[0]}: {f[1]} in "                 # same words as the training line
q_para = lambda f: f"note {f[0]}: {f[1]} in "         # same facts, different lead-in

@torch.no_grad()
def answer(m, prompt):                                              # greedy decode up to 12 characters
    x = torch.tensor([[stoi[c] for c in prompt]])
    for _ in range(12): x = torch.cat([x, m(x[:, -T:])[0, -1].argmax().view(1, 1)], 1)
    return "".join(chars[i] for i in x[0, len(prompt):])
def accuracy(m, facts, q, store=None):                              # store given: retrieve the matching line and put it first
    hit = 0
    for f in facts:
        p = q(f)
        if store is not None: p = store[f[0]] + p                   # retrieval: look up the zone name in the store
        hit += answer(m, p).startswith(f[2])
    return hit / len(facts)

def finetune(m, facts, steps=150, lr=1e-3, lora=False, seed=0, replay=0.0):
    m = copy.deepcopy(m); g = torch.Generator().manual_seed(seed)
    if lora:
        for p in m.parameters(): p.requires_grad = False
        for b in m.blocks:
            for l in (b.qkv, b.proj, b.fc1, b.fc2): l.add_lora(4)
    params = [p for p in m.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    data = [stoi[c] for c in "".join(fact_line(f) for f in facts)]; d = torch.tensor(data)
    for s in range(steps):
        st = torch.randint(0, len(d) - 48, (16,), generator=g); x = torch.stack([d[j:j + 48] for j in st]); y = torch.stack([d[j + 1:j + 49] for j in st])
        if replay: xr, yr = batch(train, g, int(16 * replay / (1 - replay)) or 1); x, y = torch.cat([x, xr[:, :48]]), torch.cat([y, yr[:, :48]])
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return m, sum(p.numel() for p in params)

if __name__ == "__main__":
    t0 = time.time(); base = pretrain(); total = sum(p.numel() for p in base.parameters())
    print(f"pretrained {total:,}-parameter model in {time.time() - t0:.0f} s; held-out text {bpc(base):.3f} bits per character")
    print(f"chance level for a month answer: {1 / 12:.3f}\n")
    res = {k: [] for k in ["base", "ft", "ftr", "lora", "rag"]}
    for seed in range(5):
        facts = make_facts(seed); store = {f[0]: fact_line(f) for f in facts}
        ft, nft = finetune(base, facts, lr=3e-4, steps=300, seed=seed); ftr, _ = finetune(base, facts, lr=3e-4, steps=300, seed=seed, replay=0.5)
        lo, nlo = finetune(base, facts, lr=3e-3, steps=300, lora=True, seed=seed, replay=0.5)
        row = {"base": (accuracy(base, facts, q_exact), accuracy(base, facts, q_para), bpc(base)),
               "ft": (accuracy(ft, facts, q_exact), accuracy(ft, facts, q_para), bpc(ft)),
               "ftr": (accuracy(ftr, facts, q_exact), accuracy(ftr, facts, q_para), bpc(ftr)),
               "lora": (accuracy(lo, facts, q_exact), accuracy(lo, facts, q_para), bpc(lo)),
               "rag": (accuracy(base, facts, q_exact, store), accuracy(base, facts, q_para, store), bpc(base))}
        for k, v in row.items(): res[k].append(v)
    print(f"trainable parameters: full fine-tuning {nft:,}; LoRA rank 4 {nlo:,} ({100 * nlo / total:.1f} percent of the model)\n")
    print("mean over 5 random fact tables of 24 facts          exact wording   new lead-in   held-out bpc")
    names = {"base": "no new knowledge", "ft": "full fine-tuning, facts only", "ftr": "full fine-tuning + replay", "lora": "LoRA rank 4 + replay", "rag": "retrieval in the prompt"}
    for k in res: a = np.mean(res[k], 0); print(f"  {names[k]:30s}  {a[0]:10.2f}  {a[1]:12.2f}  {a[2]:12.3f}")
    # ---- an update: 6 of the 24 facts change month
    print("\nupdating 6 facts (new month):")
    upd = {"ft": [], "ftkeep": [], "ftall": [], "rag": [], "ragkeep": []}
    for seed in range(5):
        facts = make_facts(seed); ft, _ = finetune(base, facts, lr=3e-4, steps=300, seed=seed, replay=0.5); r = random.Random(100 + seed)
        new = [(f[0], f[1], r.choice([m for m in MONTHS if m != f[2]])) for f in facts[:6]]; rest = facts[6:]
        ft2, _ = finetune(ft, new, lr=3e-4, steps=60, seed=seed, replay=0.5)                 # continue training on the changed lines only
        ft3, _ = finetune(base, new + rest, lr=3e-4, steps=300, seed=seed, replay=0.5)   # retrain from the base model on the whole updated table
        upd["ftall"].append(accuracy(ft3, new + rest, q_exact))
        store = {f[0]: fact_line(f) for f in new + rest}                 # retrieval: edit the store, no training
        upd["ft"].append(accuracy(ft2, new, q_exact)); upd["ftkeep"].append(accuracy(ft2, rest, q_exact)); upd["rag"].append(accuracy(base, new, q_exact, store)); upd["ragkeep"].append(accuracy(base, rest, q_exact, store))
    print(f"  fine-tune again on the 6 changed lines: changed facts right {np.mean(upd['ft']):.2f}, the other 18 still right {np.mean(upd['ftkeep']):.2f}")
    print(f"  retrain from the base model on all 24 updated facts: all 24 right {np.mean(upd['ftall']):.2f}   (all answers scored with the exact wording)")
    print(f"  edit the store and retrieve (no training): changed facts right {np.mean(upd['rag']):.2f}, the other 18 right {np.mean(upd['ragkeep']):.2f}")
    json.dump({"res": {k: np.mean(v, 0).tolist() for k, v in res.items()}, "upd": {k: float(np.mean(v)) for k, v in upd.items()}, "params": [total, nft, nlo]}, open("farm2.json", "w"))
    print(f"\n{time.time() - t0:.0f} s")
```

Output:

```text
pretrained 110,384-parameter model in 73 s; held-out text 1.992 bits per character
chance level for a month answer: 0.083

trainable parameters: full fine-tuning 110,384; LoRA rank 4 8,192 (7.4 percent of the model)

mean over 5 random fact tables of 24 facts          exact wording   new lead-in   held-out bpc
  no new knowledge                      0.06          0.00         1.992
  full fine-tuning, facts only          0.52          0.75         4.433
  full fine-tuning + replay             0.27          0.58         2.124
  LoRA rank 4 + replay                  0.42          0.61         2.186
  retrieval in the prompt               0.90          0.67         1.992

updating 6 facts (new month):
  fine-tune again on the 6 changed lines: changed facts right 0.47, the other 18 still right 0.14
  retrain from the base model on all 24 updated facts: all 24 right 0.32   (all answers scored with the exact wording)
  edit the store and retrieve (no training): changed facts right 0.77, the other 18 right 0.90

150 s
```

What the run shows, and what it does not:

- **Facts only, full fine-tuning** gets 0.52 right with the exact wording and 0.75 with a different lead-in, and the held-out loss more than doubles (1.99 to 4.43). The model learned the facts by giving up general text.
- **Replay** (half of every batch is ordinary text) holds the loss near the original (2.12 for full fine-tuning, 2.19 for LoRA rank 4 with 8,192 trainable parameters, 7.4 percent of the model) but learns the facts worse: 0.27 and 0.42 with the exact wording. With this little training, protecting old skills and learning new facts compete.
- **Retrieval** answers 0.90 right with the exact wording and 0.67 with a new lead-in, with the model untouched (1.99). It was never trained on these zones. Its weak spot is reading: a lead-in it has not seen costs it a quarter of the accuracy.
- **An update of 6 facts**: retraining on the changed lines alone left 0.47 of them right and the other 18 at 0.14, because the continued training overwrote the rest. Retraining on all 24 updated facts from the base model reached 0.32 overall. Editing the store gave 0.77 on the changed facts and 0.90 on the others, with no training.
- **Limits.** Five fact tables, one model, one small training recipe; the fine-tuning numbers would move with learning rate and steps, and a 110 thousand parameter model forgets far faster than a large one. The direction, facts into weights costs old skills and costs a training run per update, is what the published papers also report; the exact values are not transferable.

## What comes next

Part 3 turns this into a deployed system: where the model runs (cloud, a local server, the phone), how retrieval, tools and photos are wired together, and what has to connect to existing farm records.

## References

- Balaguer et al., [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture](https://arxiv.org/abs/2401.08406), 2024.
- Ovadia, Brief, Mishaeli et al., [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs](https://arxiv.org/abs/2312.05934), 2023.
- Gekhman, Yona, Aharoni et al., [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904), 2024.
- Hu, Shen, Wallis et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.
- Biderman, Portes, Ortiz et al., [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673), 2024.
- University of Illinois, [AI-AgriBench](https://digitalag.illinois.edu/2026/01/28/building-trust-in-ai-agronomy-through-transparent-benchmarking), January 2026.
