---
title: "How Language Models Work, Part 9: Where the Data Comes From"
date: 2027-04-11 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [language models, LLM, training data, copyright, privacy, memorization, beginners, teens]
description: "Part 9 of a 12-part series for readers aged 14 to 17. Where the text for training comes from, how much of it there is, why a model can repeat what it saw, and what a court order, a government report and a security paper say about copyright and privacy. Includes a measurement of copying on the Part 8 model."
---

Part 8 ended with a model that memorised a book because it had too little text. Real models get far more. This post covers where that text comes from, what happens to it, and two questions that come with it: who owns the text, and what if it contains private information. For both, it reports what documents say and does not pick a side.

## Where the text comes from

- **Web crawls.** [Common Crawl](https://commoncrawl.org/overview) is a public archive of web pages. Its own overview says the corpus "contains petabytes of data, regularly collected since 2008".
- **Curated mixes.** [The Pile](https://arxiv.org/abs/2101.00027) is "an 825 GiB English text corpus targeted at training large-scale language models", built from "22 diverse high-quality subsets", many of which "derive from academic or professional sources". Its authors also document "potentially concerning aspects of the data for prospective users".
- **Public data only, as a choice.** The [LLaMA paper](https://arxiv.org/abs/2302.13971) says its models were trained "on trillions of tokens" and that it is possible to train strong models "using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets".

## The maths tool: orders of magnitude

When numbers differ by factors of millions, a normal chart is useless. A **log scale** puts 10, 1,000 and 100,000 at equal distances: each step up is a multiplication by 100, not an addition. Logs appeared in Part 5 (surprise is `-ln(p)`); here they turn multiplication into counting steps.

![Left: log-scale bar chart comparing the 149 kilobytes of text used in Parts 2 to 8 with the 825 gibibytes of the Pile, about 6 million times larger. Right: bar chart of how many generated six-word runs are copied from the training book: 0 percent after 200 steps, 10 percent and 27 percent after 1,500 steps]({{ site.baseurl }}/assets/pic/teen-llm-data-memory.png){: width="900" }
*Left: the book used in this series (149,000 bytes, counted from the cleaned file) against the Pile (825 GiB, from its paper's abstract). Right: copying, measured below.*

The Pile is about 6 million times larger than the book used in this series (825 GiB is roughly 8.9 x 10^11 bytes; the book is 1.5 x 10^5; the ratio is the author's arithmetic). That is why the Part 8 Transformer had room to memorise, and why a model trained on millions of books has less spare room to do so, though still some.

## Measuring copying

How much does a model copy its training text? One way to measure: give the model the first 8 words of a random passage from the training book, let it write 20 more words, and count how many 6-word runs in its output appear word-for-word in the book. The code below adds to the Part 8 program.

```python
N = 6
seen = set(tuple(train_data[i:i + N].tolist()) for i in range(len(train_data) - N))

def copy_rate(model, greedy, n_prompts=100, new=20, seed=1):
    g = torch.Generator().manual_seed(seed); model.eval()
    hit = tot = 0
    for _ in range(n_prompts):
        s = torch.randint(0, len(train_data) - 40, (1,), generator=g).item()
        ctx = train_data[s:s + 8].tolist()
        for _ in range(new):
            c = ctx[-T:]; x = torch.tensor(c + [0] * (T - len(c)))[None]      # pad on the right; the mask hides it
            with torch.no_grad(): lg = model(x)[0, len(c) - 1]
            nxt = lg.argmax().item() if greedy else torch.multinomial(F.softmax(lg, -1), 1, generator=g).item()
            ctx.append(nxt)
        gen = ctx[8:]
        for i in range(len(gen) - N + 1):
            w = tuple(gen[i:i + N])
            if 500 in w: continue
            tot += 1; hit += w in seen
    model.train(); return hit, tot
```

Real output on the Part 8 Transformer, with the one-word network for comparison:

```
bigram         300 steps  sampled    1 of 571 six-word runs are in the book (0%)
bigram        1500 steps  sampled    0 of 558 six-word runs are in the book (0%)
transformer    200 steps  sampled    0 of 487 six-word runs are in the book (0%)
transformer   1500 steps  sampled   61 of 615 six-word runs are in the book (10%)
transformer   1500 steps  greedy   146 of 548 six-word runs are in the book (27%)
```

"Sampled" picks each word at random according to its chances, as in Part 2. "Greedy" always takes the most likely word. At 1,500 steps, the Transformer reproduces 10% of its 6-word runs when sampled and 27% when greedy; early in training, 0%. (The greedy one-word network produced no countable runs, because it repeats the placeholder `<other>`, so it is left out.) One real example, greedy, after 1,500 steps:

```
prompt: she heard a little <other> of feet in
model : the distance , and she <other> it to <other> it . they couldn't ...
book  : the distance , and she hastily <other> her eyes to see what was coming ...
```

Five words ("the distance , and she") are copied, then the model drifts away. The placeholder `<other>` stands for words outside the 500 most common, which makes the measurement conservative.

## Privacy: what a model can give back

The same effect matters when training text contains private information. A [2021 security paper](https://arxiv.org/abs/2012.07805) attacked GPT-2, a model "trained on scrapes of the public Internet", and was "able to extract hundreds of verbatim text sequences from the model's training data". The extracted examples "include (public) personally identifiable information (names, phone numbers, and email addresses), IRC conversations, code, and 128-bit UUIDs". Two findings: the attack worked "even though each of the above sequences are included in just one document in the training data", and "larger models are more vulnerable than smaller models".

For you, the lesson is practical: text you post publicly can end up in a crawl, and text you type into a chatbot may be kept or used depending on that service's rules. Do not enter your address, school, passwords or other people's private details. Many chat services also set a minimum age; check the terms before using one.

## Copyright: what the documents say

The U.S. Copyright Office's report on training, [Part 3 of its AI series](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf) (pre-publication version, 2025), states: "Dozens of lawsuits are pending in the United States, focusing on the application of copyright's fair use doctrine". It describes the debate: "Some warn that requiring AI companies to license copyrighted works would throttle a transformative technology". "Others fear that unlicensed training will corrode the creative ecosystem". The Office writes: "Without opining on specific cases, we provide an analytical framework for identifying relevant facts and policy considerations."

One case has produced a ruling on the facts. In [Bartz v. Anthropic](https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0.pdf), a U.S. district court in California ruled on 23 June 2025, at the summary-judgment stage, that "the use of the books at issue to train Claude and its precursors was exceedingly transformative and was a fair use under Section 107 of the Copyright Act". The same order went the other way on a different question: "Anthropic had no entitlement to use pirated copies for its central library", and it denied summary judgment on those copies. The order also notes that the authors did not allege that any output given to users infringed their works.

Read these carefully. The order is from one trial court, in one case, about one company's books. It is not a rule for every country or every dataset, and the Copyright Office describes many lawsuits still open. Your own country's law may differ. Whether training needs permission is a question that courts and parliaments are deciding now.

## Try it

1. Run the copying program. Change the 6 in `N = 6` to 4, then 10. How does the percentage change, and why?
2. Run it on the model at 600 steps. Where between 200 and 1,500 steps does copying start?
3. Open [Project Gutenberg](https://www.gutenberg.org/ebooks/11) and find the passage in the example above in the book. Which words did the model change?
4. Pick any website you use. Would you want it in a training set? Who should decide?

## Limits

- The measurement uses one small model and one book. It shows that copying rises with over-training. It does not say how often a large model copies its data.
- The six-word check misses copying with small changes, and counts common phrases ("she said to the") as copies.
- The 825 GiB figure is from the Pile's abstract; real models use data of different sizes, and many do not publish it.
- The legal text above is a summary of three documents and is not legal advice. Orders and reports are quoted from their own text; a pre-publication report and a summary-judgment order may be revised or appealed.
