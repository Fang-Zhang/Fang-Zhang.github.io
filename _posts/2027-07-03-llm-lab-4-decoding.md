---
title: "Improving LLM Performance, Measured (4/6): Greedy Decoding Scored Best on Word Validity but Repeated Itself, and Speculative Decoding Gained 19 Percent at Best on a CPU"
date: 2027-07-03 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [llm, decoding, sampling, speculative decoding, latency, series]
description: "Part 4 of 6. Temperature, top-p and speculative decoding on a small language model: quality against diversity for ten sampling settings, and measured speedup against the published formula."
---

## What the literature says

Decoding changes the output of a trained model without changing its weights. Three public sources cover the two questions here, quality against diversity and latency.

- **Likelihood-maximising decoding degenerates.** [Holtzman et al.](https://arxiv.org/abs/1904.09751) report that "using likelihood as a decoding objective leads to text that is bland and strangely repetitive", that "decoding strategies alone can dramatically effect the quality of machine text, even when generated from exactly the same neural language model", and propose nucleus sampling, which samples from the "dynamic nucleus of the probability distribution" and so truncates "the less reliable tail".
- **Speculative decoding is exact.** [Leviathan et al.](https://arxiv.org/abs/2211.17192) sample from a large model faster "without any changes to the outputs", with a smaller approximation model proposing tokens that the large model checks in parallel, and report a 2X to 3X speedup on T5-XXL with identical outputs. Their Theorem 3.8 gives the expected wall-time improvement as (1 - a^(g+1)) / ((1 - a)(g c + 1)), where a is the acceptance rate, g the number of proposed tokens and c the cost of the draft model relative to the target.
- **Same idea, larger scale.** [Chen et al.](https://arxiv.org/abs/2302.01318) describe speculative sampling, a "modified rejection sampling scheme which preserves the distribution of the target model", and report a 2 to 2.5x decoding speedup on a 70 billion parameter model in a distributed setup.

All claims come from abstracts, plus the formula from the full text of the first paper. The experiment below checks the shape of both effects on a small model and compares the measured speedup with the formula.

## Experiment 1: ten sampling settings

Setup: four character-level GPTs (4 layers, width 128, 842,624 parameters, held-out loss 2.618 plus or minus 0.020 bits per character; 800 training steps, one seed per model) from the same prose corpus as the earlier parts. Each model continues 16 held-out 64-character prompts by 160 characters under each setting. Two measures per setting, averaged over the four models:

- **Valid-word share**: the share of generated words of three or more letters that occur in the training text. A crude stand-in for fluency.
- **Diversity**: the share of distinct 8-character chunks in the output. A value near 0 means the text loops.

Results (mean over 4 models; the standard deviation across models is 0.02 to 0.08 for valid-word share and at most 0.05 for diversity):

- **Greedy**: valid-word share 0.800, diversity 0.133.
- **Temperature 0.3, 0.5, 0.7, 1.0, 1.3**: valid-word share 0.787, 0.691, 0.546, 0.368, 0.199; diversity 0.619, 0.861, 0.970, 0.995, 0.999.
- **With top-p**: temperature 0.7 with p = 0.9: 0.655 and 0.919; temperature 1.0 with p = 0.9: 0.466 and 0.984; temperature 1.0 with p = 0.5: 0.745 and 0.754; temperature 1.3 with p = 0.9: 0.262 and 0.998.

## Experiment 2: speculative decoding

Setup: for each seed a draft model of 2 layers and width 64 (124,800 parameters, held-out loss 3.500 plus or minus 0.023 against 2.618 for the target) is trained for the same number of steps. Both models run on one CPU thread with batch size 1, 24 prompts of 64 characters and 48 new characters each. The implementation follows the accept-or-resample rule of the two papers; plain decoding and speculative decoding were timed back to back.

- **Exactness.** With greedy decoding the speculative output equalled the target model's own greedy output in 96 of 96 runs for every number of draft tokens. The sampled case was not tested for distributional equality.
- **Draft cost.** One draft forward pass cost 0.25 (plus or minus 0.02) of a target forward pass, although the draft has 15 percent of the parameters; small matrices run closer to fixed overhead.
- **Acceptance.** 0.46 to 0.50 for greedy and 0.58 to 0.60 for sampling at temperature 1; tokens per target call rose from 1.57 (1 draft token) to 2.38 (8) when sampling.
- **Speed relative to plain decoding.** Sampling at temperature 1: 1.18, 1.19, 1.01 and 0.75 for 1, 2, 4 and 8 draft tokens. Greedy: 1.15, 1.09, 0.94 and 0.70. The formula, with the measured acceptance rate and cost ratio, gave 1.27, 1.31, 1.16 and 0.82 for sampling and 1.17, 1.13, 0.97 and 0.67 for greedy.

![Two panels. Left: share of generated words found in the training vocabulary against share of distinct 8-character chunks for ten decoding settings; greedy sits at 0.80 and 0.13, temperature 1.3 at 0.20 and 1.00. Right: speculative decoding speed relative to plain decoding for 1, 2, 4 and 8 draft tokens; sampling gives 1.18, 1.19, 1.01 and 0.75, greedy gives 1.15, 1.09, 0.94 and 0.70, and the formula curves sit at or above the measurements]({{ site.baseurl }}/assets/pic/llm-lab-4-decoding.png){: width="900" }
_Figure: left, mean and standard deviation over 4 models; right, mean speed over 4 models. Measured here._

## Engineering reading

These points are this series' reading of the sources and the measurement.

1. **A single quality number picked the worst output.** By valid-word share alone, greedy decoding (0.800) beat every sampled setting. Its diversity was 0.133, which means about seven of eight 8-character chunks were repeats: the loops that Holtzman et al. describe as "bland and strangely repetitive". Any decoding comparison needs a quality measure and a repetition measure together.
2. **Temperature trades the two measures almost linearly.** From 0.3 to 1.3, diversity rose from 0.62 to 1.00 while valid-word share fell from 0.79 to 0.20. The usable region on this model was a temperature of about 0.5 to 0.7, where both measures were moderate.
3. **Top-p was no clear gain over temperature alone on these measures.** Interpolating the temperature-only curve at equal diversity, the three top-p settings with temperature up to 1.0 sit 0.01 to 0.04 above it in valid-word share (for example 0.655 against about 0.61 at diversity 0.92), while the standard deviation across models is 0.03 to 0.06. This does not test the human-judged quality claim of nucleus sampling.
4. **Speculative decoding paid off only for short proposals.** With a draft costing 0.25 of the target and a 0.6 acceptance rate, 1 to 2 draft tokens gave 15 to 19 percent faster sampling, 4 gave nothing, and 8 made decoding 25 percent slower, because each failed proposal pays for draft tokens that are thrown away. The formula predicts this shape; it overestimates the measured speed by 0.01 to 0.14 here, which is expected from overheads it ignores.
5. **The published 2x to 3x needs a much cheaper draft or a more agreeable pair.** At a cost ratio of 0.25 and an acceptance rate of 0.6, the formula's best case is about 1.3. By the same formula, a gain near 3 times needs both a high acceptance rate (0.8) and a draft costing a few percent of the target (for example 5 percent with 5 proposed tokens), whereas acceptance of 0.8 at the cost ratio measured here gives about 1.7 at best. The cited setups used drafts around two orders of magnitude smaller than the target.
6. **Greedy speedups here are within timing noise.** The greedy runs showed 1.15 plus or minus 0.09 and 1.09 plus or minus 0.19 for 1 and 2 draft tokens, and the plain baselines for greedy and sampling on the same model differed by 20 percent (2.70 against 2.25 milliseconds per token), so differences of that size are not resolved.

## Limits

- Models of 0.84 million and 0.12 million parameters, character level, one training run per model, 4 models. The conclusions are about shape, not size.
- The valid-word share is a vocabulary lookup, not fluency; the diversity measure counts repeats within the generated text only. No human or model-based judgment was used, and the top-p result concerns only these two measures.
- Timings are single-thread CPU, batch 1, no key-value cache (each step recomputes the whole context), with four processes running side by side. GPU serving, batching and cached attention shift the cost ratio and therefore the best number of draft tokens.
- Exactness was verified for greedy decoding only; for sampling, the equality of distributions rests on the proofs in the cited papers.
- Literature claims come from abstracts except the formula, which was checked against the full text of the first paper. No result was reproduced.

## Code and output

Experiment code (model and training loop from Part 1):

```python
import sys, json, math, re, time, torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
import torch.nn.functional as F, lm
P = lm.load_prose(); PTR, PVA = lm.split(P); TR = lm.enc("".join(PTR)); VA = lm.enc("".join(PVA))
WORDS = {w for w in re.findall(r"[a-z]{3,}", "".join(PTR).lower())}
def train_model(seed, D, H, L, steps=800):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = lm.GPT(D=D, H=H, L=L); lr, warmup, B = 3e-3, 100, 16
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01); probs = torch.ones(1)
    for t in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * min(1, (t + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
        x, y = lm.batch([TR], probs, B, g); loss = F.cross_entropy(m(x).reshape(-1, lm.V), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m
def dist(logits, T, top_p):
    if T == 0: return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    p = F.softmax(logits / T, -1)
    if top_p < 1:
        sp, si = p.sort(-1, descending=True); keep = (sp.cumsum(-1) - sp) < top_p; sp = sp * keep; sp = sp / sp.sum(-1, keepdim=True); p = torch.zeros_like(p).scatter(-1, si, sp)
    return p
@torch.no_grad()
def gen_batch(m, prompts, n, T, top_p, g):
    x = prompts.clone()
    for _ in range(n):
        p = dist(m(x[:, -lm.CTX:])[:, -1], T, top_p); x = torch.cat([x, torch.multinomial(p, 1, generator=g)], 1)
    return x[:, prompts.shape[1]:]
def quality(txt_rows):
    words = [w for t in txt_rows for w in re.findall(r"[a-z]{3,}", t.lower())]
    valid = sum(w in WORDS for w in words) / max(1, len(words))
    grams = [t[i:i + 8] for t in txt_rows for i in range(len(t) - 7)]; distinct = len(set(grams)) / max(1, len(grams))
    return valid, distinct
def prompts_from(src, n, plen, g):
    ix = torch.randint(0, len(src) - plen - 1, (n,), generator=g); return torch.stack([src[i:i + plen] for i in ix])
@torch.no_grad()
def plain(m, x, n, T, g):
    for _ in range(n):
        p = dist(m(x)[:, -1], T, 1.0); x = torch.cat([x, torch.multinomial(p, 1, generator=g)], 1)
    return x
@torch.no_grad()
def spec(tm, dm, x, n, k, T, g):
    L0 = x.shape[1]; calls = 0; acc = 0; prop = 0
    while x.shape[1] - L0 < n:
        kk = min(k, L0 + n - x.shape[1] - 1) if L0 + n - x.shape[1] > 1 else 0; d = x; qs = []
        for _ in range(kk):
            q = dist(dm(d)[:, -1], T, 1.0); qs.append(q); d = torch.cat([d, torch.multinomial(q, 1, generator=g)], 1)
        pl = dist(tm(d)[:, -(kk + 1):], T, 1.0); calls += 1; ok = True; cur = x
        for i in range(kk):
            tok = d[:, x.shape[1] + i]; pt, qt = pl[0, i, tok.item()], qs[i][0, tok.item()]; prop += 1
            if T == 0: accept = pt > 0
            else: accept = torch.rand(1, generator=g).item() < min(1.0, (pt / qt).item())
            if accept: cur = torch.cat([cur, tok[:, None]], 1); acc += 1
            else:
                r = (pl[0, i] - qs[i][0]).clamp(min=0); r = pl[0, i] if r.sum() == 0 else r / r.sum()
                cur = torch.cat([cur, torch.multinomial(r[None], 1, generator=g)], 1); ok = False; break
        if ok: cur = torch.cat([cur, torch.multinomial(pl[0, kk][None], 1, generator=g)], 1)
        x = cur
    return x[:, :L0 + n], calls, acc, prop
if __name__ == "__main__":
    seed = int(sys.argv[1]); out = {"seed": seed}
    tm = train_model(seed, 128, 4, 4); dm = train_model(seed + 100, 64, 4, 2)
    out["val_bpc"] = [lm.evaluate(tm, VA, 400), lm.evaluate(dm, VA, 400)]
    out["params"] = [sum(p.numel() for p in tm.parameters()), sum(p.numel() for p in dm.parameters())]
    json.dump(out, open(f"f4_seed{seed}.json", "w"))
    g = torch.Generator().manual_seed(1000 + seed); pr = prompts_from(VA, 16, 64, g); sweep = []
    for T, tp in [(0, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (1.0, 1.0), (1.3, 1.0), (0.7, 0.9), (1.0, 0.9), (1.0, 0.5), (1.3, 0.9)]:
        y = gen_batch(tm, pr, 160, T, tp, torch.Generator().manual_seed(seed)); txt = ["".join(chr(int(c)) for c in r) for r in y]; v, dd = quality(txt)
        sweep.append([T, tp, v, dd]); print(T, tp, round(v, 3), round(dd, 3), flush=True)
    out["sweep"] = sweep; json.dump(out, open(f"f4_seed{seed}.json", "w"))
    # speculative decoding: batch 1, prompt 64 chars, 48 new tokens, no window trimming (64+48+k <= 128)
    g = torch.Generator().manual_seed(2000 + seed); pr = prompts_from(VA, 24, 64, g); N = 48
    def timeit(fn):
        t0 = time.perf_counter(); r = fn(); return time.perf_counter() - t0, r
    tt = 0
    for i in range(24): t, _ = timeit(lambda: dm(pr[i:i + 1])); tt += t
    tc = 0
    for i in range(24): t, _ = timeit(lambda: tm(pr[i:i + 1])); tc += t
    out["cost_ratio"] = tt / tc
    res = []
    for T in (0, 1.0):
        gg = torch.Generator().manual_seed(7); t0 = time.perf_counter(); base = [plain(tm, pr[i:i + 1], N, T, gg) for i in range(24)]; tp_ = time.perf_counter() - t0
        res.append(["plain", T, 0, tp_ / (24 * N), 1.0, None, None])
        for k in (1, 2, 4, 8):
            gg = torch.Generator().manual_seed(7); t0 = time.perf_counter(); outs = [spec(tm, dm, pr[i:i + 1], N, k, T, gg) for i in range(24)]; ts = time.perf_counter() - t0
            calls = sum(o[1] for o in outs); acc = sum(o[2] for o in outs); prop = sum(o[3] for o in outs)
            same = sum(bool((o[0] == b).all()) for o, b in zip(outs, base)) if T == 0 else None
            res.append(["spec", T, k, ts / (24 * N), (24 * N) / calls, acc / max(1, prop), same]); print(res[-1], flush=True)
    out["spec"] = res; json.dump(out, open(f"f4_seed{seed}.json", "w"))
```

Statistics output:

```
seeds: 4
target/draft params: [842624, 124800]  held-out bpc target 2.618±0.020 draft 3.500±0.023

sampling sweep (160 chars x 16 prompts per model; mean ± sd over 4 models):
temp top_p  valid-word share   distinct 8-grams
0    1.0   0.800 ± 0.082     0.133 ± 0.024
0.3  1.0   0.787 ± 0.047     0.619 ± 0.050
0.5  1.0   0.691 ± 0.019     0.861 ± 0.029
0.7  1.0   0.546 ± 0.029     0.970 ± 0.011
1.0  1.0   0.368 ± 0.025     0.995 ± 0.003
1.3  1.0   0.199 ± 0.045     0.999 ± 0.001
0.7  0.9   0.655 ± 0.028     0.919 ± 0.015
1.0  0.9   0.466 ± 0.056     0.984 ± 0.006
1.0  0.5   0.745 ± 0.036     0.754 ± 0.031
1.3  0.9   0.262 ± 0.037     0.998 ± 0.001

draft/target forward cost ratio c = 0.25 ± 0.02

speculative decoding (48 tokens x 24 prompts per model, batch 1, one thread; mean over 4 models)
T    k  ms/token  tokens per target call  acceptance  speedup vs plain  predicted speedup  greedy outputs identical
0    plain 2.70
0    1  2.35     1.45                    0.46        1.15 ± 0.09      1.17              96 of 96
0    2  2.49     1.65                    0.47        1.09 ± 0.19      1.13              96 of 96
0    4  2.90     1.83                    0.50        0.94 ± 0.18      0.97              96 of 96
0    8  3.93     1.97                    0.50        0.70 ± 0.14      0.67              96 of 96
1.0  plain 2.25
1.0  1  1.90     1.57                    0.58        1.18 ± 0.01      1.27
1.0  2  1.89     1.92                    0.60        1.19 ± 0.01      1.31
1.0  4  2.24     2.26                    0.60        1.01 ± 0.08      1.16
1.0  8  3.01     2.38                    0.60        0.75 ± 0.02      0.82
```

Next: Part 5 covers retrieval and context: what retrieval-augmented generation and longer context windows change, and which differences are within noise.

## References

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), 2019.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), 2022.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), 2023.
