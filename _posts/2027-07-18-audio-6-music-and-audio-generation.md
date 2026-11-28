---
title: "Audio Models, Explained (6/7): Flattening Four Codebooks Gave the Lowest Held-Out Error at 10.7 Bits per Frame but Cost 46 Times More Sampling Time Than the Parallel Pattern in a Small Token Model"
date: 2027-07-18 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [audio, music generation, audiolm, musicgen, language models, series]
description: "Part 6 of 7. How music and general audio are generated as codec tokens: AudioLM's three stages and MusicGen's codebook interleaving, with a small token model that compares the flatten, parallel and delay patterns."
---

## Audio as a language modelling problem

Once audio is a sequence of integers (Part 2), generating it is next-token prediction. Two things make it harder than text. The sequences are long: a codec at 50 frames per second with several codebooks per frame produces hundreds of tokens per second of sound. And the tokens at one frame are not independent, because each residual codebook encodes what the previous one left over. Two papers frame the design space.

## What the sources say

- **AudioLM (hierarchy).** [AudioLM](https://arxiv.org/abs/2209.03143) "maps the input audio to a sequence of discrete tokens and casts audio generation as a language modeling task". Its abstract states that "existing audio tokenizers provide different trade-offs between reconstruction quality and long-term structure", and it combines both kinds: semantic tokens from a masked-language-model network (w2v-BERT) for structure, and SoundStream acoustic tokens (50 Hz for 16 kHz audio, a 320-fold reduction) for fidelity. Generation runs in three stages: semantic modelling for long-term coherence, coarse acoustic modelling conditioned on the semantic tokens, and fine acoustic modelling, with random crops of 30, 10 and 3 seconds for the three stages and about 0.3 billion parameters per stage. Prompted with 3 seconds of speech from an unseen speaker it continues the voice, and it continues piano music "despite being trained without any symbolic representation of music".
- **MusicGen (one model, interleaved codebooks).** [MusicGen](https://arxiv.org/abs/2306.05284) uses "a single-stage transformer LM together with efficient token interleaving patterns, which eliminates the need for cascading several models, e.g., hierarchically or upsampling". Its EnCodec tokenizer runs at 32 kHz, 50 Hz, with four residual quantizers of 2,048 entries each. Because "the first codebook is the most important one", the order in which the four codebooks are laid into the sequence matters. It compares flattening (one token per step, four times the steps), a "parallel" pattern (all four codebooks of a frame predicted at once), and a "delay" pattern in which codebook k is shifted by k steps, so that a frame's finer codebooks are predicted after its coarser ones are known. The paper trains 300M, 1.5B and 3.3B parameter models on 20K hours of licensed music, conditions on text (T5, FLAN-T5 or CLAP) or on a melody (chromagram), samples with top-k 250 at temperature 1.0 with classifier-free guidance at scale 3.0, and reports Fréchet audio distance, KL divergence, a CLAP score and human ratings (84.8 out of 100 overall quality for the 3.3B model against 80.5 for the best baseline).

## Experiments

**Tokens.** The dropout codec of Part 2 encodes every one-second clip of the digit dataset (3,000 clips, six voices) into 50 frames of its first four codebooks (256 entries each; 1,600 bit/s). Every codebook uses all 256 entries, with a frequency-weighted entropy of 4.65 to 4.80 bits out of 8. The five training voices give 2,500 clips; the sixth, never seen in training, gives 500 held-out clips.

**Models.** A causal transformer (four layers, width 128, four heads, 1.0 to 1.07 million parameters) conditioned on the digit label, which stands in for text conditioning. Three layouts of the same 4 × 50 tokens: **flatten** (200 steps, one token per step, codebooks in order inside each frame), **parallel** (50 steps, four independent output heads per step) and **delay** (53 steps, codebook k shifted by k steps). Each was trained for 3,000 steps with batch 32 (AdamW, one-cycle schedule, seed 0).

**Measurements.** Held-out error in nats per frame (summed over the four codebooks, so the patterns are comparable); sampling time and steps for 100 clips (10 per digit, temperature 1); and whether the digit recogniser of Part 3 reads the digit in the decoded audio.

![Left: spectrograms of one digit decoded from real tokens and from tokens sampled by the three patterns; the sampled ones are blurred and the flatten sample is noisy. Middle: held-out error in nats per frame against training step for flatten, parallel and delay patterns; all three reach a minimum near step 400 to 800 and then rise, flatten staying lowest. Right: time to sample 100 clips on a log scale, 74.4 seconds for flatten with 200 steps against 1.6 and 1.9 seconds for parallel and delay with 50 and 53 steps.](/assets/pic/audio-6-token-lm.png)

**1. Held-out error.** After 3,000 steps, on all 500 held-out clips: flatten 7.40 nats (10.7 bits) per frame, delay 8.03 (11.6 bits), parallel 8.42 (12.1 bits); a uniform guess is 22.2 nats (32 bits). All three overfit: with 2,500 training clips, held-out error is lowest at step 400 to 800 and then rises, while training error keeps falling. At the lowest point of the curves (measured on the first 100 held-out clips, which are the digits 0 and 1) the order is the same with smaller gaps: flatten 7.43, delay 7.52, parallel 7.64 nats.

**2. Sampling cost.** 100 clips took 74.4 s for flatten (200 steps), 1.9 s for delay (53 steps) and 1.6 s for parallel (50 steps). The implementation has no cache of past attention keys, so each step recomputes the whole sequence and the 46-fold time ratio overstates the 4-fold ratio in steps.

**3. Does it sound like the digit?** Decoding the held-out clips' own real tokens (four codebooks) and passing them to the recogniser gave the right digit for 22 of 100 clips (38 gave no output at all); uniformly random tokens gave 1 percent. Sampled clips: flatten 9 percent, parallel 8 percent, delay 10 percent (empty output 53, 45 and 56 percent). The generated clips are well below even the ceiling set by the codec, and the three patterns are indistinguishable on this test.

**4. Copying.** The best same-position token match between a clip and any training clip of the same digit was 63.9 percent for real held-out clips (silence frames share tokens) and 57.8, 55.7 and 56.5 percent for the flatten, parallel and delay samples, so the models did not reproduce training clips.

## Engineering reading

These points are this series' reading of the measurements and the sources.

1. **The layout trades fidelity of the joint distribution against steps.** Flattening models the exact dependencies between the codebooks of a frame, at four times the steps; the parallel pattern assumes they are independent given the past; the delay pattern keeps most of the dependencies at the cost of three extra steps. In these runs the held-out error ordered the same way (flatten, delay, parallel), with the gap between flatten and parallel about 1.4 bits per frame after 3,000 steps and about 0.2 nats at the best checkpoints. MusicGen's authors present the patterns as a way of "shedding light on the importance of exact modeling of the parallel codebook sequences".
2. **The codec bounds the generator.** Real tokens at four codebooks were recognised only 22 percent of the time, so no model of those tokens could do better with this recogniser. This is the trade-off AudioLM names between reconstruction quality and structure, seen from the reconstruction side; AudioLM's semantic stage and 3-second fine stage exist to separate the two jobs.
3. **Data, not architecture, limited this model.** A one-million-parameter model on 2,500 clips overfits within hundreds of steps. MusicGen's smallest model has 300 times as many parameters and 20K hours of data.
4. **The label is a weak substitute for structure.** A digit label gives no timing or content for the model to follow; AudioLM's semantic tokens and MusicGen's text or melody conditioning carry that role at scale. How such outputs are judged (Fréchet audio distance, CLAP score, human ratings) is the subject of Part 7.

## Limits

- One seed per pattern, 100 sampled clips per pattern; with 100 clips, differences of a few points in recognition rates are within noise.
- The held-out curves in the figure use the first 100 held-out clips (digits 0 and 1 only); the final numbers use all 500.
- Sampling used no cache of past keys, plain temperature 1.0, no top-k and no guidance, which MusicGen uses.
- The tokens come from a weak four-codebook codec, and the recogniser was trained on real speech, not on codec output, so the recognition figures mix model error with codec error.
- Speech digits of one second, not music; no semantic tokens, text encoder or melody conditioning; no listening test was run.
- Paper figures (parameter counts, hours, ratings) come from different data, models and listeners and are not comparable with these runs.

## Code and output

```python
import glob, numpy as np, torch
from codec import *
X, spk = load_all(); dig = np.array([int(f.split("/")[-1].split("_")[0]) for f in sorted(glob.glob(R + "*.wav"))]); m = Codec(); m.load_state_dict(torch.load("codec_drop.pt")); m.eval(); toks = []
with torch.no_grad():
    for i in range(0, len(X), 100):
        x = X[i:i + 100]; _, _, ids, _ = m(x, torch.full((len(x),), 4)); toks.append(ids[:, :, :4])
toks = torch.cat(toks); print("tokens", tuple(toks.shape), "digits", np.bincount(dig).tolist(), "speakers", sorted(set(spk)))
for k in range(4): print("codebook", k, "distinct used", len(torch.unique(toks[:, :, k])), "entropy bits", float(-(lambda p: (p * p.log2()).sum())(torch.bincount(toks[:, :, k].flatten(), minlength=256).float().div(toks[:, :, k].numel()).clamp_min(1e-12))))
torch.save(dict(toks=toks, dig=torch.tensor(dig), spk=list(spk), X=X), "lm_data.pt")
```

```python
import sys, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
K, V, PAD, S0, D = 4, 256, 256, 50, 128
def to_grid(t, pattern):  # t: (B, 50, 4) -> targets grid (B, S, 4) with PAD where empty
    B = t.shape[0]
    if pattern != "delay": return t
    G = torch.full((B, S0 + K - 1, K), PAD, dtype=torch.long)
    for k in range(K): G[:, k:k + S0, k] = t[:, :, k]
    return G
def from_grid(G, pattern):
    if pattern != "delay": return G
    return torch.stack([G[:, k:k + S0, k] for k in range(K)], 2)
class LM(nn.Module):
    def __init__(s, pattern):
        super().__init__(); s.p = pattern; s.L = {"flatten": S0 * K, "parallel": S0, "delay": S0 + K - 1}[pattern]
        s.emb = nn.ModuleList([nn.Embedding(V + 1, D) for _ in range(K)]); s.dig = nn.Embedding(10, D); s.pos = nn.Embedding(s.L, D)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 512, 0.1, batch_first=True, norm_first=True, activation="gelu"), 4); s.ln = nn.LayerNorm(D)
        s.out = nn.Linear(D, V * (1 if pattern == "flatten" else K))
    def forward(s, inp, d):  # inp: previous-step tokens; flatten: (B, L) ; else (B, L, K); returns logits
        if s.p == "flatten": cb = (torch.arange(s.L) - 1) % K; x = sum(s.emb[k](inp) * (cb == k)[None, :, None] for k in range(K))
        else: x = sum(s.emb[k](inp[:, :, k]) for k in range(K))
        x = x + s.pos(torch.arange(s.L))[None] + s.dig(d)[:, None]; mask = torch.triu(torch.full((s.L, s.L), float("-inf")), 1)
        h = s.ln(s.tf(x, mask=mask, is_causal=True)); o = s.out(h)
        return o if s.p == "flatten" else o.view(o.shape[0], s.L, K, V)
    def prep(s, t):  # tokens (B,50,4) -> inputs (shifted right with BOS=PAD), targets
        G = to_grid(t, s.p)
        if s.p == "flatten": y = G.reshape(len(t), -1); x = torch.cat([torch.full((len(t), 1), PAD), y[:, :-1]], 1); return x, y
        x = torch.cat([torch.full((len(t), 1, K), PAD), G[:, :-1]], 1); return x, G
    def nll(s, t, d):  # summed CE per clip (nats), over valid targets
        x, y = s.prep(t); o = s(x, d); ce = F.cross_entropy(o.reshape(-1, V), y.reshape(-1), ignore_index=PAD, reduction="none").view(y.shape); return ce.flatten(1).sum(1)
    @torch.no_grad()
    def sample(s, d, temp=1.0):
        B = len(d); 
        if s.p == "flatten":
            y = torch.full((B, s.L), PAD); x = torch.full((B, s.L), PAD)
            for j in range(s.L):
                o = s(x, d)[:, j] / temp; y[:, j] = torch.multinomial(o.softmax(-1), 1)[:, 0]
                if j + 1 < s.L: x[:, j + 1] = y[:, j]
            return y.view(B, S0, K)
        G = torch.full((B, s.L, K), PAD); x = torch.full((B, s.L, K), PAD)
        for j in range(s.L):
            o = s(x, d)[:, j] / temp; smp = torch.multinomial(o.softmax(-1).reshape(-1, V), 1).view(B, K)
            if s.p == "delay": valid = torch.tensor([0 <= j - k < S0 for k in range(K)]); smp = torch.where(valid[None], smp, torch.full_like(smp, PAD))
            G[:, j] = smp
            if j + 1 < s.L: x[:, j + 1] = G[:, j]
        return from_grid(G, s.p)
if __name__ == "__main__":
    pattern = sys.argv[1]; steps = int(sys.argv[2]); torch.set_num_threads(int(sys.argv[3])); torch.manual_seed(0); np.random.seed(0)
    data = torch.load("lm_data.pt", weights_only=False); toks = data["toks"]; dig = data["dig"]; spk = np.array(data["spk"]); tr = torch.tensor(np.where(spk != "yweweler")[0]); ho = torch.tensor(np.where(spk == "yweweler")[0])
    m = LM(pattern); print("pattern", pattern, "params", sum(p.numel() for p in m.parameters()), "sequence steps", m.L, "train clips", len(tr), flush=True)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05); t0 = time.time()
    for st in range(steps):
        m.train(); i = tr[torch.randint(len(tr), (32,))]; loss = m.nll(toks[i], dig[i]).mean() / S0; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if st % 200 == 0:
            m.eval()
            with torch.no_grad(): hv = m.nll(toks[ho[:100]], dig[ho[:100]]).mean().item() / S0
            print(st, f"train nats/frame {loss.item()*1:.3f}  held-out {hv:.3f}  t {time.time()-t0:.0f}s", flush=True)
    torch.save(m.state_dict(), f"lm_{pattern}.pt"); print("saved", time.time() - t0)
```

```python
import json, time, numpy as np, torch, math
from codec import Codec
import asr
from lm import *
torch.manual_seed(1); np.random.seed(1)
data = torch.load("lm_data.pt", weights_only=False); toks = data["toks"]; dig = data["dig"]; spk = np.array(data["spk"]); tr = np.where(spk != "yweweler")[0]; ho = np.where(spk == "yweweler")[0]
cod = Codec(); cod.load_state_dict(torch.load("codec_drop.pt")); cod.eval()
def decode(ids):
    z = sum(cod.vq[k].embed[ids[:, :, k]] for k in range(K)).permute(0, 2, 1)
    with torch.no_grad(): return cod.dec(z)[:, 0].numpy()
ctc = asr.CTCModel(); ctc.load_state_dict(torch.load("asr_ctc.pt")); ctc.eval()
def recog(ids, d):
    w = decode(ids); m, lens = asr.feats(list(w))
    with torch.no_grad(): pr = asr.ctc_decode(ctc(m, lens), lens)
    return float(np.mean([p == [int(x)] for p, x in zip(pr, d)])) * 100, float(np.mean([len(p) == 0 for p in pr])) * 100, w
res = {}; D100 = torch.arange(10).repeat_interleave(10)
sel = torch.tensor([ho[(dig[ho] == dg).numpy()][j] for dg in range(10) for j in range(10)])
a, e, wreal = recog(toks[sel], dig[sel]); res["real_decoded"] = dict(acc=a, empty=e); print(f"control: held-out clips encoded at 4 codebooks and decoded: digit recognised {a:.0f}% (empty output {e:.0f}%)")
a, e, _ = recog(torch.randint(0, 256, (100, 50, 4)), D100); res["random"] = dict(acc=a, empty=e); print(f"control: uniformly random tokens decoded: recognised {a:.0f}% (empty {e:.0f}%)")
trd = {dg: toks[tr[(dig[tr] == dg).numpy()]] for dg in range(10)}
def match(g, d):  # best same-position token match fraction against any training clip of the same digit
    return float(np.mean([(trd[int(x)] == gi[None]).float().mean((1, 2)).max().item() for gi, x in zip(g, d)]))
res["real_match"] = match(toks[sel], dig[sel]); print(f"held-out real clips: best token match to a training clip of the same digit {res['real_match']*100:.1f}%")
ex = {"real": wreal[70].tolist()}
for p in ("flatten", "parallel", "delay"):
    m = LM(p); m.load_state_dict(torch.load(f"lm_{p}.pt")); m.eval()
    with torch.no_grad(): nll = torch.cat([m.nll(toks[ho[i:i + 100]], dig[ho[i:i + 100]]) for i in range(0, 500, 100)]).mean().item() / S0
    t0 = time.time(); g = m.sample(D100); dt = time.time() - t0; a, e, w = recog(g, D100); mt = match(g, D100)
    res[p] = dict(nats=nll, bits=nll / math.log(2), steps=m.L, sec=dt, acc=a, empty=e, match=mt); ex[p] = w[70].tolist()
    print(f"{p:9s}: held-out {nll:.2f} nats = {nll/math.log(2):.1f} bits per frame (uniform: {4*8} bits) | sequence steps {m.L} | 100 clips sampled in {dt:.1f} s | digit recognised {a:.0f}% (empty {e:.0f}%) | best token match to training {mt*100:.1f}%")
res["ex"] = ex; json.dump(res, open("lm.json", "w"))
```

```text
# --- lm_flatten.log ---
pattern flatten params 984832 sequence steps 200 train clips 2500
0 train nats/frame 22.358  held-out 22.224  t 2s
200 train nats/frame 8.715  held-out 7.712  t 215s
400 train nats/frame 7.882  held-out 7.478  t 429s
600 train nats/frame 8.340  held-out 7.432  t 643s
800 train nats/frame 8.007  held-out 7.437  t 818s
1000 train nats/frame 8.068  held-out 7.496  t 923s
1200 train nats/frame 7.755  held-out 7.582  t 1029s
1400 train nats/frame 7.753  held-out 7.665  t 1134s
1600 train nats/frame 7.342  held-out 7.708  t 1238s
1800 train nats/frame 8.002  held-out 7.783  t 1343s
2000 train nats/frame 7.125  held-out 7.837  t 1448s
2200 train nats/frame 7.064  held-out 7.892  t 1553s
2400 train nats/frame 7.727  held-out 7.917  t 1659s
2600 train nats/frame 7.442  held-out 7.923  t 1763s
2800 train nats/frame 6.794  held-out 7.928  t 1868s
saved 1971.3454406261444
# --- lm_parallel.log ---
pattern parallel params 1064704 sequence steps 50 train clips 2500
0 train nats/frame 21.971  held-out 21.774  t 1s
200 train nats/frame 8.546  held-out 7.726  t 51s
400 train nats/frame 7.955  held-out 7.640  t 102s
600 train nats/frame 8.008  held-out 7.721  t 152s
800 train nats/frame 7.588  held-out 7.864  t 203s
1000 train nats/frame 7.796  held-out 8.013  t 253s
1200 train nats/frame 6.885  held-out 8.199  t 304s
1400 train nats/frame 7.287  held-out 8.388  t 356s
1600 train nats/frame 6.836  held-out 8.509  t 408s
1800 train nats/frame 6.716  held-out 8.664  t 460s
2000 train nats/frame 6.626  held-out 8.781  t 511s
2200 train nats/frame 6.680  held-out 8.846  t 561s
2400 train nats/frame 6.133  held-out 8.918  t 613s
2600 train nats/frame 6.345  held-out 8.941  t 663s
2800 train nats/frame 6.467  held-out 8.939  t 714s
saved 763.7216904163361
# --- lm_delay.log ---
pattern delay params 1065088 sequence steps 53 train clips 2500
0 train nats/frame 22.210  held-out 22.029  t 1s
200 train nats/frame 8.709  held-out 7.606  t 53s
400 train nats/frame 8.243  held-out 7.521  t 106s
600 train nats/frame 7.883  held-out 7.577  t 158s
800 train nats/frame 7.569  held-out 7.636  t 211s
1000 train nats/frame 7.176  held-out 7.784  t 263s
1200 train nats/frame 7.523  held-out 7.933  t 316s
1400 train nats/frame 6.584  held-out 8.110  t 372s
1600 train nats/frame 6.926  held-out 8.242  t 426s
1800 train nats/frame 6.545  held-out 8.361  t 480s
2000 train nats/frame 6.352  held-out 8.456  t 534s
2200 train nats/frame 6.318  held-out 8.530  t 587s
2400 train nats/frame 6.149  held-out 8.585  t 641s
2600 train nats/frame 6.058  held-out 8.613  t 693s
2800 train nats/frame 6.198  held-out 8.615  t 746s
saved 787.658124923706
# --- lm_out.txt ---
control: held-out clips encoded at 4 codebooks and decoded: digit recognised 22% (empty output 38%)
control: uniformly random tokens decoded: recognised 1% (empty 52%)
held-out real clips: best token match to a training clip of the same digit 63.9%
flatten  : held-out 7.40 nats = 10.7 bits per frame (uniform: 32 bits) | sequence steps 200 | 100 clips sampled in 74.4 s | digit recognised 9% (empty 53%) | best token match to training 57.8%
parallel : held-out 8.42 nats = 12.1 bits per frame (uniform: 32 bits) | sequence steps 50 | 100 clips sampled in 1.6 s | digit recognised 8% (empty 45%) | best token match to training 55.7%
delay    : held-out 8.03 nats = 11.6 bits per frame (uniform: 32 bits) | sequence steps 53 | 100 clips sampled in 1.9 s | digit recognised 10% (empty 56%) | best token match to training 56.5%
```

## References

- Z. Borsos et al., [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143), 2022.
- J. Copet et al., [Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284), 2023.
- A. Défossez et al., [High Fidelity Neural Audio Compression (EnCodec)](https://arxiv.org/abs/2210.13438), 2022.
- N. Zeghidour et al., [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312), 2021.
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).
