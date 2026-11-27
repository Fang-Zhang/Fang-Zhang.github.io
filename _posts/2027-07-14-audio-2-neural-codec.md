---
title: "Audio Models, Explained (2/7): Eight Residual Codebooks Cut 16,000 Bytes per Second to 400 and Matched 24 kbit/s PCM at 3.2 kbit/s, but the Quantizer-Dropout Gain Did Not Reproduce"
date: 2027-07-14 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [audio, neural codec, vector quantization, soundstream, encodec, series]
description: "Part 2 of 7. How a neural audio codec turns a waveform into a short sequence of integers: encoder, residual vector quantization, decoder. A 1.26M-parameter codec trained from scratch on spoken digits, measured against the number of codebooks and against plain PCM."
---

## Why sound needs tokens

[Part 1](/posts/audio-1-sound-as-data/) ended on a problem: a spectrogram can be matched closely while the waveform behind it is different, and 16,000 samples per second is too long a sequence to model directly. A neural audio codec addresses both. A convolutional encoder shortens the waveform into a sequence of vectors, a quantizer replaces each vector with a few integers, and a decoder turns the integers back into a waveform. The integers are the tokens that the language-model approaches to speech and music in Parts 4 to 6 predict. This part builds the smallest honest version of that pipeline and measures what the bit budget buys.

## What the sources say

- **Architecture and rate.** [SoundStream](https://arxiv.org/abs/2107.03312) uses a fully convolutional encoder that "produces a sequence of embeddings at a lower sampling rate, which are quantized by a residual vector quantizer", and a convolutional decoder. With strides of (2, 4, 5, 8), "one embedding is computed every M = 2 x 4 x 5 x 8 = 320 input samples", which at 24 kHz gives embeddings at 75 Hz.
- **Residual vector quantization.** One codebook is a coarse quantizer; the residual it leaves is quantized by a second codebook, and so on. [EnCodec](https://arxiv.org/abs/2210.13438) describes the process as "computing the residual after quantization, and further quantizing it using a second codebook, and so forth", with at most 32 codebooks of 1,024 entries, "e.g. 10 bits per codebook", which gives bandwidths of 1.5, 3, 6, 12 or 24 kbps at 24 kHz. The arithmetic checks: 75 frames per second x 10 bits = 750 bit/s per codebook, so 2 codebooks are 1.5 kbps and 32 are 24 kbps.
- **Training the codebooks.** Both papers update each codebook with an exponential moving average (decay 0.99), use a straight-through estimator so that gradients pass the quantizer as if it were the identity, and replace unused entries. SoundStream initialises with k-means on the first batch and replaces entries whose assignment average falls below 2.
- **Quantizer dropout.** SoundStream's "for each input example, we sample nq uniformly at random in [1, Nq] and only use quantizers Qi for i = 1 ... nq", so that a single model serves several bit rates. Its experiment compared a model trained at 18 kbps and evaluated with fewer quantizers against one trained with dropout; the gap "vanishes when using the quantizer dropout strategy".
- **Entropy coding.** EnCodec reports that a small Transformer over the quantized units can "further compress the obtained representation by up to 40%".

## Experiments

**Setup.** Data: the same [spoken-digit recordings](https://github.com/Jakobovski/free-spoken-digit-dataset) as Part 1 at their native 8 kHz, each padded with zeros to exactly 1 s (8,000 samples) and peak-normalised. Training uses the 2,500 clips of five speakers; the 500 clips of the sixth speaker are held out. The codec is a 1,261,697-parameter convolutional encoder and decoder with strides (2, 4, 5, 4), which is 160 samples per frame and 50 frames per second, a 64-dimensional latent, and up to 8 residual codebooks of 256 entries. Each token is therefore 8 bits and each codebook costs 50 x 8 = 400 bit/s; all eight cost 3,200 bit/s against 128,000 bit/s for 16-bit PCM. Codebooks use EMA updates (decay 0.99), a straight-through estimator, k-means-style initialisation from the first batch and replacement of dead entries by random inputs. The loss is 10 x waveform L1, plus 0.1 x a multi-resolution log-spectrogram loss, minus 0.1 x SI-SDR, plus 0.25 x the commitment loss; there is no adversarial loss. Three models were trained for 2,500 steps of batch 16 (about 16 passes over the training clips), one seed each:

- **drop**: eight codebooks, with the number of active codebooks drawn uniformly from 1 to 8 for each example;
- **f8**: eight codebooks always on;
- **f2**: two codebooks only.

Quality is measured by SI-SDR (scale-invariant signal-to-distortion ratio, higher is better, 0 dB means as much error as signal) and by a log-spectral distance (LSD, lower is better) on the held-out speaker. The baseline is plain uniform PCM at the same 8 kHz with 1 to 8 bits.

![Three parts. Left: SI-SDR in dB on a held-out speaker against bit rate. The three small codecs run from about -3 dB at 400 bit/s to 2.0 dB (trained with quantizer dropout), 3.6 dB (no dropout) and 0.3 dB (trained with 2 codebooks) at 3,200 bit/s; uniform PCM is -6.7 dB at 8 kbit/s, 1.4 dB at 24 kbit/s and 31.8 dB at 64 kbit/s. Middle: the share of latent energy left falls from 35.7 percent after one residual quantizer stage to 1.0 percent after eight. Right: four spectrograms of a spoken seven (original, 1, 2 and 8 codebooks) in which the harmonic bands of the vowel are blurred at one codebook and partly restored at eight; the black area on the right is zero padding.](/assets/pic/audio-2-codec.png)

**1. Quality against bit rate.**

| codebooks used | bit rate | drop SI-SDR / LSD | f8 SI-SDR / LSD | f2 SI-SDR / LSD |
|---|---|---|---|---|
| 1 | 400 bit/s | -2.7 dB / 8.0 dB | -2.9 dB / 7.1 dB | -2.4 dB / 5.2 dB |
| 2 | 800 bit/s | 0.0 dB / 7.4 dB | 0.2 dB / 6.7 dB | 0.2 dB / 5.0 dB |
| 4 | 1,600 bit/s | 1.6 dB / 7.1 dB | 2.4 dB / 6.3 dB | 0.2 dB / 5.0 dB |
| 8 | 3,200 bit/s | 2.0 dB / 7.0 dB | 3.6 dB / 6.1 dB | 0.3 dB / 5.0 dB |

Uniform PCM gives -6.7 dB / 11.8 dB at 8 kbit/s (1 bit), -3.5 / 9.8 at 16 kbit/s, 1.4 / 7.9 at 24 kbit/s (3 bits), 7.2 / 6.2 at 32 kbit/s, 19.5 / 3.5 at 48 kbit/s and 31.8 / 1.8 at 64 kbit/s. At 3,200 bit/s the 8-codebook models match or exceed 3-bit PCM at 24,000 bit/s on both measures, which is 7.5 times fewer bits; 4-bit PCM at 32,000 bit/s is still clearly better on SI-SDR (7.2 dB against 2.0 to 3.6 dB). The extra codebooks of f2, which was never trained with more than two, add nothing (0.2 dB to 0.3 dB), while f2 has the best LSD of the three, so the two measures do not rank the models the same way.

**2. What a stage does.** In the dropout model the share of latent energy left after stages 1 to 8 was 35.7, 18.4, 10.0, 5.6, 3.4, 2.1, 1.4 and 1.0 percent: each added codebook removed roughly 40 to 50 percent of what remained over the first four stages, and less afterwards. The always-on model went from 49.1 to 4.7 percent over the same eight stages (the two latent spaces have different scales, so only the shape of each curve compares). All 251 to 256 entries of each 256-entry codebook were used on the held-out speaker, so no codebook collapsed. The perplexity of the usage distribution was 15.6 to 21.3, equivalent to 16 to 21 equally likely entries, or about 4.0 to 4.4 bits of marginal entropy against the 8 bits spent per token.

**3. Tokens are enough.** Encoding 20 held-out clips to an integer tensor of shape (20, 50, 8) and decoding from the codebook entries alone reproduced the forward-pass output to 1.1e-7. One second of audio is 16,000 bytes as PCM, 400 bytes (400 tokens, 40 times smaller) with 8 codebooks and 50 bytes (320 times smaller) with one. The sequence a language model would see is 400 tokens per second, 20 times fewer steps than the 8,000 samples.

**4. A failed first attempt.** The first training run used only L1 and log-spectrogram losses (learning rate 5e-4). It matched spectra to about 4.5 dB LSD, but its SI-SDR was -19.4 dB at 8 codebooks and -20.8 dB at one for the dropout model, with almost no change between the two: the decoder produced audio with the right spectrogram and an unrelated waveform, the situation of Part 1 reappearing inside a model. An unquantised encoder and decoder trained with those losses behaved the same way over 400 steps (SI-SDR -19 dB) and reached 4 to 5 dB once SI-SDR was added to the loss; that change, with a learning rate of 1e-3, is the configuration reported above.

## Engineering reading

These points are this series' reading of the measurements and the sources.

1. **A codec is a rate choice.** Bit rate is frames per second times codebooks times bits per codebook: 50 x 8 x 8 here, 75 x 32 x 10 at the top setting of EnCodec. Everything else is quality at that rate, and the table shows how steep the curve is at the bottom: the first codebook alone gave negative SI-SDR.
2. **Residual quantization buys coarse-to-fine structure.** The first stage removed the most energy and every later stage removed a smaller share, so the first tokens carry the broad shape and the later ones refine it. This ordering is what makes it possible to use only the first few tokens when the budget is small.
3. **Spectrum-only training does not make a waveform codec.** Matching log spectra got an LSD of 4.5 dB and a waveform that was unrelated to the original. A codec that must be inverted to sound needs a waveform-sensitive term, which production codecs get from adversarial training on top of their reconstruction losses.
4. **The quantizer-dropout benefit did not appear at this scale.** SoundStream found that dropout removes the loss from using fewer quantizers than were trained. Here the always-on model also degraded smoothly as codebooks were removed, was level with the dropout model at one codebook (-2.9 dB against -2.7 dB) and ahead of it from two codebooks on (0.2, 0.8 and 1.6 dB at 2, 4 and 8 codebooks). With one seed, 2,500 steps and no adversarial loss, this shows that the effect is not guaranteed at small scale; it says nothing against the result in the original setting. The bit-rate-specific two-codebook model was not better than the others in SI-SDR (0.2 dB at two codebooks against 0.0 and 0.2), only in LSD.
5. **Token entropy is lower than token size.** Usage perplexities of 16 to 21 against 256 entries suggest that an entropy coder could spend fewer bits than the nominal 8 per token on this data. The estimate treats tokens as independent and ignores neighbours, so it is an upper bound on the achievable bits, in the direction of EnCodec's reported 40 percent but not comparable with it.

## Limits

- The codec is 1.26M parameters trained for 2,500 steps of batch 16 on 2,500 one-second clips of spoken digits at 8 kHz; production codecs are trained on far more audio for far longer, with adversarial discriminators. The absolute quality (2 to 3.6 dB SI-SDR) is poor, and a listening test was not done; every comparison is numerical.
- One held-out speaker is a small test of generalisation, and the digits are the same ten words as in training; no claim is made for continuous speech, noise or music.
- One seed per model. Differences of 1 to 2 dB between models (the dropout comparison in particular) were not tested against seed-to-seed variation.
- The PCM baseline is plain uniform quantisation with no companding or noise shaping, which is a weak baseline; a mu-law or ADPCM coder would do better at the same bit rate.
- The dead-entry rule uses an EMA-count threshold of 0.02, not SoundStream's threshold of 2, because the counts here are normalised differently; the first-batch initialisation draws random frames instead of running k-means.
- The perplexity-based entropy figure ignores dependence between tokens and between stages; it is not an achieved compression result.
- SI-SDR is computed per clip including the zero-padded tail.

## Code and output

```python
import glob, wave, sys, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
R = "free-spoken-digit-dataset-master/recordings/"; T = 8000; NQ = 8; NCB = 256; D = 64; STRIDES = (2, 4, 5, 4)
def load_all():
    X = []; spk = []
    for f in sorted(glob.glob(R + "*.wav")):
        w = wave.open(f); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768; x = x / (np.abs(x).max() + 1e-8) * 0.9
        y = np.zeros(T, np.float32); y[:min(T, len(x))] = x[:T]; X.append(y); spk.append(f.split("_")[-2])
    return torch.tensor(np.stack(X))[:, None, :], np.array(spk)
class RU(nn.Module):
    def __init__(s, c, d): super().__init__(); s.a = nn.Conv1d(c, c, 3, padding=d, dilation=d); s.b = nn.Conv1d(c, c, 1)
    def forward(s, x): return x + s.b(F.elu(s.a(F.elu(x))))
class Enc(nn.Module):
    def __init__(s, C=16):
        super().__init__(); L = [nn.Conv1d(1, C, 7, padding=3)]; c = C
        for st in STRIDES: L += [RU(c, 1), RU(c, 3), nn.ELU(), nn.Conv1d(c, 2 * c, 2 * st + 1, stride=st, padding=st)]; c *= 2
        L += [nn.ELU(), nn.Conv1d(c, D, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, x): return s.net(x)
class Dec(nn.Module):
    def __init__(s, C=16):
        super().__init__(); c = C * 2 ** len(STRIDES); L = [nn.Conv1d(D, c, 3, padding=1)]
        for st in reversed(STRIDES): L += [nn.ELU(), nn.Upsample(scale_factor=st, mode="nearest"), nn.Conv1d(c, c // 2, 2 * st + 1, padding=st), RU(c // 2, 1), RU(c // 2, 3)]; c //= 2
        L += [nn.ELU(), nn.Conv1d(c, 1, 7, padding=3)]; s.net = nn.Sequential(*L)
    def forward(s, z): return s.net(z)
class VQ(nn.Module):
    def __init__(s, n=NCB, d=D, decay=0.99):
        super().__init__(); s.n = n; s.decay = decay; s.register_buffer("embed", torch.zeros(n, d)); s.register_buffer("cnt", torch.ones(n)); s.register_buffer("sm", torch.zeros(n, d)); s.register_buffer("ready", torch.tensor(0))
    def forward(s, r, active):  # r: (M, D) residual frames; active: (M,) bool
        if s.training and s.ready.item() == 0:
            idx = torch.randperm(len(r))[:s.n]; s.embed.copy_(r.detach()[idx] if len(idx) == s.n else r.detach()[torch.randint(len(r), (s.n,))]); s.sm.copy_(s.embed); s.ready.fill_(1)
        rd = r.detach(); d = (rd ** 2).sum(1, keepdim=True) - 2 * rd @ s.embed.T + (s.embed ** 2).sum(1)[None]; i = d.argmin(1); q = s.embed[i]
        if s.training and active.any():
            ra = r[active].detach(); ia = i[active]; oh = F.one_hot(ia, s.n).float(); s.cnt.mul_(s.decay).add_(oh.sum(0), alpha=1 - s.decay); s.sm.mul_(s.decay).add_(oh.T @ ra, alpha=1 - s.decay)
            s.embed.copy_(s.sm / s.cnt.clamp_min(1e-5)[:, None]); dead = s.cnt < 2e-2
            if dead.any(): k = int(dead.sum()); rr = ra[torch.randint(len(ra), (k,))]; s.embed[dead] = rr; s.sm[dead] = rr; s.cnt[dead] = 1.0
        return q, i
class Codec(nn.Module):
    def __init__(s, C=16): super().__init__(); s.enc = Enc(C); s.dec = Dec(C); s.vq = nn.ModuleList([VQ() for _ in range(NQ)])
    def quant(s, y, nq):  # y: (B, D, S); nq: (B,) number of stages per example
        B, _, S = y.shape; r = y.permute(0, 2, 1).reshape(-1, D); nqf = nq[:, None].expand(B, S).reshape(-1); out = torch.zeros_like(r); commit = 0.0; ids = []; res = r
        for k in range(NQ):
            act = nqf > k; q, i = s.vq[k](res, act); commit = commit + F.mse_loss(res[act], q[act].detach()) if act.any() else commit
            out = out + q * act[:, None]; res = res - q.detach() * act[:, None] if False else res - q * act[:, None]; ids.append(i)
        ste = r + (out - r).detach(); return ste.reshape(B, S, D).permute(0, 2, 1), commit, torch.stack(ids, 1).reshape(B, S, NQ), [None]
    def forward(s, x, nq): y = s.enc(x); z, c, ids, _ = s.quant(y, nq); return s.dec(z), c, ids, y
def stft_loss(x, y):
    l = 0
    for n in (512, 256, 128):
        w = torch.hann_window(n); X = torch.stft(x[:, 0], n, n // 4, window=w, return_complex=True).abs(); Y = torch.stft(y[:, 0], n, n // 4, window=w, return_complex=True).abs()
        l = l + F.l1_loss(torch.log(X + 1e-5), torch.log(Y + 1e-5)) + torch.norm(X - Y) / (torch.norm(X) + 1e-8)
    return l / 3
def sisdr(x, y):
    x = x - x.mean(-1, keepdim=True); y = y - y.mean(-1, keepdim=True); a = (y * x).sum(-1, keepdim=True) / (x ** 2).sum(-1, keepdim=True).clamp_min(1e-9); e = a * x; return 10 * torch.log10((e ** 2).sum(-1) / ((y - e) ** 2).sum(-1).clamp_min(1e-12))
if __name__ == "__main__":
    mode = sys.argv[1]; steps = int(sys.argv[2]); torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    X, spk = load_all(); tr = torch.tensor(np.where(spk != "yweweler")[0]); m = Codec(); opt = torch.optim.Adam(m.parameters(), 1e-3); print("params", sum(p.numel() for p in m.parameters() if p.requires_grad), "train", len(tr), flush=True); t0 = time.time(); B = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    for st in range(steps):
        x = X[tr[torch.randint(len(tr), (B,))]]; nq = torch.randint(1, NQ + 1, (B,)) if mode == "drop" else torch.full((B,), int(mode[1:]) if mode.startswith("f") else NQ)
        xh, c, _, _ = m(x, nq); loss = 10 * F.l1_loss(xh, x) + 0.1 * stft_loss(x, xh) - 0.1 * sisdr(x[:, 0], xh[:, 0]).mean() + 0.25 * c; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if st % 100 == 0: print(st, f"loss {loss.item():.4f} commit {float(c):.4f} t {time.time()-t0:.0f}s", flush=True)
    torch.save(m.state_dict(), f"codec_{mode}.pt")
```

```python
import json, numpy as np, torch, torch.nn.functional as F, librosa
from codec import *
torch.manual_seed(0); X, spk = load_all(); te = np.where(spk == "yweweler")[0]; Xt = X[te]; print(f"held-out speaker: {len(te)} clips; train speakers: {sorted(set(spk) - {'yweweler'})}")
def lsd(x, y):
    n = 256; w = torch.hann_window(n); a = torch.stft(x[:, 0], n, n // 4, window=w, return_complex=True).abs(); b = torch.stft(y[:, 0], n, n // 4, window=w, return_complex=True).abs()
    return (20 * torch.log10((a + 1e-4) / (b + 1e-4))).abs().mean().item()
def mods():
    d = {}
    for k in ("drop", "f8", "f2"):
        m = Codec(); m.load_state_dict(torch.load(f"codec_{k}.pt")); m.eval(); d[k] = m
    return d
M = mods(); res = {}
print("bitrate: 8 bits per token, 50 frames per second, so each codebook costs 400 bit/s; raw 8 kHz 16-bit PCM is 128,000 bit/s")
print("\n1. Reconstruction on the held-out speaker against the number of codebooks used")
with torch.no_grad():
    for k, m in M.items():
        res[k] = {}
        for nq in range(1, NQ + 1):
            xs = []; ss = []; ls = []
            for i in range(0, len(Xt), 100):
                x = Xt[i:i + 100]; xh, _, _, _ = m(x, torch.full((len(x),), nq)); ss.append(sisdr(x[:, 0], xh[:, 0])); ls.append(lsd(x, xh) * len(x))
            sd = torch.cat(ss); res[k][nq] = (sd.mean().item(), sum(ls) / len(Xt))
        print(f"  model {k:>4}: " + "  ".join(f"n={nq}: {res[k][nq][0]:5.1f} dB SI-SDR / {res[k][nq][1]:4.1f} dB LSD" for nq in (1, 2, 4, 8)))
print("\n2. Uniform PCM at the same sampling rate, held-out speaker")
pcm = {}
for b in (1, 2, 3, 4, 6, 8):
    L = 2 ** b; xq = torch.round((Xt.clamp(-1, 1 - 1e-6) + 1) / 2 * L - 0.5).clamp(0, L - 1); xq = (xq + 0.5) / L * 2 - 1; pcm[b] = (sisdr(Xt[:, 0], xq[:, 0]).mean().item(), lsd(Xt, xq)); print(f"  {b} bits/sample = {b*8000:>6,} bit/s: SI-SDR {pcm[b][0]:5.1f} dB, LSD {pcm[b][1]:4.1f} dB")
print("\n3. Residual shrinkage and codebook use (all 8 stages applied, held-out speaker)")
R3 = {}
for name in ("drop", "f8"):
    m = M[name]
    with torch.no_grad():
        y = torch.cat([m.enc(Xt[i:i + 100]) for i in range(0, len(Xt), 100)]); B, _, S = y.shape; r = y.permute(0, 2, 1).reshape(-1, D); out = torch.zeros_like(r); resid = r.clone(); e0 = (r ** 2).sum().item(); rem = []; use = []; ppl = []
        for k in range(NQ):
            q, i = m.vq[k](resid, torch.ones(len(resid), dtype=torch.bool)); out += q; resid = resid - q; rem.append((resid ** 2).sum().item() / e0); cnt = torch.bincount(i, minlength=NCB).float(); p = cnt / cnt.sum(); use.append(int((cnt > 0).sum())); ppl.append(float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())))
    R3[name] = (rem, use, ppl); print(f"  model {name}"); print("    stages used:        " + "  ".join(f"{k+1:>6}" for k in range(NQ))); print("    residual energy:    " + "  ".join(f"{v*100:5.1f}%" for v in rem)); print("    codebook entries used (of 256): " + "  ".join(str(u) for u in use)); print("    perplexity:         " + "  ".join(f"{p:6.1f}" for p in ppl))
rem, use, ppl = R3["drop"]; m = M["drop"]
print("\n4. Tokens alone are enough to decode")
with torch.no_grad():
    x = Xt[:20]; xh, _, ids_t, _ = m(x, torch.full((20,), NQ)); z = sum(m.vq[k].embed[ids_t[:, :, k]] for k in range(NQ)).permute(0, 2, 1); x2 = m.dec(z)
print(f"  ids tensor {tuple(ids_t.shape)} integers in [0, {NCB-1}]; max |decode(ids) - forward output| = {(x2 - xh).abs().max().item():.1e}")
print(f"  1 s of audio: PCM 16 bit = {8000*2:,} bytes; {NQ} codebooks = {NQ*50:,} tokens = {NQ*50} bytes ({8000*2/(NQ*50):.0f}x smaller); 1 codebook = 50 bytes ({8000*2/50:.0f}x smaller)")
json.dump({"res": {k: {str(n): v for n, v in d.items()} for k, d in res.items()}, "pcm": {str(b): v for b, v in pcm.items()}, "rem": rem, "use": use, "ppl": ppl}, open("h2.json", "w"))
```

```text
held-out speaker: 500 clips; train speakers: ['george', 'jackson', 'lucas', 'nicolas', 'theo']
bitrate: 8 bits per token, 50 frames per second, so each codebook costs 400 bit/s; raw 8 kHz 16-bit PCM is 128,000 bit/s

1. Reconstruction on the held-out speaker against the number of codebooks used
  model drop: n=1:  -2.7 dB SI-SDR /  8.0 dB LSD  n=2:   0.0 dB SI-SDR /  7.4 dB LSD  n=4:   1.6 dB SI-SDR /  7.1 dB LSD  n=8:   2.0 dB SI-SDR /  7.0 dB LSD
  model   f8: n=1:  -2.9 dB SI-SDR /  7.1 dB LSD  n=2:   0.2 dB SI-SDR /  6.7 dB LSD  n=4:   2.4 dB SI-SDR /  6.3 dB LSD  n=8:   3.6 dB SI-SDR /  6.1 dB LSD
  model   f2: n=1:  -2.4 dB SI-SDR /  5.2 dB LSD  n=2:   0.2 dB SI-SDR /  5.0 dB LSD  n=4:   0.2 dB SI-SDR /  5.0 dB LSD  n=8:   0.3 dB SI-SDR /  5.0 dB LSD

2. Uniform PCM at the same sampling rate, held-out speaker
  1 bits/sample =  8,000 bit/s: SI-SDR  -6.7 dB, LSD 11.8 dB
  2 bits/sample = 16,000 bit/s: SI-SDR  -3.5 dB, LSD  9.8 dB
  3 bits/sample = 24,000 bit/s: SI-SDR   1.4 dB, LSD  7.9 dB
  4 bits/sample = 32,000 bit/s: SI-SDR   7.2 dB, LSD  6.2 dB
  6 bits/sample = 48,000 bit/s: SI-SDR  19.5 dB, LSD  3.5 dB
  8 bits/sample = 64,000 bit/s: SI-SDR  31.8 dB, LSD  1.8 dB

3. Residual shrinkage and codebook use (all 8 stages applied, held-out speaker)
  model drop
    stages used:             1       2       3       4       5       6       7       8
    residual energy:     35.7%   18.4%   10.0%    5.6%    3.4%    2.1%    1.4%    1.0%
    codebook entries used (of 256): 256  256  256  256  256  254  253  251
    perplexity:           15.6    17.1    18.2    18.1    18.8    18.7    19.8    21.3
  model f8
    stages used:             1       2       3       4       5       6       7       8
    residual energy:     49.1%   31.2%   21.4%   15.2%   11.0%    8.1%    6.1%    4.7%
    codebook entries used (of 256): 254  256  256  256  255  256  256  255
    perplexity:            8.3    10.3     9.9    11.2    11.5    13.9    15.3    16.1

4. Tokens alone are enough to decode
  ids tensor (20, 50, 8) integers in [0, 255]; max |decode(ids) - forward output| = 1.1e-07
  1 s of audio: PCM 16 bit = 16,000 bytes; 8 codebooks = 400 tokens = 400 bytes (40x smaller); 1 codebook = 50 bytes (320x smaller)
```

## References

- [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312), 2021.
- [High Fidelity Neural Audio Compression (EnCodec)](https://arxiv.org/abs/2210.13438), 2022.
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).
