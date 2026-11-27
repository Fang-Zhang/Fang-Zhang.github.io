---
title: "Audio Models, Explained (3/7): A CTC Recogniser Kept Working at 10 Digits After Training on 3 to 5, While an Encoder-Decoder on the Same Encoder Stopped at About 5"
date: 2027-07-15 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [audio, speech recognition, ctc, whisper, series]
description: "Part 3 of 7. How speech recognisers align audio to text: CTC against encoder-decoder attention, built from scratch on spoken digits and compared with Whisper tiny, including what happens at sequence lengths never seen in training."
---

## The problem: audio has no word boundaries

Recognising speech means mapping a sequence of 50 to 100 acoustic frames per second onto a much shorter sequence of characters or words, without knowing which frames belong to which output. Two families dominate. The first, [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (connectionist temporal classification), has the network emit a label or a blank at every frame and sums over all alignments that collapse to the right text. The second, used by [Whisper](https://arxiv.org/abs/2212.04356), reads the whole audio with an encoder and lets an autoregressive decoder produce the text token by token, looking back at the audio through cross-attention. This part builds one small model of each kind on the same encoder and compares them with Whisper tiny on a task all three can attempt.

## What the sources say

- **CTC.** The network has one extra output, "the probability of observing a 'blank', or no label", and a many-to-one map removes "all blanks and repeated labels from the paths". Its objective "depends only on the sequence of labels, and not on their duration or segmentation", so no frame-level labels are needed.
- **Whisper's architecture.** The authors "use an off-the-shelf architecture" and "chose an encoder-decoder Transformer". Audio is resampled to 16,000 Hz and turned into an 80-channel log-Mel spectrogram on 25 ms windows with a 10 ms stride; the encoder starts with "a small stem consisting of two convolution layers with a filter width of 3 and the GELU activation function ... where the second convolution layer has a stride of two", followed by sinusoidal position embeddings. The tiny model has 4 layers, width 384 and 39M parameters (Table 1 of the paper).
- **Whisper's data and format.** Training used 680,000 hours of weakly supervised audio-transcript pairs in 30-second segments. The decoder is steered by special tokens: it begins with `<|startoftranscript|>`, then a language token, then `<|transcribe|>` or `<|translate|>`, then whether to predict timestamps.
- **Whisper's reported accuracy and failure modes.** On LibriSpeech test-clean the tiny model reaches a word error rate of 7.6 percent with greedy decoding (Table 8) and 6.7 percent with beam search and temperature fallback (Table 9); the largest model reaches 2.7 and 2.5. The paper also lists the failure modes of its decoder: "getting stuck in repeat loops, not transcribing the first or last few words of an audio segment, or complete hallucination where the model will output a transcript entirely unrelated to the actual audio".
- **Robustness.** The paper's comparison with a LibriSpeech-trained wav2vec 2.0 shows equal error on LibriSpeech test-clean (2.7 percent for both) and very different error elsewhere, for instance 24.5 percent against 6.2 percent on Artie and 29.9 percent against 9.0 percent on Common Voice.
- **Self-supervised pre-training.** [wav2vec 2.0](https://arxiv.org/abs/2006.11477) masks the latent speech representation and solves a contrastive task; fine-tuned with only ten minutes of labelled data it reaches 4.8 and 8.2 percent word error rate on LibriSpeech clean and other test sets, and 1.8 and 3.3 percent with all 960 hours of labels.

## Experiments

**Task.** Spoken-digit sequences, because they are a real sequence-recognition problem (unknown boundaries, variable length, a new speaker at test time) small enough for a CPU. The [spoken-digit recordings](https://github.com/Jakobovski/free-spoken-digit-dataset) of Part 1 and Part 2 are concatenated with random 40 to 150 ms gaps into sequences of 3 to 5 digits for training (random gain of plus or minus 6 dB and light Gaussian noise added), from the five training speakers. Test sequences use only the sixth speaker, whose recordings were never seen. The score is the digit error rate (DER): total edit distance (substitutions, insertions and deletions) divided by the total number of reference digits.

**Models.** Both models share one encoder: 40-band log-Mel features (25 ms windows, 10 ms hop at 8 kHz), a two-convolution stem with the second convolution of stride two (20 ms per frame), sinusoidal position embeddings and four pre-norm Transformer layers of width 128 with 4 heads. The CTC model adds a linear layer with 11 outputs (blank and ten digits), 859,531 parameters in total. The encoder-decoder model adds a two-layer Transformer decoder with cross-attention and an end-of-sequence token, 1,390,604 parameters. Each was trained for 5,000 steps at batch size 32 (160,000 sequences) with AdamW, a learning rate of 1e-3 with warm-up and cosine decay, one seed, about 55 minutes on CPU. Whisper tiny was run zero-shot, with no training on this task: the weights came from the `openai-mirror/whisper-tiny` copy on ModelScope, because the usual hosting site is blocked on the machine used; the 8 kHz audio was resampled to 16 kHz, the decoder used default greedy decoding in English transcription mode, and the text output was converted to digits by taking every digit character or number word in order. The count of parameters of the loaded model is 37,760,640, against 39M in the paper's table.

![Three parts. Left: log-Mel spectrogram of four spoken digits (1, 2, 3, 4) above the CTC output probabilities over time: the blank probability stays near 1 and each digit appears as a narrow spike under its word. Middle: decoder cross-attention of the encoder-decoder model, a staircase in which each output step attends to a different time region in order, and the end-of-sequence step attends to the end. Right: digit error rate against the number of digits per recording for CTC, the encoder-decoder model and Whisper tiny; the encoder-decoder error rises from about 12 percent at 5 digits to 53 percent at 10 digits, CTC from about 16 to 25 percent, and Whisper tiny stays between 10 and 25 percent beyond 5 digits.](/assets/pic/audio-3-asr.png)

**1. Accuracy in and out of the training range.** 200 held-out-speaker sequences per row.

| digits per recording | system | digit error rate | exactly right | output shorter than reference |
|---|---|---|---|---|
| 3 to 5 (trained) | CTC | 15.7% | 54.5% | 11.5% |
| 3 to 5 (trained) | encoder-decoder | 13.2% | 54.5% | 1.0% |
| 3 to 5 (trained) | Whisper tiny | 15.3% | 59.0% | 34.5% |
| 8 to 10 (never seen) | CTC | 22.6% | 11.0% | 46.0% |
| 8 to 10 (never seen) | encoder-decoder | 49.3% | 0.0% | 100.0% |
| 8 to 10 (never seen) | Whisper tiny | 14.3% | 34.5% | 46.5% |

Inside the training range the three systems are within 2.5 points of each other, which at 200 sequences and one seed is not a ranking. Outside it they separate. The encoder-decoder model's average output length was 5.21 digits for references of 9.03, and every output was shorter than the reference.

**2. The sweep.** 100 sequences per length, DER in percent, and the mean number of digits the encoder-decoder model produced.

| digits | CTC | encoder-decoder | Whisper tiny | encoder-decoder mean output length |
|---|---|---|---|---|
| 2 | 16.5 | 25.0 | 32.0 | 2.33 |
| 3 | 11.3 | 7.7 | 18.7 | 3.00 |
| 4 | 16.2 | 11.5 | 15.5 | 4.00 |
| 5 | 15.6 | 11.8 | 13.0 | 4.95 |
| 6 | 16.0 | 25.0 | 12.5 | 5.14 |
| 7 | 19.4 | 35.4 | 10.6 | 5.11 |
| 8 | 21.4 | 42.9 | 22.4 | 5.24 |
| 9 | 21.8 | 48.2 | 25.0 | 5.21 |
| 10 | 25.4 | 52.9 | 18.7 | 5.31 |

The encoder-decoder model's output length follows the reference up to 5 digits and then stays at about 5.1 to 5.3, the largest length it saw during training. CTC error rises gradually, from 16.0 percent at 6 digits to 25.4 percent at 10.

**3. What each model does with the audio.** Over 100 held-out four-digit sequences the CTC output is blank in 83.0 percent of frames, and each emitted digit occupies 3.64 non-blank frames of 20 ms, about 73 ms for a word of several hundred milliseconds: the model marks where a digit is, not how long it lasts. The encoder-decoder model learned an alignment on its own: its cross-attention in the figure is a staircase, one region of the audio per output digit, with the end-of-sequence step looking at the end.

**4. Whisper's output.** Four examples of the raw text for 3 to 5 digit inputs: `6 9 2` came back as a short sentence ending in "two.", `8 4 2 0` as "8, 4, 2, 0.", `2 8 1 6` as "2 8 1 6." and `7 7 2 0 5` as "7772-0. Bye.". Extra words and punctuation are ignored by the digit extraction, so a failure shows up only as missing or wrong digits.

## Engineering reading

These points are this series' reading of the measurements and the sources.

1. **CTC fixes the output rate to the frame rate; a decoder does not.** CTC cannot emit more labels than frames and does not need to know the length ahead of time, so a recording twice as long simply yields more spikes. The decoder has to decide when to stop, and in this experiment that decision was learned from sequences of at most 5 digits and applied unchanged to 9.
2. **The failure is in the decoder, not the encoder.** Both models share the encoder and the encoder-decoder error matched CTC up to 5 digits, so the divergence beyond it comes from the added decoding mechanism. Why exactly it stops at 5 (the end-of-sequence prior, the position embeddings of the decoder, the attention) is not isolated by these runs.
3. **Large-scale decoders fail in the same direction, at a different scale.** Whisper's own limitation list includes decoders that skip words at the start or end and hallucinate. The behaviour here, outputs that end early on long input, is of the same kind. This is an analogy; the causes were not compared.
4. **Scale buys robustness to conditions the training did not cover.** Whisper tiny, with about 44 times the parameters of the CTC model and 680,000 hours of training data, had never seen this task, a bandwidth of 8 kHz or isolated digit sequences, yet it was level with the in-domain models at 3 to 5 digits and the best of the three at 6, 7 and 10 digits (CTC was best at 8 and 9). Its error was not zero or stable: it dropped to 10.6 percent at 7 digits and rose to 25.0 percent at 9.
5. **CTC trades expressiveness for a guarantee.** The model can say which digits appear and roughly where, but it cannot condition one output on another, so it has no language model of its own; word-level systems built on it add one outside. A decoder conditions on everything it has written, which is why Whisper can output punctuation and casing, and also why it can write text that was never spoken.

## Limits

- The task is ten isolated words from one dataset, recorded by six speakers; the digit error rate here says nothing about word error rate on conversational speech, and the in-domain models were trained on exactly this vocabulary.
- One test speaker, 200 sequences per bucket and 100 per sweep length, and one training seed per model. The sweep curves are noisy (the Whisper curve jumps between neighbouring lengths); differences of a few points are not established.
- Whisper tiny received 8 kHz audio resampled to 16 kHz, a bandwidth it was not trained on, from a mirror of its released weights; its results in the paper use beam search of 5 with temperature fallback and ours use greedy decoding. The digit extraction ignores extra words but also accepts any digit written anywhere in the output.
- The encoder-decoder model's length failure may be reduced by training on longer sequences, by relative position encodings or by other decoding rules; none was tried. CTC's gradual rise from 16 to 25 percent also coincides with positions longer than any in training and was not isolated.
- No language model, beam search or external data was used for the small models.
- Wav2vec 2.0 and Whisper numbers are quoted from the papers and come from different test conditions than this experiment.

## Code and output

```python
import sys, glob, wave, math, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, librosa
R = "free-spoken-digit-dataset-master/recordings/"; SR = 8000; HOP = 80; WIN = 200; NFFT = 256; NMEL = 40; D = 128; BOS, EOS = 10, 11
MEL = torch.tensor(librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=NMEL, fmax=4000), dtype=torch.float32); WINDOW = torch.hann_window(WIN)
HELD = "yweweler"
def load_clips():
    C = {}
    for f in sorted(glob.glob(R + "*.wav")):
        d, s, _ = f.split("/")[-1][:-4].split("_"); w = wave.open(f); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        C.setdefault(s, []).append((int(d), x / (np.abs(x).max() + 1e-9) * 0.9))
    return C
def make_seq(C, speakers, L, rng):
    ds, xs = [], []
    for _ in range(L):
        s = speakers[rng.integers(len(speakers))]; d, x = C[s][rng.integers(len(C[s]))]; ds.append(d); xs.append(x); xs.append(np.zeros(int(rng.uniform(0.04, 0.15) * SR), dtype=np.float32))
    return ds, np.concatenate(xs[:-1])
def feats(waves):
    N = [len(w) for w in waves]; M = max(N); x = torch.zeros(len(waves), M)
    for i, w in enumerate(waves): x[i, :len(w)] = torch.from_numpy(w)
    S = torch.stft(x, NFFT, HOP, WIN, WINDOW, center=True, return_complex=True).abs() ** 2
    m = torch.log10(torch.einsum("mf,bft->bmt", MEL, S) + 1e-5); m = (m + 2.5) / 1.5
    T1 = [n // HOP + 1 for n in N]; T2 = [(t + 1) // 2 for t in T1]; return m, torch.tensor(T2)
def sinus(n, d):
    pos = torch.arange(n)[:, None]; i = torch.arange(0, d, 2)[None]; a = pos / 10000 ** (i / d); pe = torch.zeros(n, d); pe[:, 0::2] = a.sin(); pe[:, 1::2] = a.cos(); return pe
class Enc(nn.Module):
    def __init__(s, L=4):
        super().__init__(); s.c1 = nn.Conv1d(NMEL, D, 3, padding=1); s.c2 = nn.Conv1d(D, D, 3, stride=2, padding=1)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, 0.1, "gelu", batch_first=True, norm_first=True), L, enable_nested_tensor=False); s.ln = nn.LayerNorm(D)
    def forward(s, m, lens):
        h = F.gelu(s.c2(F.gelu(s.c1(m)))).transpose(1, 2); T = h.shape[1]; h = h + sinus(T, D); mask = torch.arange(T)[None] >= lens[:, None]
        return s.ln(s.tr(h, src_key_padding_mask=mask)), mask
class CTCModel(nn.Module):
    def __init__(s): super().__init__(); s.enc = Enc(); s.out = nn.Linear(D, 11)
    def forward(s, m, lens): h, _ = s.enc(m, lens); return s.out(h).log_softmax(-1)
class DecLayer(nn.Module):
    def __init__(s):
        super().__init__(); s.sa = nn.MultiheadAttention(D, 4, 0.1, batch_first=True); s.ca = nn.MultiheadAttention(D, 4, 0.1, batch_first=True)
        s.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)); s.n1, s.n2, s.n3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
    def forward(s, x, h, hmask):
        T = x.shape[1]; cm = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); y = s.n1(x); x = x + s.sa(y, y, y, attn_mask=cm, need_weights=False)[0]
        o, w = s.ca(s.n2(x), h, h, key_padding_mask=hmask, need_weights=True, average_attn_weights=True); x = x + o; return x + s.ff(s.n3(x)), w
class AttModel(nn.Module):
    def __init__(s): super().__init__(); s.enc = Enc(); s.emb = nn.Embedding(12, D); s.layers = nn.ModuleList([DecLayer(), DecLayer()]); s.ln = nn.LayerNorm(D); s.out = nn.Linear(D, 12)
    def dec(s, tok, h, hm):
        x = s.emb(tok) + sinus(tok.shape[1], D)
        for l in s.layers: x, w = l(x, h, hm)
        return s.out(s.ln(x)), w
    def forward(s, m, lens, tok): h, hm = s.enc(m, lens); return s.dec(tok, h, hm)[0]
    @torch.no_grad()
    def greedy(s, m, lens, maxlen=16, want_att=False):
        h, hm = s.enc(m, lens); tok = torch.full((len(m), 1), BOS); att = None
        for _ in range(maxlen):
            lo, w = s.dec(tok, h, hm); tok = torch.cat([tok, lo[:, -1:].argmax(-1)], 1); att = w
        res = []
        for r in tok[:, 1:].tolist(): res.append(r[:r.index(EOS)] if EOS in r else r)
        return (res, att) if want_att else res
def ctc_decode(lp, lens):
    res = []
    for b in range(len(lp)):
        a = lp[b, :lens[b]].argmax(-1).tolist(); out = []; prev = 0
        for t in a:
            if t != 0 and t != prev: out.append(t - 1)
            prev = t
        res.append(out)
    return res
def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]
if __name__ == "__main__":
    kind, steps = sys.argv[1], int(sys.argv[2]); BS = int(sys.argv[3]) if len(sys.argv) > 3 else 32; torch.manual_seed(0); rng = np.random.default_rng(0)
    C = load_clips(); tr = [s for s in C if s != HELD]; model = CTCModel() if kind == "ctc" else AttModel(); print("params", sum(p.numel() for p in model.parameters()), "speakers", tr, flush=True)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, (t + 1) / 300) * max(0.05, 0.5 * (1 + math.cos(math.pi * t / steps)))); t0 = time.time(); ema = None
    for it in range(steps):
        seqs = [make_seq(C, tr, int(rng.integers(3, 6)), rng) for _ in range(BS)]; waves = [w * 10 ** (rng.uniform(-6, 6) / 20) + rng.normal(0, 10 ** (-rng.uniform(30, 50) / 20), len(w)).astype(np.float32) * 0.1 for _, w in seqs]; m, lens = feats(waves)
        if kind == "ctc":
            lp = model(m, lens); tg = torch.cat([torch.tensor(d) + 1 for d, _ in seqs]); tl = torch.tensor([len(d) for d, _ in seqs])
            loss = F.ctc_loss(lp.transpose(0, 1), tg, lens, tl, zero_infinity=True)
        else:
            L = max(len(d) for d, _ in seqs) + 1; tin = torch.full((BS, L), EOS); tout = torch.full((BS, L), -100)
            for i, (d, _) in enumerate(seqs): tin[i, 0] = BOS; tin[i, 1:len(d) + 1] = torch.tensor(d); tout[i, :len(d)] = torch.tensor(d); tout[i, len(d)] = EOS
            loss = F.cross_entropy(model(m, lens, tin).reshape(-1, 12), tout.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if it % 100 == 0: print(it, f"loss {ema:.4f} t {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), f"asr_{kind}.pt"); print("saved", flush=True)
```

```python
import json, re, numpy as np, torch, librosa
from asr import *
from transformers import WhisperForConditionalGeneration, WhisperProcessor
torch.manual_seed(0); C = load_clips(); rng = np.random.default_rng(123)
ctc = CTCModel(); ctc.load_state_dict(torch.load("asr_ctc.pt")); ctc.eval(); att = AttModel(); att.load_state_dict(torch.load("asr_att.pt")); att.eval()
wp = WhisperProcessor.from_pretrained("whisper-tiny"); wm = WhisperForConditionalGeneration.from_pretrained("whisper-tiny").eval()
WORDS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
def parse(t): return [int(m) if m.isdigit() else WORDS[m] for m in re.findall(r"[0-9]|zero|oh|one|two|three|four|five|six|seven|eight|nine", t.lower())]
def whisper(waves):
    out = []
    for i in range(0, len(waves), 20):
        w16 = [librosa.resample(w, orig_sr=SR, target_sr=16000) for w in waves[i:i + 20]]; inp = wp(w16, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad(): ids = wm.generate(inp.input_features, language="en", task="transcribe", max_new_tokens=60)
        out += wp.batch_decode(ids, skip_special_tokens=True)
    return out
res = {}; samples = {}
for name, (lo, hi) in (("3-5 digits", (3, 5)), ("8-10 digits", (8, 10))):
    data = [make_seq(C, [HELD], int(rng.integers(lo, hi + 1)), rng) for _ in range(200)]; refs = [d for d, _ in data]; waves = [w for _, w in data]
    preds = {"ctc": [], "att": [], "whisper-tiny": []}; wt = whisper(waves); preds["whisper-tiny"] = [parse(t) for t in wt]
    for i in range(0, 200, 50):
        m, lens = feats(waves[i:i + 50])
        with torch.no_grad(): preds["ctc"] += ctc_decode(ctc(m, lens), lens)
        preds["att"] += att.greedy(m, lens, maxlen=14)
    res[name] = {}
    for k, p in preds.items():
        E = sum(edit(r, q) for r, q in zip(refs, p)); N = sum(len(r) for r in refs); ex = np.mean([r == q for r, q in zip(refs, p)]); short = np.mean([len(q) < len(r) for r, q in zip(refs, p)]); mlen = np.mean([len(q) for q in p])
        res[name][k] = dict(der=E / N * 100, exact=ex * 100, short=short * 100, mean_len=mlen, mean_ref=np.mean([len(r) for r in refs])); print(f"{name:12s} {k:13s} DER {E/N*100:5.1f}%  exact {ex*100:5.1f}%  shorter-than-ref {short*100:5.1f}%  mean output len {mlen:5.2f} (ref {np.mean([len(r) for r in refs]):.2f})", flush=True)
    if lo == 3:
        samples["whisper"] = [(r, t) for r, t in list(zip(refs, wt))[:4]]
# CTC alignment statistics on 3-5 digit set
data = [make_seq(C, [HELD], 4, rng) for _ in range(100)]; m, lens = feats([w for _, w in data])
with torch.no_grad(): lp = ctc(m, lens)
a = lp.argmax(-1); valid = torch.arange(a.shape[1])[None] < lens[:, None]; blank = ((a == 0) & valid).sum().item() / valid.sum().item(); nonblank_frames = ((a != 0) & valid).sum().item(); emitted = sum(len(x) for x in ctc_decode(lp, lens))
print(f"CTC: {blank*100:.1f}% of frames are blank; {nonblank_frames/emitted:.2f} non-blank frames per emitted digit; frame rate {1000*HOP*2/SR:.0f} ms; mean seconds per sequence {np.mean([len(w) for _, w in data])/SR:.2f}")
# figure data
d, w = data[0]; m1, l1 = feats([w]); 
with torch.no_grad(): post = ctc(m1, l1)[0].exp().numpy(); hy, aw = att.greedy(m1, l1, maxlen=len(d) + 1, want_att=True)
json.dump(dict(res=res, blank=blank, per_digit=nonblank_frames / emitted, ref=d, ctc_hyp=ctc_decode(torch.log(torch.tensor(post))[None], l1)[0], att_hyp=hy[0], mel=m1[0].numpy().tolist(), post=post.tolist(), att=aw[0].numpy().tolist(), whisper_samples=samples["whisper"]), open("h3.json", "w"))
print("whisper samples:", samples["whisper"])
```

```python
import json, numpy as np, torch
exec(open("h3_eval.py").read().split("res = {}; samples = {}")[0])
rng = np.random.default_rng(7); out = {}
print("digits  ctc_DER  att_DER  whisper_DER  att_mean_len")
for L in range(2, 11):
    data = [make_seq(C, [HELD], L, rng) for _ in range(100)]; refs = [d for d, _ in data]; waves = [w for _, w in data]; P = {"ctc": [], "att": []}
    for i in range(0, 100, 50):
        m, lens = feats(waves[i:i + 50])
        with torch.no_grad(): P["ctc"] += ctc_decode(ctc(m, lens), lens)
        P["att"] += att.greedy(m, lens, maxlen=14)
    P["whisper"] = [parse(t) for t in whisper(waves)]; N = L * 100
    out[L] = {k: sum(edit(r, q) for r, q in zip(refs, p)) / N * 100 for k, p in P.items()}; out[L]["att_len"] = float(np.mean([len(q) for q in P["att"]]))
    print(f"{L:5d}  {out[L]['ctc']:7.1f}  {out[L]['att']:7.1f}  {out[L]['whisper']:10.1f}  {out[L]['att_len']:8.2f}", flush=True)
json.dump(out, open("h3_sweep.json", "w"))
```

```text
3-5 digits   ctc           DER  15.7%  exact  54.5%  shorter-than-ref  11.5%  mean output len  4.14 (ref 4.05)
3-5 digits   att           DER  13.2%  exact  54.5%  shorter-than-ref   1.0%  mean output len  4.04 (ref 4.05)
3-5 digits   whisper-tiny  DER  15.3%  exact  59.0%  shorter-than-ref  34.5%  mean output len  3.56 (ref 4.05)
8-10 digits  ctc           DER  22.6%  exact  11.0%  shorter-than-ref  46.0%  mean output len  8.66 (ref 9.03)
8-10 digits  att           DER  49.3%  exact   0.0%  shorter-than-ref 100.0%  mean output len  5.21 (ref 9.03)
8-10 digits  whisper-tiny  DER  14.3%  exact  34.5%  shorter-than-ref  46.5%  mean output len  8.62 (ref 9.03)
CTC: 83.0% of frames are blank; 3.64 non-blank frames per emitted digit; frame rate 20 ms; mean seconds per sequence 1.72
```

```text
digits  ctc_DER  att_DER  whisper_DER  att_mean_len
    2     16.5     25.0        32.0      2.33
    3     11.3      7.7        18.7      3.00
    4     16.2     11.5        15.5      4.00
    5     15.6     11.8        13.0      4.95
    6     16.0     25.0        12.5      5.14
    7     19.4     35.4        10.6      5.11
    8     21.4     42.9        22.4      5.24
    9     21.8     48.2        25.0      5.21
   10     25.4     52.9        18.7      5.31
```

## References

- A. Graves et al., [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf), ICML 2006.
- A. Radford et al., [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356), 2022.
- A. Baevski et al., [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477), 2020.
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).
